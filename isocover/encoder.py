"""The map itself: a convolutional encoder, observation -> z in R^d.

Architecture (identical for the 64x64x3 DMLab frames and the 84x84x1 Atari
frames; only the input shape differs, and the convolutional trunk computes
its own flattened width from whatever shape it is given):

  Conv2d(C, 32, kernel 8, stride 4) -> LeakyReLU
  Conv2d(32, 64, kernel 4, stride 2) -> LeakyReLU
  Conv2d(64, 64, kernel 3, stride 1) -> LeakyReLU
  Flatten
  Linear(flat, 512) -> LayerNorm(512) -> ReLU
  Linear(512, 512)  -> LayerNorm(512) -> ReLU
  Linear(512, d)                                  # linear output, NO clamp

The convolutional trunk (the three Conv2d/LeakyReLU layers) is the widely
used "NatureCNN"-style trunk popularized by DQN (Mnih et al., 2015) and
reused by the original Random Network Distillation implementation
(Burda et al., 2018, "Exploration by Random Network Distillation",
github.com/openai/random-network-distillation); the head (two
Linear/LayerNorm/ReLU blocks then a linear projection to d) is this
project's own addition, mapping the trunk's features to the d-dimensional
map SIGReg regularizes. Nothing clamps z: it is free to take any real value,
which is what lets it be pushed toward an ordinary, unbounded N(0, I_d).

Two live copies are kept during training: the ONLINE network (updated by
gradient descent every encoder step) and an EMA (exponential moving average)
copy. The final recipe uses an EMA rate of 1.0, i.e. no smoothing at all --
the EMA copy is just a same-step mirror of the online network -- so the
reward always reads the live, currently-training map (see reward.py). The
EMA machinery is kept here anyway since it costs nothing when the rate is 1.0
and is what a slower-EMA ablation would use.
"""

from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn


def _layer_init(layer: nn.Module, std: float = math.sqrt(2.0), bias: float = 0.0):
  nn.init.orthogonal_(layer.weight, std)
  nn.init.constant_(layer.bias, bias)
  return layer


class MLPHead(nn.Module):
  """The encoder's head: flat features -> z (B, d). Linear output, no clamp."""

  def __init__(self, in_dim: int, d: int, hidden=(512, 512)):
    super().__init__()
    layers, width = [], int(in_dim)
    for h in hidden:
      layers.append(nn.Linear(width, int(h)))
      layers.append(nn.LayerNorm(int(h)))
      layers.append(nn.ReLU())
      width = int(h)
    layers.append(nn.Linear(width, int(d)))
    self.net = nn.Sequential(*layers)
    self.in_dim, self.d = int(in_dim), int(d)

  def forward(self, x):
    return self.net(x)


class ConvEncoder(nn.Module):
  """frame (B, H, W, C) uint8 or float -> z (B, d). Memoryless: one frame in,
  one point out, nothing carried between calls (any memory in this project
  lives in the policy's LSTM, above the encoder, never inside it -- see
  ppo.py)."""

  def __init__(self, frame_shape, d: int, hidden=(512, 512), seed: int = 0):
    super().__init__()
    h, w, c = (int(v) for v in frame_shape)
    self.frame_shape = (h, w, c)
    with torch.random.fork_rng(devices=[]):
      torch.manual_seed(int(seed))
      self.conv = nn.Sequential(
          _layer_init(nn.Conv2d(c, 32, 8, stride=4)), nn.LeakyReLU(),
          _layer_init(nn.Conv2d(32, 64, 4, stride=2)), nn.LeakyReLU(),
          _layer_init(nn.Conv2d(64, 64, 3, stride=1)), nn.LeakyReLU(),
          nn.Flatten())
      with torch.no_grad():
        flat = self.conv(torch.zeros(1, c, h, w)).shape[1]
      self.head = MLPHead(int(flat), int(d), hidden=hidden)
    self.flat_dim, self.d = int(flat), int(d)

  def forward(self, frames: torch.Tensor) -> torch.Tensor:
    x = frames.permute(0, 3, 1, 2)
    x = x.float() / 255.0 if x.dtype == torch.uint8 else x / 255.0
    return self.head(self.conv(x))


def make_conv_encoder(frame_shape, d: int, hidden=(512, 512), seed: int = 0) -> ConvEncoder:
  """``ConvEncoder`` whose initial weights depend only on ``seed`` (so two
  calls with the same seed produce byte-identical initial parameters,
  independent of what random draws happened earlier in the process)."""
  return ConvEncoder(frame_shape, d, hidden, seed)


@torch.no_grad()
def ema_update(target: nn.Module, online: nn.Module, rate: float) -> None:
  """target <- (1 - rate) * target + rate * online (parameters); buffers are
  copied verbatim. ``rate=1.0`` (the final recipe: ``enc_ema_rate=1.0``) makes
  the target an exact same-step copy of the online network every call --
  i.e. no EMA smoothing, the reward reads the live network. ``rate=0.0``
  would freeze the target forever."""
  rate = float(rate)
  tgt, src = list(target.parameters()), list(online.parameters())
  if tgt and hasattr(torch, "_foreach_lerp_"):
    torch._foreach_lerp_(tgt, src, rate)
  else:
    for pt, po in zip(tgt, src):
      pt.lerp_(po, rate)
  for bt, bo in zip(target.buffers(), online.buffers()):
    bt.copy_(bo)


def make_encoder_pair(frame_shape, d: int, hidden=(512, 512), seed: int = 0,
                       device="cpu"):
  """Build the (online, ema) pair this project always trains together: the
  EMA copy starts as an exact clone of the online network and has its
  gradient tracking turned off (it is only ever written to by
  ``ema_update``, never by an optimizer)."""
  online = make_conv_encoder(frame_shape, d, hidden, seed).to(device)
  ema = copy.deepcopy(online)
  for p in ema.parameters():
    p.requires_grad_(False)
  return online, ema
