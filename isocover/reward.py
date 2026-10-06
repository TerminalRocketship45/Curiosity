"""The intrinsic reward: r = ||z||^2, and its normalization.

Plain-language idea. Once the map is (approximately) shaped like a standard
Gaussian N(0, I_d), there is a clean dictionary between "distance from the
center" and "how rare this point is", via the Gaussian's own change-of-
variables formula. For z ~ N(0, I_d), the log-density is

  log p(z) = -d/2 log(2*pi) - ||z||^2 / 2,

so -log p(z) = ||z||^2 / 2 + constant: the squared distance from the origin
IS (up to an additive constant and a factor of 2) the negative log-density,
i.e. the rarity, of that point under the map's own target distribution. A
point far from the center is a point the Gaussian says should rarely occur.
Rewarding ||z||^2 is therefore rewarding the agent for reaching observations
the map currently treats as rare -- exactly the "be curious about the
unfamiliar" signal intrinsic-reward methods are after, but read directly off
the shape of a representation the encoder is already being trained to keep
Gaussian, with no separate novelty-detector network (contrast RND, which
trains a SEPARATE predictor network and rewards ITS prediction error).

  r_t = ||z(s_{t+1})||^2

read from the LIVE (online) encoder -- the final recipe uses an EMA rate of
1.0 (see encoder.py), so "live" and "EMA" are the same network at every step
anyway, but the reward always reads whichever copy the run's ``enc_ema_rate``
designates, via the ``use_ema`` argument below, for compatibility with a
slower-EMA ablation.

Normalization. PPO (ppo.py) needs the reward's SCALE to be roughly stable
over training, since GAE and the value-function regression both assume a
roughly stationary reward distribution. Following the convention introduced
by Random Network Distillation (Burda et al., 2018) and used by every
intrinsic-reward arm in this project: keep a running estimate of the
variance of the DISCOUNTED intrinsic return (not of the per-step reward
itself -- the return's scale is what actually matters to GAE), and divide
every reward in the rollout by its standard deviation. No mean-subtraction:
only the scale is normalized, never the sign or the mean.
"""

from __future__ import annotations

import math

import numpy as np
import torch


class RunningMeanStd:
  """Welford's online algorithm for a running mean and variance, updated in
  batches (as a rollout's worth of values arrives). Kept in float64 for
  numerical stability over long runs. Identical in spirit to the
  implementation used by the original RND codebase and this project's own
  trainer."""

  def __init__(self, shape=()):
    self.mean = np.zeros(shape, dtype=np.float64)
    self.var = np.ones(shape, dtype=np.float64)
    self.count = 1e-4

  def update_from_moments(self, batch_mean, batch_var, batch_count):
    delta = batch_mean - self.mean
    tot = self.count + batch_count
    new_mean = self.mean + delta * batch_count / tot
    m2 = (self.var * self.count + batch_var * batch_count
          + np.square(delta) * self.count * batch_count / tot)
    self.mean, self.var, self.count = new_mean, m2 / tot, tot

  def update(self, x):
    x = np.asarray(x, dtype=np.float64)
    axis = tuple(range(x.ndim - self.mean.ndim)) if self.mean.ndim else None
    self.update_from_moments(x.mean(axis=axis), x.var(axis=axis),
                              x.size // max(self.mean.size, 1))

  def state_dict(self):
    return {"mean": np.array(self.mean), "var": np.array(self.var),
            "count": float(self.count)}

  def load_state_dict(self, s):
    self.mean = np.array(s["mean"], dtype=np.float64)
    self.var = np.array(s["var"], dtype=np.float64)
    self.count = float(s["count"])


def zsq_reward(encoder_net, arrived_frames: torch.Tensor) -> torch.Tensor:
  """r = ||z(s')||^2, z from ``encoder_net`` (pass the EMA copy for the
  ``enc_ema_rate < 1`` case; pass the online copy when ``enc_ema_rate == 1.0``,
  the final recipe, where they are the same network every step anyway).

  Args:
    encoder_net: a module mapping frames -> (B, d) embeddings (no gradient is
      taken here; wrap the call in ``torch.no_grad()`` at the call site when
      collecting rollouts, as the training scripts do).
    arrived_frames: (B, *frame_shape) the observation the action arrived at.

  Returns:
    (B,) tensor of rewards, one per environment.
  """
  z = encoder_net(arrived_frames)
  return z.pow(2).sum(dim=1)


class IntrinsicRewardNormalizer:
  """The project's standard intrinsic-reward normalization.

  Maintains a per-environment discounted running sum (NOT reset at episode
  ends -- the discounted return of an ongoing, non-episodic process), feeds a
  count-based running variance estimate from each rollout's worth of those
  sums, and divides every raw reward in the rollout by the resulting running
  standard deviation. No mean-subtraction.
  """

  def __init__(self, gamma_int: float, num_envs: int):
    self.gamma_int = float(gamma_int)
    self.rms = RunningMeanStd()
    self._running_return = np.zeros(int(num_envs), dtype=np.float64)

  def normalize(self, raw_rewards: torch.Tensor) -> torch.Tensor:
    """``raw_rewards``: (T, N) raw ``||z||^2`` rewards for one rollout of T
    steps across N environments. Returns the same shape, divided by the
    (updated) running standard deviation of the discounted return."""
    t_len, n = raw_rewards.shape
    rets = torch.empty_like(raw_rewards, dtype=torch.float64)
    running = torch.as_tensor(self._running_return, dtype=torch.float64,
                               device=raw_rewards.device)
    raw64 = raw_rewards.double()
    for t in range(t_len):
      running = running * self.gamma_int + raw64[t]
      rets[t] = running
    self._running_return = running.cpu().numpy()
    mean = rets.mean()
    var = ((rets - mean) ** 2).mean()
    self.rms.update_from_moments(float(mean), float(var), rets.numel())
    scale = math.sqrt(float(self.rms.var)) + 1e-8
    return raw_rewards / scale

  def state_dict(self):
    return {"rms": self.rms.state_dict(),
            "running_return": np.array(self._running_return)}

  def load_state_dict(self, s):
    self.rms.load_state_dict(s["rms"])
    self._running_return = np.array(s["running_return"], dtype=np.float64)
