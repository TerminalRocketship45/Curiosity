"""Recurrent PPO with two value heads (extrinsic and intrinsic).

Architecture, a firm design choice for every environment in this release
(DMLab, Atari): the encoder sees only the CURRENT frame; all memory lives in
an LSTM sitting on top of it.

    current frame -> encoder (no memory) -> LSTM -> actor head, two critic
                                                      heads (ext, int)

With ``--policy-input embedding-only`` (the final recipe for both ``||z||^2``
and the PPO baseline), the "encoder" the policy reads from is the SAME
SIGReg map that the curiosity reward is computed from (for the PPO baseline,
with no intrinsic reward, the map is simply not shaped by ``map_loss.py`` --
train.py still needs a perception pathway for the policy, so the baseline
either reuses an (untrained, or separately-pretrained) encoder, or, as the
real baseline runs did, a plain conv trunk; see train.py's ``--arm ppo``
path). There is no separate pixel pathway for the policy: the policy acts
entirely on z.

Two value heads, two discount factors. Extrinsic (task) reward uses
``gamma_ext`` (0.999 in the final recipe: value the far future almost as much
as the present, appropriate for a sparse task reward); intrinsic reward uses
``gamma_int`` (0.99: a shorter effective horizon, appropriate for a dense,
per-step curiosity signal). Both streams get their own GAE advantage and
their own value-function regression target; the two advantages are combined
as

    advantage = ext_coef * adv_ext + int_coef * adv_int

(the final recipe: ``ext_coef=2.0``, ``int_coef=1.0``) and this COMBINED
advantage is what is normalized (mean 0, unit std, per minibatch) before
being used in the clipped PPO objective -- i.e. each stream keeps its own
raw scale until they are combined, and normalization happens once, after
combining, exactly matching the "combined" mode (the default, and the one
every final run in this project used; the trainer this was extracted from
also offers a "per_stream" mode that normalizes each stream before combining,
but no final run in this project used it, so it is not reproduced here).

Nothing here is specific to DMLab or Atari; ``envs/dmlab.py`` and
``envs/atari.py`` only need to hand back frames, discrete actions, and
episode-done flags in the shapes used below.
"""

from __future__ import annotations

import dataclasses
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def layer_init(layer: nn.Module, std: float = math.sqrt(2.0), bias: float = 0.0):
  nn.init.orthogonal_(layer.weight, std)
  nn.init.constant_(layer.bias, bias)
  return layer


class PixelBody(nn.Module):
  """The PPO BASELINE's own perception pathway: a plain conv trunk reading
  raw pixels directly, with no SIGReg/temporal shaping of its features at
  all (unlike the ``||z||^2`` arm, the baseline has no intrinsic reward and
  no map -- the real final baseline runs in this project used exactly this
  kind of trunk, trained end to end by the PPO loss alone, rather than the
  ``||z||^2`` arm's ``embedding_only`` pathway into the SIGReg map). Same
  "NatureCNN"-style trunk as ``isocover.encoder.ConvEncoder`` (see its
  docstring for the citation), ReLU rather than LeakyReLU (this trunk is
  not an input to SIGReg, so there is no reason to match that choice), plus
  one more linear layer to the LSTM's input width.
  """

  def __init__(self, frame_shape, hidden: int = 512, seed: int = 0):
    super().__init__()
    h, w, c = (int(v) for v in frame_shape)
    with torch.random.fork_rng(devices=[]):
      torch.manual_seed(int(seed))
      self.conv = nn.Sequential(
          layer_init(nn.Conv2d(c, 32, 8, stride=4)), nn.ReLU(),
          layer_init(nn.Conv2d(32, 64, 4, stride=2)), nn.ReLU(),
          layer_init(nn.Conv2d(64, 64, 3, stride=1)), nn.ReLU(),
          nn.Flatten())
      with torch.no_grad():
        flat = self.conv(torch.zeros(1, c, h, w)).shape[1]
      self.head = nn.Sequential(layer_init(nn.Linear(flat, hidden)), nn.ReLU())
    self.out_dim = int(hidden)

  def forward(self, frames):
    x = frames.permute(0, 3, 1, 2)
    x = x.float() / 255.0 if x.dtype == torch.uint8 else x / 255.0
    return self.head(self.conv(x))


# --------------------------------------------------------------- policy --

class RecurrentActorCritic(nn.Module):
  """frame -> body (no memory) -> LSTM -> actor logits + two value heads.

  ``body`` is any module mapping ``(B, *frame_shape) -> (B, feat_dim)``,
  called FRESH (with gradient tracking) on every ``step``/``sequence`` call --
  never precomputed and detached. This is what makes the ``||z||^2`` arm's
  "LIVE" wiring work: pass the SAME object as the SIGReg encoder's online
  network for ``body``, and because it is a registered submodule here, both
  ``RecurrentActorCritic.parameters()`` (this module's own PPO optimizer) AND
  the encoder's own optimizer (``map_loss.py``'s SIGReg/temporal training, a
  separate optimizer instance) update the identical parameter tensors -- two
  independent optimizers acting on one shared set of weights, exactly as the
  original research trainer's "encoder's own optimizer" plus "policy
  optimizer's second param group" design. For the plain PPO baseline, pass a
  fresh ``PixelBody`` instead: it is then trained ONLY by the PPO loss below,
  with no intrinsic reward and no SIGReg shaping at all.
  """

  def __init__(self, body: nn.Module, feat_dim: int, n_actions: int,
               critics=("ext", "int"), lstm_hidden: int = 512,
               lstm_layers: int = 1, seed: int = 0):
    super().__init__()
    self.body = body
    self.feat_dim = int(feat_dim)
    self.n_actions = int(n_actions)
    self.lstm_hidden = int(lstm_hidden)
    self.lstm_layers = int(lstm_layers)
    with torch.random.fork_rng(devices=[]):
      torch.manual_seed(int(seed))
      self.lstm = nn.LSTM(self.feat_dim, self.lstm_hidden, self.lstm_layers)
      for name, param in self.lstm.named_parameters():
        if "bias" in name:
          nn.init.constant_(param, 0.0)
        else:
          nn.init.orthogonal_(param, 1.0)
      self.actor = layer_init(nn.Linear(self.lstm_hidden, self.n_actions), std=0.01)
      self.critics = nn.ModuleDict(
          {c: layer_init(nn.Linear(self.lstm_hidden, 1), std=1.0) for c in critics})

  def initial_state(self, n: int, device=None, dtype=torch.float32):
    z = torch.zeros(self.lstm_layers, int(n), self.lstm_hidden, device=device, dtype=dtype)
    return (z, z.clone())

  def encode(self, frames):
    return self.body(frames)

  def _heads(self, h):
    values = {c: head(h).squeeze(-1) for c, head in self.critics.items()}
    return self.actor(h), values

  def step(self, frame, state=None, reset=None):
    """One timestep. frame: (N, *frame_shape). Returns (logits (N, A),
    values {name: (N,)}, next_state)."""
    n = frame.shape[0]
    if state is None:
      state = self.initial_state(n, frame.device)
    if reset is not None:
      state = reset_state(state, reset)
    feat = self.encode(frame)
    out, state = self.lstm(feat.unsqueeze(0), state)
    logits, values = self._heads(out.squeeze(0))
    return logits, values, state

  def sequence(self, frames, state=None, reset_before=None):
    """A whole (T, N, *frame_shape) rollout slice, replayed from ``state``.
    The body is called ONCE, on the ``(T*N, *frame_shape)`` flattening (every
    row still one frame -- the body itself is always memoryless).

    ``reset_before`` (T, N) or None: zero the state entering step t (build it
    with ``reset_before_from_done``). Returns (logits (T,N,A), values
    {name: (T,N)}, final_state)."""
    t_len, n = int(frames.shape[0]), int(frames.shape[1])
    if state is None:
      state = self.initial_state(n, frames.device)
    feat = self.encode(frames.reshape(t_len * n, *frames.shape[2:]))
    feat = feat.view(t_len, n, self.feat_dim)
    outs = []
    for t in range(t_len):
      if reset_before is not None:
        state = reset_state(state, reset_before[t])
      out, state = self.lstm(feat[t:t + 1], state)
      outs.append(out)
    h = torch.cat(outs, dim=0).reshape(t_len * n, self.lstm_hidden)
    logits, values = self._heads(h)
    values = {c: v.view(t_len, n) for c, v in values.items()}
    return logits.view(t_len, n, self.n_actions), values, state


def reset_state(state, reset):
  """Zero both the hidden and cell state of the rows where ``reset`` is true."""
  h, c = state
  keep = 1.0 - reset.reshape(1, -1, 1).to(h.dtype)
  return (h * keep, c * keep)


def reset_before_from_done(done: torch.Tensor) -> torch.Tensor:
  """``done (T, N)`` (done[t] = the episode ended AFTER action t) -> ``(T, N)``
  "zero the state entering step t". Row 0 is always False: the state at the
  start of a rollout slice is whatever the previous slice left (already
  reset if that slice's last step ended an episode), so a rollout boundary
  falling mid-episode must NOT cut the episode here."""
  out = torch.zeros_like(done)
  out[1:] = done[:-1]
  return out


# ----------------------------------------------------------------- GAE --

def gae(rewards, values, dones, last_value, gamma, lam, episodic):
  """(T, N) Generalized Advantage Estimation.

  ``dones[t]``: the episode ended after action t. ``episodic=False``
  bootstraps through episode ends (used for the non-episodic intrinsic
  return, matching RND's convention); ``episodic=True`` zeros the bootstrap
  at episode ends (used for the extrinsic/task return).
  """
  steps = rewards.shape[0]
  adv = torch.zeros_like(rewards)
  last = torch.zeros_like(last_value)
  for t in reversed(range(steps)):
    next_v = last_value if t == steps - 1 else values[t + 1]
    nonterm = (1.0 - dones[t].float()) if episodic else torch.ones_like(dones[t], dtype=rewards.dtype)
    delta = rewards[t] + gamma * next_v * nonterm - values[t]
    last = delta + gamma * lam * nonterm * last
    adv[t] = last
  return adv, adv + values


def explained_variance(values, returns):
  return 1.0 - (returns - values).var() / returns.var().clamp_min(1e-12)


# ---------------------------------------------------------------- config --

@dataclasses.dataclass
class PPOConfig:
  """Final-recipe PPO hyperparameters (see configs/dmlab_3d.yaml,
  configs/atari_2d.yaml for the exact source config.json each value was
  read from)."""
  num_envs: int = 64
  num_steps: int = 128
  epochs: int = 4
  num_minibatches: int = 4
  lr: float = 1e-4
  gamma_ext: float = 0.999
  gamma_int: float = 0.99
  gae_lambda: float = 0.95
  clip_coef: float = 0.2
  ent_coef: float = 0.001
  vf_coef: float = 0.5
  max_grad_norm: float = 0.5
  norm_adv: bool = True
  ext_coef: float = 2.0
  int_coef: float = 1.0
  int_episodic: bool = False  # non-episodic intrinsic return (bootstraps through done), as in RND


def combined_advantage(adv_ext, adv_int, cfg: PPOConfig):
  """The "combined" advantage-normalization mode: combine raw per-stream
  advantages with their coefficients FIRST, normalize the result once. This
  is the default, and the only mode every final run in this project used."""
  return cfg.ext_coef * adv_ext + cfg.int_coef * adv_int


def ppo_update(agent: RecurrentActorCritic, optimizer: torch.optim.Optimizer,
               frames: torch.Tensor, actions: torch.Tensor, logp_old: torch.Tensor,
               done: torch.Tensor, adv: torch.Tensor, returns: dict,
               cfg: PPOConfig, generator=None, clip_params=None):
  """One PPO update: ``cfg.epochs`` passes over ``cfg.num_minibatches``
  minibatches of ENVIRONMENTS (not of individual timesteps -- an LSTM policy
  needs whole, contiguous env trajectories to replay its hidden state
  correctly), as in cleanRL's ``ppo_atari_lstm.py``. The body (encoder) is
  called FRESH, per minibatch, inside ``agent.sequence`` -- never precomputed
  outside this function -- so a gradient reaches the body's own parameters
  exactly where the original design intends (see ``RecurrentActorCritic``'s
  docstring for the ``||z||^2`` arm's "LIVE" wiring).

  Args:
    frames: (T, N, *frame_shape) raw observations for every step in the
      rollout.
    actions: (T, N) long discrete actions taken.
    logp_old: (T, N) log pi_old(action | state) at collection time.
    done: (T, N) bool, done[t] = episode ended after action t.
    adv: (T, N) the COMBINED advantage (see ``combined_advantage``).
    returns: {"ext": (T, N), "int": (T, N)} GAE returns, one per critic head.
    generator: torch.Generator for the environment-minibatch permutation
      (pass a seeded one for reproducibility).
    clip_params: parameters to pass to gradient clipping; defaults to
      ``agent.parameters()`` (which already includes the body's parameters
      when the body is a registered submodule, as it always is here).

  Returns:
    dict of float diagnostics (policy loss, value loss, entropy, approx KL,
    grad norm), averaged over the epochs*minibatches updates performed.
  """
  t_len, n = frames.shape[0], frames.shape[1]
  n_mb = int(cfg.num_minibatches)
  if n % n_mb:
    raise ValueError("num_minibatches must divide num_envs")
  mb_envs = n // n_mb
  reset_before = reset_before_from_done(done)
  device = frames.device
  sums = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0,
          "approx_kl": 0.0, "grad_norm": 0.0}
  n_updates = 0
  if clip_params is None:
    clip_params = list(agent.parameters())
  for _ in range(cfg.epochs):
    perm = torch.randperm(n, generator=generator, device=device) if generator is not None \
        else torch.randperm(n, device=device)
    for start in range(0, n, mb_envs):
      idx = perm[start:start + mb_envs]
      logits, values, _ = agent.sequence(frames[:, idx], None, reset_before[:, idx])
      act = actions[:, idx].reshape(-1)
      logp_all = F.log_softmax(logits.reshape(-1, agent.n_actions), dim=-1)
      new_logp = logp_all.gather(1, act[:, None]).squeeze(1)
      entropy = -(logp_all.exp() * logp_all).sum(-1).mean()
      old_logp = logp_old[:, idx].reshape(-1)
      logratio = new_logp - old_logp
      ratio = logratio.exp()
      mb_adv = adv[:, idx].reshape(-1)
      if cfg.norm_adv:
        mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
      pg = torch.max(-mb_adv * ratio,
                      -mb_adv * torch.clamp(ratio, 1.0 - cfg.clip_coef, 1.0 + cfg.clip_coef)).mean()
      vloss = sum(0.5 * ((values[c].reshape(-1) - returns[c][:, idx].reshape(-1)) ** 2).mean()
                  for c in returns)
      loss = pg - cfg.ent_coef * entropy + cfg.vf_coef * vloss
      optimizer.zero_grad(set_to_none=True)
      loss.backward()
      grad_norm = nn.utils.clip_grad_norm_(clip_params, cfg.max_grad_norm)
      optimizer.step()
      with torch.no_grad():
        sums["policy_loss"] += float(pg)
        sums["value_loss"] += float(vloss)
        sums["entropy"] += float(entropy)
        sums["approx_kl"] += float(((ratio - 1) - logratio).mean())
        sums["grad_norm"] += float(grad_norm)
      n_updates += 1
  return {k: v / max(n_updates, 1) for k, v in sums.items()}
