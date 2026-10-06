"""Reproducibility checks: the same seed must give byte-identical results.

A full environment run (DMLab/Atari) is slow and needs optional dependencies
(deepmind_lab, ale_py) this test suite does not require, so these tests
instead check the two things that actually determine whether two runs with
the same ``--seed`` diverge: (1) ``isocover.seeding`` itself, and (2) a fixed
synthetic batch pushed through the map loss and through one PPO update,
on CPU, with no environment involved. If these are bit-identical across two
independently-seeded runs, a real training run with the same seed will not
diverge for any reason internal to this package (environment-side
non-determinism, e.g. from multiprocessing scheduling, is a separate concern
documented in the README).
"""

from __future__ import annotations

import copy

import torch

from isocover import ppo as ppo_lib
from isocover import seeding
from isocover.map_loss import MapLossConfig, SigRegMapLoss


def test_seed_everything_reproducible():
  seeding.seed_everything(123)
  import random
  import numpy as np
  a = (random.random(), np.random.rand(), torch.rand(3))
  seeding.seed_everything(123)
  b = (random.random(), np.random.rand(), torch.rand(3))
  assert a[0] == b[0]
  assert a[1] == b[1]
  assert torch.equal(a[2], b[2])


def test_derive_is_deterministic_and_role_sensitive():
  assert seeding.derive(7, "a") == seeding.derive(7, "a")
  assert seeding.derive(7, "a") != seeding.derive(7, "b")
  assert seeding.derive(7, "a") != seeding.derive(8, "a")


def _fixed_batch(seed, d=16, b=32, n_actions=4):
  g = torch.Generator().manual_seed(seed)
  frame_shape = (64, 64, 3)  # large enough for the conv trunk's 8/4/3 kernels
  prev = torch.randint(0, 255, (b,) + frame_shape, generator=g, dtype=torch.uint8)
  mid = torch.randint(0, 255, (b,) + frame_shape, generator=g, dtype=torch.uint8)
  nxt = torch.randint(0, 255, (b,) + frame_shape, generator=g, dtype=torch.uint8)
  action_mid = torch.randint(0, n_actions, (b,), generator=g)
  return prev, mid, nxt, action_mid


def test_map_loss_reproducible_on_fixed_batch():
  seed = 42
  cfg = MapLossConfig(d=16, n_slices=32, sigreg_weight=0.2, temporal_weight=0.003,
                       cos_tau=0.9, cos_var_weight=0.005, dyn_weight=0.1,
                       dyn_hidden=16, n_actions=4)
  prev, mid, nxt, action_mid = _fixed_batch(0)

  def run():
    torch.manual_seed(seed)
    loss_fn = SigRegMapLoss(copy.deepcopy(cfg), seed=seed, device="cpu")
    from isocover import encoder as enc_lib
    net = enc_lib.make_conv_encoder((64, 64, 3), cfg.d, seed=seed)
    z_all = net(torch.cat([prev, mid, nxt], dim=0))
    b = mid.shape[0]
    zp, zm, zn = z_all[:b], z_all[b:2 * b], z_all[2 * b:]
    total, diag = loss_fn(zp, zm, zn, prev, mid, nxt, action_mid=action_mid)
    return total

  total_a = run()
  total_b = run()
  assert torch.equal(total_a, total_b)


def test_ppo_update_reproducible_on_fixed_batch():
  """Two independently constructed (agent, optimizer) pairs, same seed, fed
  the exact same synthetic rollout: after one ppo_update, their parameters
  must be bit-identical (CPU)."""
  seed = 7
  t, n, feat_dim, n_actions = 6, 4, 8, 3

  class _Body(torch.nn.Module):
    def __init__(self, seed):
      super().__init__()
      torch.manual_seed(seed)
      self.lin = torch.nn.Linear(3, feat_dim)

    def forward(self, frames):
      return self.lin(frames.float().mean(dim=(1, 2)))

  def build():
    torch.manual_seed(seed)
    agent = ppo_lib.RecurrentActorCritic(_Body(seed), feat_dim, n_actions,
                                          critics=("value",), lstm_hidden=16, seed=seed)
    optimizer = torch.optim.Adam(agent.parameters(), lr=1e-3)
    return agent, optimizer

  g = torch.Generator().manual_seed(0)
  frames = torch.randint(0, 255, (t, n, 5, 5, 3), generator=g, dtype=torch.uint8)
  actions = torch.randint(0, n_actions, (t, n), generator=g)
  done = torch.zeros(t, n, dtype=torch.bool)
  logp_old = torch.zeros(t, n)
  adv = torch.randn(t, n, generator=g)
  returns = {"value": torch.randn(t, n, generator=g)}
  cfg = ppo_lib.PPOConfig(num_envs=n, num_steps=t, epochs=2, num_minibatches=2)

  results = []
  for _ in range(2):
    agent, optimizer = build()
    mb_gen = torch.Generator().manual_seed(99)
    ppo_lib.ppo_update(agent, optimizer, frames, actions, logp_old, done, adv,
                        returns, cfg, generator=mb_gen)
    results.append([p.clone() for p in agent.parameters()])

  for pa, pb in zip(*results):
    assert torch.equal(pa, pb)
