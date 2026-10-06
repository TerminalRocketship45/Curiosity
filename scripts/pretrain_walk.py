#!/usr/bin/env python
"""Pretrain the map on random-walk data (warm start before online training).

Standing protocol: the encoder is PRETRAINED on random-walk data from the
target environment, then KEEPS TRAINING once PPO starts (see train.py) --
never frozen. The data is a UNIFORM RANDOM ACTION walk and nothing else: an
Atari/DMLab state includes the emulator's/engine's full internal state, which
has no known parameterization, so "sample a state uniformly" has no referent
here, and a random walk is the only honest, non-privileged way to collect
pretraining frames.

Usage (see the README's "Reproduce our results" section for copy-paste
commands with expected wall-clock time):

    python scripts/pretrain_walk.py --env dmlab --maze-seed 999 \\
        --latent-dim 1024 --steps 4000 --out checkpoints/dmlab_maze999_walk.pt

    python scripts/pretrain_walk.py --env atari --atari-task montezuma \\
        --latent-dim 1024 --steps 4000 --out checkpoints/montezuma_walk.pt
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
  sys.path.insert(0, _ROOT)

import isocover  # noqa: E402,F401  (sets CPU thread caps before torch/numpy import)
import numpy as np  # noqa: E402
import torch  # noqa: E402

from isocover import encoder as enc_lib  # noqa: E402
from isocover import seeding  # noqa: E402
from isocover.map_loss import MapLossConfig, SigRegMapLoss  # noqa: E402


class WalkBuffer:
  """Collects a fixed number of random-walk steps per environment and serves
  random (prev, mid, next) triples that never cross an episode boundary."""

  def __init__(self, vec, frame_shape, rows: int, device, dyn_on: bool = False):
    self.N = vec.num_envs
    self.rows = int(rows)
    self.frame_shape = tuple(frame_shape)
    self.device = device
    self.frames = torch.empty((self.rows, self.N) + self.frame_shape,
                               dtype=torch.uint8, device=device)
    self.done = torch.zeros((self.rows, self.N), dtype=torch.bool, device=device)
    self.actions = (torch.zeros((self.rows, self.N), dtype=torch.long, device=device)
                     if dyn_on else None)

  def collect(self, vec, n_actions: int, seed: int):
    gen = torch.Generator(device=self.device)
    gen.manual_seed(seeding.derive(seed, "pretrain/walk"))
    obs, _ = vec.reset()
    t0, ends = time.time(), 0
    for t in range(self.rows):
      self.frames[t] = torch.as_tensor(obs, device=self.device)
      acts_t = torch.randint(0, n_actions, (self.N,), generator=gen, device=self.device)
      acts = acts_t.cpu().numpy()
      if self.actions is not None:
        self.actions[t] = acts_t
      obs, _, done, _ = vec.step(acts)
      self.done[t] = torch.as_tensor(done, device=self.device)
      ends += int(done.sum())
    return {"rows": self.rows, "env_steps": self.rows * self.N,
            "episode_ends": ends, "collect_s": time.time() - t0}

  def sample_triples(self, batch: int, generator: torch.Generator):
    """Sample ``batch`` valid (t-1, t, t+1) triples (same env, no episode
    boundary crossed between t-1 and t+1)."""
    valid_t = torch.arange(1, self.rows - 1, device=self.device)
    # A triple centered at t is valid if neither done[t-1] nor done[t] ended
    # an episode (an episode-ending step's NEXT row belongs to a new episode).
    ok = ~(self.done[:-2] | self.done[1:-1])  # (rows-2, N), aligned to valid_t - 1
    idx_t, idx_n = torch.where(ok)
    if idx_t.numel() == 0:
      raise RuntimeError("no valid triples collected -- increase --walk-rows")
    pick = torch.randint(0, idx_t.numel(), (batch,), generator=generator, device=self.device)
    t = valid_t[idx_t[pick]]
    n = idx_n[pick]
    prev, mid, nxt = self.frames[t - 1, n], self.frames[t, n], self.frames[t + 1, n]
    action_mid = self.actions[t, n] if self.actions is not None else None
    return prev, mid, nxt, action_mid


def build_env(args):
  if args.env == "dmlab":
    from isocover.envs.dmlab import DMLabVecEnv, DEFAULT_LEVEL, FRAME_SHAPE, N_ACTIONS
    vec = DMLabVecEnv(args.num_envs, level=DEFAULT_LEVEL, reset_seed=args.maze_seed)
    return vec, FRAME_SHAPE, N_ACTIONS
  if args.env == "atari":
    from isocover.envs.atari import AtariVecEnv, SCREEN
    vec = AtariVecEnv(args.atari_task, args.num_envs, seed=args.seed)
    return vec, (SCREEN, SCREEN, 1), vec.single_action_space_n
  raise ValueError("unknown --env %r" % (args.env,))


def main(argv=None):
  p = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  p.add_argument("--env", required=True, choices=("dmlab", "atari"))
  p.add_argument("--atari-task", default="montezuma", choices=("montezuma", "venture"))
  p.add_argument("--maze-seed", type=int, default=999, help="--env dmlab only")
  p.add_argument("--latent-dim", type=int, default=1024)
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--num-envs", type=int, default=16)
  p.add_argument("--walk-rows", type=int, default=512, help="steps per environment")
  p.add_argument("--steps", type=int, default=4000, help="encoder training steps")
  p.add_argument("--batch", type=int, default=512)
  p.add_argument("--sigreg-weight", type=float, default=0.2)
  p.add_argument("--n-slices", type=int, default=1024)
  p.add_argument("--temporal-weight", type=float, default=0.003)
  p.add_argument("--cos-tau", type=float, default=0.9)
  p.add_argument("--cos-var-weight", type=float, default=0.0)
  p.add_argument("--dyn-weight", type=float, default=0.0)
  p.add_argument("--dyn-hidden", type=int, default=512)
  p.add_argument("--lr", type=float, default=1e-3)
  p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
  p.add_argument("--deterministic", action="store_true")
  p.add_argument("--out", required=True)
  args = p.parse_args(argv)

  sub_seeds = seeding.seed_everything(args.seed, deterministic=args.deterministic)
  dev = torch.device(args.device)

  vec, frame_shape, n_actions = build_env(args)
  dyn_on = args.dyn_weight > 0.0
  try:
    buf = WalkBuffer(vec, frame_shape, args.walk_rows, dev, dyn_on=dyn_on)
    walk_stats = buf.collect(vec, n_actions, args.seed)
  finally:
    vec.close()

  online, ema = enc_lib.make_encoder_pair(
      frame_shape, args.latent_dim, seed=seeding.derive(args.seed, "encoder/init"),
      device=dev)
  cfg = MapLossConfig(d=args.latent_dim, n_slices=args.n_slices,
                       sigreg_weight=args.sigreg_weight,
                       temporal_weight=args.temporal_weight, cos_tau=args.cos_tau,
                       cos_var_weight=args.cos_var_weight, dyn_weight=args.dyn_weight,
                       dyn_hidden=args.dyn_hidden,
                       n_actions=(n_actions if dyn_on else None))
  loss_fn = SigRegMapLoss(cfg, seed=seeding.derive(args.seed, "encoder/loss"), device=dev)
  optimizer = torch.optim.AdamW(
      list(online.parameters()) + loss_fn.dyn_parameters(), lr=args.lr)
  mb_gen = torch.Generator(device=dev)
  mb_gen.manual_seed(seeding.derive(args.seed, "pretrain/minibatch"))

  t0 = time.time()
  for step in range(args.steps):
    prev, mid, nxt, action_mid = buf.sample_triples(args.batch, mb_gen)
    z_all = online(torch.cat([prev, mid, nxt], dim=0))
    b = mid.shape[0]
    zp, zm, zn = z_all[:b], z_all[b:2 * b], z_all[2 * b:]
    total, diag = loss_fn(zp, zm, zn, prev, mid, nxt, action_mid=action_mid)
    optimizer.zero_grad(set_to_none=True)
    total.backward()
    optimizer.step()
    enc_lib.ema_update(ema, online, rate=1.0)
    if step % 200 == 0 or step == args.steps - 1:
      print("step %5d / %d  loss %.4f  sigreg %.4f  mean_sq_norm %.2f (d=%d)"
            % (step, args.steps, float(diag["loss"]), float(diag["sigreg"]),
               float(diag["mean_sq_norm"]), args.latent_dim))

  os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
  torch.save({
      "online": online.state_dict(),
      "ema": ema.state_dict(),
      "loss": loss_fn.state_dict(),
      "steps": args.steps,
      "data_source": "walk",
      "env": args.env,
      "atari_task": args.atari_task if args.env == "atari" else None,
      "maze_seed": args.maze_seed if args.env == "dmlab" else None,
      "latent_dim": args.latent_dim,
      "frame_shape": list(frame_shape),
      "config": vars(args),
      "sub_seeds": sub_seeds,
      "walk_stats": walk_stats,
  }, args.out)
  print("saved %s (%.1f s total, walk collection %.1f s)"
        % (args.out, time.time() - t0, walk_stats["collect_s"]))


if __name__ == "__main__":
  main()
