#!/usr/bin/env python
"""Evaluate a trained checkpoint: extrinsic score and exploration coverage.

Coverage definitions (both MEASUREMENT ONLY -- neither is ever available to
the policy or the encoder during training):
  Atari:  number of distinct values of the "room" RAM byte seen (the standard
          "visited rooms" exploration measure for Montezuma's Revenge /
          Venture).
  DMLab:  number of distinct 100-world-unit maze cells visited, computed from
          the DEBUG.POS.TRANS side channel (never fed to the agent).

Usage:
    python scripts/evaluate.py --env atari --atari-task montezuma \\
        --ckpt runs/montezuma_zsq_s0/ckpt_final.pt --arm zsq --episodes 20

    python scripts/evaluate.py --env dmlab --maze-seed 999 \\
        --ckpt runs/dmlab_zsq_s0/ckpt_final.pt --arm zsq --episodes 20
"""

from __future__ import annotations

import argparse
import json
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
  sys.path.insert(0, _ROOT)

import isocover  # noqa: E402,F401  (sets CPU thread caps before torch/numpy import)
import numpy as np  # noqa: E402
import torch  # noqa: E402

from isocover import encoder as enc_lib  # noqa: E402
from isocover import ppo as ppo_lib  # noqa: E402
from isocover import seeding  # noqa: E402

CELL = 100.0  # world units per DMLab maze cell (isocover.envs.dmlab convention)


def cell_of(pos):
  pos = np.asarray(pos, dtype=np.float64)
  return (int(np.floor(pos[0] / CELL)), int(np.floor(pos[1] / CELL)))


def build_env(args):
  if args.env == "dmlab":
    from isocover.envs.dmlab import DMLabVecEnv, DEFAULT_LEVEL, FRAME_SHAPE, N_ACTIONS
    vec = DMLabVecEnv(args.num_envs, level=DEFAULT_LEVEL, reset_seed=args.maze_seed,
                       debug_obs=("DEBUG.POS.TRANS",))
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
  p.add_argument("--maze-seed", type=int, default=999)
  p.add_argument("--arm", required=True, choices=("zsq", "ppo"))
  p.add_argument("--latent-dim", type=int, default=1024, help="--arm zsq only")
  p.add_argument("--hidden", type=int, default=512)
  p.add_argument("--ckpt", required=True)
  p.add_argument("--episodes", type=int, default=20,
                 help="total episodes, pooled across --num-envs parallel environments")
  p.add_argument("--num-envs", type=int, default=8)
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--deterministic-policy", action="store_true",
                 help="argmax action instead of sampling")
  p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
  p.add_argument("--out", default=None, help="optional path to write a JSON summary")
  args = p.parse_args(argv)

  seeding.seed_everything(args.seed)
  dev = torch.device(args.device)
  vec, frame_shape, n_actions = build_env(args)
  N = args.num_envs

  payload = torch.load(args.ckpt, map_location=dev)
  if args.arm == "zsq":
    online = enc_lib.make_conv_encoder(frame_shape, args.latent_dim, seed=0).to(dev)
    online.load_state_dict(payload["encoder_online"])
    body, feat_dim, critics = online, args.latent_dim, ("ext", "int")
  else:
    body = ppo_lib.PixelBody(frame_shape, hidden=args.hidden, seed=0).to(dev)
    feat_dim, critics = args.hidden, ("value",)
  agent = ppo_lib.RecurrentActorCritic(body, feat_dim, n_actions, critics=critics,
                                        lstm_hidden=args.hidden, seed=0).to(dev)
  agent.load_state_dict(payload["agent"])
  agent.eval()

  try:
    obs, infos = vec.reset()
    cur_frame = torch.as_tensor(obs, device=dev)
    lstm_state = None
    scores = np.zeros(N, dtype=np.float64)
    finished_scores = []
    finished_coverage = []  # len(visited_rooms) or len(visited_cells) per finished episode
    visited_rooms = [set() for _ in range(N)]
    visited_cells = [set() for _ in range(N)]
    for i, info in enumerate(infos):
      if "room" in info:
        visited_rooms[i].add(info["room"])
      if args.env == "dmlab" and "debug" in info:
        visited_cells[i].add(cell_of(info["debug"]["DEBUG.POS.TRANS"]))

    with torch.no_grad():
      while len(finished_scores) < args.episodes:
        logits, _, lstm_state = agent.step(cur_frame, lstm_state)
        if args.deterministic_policy:
          action = logits.argmax(dim=-1)
        else:
          action = torch.multinomial(torch.softmax(logits, dim=-1), 1).squeeze(1)
        obs, rew, done, infos = vec.step(action.cpu().numpy())
        scores += rew
        for i, info in enumerate(infos):
          if "room" in info:
            visited_rooms[i].add(info["room"])
          if args.env == "dmlab" and "debug" in info:
            visited_cells[i].add(cell_of(info["debug"]["DEBUG.POS.TRANS"]))
          if done[i]:
            finished_scores.append(float(scores[i]))
            coverage = (len(visited_cells[i]) if args.env == "dmlab"
                        else len(visited_rooms[i]))
            finished_coverage.append(coverage)
            scores[i] = 0.0
            visited_rooms[i] = set()
            visited_cells[i] = set()
        cur_frame = torch.as_tensor(obs, device=dev)
        lstm_state = ppo_lib.reset_state(lstm_state, torch.as_tensor(done, device=dev))
  finally:
    vec.close()

  finished_scores = finished_scores[:args.episodes]
  finished_coverage = finished_coverage[:args.episodes]
  coverage_name = "cells_visited" if args.env == "dmlab" else "rooms_visited"
  summary = {
      "env": args.env, "arm": args.arm, "episodes": len(finished_scores),
      "score_mean": float(np.mean(finished_scores)) if finished_scores else None,
      "score_std": float(np.std(finished_scores)) if finished_scores else None,
      "score_all": finished_scores,
      coverage_name + "_mean": float(np.mean(finished_coverage)) if finished_coverage else None,
      coverage_name + "_std": float(np.std(finished_coverage)) if finished_coverage else None,
      coverage_name + "_all": finished_coverage,
  }
  print(json.dumps(summary, indent=2))
  if args.out:
    with open(args.out, "w") as f:
      json.dump(summary, f, indent=2)


if __name__ == "__main__":
  main()
