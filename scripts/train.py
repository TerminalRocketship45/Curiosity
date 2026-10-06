#!/usr/bin/env python
"""Train PPO with the ``||z||^2`` intrinsic reward, or the plain PPO baseline
(intrinsic off), on one DMLab maze or one Atari game.

This script is the public, from-scratch reimplementation of this project's
research trainer, scoped to exactly the two arms this release ships:

  --arm zsq   the method: the policy acts on the SIGReg map's own embedding
              (``--policy-input embedding-only``), the map keeps training
              online (SIGReg + cosine-hinge [+ cosine-variance] [+ dynamics],
              see isocover/map_loss.py), and the intrinsic reward is
              ``||z||^2`` of the LIVE map (isocover/reward.py).
  --arm ppo   the baseline: no intrinsic reward, no map at all -- a plain
              conv trunk (isocover/ppo.py's ``PixelBody``) trained end to end
              by the extrinsic PPO loss only.

See the README's "Reproduce our results" section for copy-paste commands,
expected wall-clock time, and step budgets for each environment, and
configs/dmlab_3d.yaml / configs/atari_2d.yaml for the exact final-recipe
hyperparameter values (each cited to the real run config.json it came from).

Reproducibility. ``--seed`` seeds Python/NumPy/Torch (CPU and CUDA), the
policy's own weight initialization and action sampling, and the map's weight
initialization and SIGReg slice directions, all via derived sub-seeds (see
isocover/seeding.py). For Atari, each of the ``--num-envs`` worker processes
additionally gets its own sub-seed (``--seed + worker index``), so two runs
with the same ``--seed`` and ``--num-envs`` see the identical sequence of
resets. DMLab's maze is FIXED (every episode resets to the same
``--maze-seed`` layout, by design -- see envs/dmlab.py), so there is no
per-worker environment seed to vary there; what ``--seed`` controls on DMLab
is everything about the agent (weights, action sampling), not the
(deterministic) environment itself. ``--deterministic`` additionally asks
cuDNN for deterministic algorithms (slower; see isocover/seeding.py's
docstring for what this does and does not guarantee). Every run directory
gets a ``config.json`` recording the fully resolved arguments, the derived
sub-seeds, the git commit (if available) and the installed package versions,
so a run can be audited or reproduced later.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
  sys.path.insert(0, _ROOT)

import isocover  # noqa: E402,F401  (sets CPU thread caps before torch/numpy import)
import numpy as np  # noqa: E402
import torch  # noqa: E402

from isocover import encoder as enc_lib  # noqa: E402
from isocover import ppo as ppo_lib  # noqa: E402
from isocover import reward as reward_lib  # noqa: E402
from isocover import seeding  # noqa: E402
from isocover.map_loss import MapLossConfig, SigRegMapLoss  # noqa: E402


# ------------------------------------------------------------- environment --

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


# ------------------------------------------------------- triples for the map

def sample_triples(frames_full, done, actions, batch, generator, device):
  """``frames_full``: (T+1, N, ...) uint8, frames_full[k] is the observation
  BEFORE action k (k=0..T-1) with one extra trailing frame. ``done``: (T, N)
  bool, done[k] = action k ended an episode. ``actions``: (T, N) long, or
  None. Returns (prev, mid, next, action_mid) batches of size ``batch``,
  sampled only from triples that do not cross an episode boundary."""
  rows = frames_full.shape[0]
  valid_t = torch.arange(1, rows - 1, device=device)
  ok = ~(done[:rows - 2] | done[1:rows - 1])
  idx_t, idx_n = torch.where(ok)
  pick = torch.randint(0, idx_t.numel(), (batch,), generator=generator, device=device)
  t = valid_t[idx_t[pick]]
  n = idx_n[pick]
  prev, mid, nxt = frames_full[t - 1, n], frames_full[t, n], frames_full[t + 1, n]
  action_mid = actions[t, n] if actions is not None else None
  return prev, mid, nxt, action_mid


# ------------------------------------------------------------------- config --

def git_commit():
  try:
    out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=_ROOT,
                          capture_output=True, text=True, timeout=5)
    return out.stdout.strip() if out.returncode == 0 else None
  except Exception:  # noqa: BLE001
    return None


def package_versions():
  versions = {}
  for name in ("torch", "numpy", "gymnasium", "ale_py"):
    try:
      mod = __import__(name)
      versions[name] = getattr(mod, "__version__", "unknown")
    except ImportError:
      versions[name] = None
  try:
    import deepmind_lab  # noqa: F401
    versions["deepmind_lab"] = "installed"
  except ImportError:
    versions["deepmind_lab"] = None
  return versions


def build_parser():
  p = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  p.add_argument("--env", required=True, choices=("dmlab", "atari"))
  p.add_argument("--atari-task", default="montezuma", choices=("montezuma", "venture"))
  p.add_argument("--maze-seed", type=int, default=999, help="--env dmlab only")
  p.add_argument("--arm", required=True, choices=("zsq", "ppo"),
                 help="zsq = ||z||^2 (the method); ppo = the baseline, intrinsic off")
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--deterministic", action="store_true",
                 help="cudnn.deterministic=True; slower, see isocover/seeding.py")
  p.add_argument("--total-steps", type=int, required=True)
  p.add_argument("--num-envs", type=int, default=64)
  p.add_argument("--num-steps", type=int, default=128, help="rollout length per env per update")
  p.add_argument("--latent-dim", type=int, default=1024, help="--arm zsq only")
  p.add_argument("--enc-init-ckpt", default=None,
                 help="--arm zsq only: a checkpoint from scripts/pretrain_walk.py. "
                      "Required unless --allow-cold-encoder is passed.")
  p.add_argument("--allow-cold-encoder", action="store_true",
                 help="--arm zsq only: start the map from scratch (no warm start). "
                      "For smoke tests only -- every reported result used a warm start.")
  # PPO hyperparameters (defaults = the final recipe; see configs/*.yaml)
  p.add_argument("--lr", type=float, default=1e-4)
  p.add_argument("--gamma-ext", type=float, default=0.999)
  p.add_argument("--gamma-int", type=float, default=0.99)
  p.add_argument("--gae-lambda", type=float, default=0.95)
  p.add_argument("--clip-coef", type=float, default=0.2)
  p.add_argument("--ent-coef", type=float, default=0.001)
  p.add_argument("--vf-coef", type=float, default=0.5)
  p.add_argument("--max-grad-norm", type=float, default=0.5)
  p.add_argument("--epochs", type=int, default=4)
  p.add_argument("--num-minibatches", type=int, default=4)
  p.add_argument("--int-coef", type=float, default=1.0)
  p.add_argument("--ext-coef", type=float, default=2.0)
  p.add_argument("--hidden", type=int, default=512, help="LSTM hidden size")
  # The map's own losses (--arm zsq only; defaults = the final recipe)
  p.add_argument("--sigreg-weight", type=float, default=0.2)
  p.add_argument("--n-slices", type=int, default=1024)
  p.add_argument("--temporal-weight", type=float, default=0.003)
  p.add_argument("--cos-tau", type=float, default=0.9)
  p.add_argument("--cos-var-weight", type=float, default=0.005)
  p.add_argument("--dyn-weight", type=float, default=0.0,
                 help="0.1 for Atari in the final recipe, 0.0 for DMLab")
  p.add_argument("--dyn-hidden", type=int, default=512)
  p.add_argument("--enc-lr", type=float, default=1e-3)
  p.add_argument("--enc-ema-rate", type=float, default=1.0)
  p.add_argument("--enc-batch", type=int, default=512)
  p.add_argument("--enc-updates-per-ppo-update", type=int, default=16)
  p.add_argument("--store-rows", type=int, default=None,
                 help="--arm zsq only: how many of the most recent rollout "
                      "steps (per env) to keep for sampling encoder-training "
                      "triples. Default: one rollout's worth (--num-steps + 1). "
                      "The original research trainer kept a much larger "
                      "cross-rollout ring buffer (--store-rows, typically "
                      "500k-1M steps); this release keeps only the current "
                      "rollout for simplicity, which is a smaller, honestly-"
                      "documented departure from the exact original sampling "
                      "rule (see README's 'What this release simplifies').")
  p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
  p.add_argument("--logdir", required=True)
  p.add_argument("--print-every", type=int, default=10)
  p.add_argument("--ckpt-every", type=int, default=0, help="0 = only at the end")
  return p


def main(argv=None):
  args = build_parser().parse_args(argv)
  if args.arm == "zsq" and args.enc_init_ckpt is None and not args.allow_cold_encoder:
    raise SystemExit("--arm zsq needs --enc-init-ckpt (from scripts/pretrain_walk.py), "
                      "or pass --allow-cold-encoder for a smoke test")
  os.makedirs(args.logdir, exist_ok=True)

  sub_seeds = seeding.seed_everything(args.seed, deterministic=args.deterministic)
  dev = torch.device(args.device)

  vec, frame_shape, n_actions = build_env(args)
  N, T = args.num_envs, args.num_steps

  # --------------------------------------------------------------- the map --
  encoder = loss_fn = enc_optimizer = None
  if args.arm == "zsq":
    online, ema = enc_lib.make_encoder_pair(
        frame_shape, args.latent_dim, seed=seeding.derive(args.seed, "encoder/init"),
        device=dev)
    if args.enc_init_ckpt:
      payload = torch.load(args.enc_init_ckpt, map_location=dev)
      if payload.get("data_source") != "walk":
        raise SystemExit("--enc-init-ckpt must come from random-walk pretraining "
                          "(scripts/pretrain_walk.py); got data_source=%r"
                          % (payload.get("data_source"),))
      online.load_state_dict(payload["online"])
      ema.load_state_dict(payload["ema"])
    cfg = MapLossConfig(d=args.latent_dim, n_slices=args.n_slices,
                         sigreg_weight=args.sigreg_weight,
                         temporal_weight=args.temporal_weight, cos_tau=args.cos_tau,
                         cos_var_weight=args.cos_var_weight, dyn_weight=args.dyn_weight,
                         dyn_hidden=args.dyn_hidden,
                         n_actions=(n_actions if args.dyn_weight > 0 else None))
    loss_fn = SigRegMapLoss(cfg, seed=seeding.derive(args.seed, "encoder/loss"), device=dev)
    enc_optimizer = torch.optim.AdamW(
        list(online.parameters()) + loss_fn.dyn_parameters(), lr=args.enc_lr)
    body, feat_dim = online, args.latent_dim
    critics = ("ext", "int")
  else:
    body = ppo_lib.PixelBody(frame_shape, hidden=args.hidden,
                              seed=seeding.derive(args.seed, "ppo/body"))
    body = body.to(dev)
    feat_dim = args.hidden
    critics = ("value",)

  agent = ppo_lib.RecurrentActorCritic(
      body, feat_dim, n_actions, critics=critics, lstm_hidden=args.hidden,
      seed=seeding.derive(args.seed, "ppo/agent")).to(dev)
  optimizer = torch.optim.Adam(agent.parameters(), lr=args.lr, eps=1e-5)

  cfg_ppo = ppo_lib.PPOConfig(
      num_envs=N, num_steps=T, epochs=args.epochs, num_minibatches=args.num_minibatches,
      lr=args.lr, gamma_ext=args.gamma_ext, gamma_int=args.gamma_int,
      gae_lambda=args.gae_lambda, clip_coef=args.clip_coef, ent_coef=args.ent_coef,
      vf_coef=args.vf_coef, max_grad_norm=args.max_grad_norm, norm_adv=True,
      ext_coef=args.ext_coef, int_coef=args.int_coef, int_episodic=False)

  int_norm = reward_lib.IntrinsicRewardNormalizer(args.gamma_int, N) if args.arm == "zsq" else None
  mb_gen = torch.Generator(device=dev)
  mb_gen.manual_seed(seeding.derive(args.seed, "ppo/minibatch"))
  tri_gen = torch.Generator(device=dev)
  tri_gen.manual_seed(seeding.derive(args.seed, "encoder/triples"))

  # --------------------------------------------------------------- config.json
  resolved = vars(args).copy()
  resolved.update({"sub_seeds": sub_seeds, "git_commit": git_commit(),
                    "package_versions": package_versions(),
                    "frame_shape": list(frame_shape), "n_actions": n_actions})
  with open(os.path.join(args.logdir, "config.json"), "w") as f:
    json.dump(resolved, f, indent=2, default=str)

  # ------------------------------------------------------------------ loop --
  try:
    obs, _ = vec.reset()
    cur_frame = torch.as_tensor(obs, device=dev)
    lstm_state = None
    global_step = 0
    update = 0
    t0 = time.time()
    while global_step < args.total_steps:
      update += 1
      frames = torch.empty((T, N) + frame_shape, dtype=torch.uint8, device=dev)
      arrived = torch.empty((T, N) + frame_shape, dtype=torch.uint8, device=dev)
      actions_t = torch.empty((T, N), dtype=torch.long, device=dev)
      logp_t = torch.empty((T, N), device=dev)
      done_t = torch.zeros((T, N), dtype=torch.bool, device=dev)
      values_t = {c: torch.empty((T, N), device=dev) for c in critics}
      r_ext = torch.zeros((T, N), device=dev)
      r_int_raw = torch.zeros((T, N), device=dev)
      start_state = lstm_state

      for t in range(T):
        frames[t] = cur_frame
        with torch.no_grad():
          logits, values, lstm_state = agent.step(cur_frame, lstm_state)
          log_probs = torch.log_softmax(logits, dim=-1)
          action = torch.multinomial(log_probs.exp(), 1).squeeze(1)
          logp = log_probs.gather(1, action[:, None]).squeeze(1)
        for c in critics:
          values_t[c][t] = values[c]
        actions_t[t] = action
        logp_t[t] = logp
        obs, rew, done, infos = vec.step(action.cpu().numpy())
        done_tensor = torch.as_tensor(done, device=dev)
        done_t[t] = done_tensor
        r_ext[t] = torch.as_tensor(rew, device=dev)
        arrived_np = np.stack([
            infos[i]["arrived_image"] if done[i] else obs[i] for i in range(N)])
        arrived[t] = torch.as_tensor(arrived_np, device=dev)
        cur_frame = torch.as_tensor(obs, device=dev)
        lstm_state = ppo_lib.reset_state(lstm_state, done_tensor)
        global_step += N
        if args.arm == "zsq":
          with torch.no_grad():
            # The reward always reads the EMA copy (enc_ema_rate=1.0, the
            # final recipe, makes it an exact same-step mirror of the online
            # network -- see encoder.py's docstring).
            r_int_raw[t] = reward_lib.zsq_reward(ema, arrived[t])

      with torch.no_grad():
        _, last_values, _ = agent.step(cur_frame, lstm_state)

      returns = {}
      if args.arm == "ppo":
        adv_e, returns["value"] = ppo_lib.gae(r_ext, values_t["value"], done_t,
                                               last_values["value"], args.gamma_ext,
                                               args.gae_lambda, True)
        combined_adv = adv_e
      else:
        r_int_norm = int_norm.normalize(r_int_raw)
        adv_e, returns["ext"] = ppo_lib.gae(r_ext, values_t["ext"], done_t,
                                             last_values["ext"], args.gamma_ext,
                                             args.gae_lambda, True)
        adv_i, returns["int"] = ppo_lib.gae(r_int_norm, values_t["int"], done_t,
                                             last_values["int"], args.gamma_int,
                                             args.gae_lambda, cfg_ppo.int_episodic)
        combined_adv = ppo_lib.combined_advantage(adv_e, adv_i, cfg_ppo)

      diag = ppo_lib.ppo_update(agent, optimizer, frames, actions_t, logp_t,
                                 done_t, combined_adv, returns, cfg_ppo, generator=mb_gen)

      enc_diag = {}
      if args.arm == "zsq":
        frames_full = torch.cat([frames, arrived[-1:]], dim=0)  # (T+1, N, ...)
        for _ in range(args.enc_updates_per_ppo_update):
          prev, mid, nxt, action_mid = sample_triples(
              frames_full, done_t, actions_t if args.dyn_weight > 0 else None,
              args.enc_batch, tri_gen, dev)
          z_all = online(torch.cat([prev, mid, nxt], dim=0))
          b = mid.shape[0]
          zp, zm, zn = z_all[:b], z_all[b:2 * b], z_all[2 * b:]
          total, enc_diag = loss_fn(zp, zm, zn, prev, mid, nxt, action_mid=action_mid)
          enc_optimizer.zero_grad(set_to_none=True)
          total.backward()
          enc_optimizer.step()
          enc_lib.ema_update(ema, online, args.enc_ema_rate)

      if update % args.print_every == 0:
        sps = global_step / (time.time() - t0)
        msg = ("update %d  step %d/%d  %.0f sps  policy_loss %.4f  value_loss %.4f  "
               "entropy %.4f  ep_ext_reward_mean %.3f"
               % (update, global_step, args.total_steps, sps, diag["policy_loss"],
                  diag["value_loss"], diag["entropy"], float(r_ext.sum(0).mean())))
        if enc_diag:
          msg += "  sigreg %.4f  mean_sq_norm %.1f" % (
              float(enc_diag["sigreg"]), float(enc_diag["mean_sq_norm"]))
        print(msg, flush=True)

      if args.ckpt_every and update % args.ckpt_every == 0:
        _save_checkpoint(args.logdir, "latest", agent, optimizer, online if args.arm == "zsq" else None,
                          ema if args.arm == "zsq" else None, enc_optimizer)
  finally:
    vec.close()

  _save_checkpoint(args.logdir, "final", agent, optimizer, online if args.arm == "zsq" else None,
                    ema if args.arm == "zsq" else None, enc_optimizer)
  with open(os.path.join(args.logdir, "done.json"), "w") as f:
    json.dump({"global_step": global_step, "elapsed_s": time.time() - t0}, f, indent=2)
  print("done: %d steps in %.1f s" % (global_step, time.time() - t0))


def _save_checkpoint(logdir, tag, agent, optimizer, online, ema, enc_optimizer):
  payload = {"agent": agent.state_dict(), "optimizer": optimizer.state_dict()}
  if online is not None:
    payload["encoder_online"] = online.state_dict()
    payload["encoder_ema"] = ema.state_dict()
    payload["enc_optimizer"] = enc_optimizer.state_dict()
  torch.save(payload, os.path.join(logdir, "ckpt_%s.pt" % tag))


if __name__ == "__main__":
  main()
