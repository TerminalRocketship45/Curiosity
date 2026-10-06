"""A fixed DeepMind Lab 3D maze as a vectorized environment.

Level: ``contributed/dmlab30/explore_goal_locations_large`` (140 open cells).
The maze's walls, goal, and start position are all drawn from the episode's
RESET SEED, not from the level name, so resetting with the SAME seed every
episode gives a single fixed maze that coverage can be pooled over (exactly
as this project's MiniGrid and continuous-maze experiments pool coverage over
one fixed layout). "Maze 999" in this project's run names means
``explore_goal_locations_large`` reset with seed 999; the default maze uses
seed 12345. Different seeds give genuinely different wall layouts (not just a
different goal placement), so an encoder pretrained on one maze seed must
never be reused on another.

Action set: the 15-action "PopArt" discretization of DMLab's native 7-D
continuous action (look yaw/pitch, strafe, move, fire, jump, crouch), as used
by IMPALA/PopArt-style DMLab agents (Hessel et al., 2019,
"Multi-task Deep Reinforcement Learning with PopArt",
github.com/deepmind/lab, ``game_scripts/common/actions.lua`` defines the
native action; this discretization is the standard public one reused across
many DMLab agent implementations).

Rendering note: on headless nodes with software (OSMesa) rendering and no
system EGL/GL, DeepMind Lab's default PBO (pixel buffer object) readback path
can abort; passing ``use_pbos='false'`` selects the ``glReadPixels`` fallback,
which works in that environment and is otherwise a no-op.

Position (``DEBUG.POS.TRANS`` / ``DEBUG.MAZE.LAYOUT``) is MEASUREMENT ONLY:
it comes back on a side channel for the coverage counters and must never
enter a policy or an encoder input.
"""

from __future__ import annotations

import multiprocessing as mp

import numpy as np

DEFAULT_LEVEL = "contributed/dmlab30/explore_goal_locations_large"
DEFAULT_RESET_SEED = 12345
ACTION_REPEAT = 4
SCREEN = 64
N_ACTIONS = 15
LEVEL_LENGTH = 1350      # agent actions before the level itself ends
TRUNCATE_AT = 1349       # one earlier, so the last frame is always renderable
FRAME_SHAPE = (SCREEN, SCREEN, 3)

# The 15-action PopArt discretization: (look_yaw, look_pitch, strafe, move,
# fire, jump, crouch), DeepMind Lab's native 7-D action.
POPART_ACTION_SET = [
    (0, 0, 0, 1, 0, 0, 0),      # forward
    (0, 0, 0, -1, 0, 0, 0),     # backward
    (0, 0, -1, 0, 0, 0, 0),     # strafe left
    (0, 0, 1, 0, 0, 0, 0),      # strafe right
    (-10, 0, 0, 0, 0, 0, 0),    # small look left
    (10, 0, 0, 0, 0, 0, 0),     # small look right
    (-60, 0, 0, 0, 0, 0, 0),    # large look left
    (60, 0, 0, 0, 0, 0, 0),     # large look right
    (0, 10, 0, 0, 0, 0, 0),     # look down
    (0, -10, 0, 0, 0, 0, 0),    # look up
    (-10, 0, 0, 1, 0, 0, 0),    # forward + small look left
    (10, 0, 0, 1, 0, 0, 0),     # forward + small look right
    (-60, 0, 0, 1, 0, 0, 0),    # forward + large look left
    (60, 0, 0, 1, 0, 0, 0),     # forward + large look right
    (0, 0, 0, 0, 1, 0, 0),      # fire
]


def _dmlab_worker(remote, level, reset_seed, screen, repeat, truncate_at, debug_obs):
  import deepmind_lab
  obs_names = ["RGB_INTERLEAVED"] + list(debug_obs)
  config = {"height": str(screen), "width": str(screen), "logLevel": "WARN",
            "use_pbos": "false"}
  env = deepmind_lab.Lab(level=level, observations=obs_names, config=config)
  steps = 0

  def do_reset():
    nonlocal steps
    env.reset(seed=int(reset_seed))
    steps = 0
    raw = env.observations()
    info = {}
    if debug_obs:
      info["debug"] = {name: raw[name] for name in debug_obs}
    return raw["RGB_INTERLEAVED"], info

  try:
    while True:
      cmd, data = remote.recv()
      if cmd == "reset":
        img, info = do_reset()
        remote.send((img, info))
      elif cmd == "step":
        raw_action = np.array(POPART_ACTION_SET[int(data)], dtype=np.intc)
        reward = env.step(raw_action, num_steps=repeat)
        steps += 1
        running = env.is_running()
        done = (not running) or (truncate_at is not None and steps >= truncate_at)
        if running:
          raw = env.observations()
          img = raw["RGB_INTERLEAVED"]
          info = {"debug": {name: raw[name] for name in debug_obs}} if debug_obs else {}
        else:
          img, info = None, {}
        if done:
          arrived = img  # the frame this action truly arrived at (may be None
                          # if the level itself ended mid-render; callers that
                          # need the frame should prefer a truncate_at below
                          # the level's own length, as this module's default
                          # does, so this is always a real frame in practice)
          img, reset_info = do_reset()
          info["arrived_image"] = arrived
          info.update(reset_info)
        remote.send((img, float(reward), bool(done), info))
      elif cmd == "close":
        env.close()
        remote.close()
        break
  except (KeyboardInterrupt, EOFError):
    env.close()


class DMLabVecEnv:
  """``num_envs`` DeepMind Lab processes stepped together, each resetting to
  the SAME fixed maze (``reset_seed``) every episode.

  Same interface as ``envs.atari.AtariVecEnv``: ``reset() -> (images,
  infos)``, ``step(actions) -> (images, rewards, dones, infos)``, with the
  arrived frame in ``info["arrived_image"]`` on the step that ends an
  episode. Must be constructed under ``if __name__ == "__main__":`` (spawn
  context).
  """

  def __init__(self, num_envs: int, level: str = DEFAULT_LEVEL,
               reset_seed: int = DEFAULT_RESET_SEED, screen: int = SCREEN,
               repeat: int = ACTION_REPEAT, truncate_at: int = TRUNCATE_AT,
               debug_obs=()):
    self.num_envs = int(num_envs)
    self.frame_shape = (int(screen), int(screen), 3)
    self.single_action_space_n = N_ACTIONS
    ctx = mp.get_context("spawn")
    self.remotes, work = zip(*[ctx.Pipe() for _ in range(self.num_envs)])
    self.procs = []
    for remote, w in zip(self.remotes, work):
      p = ctx.Process(target=_dmlab_worker,
                       args=(w, level, reset_seed, screen, repeat, truncate_at,
                             tuple(debug_obs)),
                       daemon=True)
      p.start()
      self.procs.append(p)
      w.close()
      del remote

  def reset(self):
    for remote in self.remotes:
      remote.send(("reset", None))
    got = [r.recv() for r in self.remotes]
    return np.stack([g[0] for g in got]), [g[1] for g in got]

  def step(self, actions):
    for remote, a in zip(self.remotes, actions):
      remote.send(("step", int(a)))
    got = [r.recv() for r in self.remotes]
    obs, rew, done, info = zip(*got)
    return (np.stack(obs), np.array(rew, np.float32),
            np.array(done, bool), list(info))

  def close(self):
    for remote in self.remotes:
      try:
        remote.send(("close", None))
      except Exception:  # noqa: BLE001
        pass
    for p in self.procs:
      p.join(timeout=5)
      if p.is_alive():
        p.terminate()
