"""Montezuma's Revenge / Venture as a vectorized environment.

Preprocessing matches the published hard-exploration recipe (Burda et al.,
2018, Random Network Distillation; see also the original
``openai/random-network-distillation`` and the common community port
``jcwleo/random-network-distillation-pytorch``): 84x84 grayscale, action
repeat 4, sticky actions with probability 0.25, no "done on life loss", a
4,500-agent-step episode cap, no reward clipping.

The one deliberate departure from that published recipe: NO FRAME STACK. The
published recipe stacks 4 frames so a feed-forward policy can see velocity;
this project's architecture puts all memory in an LSTM on top of the
encoder, and gives the encoder only the CURRENT frame (see ppo.py's module
docstring), so the observation here is a single ``(84, 84, 1)`` uint8 frame,
not a stack.

Implementation note: this wraps ``ale_py`` through
``gymnasium.wrappers.AtariPreprocessing`` (frame skip with max-over-last-two,
grayscale, resize), which is the standard, public way to reproduce this
preprocessing without a third-party vectorized-Atari package.
"""

from __future__ import annotations

import multiprocessing as mp
import sys

import numpy as np

ATARI_TASKS = {
    "montezuma": "ALE/MontezumaRevenge-v5",
    "venture": "ALE/Venture-v5",
}
ACTION_REPEAT = 4
STICKY_ACTION_PROB = 0.25
MAX_STEPS_PER_EPISODE = 4500  # agent steps (18,000 game frames at repeat 4)
SCREEN = 84

# Room-id RAM byte, MEASUREMENT ONLY (never fed to a policy or an encoder):
# the standard "visited rooms" exploration diagnostic. Montezuma: byte 3.
# Venture: byte 90, per the Atari Annotated RAM Interface (Anand et al. 2019,
# "Unsupervised State Representation Learning in Atari",
# atariari/benchmark/ram_annotations.py).
ROOM_RAM_BYTES = {"montezuma": 3, "venture": 90}
DEFAULT_ROOM_BYTE = 3


def make_single_env(task: str, seed: int, screen: int = SCREEN,
                     repeat: int = ACTION_REPEAT, sticky: float = STICKY_ACTION_PROB,
                     max_steps: int = MAX_STEPS_PER_EPISODE):
  """One preprocessed gymnasium env whose observation is a single ``(84, 84,
  1)`` uint8 frame (no frame stack -- see the module docstring)."""
  import ale_py
  import gymnasium as gym
  from gymnasium.wrappers import AtariPreprocessing
  gym.register_envs(ale_py)
  env = gym.make(ATARI_TASKS.get(task, task), frameskip=1,
                  repeat_action_probability=float(sticky),
                  full_action_space=False,
                  max_episode_steps=int(max_steps) * int(repeat))
  env = AtariPreprocessing(env, noop_max=30, frame_skip=int(repeat),
                            screen_size=int(screen), terminal_on_life_loss=False,
                            grayscale_obs=True, grayscale_newaxis=True,
                            scale_obs=False)
  env.reset(seed=int(seed))
  env.action_space.seed(int(seed))
  return env


def room_of(env, byte: int = DEFAULT_ROOM_BYTE) -> int:
  """Current room id from one emulator RAM byte. Evaluation/diagnostics
  only -- never part of the observation."""
  try:
    return int(env.unwrapped.ale.getRAM()[int(byte)])
  except Exception:  # noqa: BLE001
    return -1


def _atari_worker(remote, task, screen, seed, repeat, sticky, max_steps):
  env = make_single_env(task, seed, screen, repeat, sticky, max_steps)
  rb = ROOM_RAM_BYTES.get(str(task), DEFAULT_ROOM_BYTE)
  try:
    while True:
      cmd, data = remote.recv()
      if cmd == "reset":
        obs, _ = env.reset()
        remote.send((obs, {"room": room_of(env, rb)}))
      elif cmd == "step":
        obs, reward, term, trunc, _ = env.step(int(data))
        info = {"room": room_of(env, rb)}
        if term or trunc:
          info["arrived_image"] = obs
          info["terminated"] = bool(term)
          obs, _ = env.reset()
          info["reset_room"] = room_of(env, rb)
          remote.send((obs, float(reward), True, info))
        else:
          remote.send((obs, float(reward), False, info))
      elif cmd == "close":
        env.close()
        remote.close()
        break
  except (KeyboardInterrupt, EOFError):
    env.close()


class AtariVecEnv:
  """``num_envs`` ALE processes stepped together.

  ``reset() -> (images, infos)``, ``step(actions) -> (images, rewards, dones,
  infos)``. Autoreset is same-step; the true final frame of an ending episode
  is in ``info["arrived_image"]``.

  Must be constructed under ``if __name__ == "__main__":`` -- this uses the
  "spawn" multiprocessing context, which re-imports the main module in each
  worker.
  """

  def __init__(self, task: str, num_envs: int, seed: int = 0, screen: int = SCREEN,
               repeat: int = ACTION_REPEAT, sticky: float = STICKY_ACTION_PROB,
               max_steps: int = MAX_STEPS_PER_EPISODE):
    self.task = str(task)
    self.num_envs = int(num_envs)
    self.frame_shape = (int(screen), int(screen), 1)
    probe = make_single_env(task, seed, screen, repeat, sticky, max_steps)
    self.single_action_space_n = int(probe.action_space.n)
    probe.close()
    ctx = mp.get_context("spawn")
    self.remotes, work = zip(*[ctx.Pipe() for _ in range(self.num_envs)])
    self.procs = []
    for i, (remote, w) in enumerate(zip(self.remotes, work)):
      p = ctx.Process(target=_atari_worker,
                       args=(w, task, screen, seed + i, repeat, sticky, max_steps),
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


if sys.platform == "darwin":
  # macOS defaults multiprocessing to "spawn" already; nothing to do, kept
  # here only as a reminder that "fork" (Linux's default) is NOT safe with
  # ALE's internal state.
  pass
