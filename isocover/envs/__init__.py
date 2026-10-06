"""Thin environment wrappers used by this project's final runs.

``atari.py`` wraps Atari (Montezuma's Revenge, Venture) via ALE/gymnasium.
``dmlab.py`` wraps a DeepMind Lab 3D maze.

Both expose the same small vectorized-environment interface:

  reset() -> (images, infos)
  step(actions) -> (images, rewards, dones, infos)

with ``num_envs`` and ``single_action_space_n`` attributes, and the
"arrived frame" convention: on the step that ends an episode, ``info``
carries the true final observation under ``info["arrived_image"]`` (since the
returned ``images`` row for that env has already been reset to the START of
the NEXT episode), so a reward defined on "the frame the action arrived at"
is never paid on the wrong frame.
"""
