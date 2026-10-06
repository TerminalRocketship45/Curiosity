"""Explicit, reproducible seeding utilities.

House rule used throughout this codebase: every random operation takes an
explicit integer seed, and no code calls the bare global ``np.random`` or
``torch`` generators without first seeding them. ``derive`` turns one master
seed plus a short role string into a distinct, stable sub-seed, so a single
``--seed`` flag on the command line can seed a dozen independent components
(the encoder's weight init, its slice directions, the policy's weight init,
each environment worker, ...) without the caller maintaining a table of
magic numbers, and without those components accidentally sharing a stream.

The hash is SHA-256 based rather than Python's built-in ``hash()``, which is
randomly salted per process: a sub-seed must be identical every time the same
``(master_seed, role)`` pair is derived, including on a resumed run in a
fresh process, or two runs with the same ``--seed`` would silently diverge.
"""

from __future__ import annotations

import hashlib

_MAX = 2 ** 31 - 1


def derive(master_seed: int, role: str) -> int:
  """Return a deterministic sub-seed for ``role`` derived from ``master_seed``."""
  if not isinstance(master_seed, int):
    raise TypeError("master_seed must be an int, got %r" % type(master_seed))
  payload = ("%d|%s" % (int(master_seed), role)).encode("utf-8")
  digest = hashlib.sha256(payload).digest()
  return int.from_bytes(digest[:4], "big") % _MAX


def seed_everything(master_seed: int, deterministic: bool = False) -> dict:
  """Seed Python's ``random``, NumPy, and Torch (CPU and CUDA) from one master
  seed, each with its own derived sub-seed so they do not share a stream.

  Returns the dict of sub-seeds actually used, which is worth recording in a
  run's config.json alongside ``master_seed`` for a full audit trail.

  ``deterministic=True`` additionally asks cuDNN for deterministic algorithms
  (``torch.backends.cudnn.deterministic = True``, ``benchmark = False``).
  This makes convolution kernels pick slower, deterministic implementations
  and is noticeably slower on GPU; it also does not guarantee bit-identical
  results across different GPU models, CUDA/cuDNN versions, or when any op
  without a deterministic GPU implementation is used (PyTorch will raise in
  that case if ``torch.use_deterministic_algorithms(True)`` is also set,
  which this function does not set by default -- see the README's
  reproducibility notes).
  """
  import random

  import numpy as np
  import torch

  sub = {
      "python": derive(master_seed, "python"),
      "numpy": derive(master_seed, "numpy"),
      "torch_cpu": derive(master_seed, "torch_cpu"),
      "torch_cuda": derive(master_seed, "torch_cuda"),
  }
  random.seed(sub["python"])
  np.random.seed(sub["numpy"])
  torch.manual_seed(sub["torch_cpu"])
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(sub["torch_cuda"])
  if deterministic:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
  return sub


def torch_generator(master_seed: int, role: str, device="cpu"):
  """Return an explicitly-seeded ``torch.Generator`` on ``device``."""
  import torch
  g = torch.Generator(device=device)
  g.manual_seed(derive(master_seed, role))
  return g
