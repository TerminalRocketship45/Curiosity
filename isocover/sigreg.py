"""SIGReg: the sliced isotropic Gaussian regularizer.

Plain-language idea. We want the encoder's output z to be distributed like a
standard Gaussian, N(0, I_d): a bell curve centered at the origin, spread out
equally in every direction, with no preferred axis. Checking this directly in
d = 1024 dimensions is hard, but there is an exact, cheap equivalence (the
Cramer-Wold theorem): a distribution in R^d is N(0, I_d) if and only if EVERY
one-dimensional projection of it (dot the batch with some fixed unit vector)
is a standard 1-D Gaussian N(0, 1). So SIGReg:

  1. draws `n_slices` random unit directions in R^d ("slices"),
  2. projects the whole batch of z onto each direction (one number per sample
     per direction),
  3. measures, per direction, how far that 1-D sample is from N(0, 1),
  4. averages the per-direction scores into a single scalar loss.

This file implements that with the sliced Wasserstein-2 (squared) statistic
("sw2"), the variant used by every final run in this project (SIGReg weight
0.2, d = 1024, 1024 random slices; see configs/). Sliced Wasserstein-2 sorts
each 1-D projection and compares it, point for point, against the quantiles a
true N(0, 1) sample of the same size would have:

  sw2(x) = mean_i (x_(i) - Phi^{-1}((i - 1/2) / B))^2

where x_(i) is the i-th smallest of B projected samples and Phi^{-1} is the
inverse standard normal CDF (the quantile function). Averaged over slices,
this is the loss. Unlike a test statistic such as Cramer-von Mises, sw2 does
not grow with batch size B, which is why its weight transfers across batch
sizes without retuning.

Reference: SIGReg is the regularizer used by LeJEPA (Balestriero & LeCun,
2025, "LeJEPA: Provable and Scalable Self-Supervised Learning Without the
Heuristics", arXiv:2511.08544), which popularized this sliced-isotropic-
Gaussian family of losses for self-supervised representation learning. The
sliced Wasserstein-2 variant used here is a standard member of that family
(the "sliced Wasserstein distance" of Rabin et al., 2011/Bonneel et al.,
2015), not LeJEPA's own default statistic (LeJEPA uses a sliced Epps-Pulley
test); we use sw2 because its scale does not depend on the batch size.
"""

from __future__ import annotations

import torch

_QUANTILE_CACHE: dict = {}


def random_directions(dim: int, n_slices: int, generator=None, device=None,
                       dtype=torch.float32) -> torch.Tensor:
  """``(dim, n_slices)`` random unit columns ("slices").

  Each column is a uniformly random point on the unit sphere in R^dim,
  obtained the standard way: draw d i.i.d. N(0,1) coordinates and normalize.
  ``generator`` is a ``torch.Generator`` for reproducibility.
  """
  gen_device = generator.device if generator is not None else device
  d = torch.randn(int(dim), int(n_slices), generator=generator,
                   device=gen_device, dtype=dtype)
  d = d / d.norm(dim=0, keepdim=True).clamp_min(1e-12)
  return d if device is None else d.to(device)


def _project(z: torch.Tensor, n_slices: int, generator, directions):
  if z.dim() != 2:
    raise ValueError("z must be (B, d), got %r" % (tuple(z.shape),))
  if z.shape[0] < 2:
    raise ValueError("need at least 2 samples")
  if directions is None:
    directions = random_directions(z.shape[1], n_slices, generator,
                                    z.device, z.dtype)
  return z @ directions.to(device=z.device, dtype=z.dtype)


def gaussian_quantiles(batch: int, device, dtype) -> torch.Tensor:
  """``Phi^{-1}((i - 1/2) / B)`` for ``i = 1 .. B``, the quantiles a sorted
  sample of B draws from N(0, 1) would sit at. Cached per (batch, device,
  dtype), since the same shape is reused every training step."""
  key = (int(batch), str(device), dtype)
  if key not in _QUANTILE_CACHE:
    u = (torch.arange(1, batch + 1, dtype=torch.float64) - 0.5) / batch
    _QUANTILE_CACHE[key] = torch.special.ndtri(u).to(device=device, dtype=dtype)
  return _QUANTILE_CACHE[key]


def sigreg_sw2(z: torch.Tensor, n_slices: int = 1024, generator=None,
               directions=None) -> torch.Tensor:
  """Sliced Wasserstein-2 (squared) distance of ``z``'s distribution to
  N(0, I_d), the SIGReg variant used by every final run in this project.

  Args:
    z: (B, d) batch of encoder outputs.
    n_slices: number of random projection directions (1024 in every final
      run). A fresh set of directions is drawn every call unless
      ``directions`` is given explicitly.
    generator: torch.Generator used to draw the random directions (ignored if
      ``directions`` is given). Reusing the SAME generator across calls (as
      the trainer does) means every call advances a single seeded stream, so
      the whole run's sequence of slice directions is reproducible from one
      seed.
    directions: optional precomputed (d, n_slices) unit directions, e.g. for
      a parity test that must use the exact same slices as a reference
      implementation.

  Returns:
    Scalar tensor: mean over slices of mean_i (x_(i) - q_i)^2, where x_(i) is
    the i-th smallest projected value in the batch and q_i is the matching
    N(0, 1) quantile.
  """
  proj = _project(z, n_slices, generator, directions)
  proj_sorted, _ = torch.sort(proj, dim=0)
  q = gaussian_quantiles(proj.shape[0], z.device, z.dtype)[:, None]
  return ((proj_sorted - q) ** 2).mean()


def norm_penalty(z: torch.Tensor) -> torch.Tensor:
  """``(mean_batch ||z||^2 - d)^2``. Not part of the final recipe's loss (the
  final recipe gets its norm-pulling pressure from sw2 itself, since a batch
  whose norms are systematically too large or too small is also a batch whose
  projections are not N(0,1)); kept here only as a cheap diagnostic you can
  log alongside the loss to sanity-check that ``||z||^2`` is tracking ``d``.
  """
  return (z.pow(2).sum(dim=1).mean() - z.shape[1]) ** 2
