"""Plain unit tests for isocover.sigreg: shapes, and the two extremes a
Gaussian-matching loss must get right (a true Gaussian sample scores near
zero; a collapsed / degenerate sample scores large)."""

import torch

from isocover import sigreg


def test_random_directions_are_unit_norm():
  d = sigreg.random_directions(37, 20, generator=torch.Generator().manual_seed(0))
  assert d.shape == (37, 20)
  norms = d.norm(dim=0)
  assert torch.allclose(norms, torch.ones(20), atol=1e-5)


def test_sigreg_sw2_shape_is_scalar():
  z = torch.randn(64, 16)
  loss = sigreg.sigreg_sw2(z, n_slices=32, generator=torch.Generator().manual_seed(0))
  assert loss.dim() == 0


def test_sigreg_sw2_near_zero_for_true_gaussian_samples():
  torch.manual_seed(0)
  z = torch.randn(4096, 16)  # an actual N(0, I_16) sample
  loss = sigreg.sigreg_sw2(z, n_slices=256, generator=torch.Generator().manual_seed(1))
  assert float(loss) < 0.05


def test_sigreg_sw2_large_for_collapsed_batch():
  # Every sample identical: a single point, nothing like a spread-out
  # Gaussian. Every projection is a point mass at 0, maximally far from the
  # sorted-Gaussian-quantile target in sw2's own units.
  z = torch.zeros(256, 16)
  loss = sigreg.sigreg_sw2(z, n_slices=64, generator=torch.Generator().manual_seed(2))
  assert float(loss) > 0.5


def test_sigreg_sw2_large_for_rank_one_collapse():
  # All variance along a single direction: a Gaussian along one axis, a point
  # mass (zero spread) along every other -- far from isotropic.
  torch.manual_seed(3)
  z = torch.zeros(1024, 16)
  z[:, 0] = torch.randn(1024) * 5.0
  loss_collapsed = sigreg.sigreg_sw2(z, n_slices=256, generator=torch.Generator().manual_seed(4))
  z_gauss = torch.randn(1024, 16)
  loss_gaussian = sigreg.sigreg_sw2(z_gauss, n_slices=256, generator=torch.Generator().manual_seed(4))
  assert float(loss_collapsed) > float(loss_gaussian)


def test_sigreg_sw2_is_differentiable():
  z = torch.randn(64, 8, requires_grad=True)
  loss = sigreg.sigreg_sw2(z, n_slices=16, generator=torch.Generator().manual_seed(0))
  loss.backward()
  assert z.grad is not None
  assert torch.isfinite(z.grad).all()


def test_sigreg_sw2_rejects_too_small_batch():
  z = torch.randn(1, 8)
  try:
    sigreg.sigreg_sw2(z, n_slices=4)
    assert False, "expected a ValueError for batch size 1"
  except ValueError:
    pass
