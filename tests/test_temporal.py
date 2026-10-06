"""Plain unit tests for isocover.temporal: shapes, the hinge at its
threshold, and the "moved" mask."""

import torch
import torch.nn.functional as F

from isocover import temporal


def _frames(b, moved=True, seed=0):
  g = torch.Generator().manual_seed(seed)
  o_prev = torch.randint(0, 255, (b, 8, 8, 3), generator=g, dtype=torch.uint8)
  if moved:
    o_mid = (o_prev.float() + torch.randint(0, 80, (b, 8, 8, 3), generator=g).float()).clamp(0, 255).to(torch.uint8)
    o_next = (o_mid.float() + torch.randint(0, 80, (b, 8, 8, 3), generator=g).float()).clamp(0, 255).to(torch.uint8)
  else:
    o_mid, o_next = o_prev.clone(), o_prev.clone()
  return o_prev, o_mid, o_next


def test_frame_moves_shapes_and_dtype():
  o_prev, o_mid, o_next = _frames(16)
  mi, mo = temporal.frame_moves(o_prev, o_mid, o_next)
  assert mi.shape == (16,) and mo.shape == (16,)
  assert mi.dtype == torch.bool and mo.dtype == torch.bool


def test_frame_moves_all_false_when_nothing_changes():
  o_prev, o_mid, o_next = _frames(16, moved=False)
  mi, mo = temporal.frame_moves(o_prev, o_mid, o_next)
  assert not bool(mi.any())
  assert not bool(mo.any())


def test_cosine_hinge_zero_when_aligned_with_tau():
  # Every pair exactly at the threshold (cos == tau) should cost exactly 0:
  # the hinge is max(0, tau - cos), which is 0 at cos == tau.
  b, d, tau = 32, 4, 0.9
  torch.manual_seed(0)
  zm = F.normalize(torch.randn(b, d), dim=-1)
  # build zp at angle arccos(tau) from zm, in a random perpendicular plane
  perp = torch.randn(b, d)
  perp = perp - (perp * zm).sum(-1, keepdim=True) * zm
  perp = F.normalize(perp, dim=-1)
  theta = torch.acos(torch.tensor(tau))
  zp = (torch.cos(theta) * zm + torch.sin(theta) * perp)
  zn = zm.clone()
  o_prev, o_mid, o_next = _frames(b, moved=True)
  pen, diag = temporal.cosine_hinge_loss(zp, zm, zn, o_prev, o_mid, o_next, tau=tau, min_moved=1)
  # cos(zm, zp) == tau exactly (up to float error) -> that leg's hinge is 0;
  # cos(zn, zm) == 1 (zn==zm) -> also 0. So the mean penalty should be ~0.
  assert float(pen) < 1e-4


def test_cosine_hinge_positive_when_perpendicular():
  b, d = 32, 4
  torch.manual_seed(1)
  zm = F.normalize(torch.randn(b, d), dim=-1)
  perp = torch.randn(b, d)
  perp = perp - (perp * zm).sum(-1, keepdim=True) * zm
  zp = F.normalize(perp, dim=-1)  # exactly perpendicular: cos == 0
  zn = zm.clone()
  o_prev, o_mid, o_next = _frames(b, moved=True)
  pen, _ = temporal.cosine_hinge_loss(zp, zm, zn, o_prev, o_mid, o_next, tau=0.9, min_moved=1)
  assert float(pen) > 0.3  # roughly tau/2 averaged over the two legs


def test_cosine_hinge_returns_zero_below_min_moved():
  b = 4
  zp, zm, zn = torch.randn(b, 4), torch.randn(b, 4), torch.randn(b, 4)
  o_prev, o_mid, o_next = _frames(b, moved=False)  # nothing moved
  pen, diag = temporal.cosine_hinge_loss(zp, zm, zn, o_prev, o_mid, o_next, tau=0.9, min_moved=8)
  assert float(pen) == 0.0


def test_cosine_variance_zero_when_all_pairs_identical():
  # Every row of the batch has the SAME (zp, zm, zn) triple, AND zp == zn, so
  # both legs' cosines (cos(zm,zp) and cos(zn,zm) == cos(zp,zm), cosine
  # similarity being symmetric) are equal to each other as well as constant
  # across the batch -- the variance across all of cos/ must be exactly 0.
  b, d = 16, 4
  torch.manual_seed(2)
  zm1, zp1 = torch.randn(1, d), torch.randn(1, d)
  zn1 = zp1.clone()
  zp, zm, zn = zp1.expand(b, d).clone(), zm1.expand(b, d).clone(), zn1.expand(b, d).clone()
  o_prev, o_mid, o_next = _frames(b, moved=True)
  var, _ = temporal.cosine_variance_loss(zp, zm, zn, o_prev, o_mid, o_next, min_moved=1)
  assert float(var) < 1e-6
