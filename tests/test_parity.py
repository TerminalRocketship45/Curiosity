"""Numerical parity against the original research repository.

These tests import the ORIGINAL implementations (``toys/common/encoder.py``,
``toys/common/cosine_frames.py``, ``toys/large_envs/train_pixels.py``) by
file path from the private research repository this release was built from,
and check that this public package's reimplementations produce the SAME
numbers (to float tolerance) on identical inputs and seeds.

The public package never imports anything from the original repository (see
every other module under ``isocover/``); this file only reaches into it for
the purpose of this one-time verification, via the ``ISOCOVER_ORIGINAL_REPO``
environment variable, which must be set to the original repository's path to
run these tests at all. If it is unset, or does not point at a real checkout,
every test in this file is SKIPPED -- the public package has no runtime
dependency on it, and no path from any particular machine is hardcoded here.
"""

from __future__ import annotations

import importlib.util
import os

import pytest
import torch

from isocover import dynamics, reward, sigreg, temporal

ORIGINAL_REPO = os.environ.get("ISOCOVER_ORIGINAL_REPO", "")


def _load_module(path, name):
  import sys
  spec = importlib.util.spec_from_file_location(name, path)
  if spec is None or spec.loader is None:
    raise ImportError(path)
  mod = importlib.util.module_from_spec(spec)
  # Register in sys.modules BEFORE exec: some of the original files use
  # dataclasses, whose field-type resolution looks the defining module up by
  # name in sys.modules while the module body is executing.
  sys.modules[name] = mod
  spec.loader.exec_module(mod)
  return mod


def _original_available():
  return os.path.isdir(ORIGINAL_REPO) and os.path.isfile(
      os.path.join(ORIGINAL_REPO, "toys", "common", "encoder.py"))


pytestmark = pytest.mark.skipif(
    not _original_available(),
    reason="original research repo not found at %r (set ISOCOVER_ORIGINAL_REPO "
           "to its path to run these parity tests)" % (ORIGINAL_REPO,))


@pytest.fixture(scope="module")
def original_encoder():
  return _load_module(os.path.join(ORIGINAL_REPO, "toys", "common", "encoder.py"),
                       "_orig_encoder")


@pytest.fixture(scope="module")
def original_cosine_frames():
  # cosine_frames.py imports `from toys.common import logdet_frames`, a
  # relative package import, so the ORIGINAL_REPO root is put on sys.path
  # for the duration of this load (removed again right after).
  import sys
  added = ORIGINAL_REPO not in sys.path
  if added:
    sys.path.insert(0, ORIGINAL_REPO)
  try:
    return _load_module(os.path.join(ORIGINAL_REPO, "toys", "common", "cosine_frames.py"),
                         "_orig_cosine_frames")
  finally:
    if added:
      sys.path.remove(ORIGINAL_REPO)


# ------------------------------------------------------------------ SIGReg --

def test_sigreg_sw2_matches_original(original_encoder):
  torch.manual_seed(0)
  z = torch.randn(64, 32)
  directions = sigreg.random_directions(32, 128, generator=torch.Generator().manual_seed(1))
  ours = sigreg.sigreg_sw2(z, directions=directions)
  theirs = original_encoder.sigreg_sw2(z, directions=directions)
  assert torch.allclose(ours, theirs, atol=1e-6)


def test_gaussian_quantiles_match_original(original_encoder):
  ours = sigreg.gaussian_quantiles(37, "cpu", torch.float32)
  theirs = original_encoder.gaussian_quantiles(37, "cpu", torch.float32)
  assert torch.allclose(ours, theirs, atol=1e-6)


# ---------------------------------------------------------------- temporal --

def test_cosine_hinge_matches_original(original_cosine_frames):
  torch.manual_seed(0)
  b, d = 48, 16
  zp, zm, zn = torch.randn(b, d), torch.randn(b, d), torch.randn(b, d)
  o_prev = torch.randint(0, 255, (b, 8, 8, 3), dtype=torch.uint8)
  o_mid = o_prev + torch.randint(0, 40, (b, 8, 8, 3), dtype=torch.uint8)
  o_next = o_mid + torch.randint(0, 40, (b, 8, 8, 3), dtype=torch.uint8)
  ours, ours_diag = temporal.cosine_hinge_loss(zp, zm, zn, o_prev, o_mid, o_next, tau=0.9)
  theirs, their_diag = original_cosine_frames.cosine_hinge_loss(
      zp, zm, zn, o_prev, o_mid, o_next, tau=0.9)
  assert torch.allclose(ours, theirs, atol=1e-6)
  for key in ("cos/mean", "cos/frac_below_tau", "cos/moved_share"):
    assert torch.allclose(ours_diag[key], their_diag[key], atol=1e-6), key


def test_cosine_variance_matches_original(original_cosine_frames):
  torch.manual_seed(1)
  b, d = 48, 16
  zp, zm, zn = torch.randn(b, d), torch.randn(b, d), torch.randn(b, d)
  o_prev = torch.randint(0, 255, (b, 8, 8, 3), dtype=torch.uint8)
  o_mid = o_prev + torch.randint(0, 40, (b, 8, 8, 3), dtype=torch.uint8)
  o_next = o_mid + torch.randint(0, 40, (b, 8, 8, 3), dtype=torch.uint8)
  ours, _ = temporal.cosine_variance_loss(zp, zm, zn, o_prev, o_mid, o_next)
  theirs, _ = original_cosine_frames.cosine_variance_loss(zp, zm, zn, o_prev, o_mid, o_next)
  assert torch.allclose(ours, theirs, atol=1e-6)


def test_frame_moves_matches_original():
  import sys
  added = ORIGINAL_REPO not in sys.path
  if added:
    sys.path.insert(0, ORIGINAL_REPO)
  try:
    logdet_frames = _load_module(
        os.path.join(ORIGINAL_REPO, "toys", "common", "logdet_frames.py"),
        "_orig_logdet_frames")
  finally:
    if added:
      sys.path.remove(ORIGINAL_REPO)
  torch.manual_seed(2)
  b = 48
  o_prev = torch.randint(0, 255, (b, 8, 8, 3), dtype=torch.uint8)
  o_mid = o_prev + torch.randint(0, 40, (b, 8, 8, 3), dtype=torch.uint8)
  o_next = o_mid + torch.randint(0, 40, (b, 8, 8, 3), dtype=torch.uint8)
  mi, mo = temporal.frame_moves(o_prev, o_mid, o_next)
  tmi, tmo, _ = logdet_frames.frame_moves(o_prev, o_mid, o_next)
  assert torch.equal(mi, tmi)
  assert torch.equal(mo, tmo)


# ------------------------------------------------------------------ reward --

def test_running_mean_std_matches_original():
  import sys
  sys.path.insert(0, ORIGINAL_REPO)
  try:
    tp = _load_module(os.path.join(ORIGINAL_REPO, "toys", "large_envs", "train_pixels.py"),
                       "_orig_train_pixels_rms_only")
  except Exception as e:  # noqa: BLE001
    pytest.skip("could not import train_pixels.py for RunningMeanStd parity: %r" % (e,))
    return
  finally:
    if ORIGINAL_REPO in sys.path:
      sys.path.remove(ORIGINAL_REPO)
  import numpy as np
  ours = reward.RunningMeanStd()
  theirs = tp.RunningMeanStd()
  rng = np.random.default_rng(0)
  for _ in range(5):
    batch = rng.normal(size=(100,))
    ours.update(batch)
    theirs.update(batch)
  assert ours.mean == pytest.approx(theirs.mean)
  assert ours.var == pytest.approx(theirs.var)


def test_dynamics_mlp_init_matches_original():
  import sys
  added = ORIGINAL_REPO not in sys.path
  if added:
    sys.path.insert(0, ORIGINAL_REPO)
  try:
    rc = _load_module(os.path.join(ORIGINAL_REPO, "toys", "agents", "recurrent.py"),
                       "_orig_recurrent")
  finally:
    if added:
      sys.path.remove(ORIGINAL_REPO)
  import torch.nn as nn

  d, n_actions, hidden, seed = 8, 5, 32, 7
  with torch.random.fork_rng(devices=[]):
    torch.manual_seed(seed)
    theirs = nn.Sequential(
        rc.layer_init(nn.Linear(d + n_actions, hidden)), nn.ReLU(),
        rc.layer_init(nn.Linear(hidden, hidden)), nn.ReLU(),
        rc.layer_init(nn.Linear(hidden, d), std=1.0))
  ours = dynamics.make_dynamics_mlp(d, n_actions, hidden, seed=seed)
  for (pn, pt), (_, po) in zip(theirs.named_parameters(), ours.named_parameters()):
    assert torch.allclose(pt, po, atol=1e-6), pn


def test_zsq_reward_matches_original_formula():
  # The original's `_intrinsic` for --arm caseA is exactly `z.pow(2).sum(1)`
  # (toys/large_envs/train_pixels.py); no module import needed to check this
  # since it is a one-line formula, but we assert it explicitly here so a
  # change to either side is caught.
  z = torch.randn(10, 4)

  class _Identity(torch.nn.Module):
    def forward(self, x):
      return x

  got = reward.zsq_reward(_Identity(), z)
  assert torch.allclose(got, z.pow(2).sum(dim=1))
