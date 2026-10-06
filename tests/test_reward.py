"""Plain unit tests for isocover.reward: the ||z||^2 reward and its running
normalization."""

import numpy as np
import pytest
import torch

from isocover import reward


class _Identity(torch.nn.Module):
  def forward(self, x):
    return x


def test_zsq_reward_is_squared_norm():
  z = torch.tensor([[3.0, 4.0], [0.0, 0.0], [1.0, 1.0]])
  got = reward.zsq_reward(_Identity(), z)
  expected = torch.tensor([25.0, 0.0, 2.0])
  assert torch.allclose(got, expected)


def test_zsq_reward_nonnegative():
  torch.manual_seed(0)
  z = torch.randn(100, 16) * 5.0
  got = reward.zsq_reward(_Identity(), z)
  assert bool((got >= 0).all())


def test_running_mean_std_converges_to_batch_statistics():
  rms = reward.RunningMeanStd()
  rng = np.random.default_rng(0)
  data = rng.normal(loc=2.0, scale=3.0, size=(20000,))
  for chunk in np.array_split(data, 20):
    rms.update(chunk)
  assert rms.mean == pytest.approx(2.0, abs=0.1)
  assert rms.var == pytest.approx(9.0, rel=0.1)


def test_intrinsic_reward_normalizer_scales_toward_unit_std():
  # Realistic usage: normalize() is called once per short rollout (as in
  # scripts/train.py, --num-steps steps at a time), many times over a run,
  # not once over a huge batch. For a stationary raw reward raw_t ~
  # Uniform(0, 10) (mimicking ||z||^2's nonnegativity), the discounted-return
  # process X_t = sum_k gamma_int^k * raw_{t-k} has a known steady-state
  # variance, Var(raw) / (1 - gamma_int^2) -- this is the right ORDER OF
  # MAGNITUDE for the running scale to settle at. (The exact pooled-variance
  # estimate this class keeps never discounts the very first rollout's
  # ramp-up-from-zero transient, so it runs systematically somewhat above
  # this steady-state figure rather than converging to it exactly -- an
  # honest property of the plain Welford/pooled-variance algorithm this
  # class uses, matching the original research code's own RunningMeanStd,
  # which is checked for exact arithmetic correctness separately in
  # test_running_mean_std_converges_to_batch_statistics above.)
  torch.manual_seed(0)
  gamma_int, num_envs, rollout_len = 0.99, 8, 128
  norm = reward.IntrinsicRewardNormalizer(gamma_int, num_envs)
  last_normed = None
  for _ in range(200):  # 200 * 128 = 25,600 steps of stationary raw reward
    raw = torch.rand(rollout_len, num_envs) * 10.0  # nonnegative, like ||z||^2
    last_normed = norm.normalize(raw)
  assert last_normed.shape == (rollout_len, num_envs)
  assert bool((last_normed >= 0).all())  # scale only, never flips sign

  var_raw = 1.0 / 12.0 * 10.0 ** 2  # Var[Uniform(0, 10)]
  expected_var_return = var_raw / (1.0 - gamma_int ** 2)
  # Right order of magnitude (within a factor of 5 either way), not pinned to
  # an exact value -- see the note above on why this estimator runs high.
  assert expected_var_return / 5.0 < norm.rms.var < expected_var_return * 5.0


def test_intrinsic_reward_normalizer_no_mean_subtraction():
  # A constant positive reward stream should stay entirely nonnegative after
  # normalization (only the scale changes, never the mean/sign).
  norm = reward.IntrinsicRewardNormalizer(0.99, 4)
  raw = torch.full((500, 4), 3.0)
  normed = norm.normalize(raw)
  assert bool((normed > 0).all())


def test_intrinsic_reward_normalizer_state_dict_roundtrip():
  norm = reward.IntrinsicRewardNormalizer(0.99, 4)
  norm.normalize(torch.rand(10, 4))
  state = norm.state_dict()
  norm2 = reward.IntrinsicRewardNormalizer(0.99, 4)
  norm2.load_state_dict(state)
  assert norm2.rms.mean == norm.rms.mean
  assert norm2.rms.var == norm.rms.var
  assert np.allclose(norm2._running_return, norm._running_return)
