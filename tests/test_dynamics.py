"""Plain unit tests for isocover.dynamics."""

import torch

from isocover import dynamics


def test_make_dynamics_mlp_shapes():
  d, n_actions = 16, 5
  net = dynamics.make_dynamics_mlp(d, n_actions, hidden=32, seed=0)
  x = torch.randn(10, d + n_actions)
  out = net(x)
  assert out.shape == (10, d)


def test_make_dynamics_mlp_seed_reproducible():
  a = dynamics.make_dynamics_mlp(8, 4, hidden=16, seed=42)
  b = dynamics.make_dynamics_mlp(8, 4, hidden=16, seed=42)
  for pa, pb in zip(a.parameters(), b.parameters()):
    assert torch.equal(pa, pb)


def test_make_dynamics_mlp_different_seeds_differ():
  a = dynamics.make_dynamics_mlp(8, 4, hidden=16, seed=1)
  b = dynamics.make_dynamics_mlp(8, 4, hidden=16, seed=2)
  same = all(torch.equal(pa, pb) for pa, pb in zip(a.parameters(), b.parameters()))
  assert not same


def test_dynamics_loss_zero_for_perfect_predictor():
  d, n_actions, b = 4, 3, 10
  torch.manual_seed(0)
  z_t = torch.randn(b, d)
  action_t = torch.randint(0, n_actions, (b,))

  class _Perfect(torch.nn.Module):
    def forward(self, x):
      return x[:, :d]  # predicts z_t itself

  z_next = z_t.clone()  # so predicting z_t exactly matches the target
  loss = dynamics.dynamics_loss(_Perfect(), z_t, z_next, action_t, n_actions)
  assert float(loss) < 1e-6


def test_dynamics_loss_gradient_flows_to_both_zt_and_znext():
  # NO stop-gradients: the dynamics loss must backprop into BOTH z_t (as the
  # predictor's input) and z_next (as the MSE target) -- this project's
  # explicit design choice (see dynamics.py's module docstring).
  d, n_actions, b = 4, 3, 6
  net = dynamics.make_dynamics_mlp(d, n_actions, hidden=16, seed=0)
  z_t = torch.randn(b, d, requires_grad=True)
  z_next = torch.randn(b, d, requires_grad=True)
  action_t = torch.randint(0, n_actions, (b,))
  loss = dynamics.dynamics_loss(net, z_t, z_next, action_t, n_actions)
  loss.backward()
  assert z_t.grad is not None and torch.isfinite(z_t.grad).all()
  assert z_next.grad is not None and torch.isfinite(z_next.grad).all()
  assert z_t.grad.abs().sum() > 0
  assert z_next.grad.abs().sum() > 0
