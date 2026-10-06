"""The dynamics loss (2D latent-game recipe only: weight 0.1).

A small MLP predicts the NEXT map point from the current one and the action
taken:

  z_hat_{t+1} = f(z_t, one_hot(a_t))
  loss = MSE(z_hat_{t+1}, z_{t+1})

Why this helps: SIGReg and the temporal loss alone constrain the SHAPE of the
cloud of z's and how smoothly consecutive points relate, but say nothing
about whether the map is actually PREDICTABLE from the agent's own actions --
a map could satisfy both and still scramble the relationship between actions
and their effects. Asking that a small model predict z_{t+1} from (z_t, a_t)
pushes the encoder toward a map where actions have a consistent, learnable
effect, which is exactly the structure a curiosity signal built on this map
should be measuring.

Gradient flow: there are NO stop-gradients here. z_t and z_{t+1} come from
the same encoder forward pass, and the dynamics loss's gradient reaches the
encoder's parameters through BOTH z_t (as the predictor's input) and z_{t+1}
(as the MSE target). This is deliberate: SIGReg is what keeps the map from
collapsing to make this loss trivially small (e.g. by mapping every
observation to the same point), not a stop-gradient trick.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def make_dynamics_mlp(d: int, n_actions: int, hidden: int = 512,
                       seed: int = 0) -> nn.Module:
  """Build the dynamics predictor ``f(z_t, one_hot(a_t)) -> z_hat_{t+1}``.

  Two hidden layers of width ``hidden`` with ReLU, orthogonal-initialized
  (std=sqrt(2) on the hidden layers, std=1.0 on the output layer -- the same
  convention used for the policy/value heads), seeded so its initial weights
  depend only on ``seed``.
  """
  with torch.random.fork_rng(devices=[]):
    torch.manual_seed(int(seed))

    def layer_init(layer, std):
      nn.init.orthogonal_(layer.weight, std)
      nn.init.constant_(layer.bias, 0.0)
      return layer

    import math
    net = nn.Sequential(
        layer_init(nn.Linear(int(d) + int(n_actions), int(hidden)), math.sqrt(2.0)), nn.ReLU(),
        layer_init(nn.Linear(int(hidden), int(hidden)), math.sqrt(2.0)), nn.ReLU(),
        layer_init(nn.Linear(int(hidden), int(d)), 1.0))
  return net


def dynamics_loss(dyn: nn.Module, z_t: torch.Tensor, z_next: torch.Tensor,
                   action_t: torch.Tensor, n_actions: int) -> torch.Tensor:
  """MSE between the predicted and actual next embedding.

  Args:
    dyn: the predictor MLP (``make_dynamics_mlp``).
    z_t, z_next: (B, d) consecutive encoder outputs.
    action_t: (B,) long tensor of discrete actions taken at time t (the
      action that led from z_t to z_next).
    n_actions: size of the discrete action space (for the one-hot).

  Returns:
    Scalar MSE loss tensor.
  """
  a_onehot = F.one_hot(action_t.long(), int(n_actions)).to(z_t.dtype)
  z_hat_next = dyn(torch.cat([z_t, a_onehot], dim=-1))
  return F.mse_loss(z_hat_next, z_next)
