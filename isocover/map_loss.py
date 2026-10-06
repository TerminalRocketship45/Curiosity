"""SigRegMapLoss: the full encoder loss, combining SIGReg and the temporal
losses with the exact weights the final runs used.

  total = sigreg_weight   * SIGReg_sw2(z_mid)                        # 0.2, always
        + temporal_weight * cosine_hinge(z_prev, z_mid, z_next)      # 0.003, always
        + cos_var_weight  * cosine_variance(z_prev, z_mid, z_next)   # 0.005, DMLab and 2D games
        + dyn_weight      * dynamics_mse(z_mid, z_next, action_mid)  # 0.1, 2D games only (0 on DMLab)

See configs/dmlab_3d.yaml and configs/atari_2d.yaml for exactly which terms
are on for which environment, sourced from this project's real final run
configs (cited in each config file's header comment).
"""

from __future__ import annotations

import dataclasses

import torch

from isocover import seeding, sigreg, temporal, dynamics


@dataclasses.dataclass
class MapLossConfig:
  """Weights and options for ``SigRegMapLoss``. Defaults are the final recipe
  (see isocover_zsq_definition / configs/*.yaml): SIGReg sw2 weight 0.2 at
  1024 slices, cosine-hinge weight 0.003 at tau 0.9. ``cos_var_weight`` and
  ``dyn_weight`` default to 0 (off); set them for the 2D-game recipe."""
  d: int = 1024
  n_slices: int = 1024
  sigreg_weight: float = 0.2
  temporal_weight: float = 0.003
  cos_tau: float = 0.9
  cos_var_weight: float = 0.0
  dyn_weight: float = 0.0
  dyn_hidden: int = 512
  n_actions: int | None = None  # required if dyn_weight > 0


class SigRegMapLoss:
  """Stateful loss: owns the slice-direction RNG (so successive calls draw a
  reproducible sequence of random projections from one seed) and, when
  ``dyn_weight > 0``, the dynamics predictor MLP and its parameters (the
  caller must add ``dyn_parameters()`` to whatever optimizer trains the
  encoder, alongside the encoder's own parameters).
  """

  def __init__(self, cfg: MapLossConfig, seed: int = 0, device="cpu"):
    self.cfg = cfg
    self.device = torch.device(device)
    self.gen = torch.Generator(device=self.device)
    self.gen.manual_seed(seeding.derive(int(seed), "encoder/slices"))
    self.dyn = None
    if cfg.dyn_weight > 0.0:
      if cfg.n_actions is None:
        raise ValueError("MapLossConfig.dyn_weight > 0 needs n_actions set")
      self.dyn = dynamics.make_dynamics_mlp(
          cfg.d, cfg.n_actions, cfg.dyn_hidden,
          seed=seeding.derive(int(seed), "encoder/dynamics")).to(self.device)

  def dyn_parameters(self):
    return [] if self.dyn is None else list(self.dyn.parameters())

  def __call__(self, z_prev: torch.Tensor, z_mid: torch.Tensor,
               z_next: torch.Tensor, o_prev: torch.Tensor, o_mid: torch.Tensor,
               o_next: torch.Tensor, action_mid: torch.Tensor | None = None):
    """Returns (total_loss, diagnostics_dict). ``diagnostics_dict`` values are
    detached scalar tensors, safe to ``.item()`` for logging."""
    cfg = self.cfg
    reg = sigreg.sigreg_sw2(z_mid.float(), cfg.n_slices, self.gen)
    total = cfg.sigreg_weight * reg
    diag = {"sigreg": reg.detach()}

    pen, cdiag = temporal.cosine_hinge_loss(z_prev, z_mid, z_next, o_prev,
                                             o_mid, o_next, tau=cfg.cos_tau)
    total = total + cfg.temporal_weight * pen
    diag["cosine_hinge"] = pen.detach()
    diag.update(cdiag)

    if cfg.cos_var_weight > 0.0:
      penv, vdiag = temporal.cosine_variance_loss(z_prev, z_mid, z_next,
                                                   o_prev, o_mid, o_next)
      total = total + cfg.cos_var_weight * penv
      diag["cosine_variance"] = penv.detach()
      diag.update(vdiag)

    if cfg.dyn_weight > 0.0:
      if action_mid is None:
        raise ValueError("MapLossConfig.dyn_weight > 0 needs action_mid "
                          "(the action taken at z_mid's timestep)")
      dloss = dynamics.dynamics_loss(self.dyn, z_mid, z_next, action_mid,
                                      cfg.n_actions)
      total = total + cfg.dyn_weight * dloss
      diag["dynamics"] = dloss.detach()

    diag["loss"] = total.detach()
    diag["mean_sq_norm"] = z_mid.detach().pow(2).sum(dim=1).mean()
    return total, diag

  def state_dict(self):
    return {"slice_generator": self.gen.get_state(),
            "dyn": None if self.dyn is None else self.dyn.state_dict()}

  def load_state_dict(self, state):
    if "slice_generator" in state:
      self.gen.set_state(state["slice_generator"])
    if self.dyn is not None and state.get("dyn") is not None:
      self.dyn.load_state_dict(state["dyn"])
