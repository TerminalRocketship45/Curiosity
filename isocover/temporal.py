"""The temporal loss: keep consecutive map points pointing the same way.

SIGReg (sigreg.py) only constrains the overall SHAPE of the cloud of points
the encoder produces (it should look like a Gaussian); it says nothing about
how the map moves frame to frame. Left alone, SIGReg is perfectly happy with
an encoder that scrambles consecutive frames to unrelated, far-apart points,
as long as the overall cloud is still Gaussian-shaped. The temporal loss adds
the missing constraint: a step in the environment should not swing the map
point's DIRECTION (as seen from the origin) by very much.

Why the origin, and why direction (not distance). SIGReg pushes the mean of z
toward 0, so the origin is already the "center of mass" of the map -- it is
also the point curiosity measures distance from (reward = ||z||^2, see
reward.py), so it is the natural reference point for direction too. Using the
batch's own mean instead would give a noisy, moving reference the encoder
could quietly game; the origin cannot move.

Why this is motivated: the reverse triangle inequality says a single step
cannot change a vector's NORM by more than the step's own length,
  | ||z_t|| - ||z_{t-1}|| | <= ||z_t - z_{t-1}||.
The same geometry bounds the change in DIRECTION: the distance from z_{t-1} to
the ray through the origin and z_t is ||z_{t-1}|| * sin(theta), where theta is
the angle between them, and that distance is itself at most the step length:
  ||z_{t-1}|| * sin(theta) <= ||z_t - z_{t-1}||.
So bounding the angle (equivalently, keeping cos(theta) above a threshold) is
the natural way to ask "frame-to-frame steps should be small compared to how
far the map point already is from the center" -- i.e. the map should be
locally smooth in direction, so the ||z||^2 curiosity bonus cannot swing
wildly from one frame to the next.

The hinge. Rather than penalizing the angle everywhere, we only penalize it
past a threshold: cos(theta) >= tau is free, cos(theta) < tau costs
(tau - cos(theta)), i.e. a hinge loss with threshold tau. The final recipe
uses tau = 0.9 (about 25.8 degrees) and weight 0.003.

"Moved" steps only. A step where the agent's observation barely changed (a
blocked move into a wall, or standing still) carries no information about
which direction the agent "should" be able to see; such steps are excluded by
comparing the raw pixel change of each step to 0.2 times the batch's own mean
pixel change (the same convention the SIGReg/dynamics training pairs use for
"this pair came from a move, not a bump into a wall").

The combined cosine-variance term (used for the 2D latent-game recipe,
weight 0.005) penalizes the VARIANCE of those same per-step cosines across
the batch, instead of capping them from below: the goal there is that every
step's angle be roughly the SAME SIZE everywhere on the map (locally uniform
turning), not merely small everywhere. It is independent of (and can be used
together with or instead of) the hinge.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

MOVE_FRAC = 0.2  # a step counts as "moved" if its pixel change exceeds this
                 # fraction of the batch's own mean pixel change


def frame_moves(o_prev: torch.Tensor, o_mid: torch.Tensor, o_next: torch.Tensor,
                 move_frac: float = MOVE_FRAC):
  """From three consecutive observation frames, decide (with no gradient)
  which of the two steps (prev->mid, mid->next) actually moved.

  A step "moved" if its pixel-space displacement exceeds ``move_frac`` times
  the mean pixel displacement across the whole batch (both steps pooled) --
  purely a property of the frames themselves, never of privileged simulator
  state, so this is honest to compute from any environment's raw pixels.

  Returns:
    moved_in, moved_out: (B,) bool, whether the prev->mid / mid->next step
      moved.
  """
  with torch.no_grad():
    v1 = (o_mid.float() - o_prev.float()).reshape(o_mid.shape[0], -1)
    v2 = (o_next.float() - o_mid.float()).reshape(o_mid.shape[0], -1)
    n1, n2 = v1.norm(dim=-1), v2.norm(dim=-1)
    ref = torch.cat([n1, n2]).mean().clamp_min(1e-12)
    moved_in, moved_out = n1 > move_frac * ref, n2 > move_frac * ref
  return moved_in, moved_out


def cosine_hinge_loss(zp: torch.Tensor, zm: torch.Tensor, zn: torch.Tensor,
                       o_prev: torch.Tensor, o_mid: torch.Tensor,
                       o_next: torch.Tensor, tau: float = 0.9,
                       min_moved: int = 8):
  """The temporal cosine-hinge loss (final recipe: tau=0.9, weight=0.003).

  Args:
    zp, zm, zn: (B, d) encoder outputs of three consecutive frames
      (z_{t-1}, z_t, z_{t+1}).
    o_prev, o_mid, o_next: the three RAW frames those embeddings came from
      (used only to decide which steps "moved", never fed to the loss).
    tau: cosine threshold; steps with cos(theta) < tau are penalized.
    min_moved: if fewer than this many moved pairs survive in the batch, the
      loss is a zero that still carries a (null) gradient path, so a call
      with too few moved pairs is numerically safe and does not raise.

  Returns:
    (penalty, diagnostics): penalty is a scalar tensor; diagnostics is a dict
    of detached scalars useful for logging (mean cosine, fraction below tau,
    etc.), matching the original implementation's field names.
  """
  mi, mo = frame_moves(o_prev, o_mid, o_next)
  c_in = F.cosine_similarity(zm, zp, dim=-1, eps=1e-8)
  c_out = F.cosine_similarity(zn, zm, dim=-1, eps=1e-8)
  cos = torch.cat([c_in, c_out])
  mv = torch.cat([mi, mo])
  n = int(mv.sum().item())
  with torch.no_grad():
    diag = {
        "cos/mean": cos[mv].mean() if n else cos.new_zeros(()),
        "cos/frac_below_tau": (cos[mv] < tau).float().mean() if n else cos.new_zeros(()),
        "cos/moved_share": mv.float().mean(),
    }
  if n < min_moved:
    return zm.sum() * 0.0, diag
  pen = torch.clamp(tau - cos[mv], min=0.0).mean()
  diag["cos/hinge"] = pen.detach()
  return pen, diag


def cosine_variance_loss(zp: torch.Tensor, zm: torch.Tensor, zn: torch.Tensor,
                          o_prev: torch.Tensor, o_mid: torch.Tensor,
                          o_next: torch.Tensor, min_moved: int = 2):
  """The combined cosine-variance term (2D latent-game recipe, weight 0.005).

  Uses the SAME per-step cosines ``cosine_hinge_loss`` penalizes, but
  penalizes their VARIANCE across the batch instead of capping them from
  below: every step's turning angle should be roughly the same size
  everywhere on the map, not merely small everywhere. Independent of the
  hinge above (own weight; works with the hinge on or off).
  """
  mi, mo = frame_moves(o_prev, o_mid, o_next)
  c_in = F.cosine_similarity(zm, zp, dim=-1, eps=1e-8)
  c_out = F.cosine_similarity(zn, zm, dim=-1, eps=1e-8)
  cos = torch.cat([c_in, c_out])
  mv = torch.cat([mi, mo])
  n = int(mv.sum().item())
  with torch.no_grad():
    diag = {
        "cosvar/mean": cos[mv].mean() if n else cos.new_zeros(()),
        "cosvar/var": cos[mv].var(unbiased=False) if n > 1 else cos.new_zeros(()),
        "cosvar/moved_share": mv.float().mean(),
    }
  if n < min_moved:
    return zm.sum() * 0.0, diag
  var = cos[mv].var(unbiased=False)
  diag["cosvar/loss"] = var.detach()
  return var, diag
