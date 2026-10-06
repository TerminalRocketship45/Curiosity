"""IsoCover: curiosity from an isotropic Gaussian map.

An encoder phi maps observations to a latent z in R^d, trained with a sliced
isotropic Gaussian regularizer (SIGReg, sigreg.py) plus temporal losses
(temporal.py, dynamics.py) that keep consecutive frames' embeddings related.
Once phi's output distribution is approximately N(0, I_d), the squared
distance from the origin, ||z||^2, is (via the Gaussian's own change-of-
variables formula) a direct proxy for how rare an observation is under the
agent's own experience -- this is the ``reward.py`` intrinsic reward. See the
top-level README for the full plain-language explanation and
``configs/`` for the exact, cited final-recipe hyperparameters.
"""

import os as _os

# THREAD CAP -- set BEFORE torch/numpy are imported anywhere.
#
# On a machine with many CPU cores, torch's default intra-op thread pool size
# equals the core count. At the small matrix sizes this project uses (a
# handful of conv layers, policy batches in the hundreds, SIGReg's d=1024
# projection), every op is far too small to amortize waking that many
# threads, and training can be MANY TIMES SLOWER (measured 5x+ slower with a
# large default thread pool than with 1-2 threads) -- on a heavily loaded
# shared cluster node this can look like an outright hang rather than just
# slow. `torch.set_num_threads()` after `import torch` is too late: the
# OpenMP/MKL/OpenBLAS runtimes read these environment variables when they
# initialize, at import time. Python runs a package's `__init__` before any
# of its submodules, and no `isocover` submodule imports torch at module
# level, so this is the earliest point guaranteed to precede that import.
#
# `setdefault`, not assignment: an explicit value the caller already set in
# its own environment (e.g. inside an sbatch script) always wins.
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
  _os.environ.setdefault(_var, "1")

__version__ = "0.1.0"
