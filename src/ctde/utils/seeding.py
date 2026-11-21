"""
================================================================================
 SEEDING & DETERMINISM
================================================================================

WHY THIS FILE EXISTS
--------------------
Charter §7 requires that "every figure [is] regenerable from released code with
one command." That promise only holds if we control randomness from the very
first line of code. Reproducibility CANNOT be retrofitted later -- by the time
you notice results don't reproduce, the randomness has already leaked into a
hundred places.

WHAT IS A "SEED"?
-----------------
Computers cannot produce true randomness. They use *pseudo*-random number
generators (PRNGs): deterministic algorithms that produce a sequence of numbers
that LOOK random. A PRNG needs a starting point -- the "seed".

    seed = 0  ->  0.844, 0.757, 0.420, ...   (always this exact sequence)
    seed = 1  ->  0.417, 0.720, 0.000, ...   (always this different sequence)

Same seed => same "random" numbers => same experiment outcome.

WHY WE NEED >= 8 SEEDS (Charter §7)
------------------------------------
Reinforcement learning is extremely stochastic. The SAME code with the SAME
hyperparameters can succeed on seed 0 and completely fail on seed 1. This is
not a bug; it is the nature of RL. Network weights start random, exploration
is random, and the environment is random.

So a single run tells you almost nothing. If we ran k=2 once and k=3 once and
k=2 scored higher, that is not evidence -- it could easily be luck. We must run
each configuration many times with different seeds and compare the DISTRIBUTIONS.

This is central to our project: Claim 1 says performance peaks at an intermediate
k. If seed noise is larger than the difference between k values, we cannot
detect the peak at all. That is kill-criterion K3 in Charter §3.

THE FOUR SOURCES OF RANDOMNESS WE MUST CONTROL
-----------------------------------------------
  1. Python's built-in `random`   -- used by some libraries internally
  2. NumPy's global RNG           -- used everywhere in scientific Python
  3. PyTorch                      -- neural network weight initialisation
  4. CUDA/cuDNN                   -- GPU operations (if a GPU is used)

Missing ANY of these means results will not reproduce exactly.
================================================================================
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np


# ==============================================================================
# SECTION 1 : GLOBAL SEEDING
# ==============================================================================

def seed_everything(seed: int, deterministic_torch: bool = True) -> None:
    """Seed every random number generator in the process.

    Call this ONCE at the very start of any script, before creating
    environments or networks.

    Parameters
    ----------
    seed:
        The seed value. In our experiments this is the "seed index"
        (0, 1, 2, ... 7) from the config -- see Charter §7 (>= 8 seeds).
    deterministic_torch:
        If True, force PyTorch/cuDNN into deterministic mode. This makes GPU
        results exactly reproducible but can be slower, because cuDNN is not
        allowed to auto-pick the fastest (non-deterministic) algorithm.
        Keep True for published results; set False only when profiling speed.
    """
    # --- 1. Python's built-in RNG -------------------------------------------
    random.seed(seed)

    # --- 2. NumPy's *global* RNG --------------------------------------------
    # NOTE: this seeds the legacy global RNG (np.random.rand etc.). Library
    # code we do not control (e.g. parts of gym / pybullet) still uses it, so
    # we must seed it. OUR OWN code should prefer `make_rng()` below instead.
    np.random.seed(seed)

    # --- 3. Hash randomisation ----------------------------------------------
    # Python randomises string hashes per-process by default, which can change
    # dict/set iteration order and therefore change results. Pin it.
    os.environ["PYTHONHASHSEED"] = str(seed)

    # --- 4. PyTorch ----------------------------------------------------------
    # Imported lazily so that this module stays usable in environments where
    # torch is not installed (e.g. the Phase-1 downwash probe needs no torch).
    try:
        import torch
    except ImportError:
        return

    torch.manual_seed(seed)              # CPU RNG (weight init, dropout, ...)
    torch.cuda.manual_seed_all(seed)     # all GPUs (no-op if none present)

    if deterministic_torch:
        # cuDNN picks different algorithms depending on hardware/benchmarking.
        # These two flags force a fixed, reproducible choice.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ==============================================================================
# SECTION 2 : LOCAL (NON-GLOBAL) RANDOM GENERATORS
# ==============================================================================
#
# WHY NOT JUST USE THE GLOBAL np.random EVERYWHERE?
#
# Global RNG state is shared by every piece of code in the process. If the
# environment draws a random wind vector and the policy also draws random
# exploration noise from the SAME global generator, then changing anything
# about one silently shifts the number stream seen by the other.
#
# Concretely: adding one extra wind sample would change every subsequent
# exploration action. Two runs that should be comparable are no longer
# comparable, and the bug is nearly invisible.
#
# The fix is to give each subsystem its OWN independent generator.
# ==============================================================================

def make_rng(seed: int, stream: str = "") -> np.random.Generator:
    """Create an *independent* NumPy random generator.

    Parameters
    ----------
    seed:
        Base experiment seed (same value passed to `seed_everything`).
    stream:
        A short name identifying the subsystem, e.g. "wind", "init_pose",
        "exploration". Different names produce statistically independent
        streams from the same base seed.

    Returns
    -------
    np.random.Generator
        A modern NumPy generator, isolated from the global RNG.

    Example
    -------
    >>> wind_rng = make_rng(seed=0, stream="wind")
    >>> pose_rng = make_rng(seed=0, stream="init_pose")
    >>> # Drawing from wind_rng can never disturb pose_rng.
    """
    # SeedSequence turns (seed, stream-name) into a high-quality independent
    # seed. We hash the stream name to an integer so it can be mixed in.
    # `& 0xFFFFFFFF` keeps it in 32-bit range as SeedSequence expects.
    stream_id = abs(hash(stream)) & 0xFFFFFFFF if stream else 0
    seq = np.random.SeedSequence([seed, stream_id])
    return np.random.default_rng(seq)


# ==============================================================================
# SECTION 3 : RUN IDENTITY (provenance for every result)
# ==============================================================================

@dataclass(frozen=True)
class RunIdentity:
    """Everything needed to identify and reproduce a single run.

    Attached to every result file so that months later we can answer
    "which code and which settings produced this number?" -- required by
    Charter §7 ("every figure regenerable ... with one command").
    """

    seed: int
    config_name: str
    git_commit: str

    def as_dict(self) -> dict:
        return {
            "seed": self.seed,
            "config_name": self.config_name,
            "git_commit": self.git_commit,
        }


def get_git_commit(default: str = "unknown") -> str:
    """Return the current git commit hash, or `default` if unavailable.

    Recording the commit means a result can always be traced back to the
    exact code that produced it.
    """
    import subprocess

    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Not a git repo, or git not installed -- not fatal.
        return default
