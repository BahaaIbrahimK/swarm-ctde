"""
================================================================================
 RESULT LOADER  --  turn result directories into tidy data
================================================================================

WHY THIS FILE EXISTS
--------------------
Plotting scripts should say "give me the returns per rung", not know that
history lives in `<dir>/stage2_pilot/<rung>_seed<N>_sp<X>/history.json`. This
module owns the directory layout so the plotting scripts stay readable, and so
a future change to the layout is a one-file edit.

It also handles a practical problem: results now exist in two different layouts
-- the older `results/_archive/<timestamp>/stageN_.../` (all INVALID) and the newer
`results/<EXP_ID>/<date>_<task_version>/runs/`. Both are discovered here.

WHAT IT GUARDS AGAINST
----------------------
  * Pooling incompatible task versions. Results from `v1-raw4d` and
    `v2-pidassist` describe DIFFERENT experiments; putting them on one axis
    would be meaningless. `load_sweep` warns loudly if it sees a mix.
  * Silently plotting an unfinished sweep. Runs are counted and reported.
================================================================================
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


# ==============================================================================
# SECTION 1 : DATA CONTAINERS
# ==============================================================================

@dataclass
class Run:
    """One training run: a single (rung, seed, spacing) triple."""

    rung: str
    seed: int
    spacing: float
    history: list[dict]                  # one record per evaluation point
    config: dict = field(default_factory=dict)
    path: Path | None = None

    # ---- Convenience views over the learning curve ---------------------------

    @property
    def steps(self) -> np.ndarray:
        return np.array([h["step"] for h in self.history])

    @property
    def returns(self) -> np.ndarray:
        return np.array([h["eval_return"] for h in self.history])

    @property
    def explained_variance(self) -> np.ndarray:
        return np.array([h.get("explained_variance", np.nan) for h in self.history])

    @property
    def episode_len(self) -> np.ndarray:
        return np.array([h.get("eval_episode_len", np.nan) for h in self.history])

    @property
    def lateral_error(self) -> np.ndarray:
        """Sidestep-exploit monitor. Only present in task version v2+."""
        return np.array([h.get("eval_lateral_error_m", np.nan) for h in self.history])

    @property
    def coupling(self) -> np.ndarray:
        return np.array([h.get("eval_coupling_N", np.nan) for h in self.history])

    @property
    def crash_rate(self) -> np.ndarray:
        return np.array([h.get("eval_crash_rate", np.nan) for h in self.history])

    def final_return(self, tail: int = 3, horizon: int | None = None) -> float:
        """Final performance = mean of the last `tail` evaluations.

        Parameters
        ----------
        horizon:
            If given, evaluate as of this many env steps rather than the end of
            training.

        WHY A HORIZON OPTION EXISTS
        ---------------------------
        Runs can COLLAPSE late in training: the critic destabilises (explained
        variance goes negative), which corrupts the policy gradient, and the
        policy follows it down. Measured in the v3 pilot: 4 of 15 runs collapsed
        in the final 100k steps, after being stable for the preceding 300k.

        A run that reached 684 and held it for 300k steps, then crashed on its
        last two evaluations, is reported as 498 by the naive last-3 rule. That
        single number inflates the rung's variance enough to hide a real effect.

        Stage 1 established convergence by 300k, so evaluating every run at that
        SAME horizon is both defensible and uniform -- it is not cherry-picking,
        because it is applied identically to every run regardless of rung.

        Report the horizon used in any figure caption.
        """
        if horizon is None:
            return float(np.mean(self.returns[-tail:]))

        mask = self.steps <= horizon
        vals = self.returns[mask]
        if len(vals) == 0:
            return float(np.mean(self.returns[:tail]))
        return float(np.mean(vals[-tail:]))

    def collapsed(self, threshold: float = 0.85) -> bool:
        """True if the run ended far below its own best -- a late collapse.

        Used to report how many runs are affected, so the choice of horizon is
        transparent rather than hidden.
        """
        best = float(np.max(self.returns))
        final = self.final_return()
        return best > 0 and final < best * threshold

    @property
    def task_version(self) -> str:
        return self.config.get("task_version", "unknown")


@dataclass
class Sweep:
    """A collection of runs, plus what is needed to interpret them."""

    runs: list[Run]
    source: Path
    label: str = ""

    # ---- Structure -----------------------------------------------------------

    @property
    def rungs(self) -> list[str]:
        """Rungs present, ordered by k rather than alphabetically.

        Alphabetical order would put 'k10' before 'k2'. Sorting by the integer
        keeps the x-axis in the order a reader expects.
        """
        found = {r.rung for r in self.runs}

        def sort_key(rung: str) -> tuple[int, int]:
            m = re.match(r"k(\d+)", rung)
            k = int(m.group(1)) if m else 999
            return (k, 1 if "+P" in rung else 0)   # k3+P sorts after k3

        return sorted(found, key=sort_key)

    @property
    def seeds(self) -> list[int]:
        return sorted({r.seed for r in self.runs})

    @property
    def spacings(self) -> list[float]:
        return sorted({r.spacing for r in self.runs})

    @property
    def task_versions(self) -> set[str]:
        return {r.task_version for r in self.runs}

    # ---- Selection -----------------------------------------------------------

    def by_rung(self, rung: str, spacing: float | None = None) -> list[Run]:
        out = [r for r in self.runs if r.rung == rung]
        if spacing is not None:
            out = [r for r in out if abs(r.spacing - spacing) < 1e-9]
        return sorted(out, key=lambda r: r.seed)

    def final_returns(self, rung: str, spacing: float | None = None,
                      horizon: int | None = None) -> np.ndarray:
        """Final return per seed for one rung -- the input to every summary."""
        return np.array([r.final_return(horizon=horizon)
                         for r in self.by_rung(rung, spacing)])

    def num_collapsed(self) -> int:
        """How many runs collapsed late in training (see Run.collapsed)."""
        return sum(1 for r in self.runs if r.collapsed())

    # ---- Summary statistics --------------------------------------------------

    def summary(self, spacing: float | None = None,
                horizon: int | None = None) -> dict[str, dict[str, float]]:
        """Mean, std, SEM and 95% CI per rung.

        Charter §7 requires mean ± 95% CI everywhere. We use the t-distribution
        rather than 1.96 because seed counts are small (5-8): with n=5 the
        correct multiplier is 2.78, and using 1.96 would understate the interval
        by ~40% -- overstating our confidence.
        """
        from scipy import stats

        out: dict[str, dict[str, float]] = {}
        for rung in self.rungs:
            vals = self.final_returns(rung, spacing, horizon)
            if len(vals) == 0:
                continue

            n = len(vals)
            mean = float(vals.mean())
            std = float(vals.std(ddof=1)) if n > 1 else 0.0
            sem = std / np.sqrt(n) if n > 1 else 0.0
            tcrit = float(stats.t.ppf(0.975, df=n - 1)) if n > 1 else 0.0

            out[rung] = {
                "n": n,
                "mean": mean,
                "std": std,
                "sem": sem,
                "ci95": tcrit * sem,
                "ci95_low": mean - tcrit * sem,
                "ci95_high": mean + tcrit * sem,
                "min": float(vals.min()),
                "max": float(vals.max()),
            }
        return out


# ==============================================================================
# SECTION 2 : LOADING
# ==============================================================================

_RUN_DIR = re.compile(r"^(?P<rung>k\d+(?:\+P)?)_seed(?P<seed>\d+)_sp(?P<sp>[\d.]+)")


def _parse_run_dir(name: str) -> tuple[str, int, float] | None:
    """Parse `k1_seed03_sp1.00` into ('k1', 3, 1.0)."""
    m = _RUN_DIR.match(name)
    if not m:
        return None
    return m.group("rung"), int(m.group("seed")), float(m.group("sp"))


def load_run(run_dir: Path) -> Run | None:
    """Load one run directory. Returns None if it has no history yet."""
    run_dir = Path(run_dir)
    parsed = _parse_run_dir(run_dir.name)
    if parsed is None:
        return None

    history_path = run_dir / "history.json"
    if not history_path.exists():
        return None                      # run did not finish

    with open(history_path, encoding="utf-8") as f:
        history = json.load(f)
    if not history:
        return None

    config = {}
    config_path = run_dir / "run_config.json"
    if config_path.exists():
        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)

    rung, seed, spacing = parsed
    return Run(rung=rung, seed=seed, spacing=spacing,
               history=history, config=config, path=run_dir)


def load_sweep(path: str | Path, label: str = "", quiet: bool = False) -> Sweep:
    """Load every run beneath `path`.

    Accepts any of:
        results/_archive/<timestamp>/                 (searches all stages; INVALID)
        results/_archive/<timestamp>/stage2_pilot/    (one stage; INVALID)
        results/<EXP_ID>/<date>_<version>/runs/        (new layout)

    Parameters
    ----------
    quiet:
        Suppress the summary print (useful when loading many sweeps).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No such result directory: {path}")

    # Search recursively so any of the layouts above works.
    runs: list[Run] = []
    for candidate in sorted(path.rglob("*")):
        if candidate.is_dir():
            run = load_run(candidate)
            if run is not None:
                runs.append(run)

    if not runs:
        raise ValueError(
            f"No completed runs found under {path}. "
            "Runs only write history.json when they finish -- the sweep may "
            "still be in progress."
        )

    sweep = Sweep(runs=runs, source=path, label=label or path.name)

    # ---- Guard: never pool incompatible task versions -----------------------
    versions = sweep.task_versions
    if len(versions) > 1:
        print("=" * 70)
        print("  WARNING: MIXED TASK VERSIONS")
        print(f"  Found: {sorted(versions)}")
        print()
        print("  Different task versions are DIFFERENT EXPERIMENTS -- reward,")
        print("  action space, or termination differ, so their returns are not")
        print("  comparable. Plotting them on one axis is meaningless.")
        print("  Filter to a single version before drawing figures.")
        print("=" * 70)

    if not quiet:
        print(f"[loaded] {len(runs)} runs from {path}")
        print(f"         rungs    : {sweep.rungs}")
        print(f"         seeds    : {sweep.seeds}")
        print(f"         spacings : {sweep.spacings}")
        print(f"         task ver : {sorted(versions)}")

    return sweep


# ==============================================================================
# SECTION 3 : STATISTICS
# ==============================================================================

def compare_rungs(sweep: Sweep, rung_a: str, rung_b: str,
                  spacing: float | None = None, horizon: int | None = None) -> dict:
    """Welch's t-test between two rungs, with effect size.

    WHY WELCH RATHER THAN STUDENT: Welch does not assume the two groups have
    equal variance. Our rungs demonstrably do not -- in the pilot, k0's std was
    104 while k1's was 40 -- so Student's t would be the wrong test.

    WHY REPORT COHEN'S d: with 5 seeds, a real effect can easily fail to reach
    p<0.05. The effect SIZE tells us whether a larger sweep is likely to resolve
    it, which is exactly the decision the pilot exists to inform.
    """
    from scipy import stats

    a = sweep.final_returns(rung_a, spacing, horizon)
    b = sweep.final_returns(rung_b, spacing, horizon)

    if len(a) < 2 or len(b) < 2:
        return {"error": "need >= 2 seeds per rung"}

    t_stat, p_value = stats.ttest_ind(a, b, equal_var=False)

    # Cohen's d with pooled standard deviation.
    n1, n2 = len(a), len(b)
    pooled = np.sqrt(((n1 - 1) * a.var(ddof=1) + (n2 - 1) * b.var(ddof=1)) / (n1 + n2 - 2))
    cohens_d = (a.mean() - b.mean()) / pooled if pooled > 0 else 0.0

    return {
        "rung_a": rung_a,
        "rung_b": rung_b,
        "mean_a": float(a.mean()),
        "mean_b": float(b.mean()),
        "difference": float(a.mean() - b.mean()),
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "cohens_d": float(cohens_d),
        "significant": bool(p_value < 0.05),
    }


def interior_optimum_test(sweep: Sweep, spacing: float | None = None,
                          horizon: int | None = None) -> dict:
    """Test Claim 1: does an interior rung beat BOTH endpoints?

    Claim 1 requires the peak to beat both ends. Beating only the decentralized
    end (k0) would merely re-show that centralization helps -- established since
    MADDPG (2017). Beating both is the novel claim.
    """
    rungs = sweep.rungs
    if len(rungs) < 3:
        return {"error": "need >= 3 rungs to test for an interior optimum"}

    means = {r: sweep.final_returns(r, spacing, horizon).mean() for r in rungs}
    interior = rungs[1:-1]
    best = max(interior, key=lambda r: means[r])

    low_end, high_end = rungs[0], rungs[-1]
    vs_low = compare_rungs(sweep, best, low_end, spacing, horizon)
    vs_high = compare_rungs(sweep, best, high_end, spacing, horizon)

    beats_both = means[best] > means[low_end] and means[best] > means[high_end]

    return {
        "best_interior_rung": best,
        "means": means,
        "beats_both_endpoints": beats_both,
        "vs_low_endpoint": vs_low,
        "vs_high_endpoint": vs_high,
        "both_significant": vs_low.get("significant") and vs_high.get("significant"),
    }
