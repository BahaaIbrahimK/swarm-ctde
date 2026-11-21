"""
================================================================================
 SWEEP RUNNER  --  many (rung, seed) runs, in parallel
================================================================================

PHASE 3 (Charter §5).

--------------------------------------------------------------------------------
 WHY PARALLELISM IS THE RIGHT LEVER HERE
--------------------------------------------------------------------------------
Charter §13 argued we are bottlenecked by SEED COUNT, not by steps/second, and
that the workload is embarrassingly parallel. This module is where that
argument cashes out.

Measured on this machine: ~46 env-steps/sec per process, PyBullet-bound.
The runs are fully independent, so wall-clock time divides by the number of
worker processes.

We use PROCESSES, not threads: PyBullet holds the GIL and keeps global state
per physics client, so threads would serialise and could even collide.

--------------------------------------------------------------------------------
 REPRODUCIBILITY UNDER PARALLELISM
--------------------------------------------------------------------------------
Each run seeds itself from its own `RunConfig.seed` INSIDE the worker process
(see Trainer.__init__). Results therefore do not depend on scheduling order,
on how many workers are used, or on which core a job lands. Charter §7's
"regenerable with one command" survives parallel execution.
================================================================================
"""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

import numpy as np

from ctde.models.observability import ObservabilityLevel
from ctde.training.trainer import RunConfig, Trainer


# ==============================================================================
# SECTION 1 : WORKER
# ==============================================================================

def _run_one(args: tuple[RunConfig, str]) -> dict:
    """Execute a single run in a worker process.

    Must be a module-level function so it can be pickled and sent to workers.

    Returns a summary dict rather than the Trainer object, because Trainer
    holds an unpicklable PyBullet client.
    """
    config, out_dir = args
    started = time.time()

    trainer = Trainer(config, Path(out_dir))
    try:
        history = trainer.train(verbose=False)

        # ======================================================================
        # THE REPORTED NUMBER  (fixed 17 Sep 2026)
        # ======================================================================
        # This used to be the mean of the last three PERIODIC evaluations, under
        # field names beginning `final_`. That was wrong in three ways at once:
        #
        #   1. It averaged THREE DIFFERENT POLICIES (three checkpoints), so the
        #      reported number described no model that ever existed.
        #   2. Those evaluations use the PERIODIC seed block -- the same
        #      episodes used to watch the run's health. Reporting them is
        #      selecting on the data used for monitoring.
        #   3. `Trainer.evaluate(final=True)` existed, was tested, and was
        #      called ONLY from scripts/check_learning.py. The sweep -- the code
        #      that actually produces the paper's numbers -- never called it.
        #
        # (3) is the fourth instance in this project of a correct fix that never
        # reached the running code (padded critic input, metrics.py, the
        # eval_episodes override, and this). Hence test_sweep_reports_final_eval
        # in tests/test_reporting.py, which asserts on the SWEEP's output rather
        # than on Trainer's API.
        final_eval = trainer.evaluate(final=True) if history else {}
        final_episodes = [m.as_dict() for m in trainer.last_eval_metrics]
    finally:
        trainer.close()

    # Kept for the learning-curve figures ONLY. Never report these: see above.
    tail = [h["eval_return"] for h in history[-3:]] if history else [float("nan")]

    def f(key: str) -> float:
        """Pull a held-out final-evaluation value."""
        v = final_eval.get(key, float("nan"))
        return float(v) if v is not None else float("nan")

    # Per-episode records are saved so the analysis can compute PAIRED
    # differences later. Storing only means makes paired tests impossible after
    # the fact, and the sweep is expensive to repeat.
    if final_episodes:
        with open(Path(out_dir) / "final_eval_episodes.json", "w", encoding="utf-8") as fh:
            json.dump(final_episodes, fh, indent=2)

    return {
        "rung": config.level.label,
        "level_k": config.level.k,
        "level_privileged": config.level.include_privileged,
        "seed": config.seed,
        "formation_spacing_m": config.formation_spacing_m,
        # ---- HELD-OUT final evaluation: this is what gets reported -----------
        "final_eval_return": f("mean_return"),
        "final_eval_return_std": f("std_return"),
        "final_eval_episodes": f("n_eval_episodes"),
        # ---- PHYSICAL metrics (Charter §S.11) --------------------------------
        # Return is a training variable whose meaning changes every time the
        # reward is retuned (ours has changed three times). These do not.
        "final_rms_position_error_m": f("rms_position_error_m"),
        "final_max_position_error_m": f("max_position_error_m"),
        "final_min_separation_m": f("min_separation_m"),
        "final_collision_rate": f("collision_rate"),
        "final_crash_rate": f("crash_rate"),
        "final_worst_error_amplification": f("worst_error_amplification"),
        "final_control_effort": f("mean_control_effort"),
        # ---- Training-curve diagnostics: NOT results -------------------------
        # Explicitly prefixed `curve_` so they cannot be mistaken for the
        # reported numbers in a table or a plot label.
        "curve_last3_periodic_return": float(np.mean(tail)),
        "curve_best_periodic_return": (
            float(max(h["eval_return"] for h in history)) if history else float("nan")
        ),
        "curve_final_explained_variance": (
            history[-1]["explained_variance"] if history else float("nan")
        ),
        "wall_time_sec": time.time() - started,
        "output_dir": out_dir,
    }


# ==============================================================================
# SECTION 2 : SWEEP
# ==============================================================================

def run_sweep(
    levels: list[ObservabilityLevel],
    seeds: list[int],
    base_config: RunConfig,
    output_root: Path,
    spacings: list[float] | None = None,
    num_workers: int | None = None,
    verbose: bool = True,
) -> list[dict]:
    """Run the full grid of (level x seed x spacing) jobs.

    Parameters
    ----------
    levels:
        Observability rungs to sweep -- the independent variable.
    seeds:
        Charter §7 requires >= 8 for final results. The pilot uses 5, which is
        enough to see whether an effect exists at all.
    base_config:
        Template; `level`, `seed`, and `formation_spacing_m` are overridden per
        job. Everything else -- crucially the STEP BUDGET -- stays identical
        across rungs, as Charter §7 demands.
    spacings:
        Formation spacings for Claim 2. None means "use base_config's value"
        (the Claim 1 sweep, which holds spacing fixed).
    num_workers:
        Defaults to cpu_count() - 2, leaving headroom so the machine stays
        usable and workers are not starved by contention.

    Returns
    -------
    list[dict]
        One summary record per run.
    """
    spacings = spacings if spacings is not None else [base_config.formation_spacing_m]
    num_workers = num_workers or max(1, (os.cpu_count() or 4) - 2)

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    # ---- Build the job list -------------------------------------------------
    jobs: list[tuple[RunConfig, str]] = []
    for spacing in spacings:
        for level in levels:
            for seed in seeds:
                cfg = replace(
                    base_config,
                    level=level,
                    seed=seed,
                    formation_spacing_m=spacing,
                )
                name = f"{level.label}_seed{seed}_sp{spacing:.2f}"
                jobs.append((cfg, str(output_root / name)))

    if verbose:
        est_hours = (
            len(jobs) * base_config.total_env_steps / 46.0 / 3600.0 / num_workers
        )
        print("=" * 74)
        print(" SWEEP")
        print("=" * 74)
        print(f"  rungs        : {[lv.label for lv in levels]}")
        print(f"  seeds        : {seeds}")
        print(f"  spacings     : {spacings}")
        print(f"  budget/run   : {base_config.total_env_steps:,} env steps")
        print(f"  total runs   : {len(jobs)}")
        print(f"  workers      : {num_workers}")
        print(f"  est. wall    : ~{est_hours:.1f} h (at ~46 steps/s/process)")
        print("=" * 74)
        print()

    # ---- Execute ------------------------------------------------------------
    results: list[dict] = []
    started = time.time()

    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        futures = {pool.submit(_run_one, job): job for job in jobs}

        for n, future in enumerate(as_completed(futures), start=1):
            try:
                result = future.result()
                results.append(result)

                if verbose:
                    elapsed = time.time() - started
                    eta = elapsed / n * (len(jobs) - n)
                    print(
                        f"  [{n:>3}/{len(jobs)}] {result['rung']:<6} "
                        f"seed{result['seed']:<3} sp{result['formation_spacing_m']:.2f}  "
                        f"return {result['final_eval_return']:>9.2f}  "
                        f"({result['wall_time_sec'] / 60:.1f} min)  "
                        f"ETA {eta / 60:.0f} min"
                    )
            except Exception as exc:                       # noqa: BLE001
                # One failed run must not abort the sweep -- we would lose every
                # completed result. Record and continue.
                cfg, out_dir = futures[future]
                print(f"  [FAIL] {cfg.level.label} seed{cfg.seed}: {exc}")
                results.append({
                    "rung": cfg.level.label,
                    "seed": cfg.seed,
                    "formation_spacing_m": cfg.formation_spacing_m,
                    "final_eval_return": float("nan"),
                    "error": str(exc),
                    "output_dir": out_dir,
                })

    # ---- Persist ------------------------------------------------------------
    with open(output_root / "sweep_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    if verbose:
        print()
        print(f"  sweep complete in {(time.time() - started) / 60:.1f} min")
        print(f"  results -> {output_root / 'sweep_results.json'}")

    return results


# ==============================================================================
# SECTION 3 : AGGREGATION
# ==============================================================================

def aggregate_by_rung(
    results: list[dict],
    spacing: float | None = None,
    metric: str = "final_eval_return",
) -> dict[str, dict[str, float]]:
    """Group results by rung and compute mean +/- 95% CI across seeds.

    Charter §7 requires "mean +/- 95% CI, always plotted". This is where the
    per-seed noise that Claim 1 must overcome becomes visible: if the CIs at
    adjacent rungs overlap heavily, the study is underpowered (kill-criterion
    K3) and no conclusion may be drawn.

    SPACING IS NOT A NUISANCE VARIABLE  (fixed 16 Sep 2026)
    -------------------------------------------------------
    Formation spacing is Claim 2's INDEPENDENT VARIABLE: the claim is that the
    best rung k* tracks coupling strength, and spacing is how we vary coupling.
    Averaging a rung's seeds across spacings therefore destroys precisely the
    signal being tested -- and destroys it SILENTLY, producing a plausible
    number with no error.

    So this function refuses to guess. Pass `spacing` to select one condition;
    if the results contain several and none is named, it raises.

    Parameters
    ----------
    spacing
        Keep only runs at this formation spacing. Required when the results
        span more than one.
    metric
        Which per-run quantity to aggregate. Defaults to return, but the
        physical metrics (`final_rms_position_error_m`, ...) are what the paper
        reports -- see Charter §S.11.
    """
    spacings = {r["formation_spacing_m"] for r in results
                if r.get("formation_spacing_m") is not None}

    if spacing is not None:
        results = [r for r in results if r.get("formation_spacing_m") == spacing]
    elif len(spacings) > 1:
        raise ValueError(
            f"results span multiple spacings {sorted(spacings)} -- pass "
            f"spacing=... to select one. Pooling them would average across "
            f"Claim 2's independent variable and destroy the effect being tested."
        )

    by_rung: dict[str, list[float]] = {}
    for r in results:
        if not np.isnan(r.get(metric, float("nan"))):
            by_rung.setdefault(r["rung"], []).append(r[metric])

    summary: dict[str, dict[str, float]] = {}
    for rung, values in by_rung.items():
        arr = np.array(values, dtype=float)
        n = len(arr)
        mean = float(arr.mean())
        # Standard error of the mean; 1.96 for a normal 95% interval. With
        # small n this is approximate -- the final paper will use a
        # t-distribution and report exact tests.
        sem = float(arr.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0

        summary[rung] = {
            "n_seeds": n,
            "mean": mean,
            "std": float(arr.std(ddof=1)) if n > 1 else 0.0,
            "sem": sem,
            "ci95_low": mean - 1.96 * sem,
            "ci95_high": mean + 1.96 * sem,
        }

    return summary
