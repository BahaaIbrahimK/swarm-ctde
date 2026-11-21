#!/usr/bin/env python3
"""
================================================================================
 OVERNIGHT STAGED RUN
================================================================================

Designed for an unattended multi-hour window. Runs two stages in sequence and
writes a single readable report.

--------------------------------------------------------------------------------
 WHY TWO STAGES RATHER THAN ONE BIG SWEEP
--------------------------------------------------------------------------------
The learning sanity check showed the agent DOES learn (return +9.6%, explained
variance ~0.74), but its curve plateaued around 12k steps and then wobbled.
That raises a question the pilot cannot answer on its own:

    Is the training budget long enough for rungs to SEPARATE?

If every rung sits in a noisy plateau, the pilot returns "flat" -- and a flat
curve is ambiguous between "no effect" (kill-criterion K1) and "not converged"
(kill-criterion K2). Acting on that ambiguity is how a project gets abandoned
for the wrong reason.

So:

  STAGE 1  BUDGET CHECK   one rung, few seeds, LONG budget
                          -> where do returns actually converge?

  STAGE 2  PILOT          three rungs, more seeds, budget informed by Stage 1
                          -> does k matter?

Stage 1 makes Stage 2 interpretable. Running only Stage 2 risks a full night
producing an uninterpretable result.

--------------------------------------------------------------------------------
 USAGE
--------------------------------------------------------------------------------
    # Full overnight run (~9-10 h on 6 workers)
    python scripts/run_overnight.py

    # Custom
    python scripts/run_overnight.py --stage1-steps 300000 --stage2-steps 300000
    python scripts/run_overnight.py --skip-stage1        # pilot only
    python scripts/run_overnight.py --dry-run            # print the plan, exit

Progress is written continuously to the output directory, so an interrupted run
still leaves usable partial results.
================================================================================
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import argparse
import json
import os
import time
from datetime import datetime, timedelta

import numpy as np

from ctde.models.observability import build_ladder, critic_input_dim
from ctde.training.mappo import MAPPOConfig
from ctde.training.sweep import aggregate_by_rung, run_sweep
from ctde.training.trainer import RunConfig
from ctde.utils.seeding import get_git_commit

# Measured on this machine (PyBullet-bound, single process).
STEPS_PER_SEC = 46.0


# ==============================================================================
# SECTION 1 : PLANNING
# ==============================================================================

def estimate_hours(num_runs: int, steps_per_run: int, workers: int) -> float:
    return num_runs * steps_per_run / STEPS_PER_SEC / 3600.0 / workers


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Staged overnight run.")

    p.add_argument("--agents", type=int, default=4)
    p.add_argument("--spacing", type=float, default=1.0)
    p.add_argument("--workers", type=int, default=None,
                   help="Parallel processes (default: cpu_count - 2)")

    # ---- Stage 1: budget check ---------------------------------------------
    p.add_argument("--stage1-steps", type=int, default=300_000)
    p.add_argument("--stage1-seeds", type=int, default=3)
    p.add_argument("--skip-stage1", action="store_true")

    # ---- Stage 2: pilot -----------------------------------------------------
    p.add_argument("--stage2-steps", type=int, default=300_000)
    p.add_argument("--stage2-seeds", type=int, default=5)
    p.add_argument("--skip-stage2", action="store_true")

    p.add_argument("--dry-run", action="store_true",
                   help="Print the plan and time estimate, then exit.")
    return p.parse_args()


# ==============================================================================
# SECTION 2 : STAGE 1  --  BUDGET CHECK
# ==============================================================================

def run_stage1(args, out_root: Path, workers: int) -> dict:
    """One rung, several seeds, long budget.

    QUESTION: where do returns converge, and is the critic still improving?

    We use the MIDDLE rung (k1). It is the interior candidate for Claim 1, so
    if any rung is going to show interesting behaviour it should. Using an
    endpoint could mislead: k0 may plateau early simply because it has less
    information to exploit.
    """
    print("=" * 74)
    print(" STAGE 1  --  BUDGET CHECK")
    print("=" * 74)
    print("  Question: is the training budget long enough for rungs to separate?")
    print()
    print("  The learning check plateaued around 12k steps. If every rung sits in")
    print("  a noisy plateau, the pilot returns 'flat' -- which is ambiguous")
    print("  between 'no effect' (K1) and 'not converged' (K2). This resolves it.")
    print()

    ladder = build_ladder(args.agents, include_privileged_rung=False)
    level = ladder[1]                       # k1 -- the interior candidate

    base = RunConfig(
        level=level,
        num_agents=args.agents,
        formation_spacing_m=args.spacing,
        total_env_steps=args.stage1_steps,
        episode_len_sec=15.0,
        # Evaluate often: the SHAPE of the curve is the output here, not just
        # its endpoint. Stage 1 asks "where does the curve flatten?", which a
        # noisy-but-frequent estimate answers fine -- so a smaller count than
        # RunConfig's calibrated default is deliberate here, and the ENDPOINT
        # from this stage must never be quoted as a result.
        eval_every_steps=max(args.stage1_steps // 30, 2_000),
        eval_episodes=4,
        mappo=MAPPOConfig(),
    )

    stage_dir = out_root / "stage1_budget"
    results = run_sweep(
        levels=[level],
        seeds=list(range(args.stage1_seeds)),
        base_config=base,
        output_root=stage_dir,
        num_workers=workers,
        verbose=True,
    )

    # ---- Analyse the learning curves ---------------------------------------
    curves: list[list[dict]] = []
    for r in results:
        hist_path = Path(r["output_dir"]) / "history.json"
        if hist_path.exists():
            with open(hist_path, encoding="utf-8") as f:
                curves.append(json.load(f))

    analysis = analyse_convergence(curves, args.stage1_steps)

    with open(stage_dir / "convergence_analysis.json", "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2)

    print()
    print("-" * 74)
    print(" STAGE 1 ANALYSIS")
    print("-" * 74)
    for line in analysis["report_lines"]:
        print(f"  {line}")
    print()

    return analysis


def analyse_convergence(curves: list[list[dict]], total_steps: int) -> dict:
    """Decide whether training had converged by the end of the budget.

    Method: compare the mean return over the last quarter of training against
    the quarter before it. If the return is still climbing meaningfully, the
    budget is too short.
    """
    if not curves:
        return {"converged": False, "report_lines": ["No curves produced."]}

    # Align curves by evaluation index (all runs use identical schedules).
    min_len = min(len(c) for c in curves)
    returns = np.array([[c[i]["eval_return"] for i in range(min_len)] for c in curves])
    explained = np.array([[c[i]["explained_variance"] for i in range(min_len)] for c in curves])

    mean_curve = returns.mean(axis=0)
    quarter = max(min_len // 4, 1)

    third_quarter = float(mean_curve[-2 * quarter : -quarter].mean()) if min_len >= 2 * quarter else float(mean_curve[0])
    last_quarter = float(mean_curve[-quarter:].mean())

    improvement = last_quarter - third_quarter
    denom = abs(third_quarter) if abs(third_quarter) > 1e-6 else 1.0
    rel_improvement = improvement / denom

    final_ev = float(explained[:, -quarter:].mean())
    seed_spread = float(returns[:, -quarter:].mean(axis=1).std(ddof=1)) if len(curves) > 1 else 0.0
    final_mean = last_quarter

    # Still climbing by more than 2% over the final quarter => not converged.
    converged = rel_improvement < 0.02

    lines = [
        f"final return (last quarter) : {last_quarter:.3f}",
        f"previous quarter            : {third_quarter:.3f}",
        f"improvement                 : {improvement:+.3f}  ({rel_improvement:+.1%})",
        f"explained variance          : {final_ev:.3f}",
        f"seed spread (std)           : {seed_spread:.3f}",
        "",
    ]

    if converged:
        lines += [
            f"CONVERGED by {total_steps:,} steps.",
            "=> This budget is sufficient. Stage 2's pilot is interpretable.",
        ]
    else:
        lines += [
            f"STILL IMPROVING at {total_steps:,} steps ({rel_improvement:+.1%} over the last quarter).",
            "=> Rungs may not have separated yet. Treat a FLAT Stage 2 result as",
            "   kill-criterion K2 (budget too small), NOT K1 (no effect).",
        ]

    # Is seed noise large relative to the effect we hope to detect?
    if seed_spread > 0 and abs(final_mean) > 1e-6:
        noise_ratio = seed_spread / abs(final_mean)
        lines.append("")
        lines.append(f"seed noise / |mean| = {noise_ratio:.1%}")
        if noise_ratio > 0.15:
            lines.append("WARNING: seed variance is large. Differences between rungs")
            lines.append("         will need many seeds to resolve (kill-criterion K3).")

    return {
        "converged": converged,
        "final_return": last_quarter,
        "previous_quarter_return": third_quarter,
        "relative_improvement": rel_improvement,
        "final_explained_variance": final_ev,
        "seed_spread": seed_spread,
        "report_lines": lines,
    }


# ==============================================================================
# SECTION 3 : STAGE 2  --  PILOT
# ==============================================================================

def run_stage2(args, out_root: Path, workers: int) -> dict:
    """Three rungs, more seeds. The actual decision gate."""
    print("=" * 74)
    print(" STAGE 2  --  PILOT SWEEP  (DECISION GATE)")
    print("=" * 74)
    print("  Question: does critic observability (k) matter?")
    print()

    ladder = build_ladder(args.agents, include_privileged_rung=False)
    levels = [ladder[0], ladder[1], ladder[-1]]      # k0, k1, k(N-1)

    for lv in levels:
        print(f"    {lv.label:<6} critic input {critic_input_dim(lv):>3}-D")
    print()

    base = RunConfig(
        level=levels[0],
        num_agents=args.agents,
        formation_spacing_m=args.spacing,
        total_env_steps=args.stage2_steps,
        episode_len_sec=15.0,
        eval_every_steps=max(args.stage2_steps // 15, 2_000),
        # eval_episodes NOT set: stage 2 IS the pilot whose numbers get
        # reported, so it must inherit RunConfig's calibrated default (8).
        # Hardcoding 3 here would silently revert the protocol -- one crash in a
        # 3-episode block moves the mean by ~350, three times the effect we are
        # trying to resolve.
        mappo=MAPPOConfig(),
    )

    stage_dir = out_root / "stage2_pilot"
    results = run_sweep(
        levels=levels,
        seeds=list(range(args.stage2_seeds)),
        base_config=base,
        output_root=stage_dir,
        num_workers=workers,
        verbose=True,
    )

    # One spacing per stage here; aggregate explicitly so that adding a
    # spacing sweep later fails loudly instead of pooling silently.
    spacings_present = sorted({r["formation_spacing_m"] for r in results})
    summary = aggregate_by_rung(results, spacing=spacings_present[0])
    if len(spacings_present) > 1:
        print(f"  WARNING: results span spacings {spacings_present}; "
              f"reporting only {spacings_present[0]} m")
    rung_order = [lv.label for lv in levels]

    with open(stage_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "rung_order": rung_order}, f, indent=2)

    return {"summary": summary, "rung_order": rung_order}


# ==============================================================================
# SECTION 4 : REPORT
# ==============================================================================

def write_report(
    out_root: Path,
    args,
    stage1: dict | None,
    stage2: dict | None,
    elapsed_hours: float,
) -> str:
    """Compose the morning report -- the thing you read first."""
    lines = [
        "=" * 74,
        " OVERNIGHT RUN REPORT",
        "=" * 74,
        f"  finished : {datetime.now():%Y-%m-%d %H:%M}",
        f"  duration : {elapsed_hours:.1f} h",
        f"  agents   : {args.agents}      spacing: {args.spacing} m",
        "",
    ]

    # ---- Stage 1 -------------------------------------------------------------
    if stage1:
        lines += ["-" * 74, " STAGE 1 -- BUDGET CHECK", "-" * 74]
        lines += [f"  {ln}" for ln in stage1["report_lines"]]
        lines.append("")

    # ---- Stage 2 -------------------------------------------------------------
    if stage2:
        summary, rung_order = stage2["summary"], stage2["rung_order"]

        lines += ["-" * 74, " STAGE 2 -- PILOT (mean +/- 95% CI)", "-" * 74]
        lines.append(f"  {'rung':<8} {'n':<4} {'mean':<12} {'std':<10} {'95% CI'}")
        lines.append("  " + "-" * 60)

        for rung in rung_order:
            if rung not in summary:
                continue
            s = summary[rung]
            ci = f"[{s['ci95_low']:.3f}, {s['ci95_high']:.3f}]"
            lines.append(
                f"  {rung:<8} {s['n_seeds']:<4} {s['mean']:<12.3f} {s['std']:<10.3f} {ci}"
            )

        lines.append("")
        lines += [f"  {ln}" for ln in verdict_lines(summary, rung_order, stage1, args)]

    lines += ["", "=" * 74]
    report = "\n".join(lines)

    (out_root / "REPORT.txt").write_text(report, encoding="utf-8")
    return report


def verdict_lines(summary: dict, rung_order: list[str], stage1: dict | None, args) -> list[str]:
    """The verdict, qualified by what Stage 1 established about the budget."""
    present = [r for r in rung_order if r in summary]
    if len(present) < 3:
        return ["VERDICT: INCOMPLETE -- fewer than 3 rungs produced results."]

    means = [summary[r]["mean"] for r in present]
    low, mid, high = means[0], means[1], means[-1]

    ci_mid = (summary[present[1]]["ci95_low"], summary[present[1]]["ci95_high"])
    ci_low = (summary[present[0]]["ci95_low"], summary[present[0]]["ci95_high"])
    ci_high = (summary[present[-1]]["ci95_low"], summary[present[-1]]["ci95_high"])

    beats_low = ci_mid[0] > ci_low[1]
    beats_high = ci_mid[0] > ci_high[1]

    out = ["VERDICT", "-" * 60]

    if mid > low and mid > high:
        out.append("SHAPE: interior optimum -- the middle rung beat BOTH endpoints.")
        if beats_low and beats_high:
            out += [
                "Confidence intervals separate from both endpoints.",
                "=> Consistent with Claim 1. PROCEED to the full sweep (Phase 4).",
            ]
        else:
            out += [
                "BUT confidence intervals overlap at least one endpoint.",
                "=> Suggestive, not established. The full sweep with >=8 seeds is",
                "   exactly what resolves this. PROCEED, treating it as suggestive.",
            ]
    elif low < mid < high:
        out.append("SHAPE: monotonic increasing -- more critic information kept helping.")
        if stage1 and not stage1["converged"]:
            out += [
                "CAUTION: Stage 1 showed training had NOT converged at this budget.",
                "A monotonic result here may reflect an unconverged run rather than",
                "a real absence of the effect. Re-run at a larger budget BEFORE",
                "firing kill-criterion K1.",
            ]
        else:
            out += [
                "Stage 1 confirmed the budget was sufficient.",
                "=> KILL-CRITERION K1. Pivot to Claim 3 (role-conditional critics)",
                "   as primary -- Charter §11 fallback A.",
            ]
    else:
        spread = abs(high - low) / (abs(low) if abs(low) > 1e-6 else 1.0)
        if spread < 0.05:
            out.append("SHAPE: flat -- no rung meaningfully differed.")
            if stage1 and not stage1["converged"]:
                out += [
                    "Stage 1 showed training had NOT converged.",
                    "=> This is kill-criterion K2 (budget too small), NOT K1.",
                    "   Increase the budget and re-run.",
                ]
            else:
                out += [
                    "Stage 1 confirmed convergence, so the task may be SATURATED.",
                    "=> Check whether returns are at ceiling (K2) before concluding.",
                ]
        else:
            out += [
                "SHAPE: decreasing or irregular.",
                "=> Inspect the learning curves; this may indicate a training",
                "   stability problem rather than a real effect.",
            ]

    out += [
        "",
        f"REMINDER: {args.stage2_seeds} seeds is a PILOT. It decides whether the full",
        "sweep is worth running. It is not a finding and must not be reported as one.",
    ]
    return out


# ==============================================================================
# SECTION 5 : MAIN
# ==============================================================================

def main() -> int:
    args = parse_args()
    workers = args.workers or max(1, (os.cpu_count() or 4) - 2)

    # ---- Plan ---------------------------------------------------------------
    s1_runs = 0 if args.skip_stage1 else args.stage1_seeds
    s2_runs = 0 if args.skip_stage2 else 3 * args.stage2_seeds

    s1_hours = estimate_hours(s1_runs, args.stage1_steps, workers) if s1_runs else 0.0
    s2_hours = estimate_hours(s2_runs, args.stage2_steps, workers) if s2_runs else 0.0
    total_hours = s1_hours + s2_hours

    print("=" * 74)
    print(" OVERNIGHT STAGED RUN  --  PLAN")
    print("=" * 74)
    print(f"  workers            : {workers}")
    print(f"  agents / spacing   : {args.agents} / {args.spacing} m")
    print()
    if s1_runs:
        print(f"  STAGE 1 budget check : {s1_runs} runs x {args.stage1_steps:,} steps  -> ~{s1_hours:.1f} h")
    if s2_runs:
        print(f"  STAGE 2 pilot        : {s2_runs} runs x {args.stage2_steps:,} steps  -> ~{s2_hours:.1f} h")
    print(f"  {'':21}{'':.<28}   ~{total_hours:.1f} h total")
    print(f"  estimated finish   : {(datetime.now() + timedelta(hours=total_hours)):%H:%M}")
    print("=" * 74)
    print()

    if args.dry_run:
        print("  (dry run -- nothing executed)")
        return 0

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = PROJECT_ROOT / "results" / "overnight" / stamp
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "git_commit.txt").write_text(get_git_commit(), encoding="utf-8")

    started = time.time()
    stage1 = stage2 = None

    try:
        if not args.skip_stage1:
            stage1 = run_stage1(args, out_root, workers)

        if not args.skip_stage2:
            stage2 = run_stage2(args, out_root, workers)

    except KeyboardInterrupt:
        print("\n[interrupted] writing partial report...")

    elapsed = (time.time() - started) / 3600.0
    report = write_report(out_root, args, stage1, stage2, elapsed)

    print()
    print(report)
    print(f"\n  artifacts -> {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
