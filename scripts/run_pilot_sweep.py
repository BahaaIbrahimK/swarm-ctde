#!/usr/bin/env python3
"""
================================================================================
 PHASE 3 PILOT SWEEP  --  THE DECISION GATE
================================================================================

Charter §5 Phase 3, and kill-criterion K1 (Charter §3).

--------------------------------------------------------------------------------
 WHAT THIS SCRIPT IS FOR
--------------------------------------------------------------------------------
This is the cheapest possible test of whether Claim 1's effect exists at all.

    Rungs : k0, k1, k(N-1)     -- the two endpoints plus one interior point
    Seeds : 5                  -- enough to see an effect, not enough to publish
    Budget: reduced

PURPOSE: KILL THE PROJECT CHEAPLY IF THE EFFECT IS NOT THERE.

If the curve comes back flat or monotonic, kill-criterion K1 fires and we pivot
to Claim 3 as primary -- in week 6, having spent days, rather than in month 5
having spent months.

--------------------------------------------------------------------------------
 HOW TO READ THE RESULT
--------------------------------------------------------------------------------
Three rungs are enough to distinguish the two competing hypotheses:

    MONOTONIC (the field is right):     k0 < k1 < k(N-1)
    INTERIOR OPTIMUM (Claim 1):         k1 > both k0 and k(N-1)

The interior point must beat BOTH endpoints. Beating only k0 would merely
re-show that centralization helps -- known since MADDPG (2017).

IMPORTANT: the pilot is UNDERPOWERED BY DESIGN. It cannot establish Claim 1;
it can only tell us whether the full sweep is worth running. Do not report
pilot numbers as findings.

USAGE
-----
    python scripts/run_pilot_sweep.py                       # default pilot
    python scripts/run_pilot_sweep.py --steps 50000         # shorter
    python scripts/run_pilot_sweep.py --seeds 3 --workers 4
    python scripts/run_pilot_sweep.py --smoke               # ~2 min sanity run
================================================================================
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import argparse
import json
from datetime import datetime

import numpy as np

from ctde.models.observability import ObservabilityLevel, build_ladder, critic_input_dim
from ctde.training.mappo import MAPPOConfig
from ctde.training.sweep import aggregate_by_rung, run_sweep
from ctde.training.trainer import RunConfig
from ctde.utils.seeding import get_git_commit


# ==============================================================================
# SECTION 1 : ARGUMENTS
# ==============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 3 pilot sweep (decision gate).")
    p.add_argument("--agents", type=int, default=4, help="Team size (default 4)")
    p.add_argument("--seeds", type=int, default=5, help="Number of seeds (default 5)")
    p.add_argument("--steps", type=int, default=100_000, help="Env steps per run")
    p.add_argument("--spacing", type=float, default=1.0, help="Formation spacing (m)")
    p.add_argument("--workers", type=int, default=None, help="Parallel processes")
    p.add_argument("--smoke", action="store_true",
                   help="Tiny run (~2 min) to verify the pipeline end to end")
    return p.parse_args()


# ==============================================================================
# SECTION 2 : VERDICT
# ==============================================================================

def interpret(
    summary: dict[str, dict[str, float]],
    rung_order: list[str],
    total_steps: int,
    is_smoke: bool = False,
) -> str:
    """Classify the pilot's shape and state the consequence.

    Deliberately conservative: with 5 seeds we can only distinguish "worth
    pursuing" from "probably not". Overlapping confidence intervals are
    reported as INCONCLUSIVE rather than spun as support.

    GUARD -- WHY THIS MATTERS
    -------------------------
    A verdict is only meaningful if the policies actually LEARNED something.
    On a tiny budget every rung sits at untrained-network noise, and comparing
    those means produces a confident-looking but entirely spurious verdict.

    Firing kill-criterion K1 on noise would abandon the project's primary claim
    for no reason -- the most expensive possible failure mode of this script. So
    we refuse to issue any verdict below a minimum budget.
    """
    # Below this, no policy has meaningfully trained and differences between
    # rungs are initialisation noise rather than learning.
    MIN_STEPS_FOR_VERDICT = 20_000

    if is_smoke or total_steps < MIN_STEPS_FOR_VERDICT:
        return "\n".join([
            "VERDICT: NONE -- BUDGET TOO SMALL",
            "-" * 70,
            f"  This run used {total_steps:,} env steps; a verdict requires at least",
            f"  {MIN_STEPS_FOR_VERDICT:,}. At this budget no policy has learned anything,",
            "  so all rungs sit at untrained-network noise and any apparent ordering",
            "  is meaningless.",
            "",
            "  What this run DOES verify: the pipeline executes end to end --",
            "  environment, learner, parallel sweep, aggregation, and artifacts.",
            "",
            "  Run the real pilot for a verdict:",
            "      python scripts/run_pilot_sweep.py --steps 100000 --seeds 5",
        ])

    present = [r for r in rung_order if r in summary]
    if len(present) < 3:
        return "INCOMPLETE -- fewer than 3 rungs produced results."

    means = [summary[r]["mean"] for r in present]
    low, mid, high = means[0], means[1], means[-1]

    lines = [
        "VERDICT",
        "-" * 70,
    ]

    # Do the confidence intervals separate at all?
    ci_mid = (summary[present[1]]["ci95_low"], summary[present[1]]["ci95_high"])
    ci_low = (summary[present[0]]["ci95_low"], summary[present[0]]["ci95_high"])
    ci_high = (summary[present[-1]]["ci95_low"], summary[present[-1]]["ci95_high"])

    separated_from_low = ci_mid[0] > ci_low[1]
    separated_from_high = ci_mid[0] > ci_high[1]

    if mid > low and mid > high:
        lines.append("  SHAPE: interior optimum -- the middle rung beat BOTH endpoints.")
        if separated_from_low and separated_from_high:
            lines.append("  CIs separate from both endpoints.")
            lines.append("  => Consistent with Claim 1. PROCEED to the full sweep (Phase 4).")
        else:
            lines.append("  BUT confidence intervals OVERLAP an endpoint.")
            lines.append("  => Encouraging but not established. The full sweep with >=8")
            lines.append("     seeds is exactly what is needed to resolve it. PROCEED,")
            lines.append("     treating this as suggestive only.")
    elif low < mid < high:
        lines.append("  SHAPE: monotonic increasing -- more critic information kept helping.")
        lines.append("  => Consistent with the FIELD'S ASSUMPTION, not with Claim 1.")
        lines.append("  => KILL-CRITERION K1. Pivot to Claim 3 (role-conditional critics)")
        lines.append("     as the primary claim -- see Charter §11 fallback A.")
    elif abs(high - low) < 0.05 * abs(low if low != 0 else 1.0):
        lines.append("  SHAPE: flat -- no rung meaningfully differed.")
        lines.append("  => Either the task is too easy (saturated) or the budget is too")
        lines.append("     small for differences to emerge. Check whether returns are at")
        lines.append("     ceiling (kill-criterion K2) BEFORE concluding anything.")
    else:
        lines.append("  SHAPE: decreasing or irregular.")
        lines.append("  => Inspect learning curves before drawing any conclusion; this may")
        lines.append("     indicate a training-stability problem rather than a real effect.")

    lines += [
        "",
        "  REMINDER: the pilot is UNDERPOWERED BY DESIGN (5 seeds). It decides",
        "  whether the full sweep is worth running. It is NOT a finding and must",
        "  not be reported as one.",
    ]
    return "\n".join(lines)


# ==============================================================================
# SECTION 3 : MAIN
# ==============================================================================

def main() -> int:
    args = parse_args()

    # ---- Pilot ladder: two endpoints + one interior point -------------------
    full_ladder = build_ladder(args.agents, include_privileged_rung=False)
    pilot_levels = [
        full_ladder[0],                    # k0  -- decentralized endpoint
        full_ladder[1],                    # k1  -- interior candidate
        full_ladder[-1],                   # k(N-1) -- standard CTDE endpoint
    ]

    # ---- Budget -------------------------------------------------------------
    if args.smoke:
        total_steps = 2_048
        seeds = list(range(2))
        mappo = MAPPOConfig(num_steps=256, num_epochs=2, minibatch_size=128)
        episode_len = 3.0
        eval_every, eval_eps = 1_024, 1
    else:
        total_steps = args.steps
        seeds = list(range(args.seeds))
        mappo = MAPPOConfig()
        episode_len = 15.0
        # None -> inherit RunConfig's calibrated default (8). See RunConfig:
        # 3 episodes cannot resolve the effects this study reports.
        eval_every, eval_eps = max(total_steps // 10, 1_000), None

    base = RunConfig(
        level=pilot_levels[0],             # overridden per job
        num_agents=args.agents,
        formation_spacing_m=args.spacing,
        total_env_steps=total_steps,
        episode_len_sec=episode_len,
        eval_every_steps=eval_every,
        **({} if eval_eps is None else {"eval_episodes": eval_eps}),
        mappo=mappo,
    )

    # ---- Output -------------------------------------------------------------
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = "smoke" if args.smoke else "pilot"
    out_root = PROJECT_ROOT / "results" / f"phase3_{tag}" / stamp
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "git_commit.txt").write_text(get_git_commit(), encoding="utf-8")

    print("=" * 74)
    print(f" PHASE 3 {'SMOKE TEST' if args.smoke else 'PILOT SWEEP'}  --  DECISION GATE")
    print("=" * 74)
    print("  Purpose: determine cheaply whether Claim 1's effect exists at all.")
    print("  If the curve is flat or monotonic, kill-criterion K1 fires and we")
    print("  pivot to Claim 3 -- now, rather than after months of work.")
    print()
    print("  Rungs under test:")
    for lv in pilot_levels:
        print(f"    {lv.label:<6} critic input {critic_input_dim(lv):>3}-D")
    print()

    # ---- Run ----------------------------------------------------------------
    results = run_sweep(
        levels=pilot_levels,
        seeds=seeds,
        base_config=base,
        output_root=out_root,
        num_workers=args.workers,
        verbose=True,
    )

    # ---- Aggregate ----------------------------------------------------------
    # Aggregate WITHIN each spacing. Spacing is Claim 2's independent variable,
    # so pooling across it would average away the effect under test.
    spacings_present = sorted({r["formation_spacing_m"] for r in results})
    if len(spacings_present) > 1:
        raise SystemExit(
            f"this script reports ONE spacing but the results contain "
            f"{spacings_present}. Use the Claim-2 sweep, which reports each "
            f"spacing separately -- pooling them would average across Claim 2's "
            f"independent variable."
        )
    summary = aggregate_by_rung(results, spacing=spacings_present[0])
    rung_order = [lv.label for lv in pilot_levels]

    print()
    print("=" * 74)
    print(" RESULTS  (mean +/- 95% CI across seeds)")
    print("=" * 74)
    print(f"  {'rung':<8} {'n':<4} {'mean':<12} {'std':<10} {'95% CI':<24}")
    print("  " + "-" * 62)
    for rung in rung_order:
        if rung not in summary:
            continue
        s = summary[rung]
        ci = f"[{s['ci95_low']:.2f}, {s['ci95_high']:.2f}]"
        print(f"  {rung:<8} {s['n_seeds']:<4} {s['mean']:<12.3f} {s['std']:<10.3f} {ci:<24}")

    print()
    print(interpret(summary, rung_order, total_steps=total_steps, is_smoke=args.smoke))
    print("=" * 74)

    with open(out_root / "summary.json", "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "rung_order": rung_order}, f, indent=2)

    print(f"\n  artifacts -> {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
