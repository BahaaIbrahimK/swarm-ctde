#!/usr/bin/env python3
"""
================================================================================
 LEARNING SANITY CHECK  --  can the agent learn AT ALL?
================================================================================

RUN THIS BEFORE THE PILOT SWEEP.

--------------------------------------------------------------------------------
 WHY THIS EXISTS
--------------------------------------------------------------------------------
The pilot sweep costs hours and answers "does k matter?". That question is only
meaningful if the agent can learn the task in the first place.

If learning is broken -- bad reward scale, too-hard task, wrong hyperparameters
-- then EVERY rung sits at the same untrained-noise level, the pilot curve comes
back flat, and we could wrongly fire kill-criterion K1 and abandon the project's
primary claim for no reason. That is the most expensive possible mistake in this
codebase.

So: one rung, a few seeds, a moderate budget, and one question -- does the
agent actually learn?

--------------------------------------------------------------------------------
 WHY IT MUST BE RE-RUN FOR v4-correctness
--------------------------------------------------------------------------------
v4 changed FIVE things at once, any one of which could stop learning:

    observation      12-D -> 13-D (role flag)
    action dist.     unbounded Gaussian -> tanh-squashed
    reward scale     crash penalty rescaled to be monotone in survival time
    wind             applied on ALL 5 physics substeps (was 1 of 5 -- so the
                     REAL disturbance is now ~5x what v3 actually delivered)
    critic targets   value normalization added

The wind change deserves emphasis. The gust calibration (sigma = 4.0 m/s, chosen
as "best variance, still 0% crashes") was measured while wind reached the drones
at 20% strength. That calibration is therefore STALE, and the task may now be
harder than intended. This script is where we find that out -- in minutes, not
after a sweep.

--------------------------------------------------------------------------------
 WHAT GOOD OUTPUT LOOKS LIKE
--------------------------------------------------------------------------------
  * eval_return should IMPROVE (become less negative) over training
  * explained_variance should RISE above 0 -- the critic is learning to predict
    returns rather than guessing the mean
  * episode length should grow if early episodes end in crashes

If return is flat AND explained_variance stays near 0, learning is broken.
Diagnose that here, cheaply, before spending sweep hours.

USAGE
-----
    python scripts/check_learning.py                    # k1, 3 seeds, 30k steps
    python scripts/check_learning.py --seeds 1 --steps 60000
    python scripts/check_learning.py --rung k3
    python scripts/check_learning.py --no-save          # skip writing results/
================================================================================
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import argparse
import json

import numpy as np

from ctde.metrics import format_metrics_table
from ctde.models.observability import build_ladder
from ctde.training.mappo import MAPPOConfig
from ctde.training.trainer import RunConfig, Trainer
from ctde.utils.run_naming import (
    TASK_VERSION,
    make_experiment_dir,
    make_run_name,
    write_manifest,
)


def run_one_seed(level, seed: int, args) -> tuple[list[dict], dict]:
    """Train one seed and return (history, final held-out evaluation)."""
    config = RunConfig(
        level=level,
        seed=seed,
        num_agents=args.agents,
        formation_spacing_m=args.spacing,
        total_env_steps=args.steps,
        eval_every_steps=max(args.steps // 10, 1_000),
        # eval_episodes deliberately NOT set here -- it inherits RunConfig's
        # calibrated default (8). Hardcoding 3 here silently overrode that
        # default, so a corrected protocol produced numbers byte-identical to
        # the broken one. See RunConfig on why 3 cannot resolve +109.
        mappo=MAPPOConfig(),
    )

    out_dir = None
    if args.out_dir is not None:
        out_dir = Path(args.out_dir) / "runs" / make_run_name(
            level.label, seed, args.spacing, args.agents
        )

    trainer = Trainer(config, out_dir)
    try:
        history = trainer.train(verbose=args.verbose)
        # The FINAL number uses the held-out seed block -- never the episodes
        # the periodic curve was computed on. See RunConfig.eval_seeds.
        final = trainer.evaluate(final=True)
        final_metrics = trainer.last_eval_metrics
    finally:
        trainer.close()

    if out_dir is not None:
        with open(out_dir / "final_eval.json", "w", encoding="utf-8") as f:
            json.dump(final, f, indent=2)

    return history, {"final": final, "episodes": final_metrics, "config": config}


def summarise_seed(history: list[dict], config: RunConfig) -> dict:
    """Early-vs-late comparison for ONE seed."""
    returns = [h["eval_return"] for h in history]
    explained = [h["explained_variance"] for h in history]
    lengths = [h["eval_episode_len"] for h in history]

    # THE DIAGNOSTIC THAT WOULD HAVE CAUGHT v1's FAILURE.
    #
    # In v1 the return "improved" 59% while episodes stayed 20-100 steps out of
    # 720: the policy was learning to CRASH FASTER, because a per-step penalty
    # plus early termination made dying cheaper than flying badly. Rising return
    # alone is therefore NOT evidence of learning.
    max_episode_steps = config.episode_len_sec * config.ctrl_freq
    completion = [ln / max_episode_steps for ln in lengths]

    # Compare first and last thirds rather than single points, so one noisy
    # evaluation cannot flip the conclusion.
    third = max(len(returns) // 3, 1)
    early_return = float(np.mean(returns[:third]))
    late_return = float(np.mean(returns[-third:]))
    improvement = late_return - early_return
    denom = abs(early_return) if abs(early_return) > 1e-6 else 1.0

    return {
        "early_return": early_return,
        "late_return": late_return,
        "improvement": improvement,
        "rel_improvement": improvement / denom,
        "early_ev": float(np.mean(explained[:third])),
        "late_ev": float(np.mean(explained[-third:])),
        "early_completion": float(np.mean(completion[:third])),
        "late_completion": float(np.mean(completion[-third:])),
        "max_episode_steps": max_episode_steps,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify the agent can learn the task.")
    parser.add_argument("--steps", type=int, default=30_000)
    parser.add_argument("--agents", type=int, default=4)
    parser.add_argument("--seeds", type=int, default=3,
                        help="Number of seeds (0..N-1). One seed cannot "
                             "distinguish learning from a lucky initialisation.")
    parser.add_argument("--rung", type=str, default="k1",
                        help="Which rung to test (default k1)")
    parser.add_argument("--spacing", type=float, default=1.0)
    parser.add_argument("--no-save", action="store_true",
                        help="Do not write anything under results/")
    parser.add_argument("--quiet", dest="verbose", action="store_false",
                        help="Suppress per-evaluation training output")
    parser.set_defaults(verbose=True)
    args = parser.parse_args()

    # ---- Pick the rung ------------------------------------------------------
    ladder = {lv.label: lv for lv in build_ladder(args.agents)}
    if args.rung not in ladder:
        print(f"[ERROR] unknown rung '{args.rung}'. Options: {sorted(ladder)}")
        return 1
    level = ladder[args.rung]

    # ---- Output directory ---------------------------------------------------
    # Registered layout: results/<experiment_id>/<date>_<task_version>/
    # The task version is IN THE PATH, so these results can never be silently
    # pooled with the invalid v2/v3 runs in results/_archive/.
    args.out_dir = None
    if not args.no_save:
        args.out_dir = make_experiment_dir(PROJECT_ROOT / "results", "EXP01_learnability")
        write_manifest(
            args.out_dir,
            "EXP01_learnability",
            rungs=[level.label],
            seeds=list(range(args.seeds)),
            spacings=[args.spacing],
            total_env_steps=args.steps,
            num_agents=args.agents,
            notes=(
                "Post-v4 learnability check. v4 changed the observation, the "
                "action distribution, the reward scale, wind application (1 of 5 "
                "substeps -> all 5), and added value normalization. None of it "
                "was verified to still LEARN. Gust sigma=4.0 was calibrated "
                "under the broken 20%-strength wind, so it may now be too hard."
            ),
        )

    print("=" * 74)
    print(" LEARNING SANITY CHECK")
    print("=" * 74)
    print("  Question: does the agent learn AT ALL under v4-correctness?")
    print()
    print("  If learning is broken, every rung sits at untrained noise, the pilot")
    print("  curve looks flat, and we could wrongly fire kill-criterion K1 and")
    print("  abandon Claim 1 for no reason. Check that here first -- cheaply.")
    print()
    print(f"  task    : {TASK_VERSION}")
    print(f"  rung    : {level.label}")
    print(f"  agents  : {args.agents}")
    print(f"  seeds   : {args.seeds}")
    print(f"  steps   : {args.steps:,} per seed")
    print(f"  spacing : {args.spacing} m")
    print(f"  output  : {args.out_dir or '(not saved)'}")
    print("=" * 74)
    print()

    # ---- Train every seed ---------------------------------------------------
    summaries, finals, all_episodes = [], [], []
    for seed in range(args.seeds):
        print(f"\n--- seed {seed} " + "-" * 58)
        history, extra = run_one_seed(level, seed, args)
        if not history:
            print(f"\n[ERROR] seed {seed} produced no evaluation records.")
            return 1
        summaries.append(summarise_seed(history, extra["config"]))
        finals.append(extra["final"])
        all_episodes.extend(extra["episodes"])

    # ---- Analyse across seeds -----------------------------------------------
    def col(key: str) -> np.ndarray:
        return np.array([s[key] for s in summaries], dtype=float)

    max_episode_steps = summaries[0]["max_episode_steps"]

    print()
    print("=" * 74)
    print(f" ANALYSIS  (mean +/- std across {args.seeds} seeds)")
    print("=" * 74)
    print(f"  {'metric':<26} {'early':<16} {'late':<16} {'change':<14}")
    print("  " + "-" * 68)

    for label, early_key, late_key, fmt in [
        ("eval return", "early_return", "late_return", "{:.2f}"),
        ("explained variance", "early_ev", "late_ev", "{:.3f}"),
    ]:
        e, l = col(early_key), col(late_key)
        es = f"{fmt.format(e.mean())} ± {fmt.format(e.std())}"
        ls = f"{fmt.format(l.mean())} ± {fmt.format(l.std())}"
        print(f"  {label:<26} {es:<16} {ls:<16} {l.mean() - e.mean():+.3f}")

    ec, lc = col("early_completion"), col("late_completion")
    es = f"{ec.mean():.1%} ± {ec.std():.1%}"
    ls = f"{lc.mean():.1%} ± {lc.std():.1%}"
    print(f"  {'episode COMPLETION':<26} {es:<16} {ls:<16} {lc.mean() - ec.mean():+.1%}")

    print()
    print(f"  (full episode = {max_episode_steps:.0f} steps. v1 FAILED here:")
    print("   episodes stayed at 3-14% while return 'improved' 59%.)")

    # ---- Physical metrics ---------------------------------------------------
    # The headline numbers (Charter §S.11), from the HELD-OUT evaluation.
    print()
    print(format_metrics_table(all_episodes, label=f"{level.label}, held-out seeds"))

    # ---- Verdict ------------------------------------------------------------
    print()
    print("=" * 74)
    print(" VERDICT")
    print("=" * 74)

    rel = col("rel_improvement")
    late_ev = col("late_ev")
    late_completion = col("late_completion")

    # Judged on the MEAN across seeds; per-seed detail printed below so a single
    # diverging seed is visible rather than averaged away.
    learning = rel.mean() > 0.05
    critic_working = late_ev.mean() > 0.1
    surviving = late_completion.mean() > 0.8

    print(f"  {'seed':<6} {'rel. improvement':<20} {'expl. var':<14} {'completion':<12}")
    print("  " + "-" * 56)
    for i, sm in enumerate(summaries):
        print(f"  {i:<6} {sm['rel_improvement']:>+18.1%}   "
              f"{sm['late_ev']:>12.3f}   {sm['late_completion']:>10.1%}")
    print()

    exit_code = 0
    if not surviving:
        print("  FAIL -- episodes are NOT completing.")
        print(f"    completion {late_completion.mean():.1%} (need > 80%)")
        print()
        print("  THIS IS EXACTLY HOW v1 FAILED. Rising return with short episodes")
        print("  means the policy is learning to CRASH FASTER, not to fly: a")
        print("  per-step penalty plus early termination makes dying cheaper than")
        print("  flying badly.")
        print()
        print("  CHECK THE TASK BEFORE THE LEARNER. Run a PURE PID (zero action)")
        print("  at this gust strength: if IT cannot survive either, the task is")
        print("  unsurvivable and no policy can learn it.")
        print()
        print("  This exact failure happened on 16 Sep 2026: gust_std_ms=4.0 had")
        print("  been calibrated while wind reached the drones on 1 of 5 substeps.")
        print("  At full strength a pure PID crashed 40% of the time -- matching")
        print("  the trained policies' 73% completion almost exactly. Recalibrated")
        print("  to 3.0 (53.6 seed std, 0% crashes). See trainer.py.")
        print()
        print("  DO NOT run any sweep.")
        exit_code = 1
    elif learning and critic_working:
        print("  PASS -- the agent is learning, surviving, and the critic is predictive.")
        print(f"    return improved {rel.mean():.1%}")
        print(f"    episode completion {late_completion.mean():.1%}")
        print(f"    explained variance {late_ev.mean():.3f}")
        print()
        print("  => Safe to proceed to the §S.10 diagnostics and the pilot sweep.")
    elif learning and not critic_working:
        print("  PARTIAL -- the policy improves but the critic predicts poorly.")
        print(f"    explained variance {late_ev.mean():.3f} (want > 0.1)")
        print()
        print("  The critic is central to this paper: if it cannot predict returns at")
        print("  ANY rung, differences BETWEEN rungs will be dominated by noise --")
        print("  and a flat ladder would be uninterpretable rather than evidence")
        print("  of kill-criterion K1.")
        print()
        print("  Consider a longer budget, a higher critic learning rate, or checking")
        print("  the reward scale before committing to the sweep.")
        exit_code = 2
    else:
        print("  FAIL -- no meaningful learning detected.")
        print(f"    return change {rel.mean():+.1%}, explained variance {late_ev.mean():.3f}")
        print()
        print("  DO NOT run the pilot sweep yet, and do NOT interpret a flat pilot")
        print("  curve as kill-criterion K1 -- the cause would be broken learning,")
        print("  not an absent effect. Check, in order:")
        print("    1. Gust strength   -- sigma=4.0 was calibrated under 20% wind")
        print("    2. Reward scale    -- are returns dominated by one term?")
        print("    3. Task difficulty -- can drones even stay airborne? (--steps 60000)")
        print("    4. Action scaling  -- is +/-0.2 m of offset enough authority?")
        exit_code = 1

    # ---- Persist ------------------------------------------------------------
    if args.out_dir is not None:
        report = {
            "task_version": TASK_VERSION,
            "rung": level.label,
            "seeds": args.seeds,
            "steps_per_seed": args.steps,
            "spacing_m": args.spacing,
            "per_seed": summaries,
            "final_eval": finals,
            "verdict": {
                "learning": bool(learning),
                "critic_working": bool(critic_working),
                "surviving": bool(surviving),
                "exit_code": exit_code,
            },
        }
        with open(args.out_dir / "analysis" / "learnability.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print()
        print(f"  saved: {args.out_dir}")

    print("=" * 74)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
