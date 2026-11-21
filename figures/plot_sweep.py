#!/usr/bin/env python3
"""
================================================================================
 PLOT A SWEEP  --  the paper's figures
================================================================================

USAGE
-----
    # Point it at any result directory; it works out the rest.
    python figures/plot_sweep.py results/<experiment_id>/<date>_<task_version>

    # Choose where figures go
    python figures/plot_sweep.py <results_dir> --out figures/output/pilot

    # One specific figure
    python figures/plot_sweep.py <results_dir> --only performance

WHAT IT PRODUCES
----------------
  1. performance_vs_k   THE HEADLINE FIGURE. Performance against critic
                        observability, with per-seed points and 95% CIs.
                        This is where Claim 1 lives or dies.

  2. learning_curves    Return over training, one panel per rung. Shows whether
                        runs converged and how much seeds disagreed.

  3. mechanism          Explained variance vs k, beside performance vs k.
                        The §1.5 argument predicts these DIVERGE: the critic
                        keeps getting better at prediction as k grows, while
                        performance peaks and falls.

  4. task_health        Episode completion, crash rate, lateral error, coupling.
                        Confirms the result is not an artifact of a broken task.

  5. seed_variability   Per-seed spread. Shows honestly whether the study is
                        powered enough to distinguish rungs.

Every figure is written as PDF (for the paper) and PNG (for viewing).
================================================================================
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import argparse

import numpy as np

from figures.loader import Sweep, compare_rungs, interior_optimum_test, load_sweep
from figures.style import (
    COLORS, DOUBLE_COL, GOLDEN, RUNG_COLORS, RUNG_MARKERS, SINGLE_COL,
    annotate_pilot_warning, apply_style, save, significance_stars,
)

import matplotlib.pyplot as plt

# Set from --horizon in main(); every figure honours it.
HORIZON = None


# ==============================================================================
# FIGURE 1 : PERFORMANCE vs k   (the headline)
# ==============================================================================

def fig_performance_vs_k(sweep: Sweep, out_dir: Path, is_pilot: bool = True):
    """THE figure the paper stands on.

    Design decisions:
      * Individual seed points are drawn, not just the mean. Hiding the spread
        would misrepresent how much of the story is noise -- and with 5 seeds,
        the spread IS the story.
      * Error bars are 95% CI using the t-distribution (correct for small n).
      * The best interior rung is annotated only when it beats both endpoints,
        since that is the specific shape Claim 1 predicts.
    """
    fig, ax = plt.subplots(figsize=(SINGLE_COL * 1.4, SINGLE_COL * 1.4 * GOLDEN))

    rungs = sweep.rungs
    summary = sweep.summary(horizon=HORIZON)
    x = np.arange(len(rungs))

    means = [summary[r]["mean"] for r in rungs]
    errs = [summary[r]["ci95"] for r in rungs]

    # ---- Per-seed points, jittered so overlapping values stay visible -------
    rng = np.random.default_rng(0)          # fixed: jitter must be reproducible
    for i, rung in enumerate(rungs):
        vals = sweep.final_returns(rung, horizon=HORIZON)
        jitter = rng.uniform(-0.09, 0.09, size=len(vals))
        ax.scatter(
            x[i] + jitter, vals,
            s=22, alpha=0.45, zorder=2,
            color=RUNG_COLORS.get(rung, COLORS["blue"]),
            edgecolors="none",
            label="individual seeds" if i == 0 else None,
        )

    # ---- Mean with 95% CI ---------------------------------------------------
    ax.errorbar(
        x, means, yerr=errs,
        marker="o", markersize=7, linewidth=2, capsize=4, capthick=1.4,
        color=COLORS["black"], ecolor=COLORS["black"],
        markerfacecolor=COLORS["blue"], markeredgecolor=COLORS["black"],
        zorder=3, label="mean ± 95% CI",
    )

    # ---- Annotate an interior optimum, if there is one ----------------------
    test = interior_optimum_test(sweep, horizon=HORIZON)
    if "error" not in test and test["beats_both_endpoints"]:
        best = test["best_interior_rung"]
        bi = rungs.index(best)
        ax.annotate(
            f"interior optimum\n$k^*$ = {best}",
            xy=(bi, means[bi]),
            xytext=(bi + 0.35, means[bi] + (max(means) - min(means)) * 0.55),
            fontsize=8, ha="left",
            arrowprops=dict(arrowstyle="->", lw=1.0, color=COLORS["red"]),
            color=COLORS["red"],
        )

    ax.set_xticks(x)
    ax.set_xticklabels(rungs)
    ax.set_xlabel("Critic observability $k$  (teammates visible to the critic)")
    ax.set_ylabel("Final return")
    ax.set_title("Performance vs critic observability")
    ax.legend(loc="best")

    # Secondary axis labelling what each rung MEANS, so the reader does not
    # have to remember the encoding.
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(x)
    ax2.set_xticklabels(
        ["decentralized" if r == rungs[0]
         else "maximal" if r == rungs[-1]
         else "" for r in rungs],
        fontsize=7, color=COLORS["grey"],
    )
    ax2.tick_params(length=0)
    for side in ("top", "right", "left", "bottom"):
        ax2.spines[side].set_visible(False)

    if is_pilot:
        annotate_pilot_warning(fig)

    fig.tight_layout()
    return save(fig, out_dir, "fig1_performance_vs_k")


# ==============================================================================
# FIGURE 2 : LEARNING CURVES
# ==============================================================================

def fig_learning_curves(sweep: Sweep, out_dir: Path, is_pilot: bool = True):
    """Return over training, one panel per rung.

    Answers two questions the summary table cannot:
      * Did runs CONVERGE, or were they still improving when the budget ran out?
      * How much did seeds disagree, and did any collapse?
    """
    rungs = sweep.rungs
    fig, axes = plt.subplots(
        1, len(rungs),
        figsize=(DOUBLE_COL, DOUBLE_COL * 0.32),
        sharey=True,
    )
    if len(rungs) == 1:
        axes = [axes]

    for ax, rung in zip(axes, rungs):
        runs = sweep.by_rung(rung)
        color = RUNG_COLORS.get(rung, COLORS["blue"])

        # Individual seeds, faint.
        for run in runs:
            ax.plot(run.steps / 1000, run.returns, lw=0.8, alpha=0.35, color=color)

        # Mean across seeds, bold. Curves share an evaluation schedule, so we
        # can align by index; trim to the shortest in case one run stopped early.
        n = min(len(r.returns) for r in runs)
        steps = runs[0].steps[:n] / 1000
        stacked = np.vstack([r.returns[:n] for r in runs])
        mean = stacked.mean(axis=0)
        sem = stacked.std(axis=0, ddof=1) / np.sqrt(len(runs)) if len(runs) > 1 else np.zeros(n)

        ax.plot(steps, mean, lw=2, color=color, label="mean")
        ax.fill_between(steps, mean - sem, mean + sem, alpha=0.2, color=color)

        ax.set_title(f"{rung}", fontsize=9)
        ax.set_xlabel("Env steps (thousands)")

    axes[0].set_ylabel("Evaluation return")
    fig.suptitle("Learning curves by observability rung  (shaded: ± SEM)", y=1.02)

    if is_pilot:
        annotate_pilot_warning(fig)

    fig.tight_layout()
    return save(fig, out_dir, "fig2_learning_curves")


# ==============================================================================
# FIGURE 3 : MECHANISM
# ==============================================================================

def fig_mechanism(sweep: Sweep, out_dir: Path, is_pilot: bool = True):
    """Explained variance vs k, beside performance vs k.

    THIS IS THE MOST SCIENTIFICALLY INTERESTING FIGURE.

    Charter §1.5 predicts these two curves DIVERGE:

      * explained variance should rise MONOTONICALLY with k -- more information
        always makes the critic a better PREDICTOR;
      * performance should PEAK and then fall -- because beyond some k, the
        extra prediction accuracy comes from information the actor cannot act
        on, and it turns into noise in the policy gradient.

    A monotone-rising left panel beside a peaked right panel is exactly the
    signature of representational mismatch. It separates "the critic got worse"
    (which would be a boring explanation) from "the critic got better but less
    useful" (which is the paper's actual claim).
    """
    fig, (ax_left, ax_right) = plt.subplots(
        1, 2, figsize=(DOUBLE_COL, DOUBLE_COL * 0.38)
    )

    rungs = sweep.rungs
    x = np.arange(len(rungs))

    # ---- LEFT: explained variance -------------------------------------------
    ev_means, ev_errs = [], []
    for rung in rungs:
        vals = np.array([np.mean(r.explained_variance[-3:]) for r in sweep.by_rung(rung)])
        vals = vals[np.isfinite(vals)]
        ev_means.append(vals.mean() if len(vals) else np.nan)
        ev_errs.append(vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0)

    ax_left.errorbar(
        x, ev_means, yerr=ev_errs,
        marker="s", markersize=6, linewidth=2, capsize=4,
        color=COLORS["green"],
    )
    ax_left.set_xticks(x)
    ax_left.set_xticklabels(rungs)
    ax_left.set_xlabel("Critic observability $k$")
    ax_left.set_ylabel("Explained variance")
    ax_left.set_title("Critic prediction quality")
    ax_left.text(
        0.5, -0.42,
        "expected: rises monotonically\n(more info → better predictor)",
        transform=ax_left.transAxes, ha="center", va="top",
        fontsize=7, style="italic", color=COLORS["grey"],
    )

    # ---- RIGHT: performance -------------------------------------------------
    summary = sweep.summary(horizon=HORIZON)
    perf = [summary[r]["mean"] for r in rungs]
    perf_err = [summary[r]["ci95"] for r in rungs]

    ax_right.errorbar(
        x, perf, yerr=perf_err,
        marker="o", markersize=6, linewidth=2, capsize=4,
        color=COLORS["blue"],
    )
    ax_right.set_xticks(x)
    ax_right.set_xticklabels(rungs)
    ax_right.set_xlabel("Critic observability $k$")
    ax_right.set_ylabel("Final return")
    ax_right.set_title("Task performance")
    ax_right.text(
        0.5, -0.42,
        "Claim 1: peaks at an interior $k$\n(extra info becomes unusable)",
        transform=ax_right.transAxes, ha="center", va="top",
        fontsize=7, style="italic", color=COLORS["grey"],
    )

    fig.suptitle(
        "Mechanism: the critic keeps improving while performance peaks",
        y=1.02,
    )

    if is_pilot:
        annotate_pilot_warning(fig)

    # Extra bottom room so the explanatory captions clear the x-axis labels.
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.30)
    return save(fig, out_dir, "fig3_mechanism")


# ==============================================================================
# FIGURE 4 : TASK HEALTH
# ==============================================================================

def fig_task_health(sweep: Sweep, out_dir: Path, is_pilot: bool = True):
    """Four diagnostics confirming the result is not a task artifact.

    Five bugs were found in this task, four invisible without measurement.
    These panels make the relevant failure modes visible in the results
    themselves rather than trusting that they were fixed:

      episode completion -- v1 failed here (policies learned to crash faster)
      crash rate         -- termination effects contaminating the reward
      lateral error      -- THE SIDESTEP EXPLOIT. If policies drift laterally
                            they are dodging the wake, deleting the coupling
                            this paper studies. Results would be invalid.
      coupling           -- confirms downwash survived training
    """
    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COL, DOUBLE_COL * 0.62))
    rungs = sweep.rungs
    x = np.arange(len(rungs))

    panels = [
        (axes[0, 0], "episode_len", "Episode length (steps)",
         "Episode completion", COLORS["blue"], 720),
        (axes[0, 1], "crash_rate", "Crash rate",
         "Crashes (want ≈ 0)", COLORS["red"], None),
        (axes[1, 0], "lateral_error", "Mean lateral error (m)",
         "Sidestep-exploit monitor", COLORS["orange"], None),
        (axes[1, 1], "coupling", "Mean coupling (N)",
         "Downwash during training", COLORS["green"], None),
    ]

    for ax, attr, ylabel, title, color, reference in panels:
        means, errs = [], []
        for rung in rungs:
            vals = np.array([np.mean(getattr(r, attr)[-3:]) for r in sweep.by_rung(rung)])
            vals = vals[np.isfinite(vals)]
            means.append(vals.mean() if len(vals) else np.nan)
            errs.append(vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0)

        ax.errorbar(x, means, yerr=errs, marker="o", markersize=5,
                    linewidth=1.6, capsize=3, color=color)

        if reference is not None:
            ax.axhline(reference, ls="--", lw=1, color=COLORS["grey"],
                       label=f"full episode ({reference})")
            ax.legend(fontsize=7)

        ax.set_xticks(x)
        ax.set_xticklabels(rungs)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_title(title, fontsize=9)

    for ax in axes[1, :]:
        ax.set_xlabel("Critic observability $k$")

    fig.suptitle("Task health — is the result an artifact?", y=1.00)

    if is_pilot:
        annotate_pilot_warning(fig)

    fig.tight_layout()
    return save(fig, out_dir, "fig4_task_health")


# ==============================================================================
# FIGURE 5 : SEED VARIABILITY
# ==============================================================================

def fig_seed_variability(sweep: Sweep, out_dir: Path, is_pilot: bool = True):
    """Per-seed spread — an honest look at statistical power.

    Charter kill-criterion K3: if seed variance exceeds the between-rung effect,
    the study is underpowered and NO conclusion may be drawn. This figure is how
    we check that rather than assume it.
    """
    fig, (ax_box, ax_bar) = plt.subplots(
        1, 2, figsize=(DOUBLE_COL, DOUBLE_COL * 0.36)
    )

    rungs = sweep.rungs
    data = [sweep.final_returns(r, horizon=HORIZON) for r in rungs]

    # ---- LEFT: box plot with individual points -------------------------------
    bp = ax_box.boxplot(data, tick_labels=rungs, widths=0.55, patch_artist=True,
                        medianprops=dict(color=COLORS["black"], lw=1.4))
    for patch, rung in zip(bp["boxes"], rungs):
        patch.set_facecolor(RUNG_COLORS.get(rung, COLORS["blue"]))
        patch.set_alpha(0.35)

    rng = np.random.default_rng(1)
    for i, vals in enumerate(data, start=1):
        ax_box.scatter(i + rng.uniform(-0.12, 0.12, len(vals)), vals,
                       s=20, zorder=3, color=COLORS["black"], alpha=0.6)

    ax_box.set_xlabel("Critic observability $k$")
    ax_box.set_ylabel("Final return")
    ax_box.set_title("Distribution across seeds")

    # ---- RIGHT: noise vs effect size ----------------------------------------
    summary = sweep.summary(horizon=HORIZON)
    seed_std = [summary[r]["std"] for r in rungs]
    means = [summary[r]["mean"] for r in rungs]
    effect = max(means) - min(means)

    ax_bar.bar(np.arange(len(rungs)), seed_std, width=0.6,
               color=[RUNG_COLORS.get(r, COLORS["blue"]) for r in rungs], alpha=0.75)
    ax_bar.axhline(effect, ls="--", lw=1.5, color=COLORS["red"],
                   label=f"between-rung effect ({effect:.0f})")

    ax_bar.set_xticks(np.arange(len(rungs)))
    ax_bar.set_xticklabels(rungs)
    ax_bar.set_xlabel("Critic observability $k$")
    ax_bar.set_ylabel("Seed std. dev.")
    ax_bar.set_title("Noise vs effect size")
    ax_bar.legend(fontsize=7)

    # State the power verdict on the figure itself.
    underpowered = max(seed_std) > effect
    ax_bar.text(
        0.5, -0.32,
        "seed noise EXCEEDS the effect → underpowered (K3)"
        if underpowered else
        "effect exceeds seed noise → resolvable with more seeds",
        transform=ax_bar.transAxes, ha="center", va="top", fontsize=7,
        style="italic",
        color=COLORS["red"] if underpowered else COLORS["green"],
    )

    fig.suptitle("Statistical power", y=1.02)

    if is_pilot:
        annotate_pilot_warning(fig)

    fig.tight_layout()
    return save(fig, out_dir, "fig5_seed_variability")


# ==============================================================================
# TEXT SUMMARY
# ==============================================================================

def print_statistics(sweep: Sweep) -> str:
    """Numbers behind the figures, including significance tests."""
    lines = [
        "=" * 74,
        " STATISTICAL SUMMARY",
        "=" * 74,
        "",
        f"  {'rung':<8} {'n':<4} {'mean':<10} {'std':<9} {'95% CI':<22} {'min':<8} {'max':<8}",
        "  " + "-" * 68,
    ]

    summary = sweep.summary(horizon=HORIZON)
    for rung in sweep.rungs:
        s = summary[rung]
        ci = f"[{s['ci95_low']:.1f}, {s['ci95_high']:.1f}]"
        lines.append(
            f"  {rung:<8} {s['n']:<4} {s['mean']:<10.2f} {s['std']:<9.2f} "
            f"{ci:<22} {s['min']:<8.1f} {s['max']:<8.1f}"
        )

    # ---- Interior-optimum test ---------------------------------------------
    test = interior_optimum_test(sweep, horizon=HORIZON)
    lines += ["", "-" * 74, " CLAIM 1 TEST: does an interior rung beat BOTH endpoints?", "-" * 74]

    if "error" in test:
        lines.append(f"  {test['error']}")
    else:
        best = test["best_interior_rung"]
        lines.append(f"  best interior rung : {best}")
        lines.append(f"  beats both ends    : {'YES' if test['beats_both_endpoints'] else 'NO'}")
        lines.append("")

        for key, label in [("vs_low_endpoint", "vs decentralized end"),
                           ("vs_high_endpoint", "vs maximal end")]:
            c = test[key]
            if "error" in c:
                continue
            stars = significance_stars(c["p_value"])
            lines.append(
                f"  {label:<22} {c['rung_a']} − {c['rung_b']} = {c['difference']:+7.2f}   "
                f"p = {c['p_value']:.3f} {stars:<5} d = {c['cohens_d']:+.2f}"
            )

        lines += ["", "  INTERPRETATION"]
        if test["beats_both_endpoints"] and test["both_significant"]:
            lines.append("    Interior optimum, significant against both endpoints.")
            lines.append("    Consistent with Claim 1.")
        elif test["beats_both_endpoints"]:
            lines.append("    Interior optimum in the MEANS, but confidence intervals")
            lines.append("    overlap at least one endpoint. Suggestive, not established.")
            lines.append("    This is what more seeds are for.")
        else:
            lines.append("    No interior optimum in the means. Check the shape before")
            lines.append("    concluding -- see Charter §3 kill criteria.")

    # ---- Power check --------------------------------------------------------
    seed_std = [summary[r]["std"] for r in sweep.rungs]
    means = [summary[r]["mean"] for r in sweep.rungs]
    effect = max(means) - min(means)

    lines += [
        "",
        "-" * 74,
        " STATISTICAL POWER (kill-criterion K3)",
        "-" * 74,
        f"  between-rung effect : {effect:.2f}",
        f"  largest seed std    : {max(seed_std):.2f}",
        "",
    ]
    if max(seed_std) > effect:
        lines += [
            "  Seed noise EXCEEDS the between-rung effect.",
            "  More seeds are required before any claim can be made (K3).",
        ]
    else:
        lines.append("  Effect exceeds seed noise — resolvable with more seeds.")

    lines += ["", "=" * 74]
    return "\n".join(lines)


# ==============================================================================
# MAIN
# ==============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plot publication-quality figures from a sweep result directory.",
    )
    parser.add_argument("results_dir", type=str,
                        help="Path to a result directory (e.g. results/learnability_v4/2026-09-16_v4-correctness)")
    parser.add_argument("--out", type=str, default=None,
                        help="Output directory (default: figures/output/<result dir name>)")
    parser.add_argument("--only", type=str, default=None,
                        choices=["performance", "curves", "mechanism", "health", "seeds"],
                        help="Draw only one figure")
    parser.add_argument("--horizon", type=int, default=None,
                        help="Evaluate every run at this many env steps instead of "
                             "the end of training. Use when late-training collapses "
                             "distort the final-3 estimate; report it in captions.")
    parser.add_argument("--not-pilot", action="store_true",
                        help="Omit the 'PILOT DATA' stamp (use for final ≥8-seed sweeps)")
    args = parser.parse_args()

    global HORIZON
    HORIZON = args.horizon

    apply_style()

    results_dir = Path(args.results_dir)
    sweep = load_sweep(results_dir)

    n_collapsed = sweep.num_collapsed()
    if n_collapsed:
        print()
        print(f"  NOTE: {n_collapsed}/{len(sweep.runs)} runs COLLAPSED late in training")
        print("        (ended far below their own best -- critic destabilised).")
        if HORIZON is None:
            print("        Their final-3 estimate is distorted. Consider --horizon 300000.")
        else:
            print(f"        Evaluating all runs at {HORIZON:,} steps instead.")

    out_dir = Path(args.out) if args.out else PROJECT_ROOT / "figures" / "output" / results_dir.name
    is_pilot = not args.not_pilot

    print()
    figures = {
        "performance": fig_performance_vs_k,
        "curves": fig_learning_curves,
        "mechanism": fig_mechanism,
        "health": fig_task_health,
        "seeds": fig_seed_variability,
    }

    to_draw = {args.only: figures[args.only]} if args.only else figures

    for name, fn in to_draw.items():
        try:
            paths = fn(sweep, out_dir, is_pilot)
            print(f"[saved] {name:<12} -> {paths[0].name}, {paths[1].name}")
        except Exception as exc:                       # noqa: BLE001
            # One failed figure must not prevent the others being drawn.
            print(f"[FAILED] {name}: {exc}")

    # ---- Statistics ---------------------------------------------------------
    stats_text = print_statistics(sweep)
    print()
    print(stats_text)

    (out_dir / "statistics.txt").write_text(stats_text, encoding="utf-8")
    print(f"\n  figures -> {out_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
