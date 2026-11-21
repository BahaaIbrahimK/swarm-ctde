#!/usr/bin/env python3
"""
================================================================================
 ENTRY POINT: DOWNWASH PROBE
================================================================================

PHASE 1 EXIT ARTIFACT (Charter §5)

WHAT THIS SCRIPT DOES
---------------------
Runs the downwash characterisation sweep and produces three artifacts:

    1. downwash_measurements.csv          raw numbers
    2. downwash_force_vs_separation.png   the Phase 1 exit plot
    3. config.yaml                        exact settings used (provenance)

WHY IT MATTERS
--------------
This is the PREMISE CHECK for the whole paper. See the long explanation at the
top of src/ctde/physics/downwash_probe.py. In one line: if downwash coupling is
not physically real at our spacings, Claims 1 and 2 are not testable and the
project must pivot -- and we want to learn that in week 3, not month 5.

USAGE
-----
    python scripts/run_downwash_probe.py
    python scripts/run_downwash_probe.py --config configs/probe_downwash.yaml
    python scripts/run_downwash_probe.py --seed 3

HOW TO READ THE OUTPUT
----------------------
Look at the printed summary block first. It answers, in plain language:
  * Is coupling real?         (differential drift at tight spacing)
  * Where does it vanish?     (the coupling horizon)
Then open the PNG for the full picture.
================================================================================
"""

from __future__ import annotations

# ==============================================================================
# SECTION 0 : IMPORT PATH SETUP
# ==============================================================================
# This script lives in scripts/ but imports from src/ctde/. Rather than
# requiring `pip install -e .` before anything can run, we add src/ to the
# import path explicitly. This keeps the very first run friction-free.
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import argparse
import csv
from datetime import datetime

from ctde.physics.downwash_probe import (
    DownwashProbe,
    ProbeMeasurement,
    find_coupling_horizon,
    summarise,
)
from ctde.utils.config import load_config, save_config
from ctde.utils.seeding import get_git_commit, seed_everything


# ==============================================================================
# SECTION 1 : ARGUMENTS
# ==============================================================================

def parse_args() -> argparse.Namespace:
    """Command-line arguments.

    Deliberately minimal. Experiment SETTINGS belong in the YAML config
    (versioned in git, saved with results); the command line only selects
    WHICH config to run and where to put the output. See the rationale in
    src/ctde/utils/config.py.
    """
    parser = argparse.ArgumentParser(
        description="Run the downwash characterisation probe (Phase 1).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/probe_downwash.yaml",
        help="Path to the probe config (default: configs/probe_downwash.yaml)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override the seed from the config.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip plot generation (useful on headless machines without matplotlib).",
    )
    return parser.parse_args()


# ==============================================================================
# SECTION 2 : OUTPUT WRITERS
# ==============================================================================

def write_csv(measurements: list[ProbeMeasurement], out_path: Path) -> None:
    """Write raw measurements to CSV.

    The CSV is the source of truth. The plot is a rendering of it, and can
    always be regenerated -- so the numbers are saved before any plotting is
    attempted (plotting is the step most likely to fail on a headless box).
    """
    if not measurements:
        raise ValueError("No measurements to write.")

    rows = [m.as_dict() for m in measurements]

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def make_plot(measurements: list[ProbeMeasurement], out_path: Path) -> None:
    """Render the Phase 1 exit figure.

    TWO PANELS
    ----------
    LEFT   coupling vs VERTICAL separation (at horizontal offset = 0).
           This is the headline: it shows the decay of downwash with distance
           and therefore the coupling horizon that predicts k*.

    RIGHT  coupling vs HORIZONTAL offset, one line per vertical separation.
           Downwash is modelled as a Gaussian centred beneath the upper drone,
           so this should peak at 0 and fall away sideways. It tells us how
           precisely aligned drones must be to interact -- input to the
           formation geometry design in Phase 2.
    """
    import matplotlib
    matplotlib.use("Agg")          # non-interactive backend: works headless
    import matplotlib.pyplot as plt

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(13, 5))

    # ---- LEFT PANEL : vertical decay ---------------------------------------
    aligned = sorted(
        [m for m in measurements if m.horizontal_offset_m == 0.0],
        key=lambda m: m.vertical_separation_m,
    )
    if aligned:
        xs = [m.vertical_separation_m for m in aligned]
        ys = [abs(m.differential_z_drift_m) for m in aligned]

        ax_left.plot(xs, ys, marker="o", linewidth=2, color="#1f77b4")
        ax_left.set_xlabel("Vertical separation (m)")
        ax_left.set_ylabel("|differential z-drift| (m)")
        ax_left.set_title("Downwash coupling vs vertical separation\n(directly beneath)")
        ax_left.grid(alpha=0.3)

        # Mark the coupling horizon -- the number that predicts k*.
        horizon = find_coupling_horizon(measurements)
        if horizon is not None:
            ax_left.axvline(
                horizon,
                color="crimson",
                linestyle="--",
                linewidth=1.5,
                label=f"coupling horizon ~{horizon:.2f} m",
            )
            ax_left.legend()

    # ---- RIGHT PANEL : horizontal falloff ----------------------------------
    separations = sorted({m.vertical_separation_m for m in measurements})
    for v_sep in separations:
        pts = sorted(
            [m for m in measurements if m.vertical_separation_m == v_sep],
            key=lambda m: m.horizontal_offset_m,
        )
        if len(pts) < 2:
            continue        # a single point cannot show a falloff curve
        ax_right.plot(
            [m.horizontal_offset_m for m in pts],
            [abs(m.differential_z_drift_m) for m in pts],
            marker="o",
            linewidth=1.5,
            label=f"{v_sep:.2f} m",
        )

    ax_right.set_xlabel("Horizontal offset (m)")
    ax_right.set_ylabel("|differential z-drift| (m)")
    ax_right.set_title("Downwash coupling vs lateral offset\n(one line per vertical separation)")
    ax_right.grid(alpha=0.3)
    ax_right.legend(title="vertical sep.", fontsize=8)

    fig.suptitle(
        "Phase 1 premise check: is inter-agent aerodynamic coupling real?",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ==============================================================================
# SECTION 3 : MAIN
# ==============================================================================

def main() -> int:
    args = parse_args()

    # ---- Load configuration ------------------------------------------------
    config_path = PROJECT_ROOT / args.config
    config = load_config(config_path)

    if args.seed is not None:
        config["reproducibility"]["seed"] = args.seed

    seed = config["reproducibility"]["seed"]

    # ---- Seed BEFORE creating anything -------------------------------------
    # Must happen before environments exist, or their internal randomness
    # escapes our control. See src/ctde/utils/seeding.py.
    seed_everything(seed, deterministic_torch=config["reproducibility"]["deterministic_torch"])

    # ---- Prepare output directory ------------------------------------------
    # Timestamped so repeated runs never silently overwrite each other.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = (
        PROJECT_ROOT
        / config["project"]["results_root"]
        / config["output"]["subdir"]
        / f"{timestamp}_seed{seed}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save the resolved config + provenance immediately, so that even if the
    # run crashes we know exactly what was attempted.
    save_config(config, out_dir)
    (out_dir / "git_commit.txt").write_text(get_git_commit(), encoding="utf-8")

    print("=" * 74)
    print(" PHASE 1: DOWNWASH CHARACTERISATION PROBE")
    print("=" * 74)
    print(f"  config      : {config_path}")
    print(f"  seed        : {seed}")
    print(f"  physics     : {config['physics']['mode']}")
    print(f"  output dir  : {out_dir}")
    print("=" * 74)
    print()

    # ---- Run the sweep -----------------------------------------------------
    probe = DownwashProbe(config)
    measurements = probe.run_sweep(verbose=True)

    # ---- Write raw data FIRST (before plotting, which may fail) ------------
    csv_path = out_dir / config["output"]["csv_name"]
    write_csv(measurements, csv_path)
    print(f"\n[saved] {csv_path}")

    # ---- Plot ---------------------------------------------------------------
    if not args.no_plot:
        try:
            plot_path = out_dir / config["output"]["plot_name"]
            make_plot(measurements, plot_path)
            print(f"[saved] {plot_path}")
        except ImportError:
            # Not fatal: the CSV holds the data, and the plot can be
            # regenerated later from it.
            print("[warn ] matplotlib not available; skipped plot. CSV is intact.")

    # ---- Verdict ------------------------------------------------------------
    print()
    print(summarise(measurements))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
