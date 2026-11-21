#!/usr/bin/env python3
"""
================================================================================
 PHASE 2 INSPECTION  --  see the experimental apparatus
================================================================================

Prints, in one place, everything that defines the experiment's independent
variable and its confound controls:

  1. The observability ladder    -- what the critic sees at each rung
  2. Capacity control            -- proof the trunk is identical across rungs
  3. Coupling ranking            -- why "nearest" means aerodynamic influence
  4. Claim 2 manipulation check  -- proof the spacing knob changes the physics

USAGE
-----
    python scripts/inspect_ladder.py
    python scripts/inspect_ladder.py --num-drones 6
    python scripts/inspect_ladder.py --skip-env      # skip the slow parts

Run this before a sweep to confirm the apparatus is what you think it is.
================================================================================
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import argparse

import numpy as np

from ctde.envs.coupling import downwash_force
from ctde.models.networks import Actor, verify_capacity_control
from ctde.models.observability import build_ladder, critic_input_dim, describe_ladder


def banner(title: str) -> None:
    print()
    print("=" * 74)
    print(f" {title}")
    print("=" * 74)


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect the Phase 2 apparatus.")
    parser.add_argument("--num-drones", type=int, default=4)
    parser.add_argument("--skip-env", action="store_true",
                        help="Skip simulator-dependent checks (faster).")
    args = parser.parse_args()

    n = args.num_drones

    # ---- 1. The ladder ------------------------------------------------------
    banner("1. OBSERVABILITY LADDER  (the independent variable)")
    print(describe_ladder(n))

    # ---- 2. Capacity control ------------------------------------------------
    banner("2. CAPACITY CONTROL  (confound control 2 -- Charter §4.3)")
    print("The trunk must be IDENTICAL across rungs. Otherwise higher k means")
    print("more capacity, and we would measure capacity rather than information")
    print("-- the most likely reviewer objection to the whole paper.")
    print()
    dims = [critic_input_dim(lv) for lv in build_ladder(n)]
    print(verify_capacity_control(dims))

    print()
    actor_params = sum(p.numel() for p in Actor().parameters())
    print(f"  Actor parameters: {actor_params:,} -- IDENTICAL at every rung")
    print("  (confound control 1: only the critic's input set varies with k)")

    # ---- 3. Coupling ranking ------------------------------------------------
    banner("3. NEIGHBOUR RANKING  (why 'nearest' != Euclidean)")
    print("Phase 1 measured that downwash depends on ALIGNMENT, not just distance.")
    print()
    print(f"  {'configuration':<44} {'force (N)':<12}")
    print("  " + "-" * 58)
    cases = [
        ("0.6 m above, 0.00 m offset (near, aligned)", 0.6, 0.00),
        ("0.6 m above, 0.10 m offset (near, offset)", 0.6, 0.10),
        ("1.5 m above, 0.00 m offset (far,  aligned)", 1.5, 0.00),
        ("0.5 m BELOW (wake flows downward)", -0.5, 0.00),
    ]
    for label, dz, dxy in cases:
        print(f"  {label:<44} {downwash_force(dz, dxy):<12.5f}")

    print()
    print("  => A drone 0.6 m above but 10 cm off-axis exerts ~0 N, while one")
    print("     1.5 m above and aligned exerts real force. Euclidean ranking")
    print("     would put an irrelevant teammate at k=1 and corrupt the ladder.")

    if args.skip_env:
        banner("DONE  (environment checks skipped)")
        return 0

    # ---- 4. Claim 2 manipulation check --------------------------------------
    banner("4. CLAIM 2 MANIPULATION CHECK  (does spacing change the physics?)")
    print("Claim 2 says k* tracks coupling strength, varied via formation spacing.")
    print("This verifies the knob changes the PHYSICS, not just the geometry.")
    print()

    from ctde.envs.formation_aviary import FormationAviary

    print(f"  {'spacing (m)':<14} {'total coupling (N)':<22} {'max pairwise (N)':<18}")
    print("  " + "-" * 54)

    couplings = []
    for spacing in [0.7, 1.0, 1.5, 2.0, 2.5]:
        env = FormationAviary(
            num_drones=n,
            formation_spacing_m=spacing,
            base_altitude_m=0.8,
        )
        try:
            env.reset(seed=0)
            _, _, _, _, info = env.step(np.zeros((n, 4), dtype=np.float32))
            couplings.append(info["total_coupling_N"])
            print(f"  {spacing:<14.2f} {info['total_coupling_N']:<22.4f} "
                  f"{info['max_coupling_N']:<18.4f}")
        finally:
            env.close()

    print()
    print(f"  Coupling contrast (0.7 m vs 2.5 m): {couplings[0] / couplings[-1]:.1f}x")
    monotonic = all(couplings[i] > couplings[i + 1] for i in range(len(couplings) - 1))
    print(f"  Monotonic decrease: {'YES' if monotonic else 'NO'}")

    banner("DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
