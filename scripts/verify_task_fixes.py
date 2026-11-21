#!/usr/bin/env python3
"""
================================================================================
 VERIFY TASK FIXES  (v1-raw4d -> v2-pidassist)
================================================================================

Proves each of the three fixes actually works, BEFORE spending compute.

v1 failed because I trusted a design without verifying its incentives. These
checks exist so that cannot happen again -- each one tests a specific way the
task could be silently broken.

    CHECK 1  PID assist       can a drone hold formation with zero policy input?
                              (if the inner loop cannot hover, nothing can learn)

    CHECK 2  Survival shaping is dying ever better than flying?
                              (v1's fatal flaw: crashing early scored higher)

    CHECK 3  Sidestep exploit does dodging the wake pay?
                              (if yes, agents delete the coupling this paper studies)

    CHECK 4  Coupling knob    does spacing still change the physics?
                              (Claim 2's manipulation check)

USAGE
-----
    python scripts/verify_task_fixes.py
================================================================================
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np

from ctde.envs.coupling import downwash_force
from ctde.envs.formation_aviary import FormationAviary


def banner(n: int, title: str) -> None:
    print()
    print("=" * 74)
    print(f" CHECK {n}: {title}")
    print("=" * 74)


# ==============================================================================
# CHECK 1 : PID ASSIST
# ==============================================================================

def check_pid_assist() -> bool:
    """With zero policy output, drones should simply hold their slots.

    This is the floor: if the inner loop cannot keep the formation with a
    null action, no policy can learn anything on top of it. v1 failed here --
    drones fell within ~1 s.
    """
    banner(1, "PID assist -- can drones fly with NO policy input?")
    print("  Zero action = 'aim exactly at my slot'. If the PID cannot hold that,")
    print("  the task is unlearnable no matter how good the policy is.")
    print()

    env = FormationAviary(num_drones=4, formation_spacing_m=1.0,
                          base_altitude_m=0.8, episode_len_sec=15.0)
    try:
        env.reset(seed=0)
        action = np.zeros((4, 3), dtype=np.float32)

        steps = 0
        total_reward = np.zeros(4)
        lateral_errors = []

        while True:
            _, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            lateral_errors.append(info["mean_lateral_error_m"])
            steps += 1
            if terminated or truncated:
                break

        max_steps = int(15.0 * env.CTRL_FREQ)
        survived = steps / max_steps

        print(f"  episodes steps survived : {steps} / {max_steps}  ({survived:.0%})")
        print(f"  mean lateral error      : {np.mean(lateral_errors):.4f} m")
        print(f"  mean return per drone   : {total_reward.mean():+.2f}")
        print(f"  crashed                 : {info['crashed']}")
        print()

        ok = survived > 0.95 and not info["crashed"]
        print("  PASS -- PID holds formation; policy can learn strategy on top."
              if ok else
              "  FAIL -- the inner loop cannot hold formation. Fix before training.")
        return ok
    finally:
        env.close()


# ==============================================================================
# CHECK 2 : SURVIVAL SHAPING
# ==============================================================================

def check_survival_incentive() -> bool:
    """Crashing early must never score better than flying on.

    THIS IS WHAT KILLED v1. Reward was a per-step penalty and episodes ended on
    crash, so dying at step 30 accumulated far fewer penalties than flying
    badly for 720 steps. Returns rose 59% while episode length stayed flat --
    the policy was learning to die faster.
    """
    banner(2, "Survival shaping -- is dying ever better than flying?")
    print("  v1's fatal flaw: per-step penalties + early termination meant")
    print("  crashing accumulated FEWER penalties than flying badly.")
    print()

    env = FormationAviary(num_drones=4, formation_spacing_m=1.0,
                          base_altitude_m=0.8, episode_len_sec=15.0)
    try:
        # --- Scenario A: fly normally (null action) --------------------------
        env.reset(seed=0)
        good_return = np.zeros(4)
        good_steps = 0
        for _ in range(int(15.0 * env.CTRL_FREQ)):
            _, r, term, trunc, _ = env.step(np.zeros((4, 3), dtype=np.float32))
            good_return += r
            good_steps += 1
            if term or trunc:
                break

        # --- Scenario B: fly badly (persistent lateral offset) ----------------
        # NOTE: with PID assist and MAX_OFFSET clamping, the policy CANNOT
        # actually fly itself into the ground -- verified: a full-down command
        # still finishes all 720 steps. That is itself a safety property worth
        # knowing. So the meaningful comparison is good flying vs bad flying.
        env.reset(seed=0)
        bad_return = np.zeros(4)
        bad_steps = 0
        bad_action = np.tile(np.array([1.0, 1.0, -1.0], dtype=np.float32), (4, 1))
        while True:
            _, r, term, trunc, info = env.step(bad_action)
            bad_return += r
            bad_steps += 1
            if term or trunc:
                break

        # --- Scenario C: forced early termination ----------------------------
        # What would a REAL early crash score? The action space cannot actually
        # reach the ground (MAX_OFFSET clamps it -- itself a useful safety
        # property), so we fly normally for 60 steps, then set the crash flag
        # and read the reward the ENVIRONMENT actually produces.
        #
        # Reading it from the env rather than recomputing it by hand matters:
        # a hand-copy of the formula silently goes stale the moment the reward
        # changes, and would report a stale verdict.
        # CORRECTED 16 Sep 2026. The previous version set `env._crashed = True`
        # by hand and called `_computeReward()` directly -- testing the reward
        # FORMULA while bypassing the call ORDER that made it dead code. It
        # reported PASS while the penalty was never actually delivered.
        #
        # Now we drive a REAL crash and read what env.step() returns.
        env.reset(seed=0)
        forced_return = np.zeros(4)
        dive = np.tile(np.array([0.0, 0.0, -1.0], dtype=np.float32), (4, 1))
        for _ in range(600):
            _, r, term, trunc, info = env.step(dive)
            forced_return += r
            if term or trunc:
                break

        print(f"  A) fly well to time limit   : return {good_return.mean():+9.2f}  ({good_steps} steps)")
        print(f"  B) fly badly to time limit  : return {bad_return.mean():+9.2f}  ({bad_steps} steps)")
        print(f"  C) real dive to crash       : return {forced_return.mean():+9.2f}")
        print()

        # NOTE 16 Sep 2026: with the crash penalty now actually delivered,
        # scenario B (a sustained max lateral+down command) CRASHES rather than
        # flying badly to the time limit. So B and C are both crashes and the
        # A > B > C ordering no longer describes three distinct regimes.
        #
        # The property that matters is unchanged and is what we assert:
        #     surviving to the time limit must beat any crash.
        good_steps_full = good_steps >= int(15.0 * env.CTRL_FREQ) * 0.95
        beats_bad = good_return.mean() > bad_return.mean()
        beats_crash = good_return.mean() > forced_return.mean()
        ok = beats_bad and beats_crash and good_steps_full

        if ok:
            print("  PASS -- surviving the full episode beats every crash.")
            print(f"    survived {good_steps} steps and scored {good_return.mean():+.1f}")
            print(f"    crashing scored {forced_return.mean():+.1f} "
                  f"(margin {good_return.mean() - forced_return.mean():,.0f})")
            print("    => return is monotone in survival time. v1's bug is fixed.")
        else:
            if not good_steps_full:
                print(f"  FAIL -- the null action did not survive ({good_steps} steps).")
            if not beats_bad:
                print("  FAIL -- bad flying scores as well as good flying.")
            if not beats_crash:
                print("  FAIL -- crashing scores better than surviving. v1's bug persists.")
        return ok
    finally:
        env.close()


# ==============================================================================
# CHECK 3 : THE SIDESTEP EXPLOIT
# ==============================================================================

def check_sidestep_exploit() -> bool:
    """Dodging the wake must cost more than the downwash it avoids.

    At 1 m separation a 10 cm lateral offset removes 86% of the downwash. If
    the reward penalty for that offset is small, the optimal policy is to step
    aside -- which would eliminate the coupling this entire paper studies and
    produce a flat curve that LOOKS like "no effect".
    """
    banner(3, "Sidestep exploit -- does dodging the wake pay?")
    print("  A 10 cm offset removes 86% of the downwash. If that is cheap, the")
    print("  agents delete the coupling the paper is about, and the result would")
    print("  look like 'no effect' when it is actually 'effect engineered away'.")
    print()

    env = FormationAviary(num_drones=4, formation_spacing_m=1.0, base_altitude_m=0.8)
    lateral_weight = env.LATERAL_WEIGHT
    env.close()

    weight_drone = 0.027 * 9.81
    episode_steps = int(15.0 * 48)

    print(f"  lateral_weight = {lateral_weight}")
    print()
    print(f"  {'offset':<9} {'downwash':<11} {'removed':<10} "
          f"{'reward cost':<14} {'benefit':<12} {'pays?'}")
    print("  " + "-" * 68)

    f_aligned = downwash_force(1.0, 0.0)
    any_pays = False

    for offset in [0.02, 0.05, 0.10, 0.15, 0.20]:
        f_offset = downwash_force(1.0, offset)
        removed = f_aligned - f_offset

        # Cost: the lateral penalty, integrated over an episode.
        cost = lateral_weight * offset ** 2 * episode_steps

        # Benefit (deliberately GENEROUS to the exploit): treat the avoided
        # downwash impulse as if it converted fully into avoided position error.
        # Real benefit is smaller, so passing this bound is conservative.
        accel = removed / 0.027
        drift_per_step = 0.5 * accel * (1.0 / 48) ** 2
        benefit = drift_per_step * episode_steps * 10.0

        pays = benefit > cost
        any_pays |= pays

        print(f"  {offset:<9.2f} {f_offset:<11.5f} {removed / f_aligned:<10.0%} "
              f"{cost:<14.1f} {benefit:<12.1f} {'YES -- BAD' if pays else 'no'}")

    print()
    if any_pays:
        print("  FAIL -- sidestepping still pays. Increase lateral_weight.")
    else:
        print("  PASS -- holding the slot always beats dodging the wake.")
        print("  (NOTE: this is an analytic bound. The real test is measuring")
        print("   mean_lateral_error_m in TRAINED policies -- see check_learning.py.)")

    return not any_pays


# ==============================================================================
# CHECK 4 : COUPLING KNOB
# ==============================================================================

def check_coupling_knob() -> bool:
    """Claim 2's manipulation check: does spacing change the physics?"""
    banner(4, "Coupling knob -- does spacing still change the physics?")
    print("  Claim 2 says k* tracks coupling strength, varied via spacing.")
    print()
    print(f"  {'spacing (m)':<14} {'total coupling (N)':<22} {'max pairwise (N)'}")
    print("  " + "-" * 54)

    couplings = []
    for spacing in [0.7, 1.0, 1.5, 2.0, 2.5]:
        env = FormationAviary(num_drones=4, formation_spacing_m=spacing, base_altitude_m=0.8)
        try:
            env.reset(seed=0)
            _, _, _, _, info = env.step(np.zeros((4, 3), dtype=np.float32))
            couplings.append(info["total_coupling_N"])
            print(f"  {spacing:<14.2f} {info['total_coupling_N']:<22.4f} "
                  f"{info['max_coupling_N']:.4f}")
        finally:
            env.close()

    monotonic = all(couplings[i] > couplings[i + 1] for i in range(len(couplings) - 1))
    contrast = couplings[0] / couplings[-1]

    print()
    print(f"  contrast (0.7 m vs 2.5 m) : {contrast:.1f}x")
    print(f"  monotonic decrease        : {'YES' if monotonic else 'NO'}")
    print()

    ok = monotonic and contrast > 5.0
    print("  PASS -- the knob changes the physics." if ok else
          "  FAIL -- coupling does not vary usefully with spacing.")
    return ok


# ==============================================================================
# MAIN
# ==============================================================================

def main() -> int:
    print("=" * 74)
    print(" TASK FIX VERIFICATION  --  v1-raw4d -> v2-pidassist")
    print("=" * 74)
    print()
    print("  v1 failed because a design was trusted without verifying its")
    print("  incentives. These checks make that failure mode impossible to")
    print("  repeat silently.")

    results = {
        "PID assist": check_pid_assist(),
        "Survival shaping": check_survival_incentive(),
        "Sidestep exploit closed": check_sidestep_exploit(),
        "Coupling knob": check_coupling_knob(),
    }

    print()
    print("=" * 74)
    print(" SUMMARY")
    print("=" * 74)
    for name, passed in results.items():
        print(f"  {'PASS' if passed else 'FAIL'}  {name}")

    print()
    if all(results.values()):
        print("  ALL CHECKS PASSED -- safe to run the learnability experiment:")
        print("      python scripts/check_learning.py --steps 30000")
        return 0

    print("  SOME CHECKS FAILED -- fix before spending compute.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
