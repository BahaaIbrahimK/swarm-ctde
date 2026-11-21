"""
================================================================================
 BASELINE TESTS  --  the bar a learned policy must clear
================================================================================

These run the REAL environment, not a mock. The P-controller's advantage over
the pure PID is a load-bearing claim: it is what tells us the task is solvable
when a learned policy fails to improve. If a future change to the reward, the
observation, or the physics silently destroys that advantage, we need to find
out here rather than by misreading a flat ladder as kill-criterion K1.
================================================================================
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import pytest

from ctde.baselines import p_control_baseline, pid_baseline
from ctde.envs.formation_aviary import FormationAviary
from ctde.metrics import EpisodeRecorder


def _configured_gust_std() -> float:
    """The gust strength the experiment actually uses (single source of truth)."""
    from ctde.models.observability import build_ladder
    from ctde.training.trainer import RunConfig

    return RunConfig(level=build_ladder(4)[0]).gust_std_ms


def _roll(action_fn, seeds, gust_std=None):
    """Run episodes through the REAL env and return (mean return, mean RMS, crashes).

    `gust_std` defaults to the CONFIGURED value. It used to be hardcoded to 3.0,
    a level abandoned in the Sep 2026 recalibration because the PID alone
    crashes ~1 in 4 episodes there. Asserting "zero PID crashes" at sigma=3.0
    then depended on which 4 episodes were drawn, and on floating-point
    details of the machine: it passed on the original WSL machine and failed
    on the GitHub Actions runner. Same lesson as HANDOFF.md section A: a
    recalibration that never reached every caller.
    """
    if gust_std is None:
        gust_std = _configured_gust_std()
    returns, rms, crashes = [], [], 0
    for seed in seeds:
        env = FormationAviary(
            num_drones=4, formation_spacing_m=1.0, episode_len_sec=15.0,
            ctrl_freq=48, wind_mean_ms=1.0, gust_std_ms=gust_std,
            gust_tau_sec=1.0, gust_spatial_corr=0.7, chain_topology=True,
        )
        try:
            obs, _ = env.reset(seed=seed)
            rec = EpisodeRecorder(4, 720)
            total = np.zeros(4)
            info: dict = {}
            while True:
                action = action_fn(obs)
                obs, reward, term, trunc, info = env.step(action)
                rec.record(env, action, info)
                total += reward
                if term or trunc:
                    break
            if info.get("crashed"):
                crashes += 1
            returns.append(float(total.mean()))
            rms.append(rec.finalise().rms_position_error_m)
        finally:
            env.close()
    return float(np.mean(returns)), float(np.mean(rms)), crashes


# ==============================================================================

def test_p_control_shape_and_bounds():
    """Actions must be bounded and oppose the observed gap error."""
    obs = np.zeros((4, 13), dtype=np.float32)
    obs[:, 0:3] = 0.5                      # large positive gap error

    action = p_control_baseline(obs)

    assert action.shape == (4, 3)
    assert np.all(action >= -1.0) and np.all(action <= 1.0)
    # Error is positive, so the correction must be negative.
    assert np.all(action < 0), "P-control must push AGAINST the gap error"


def test_pid_baseline_is_exactly_zero():
    """The 'do nothing' reference must command nothing at all."""
    obs = np.random.default_rng(0).normal(size=(4, 13)).astype(np.float32)
    assert np.array_equal(pid_baseline(obs), np.zeros((4, 3), dtype=np.float32))


@pytest.mark.slow
def test_feedforward_pid_is_a_strong_baseline():
    """The PID WITH velocity feedforward must beat adding P gain on top.

    REPLACES `test_p_control_beats_pure_pid`, which asserted the opposite and
    was true only because the PID was handicapped.

    The inner controller was called with target_pos only, so its desired
    velocity defaulted to ZERO while the leader tracked a moving circle and each
    follower tracked a moving predecessor. A one-parameter P-controller then
    beat it by +109 return / 28% lower RMS, and that gap was read as evidence
    that RL had an exploration problem.

    With `target_vel` passed, on common episodes:

        PID + feedforward     614.7 return   12.16 cm   0/8 crashes
        P-control on top     -561.8 return   11.60 cm   8/8 crashes

    and a gain sweep puts the best stable gain at ZERO. The P-controller was
    substituting proportional gain for the velocity feedback the controller
    should have had.

    This test exists so the feedforward cannot be silently dropped again: if it
    is, the P-controller starts winning and this fails.
    """
    seeds = [900_000 + i for i in range(4)]

    pid_return, pid_rms, pid_crashes = _roll(pid_baseline, seeds)
    p_return, _, p_crashes = _roll(p_control_baseline, seeds)

    assert pid_crashes == 0, (
        f"the feedforward PID crashed {pid_crashes}/{len(seeds)} -- it is the "
        f"baseline everything is measured against and must be stable"
    )
    assert pid_return > p_return, (
        f"PID+feedforward ({pid_return:.1f}) should beat P-control on top "
        f"({p_return:.1f}); if not, the feedforward term was probably lost"
    )
    # Concrete floor, so a regression in the controller is visible as a number.
    assert pid_rms < 0.16, f"PID RMS {pid_rms*100:.2f} cm is worse than expected"


@pytest.mark.slow
def test_gust_calibration_leaves_the_pid_surviving():
    """The configured gust must be survivable by the PID ALONE.

    Checked on the FULL common evaluation block, not a short prefix. That
    distinction is the whole point: sigma=3.0 looked crash-free on 8 episodes,
    but all six of its crashes sit in episodes 8-23. Calibrating on 12 episodes
    is how sigma=3.0 was adopted in the first place.
    """
    from ctde.models.observability import build_ladder
    from ctde.training.trainer import RunConfig

    cfg = RunConfig(level=build_ladder(4)[0])
    seeds = cfg.final_eval_seeds()

    _, _, crashes = _roll(pid_baseline, seeds, gust_std=cfg.gust_std_ms)

    assert crashes == 0, (
        f"the pure PID crashes {crashes}/{len(seeds)} at the configured "
        f"gust_std_ms={cfg.gust_std_ms}. Crash variance is dice, not skill; "
        f"recalibrate before training anything."
    )
