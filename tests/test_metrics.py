"""
================================================================================
 TESTS: physical metrics and downstream error amplification
================================================================================

Charter §S.11. These metrics are what the paper reports -- centimetres, rates,
and amplification ratios -- so their arithmetic must be pinned down.
================================================================================
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import pytest

from ctde.metrics import (
    EpisodeRecorder,
    compute_error_amplification,
    summarise_amplification,
)


def test_amplification_detects_growth():
    """G_i > 1 when errors grow downstream."""
    # leader 0.01, then doubling each link
    g = compute_error_amplification([0.01, 0.02, 0.04, 0.08])
    assert len(g) == 3
    assert all(abs(x - 2.0) < 1e-6 for x in g)


def test_amplification_detects_attenuation():
    """G_i < 1 when errors shrink downstream."""
    g = compute_error_amplification([0.08, 0.04, 0.02, 0.01])
    assert all(x < 1.0 for x in g)


def test_amplification_handles_zero_denominator():
    """A perfect upstream drone must not produce a NaN that poisons summaries."""
    g = compute_error_amplification([0.0, 0.01])
    assert np.isinf(g[0])


def test_summarise_reports_worst_link():
    """The headline is the WORST link, not the average.

    A chain is only as stable as its weakest link; averaging would hide the
    failure mode that matters.
    """
    s = summarise_amplification([[1.0, 3.0], [1.0, 1.0]])
    assert s["worst_link"] == pytest.approx(3.0)
    assert s["fraction_amplifying"] == pytest.approx(0.25)  # 1 of 4 entries > 1


def test_summarise_ignores_infinities():
    """Infinite ratios must not make the whole summary NaN."""
    s = summarise_amplification([[np.inf, 2.0], [1.0, 1.5]])
    assert np.isfinite(s["mean_amplification"])
    assert s["worst_link"] == pytest.approx(2.0)


@pytest.mark.slow
def test_recorder_end_to_end():
    """The recorder must work against the real environment's public interface."""
    from ctde.envs.formation_aviary import FormationAviary

    env = FormationAviary(
        num_drones=4, formation_spacing_m=1.0, base_altitude_m=0.8,
        episode_len_sec=3.0, gust_std_ms=2.0,
    )
    try:
        max_steps = int(3.0 * env.CTRL_FREQ)
        rec = EpisodeRecorder(num_agents=4, max_steps=max_steps)

        env.reset(seed=0)
        action = np.zeros((4, 3), dtype=np.float32)
        while True:
            _, _, term, trunc, info = env.step(action)
            rec.record(env, action, info)
            if term or trunc:
                break

        m = rec.finalise()

        # Physical sanity: a PID holding formation should be within ~tens of cm.
        assert 0.0 < m.rms_position_error_m < 1.0, m.rms_position_error_m
        assert m.min_separation_m > env.COLLISION_RADIUS_M
        assert not m.collision
        assert len(m.per_drone_rms_error_m) == 4
        assert len(m.error_amplification) == 3
        assert m.episode_length_steps > 0
    finally:
        env.close()
