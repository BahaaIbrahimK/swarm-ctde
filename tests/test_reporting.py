"""
================================================================================
 REPORTING TESTS  --  evaluation protocol and result aggregation
================================================================================

These guard four defects found in the second review (Sep 2026). None of them
breaks physics; all of them corrupt what gets REPORTED, which for a paper is
just as damaging.

Same rule as tests/test_wiring.py: instantiate the REAL objects.
================================================================================
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import pytest

from ctde.models.observability import build_ladder
from ctde.training.mappo import MAPPOConfig
from ctde.training.sweep import aggregate_by_rung
from ctde.training.trainer import RunConfig


# ==============================================================================
# SECTION 1 : AGGREGATION MUST NOT POOL SPACINGS
# ==============================================================================

def test_aggregate_does_not_pool_across_spacings():
    """CLAIM 2 KILLER: spacings must never be averaged together.

    Claim 2 asserts that k* TRACKS coupling strength, which we vary via
    formation spacing. If the aggregator silently averages a rung's results
    across spacings, the very signal Claim 2 depends on is destroyed -- and
    nothing would error, so the corruption would be invisible.
    """
    results = [
        {"rung": "k1", "formation_spacing_m": 0.7, "final_eval_return": 100.0},
        {"rung": "k1", "formation_spacing_m": 0.7, "final_eval_return": 110.0},
        {"rung": "k1", "formation_spacing_m": 2.5, "final_eval_return": 500.0},
        {"rung": "k1", "formation_spacing_m": 2.5, "final_eval_return": 510.0},
    ]

    tight = aggregate_by_rung(results, spacing=0.7)
    wide = aggregate_by_rung(results, spacing=2.5)

    assert tight["k1"]["mean"] == pytest.approx(105.0)
    assert wide["k1"]["mean"] == pytest.approx(505.0)
    assert tight["k1"]["n_seeds"] == 2


def test_aggregate_refuses_to_silently_pool_mixed_spacings():
    """Aggregating mixed spacings without saying which must FAIL LOUDLY.

    A silent average of 105 and 505 is worse than an error: it looks like a
    result.
    """
    results = [
        {"rung": "k1", "formation_spacing_m": 0.7, "final_eval_return": 100.0},
        {"rung": "k1", "formation_spacing_m": 2.5, "final_eval_return": 500.0},
    ]
    with pytest.raises(ValueError, match="multiple spacings"):
        aggregate_by_rung(results)


def test_aggregate_allows_single_spacing_without_filter():
    """When only one spacing is present there is nothing to pool."""
    results = [
        {"rung": "k1", "formation_spacing_m": 1.0, "final_eval_return": 100.0},
        {"rung": "k1", "formation_spacing_m": 1.0, "final_eval_return": 120.0},
    ]
    out = aggregate_by_rung(results)
    assert out["k1"]["mean"] == pytest.approx(110.0)


# ==============================================================================
# SECTION 2 : HELD-OUT EVALUATION SEEDS
# ==============================================================================

def test_eval_seeds_are_disjoint_from_training_seeds():
    """Evaluation must not reuse the seeds training saw.

    The environment RNG stream is derived from `seed`; evaluating on the same
    seeds measures performance on disturbances already encountered.
    """
    cfg = RunConfig(level=build_ladder(4)[0], seed=3)
    train_seeds = {cfg.seed}
    eval_seeds = set(cfg.eval_seeds())

    assert not (train_seeds & eval_seeds), (
        f"evaluation seeds {eval_seeds} overlap the training seed {cfg.seed}"
    )


def test_final_eval_seeds_are_held_out_from_periodic_eval():
    """The FINAL evaluation must use seeds never used for checkpoint selection.

    Periodic evaluation guides "is it learning?"; reusing those same seeds for
    the reported number would be selecting on the test set.
    """
    cfg = RunConfig(level=build_ladder(4)[0], seed=0)
    periodic = set(cfg.eval_seeds())
    final = set(cfg.final_eval_seeds())

    assert not (periodic & final), (
        f"final eval seeds {final} overlap periodic eval seeds {periodic}"
    )
    assert len(final) >= 5, "final evaluation needs enough episodes to be stable"


def test_eval_seeds_differ_across_training_seeds():
    """Two runs with different training seeds must see different evaluations.

    Otherwise every run in a sweep is scored on an identical fixed set, which
    understates the variance we report.
    """
    a = set(RunConfig(level=build_ladder(4)[0], seed=0).eval_seeds())
    b = set(RunConfig(level=build_ladder(4)[0], seed=1).eval_seeds())
    assert a != b


def test_seed_blocks_do_not_overflow_into_each_other():
    """Episode counts must stay inside their 1000-wide blocks.

    Seeds are laid out as `seed*1000 + offset`. If `final_eval_episodes` ever
    exceeded 1000, run N's block would run into run N+1's and the independence
    the whole variance analysis assumes would be silently gone.
    """
    cfg = RunConfig(level=build_ladder(4)[0], seed=0)
    assert cfg.eval_episodes < 1000
    assert cfg.final_eval_episodes < 1000

    # PERIODIC blocks are per-run and must stay disjoint: they judge each run's
    # health independently.
    #
    # FINAL episodes are deliberately SHARED by every run (common random
    # numbers, 17 Sep 2026), so they are excluded from this check -- see
    # RunConfig.final_eval_seeds and tests/test_review_fixes.py.
    seen: set[int] = set()
    for seed in range(12):
        c = RunConfig(level=build_ladder(4)[0], seed=seed)
        block = set(c.eval_seeds())
        assert not (block & seen), f"seed {seed}'s periodic block overlaps another run"
        seen |= block

    # And no run may be scored on an episode it monitored with.
    assert not (seen & set(cfg.final_eval_seeds()))


def test_enough_eval_episodes_to_resolve_the_baseline_effect():
    """Evaluation must be able to resolve the effects we intend to report.

    MEASURED 16 Sep 2026: a single early crash costs ~-650 return, so one crash
    in a 3-episode block moves the mean by ~350 -- more than three times the
    P-control-vs-PID effect of +109. Three episodes could not tell a good policy
    from a lucky draw, and one seed was nearly misdiagnosed as 'not learning'
    when its eval block simply contained an early crash.

    Power analysis on the measured spreads (d = 1.42 on return, 3.07 on RMS)
    put the requirement at n >= 8.
    """
    cfg = RunConfig(level=build_ladder(4)[0], seed=0)
    assert cfg.eval_episodes >= 8, (
        f"eval_episodes={cfg.eval_episodes} cannot resolve the effects this "
        f"study reports; a single crash would dominate the mean"
    )
    assert cfg.final_eval_episodes >= cfg.eval_episodes, (
        "the FINAL reported evaluation must be at least as precise as the "
        "periodic one it supersedes"
    )


def test_no_script_silently_downgrades_the_eval_protocol():
    """Scripts must not hardcode a weaker `eval_episodes` than the calibrated one.

    THIS EXACT BUG HAPPENED (16 Sep 2026). RunConfig's default was raised 3 -> 8
    after measuring that 3 episodes cannot resolve the effects we report. The
    learnability script still passed `eval_episodes=3` explicitly, so the
    "corrected" run produced numbers BYTE-IDENTICAL to the broken one. Two of
    the sweep scripts had the same override, which would have reverted the
    protocol for the real pilots.

    Same shape as the padded-critic and dead-metrics defects: the fix was
    correct and never reached the code that runs.

    Any script needing a smaller count must name itself in ALLOWED below, with a
    reason. An explicit allow-list beats scanning nearby comments for
    justification words -- a heuristic like that passes whenever an unrelated
    comment happens to sit above the offending line, which is exactly how the
    first version of this test managed to pass while the bug was present.
    """
    import re

    # script name -> why a smaller count is scientifically acceptable there
    ALLOWED = {
        # Stage 1 asks "where does the learning curve flatten?". A noisy but
        # frequent estimate answers that; its ENDPOINT is never quoted.
        "run_overnight.py",
        # --smoke exists to prove the plumbing runs, not to measure anything.
        "run_pilot_sweep.py",
    }

    scripts = Path(__file__).resolve().parent.parent / "scripts"
    calibrated = RunConfig(level=build_ladder(4)[0]).eval_episodes

    offenders = []
    for path in sorted(scripts.glob("*.py")):
        if path.name in ALLOWED:
            continue
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            if line.lstrip().startswith("#"):
                continue
            m = re.search(r"\beval_episodes\s*=\s*(\d+)", line)
            if m and int(m.group(1)) < calibrated:
                offenders.append(f"{path.name}:{lineno}  {line.strip()}")

    assert not offenders, (
        f"scripts hardcode an eval_episodes below the calibrated {calibrated}:\n  "
        + "\n  ".join(offenders)
        + "\n\nEither remove the override so it inherits RunConfig's default, "
          "or add the script to ALLOWED in this test with a reason."
    )


# ==============================================================================
# SECTION 3 : EXACT SAMPLE BUDGET
# ==============================================================================

def test_training_does_not_overshoot_the_budget():
    """Charter §7 requires an IDENTICAL budget across rungs.

    The loop collected whole rollouts, so a budget not divisible by
    `num_steps` overshot: 100,000 became 100,096. Small, but it makes the
    stated budget untrue and varies with the rollout length.
    """
    from ctde.training.trainer import Trainer

    cfg = RunConfig(
        level=build_ladder(4)[0],
        total_env_steps=500,                 # not divisible by 200
        eval_every_steps=500, eval_episodes=1, episode_len_sec=1.0,
        mappo=MAPPOConfig(num_steps=200, num_epochs=1, minibatch_size=100),
    )
    trainer = Trainer(cfg)
    try:
        trainer.train(verbose=False)
        assert trainer.global_step == cfg.total_env_steps, (
            f"collected {trainer.global_step} steps, budget was {cfg.total_env_steps}"
        )
    finally:
        trainer.close()


# ==============================================================================
# SECTION 4 : PHYSICAL METRICS ARE ACTUALLY RECORDED
# ==============================================================================

@pytest.mark.slow
def test_trainer_records_physical_metrics():
    """`metrics.py` must be WIRED, not merely present.

    It was written, tested in isolation, and imported nowhere -- the same
    dead-code pattern as the padded critic input. Results still reported
    unitless return instead of the centimetres and rates the charter (§S.11)
    says are the headline.
    """
    from ctde.training.trainer import Trainer

    cfg = RunConfig(
        level=build_ladder(4)[1],
        total_env_steps=400, eval_every_steps=400, eval_episodes=2,
        episode_len_sec=1.0,
        mappo=MAPPOConfig(num_steps=200, num_epochs=1, minibatch_size=100),
    )
    trainer = Trainer(cfg)
    try:
        history = trainer.train(verbose=False)
        record = history[-1]

        for key in [
            "eval_rms_position_error_m",
            "eval_max_position_error_m",
            "eval_min_separation_m",
            "eval_collision_rate",
            "eval_worst_error_amplification",
        ]:
            assert key in record, f"physical metric '{key}' missing from history"
            assert np.isfinite(record[key]), f"'{key}' is not finite"

        # Physical sanity, in metres.
        assert 0.0 < record["eval_rms_position_error_m"] < 3.0
        assert record["eval_min_separation_m"] > 0.0
    finally:
        trainer.close()
