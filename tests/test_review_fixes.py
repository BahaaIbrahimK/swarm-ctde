"""
================================================================================
 EXTERNAL-REVIEW FIXES  --  17 September 2026
================================================================================

Each test corresponds to a defect an external reviewer found and that was then
confirmed by measurement. All of them corrupt REPORTED RESULTS without crashing,
which is the class this project has been repeatedly burned by.

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

from ctde.envs.formation_aviary import FormationAviary
from ctde.models.observability import (
    build_critic_input,
    build_ladder,
    information_added_by_rung,
)
from ctde.training.mappo import MAPPOConfig
from ctde.training.trainer import FINAL_EVAL_SEED_BASE, RunConfig


# ==============================================================================
# SECTION 1 : EPISODE LENGTH IS EXACT
# ==============================================================================

@pytest.mark.parametrize("secs,freq,expected", [
    (15.0, 48, 720), (5.0, 48, 240), (10.0, 30, 300), (1.0, 48, 48),
])
def test_episode_runs_exactly_the_configured_number_of_steps(secs, freq, expected):
    """A 15 s episode at 48 Hz must be 720 steps, not 722.

    TWO bugs compounded here. Upstream `BaseAviary.step()` increments
    `step_counter` AFTER computing obs/reward/terminated/truncated, so reading
    it directly gave the time BEFORE the step being described; and the
    truncation test used a strict `>`, so the boundary step did not trigger it.

    Result: 722 steps, "episode completion" reported above 100%, and a crash
    penalty computed against a horizon the episode did not have.
    """
    env = FormationAviary(
        num_drones=4, formation_spacing_m=1.0, episode_len_sec=secs,
        ctrl_freq=freq, gust_std_ms=0.0, chain_topology=True,
    )
    try:
        env.reset(seed=7)
        steps = 0
        while True:
            _, _, term, trunc, _ = env.step(np.zeros((4, 3), dtype=np.float32))
            steps += 1
            if term or trunc:
                break
        assert steps == expected, f"ran {steps} steps, expected {expected}"
    finally:
        env.close()


def test_completion_fraction_cannot_exceed_one():
    """The metric that made the 722-step bug visible."""
    env = FormationAviary(
        num_drones=4, formation_spacing_m=1.0, episode_len_sec=5.0,
        ctrl_freq=48, gust_std_ms=0.0, chain_topology=True,
    )
    try:
        env.reset(seed=3)
        steps = 0
        while True:
            _, _, term, trunc, _ = env.step(np.zeros((4, 3), dtype=np.float32))
            steps += 1
            if term or trunc:
                break
        assert steps / (5.0 * 48) <= 1.0
    finally:
        env.close()


# ==============================================================================
# SECTION 2 : COMMON RANDOM NUMBERS
# ==============================================================================

def test_final_eval_episodes_are_identical_across_training_seeds():
    """Every policy must be scored on the SAME episodes.

    Per-run final seeds added episode difficulty to what we report as
    between-run variance. Measured: training seed 1 scored 130 where others
    scored 450-545, because its block held an episode that crashes at step 87.
    Scoring that same block with the PURE PID gives 119.7 -- the exam was
    harder, not the policy worse.
    """
    blocks = [
        RunConfig(level=build_ladder(4)[0], seed=s).final_eval_seeds()
        for s in range(5)
    ]
    for block in blocks[1:]:
        assert block == blocks[0], "final evaluation episodes differ between runs"
    assert blocks[0][0] == FINAL_EVAL_SEED_BASE


def test_final_eval_never_overlaps_training_or_periodic_seeds():
    """Common episodes must still be unseen by every run."""
    final = set(RunConfig(level=build_ladder(4)[0], seed=0).final_eval_seeds())
    for s in range(64):
        cfg = RunConfig(level=build_ladder(4)[0], seed=s)
        assert not (final & set(cfg.eval_seeds()))
        assert cfg.seed not in final


# ==============================================================================
# SECTION 3 : THE SWEEP REPORTS A REAL HELD-OUT EVALUATION
# ==============================================================================

@pytest.mark.slow
def test_sweep_reports_final_held_out_evaluation():
    """`final_*` in a sweep record must come from evaluate(final=True).

    It used to be the mean of the last three PERIODIC evaluations -- three
    DIFFERENT policies, scored on the episodes used to monitor the run. The
    held-out path existed and was called only from a diagnostic script, never
    from the sweep that produces the paper's numbers.

    Asserted on the SWEEP's output, because that is the layer that was wrong.
    """
    import json
    import tempfile

    from ctde.training.sweep import _run_one

    cfg = RunConfig(
        level=build_ladder(4)[1], seed=0,
        total_env_steps=400, eval_every_steps=200, eval_episodes=2,
        final_eval_episodes=3, episode_len_sec=1.0,
        mappo=MAPPOConfig(num_steps=200, num_epochs=1, minibatch_size=100),
    )

    with tempfile.TemporaryDirectory() as td:
        record = _run_one((cfg, td))

        assert record["final_eval_episodes"] == 3, (
            "the reported evaluation did not use the configured held-out episodes"
        )
        # Per-episode records must be saved for later paired tests.
        saved = Path(td) / "final_eval_episodes.json"
        assert saved.exists(), "per-episode final evaluation was not persisted"
        assert len(json.loads(saved.read_text())) == 3

        # Curve diagnostics must be present but clearly NOT named as results.
        assert "curve_last3_periodic_return" in record
        for key in record:
            if key.startswith("final_"):
                assert "periodic" not in key


# ==============================================================================
# SECTION 4 : GUST MAGNITUDE IS REPORTED HONESTLY
# ==============================================================================

@pytest.mark.slow
def test_logged_gust_speed_is_per_drone_not_frobenius():
    """`mean_gust_speed_ms` must be a speed, not sqrt(sum over all drones).

    The recorder took np.linalg.norm() of the whole (N,3) array, which grows
    like sqrt(N). Measured 5.689 for a field whose true mean per-drone speed
    was 2.486 -- inflated roughly 2x at N=4.
    """
    from ctde.metrics import EpisodeRecorder

    class FakeEnv:
        def get_global_state(self):
            # Four drones, each with speed exactly 3.0
            return {
                "positions": np.zeros((4, 3)),
                "gusts": np.array([[3., 0, 0], [0, 3., 0], [0, 0, 3.], [3., 0, 0]]),
            }

        def _reference_positions(self):
            return np.zeros((4, 3))

        def _getDroneStateVector(self, i):
            return np.zeros(20)

    rec = EpisodeRecorder(4, 10)
    rec.record(FakeEnv(), np.zeros((4, 3)),
               {"min_separation_m": 1.0, "total_coupling_N": 0.0})

    assert rec._gusts[0] == pytest.approx(3.0), (
        f"logged {rec._gusts[0]}, expected the per-drone speed 3.0"
    )


# ==============================================================================
# SECTION 5 : LADDER DEGENERACY IS MEASURABLE
# ==============================================================================

def test_information_diagnostic_detects_affine_duplication():
    """The diagnostic must return 0 when a rung adds nothing new."""
    rng = np.random.default_rng(0)
    own = rng.normal(size=(200, 5))

    assert information_added_by_rung(own, own) == 0
    assert information_added_by_rung(own, own @ rng.normal(size=(5, 5))) == 0
    assert information_added_by_rung(
        own, np.hstack([own, rng.normal(size=(200, 3))])
    ) == 3


@pytest.mark.slow
def test_every_rung_adds_information_for_every_agent():
    """THE LADDER'S DEFINING PROPERTY: k must mean the same thing to every agent.

    Until 17 Sep 2026 it did not. The most-coupled teammate of a follower in a
    vertical chain IS its predecessor, whose gap and relative velocity the actor
    already observes:

        e_i  = p_i - (p_{i-1} - Delta)   ->  p_{i-1} - p_i = Delta - e_i
        edot = v_i - v_{i-1}             ->  v_{i-1} - v_i = -edot

    (The k=1 teammate's relative velocity was bit-identical to the negated
    obs[6:9].) Agent 1 therefore gained ZERO dimensions at k=1, while agents 2
    and 3 gained 6 -- so `k` denoted a different amount of information depending
    on which drone you were. For a study whose single independent variable is k,
    that is fatal.

    This test replaces one that asserted the broken state. It now requires the
    property the experiment depends on: at every rung, every agent gains the
    same amount of genuinely new information.
    """
    env = FormationAviary(
        num_drones=4, formation_spacing_m=1.0, episode_len_sec=5.0,
        ctrl_freq=48, gust_std_ms=3.0, chain_topology=True,
    )
    ladder = build_ladder(4)

    own: dict[int, list] = {i: [] for i in range(4)}
    critic: dict[str, dict[int, list]] = {
        lv.label: {i: [] for i in range(4)} for lv in ladder
    }

    try:
        obs, _ = env.reset(seed=1234)
        for _ in range(150):
            state = env.get_global_state()
            for i in range(4):
                own[i].append(np.asarray(obs[i], dtype=np.float32))
                for lv in ladder:
                    critic[lv.label][i].append(
                        build_critic_input(i, obs[i], state, lv)
                    )
            obs, _, term, trunc, _ = env.step(
                np.random.uniform(-0.3, 0.3, (4, 3)).astype(np.float32)
            )
            if term or trunc:
                break
    finally:
        env.close()

    def added(label: str, agent: int) -> int:
        return information_added_by_rung(
            np.array(own[agent]), np.array(critic[label][agent])
        )

    # k0 adds nothing by construction -- it IS the actor's observation.
    assert all(added("k0", i) == 0 for i in range(4))

    # Every teammate rung must add information to EVERY agent, and the amount
    # must be the same across the followers.
    for lv in ladder:
        if lv.k == 0 or lv.include_privileged:
            continue
        gains = {i: added(lv.label, i) for i in range(4)}
        assert all(g > 0 for g in gains.values()), (
            f"{lv.label} adds nothing for some agent: {gains}"
        )
        followers = {gains[i] for i in (1, 2, 3)}
        assert len(followers) == 1, (
            f"{lv.label} adds different amounts to different followers: {gains}"
        )

    # Rungs must be strictly ordered: no duplicate conditions in the ladder.
    teammate_rungs = [lv for lv in ladder if not lv.include_privileged]
    for lower, higher in zip(teammate_rungs, teammate_rungs[1:]):
        assert added(higher.label, 1) > added(lower.label, 1), (
            f"{higher.label} adds no more than {lower.label} -- duplicate rung"
        )

    # The privileged rung must actually be privileged.
    top = [lv for lv in ladder if lv.include_privileged][0]
    base = teammate_rungs[-1]
    assert added(top.label, 1) > added(base.label, 1), (
        "the privileged rung adds nothing beyond the top teammate rung"
    )


# ==============================================================================
# SECTION 6 : LEARNER INSTRUMENTATION
# ==============================================================================

@pytest.mark.slow
def test_learner_records_behavioural_diagnostics():
    """Gradient norms and ACTION statistics must be recorded.

    The review's point: KL and clip fraction show only that updates are not
    exploding. They cannot tell you whether the actor gradient is vanishing, or
    whether tanh has saturated -- and the logged `entropy` is the UNSQUASHED
    Gaussian's, which cannot fall when tanh saturates, so it cannot detect the
    failure it appears to be watching.

    `action_abs_mean` is the one that matters most here: it rose monotonically
    in every v4 seed while tracking error stayed flat, which is the "acting more,
    achieving nothing" signature.
    """
    from ctde.training.trainer import Trainer

    cfg = RunConfig(
        level=build_ladder(4)[1],
        total_env_steps=600, eval_every_steps=600, eval_episodes=2,
        final_eval_episodes=2, episode_len_sec=2.0,
        mappo=MAPPOConfig(num_steps=300, num_epochs=2, minibatch_size=150),
    )
    trainer = Trainer(cfg)
    try:
        history = trainer.train(verbose=False)
        record = history[-1]
        for key in [
            "actor_grad_norm", "critic_grad_norm",
            "action_saturation", "action_abs_mean", "action_std",
        ]:
            assert key in record, f"diagnostic '{key}' is not recorded"
            assert np.isfinite(record[key]), f"'{key}' is not finite"

        assert record["actor_grad_norm"] > 0, "actor gradient is exactly zero"
        assert 0.0 <= record["action_saturation"] <= 1.0
    finally:
        trainer.close()


@pytest.mark.slow
def test_actor_can_express_a_competent_controller():
    """REPRESENTATION CHECK: the network must be able to fit a good policy.

    If the actor cannot represent a competent controller, no amount of
    exploration would find one and "exploration problem" is the wrong diagnosis.

    Measured 17 Sep 2026: supervised cloning of a P mapping on on-distribution
    observations reaches R^2 > 0.998. Representation is NOT the bottleneck, so a
    learning failure must lie in the search, the reward, or the critic.
    """
    import torch

    from ctde.models.networks import Actor

    env = FormationAviary(
        num_drones=4, formation_spacing_m=1.0, episode_len_sec=4.0,
        ctrl_freq=48, gust_std_ms=2.5, chain_topology=True,
    )
    samples = []
    try:
        obs, _ = env.reset(seed=800)
        for _ in range(180):
            samples.append(np.asarray(obs, dtype=np.float32).copy())
            obs, _, term, trunc, _ = env.step(
                np.random.uniform(-0.2, 0.2, (4, 3)).astype(np.float32)
            )
            if term or trunc:
                break
    finally:
        env.close()

    X = np.concatenate(samples, axis=0)
    Y = np.clip(-3.0 * X[:, 0:3], -1.0, 1.0)

    xt, yt = torch.tensor(X), torch.tensor(Y, dtype=torch.float32)
    torch.manual_seed(0)
    actor = Actor(obs_dim=13, action_dim=3)
    opt = torch.optim.Adam(actor.parameters(), lr=3e-4)

    for _ in range(800):
        loss = ((torch.tanh(actor.net(xt)) - yt) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

    with torch.no_grad():
        mse = float(((torch.tanh(actor.net(xt)) - yt) ** 2).mean())
    r2 = 1.0 - mse / float(yt.var())

    assert r2 > 0.9, (
        f"the actor could only reach R^2={r2:.3f} cloning a simple controller; "
        f"the architecture cannot express a competent policy"
    )


# ==============================================================================
# SECTION 7 : GATE 0 -- CHECKPOINTS AND FAILURE COST
# ==============================================================================

def test_checkpoint_carries_the_value_normalizer():
    """A checkpoint without normalizer statistics restores a WRONG critic.

    The critic trains against normalised returns and its output is denormalised
    using running mean/variance. Those are learned state. Omitting them yields a
    model that loads cleanly, runs, and is silently mis-scaled.
    """
    from ctde.training.mappo import MAPPO

    learner = MAPPO(num_agents=4, level=build_ladder(4)[0], obs_dim=13,
                    action_dim=3, config=MAPPOConfig())
    learner.value_normalizer.update(np.random.normal(500, 50, (200, 4)))

    state = learner.state_dict()
    assert "value_normalizer" in state
    assert "actor_opt" in state and "critic_opt" in state

    restored = MAPPO(num_agents=4, level=build_ladder(4)[0], obs_dim=13,
                     action_dim=3, config=MAPPOConfig())
    restored.load_state_dict(state)
    assert np.allclose(restored.value_normalizer.mean, learner.value_normalizer.mean)
    assert np.allclose(restored.value_normalizer.var, learner.value_normalizer.var)


def test_checkpoint_from_a_different_rung_is_rejected():
    """The old guard could never fire.

    It compared `critic_obs_dim`, which since the padding fix is IDENTICAL at
    every rung by design -- that is the point of exact capacity control. So a k0
    checkpoint loaded into a k2 learner succeeded silently, producing a model
    whose label and whose actual information content disagreed. The RUNG is what
    must match.
    """
    from ctde.training.mappo import MAPPO

    ladder = build_ladder(4)
    low = MAPPO(num_agents=4, level=ladder[0], obs_dim=13, action_dim=3,
                config=MAPPOConfig())
    high = MAPPO(num_agents=4, level=ladder[2], obs_dim=13, action_dim=3,
                 config=MAPPOConfig())

    # Precondition: the widths DO match, which is why the old guard was useless.
    assert low.critic_obs_dim == high.critic_obs_dim

    with pytest.raises(ValueError, match="rung"):
        high.load_state_dict(low.state_dict())


@pytest.mark.slow
def test_crashing_is_never_better_than_flying_badly():
    """Return must be monotone in survival time, for ANY policy.

    MEASURED HOLE (17 Sep 2026): a drone flying badly but legally scores as low
    as -1.07 per step while ALIVE, while crashing forfeited only the 1.00/step
    survival bonus. A persistently-bad agent could therefore PROFIT by
    terminating -- the "learn to die" failure that killed v1, reachable again
    through a different door.

    Fixed by charging every unflown step at FAILURE_STEP_COST, strictly worse
    than the worst survivable per-step reward, so the episode behaves as if it
    ran to the full horizon in an absorbing failure state.
    """
    def roll(action_fn, seed):
        env = FormationAviary(
            num_drones=4, formation_spacing_m=1.0, episode_len_sec=15.0,
            ctrl_freq=48, gust_std_ms=2.5, chain_topology=True,
        )
        try:
            obs, _ = env.reset(seed=seed)
            total = np.zeros(4)
            info: dict = {}
            while True:
                obs, reward, term, trunc, info = env.step(action_fn(obs))
                total += reward
                if term or trunc:
                    break
            return float(total.mean()), bool(info.get("crashed"))
        finally:
            env.close()

    seeds = [900_000, 900_001]
    bad = [roll(lambda o: np.full((4, 3), 0.9, dtype=np.float32), s) for s in seeds]
    dive = [roll(lambda o: np.array([[0, 0, -1.0]] * 4, dtype=np.float32), s)
            for s in seeds]

    assert all(not crashed for _, crashed in bad), "the 'bad' policy must survive"
    assert all(crashed for _, crashed in dive), "the 'dive' policy must crash"

    worst_surviving = min(r for r, _ in bad)
    best_crashing = max(r for r, _ in dive)

    assert best_crashing < worst_surviving, (
        f"crashing ({best_crashing:.1f}) beats flying badly ({worst_surviving:.1f}) "
        f"-- early termination is profitable and the policy can learn to die"
    )
