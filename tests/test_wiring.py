"""
================================================================================
 WIRING TESTS  --  does the RUNTIME PATH do what the charter claims?
================================================================================

WHY THIS FILE EXISTS
--------------------
A second external review (Sep 2026) found that three "fixes" from the first
audit were implemented but NEVER CONNECTED to the training path:

  * `_build_padded_critic_input()` existed; MAPPO never called it.
  * `get_global_state()` exposed `air_velocity`; the critic still got `wind_vec`.
  * `metrics.py` existed; nothing imported it.

Every one of these had a PASSING test. The tests passed because they built
their own objects:

    critic = Critic(input_dim=max_dim)          # constructed by hand
    assert params are equal                     # trivially true

That assertion says nothing about what `MAPPO` actually constructs. A test that
builds its own objects can NEVER detect an unconnected code path.

THE RULE THIS FILE ENFORCES
---------------------------
    Instantiate the REAL objects the experiment uses -- Trainer, MAPPO, the
    environment -- and assert on THEIR state.

If a property cannot be checked that way, the code is wrong, not the test.
================================================================================
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import pytest

from ctde.models.observability import build_ladder, max_critic_input_dim
from ctde.training.mappo import MAPPO, MAPPOConfig


def _learners(num_agents: int = 4):
    """One REAL MAPPO learner per rung."""
    cfg = MAPPOConfig(num_steps=8, num_epochs=1, minibatch_size=8)
    return [MAPPO(num_agents=num_agents, level=lv, config=cfg)
            for lv in build_ladder(num_agents)]


# ==============================================================================
# SECTION 1 : CAPACITY CONTROL, ON REAL LEARNERS
# ==============================================================================

def test_real_mappo_critics_have_identical_input_width():
    """BLOCKER 1: every rung's critic must take the SAME input width.

    Measured before the fix, on real learners:
        k0   13-D    83,585 params
        k3+P 34-D    84,929 params      spread 1,344

    The previous test constructed `Critic(input_dim=max_dim)` by hand, so it
    could not see that MAPPO was building something else entirely.
    """
    dims = {m.level.label: m.critic_obs_dim for m in _learners()}
    assert len(set(dims.values())) == 1, (
        f"critic input widths differ across rungs: {dims}"
    )
    assert set(dims.values()) == {max_critic_input_dim(4, own_obs_dim=13)}


def test_real_mappo_critics_have_identical_parameter_count():
    """The capacity-control claim, checked on the objects that actually train."""
    counts = {
        m.level.label: sum(p.numel() for p in m.critic.parameters())
        for m in _learners()
    }
    assert len(set(counts.values())) == 1, (
        f"critic parameter counts differ across rungs: {counts}"
    )


def test_real_act_returns_fixed_width_critic_input():
    """`MAPPO.act()` -- the function the trainer calls -- must emit padded input."""
    rng = np.random.default_rng(0)
    obs = rng.normal(size=(4, 13)).astype(np.float32)
    global_state = {
        "positions": rng.normal(size=(4, 3)),
        "velocities": rng.normal(size=(4, 3)),
        "wind_vec": rng.normal(size=3),
        "gusts": rng.normal(size=(4, 3)),
        "air_velocity": rng.normal(size=(4, 3)),
    }

    expected = max_critic_input_dim(4, own_obs_dim=13)
    for learner in _learners():
        _, _, _, critic_obs = learner.act(obs, global_state)
        assert critic_obs.shape == (4, expected), (
            f"{learner.level.label}: act() produced {critic_obs.shape}, "
            f"expected (4, {expected})"
        )


def test_real_act_zeroes_excluded_teammate_slots():
    """k0 must carry no teammate information; higher rungs strictly more."""
    rng = np.random.default_rng(1)
    obs = rng.normal(size=(4, 13)).astype(np.float32)
    global_state = {
        "positions": rng.normal(size=(4, 3)),
        "velocities": rng.normal(size=(4, 3)),
        "wind_vec": rng.normal(size=3),
        "gusts": rng.normal(size=(4, 3)),
        "air_velocity": rng.normal(size=(4, 3)),
    }

    ladder = _learners()
    k0 = ladder[0].act(obs, global_state)[3]
    k3 = ladder[3].act(obs, global_state)[3]

    assert np.allclose(k0[:, 13:], 0.0), "k0 must have every teammate slot zeroed"
    assert np.count_nonzero(k3[:, 13:]) > np.count_nonzero(k0[:, 13:])


# ==============================================================================
# SECTION 2 : PRIVILEGED INPUT, ON THE REAL PATH
# ==============================================================================

@pytest.mark.slow
def test_privileged_rung_critic_actually_receives_the_disturbance():
    """BLOCKER 2: the +P rung must CONSUME the gust, not merely have it exposed.

    Measured before the fix, from a real `act()` call:
        air_velocity (intended)   [-0.933, -9.504,  3.199]
        critic received           [ 0.366,  0.869,  0.105]   <- steady mean

    The earlier test asserted only that `get_global_state()` CONTAINED gusts.
    It never checked what the critic consumed.
    """
    from ctde.envs.formation_aviary import FormationAviary

    env = FormationAviary(
        num_drones=4, formation_spacing_m=1.0, base_altitude_m=0.8,
        gust_std_ms=4.0,
    )
    try:
        obs, _ = env.reset(seed=1)
        for _ in range(50):
            obs, _, _, _, _ = env.step(np.zeros((4, 3), dtype=np.float32))

        state = env.get_global_state()
        learner = _learners()[-1]                      # the +P rung
        _, _, _, critic_obs = learner.act(obs, state)

        privileged = critic_obs[0][-3:]
        np.testing.assert_allclose(
            privileged, state["air_velocity"][0], atol=1e-3,
            err_msg="privileged block must carry per-drone air velocity",
        )
        assert not np.allclose(privileged, state["wind_vec"], atol=1e-3), (
            "privileged block is still the steady mean wind"
        )
    finally:
        env.close()


@pytest.mark.slow
def test_non_privileged_rungs_get_no_disturbance_information():
    """Only the +P rung may see the disturbance; others must have it zeroed."""
    from ctde.envs.formation_aviary import FormationAviary

    env = FormationAviary(
        num_drones=4, formation_spacing_m=1.0, base_altitude_m=0.8,
        gust_std_ms=4.0,
    )
    try:
        obs, _ = env.reset(seed=1)
        for _ in range(50):
            obs, _, _, _, _ = env.step(np.zeros((4, 3), dtype=np.float32))
        state = env.get_global_state()

        for learner in _learners()[:-1]:               # every rung except +P
            _, _, _, critic_obs = learner.act(obs, state)
            assert np.allclose(critic_obs[:, -3:], 0.0), (
                f"{learner.level.label} must not receive privileged state"
            )
    finally:
        env.close()


# ==============================================================================
# SECTION 3 : WIND ACROSS PHYSICS SUBSTEPS
# ==============================================================================

@pytest.mark.slow
def test_wind_is_applied_on_every_physics_substep():
    """BLOCKER 3: wind must act on all substeps, not just the first.

    `_preprocessAction()` runs ONCE per control step, but PyBullet then executes
    `PYB_STEPS_PER_CTRL` physics substeps and CLEARS external forces after each.
    Applying wind only in `_preprocessAction()` therefore delivered 1/5 of the
    intended impulse at ctrl_freq=48.

    Measured before the fix: 1 call per env.step(), 5 substeps -> 20%.
    """
    from ctde.envs.formation_aviary import FormationAviary

    calls = {"n": 0}

    class Counted(FormationAviary):
        def _apply_wind(self):
            calls["n"] += 1
            return super()._apply_wind()

    env = Counted(
        num_drones=4, formation_spacing_m=1.0, base_altitude_m=0.8,
        gust_std_ms=4.0,
    )
    try:
        env.reset(seed=0)
        calls["n"] = 0
        env.step(np.zeros((4, 3), dtype=np.float32))

        assert calls["n"] == env.PYB_STEPS_PER_CTRL, (
            f"wind applied {calls['n']} times per control step, "
            f"expected {env.PYB_STEPS_PER_CTRL} (once per physics substep)"
        )
    finally:
        env.close()


@pytest.mark.slow
def test_wind_impulse_is_control_frequency_invariant():
    """The same wind over the same wall-clock time must impart the same impulse.

    This is the decisive PHYSICAL check. If wind is applied per CONTROL step
    rather than per PHYSICS substep, the delivered impulse scales with
    `ctrl_freq` and the disturbance silently depends on a simulation setting.

    Measured after the substep fix, lateral drift over 1 s of 8 m/s wind:
        ctrl_freq= 48   0.14715 m
        ctrl_freq=120   0.14136 m
        ctrl_freq=240   0.14141 m        -> agree within ~4%

    NOTE ON AN EARLIER VERSION OF THIS TEST: it hand-rolled the physics loop
    (`_apply_wind`, `_physics`, `stepSimulation`) instead of calling
    `env.step()`. It therefore bypassed the very integration path it was meant
    to validate and reported a false failure. Fixed to use the public path --
    the same rule as the rest of this file.
    """
    from ctde.envs.formation_aviary import FormationAviary

    def drift_after_one_second(ctrl_freq: int) -> float:
        env = FormationAviary(
            num_drones=1, formation_spacing_m=1.0, base_altitude_m=2.0,
            episode_len_sec=5.0, gust_std_ms=0.0, wind_mean_ms=8.0,
            ctrl_freq=ctrl_freq,
        )
        try:
            env.reset(seed=0)
            start = env._getDroneStateVector(0)[0:2].copy()
            for _ in range(ctrl_freq):              # exactly 1 second
                env.step(np.zeros((1, 3), dtype=np.float32))
            end = env._getDroneStateVector(0)[0:2]
            return float(np.linalg.norm(end - start))
        finally:
            env.close()

    slow = drift_after_one_second(48)
    fast = drift_after_one_second(240)

    relative_difference = abs(slow - fast) / max(fast, 1e-9)
    assert relative_difference < 0.20, (
        f"wind impulse depends on ctrl_freq: {slow:.5f} m at 48 Hz vs "
        f"{fast:.5f} m at 240 Hz ({relative_difference:.1%} apart)"
    )


# ==============================================================================
# SECTION 4 : BOUNDED ACTIONS
# ==============================================================================

def test_policy_actions_are_bounded():
    """BLOCKER 4: sampled actions must lie in [-1, 1].

    PPO stores the log-probability of the SAMPLED action, but the environment
    clips to [-1,1]. Whenever clipping occurs,

        a_executed  !=  a_in_the_likelihood_ratio

    which biases the objective. Saturation is rare at the current log_std, but
    the learned MEAN is unbounded, so it can become common.
    """
    import torch
    from ctde.models.networks import Actor

    actor = Actor(obs_dim=13, action_dim=3)
    torch.manual_seed(0)

    # Drive the mean far outside the bound to force the issue.
    with torch.no_grad():
        actor.net[-1].bias.fill_(5.0)

    dist_sample = actor.distribution(torch.zeros(500, 13)).sample()
    assert dist_sample.abs().max() <= 1.0 + 1e-5, (
        f"policy produced |a| = {dist_sample.abs().max():.3f} > 1; "
        "an unbounded distribution biases the PPO ratio under clipping"
    )


def test_bounded_policy_log_prob_is_finite_and_correct_shape():
    """The squashing correction must not produce NaN or inf."""
    import torch
    from ctde.models.networks import Actor

    actor = Actor(obs_dim=13, action_dim=3)
    obs = torch.randn(64, 13)
    dist = actor.distribution(obs)
    actions = dist.sample()
    log_probs = dist.log_prob(actions).sum(dim=-1)

    assert log_probs.shape == (64,)
    assert torch.isfinite(log_probs).all(), "log-probabilities must be finite"
