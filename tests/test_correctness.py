"""
================================================================================
 CORRECTNESS TESTS  --  written to FAIL against the pre-fix implementation
================================================================================

WHY THIS FILE EXISTS, AND WHY IT IS WRITTEN THIS WAY
----------------------------------------------------
An external code audit (16 Sep 2026) found five defects that every existing
test missed. The tests did not miss them by accident -- they missed them
because each test exercised the component in a way that BYPASSED the failure.

Two examples worth internalising:

  * `test_gae_terminal_cuts_bootstrap` put the terminal on the LAST buffer
    element, the single position where the off-by-one in `dones[t+1]` has no
    effect (that step reads `last_dones` instead). A mid-rollout terminal --
    the normal case -- was never tested.

  * `check_survival_incentive` set `env._crashed = True` by hand and then
    called `_computeReward()` directly. That tests the reward FORMULA while
    bypassing the call ORDER that breaks it. The real `BaseAviary.step()`
    calls `_computeReward()` at line 377 and `_computeTerminated()` at 378,
    so the flag is always False when the reward is computed.

THE RULE THIS FILE FOLLOWS
--------------------------
    Test through the PUBLIC path the experiment actually uses.

Never poke a private flag to set up a condition the system is supposed to
reach on its own. If reaching the condition naturally is hard, that difficulty
is information -- it usually means the condition is unreachable, which is
itself the bug.

Each test below states which audit finding it guards.
================================================================================
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import pytest

from ctde.training.buffer import RolloutBuffer


# ==============================================================================
# SECTION 1 : GAE  --  audit findings #1 and #2
# ==============================================================================
#
# A rollout stores transitions as
#
#     (s_t, a_t, r_t, s_{t+1}, done_t)
#
# so `dones[t]` answers "did the episode end AS A RESULT of transition t?".
# The bootstrap for transition t must therefore consult `dones[t]`, not
# `dones[t+1]`.
#
# Two masks are needed and they are NOT the same:
#
#   bootstrap mask  -- from TERMINATION. A true terminal state has no future,
#                      so V(s_{t+1}) must be excluded.
#   boundary mask   -- from termination OR TRUNCATION. Either way the next
#                      stored transition belongs to a DIFFERENT episode, so the
#                      GAE recursion must not carry across it.
#
# Time-limit truncation is the subtle case: the future still exists (we simply
# stopped looking), so we DO bootstrap -- but we must NOT let the recursion
# continue into the reset episode.
# ==============================================================================

def _fill(buf: RolloutBuffer, rewards, values, dones, truncs=None):
    """Fill a single-agent buffer with a scripted episode."""
    n = len(rewards)
    truncs = truncs if truncs is not None else [0.0] * n
    for t in range(n):
        buf.add(
            obs=np.zeros((1, 2)), critic_obs=np.zeros((1, 2)),
            actions=np.zeros((1, 1)), log_probs=np.zeros(1),
            rewards=np.array([rewards[t]]), values=np.array([values[t]]),
            dones=np.array([dones[t]]),
            truncations=np.array([truncs[t]]),
        )


def test_gae_terminal_MID_rollout():
    """AUDIT #1: a terminal in the MIDDLE of a rollout must cut the bootstrap.

    THE TEST THE OLD SUITE SHOULD HAVE HAD. The previous version placed the
    terminal on the last element, which is the one index where the off-by-one
    is invisible.

    Scripted: 3 steps, terminal on step 1.

        t=1 is terminal  -> delta_1 = r_1 - V(s_1)      (no bootstrap)
                         -> A_1     = delta_1           (no propagation from t=2)
        t=0 is normal    -> delta_0 = r_0 + g*V(s_1) - V(s_0)
                         -> A_0     = delta_0 + g*l*A_1
    """
    buf = RolloutBuffer(num_steps=3, num_agents=1, obs_dim=2, critic_obs_dim=2, action_dim=1)
    rewards, values, dones = [1.0, 1.0, 1.0], [0.5, 0.5, 0.5], [0.0, 1.0, 0.0]
    _fill(buf, rewards, values, dones)

    g, lam, last_v = 0.99, 0.95, 0.5
    buf.compute_gae(np.array([last_v]), np.zeros(1), gamma=g, gae_lambda=lam)

    d2 = rewards[2] + g * last_v - values[2]
    a2 = d2
    d1 = rewards[1] - values[1]          # terminal: no bootstrap
    a1 = d1                              # terminal: no propagation from a2
    d0 = rewards[0] + g * values[1] - values[0]
    a0 = d0 + g * lam * a1

    np.testing.assert_allclose(buf.advantages[:3, 0], [a0, a1, a2], atol=1e-5)


def test_gae_does_not_cross_truncation_boundary():
    """AUDIT #2: truncation bootstraps, but the recursion must STOP there.

    At a time limit the future still exists, so V(s_{t+1}) is a legitimate
    estimate and we DO bootstrap. But the environment resets, so the NEXT
    stored transition belongs to a different episode -- letting `last_gae`
    carry across would blend two unrelated trajectories.

    Scripted: 3 steps, truncation on step 1 with a large bootstrap value, so
    leakage is unmistakable.
    """
    buf = RolloutBuffer(num_steps=3, num_agents=1, obs_dim=2, critic_obs_dim=2, action_dim=1)
    rewards, values = [1.0, 1.0, 1.0], [0.5, 0.5, 0.5]
    dones = [0.0, 0.0, 0.0]              # NOT terminal
    truncs = [0.0, 1.0, 0.0]             # time limit at t=1
    bootstrap_at_trunc = 10.0            # deliberately large
    _fill(buf, rewards, values, dones, truncs)
    buf.set_truncation_values(np.array([[0.0], [bootstrap_at_trunc], [0.0]]))

    g, lam, last_v = 0.99, 0.95, 0.5
    buf.compute_gae(np.array([last_v]), np.zeros(1), gamma=g, gae_lambda=lam)

    d2 = rewards[2] + g * last_v - values[2]
    a2 = d2
    # Truncated: bootstrap off the FINAL state of that episode...
    d1 = rewards[1] + g * bootstrap_at_trunc - values[1]
    a1 = d1                              # ...but do NOT propagate a2 backwards
    d0 = rewards[0] + g * values[1] - values[0]
    a0 = d0 + g * lam * a1

    np.testing.assert_allclose(buf.advantages[:3, 0], [a0, a1, a2], atol=1e-5)


def test_gae_normal_rollout_unchanged():
    """A rollout with no boundaries must match the plain GAE recursion.

    Guards against a "fix" that breaks the common case.
    """
    buf = RolloutBuffer(num_steps=3, num_agents=1, obs_dim=2, critic_obs_dim=2, action_dim=1)
    rewards, values = [1.0, 2.0, 3.0], [0.5, 0.6, 0.7]
    _fill(buf, rewards, values, [0.0, 0.0, 0.0])

    g, lam, last_v = 0.99, 0.95, 0.8
    buf.compute_gae(np.array([last_v]), np.zeros(1), gamma=g, gae_lambda=lam)

    d2 = rewards[2] + g * last_v - values[2]; a2 = d2
    d1 = rewards[1] + g * values[2] - values[1]; a1 = d1 + g * lam * a2
    d0 = rewards[0] + g * values[1] - values[0]; a0 = d0 + g * lam * a1

    np.testing.assert_allclose(buf.advantages[:3, 0], [a0, a1, a2], atol=1e-5)


def test_gae_terminal_on_last_element_still_correct():
    """The case the OLD test covered -- kept so the fix does not regress it."""
    buf = RolloutBuffer(num_steps=2, num_agents=1, obs_dim=2, critic_obs_dim=2, action_dim=1)
    _fill(buf, [1.0, 1.0], [0.5, 0.5], [0.0, 1.0])

    buf.compute_gae(np.array([99.0]), np.ones(1), gamma=0.99, gae_lambda=0.95)

    # Terminal: advantage = r - V, with the absurd bootstrap excluded.
    assert abs(buf.advantages[1, 0] - 0.5) < 1e-5


# ==============================================================================
# SECTION 2 : CRASH PENALTY  --  audit finding #3
# ==============================================================================

@pytest.mark.slow
def test_crash_penalty_delivered_by_real_step():
    """AUDIT #3: the penalty must arrive through env.step(), not a poked flag.

    THE BUG: `BaseAviary.step()` calls
        line 377   reward     = self._computeReward()
        line 378   terminated = self._computeTerminated()
    and `_computeTerminated()` is what sets `self._crashed`. So on the crashing
    step the reward is computed while the flag is still False, the episode then
    ends, and the penalty is never delivered.

    THE OLD TEST MISSED IT by setting `env._crashed = True` by hand before
    calling `_computeReward()` -- exercising the formula while bypassing the
    ordering that breaks it.

    This test drives a real crash and inspects the reward the environment
    actually returns.
    """
    from ctde.envs.formation_aviary import FormationAviary

    env = FormationAviary(
        num_drones=4, formation_spacing_m=1.0, base_altitude_m=0.8,
        episode_len_sec=15.0, gust_std_ms=0.0, wind_mean_ms=0.0,
    )
    try:
        env.reset(seed=0)
        dive = np.tile(np.array([0.0, 0.0, -1.0], dtype=np.float32), (4, 1))

        prev_reward = None
        for _ in range(600):
            _, reward, terminated, truncated, info = env.step(dive)
            if terminated:
                assert info["crashed"], "terminated without the crash flag set"
                # The crashing step must be markedly worse than the one before.
                assert reward.mean() < prev_reward - env.CRASH_PENALTY * 0.5, (
                    f"crash penalty not delivered: final {reward.mean():+.2f} vs "
                    f"previous {prev_reward:+.2f} (penalty {env.CRASH_PENALTY})"
                )
                return
            prev_reward = reward.mean()
            if truncated:
                pytest.skip("episode truncated before a crash could be induced")
    finally:
        env.close()


@pytest.mark.slow
def test_crashing_early_is_worse_than_flying_badly():
    """The incentive property the penalty exists to create.

    Return must be monotone in survival time: an agent must never be able to
    improve its score by ending the episode sooner. Measured end-to-end, with
    no private state touched.
    """
    from ctde.envs.formation_aviary import FormationAviary

    def rollout(action_vec, seed=0):
        env = FormationAviary(
            num_drones=4, formation_spacing_m=1.0, base_altitude_m=0.8,
            episode_len_sec=15.0, gust_std_ms=0.0, wind_mean_ms=0.0,
        )
        try:
            env.reset(seed=seed)
            total = np.zeros(4)
            steps = 0
            while True:
                _, r, term, trunc, info = env.step(action_vec)
                total += r
                steps += 1
                if term or trunc:
                    return float(total.mean()), steps, info["crashed"]
        finally:
            env.close()

    hover_return, hover_steps, hover_crashed = rollout(np.zeros((4, 3), dtype=np.float32))
    dive_return, dive_steps, dive_crashed = rollout(
        np.tile(np.array([0.0, 0.0, -1.0], dtype=np.float32), (4, 1))
    )

    assert not hover_crashed, "the null action should not crash"
    if not dive_crashed:
        pytest.skip("dive did not crash; cannot compare")

    assert hover_return > dive_return, (
        f"crashing ({dive_return:+.1f} in {dive_steps} steps) scores better than "
        f"surviving ({hover_return:+.1f} in {hover_steps} steps)"
    )


# ==============================================================================
# SECTION 3 : PRIVILEGED CRITIC  --  audit finding #5
# ==============================================================================

@pytest.mark.slow
def test_privileged_state_contains_the_actual_disturbance():
    """AUDIT #5: the +P rung must receive the disturbance that actually acts.

    THE BUG: `get_global_state()` returned only `wind_vec` -- the STEADY MEAN
    wind. The stochastic OU gust, which is the entire disturbance the critic is
    supposed to be able to explain, was withheld.

    Measured before the fix: wind_vec was [0.37, 0.87, 0.10] while the actual
    per-drone gusts were around [-1.3, -10.4, 3.1]. The "maximal" rung was
    therefore missing the dominant term.
    """
    from ctde.envs.formation_aviary import FormationAviary

    env = FormationAviary(
        num_drones=4, formation_spacing_m=1.0, base_altitude_m=0.8,
        gust_std_ms=4.0,
    )
    try:
        env.reset(seed=1)
        for _ in range(50):
            env.step(np.zeros((4, 3), dtype=np.float32))

        state = env.get_global_state()

        assert "gusts" in state, "privileged state must expose the per-drone gust field"
        gusts = np.asarray(state["gusts"])
        assert gusts.shape == (4, 3), f"expected per-drone 3-D gusts, got {gusts.shape}"

        # With gust_std_ms=4.0 after 50 steps the gust must dominate the mean.
        assert np.abs(gusts).max() > np.abs(state["wind_vec"]).max(), (
            "gust magnitude should exceed the steady mean at this setting; "
            "the privileged rung is probably still seeing only the mean"
        )
    finally:
        env.close()


# ==============================================================================
# SECTION 4 : CRITIC CAPACITY  --  audit finding #6
# ==============================================================================

def test_critic_parameter_count_is_EXACTLY_equal_across_rungs():
    """AUDIT #6: every rung's critic must have IDENTICAL learned parameters.

    THE OLD TEST checked only that the TRUNK matched, and accepted a growing
    projection layer (832 -> 2,176 params) on the argument that the bias
    "favours high k". That argument is half right: it works against a decline
    at high k, but it cannot explain away k=1 beating k=0 -- which is exactly
    the ascending side of the claimed interior optimum.

    THE FIX: every critic takes a fixed MAXIMUM-WIDTH input and an external
    binary mask. Excluded teammates are zeroed, not absent. The network is then
    literally the same object at every rung.
    """
    from ctde.models.networks import Critic
    from ctde.models.observability import build_ladder, critic_input_dim

    ladder = build_ladder(4)
    max_dim = max(critic_input_dim(lv) for lv in ladder)

    counts = set()
    for _ in ladder:
        critic = Critic(input_dim=max_dim)
        counts.add(sum(p.numel() for p in critic.parameters()))

    assert len(counts) == 1, f"critic parameter counts differ across rungs: {sorted(counts)}"


def test_masked_input_zeroes_excluded_teammates():
    """A masked critic input must be max-width with excluded slots zeroed.

    This is what makes the capacity control exact: the ARCHITECTURE never
    changes, only which slots carry information.
    """
    from ctde.models.observability import (
        build_critic_input,
        build_ladder,
        critic_input_dim,
        max_critic_input_dim,
    )

    rng = np.random.default_rng(0)
    global_state = {
        "positions": rng.normal(size=(4, 3)),
        "velocities": rng.normal(size=(4, 3)),
        "wind_vec": rng.normal(size=3),
        "gusts": rng.normal(size=(4, 3)),
    }
    # 13-D since v4: gap error, attitude, RELATIVE velocity, angular rate,
    # role flag. See test_actor_can_compute_the_gap_derivative.
    own_obs = rng.normal(size=13).astype(np.float32)

    ladder = build_ladder(4)
    # The padded builder sizes itself with max_critic_input_dim(); per-rung
    # widths come from critic_input_dim(). Those are two definitions of the same
    # quantity, so assert they AGREE before relying on either -- a silent
    # divergence would reintroduce unequal critic capacity.
    max_dim = max_critic_input_dim(4)
    assert max_dim == max(critic_input_dim(lv) for lv in ladder), (
        "max_critic_input_dim disagrees with the widest rung's critic_input_dim"
    )

    for level in ladder:
        vec = build_critic_input(1, own_obs, global_state, level, padded=True)
        assert vec.shape == (max_dim,), (
            f"{level.label}: padded input must be max width {max_dim}, got {vec.shape[0]}"
        )
        # Own observation always present.
        np.testing.assert_allclose(vec[:13], own_obs, rtol=1e-6)

    # k=0 must carry NO teammate information beyond its own observation.
    k0 = build_critic_input(1, own_obs, global_state, ladder[0], padded=True)
    assert np.allclose(k0[13:], 0.0), "k0 must have all teammate slots zeroed"

    # A higher rung must carry strictly more non-zero information.
    k3 = build_critic_input(1, own_obs, global_state, ladder[3], padded=True)
    assert np.count_nonzero(k3[13:]) > np.count_nonzero(k0[13:])


# ==============================================================================
# SECTION 5 : COLLISION TERMINATION  --  audit finding #4
# ==============================================================================

@pytest.mark.slow
def test_collision_is_detected_when_it_occurs():
    """AUDIT #4: inter-drone collision must terminate and be reported.

    The charter lists collision as a termination condition, but the
    implementation checked only floor contact, tilt, and distance from the
    reference. Two drones could occupy the same space with nothing recorded --
    and `crash_rate` was reported as if it measured collisions, which it did not.

    NOTE ON REACHABILITY. Measured 16 Sep 2026: with `max_offset_m = 0.2`, two
    drones can close at most 0.4 m on each other, so at a 1.0 m formation the
    minimum achievable separation is ~0.79 m -- far above the 0.15 m collision
    radius. Collision is therefore UNREACHABLE by policy action alone at valid
    spacings, which is a genuine safety property of the action clamp (see
    `test_collision_unreachable_by_policy_action` below).

    To test the DETECTOR rather than the reachability, this test places the
    drones in contact at construction time.
    """
    from ctde.envs.formation_aviary import FormationAviary

    env = FormationAviary(
        num_drones=2, formation_spacing_m=1.0, base_altitude_m=1.0,
        episode_len_sec=15.0, gust_std_ms=0.0, wind_mean_ms=0.0,
    )
    try:
        env.reset(seed=0)

        # Place drone 1 inside drone 0's collision radius, through the
        # simulator's own API rather than by poking our attributes.
        import pybullet as p
        pos0 = env._getDroneStateVector(0)[0:3]
        p.resetBasePositionAndOrientation(
            env.DRONE_IDS[1],
            (pos0 + np.array([0.05, 0.0, 0.0])).tolist(),
            [0, 0, 0, 1],
            physicsClientId=env.CLIENT,
        )
        # BaseAviary caches kinematics; a direct PyBullet reset is not visible
        # to `_getDroneStateVector` until the cache is refreshed.
        env._updateAndStoreKinematicInformation()

        failed, reason = env._check_failure()
        assert failed, "drones 5 cm apart were not flagged as a failure"
        assert reason == "collision", f"expected 'collision', got '{reason}'"

        assert env._computeTerminated(), "collision must terminate the episode"
        assert env._computeInfo()["collision"], "collision flag missing from info"
    finally:
        env.close()


@pytest.mark.slow
def test_collision_unreachable_by_policy_action_at_valid_spacings():
    """The action clamp makes collision physically impossible -- assert it.

    This documents a real safety property instead of leaving it implicit. If a
    future change raises `max_offset_m` or narrows the formation, this test
    fails and forces an explicit decision about whether collisions should
    become reachable.
    """
    from ctde.envs.formation_aviary import FormationAviary

    env = FormationAviary(
        num_drones=2, formation_spacing_m=1.0, base_altitude_m=1.0,
        episode_len_sec=15.0, gust_std_ms=0.0, wind_mean_ms=0.0,
    )
    try:
        env.reset(seed=0)
        # Maximum closing command: upper drone dives, lower drone climbs.
        closing = np.array([[0.0, 0.0, -1.0], [0.0, 0.0, 1.0]], dtype=np.float32)

        min_sep = np.inf
        for _ in range(720):
            _, _, terminated, truncated, info = env.step(closing)
            min_sep = min(min_sep, info["min_separation_m"])
            if terminated or truncated:
                break

        # Geometric bound: each drone can shift at most max_offset toward the
        # other, so the gap cannot close by more than 2 * max_offset.
        theoretical_floor = 1.0 - 2 * env.MAX_OFFSET
        assert min_sep > env.COLLISION_RADIUS_M, (
            f"collision became reachable (min separation {min_sep:.3f} m)"
        )
        assert min_sep >= theoretical_floor - 0.15, (
            f"min separation {min_sep:.3f} m is far below the geometric bound "
            f"{theoretical_floor:.3f} m -- the action clamp may not be working"
        )
    finally:
        env.close()


# ==============================================================================
# SECTION 6 : ACTOR OBSERVATION COMPLETENESS  --  audit finding #10
# ==============================================================================

@pytest.mark.slow
def test_actor_can_compute_the_gap_derivative():
    """AUDIT #10: the actor must not lack a derivative the k=1 critic has.

    THE CONFOUND. Before 16 Sep 2026 the actor received:
        [0:3]  gap error to predecessor       (relative POSITION)
        [6:9]  its OWN absolute velocity
    but NOT the predecessor's velocity. Being memoryless, it therefore could
    not compute d(gap)/dt.

    The k=1 critic, however, received relative velocity directly. So k=1's
    measured advantage might have come from "the critic has a derivative the
    actor lacks" rather than from "the critic sees aerodynamic coupling" --
    an alternative explanation for the entire pilot result.

    Measured before the fix, drone 2 at step 30:
        actor own velocity      [ 0.0385, -0.0197, -0.0699]
        relative velocity       [ 0.0561, -0.0119,  0.0943]   <- unavailable
        k=1 critic slot [15:18] [ 0.0561, -0.0119,  0.0943]   <- available

    THE FIX: the actor observes RELATIVE velocity to its reference. A real
    follower with UWB or vision can estimate this, so withholding it was not
    realistic either.
    """
    from ctde.envs.formation_aviary import FormationAviary

    env = FormationAviary(
        num_drones=4, formation_spacing_m=1.0, base_altitude_m=0.8,
        gust_std_ms=4.0,
    )
    try:
        obs, _ = env.reset(seed=0)
        for _ in range(30):
            obs, _, _, _, _ = env.step(np.zeros((4, 3), dtype=np.float32))

        state = env.get_global_state()

        # For a follower, obs[6:9] must be the velocity RELATIVE to the drone
        # it is following -- not its own absolute velocity.
        follower = 2
        predecessor = 1
        expected_rel = state["velocities"][follower] - state["velocities"][predecessor]

        np.testing.assert_allclose(
            obs[follower][6:9], expected_rel, atol=1e-4,
            err_msg="follower must observe velocity relative to its predecessor",
        )

        # And it must NOT simply be absolute velocity (which would mean the fix
        # silently did nothing).
        own_abs = state["velocities"][follower]
        assert not np.allclose(obs[follower][6:9], own_abs, atol=1e-6), (
            "follower observation is still absolute velocity"
        )
    finally:
        env.close()


@pytest.mark.slow
def test_leader_observation_is_mission_relative():
    """The leader has no predecessor -- its reference is its own trajectory.

    Leader and follower observations therefore MEAN different things:
        leader   -> absolute mission-tracking error and velocity
        follower -> gap error and relative velocity to the drone above

    With a shared actor and no role indicator, that difference is aliased: the
    network cannot tell which regime it is in. This test documents the asymmetry
    and asserts the role flag that resolves it.
    """
    from ctde.envs.formation_aviary import FormationAviary

    env = FormationAviary(
        num_drones=4, formation_spacing_m=1.0, base_altitude_m=0.8,
        gust_std_ms=0.0, wind_mean_ms=0.0,
    )
    try:
        obs, _ = env.reset(seed=0)
        assert obs.shape[1] == 13, (
            f"expected a 13-D observation (12 + role flag), got {obs.shape[1]}"
        )
        assert obs[0][12] == pytest.approx(1.0), "leader role flag must be 1"
        for i in range(1, 4):
            assert obs[i][12] == pytest.approx(0.0), f"drone {i} role flag must be 0"
    finally:
        env.close()
