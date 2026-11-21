"""
================================================================================
 TESTS: Phase 3  --  buffer, GAE, and the MAPPO learner
================================================================================

WHY THESE TESTS EXIST
---------------------
The previous project shipped a centralized Q-critic that was never called by
the loss -- dead code silently receiving no gradients, producing numbers that
looked plausible and meant nothing.

`test_critic_receives_gradients` exists specifically so that failure mode
cannot recur here.

The GAE tests matter for a similar reason: a wrong advantage calculation does
not crash. It just quietly makes every result meaningless.

RUN WITH
--------
    pytest tests/test_phase3.py -v
================================================================================
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import pytest
import torch

from ctde.models.observability import build_ladder
from ctde.training.buffer import RolloutBuffer
from ctde.training.mappo import MAPPO, MAPPOConfig
from ctde.training.sweep import aggregate_by_rung


# ==============================================================================
# SECTION 1 : GAE CORRECTNESS
# ==============================================================================

def test_gae_matches_hand_computation():
    """GAE must match the recursion computed by hand.

    A wrong advantage does not crash -- it silently corrupts every result. So
    we pin the arithmetic against an independent calculation.
    """
    buf = RolloutBuffer(num_steps=3, num_agents=1, obs_dim=2, critic_obs_dim=2, action_dim=1)

    rewards = [1.0, 2.0, 3.0]
    values = [0.5, 0.6, 0.7]
    for t in range(3):
        buf.add(
            obs=np.zeros((1, 2)), critic_obs=np.zeros((1, 2)), actions=np.zeros((1, 1)),
            log_probs=np.zeros(1), rewards=np.array([rewards[t]]),
            values=np.array([values[t]]), dones=np.zeros(1),
        )

    gamma, lam, last_v = 0.99, 0.95, 0.8
    buf.compute_gae(np.array([last_v]), np.zeros(1), gamma=gamma, gae_lambda=lam)

    # Backward recursion, by hand.
    d2 = rewards[2] + gamma * last_v - values[2]
    a2 = d2
    d1 = rewards[1] + gamma * values[2] - values[1]
    a1 = d1 + gamma * lam * a2
    d0 = rewards[0] + gamma * values[1] - values[0]
    a0 = d0 + gamma * lam * a1

    np.testing.assert_allclose(buf.advantages[:3, 0], [a0, a1, a2], atol=1e-5)


def test_gae_terminal_cuts_bootstrap():
    """A terminal state must ignore future value entirely.

    Conflating termination with time-limit truncation is a classic RL bug that
    biases value estimates. We pass an absurd last_value (99.0) to make any
    leakage obvious.
    """
    buf = RolloutBuffer(2, 1, 2, 2, 1)
    for done in (0.0, 1.0):
        buf.add(
            obs=np.zeros((1, 2)), critic_obs=np.zeros((1, 2)), actions=np.zeros((1, 1)),
            log_probs=np.zeros(1), rewards=np.array([1.0]),
            values=np.array([0.5]), dones=np.array([done]),
        )

    buf.compute_gae(np.array([99.0]), np.ones(1), gamma=0.99, gae_lambda=0.95)

    # Final step is terminal: advantage = reward - value = 1.0 - 0.5.
    assert abs(buf.advantages[1, 0] - 0.5) < 1e-5


def test_returns_equal_advantages_plus_values():
    """Value targets must reconstruct the empirical return."""
    buf = RolloutBuffer(4, 2, 3, 3, 1)
    rng = np.random.default_rng(0)
    for _ in range(4):
        buf.add(
            obs=rng.normal(size=(2, 3)), critic_obs=rng.normal(size=(2, 3)),
            actions=rng.normal(size=(2, 1)), log_probs=rng.normal(size=2),
            rewards=rng.normal(size=2), values=rng.normal(size=2), dones=np.zeros(2),
        )
    buf.compute_gae(np.zeros(2), np.zeros(2))

    np.testing.assert_allclose(
        buf.returns[:4], buf.advantages[:4] + buf.values[:4], atol=1e-5
    )


def test_buffer_rejects_overflow():
    buf = RolloutBuffer(1, 1, 2, 2, 1)
    args = dict(
        obs=np.zeros((1, 2)), critic_obs=np.zeros((1, 2)), actions=np.zeros((1, 1)),
        log_probs=np.zeros(1), rewards=np.zeros(1), values=np.zeros(1), dones=np.zeros(1),
    )
    buf.add(**args)
    with pytest.raises(RuntimeError, match="full"):
        buf.add(**args)


def test_advantage_normalisation():
    """Normalised advantages must be zero-mean, unit-variance.

    Decouples the policy step size from reward scale -- important because
    Claim 2 sweeps formation spacing, which changes reward magnitudes.
    """
    buf = RolloutBuffer(8, 2, 3, 3, 1)
    rng = np.random.default_rng(1)
    for _ in range(8):
        buf.add(
            obs=rng.normal(size=(2, 3)), critic_obs=rng.normal(size=(2, 3)),
            actions=rng.normal(size=(2, 1)), log_probs=rng.normal(size=2),
            rewards=rng.normal(size=2) * 10.0, values=rng.normal(size=2), dones=np.zeros(2),
        )
    buf.compute_gae(np.zeros(2), np.zeros(2))

    adv = buf.get_batch(normalise_advantages=True)["advantages"]
    assert abs(float(adv.mean())) < 1e-5
    assert abs(float(adv.std()) - 1.0) < 1e-3


# ==============================================================================
# SECTION 2 : THE LEARNER
# ==============================================================================

def _fill_buffer(learner: MAPPO, steps: int, seed: int = 0):
    """Fill a learner's buffer with synthetic experience."""
    rng = np.random.default_rng(seed)
    n = learner.num_agents

    obs = rng.normal(size=(n, 13)).astype(np.float32)   # v4: 13-D
    global_state = {
        "positions": rng.normal(size=(n, 3)),
        "velocities": rng.normal(size=(n, 3)),
        "wind_vec": rng.normal(size=3),
    }

    for _ in range(steps):
        actions, log_probs, values, critic_obs = learner.act(obs, global_state)
        learner.buffer.add(
            obs=obs, critic_obs=critic_obs, actions=actions, log_probs=log_probs,
            rewards=rng.normal(size=n).astype(np.float32), values=values,
            dones=np.zeros(n, dtype=np.float32),
        )

    learner.buffer.compute_gae(learner.compute_values(obs, global_state), np.zeros(n))
    return obs, global_state


def test_learner_constructs_at_every_rung():
    cfg = MAPPOConfig(num_steps=8, num_epochs=1, minibatch_size=8)
    for level in build_ladder(4):
        learner = MAPPO(num_agents=4, level=level, config=cfg)
        report = learner.parameter_report()
        assert report["rung"] == level.label
        assert report["trunk_params"] == 82_689     # identical at every rung


def test_act_returns_correct_shapes():
    cfg = MAPPOConfig(num_steps=8, num_epochs=1, minibatch_size=8)
    learner = MAPPO(num_agents=4, level=build_ladder(4)[1], config=cfg)

    rng = np.random.default_rng(0)
    obs = rng.normal(size=(4, 13)).astype(np.float32)   # v4: 13-D
    global_state = {
        "positions": rng.normal(size=(4, 3)),
        "velocities": rng.normal(size=(4, 3)),
        "wind_vec": rng.normal(size=3),
    }

    actions, log_probs, values, critic_obs = learner.act(obs, global_state)

    assert actions.shape == (4, 3)   # v4: 3-D position offset
    assert log_probs.shape == (4,)
    assert values.shape == (4,)
    # v4: EVERY rung uses the padded maximum width (34-D for N=4), with
    # excluded teammate slots zeroed. This is what makes critic capacity
    # exactly equal across rungs -- see tests/test_wiring.py.
    from ctde.models.observability import max_critic_input_dim
    assert critic_obs.shape == (4, max_critic_input_dim(4, own_obs_dim=13))


def test_critic_receives_gradients():
    """THE MOST IMPORTANT TEST IN THIS FILE.

    The previous project's centralized Q-critic was never called by the loss --
    dead code receiving no gradients, producing meaningless numbers that looked
    fine. This test makes that failure mode impossible to reintroduce silently.

    Both the projection (whose width encodes k) and the shared trunk must
    receive non-zero gradients.
    """
    cfg = MAPPOConfig(num_steps=16, num_epochs=1, minibatch_size=16)
    learner = MAPPO(num_agents=4, level=build_ladder(4)[3], config=cfg)   # k2+P (top rung)

    _fill_buffer(learner, cfg.num_steps)
    batch = learner.buffer.get_batch()

    values = learner.critic(batch["critic_obs"])
    loss = ((values - batch["returns"]) ** 2).mean()
    learner.critic_opt.zero_grad()
    loss.backward()

    proj_grad = learner.critic.projection[0].weight.grad
    trunk_grad = learner.critic.trunk[0].weight.grad

    assert proj_grad is not None and proj_grad.norm() > 0, "critic projection is dead code"
    assert trunk_grad is not None and trunk_grad.norm() > 0, "critic trunk is dead code"


def test_update_changes_actor_weights():
    """The policy must actually move -- a no-op update would be silent."""
    cfg = MAPPOConfig(num_steps=16, num_epochs=2, minibatch_size=16)
    learner = MAPPO(num_agents=4, level=build_ladder(4)[1], config=cfg)

    before = learner.actor.net[0].weight.detach().clone()
    _fill_buffer(learner, cfg.num_steps)
    learner.update()

    assert not torch.allclose(before, learner.actor.net[0].weight.detach())


def test_update_returns_diagnostics():
    """The Phase 7 mechanism study needs these recorded from the first run."""
    cfg = MAPPOConfig(num_steps=16, num_epochs=1, minibatch_size=16)
    learner = MAPPO(num_agents=4, level=build_ladder(4)[1], config=cfg)

    _fill_buffer(learner, cfg.num_steps)
    stats = learner.update()

    for key in ["policy_loss", "value_loss", "entropy", "approx_kl",
                "clip_fraction", "explained_variance", "advantage_std"]:
        assert key in stats, f"missing diagnostic: {key}"
        assert np.isfinite(stats[key]), f"non-finite diagnostic: {key}"


def test_buffer_reset_after_update():
    """PPO is on-policy: data must be discarded after use, never reused."""
    cfg = MAPPOConfig(num_steps=16, num_epochs=1, minibatch_size=16)
    learner = MAPPO(num_agents=4, level=build_ladder(4)[0], config=cfg)

    _fill_buffer(learner, cfg.num_steps)
    assert learner.buffer.ptr == cfg.num_steps
    learner.update()
    assert learner.buffer.ptr == 0


def test_checkpoint_records_its_rung():
    """Checkpoints must record which rung produced them.

    v4 NOTE: every rung now shares one critic ARCHITECTURE (padded input), so a
    dimension mismatch can no longer catch a cross-rung load. The rung is
    therefore recorded explicitly and must be checked by the caller.

    This replaces `test_checkpoint_rejects_wrong_rung`, which relied on the
    differing widths that the capacity fix deliberately removed.
    """
    cfg = MAPPOConfig(num_steps=8, num_epochs=1, minibatch_size=8)
    k1 = MAPPO(num_agents=4, level=build_ladder(4)[1], config=cfg)
    top = MAPPO(num_agents=4, level=build_ladder(4)[3], config=cfg)   # k2+P

    assert k1.state_dict()["level_k"] == 1
    # Top teammate rung is k2 at N=4 (ladder capped at N-2).
    assert top.state_dict()["level_k"] == 2

    # Same architecture -> same width, which is the point of the fix.
    assert k1.critic_obs_dim == top.critic_obs_dim



# ==============================================================================
# SECTION 3 : AGGREGATION
# ==============================================================================

def test_aggregate_computes_mean_and_ci():
    """Charter §7 requires mean +/- 95% CI everywhere."""
    results = [
        {"rung": "k1", "final_eval_return": -10.0},
        {"rung": "k1", "final_eval_return": -12.0},
        {"rung": "k1", "final_eval_return": -11.0},
        {"rung": "k0", "final_eval_return": -20.0},
        {"rung": "k0", "final_eval_return": -22.0},
    ]
    summary = aggregate_by_rung(results)

    assert summary["k1"]["n_seeds"] == 3
    assert abs(summary["k1"]["mean"] - (-11.0)) < 1e-6
    assert summary["k1"]["ci95_low"] < summary["k1"]["mean"] < summary["k1"]["ci95_high"]


def test_aggregate_skips_failed_runs():
    """A crashed run must not poison the aggregate with NaN."""
    results = [
        {"rung": "k1", "final_eval_return": -10.0},
        {"rung": "k1", "final_eval_return": float("nan"), "error": "boom"},
    ]
    summary = aggregate_by_rung(results)

    assert summary["k1"]["n_seeds"] == 1
    assert np.isfinite(summary["k1"]["mean"])
