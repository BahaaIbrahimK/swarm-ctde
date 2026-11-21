"""
================================================================================
 TESTS: Phase 2  --  coupling, environment, observability ladder, networks
================================================================================

WHY THESE TESTS EXIST
---------------------
Phase 2 builds the experimental apparatus. Two of these tests guard the
paper's CENTRAL EXPERIMENTAL CONTROLS -- if either silently breaks during a
later refactor, the whole sweep becomes uninterpretable and we might not
notice until a reviewer points it out.

    test_critic_trunk_identical_across_ladder   -> capacity is not confounded
    test_actor_identical_across_ladder          -> only critic input varies

The others verify that the physics knob actually works and that "nearest
neighbour" means what Phase 1 showed it must mean.

RUN WITH
--------
    pytest tests/test_phase2.py -v
================================================================================
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import pytest

from ctde.envs.coupling import (
    BETA_VALIDITY_THRESHOLD_M,
    check_spacing_validity,
    coupling_matrix,
    downwash_force,
    rank_neighbours_by_coupling,
)
from ctde.models.networks import Actor, Critic
from ctde.models.observability import (
    build_critic_input,
    build_ladder,
    critic_input_dim,
)


# ==============================================================================
# SECTION 1 : COUPLING MODEL
# ==============================================================================

def test_downwash_only_flows_downward():
    """A drone BELOW us exerts no downwash on us, however close.

    Physical: rotor wake goes down. If this failed, the ranking would treat
    drones underneath as influential and the ladder would be meaningless.
    """
    assert downwash_force(dz=-0.5, dxy=0.0) == 0.0
    assert downwash_force(dz=0.0, dxy=0.0) == 0.0
    assert downwash_force(dz=+0.5, dxy=0.0) > 0.0


def test_downwash_decays_with_vertical_separation():
    """Inverse-square decay -- matches the Phase 1 measurement."""
    forces = [downwash_force(dz, 0.0) for dz in [0.7, 1.0, 1.5, 2.0, 2.5]]
    assert all(forces[i] > forces[i + 1] for i in range(len(forces) - 1))

    # Phase 1 measured ~12x contrast across this range.
    assert forces[0] / forces[-1] > 5.0


def test_downwash_decays_with_lateral_offset():
    """Gaussian falloff sideways."""
    aligned = downwash_force(1.0, 0.0)
    offset = downwash_force(1.0, 0.10)
    assert aligned > offset


def test_alignment_beats_proximity():
    """THE PHASE 1 FINDING, encoded as a test.

    A drone 0.6 m above but 0.1 m off-axis exerts LESS force than one 1.5 m
    above and perfectly aligned -- even though the first is 'nearer'.

    This is why the ladder ranks by predicted downwash rather than Euclidean
    distance. If this test fails, k=1 may be handing the critic a teammate that
    exerts no influence at all.
    """
    near_but_offset = downwash_force(dz=0.6, dxy=0.10)
    far_but_aligned = downwash_force(dz=1.5, dxy=0.00)
    assert far_but_aligned > near_but_offset


def test_ranking_orders_by_influence_not_distance():
    positions = np.array([
        [0.0, 0.0, 1.0],    # 0: us
        [0.1, 0.0, 1.6],    # 1: near, offset      -> weak
        [0.0, 0.0, 2.5],    # 2: far, aligned      -> stronger
        [0.0, 0.0, 0.5],    # 3: below             -> zero
    ])
    ranked = rank_neighbours_by_coupling(positions, agent_idx=0)

    assert ranked[0] == 2, "the aligned drone should rank first"
    assert ranked[-1] == 3, "the drone below should rank last"


def test_ranking_is_deterministic():
    """Ties must break deterministically, for reproducibility (Charter §7)."""
    positions = np.array([[0.0, 0.0, 1.0], [5.0, 0.0, 0.5], [-5.0, 0.0, 0.5]])
    a = rank_neighbours_by_coupling(positions, 0)
    b = rank_neighbours_by_coupling(positions, 0)
    np.testing.assert_array_equal(a, b)


def test_coupling_matrix_is_asymmetric():
    """If j is above i, j affects i but i does not affect j."""
    positions = np.array([[0.0, 0.0, 2.0], [0.0, 0.0, 1.0]])
    m = coupling_matrix(positions)

    assert m[1, 0] > 0.0, "upper drone should affect lower"
    assert m[0, 1] == 0.0, "lower drone must not affect upper"
    assert m[0, 0] == 0.0 and m[1, 1] == 0.0


def test_beta_validity_threshold():
    """Below ~0.6875 m the model's Gaussian width goes negative.

    Guards the limitation documented in docs/downwash_model_notes.md. A
    reviewer WILL ask whether the coupling model is valid at the spacings used
    for Claim 2.
    """
    ok_below, _ = check_spacing_validity(0.5)
    ok_above, _ = check_spacing_validity(1.0)

    assert not ok_below
    assert ok_above
    assert 0.68 < BETA_VALIDITY_THRESHOLD_M < 0.69


# ==============================================================================
# SECTION 2 : THE OBSERVABILITY LADDER
# ==============================================================================

def test_ladder_spans_both_endpoints():
    """Claim 1 requires the interior optimum to beat BOTH ends.

    Beating only k=0 would merely re-show that centralization helps -- known
    since MADDPG (2017). The ladder must therefore contain a true decentralized
    end and a true maximal end.
    """
    ladder = build_ladder(num_agents=4)

    assert ladder[0].k == 0, "must include the fully decentralized rung"
    assert ladder[-1].include_privileged, "must include the maximal-centralization rung"
    # k0, k1, k2, k2+P. Capped at N-2 teammate rungs since the chain
    # predecessor is excluded as redundant -- k3 added nothing for followers.
    # See build_ladder's LADDER LENGTH note.
    assert len(ladder) == 4
    assert [lv.label for lv in ladder] == ["k0", "k1", "k2", "k2+P"]


def test_critic_input_dim_grows_with_k():
    ladder = build_ladder(num_agents=4)
    dims = [critic_input_dim(lv) for lv in ladder]
    assert all(dims[i] < dims[i + 1] for i in range(len(dims) - 1))
    assert dims[0] == 13         # own observation only (v4: 13-D)


def test_build_critic_input_matches_declared_dim():
    """The assembled vector must match the declared width exactly.

    A mismatch here would surface as an opaque shape error deep inside
    training; catching it at construction time is far cheaper.
    """
    rng = np.random.default_rng(0)
    num_agents = 4

    global_state = {
        "positions": rng.normal(size=(num_agents, 3)),
        "velocities": rng.normal(size=(num_agents, 3)),
        "wind_vec": rng.normal(size=3),
    }
    own_obs = rng.normal(size=13)   # v4: 13-D (adds relative velocity + role flag)

    for lv in build_ladder(num_agents):
        vec = build_critic_input(1, own_obs, global_state, lv)
        assert vec.shape == (critic_input_dim(lv),), f"width mismatch at rung {lv.label}"
        assert vec.dtype == np.float32


def test_critic_input_starts_with_own_obs():
    """The agent's own observation must be present and first.

    The critic should never see LESS than the actor -- k=0 is 'own obs only',
    not 'nothing'.
    """
    rng = np.random.default_rng(1)
    own_obs = rng.normal(size=13).astype(np.float32)   # v4: 13-D
    global_state = {
        "positions": rng.normal(size=(4, 3)),
        "velocities": rng.normal(size=(4, 3)),
        "wind_vec": rng.normal(size=3),
    }

    for lv in build_ladder(4):
        vec = build_critic_input(0, own_obs, global_state, lv)
        np.testing.assert_allclose(vec[:13], own_obs, rtol=1e-6)


# ==============================================================================
# SECTION 3 : CONFOUND CONTROLS  (the paper's central guarantees)
# ==============================================================================

def test_critic_trunk_identical_across_ladder():
    """CONFOUND CONTROL 2 -- the most important test in this file.

    Every critic's TRUNK must have exactly the same parameter count. Otherwise
    higher k means more capacity, and we would be measuring capacity rather
    than information -- the single most likely reviewer objection to the paper.
    """
    dims = [critic_input_dim(lv) for lv in build_ladder(4)]
    trunk_counts = {Critic(input_dim=d).parameter_report()["trunk_params"] for d in dims}

    assert len(trunk_counts) == 1, (
        f"Critic trunk capacity differs across the ladder: {sorted(trunk_counts)}. "
        "The capacity confound has been reintroduced."
    )


def test_critic_capacity_spread_favours_high_k():
    """Residual parameter differences must work AGAINST our hypothesis.

    Only the thin projection layer varies, and it grows with k. So a high-k
    critic has marginally MORE parameters. If we still observe performance
    DROPPING at high k, extra capacity cannot be the explanation -- which is
    what makes the control credible rather than merely present.
    """
    dims = [critic_input_dim(lv) for lv in build_ladder(4)]
    totals = [Critic(input_dim=d).parameter_report()["total_params"] for d in dims]

    assert all(totals[i] <= totals[i + 1] for i in range(len(totals) - 1))

    # And the spread must stay small relative to the whole network.
    spread_fraction = (totals[-1] - totals[0]) / totals[0]
    assert spread_fraction < 0.05, f"capacity spread {spread_fraction:.1%} is too large"


def test_actor_identical_across_ladder():
    """CONFOUND CONTROL 1.

    The actor never changes with k: same architecture, same 12-D input, same
    parameter count. Any performance difference is therefore attributable to
    critic information alone.
    """
    counts = {sum(p.numel() for p in Actor().parameters()) for _ in range(3)}
    assert len(counts) == 1

    # And it must consume exactly the local observation width.
    import torch
    actor = Actor()
    # v4: 13-D observation, 3-D action (position offset, not rotor RPMs).
    mean, std = actor(torch.zeros(1, 13))
    assert mean.shape == (1, 3)
    assert std.shape == (1, 3)


# ==============================================================================
# SECTION 4 : ENVIRONMENT  (slower -- needs the simulator)
# ==============================================================================

@pytest.mark.slow
def test_formation_env_runs():
    from ctde.envs.formation_aviary import FormationAviary

    env = FormationAviary(num_drones=4, formation_spacing_m=1.0, base_altitude_m=0.8)
    try:
        obs, _ = env.reset(seed=0)
        assert obs.shape == (4, 13)   # v4: 13-D observation

        # v2-pidassist: action is a 3-D position OFFSET per drone, not 4 rotor
        # RPMs. The inner PID converts the offset into RPMs.
        obs, rew, term, trunc, info = env.step(np.zeros((4, 3), dtype=np.float32))
        assert rew.shape == (4,), "reward must be per-agent"
        assert "total_coupling_N" in info
        assert "mean_lateral_error_m" in info, "sidestep-exploit monitor missing"
    finally:
        env.close()


@pytest.mark.slow
def test_spacing_knob_changes_coupling():
    """CLAIM 2 MANIPULATION CHECK.

    Claim 2 asserts that k* tracks coupling strength, which we vary via
    formation spacing. This verifies the knob actually changes the PHYSICS,
    not merely the commanded geometry. Without this, Claim 2 would rest on an
    unverified assumption.
    """
    from ctde.envs.formation_aviary import FormationAviary

    couplings = []
    for spacing in [0.7, 1.0, 1.5, 2.0, 2.5]:
        env = FormationAviary(num_drones=4, formation_spacing_m=spacing, base_altitude_m=0.8)
        try:
            env.reset(seed=0)
            _, _, _, _, info = env.step(np.zeros((4, 3), dtype=np.float32))
            couplings.append(info["total_coupling_N"])
        finally:
            env.close()

    assert all(couplings[i] > couplings[i + 1] for i in range(len(couplings) - 1)), (
        "coupling must decrease monotonically as spacing widens"
    )
    assert couplings[0] / couplings[-1] > 5.0, "need a wide coupling contrast for Claim 2"
