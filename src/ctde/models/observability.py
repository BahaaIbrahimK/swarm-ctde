"""
================================================================================
 THE OBSERVABILITY LADDER  --  the independent variable of the paper
================================================================================

PHASE 2 (Charter §5). Implements Charter §4.3.

--------------------------------------------------------------------------------
 WHAT THIS IS
--------------------------------------------------------------------------------
`k` controls HOW MUCH the centralized critic sees. It is the one thing we vary.

    k = 0        own observation only            (independent learner)
    k = 1        own + 1 most-coupled teammate   (minimal centralization)
    k = 2        own + 2 most-coupled teammates
    ...
    k = N-1      own + ALL teammates             (standard CTDE -- the incumbent)
    k = N-1+P    all teammates + privileged sim state (wind, true coupling)

The field's default is "use k = N-1 or higher; more information is never
harmful". Claim 1 says performance actually PEAKS at an interior k.

--------------------------------------------------------------------------------
 THE THREE CONFOUND CONTROLS  (Charter §4.3 -- non-negotiable)
--------------------------------------------------------------------------------
Without these the sweep is uninterpretable and a reviewer will say so.

 1. THE ACTOR IS BYTE-IDENTICAL ACROSS ALL k.
    Same architecture, same input (12-D local obs), same parameter count. Only
    the CRITIC's input set changes. Therefore any performance difference is
    attributable to critic information alone -- nothing else varied.

 2. CRITIC PARAMETER COUNT IS HELD CONSTANT ACROSS k.
    Naively, larger k means a wider critic input, hence more first-layer
    weights. Then a k=3 critic would have more capacity than a k=1 critic, and
    we would be measuring CAPACITY, not INFORMATION.

    THIS IS THE MOST LIKELY REVIEWER OBJECTION TO THE WHOLE PAPER.

    Fix: a fixed-width projection layer. Every k projects its variable-width
    input to the SAME hidden dimension before the shared trunk. Only the
    projection's input width differs; the trunk is identical. We report exact
    parameter counts per k in the paper.

 3. NEIGHBOUR ENCODING IS PERMUTATION-INVARIANT AND PHYSICALLY ORDERED.
    Teammates are ordered by PREDICTED DOWNWASH INFLUENCE (see envs/coupling.py),
    not by agent index and not by Euclidean distance. Phase 1 showed why: a
    drone 0.6 m above but 0.1 m off-axis exerts ~0 N, while one 1.5 m above and
    aligned exerts 0.034 N. Euclidean ranking would put an irrelevant drone at
    k=1 and silently corrupt the ladder.

    So k varies INFORMATION CONTENT, not encoding difficulty or arbitrary
    ordering.
================================================================================
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ctde.envs.coupling import rank_neighbours_by_coupling


# ==============================================================================
# WHICH TEAMMATES COUNT AS "NEW INFORMATION"
# ==============================================================================
# REDEFINED 17 Sep 2026, after an external review and a rank measurement.
#
# THE DEFECT. In a vertical chain the most-coupled teammate IS the predecessor,
# and the actor's own observation already contains the gap to and relative
# velocity of that predecessor:
#
#     e_i = p_i - (p_{i-1} - Delta)   ->   p_{i-1} - p_i = Delta - e_i
#     edot_i = v_i - v_{i-1}          ->   v_{i-1} - v_i = -edot_i
#
# (Verified numerically: the k=1 teammate's relative velocity was bit-identical
# to the negated obs[6:9] for every follower.)
#
# So the FIRST teammate slot was spending itself on information the critic
# already had. Dimensions added beyond the actor's observation, measured:
#
#     rung   agent0  agent1  agent2  agent3
#     k1        6       0       6       6     <- agent 1 gained NOTHING
#
# Agent 1 was fully degenerate because its predecessor is the leader, whose
# motion the actor already tracks analytically. Agents 2-3 were not, because
# their predecessors are gust-perturbed. The ladder was therefore INCONSISTENT
# ACROSS AGENTS: `k` denoted a different amount of added information depending
# on which drone you were -- fatal for a controlled experiment whose single
# independent variable is supposed to be `k`.
#
# THE FIX. Exclude the chain predecessor when counting teammate slots, so every
# rung adds genuinely novel information for every agent:
#
#     rung   agent0  agent1  agent2  agent3
#     k1        6       6       6       6
#     k2       12      12      12      12
#
# NOTHING IS LOST AT THE TOP RUNG. Measured: giving a follower ALL teammates
# adds 12 dimensions; giving it all-except-predecessor also adds 12. The
# predecessor block is pure redundancy, so the maximal rung still conveys the
# full team state and Claim 1's endpoint test is unaffected.
#
# The leader has no predecessor, so nothing is excluded for it.

def _informative_neighbours(positions: np.ndarray, agent_idx: int) -> np.ndarray:
    """Teammates ranked by coupling, EXCLUDING the already-sensed predecessor.

    See the block comment above for why the predecessor is excluded and for the
    measurement showing no information is lost by doing so.
    """
    ranked = rank_neighbours_by_coupling(positions, agent_idx)
    predecessor = agent_idx - 1
    if predecessor < 0:
        return ranked
    return np.array([o for o in ranked if o != predecessor], dtype=ranked.dtype)


# ==============================================================================
# SECTION 1 : LADDER SPECIFICATION
# ==============================================================================

@dataclass(frozen=True)
class ObservabilityLevel:
    """One rung of the ladder.

    Attributes
    ----------
    k:
        Number of teammates the critic sees (ordered by coupling strength).
    include_privileged:
        Whether the critic additionally receives privileged simulator state
        (wind vector, true coupling forces) that NO real drone could ever
        measure. This is the top rung, `k = N-1+P`, representing the
        "asymmetric actor-critic" style of maximal centralization.
    label:
        Short name used in plots and result directories.
    """

    k: int
    include_privileged: bool = False

    @property
    def label(self) -> str:
        return f"k{self.k}+P" if self.include_privileged else f"k{self.k}"


def build_ladder(num_agents: int, include_privileged_rung: bool = True) -> list[ObservabilityLevel]:
    """Construct the full ladder for a team of `num_agents`.

    Example, num_agents=4:
        k0, k1, k2, k3, k3+P     -> 5 rungs

    The endpoints matter for Claim 1's significance test: the interior optimum
    must beat BOTH k=0 (decentralized critic) and the maximal rung. Beating only
    k=0 would merely re-show that centralization helps -- known since 2017.

    LADDER LENGTH  (corrected 17 Sep 2026)
    --------------------------------------
    The top rung used to be k = N-1, "sees every teammate". Since the chain
    predecessor is now excluded as redundant (see `_informative_neighbours`), a
    FOLLOWER only has N-2 informative teammates -- so k = N-1 gave it nothing
    that k = N-2 had not already given.

    Measured dimensions added beyond the actor's own observation, N=4:

        rung    ag0   ag1   ag2   ag3
        k2       12    12    12    12
        k3       18    12    12    12     <- followers gain NOTHING

    A duplicate rung is not harmless: it would consume a fifth of the sweep
    budget measuring the same condition twice, and an "interior optimum" that
    happened to sit between two identical rungs would be an artefact.

    The ladder is therefore capped at N-2 teammate rungs, and the privileged
    rung sits on top of that maximum. For N=4 this gives k0, k1, k2, k2+P.

    NOTE ON TEAM SIZE. N=4 yields only three information levels (0, 6, 12
    dimensions), which is a thin ladder for detecting an INTERIOR optimum --
    there is exactly one interior point. N=6 gives k0..k4 all distinct
    (0/6/12/18/24) and three interior points. If Claim 1 is to be tested
    seriously, N=6 is the defensible team size.
    """
    # Followers have N-2 informative teammates (the predecessor is excluded);
    # the leader has N-1. Cap at the smaller so no rung is a duplicate.
    max_informative = max(num_agents - 2, 0)

    levels = [ObservabilityLevel(k=k) for k in range(max_informative + 1)]

    if include_privileged_rung:
        levels.append(
            ObservabilityLevel(k=max_informative, include_privileged=True)
        )

    return levels


# ==============================================================================
# SECTION 2 : CRITIC INPUT CONSTRUCTION
# ==============================================================================

def critic_input_dim(
    level: ObservabilityLevel,
    own_obs_dim: int = 13,
    teammate_feat_dim: int = 6,
    privileged_dim: int = 3,
) -> int:
    """Input width of the critic at this rung.

    Composition
    -----------
        own_obs                     own_obs_dim              (12)
      + k * teammate features       k * teammate_feat_dim    (k * 6)
      + privileged (optional)       privileged_dim           (3, wind vector)

    Teammate features are 6-D (relative position 3 + relative velocity 3)
    rather than the full 12-D observation. Rationale: relative quantities are
    what matter for coupling, and they keep the per-teammate cost low so that
    the ESTIMATION-COST force in the §1.5 mechanism does not swamp the
    variance-reduction force before we can observe the trade-off.
    """
    dim = own_obs_dim + level.k * teammate_feat_dim
    if level.include_privileged:
        dim += privileged_dim
    return dim


def max_critic_input_dim(
    num_agents: int,
    own_obs_dim: int = 13,
    teammate_feat_dim: int = 6,
    privileged_dim: int = 3,
) -> int:
    """Width of the PADDED critic input -- identical at every rung.

    EXACT CAPACITY CONTROL (added 16 Sep 2026).

    The previous scheme gave each rung its own projection layer, sized to that
    rung's input. Parameter counts therefore differed: 832 at k0 rising to
    2,176 at k3+P.

    The old defence was that the extra parameters FAVOUR high k, so they cannot
    explain a decline at high k. That is true but only half the argument: it
    does NOT explain away k=1 beating k0, which is the ascending side of the
    claimed interior optimum -- precisely the part the paper needs.

    The fix is to give every critic the SAME maximum-width input and zero the
    slots a given rung excludes. The network is then literally the same object
    at every rung; only the information content of its input changes.

    SLOT COUNT  (corrected 17 Sep 2026)
    -----------------------------------
    This reserved N-1 teammate slots, matching the old top rung k = N-1. The
    ladder is now capped at N-2 (the chain predecessor is excluded as
    redundant), so one slot could never be filled at any rung.

    Six permanently-zero inputs would not have broken capacity equality -- every
    rung shared them -- but they are dead width in every critic, and a
    "max width" that no rung can reach invites exactly the kind of quiet
    drift between two definitions that this project has been bitten by. The
    reservation now matches what the ladder can actually use.
    """
    max_informative = max(num_agents - 2, 0)
    return own_obs_dim + max_informative * teammate_feat_dim + privileged_dim


def build_critic_input(
    agent_idx: int,
    own_obs: np.ndarray,
    global_state: dict,
    level: ObservabilityLevel,
    padded: bool = False,
    num_agents: int | None = None,
) -> np.ndarray:
    """Assemble the critic's input vector for one agent at one rung.

    Parameters
    ----------
    agent_idx:
        Whose critic we are building input for.
    own_obs:
        That agent's 12-D local observation -- IDENTICAL to what its actor sees.
    global_state:
        From `FormationAviary.get_global_state()`.
    level:
        The ladder rung.

    Returns
    -------
    np.ndarray
        1-D float32 vector of length `critic_input_dim(level)`.

    Notes
    -----
    Teammates are selected by COUPLING RANK, so k=1 means "the teammate whose
    wake affects me most" -- not "agent 1", and not "the Euclidean-nearest".
    """
    positions = global_state["positions"]
    velocities = global_state["velocities"]

    if padded:
        return _build_padded_critic_input(
            agent_idx, own_obs, global_state, level,
            num_agents if num_agents is not None else positions.shape[0],
        )

    parts: list[np.ndarray] = [np.asarray(own_obs, dtype=np.float32).ravel()]

    if level.k > 0:
        # Rank teammates by aerodynamic influence, EXCLUDING the predecessor the
        # actor already senses -- see _informative_neighbours.
        ranked = _informative_neighbours(positions, agent_idx)

        own_pos = positions[agent_idx]
        own_vel = velocities[agent_idx]

        # Take the k most-coupled. If the team is smaller than k+1, pad with
        # zeros so the input width stays fixed (required for a fixed critic
        # architecture).
        for slot in range(level.k):
            if slot < len(ranked):
                other = ranked[slot]
                rel_pos = positions[other] - own_pos
                rel_vel = velocities[other] - own_vel
                parts.append(np.concatenate([rel_pos, rel_vel]).astype(np.float32))
            else:
                parts.append(np.zeros(6, dtype=np.float32))

    if level.include_privileged:
        # The wind vector: a hidden common cause acting on every drone. No real
        # drone can measure this. Including it is the maximal-centralization
        # rung -- and, per Claim 4, the most likely to overfit the simulator.
        # The TRUE disturbance this agent feels: steady mean PLUS its own gust.
        #
        # This branch used to append `wind_vec` -- the single GLOBAL mean wind --
        # while each drone actually experiences a different OU gust. The padded
        # builder was corrected in September; this unpadded one was not, so the
        # two paths disagreed about what "privileged" means. MAPPO uses the
        # padded path, so training was right and only diagnostics read the stale
        # value, but a silent disagreement between two builders of the same
        # vector is exactly the kind of drift that produces unexplainable
        # results later.
        if "air_velocity" in global_state:
            air = np.asarray(global_state["air_velocity"], dtype=np.float32)
            parts.append((air[agent_idx] if air.ndim == 2 else air).ravel())
        else:
            parts.append(np.asarray(global_state["wind_vec"], dtype=np.float32).ravel())

    return np.concatenate(parts).astype(np.float32)


# ==============================================================================
# SECTION 3 : SELF-CHECK
# ==============================================================================

def describe_ladder(num_agents: int, include_privileged_rung: bool = True) -> str:
    """Human-readable ladder summary, for logging at the start of a sweep.

    Printing this before training makes the independent variable explicit in
    every log -- useful when reading results months later.
    """
    levels = build_ladder(num_agents, include_privileged_rung)

    lines = [
        f"OBSERVABILITY LADDER  (num_agents={num_agents})",
        "-" * 62,
        f"  {'rung':<8} {'teammates':<11} {'privileged':<12} {'critic input dim':<16}",
        "-" * 62,
    ]

    for lv in levels:
        lines.append(
            f"  {lv.label:<8} {lv.k:<11} {str(lv.include_privileged):<12} "
            f"{critic_input_dim(lv):<16}"
        )

    lines += [
        "-" * 62,
        "  actor input dim : 12  (IDENTICAL at every rung -- confound control 1)",
        "  teammate order  : by predicted downwash force, not Euclidean distance",
    ]
    return "\n".join(lines)


def _build_padded_critic_input(
    agent_idx: int,
    own_obs: np.ndarray,
    global_state: dict,
    level: ObservabilityLevel,
    num_agents: int,
    own_obs_dim: int = 13,
    teammate_feat_dim: int = 6,
    privileged_dim: int = 3,
) -> np.ndarray:
    """Fixed-width critic input with excluded slots zeroed.

    Layout (always the same width, regardless of rung):

        [ own_obs (12) |
          teammate slot 1 (6) | ... | teammate slot N-1 (6) |
          privileged (3) ]

    A rung with observability k fills the first k teammate slots -- ordered by
    predicted downwash influence -- and leaves the rest at zero. The privileged
    block carries the ACTUAL air velocity (steady mean + stochastic gust) and is
    zeroed unless the rung includes it.

    WHY ZEROING RATHER THAN OMITTING: it makes the critic architecture literally
    identical at every rung, so a reviewer cannot attribute any difference to
    network capacity. See `max_critic_input_dim` for the history.
    """
    positions = global_state["positions"]
    velocities = global_state["velocities"]

    total = max_critic_input_dim(num_agents, own_obs_dim, teammate_feat_dim, privileged_dim)
    vec = np.zeros(total, dtype=np.float32)

    vec[:own_obs_dim] = np.asarray(own_obs, dtype=np.float32).ravel()

    if level.k > 0:
        ranked = _informative_neighbours(positions, agent_idx)
        own_pos, own_vel = positions[agent_idx], velocities[agent_idx]

        for slot in range(min(level.k, len(ranked))):
            other = ranked[slot]
            start = own_obs_dim + slot * teammate_feat_dim
            vec[start:start + 3] = positions[other] - own_pos
            vec[start + 3:start + 6] = velocities[other] - own_vel

    if level.include_privileged:
        # The TRUE disturbance: steady mean plus stochastic gust. Using
        # `wind_vec` alone was the bug that made this rung not privileged.
        #
        # The offset must match how max_critic_input_dim() sized this vector:
        # N-2 teammate slots, not N-1. Using N-1 here wrote past the end.
        start = own_obs_dim + max(num_agents - 2, 0) * teammate_feat_dim
        if "air_velocity" in global_state:
            air = np.asarray(global_state["air_velocity"], dtype=np.float32)
            vec[start:start + 3] = air[agent_idx] if air.ndim == 2 else air
        else:
            vec[start:start + 3] = np.asarray(global_state["wind_vec"], dtype=np.float32)

    return vec


# ==============================================================================
# SECTION 4 : LADDER DEGENERACY DIAGNOSTIC  (added 17 Sep 2026)
# ==============================================================================
# An external review observed that for a follower in a vertical chain, the
# most-coupled teammate IS its predecessor -- and the actor's own observation
# already contains the gap to, and relative velocity of, that predecessor.
#
#     e_i      = p_i - (p_{i-1} - Delta)        -> p_{i-1} - p_i = Delta - e_i
#     edot_i   = v_i - v_{i-1}                  -> v_{i-1} - v_i = -edot_i
#
# So the k=1 teammate block can be an affine function of information the critic
# already has, in which case k0 vs k1 does not vary what the critic KNOWS and a
# flat result at that rung would prove nothing.
#
# MEASURED (seed 1234, N=4, chain, sigma=3.0), new dimensions each rung adds
# beyond the agent's own observation:
#
#     rung    agent 0   agent 1   agent 2   agent 3
#     k0         0         0         0         0
#     k1         6         0         6         6      <- agent 1 degenerate
#     k2        12         6        12        12
#     k3        18        12        18        18
#
# Agent 1 at k=1 adds EXACTLY ZERO new dimensions: its predecessor is the
# leader, whose motion is fully determined by the analytical trajectory the
# actor already tracks. Agents 2 and 3 are NOT degenerate, because their
# predecessors are themselves gust-perturbed, so absolute teammate state carries
# information the purely relative observation does not.
#
# The ladder is therefore not uniformly broken -- it is INCONSISTENT ACROSS
# AGENTS, which for a controlled experiment is arguably worse: "k" denotes a
# different amount of added information depending on which drone you are.
#
# This function exists so that claim is checked by measurement rather than
# argued from geometry, and so any redefinition of the rungs can be validated.

def information_added_by_rung(
    own_obs_samples: np.ndarray,
    critic_input_samples: np.ndarray,
    tol: float = 1e-6,
) -> int:
    """Dimensions in the critic input that are NOT affine functions of own obs.

    Both arguments are (num_samples, dim) arrays gathered from real rollouts.

    Returns the rank of the critic input after projecting out the column space
    of the actor's observation -- i.e. how much genuinely new information the
    rung supplies. Zero means the rung is degenerate for that agent.
    """
    A = np.asarray(own_obs_samples, dtype=float)
    B = np.asarray(critic_input_samples, dtype=float)
    A = A - A.mean(axis=0)
    B = B - B.mean(axis=0)

    Q, _ = np.linalg.qr(A)
    residual = B - Q @ (Q.T @ B)

    sv = np.linalg.svd(residual, compute_uv=False)
    if len(sv) == 0:
        return 0
    return int((sv > tol * max(sv[0], 1.0)).sum())
