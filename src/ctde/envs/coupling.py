"""
================================================================================
 AERODYNAMIC COUPLING  --  ranking teammates by how much they actually matter
================================================================================

WHY THIS FILE EXISTS
--------------------
The observability ladder (Charter §4.3) is defined as:

    k=0  -> critic sees own observation only
    k=1  -> own + NEAREST neighbour
    k=2  -> own + 2 nearest neighbours
    ...

The word "nearest" hides a decision that matters enormously, and Phase 1's
probe is what exposed it.

WHAT PHASE 1 MEASURED (see docs/downwash_model_notes.md)
--------------------------------------------------------
Downwash force is NOT a function of Euclidean distance. It is:

    alpha = DW_COEFF_1 * (PROP_RADIUS / (4*dz))^2        # inverse-square in dz
    beta  = DW_COEFF_2 * dz + DW_COEFF_3                 # Gaussian width
    F     = alpha * exp(-0.5 * (dxy/beta)^2)             # lateral falloff

Three consequences, all measured in the Phase 1 sweep:

  1. ONLY DRONES ABOVE MATTER. Downwash flows downward. A drone below you
     exerts zero influence, no matter how close.

  2. LATERAL ALIGNMENT DOMINATES AT CLOSE RANGE. At dz=0.60 m the coupling
     collapses to ~0 by just 5 cm of lateral offset. At dz=2.50 m it stays
     nearly flat across the whole offset range -- close wakes are NARROW and
     INTENSE, distant wakes are BROAD and WEAK.

  3. THEREFORE EUCLIDEAN DISTANCE IS THE WRONG RANKING. A teammate 0.6 m below
     but 0.1 m to the side is aerodynamically irrelevant, while one 1.5 m below
     and perfectly aligned matters substantially -- even though the first is
     "nearer".

If we ranked neighbours by Euclidean distance, k=1 could hand the critic a
drone that exerts no influence at all. The ladder would then vary "how many
arbitrary teammates the critic sees" rather than "how much coupling-relevant
information the critic has" -- and Claim 1 would be testing the wrong thing.

WHAT THIS MODULE DOES
---------------------
Ranks teammates by PREDICTED DOWNWASH INFLUENCE, using the same model the
simulator runs. This makes the observability ladder mean:

    k=1  ->  the teammate whose wake affects me most

which is exactly the quantity the §1.5 mechanism argument is about.

A NOTE ON WHY THIS IS NOT CHEATING
----------------------------------
The ranking uses only positions -- information the critic is being given
anyway. It does not inject privileged force values into the actor. It is a
principled ORDERING of the same teammate states, replacing an arbitrary one.
We report it explicitly in the paper as part of the ladder definition.
================================================================================
"""

from __future__ import annotations

import numpy as np


# ==============================================================================
# SECTION 1 : THE DOWNWASH MODEL (mirrors BaseAviary._downwash)
# ==============================================================================
#
# These constants come from the Crazyflie 2.x URDF and are read at runtime from
# the environment. They are duplicated here as defaults so this module can be
# unit-tested without constructing a simulator.
#
# Verified against the installed gym-pybullet-drones on 23 Aug 2026:
#     dw_coeff_1 2267.180000, dw_coeff_2 0.160000, dw_coeff_3 -0.110000
#     prop_radius 0.023135
# ==============================================================================

DEFAULT_DW_COEFF_1 = 2267.18
DEFAULT_DW_COEFF_2 = 0.16
DEFAULT_DW_COEFF_3 = -0.11
DEFAULT_PROP_RADIUS = 0.023135

# Below this vertical separation the Gaussian width `beta` goes NEGATIVE:
#     beta = 0.16*dz - 0.11 = 0  at  dz = 0.6875 m
# Beta is physically a width, so a negative value is meaningless -- the model is
# being extrapolated outside SiQi Zhou's calibration range. It does not crash
# (beta is squared), which makes the failure SILENT. See Charter limitations and
# docs/downwash_model_notes.md.
BETA_VALIDITY_THRESHOLD_M = 0.6875


def downwash_force(
    dz: float,
    dxy: float,
    dw_coeff_1: float = DEFAULT_DW_COEFF_1,
    dw_coeff_2: float = DEFAULT_DW_COEFF_2,
    dw_coeff_3: float = DEFAULT_DW_COEFF_3,
    prop_radius: float = DEFAULT_PROP_RADIUS,
) -> float:
    """Predicted downwash force (N) from a drone `dz` above and `dxy` to the side.

    Parameters
    ----------
    dz:
        Vertical separation, POSITIVE when the other drone is ABOVE.
        Zero or negative returns 0.0 -- downwash only flows downward.
    dxy:
        Lateral (horizontal) separation, always non-negative.

    Returns
    -------
    float
        Force magnitude in newtons. For scale, a Crazyflie weighs 0.265 N.

    Notes
    -----
    Mirrors `BaseAviary._downwash()` exactly, so the ranking agrees with the
    forces the simulator actually applies.
    """
    # Downwash flows DOWN. A drone at or below our altitude exerts nothing.
    if dz <= 0:
        return 0.0

    # Match the simulator's cutoff: it ignores drones more than 10 m away.
    if dxy >= 10.0:
        return 0.0

    alpha = dw_coeff_1 * (prop_radius / (4.0 * dz)) ** 2
    beta = dw_coeff_2 * dz + dw_coeff_3

    # Guard against beta == 0 exactly (dz = 0.6875), which would divide by zero.
    # The simulator has the same latent issue; we return alpha (the dxy=0 value)
    # because as beta -> 0 the Gaussian tends to 1 at dxy=0 and 0 elsewhere.
    if abs(beta) < 1e-9:
        return alpha if dxy < 1e-9 else 0.0

    return float(alpha * np.exp(-0.5 * (dxy / beta) ** 2))


# ==============================================================================
# SECTION 2 : NEIGHBOUR RANKING
# ==============================================================================

def rank_neighbours_by_coupling(
    positions: np.ndarray,
    agent_idx: int,
    **model_kwargs,
) -> np.ndarray:
    """Rank all other agents by how strongly they affect `agent_idx`.

    THIS DEFINES WHAT "NEAREST NEIGHBOUR" MEANS IN THE OBSERVABILITY LADDER.

    Parameters
    ----------
    positions:
        (num_agents, 3) array of XYZ positions.
    agent_idx:
        Index of the agent whose perspective we take.

    Returns
    -------
    np.ndarray
        Indices of the other agents, ordered by DECREASING influence on
        `agent_idx`. Length is num_agents - 1.

    Example
    -------
    >>> pos = np.array([
    ...     [0.0, 0.0, 1.0],   # 0: us
    ...     [0.1, 0.0, 1.6],   # 1: above but 10cm off-axis -> weak
    ...     [0.0, 0.0, 2.5],   # 2: far above but ALIGNED   -> stronger
    ...     [0.0, 0.0, 0.5],   # 3: below                   -> zero
    ... ])
    >>> rank_neighbours_by_coupling(pos, 0)
    array([2, 1, 3])
    """
    num_agents = positions.shape[0]
    own_pos = positions[agent_idx]

    influences: list[tuple[int, float]] = []

    for other in range(num_agents):
        if other == agent_idx:
            continue

        # dz > 0 means the OTHER drone is above us, so its wake reaches us.
        dz = float(positions[other, 2] - own_pos[2])
        dxy = float(np.linalg.norm(positions[other, 0:2] - own_pos[0:2]))

        influences.append((other, downwash_force(dz, dxy, **model_kwargs)))

    # Sort by influence descending. Ties (e.g. all-zero when no drone is above)
    # break by index, which keeps the ordering deterministic -- important for
    # reproducibility (Charter §7).
    influences.sort(key=lambda pair: (-pair[1], pair[0]))

    return np.array([idx for idx, _ in influences], dtype=np.int64)


def coupling_matrix(positions: np.ndarray, **model_kwargs) -> np.ndarray:
    """Full pairwise coupling matrix.

    Returns
    -------
    np.ndarray
        (n, n) array where entry [i, j] is the force drone j exerts on drone i.
        The diagonal is zero. The matrix is ASYMMETRIC by nature: if j is above
        i, then j affects i but i does not affect j.

    Used for logging the realised coupling strength during training, which lets
    us verify that the formation spacing knob (Claim 2) actually changed the
    physics rather than only the commanded geometry.
    """
    n = positions.shape[0]
    matrix = np.zeros((n, n), dtype=np.float64)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            dz = float(positions[j, 2] - positions[i, 2])
            dxy = float(np.linalg.norm(positions[j, 0:2] - positions[i, 0:2]))
            matrix[i, j] = downwash_force(dz, dxy, **model_kwargs)

    return matrix


# ==============================================================================
# SECTION 3 : VALIDITY CHECKING
# ==============================================================================

def check_spacing_validity(vertical_separation_m: float) -> tuple[bool, str]:
    """Check whether a spacing lies inside the downwash model's valid domain.

    WHY THIS MATTERS FOR THE PAPER
    ------------------------------
    Claim 2 asserts that k* tracks coupling strength, which we vary via
    formation spacing. A reviewer will therefore ask whether the coupling model
    is trustworthy at the spacings used.

    Below dz = 0.6875 m the model's Gaussian width goes negative -- physically
    meaningless, and silent because beta is squared. We must either stay above
    that threshold or disclose the extrapolation explicitly.

    Returns
    -------
    (is_valid, message)
    """
    if vertical_separation_m >= BETA_VALIDITY_THRESHOLD_M:
        return True, f"{vertical_separation_m:.3f} m is inside the model's valid range."

    beta = DEFAULT_DW_COEFF_2 * vertical_separation_m + DEFAULT_DW_COEFF_3
    return False, (
        f"{vertical_separation_m:.3f} m is BELOW the validity threshold "
        f"({BETA_VALIDITY_THRESHOLD_M:.4f} m). The Gaussian width beta = {beta:+.4f} "
        "is negative, so the downwash model is being extrapolated outside its "
        "calibration range. This does not crash (beta is squared) but must be "
        "disclosed in the paper's limitations. See docs/downwash_model_notes.md."
    )
