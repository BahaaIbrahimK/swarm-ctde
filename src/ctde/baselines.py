"""
================================================================================
 REFERENCE CONTROLLERS  --  what the learned policy must beat
================================================================================

Charter §7 requires results to be judged against something meaningful. Return
alone cannot do that: it is unitless, its scale has changed three times, and
"the policy improved 4%" says nothing about whether the policy is any good.

--------------------------------------------------------------------------------
 WHY "BETTER THAN NOTHING" IS NOT A BASELINE
--------------------------------------------------------------------------------
The obvious reference is the pure PID (action = 0). It is too weak. Measured at
sigma = 3.0 on held-out gust seeds (16 Sep 2026):

    pure PID (action = 0)            return 495.8    RMS 18.00 cm
    constant offset 0.05             return 485.8    RMS 18.39 cm
    random noise at policy init std  return 481.3    RMS 18.56 cm

Everything small scores the same, because the reward landscape is FLAT near
zero. A policy could sit at the PID baseline forever and look like it had
"learned" relative to random.

The proportional controller below does not:

    P-control on gap error, gain 3   return 604.8    RMS 12.92 cm

That is +109 return and 28% lower tracking error than the PID, from a
controller with ONE tunable number. It is the honest bar.

--------------------------------------------------------------------------------
 SUPERSEDED 17 Sep 2026 -- READ THIS BEFORE USING THE NUMBERS ABOVE
--------------------------------------------------------------------------------
The measurements above were taken while the inner PID was given target_pos ONLY,
so its desired velocity defaulted to ZERO -- while the leader tracked a moving
circle and every follower tracked a moving predecessor. The baseline was
handicapped by a missing feedforward term, not by any limitation of classical
control.

With `target_vel` passed (the leader's trajectory derivative; each follower's
measured predecessor velocity), on common episodes 900000-900007:

    PID + feedforward (action = 0)     return 614.7    RMS 12.16 cm   crash 0/8
    P-control (gain 3) on top          return -561.8   RMS 11.60 cm   crash 8/8

The unaided PID now MATCHES what the P-controller previously achieved, and
adding the P term on top crashes every episode. Re-tuning the gain confirms it:

    gain   0.00   0.10   0.25   0.50   1.00   2.00   3.00
    return 602.5  550.2  455.1  286.1   62.7 -471.3 -559.1
    crash   0/6    1/6    1/6    2/6    5/6    6/6    6/6

THE BEST STABLE GAIN IS ZERO. The P-controller's entire advantage was
substituting proportional gain for the velocity feedback the controller should
have had.

WHAT THIS INVALIDATES. The conclusion that "a trivial controller beats RL, so
the learner has an exploration problem" does not hold: the comparison was
against a deliberately handicapped baseline. The real bar for the learned policy
is the PID WITH feedforward, and that bar is considerably higher.

--------------------------------------------------------------------------------
 WHAT IT PROVES, AND WHAT IT THREATENS  (historical -- pre-feedforward)
--------------------------------------------------------------------------------
PROVES: the task is solvable, the solution is expressible in the 3-D offset
action space, and it is reachable from the 13-D observation. When a learned
policy fails to improve, the task is not at fault.

THREATENS: if NO rung of the observability ladder beats this controller, then
the honest conclusion is that this task does not require a learned policy at
all. That must be reported rather than buried -- it is kill-criterion K1 applied
to the task rather than to k.

Note the useful gain band is narrow: gain 3 completes 5/5 episodes, gain 8
crashes 5/5. The landscape is flat near zero and sharply punishing further out,
which is exactly why exploration from a zero-initialised policy is hard here.
================================================================================
"""

from __future__ import annotations

import numpy as np

# RE-TUNED 17 Sep 2026 against the CORRECTED controller (see below). Gain 3 was
# calibrated when the inner PID had no velocity feedforward; on the corrected
# controller it crashes 8/8. The best stable gain is now ZERO.
DEFAULT_P_GAIN = 3.0   # kept for reproducing the pre-feedforward measurements


def pid_baseline(obs: np.ndarray) -> np.ndarray:
    """Do nothing: let the inner PID hold the formation slot unaided.

    The weakest meaningful reference. Use `p_control_baseline` for the bar that
    actually discriminates.
    """
    obs = np.asarray(obs)
    return np.zeros((obs.shape[0], 3), dtype=np.float32)


def p_control_baseline(obs: np.ndarray, gain: float = DEFAULT_P_GAIN) -> np.ndarray:
    """Proportional feedback on the observed gap error.

    `obs[:, 0:3]` is each drone's gap error -- the difference between where it
    is and where its reference says it should be. Commanding an offset OPPOSITE
    to that error drives it toward zero.

        action = clip(-gain * gap_error, -1, 1)

    This is deliberately the simplest controller that uses the observation at
    all. A learned policy that cannot beat it is not earning its complexity.

    Parameters
    ----------
    obs
        (num_agents, 13) observation array.
    gain
        Proportional gain. The stable band is narrow -- see the module
        docstring. Values around 8 crash every episode.
    """
    gap_error = np.asarray(obs)[:, 0:3]
    return np.clip(-gain * gap_error, -1.0, 1.0).astype(np.float32)


BASELINES = {
    "pid": pid_baseline,
    "p_control": p_control_baseline,
}
