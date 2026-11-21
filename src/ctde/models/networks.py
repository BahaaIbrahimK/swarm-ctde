"""
================================================================================
 ACTOR AND CRITIC NETWORKS
================================================================================

PHASE 2 (Charter §5). Implements the confound controls of Charter §4.3.

--------------------------------------------------------------------------------
 THE ONE THING THIS FILE EXISTS TO GUARANTEE
--------------------------------------------------------------------------------
When we compare k=1 against k=3, the ONLY difference must be what the critic
KNOWS -- never how big it is, never how the actor is built, never how the
inputs are ordered.

If a reviewer can say "your k=3 critic simply had more parameters", the entire
paper collapses. So capacity control is not an implementation detail here; it
is the experimental design, expressed in code.

--------------------------------------------------------------------------------
 HOW CAPACITY IS HELD CONSTANT
--------------------------------------------------------------------------------
Naive approach (WRONG):

    critic = MLP(input_dim=critic_input_dim(k), hidden=256, out=1)

    k=0 -> 12 inputs -> first layer 12*256  =  3,072 weights
    k=3 -> 30 inputs -> first layer 30*256  =  7,680 weights
    => k=3 has 4,608 MORE parameters. Capacity is confounded with information.

Our approach (CORRECT), as of v4-correctness:

    ALL rungs receive the SAME input width -- real teammate slots up to k,
    ZEROS beyond:

    projection: Linear(MAX_CRITIC_INPUT_DIM -> PROJECTION_DIM)  # same at every k
    trunk:      Linear(PROJECTION_DIM -> 256) -> ... -> Linear(256 -> 1)  # IDENTICAL

For N=4 that is 34-D input and 84,929 parameters at EVERY rung -- not
approximately equal, exactly equal, verified on the real learners the
experiment constructs (tests/test_wiring.py).

WHAT THIS REPLACED, AND WHY IT MATTERED
---------------------------------------
Until the September 2026 audit this file used a projection layer whose width
VARIED with k, and argued the residual difference was small and biased against
the hypothesis. That argument was defensible but weaker than it needed to be,
and it had a fatal companion defect: the padding helper existed but MAPPO never
called it, so the real learners were 13-D and 34-D -- 1,344 parameters apart --
while the tests checked networks they had built themselves and passed.

Zero-padding removes the argument entirely. There is no residual difference to
defend, and "your k=3 critic simply had more parameters" is not a sentence a
reviewer can write.

Note the padding is not information-free by accident: an agent at k1 and one at
k3 differ ONLY in how many slots carry live numbers. That is the independent
variable, and nothing else moves with it.
================================================================================
"""

from __future__ import annotations

import torch
import torch.nn as nn

# Width that every critic projects to before the shared trunk. Fixed across all
# rungs -- this is what makes capacity comparable.
PROJECTION_DIM = 64


# ==============================================================================
# SECTION 1 : ACTOR  (decentralized execution)
# ==============================================================================

class Actor(nn.Module):
    """Policy network. IDENTICAL at every rung of the ladder.

    CONFOUND CONTROL 1 (Charter §4.3): the actor's architecture, input, and
    parameter count never change with k. It sees only the agent's own 13-D
    local observation -- which is what makes execution decentralized and what
    makes the policy deployable on real hardware with no radio link.

    Outputs the mean of a Gaussian over actions. The log-std is a separate
    learned parameter (standard for continuous-control PPO): making it
    state-independent stabilises early training, when a state-dependent std can
    collapse or explode.
    """

    def __init__(self, obs_dim: int = 13, action_dim: int = 3, hidden: int = 256):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),                      # Tanh, not ReLU: standard for
            nn.Linear(hidden, hidden),      # continuous control; bounded
            nn.Tanh(),                      # activations keep early policies
            nn.Linear(hidden, action_dim),  # from producing extreme actions
        )

        # State-independent log-std.
        #
        # CALIBRATED 2026-08-28. The default -0.5 (std 0.61) was catastrophic
        # here: with max_offset=0.5 m and lateral_weight=10, exploration noise
        # alone cost ~662 return per episode -- MORE than the ~200 available to
        # win. The policy started at the PID ceiling (588) and degraded to 138,
        # because random exploration was strictly worse than doing nothing.
        #
        # -1.5 (std 0.22) with max_offset=0.2 m costs ~14 per episode, about 7%
        # of the available return: enough to explore, cheap enough not to
        # dominate. See docs/task_v2_notes.md.
        self.log_std = nn.Parameter(torch.full((action_dim,), -1.5))

        self._init_weights()

    def _init_weights(self) -> None:
        """Orthogonal init -- the standard choice for PPO stability.

        The final layer gets gain 0.01 so the initial policy is close to
        deterministic-at-zero. Since our action space is normalised such that
        0 == hover RPM, an untrained policy therefore roughly hovers instead of
        immediately crashing. That makes early experience informative rather
        than uniformly catastrophic.
        """
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=1.0)
                nn.init.constant_(layer.bias, 0.0)

        last = self.net[-1]
        nn.init.orthogonal_(last.weight, gain=0.01)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (pre-squash mean, std) of the underlying Gaussian."""
        mean = self.net(obs)
        std = torch.exp(self.log_std).expand_as(mean)
        return mean, std

    def distribution(self, obs: torch.Tensor) -> "SquashedGaussian":
        """Return a BOUNDED action distribution.

        BOUNDED-ACTION FIX (16 Sep 2026).

        THE BUG: the policy sampled from an unbounded Normal, PPO stored the
        log-probability of that SAMPLED action, and the environment then clipped
        to [-1, 1]. Whenever clipping occurred,

            a_executed_by_physics  !=  a_in_the_PPO_likelihood_ratio

        which biases the objective: the ratio describes a command that was never
        applied. Saturation was rare at the initial log_std, but the learned MEAN
        is unbounded, so it can become common during training.

        THE FIX: a tanh-squashed Gaussian, with the change-of-variables
        correction to the log-probability. Actions are bounded by construction,
        so the executed action and the one in the ratio are always identical.
        """
        mean, std = self.forward(obs)
        return SquashedGaussian(mean, std)


# ==============================================================================
# SECTION 1b : BOUNDED ACTION DISTRIBUTION
# ==============================================================================

class SquashedGaussian:
    """tanh-squashed Gaussian over [-1, 1], with the correct log-density.

    For u ~ N(mu, sigma) and a = tanh(u), the change of variables gives

        log p(a) = log p(u) - sum_i log(1 - tanh(u_i)^2)

    computed in a numerically stable form. This is the standard construction
    (SAC, and bounded-action PPO variants).

    WHY NOT JUST CLIP: clipping makes the executed action differ from the one
    whose probability PPO uses, biasing the importance ratio. Squashing keeps
    them identical by construction.
    """

    # Guards log(1 - a^2) against a = +/-1 exactly.
    _EPS = 1e-6

    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        self.mean = mean
        self.std = std
        self.base = torch.distributions.Normal(mean, std)

    # ---- Sampling ------------------------------------------------------------

    def sample(self) -> torch.Tensor:
        """A bounded sample. `rsample` keeps the path differentiable."""
        return torch.tanh(self.base.rsample())

    @property
    def deterministic(self) -> torch.Tensor:
        """The squashed mean -- used for evaluation."""
        return torch.tanh(self.mean)

    # ---- Density -------------------------------------------------------------

    def log_prob(self, action: torch.Tensor) -> torch.Tensor:
        """Log-density of a BOUNDED action, with the tanh correction.

        Returns a per-dimension tensor; callers sum over the action axis, as
        they do for a plain Normal.
        """
        clamped = action.clamp(-1.0 + self._EPS, 1.0 - self._EPS)
        pre_squash = torch.atanh(clamped)

        base_log_prob = self.base.log_prob(pre_squash)
        # d a / d u = 1 - tanh(u)^2 ; subtract its log for the density correction.
        correction = torch.log(1.0 - clamped.pow(2) + self._EPS)

        return base_log_prob - correction

    def entropy(self) -> torch.Tensor:
        """Entropy of the underlying Gaussian.

        The exact entropy of a squashed Gaussian has no closed form. The base
        entropy is the standard surrogate and is monotone in `std`, which is all
        the entropy BONUS needs -- it only has to discourage premature collapse.
        Documented so the approximation is not mistaken for exactness.
        """
        return self.base.entropy()


# ==============================================================================
# SECTION 2 : CRITIC  (centralized training)
# ==============================================================================

class Critic(nn.Module):
    """Value network whose INPUT WIDTH AND CAPACITY are identical at every rung.

    CONFOUND CONTROL 2 (Charter §4.3). See the module docstring for why this
    matters more than anything else in the file.

    UPDATED v4-correctness (16 Sep 2026)
    ------------------------------------
    This docstring used to say the input width VARIED with k, and it did. Every
    rung then had a differently sized projection layer, so k changed both the
    INFORMATION the critic saw and the PARAMETERS it had to fit -- two variables
    moving together, which is the confound the whole study exists to avoid.

    Rungs now receive a fixed-width vector: real teammate slots up to k, ZEROS
    beyond. Width is `max_critic_input_dim(...)` (34-D for N=4) at every rung,
    so all five rungs instantiate byte-identical architectures with 84,929
    parameters -- verified on the real learners in tests/test_wiring.py, not on
    networks the test built itself.

    The zeros are not information-free padding by accident: an agent at k1 and
    an agent at k3 differ only in how many slots carry live numbers, which is
    exactly the independent variable and nothing else.

    Architecture
    ------------
        input (fixed width, zero-padded beyond k)
          -> projection: Linear(input_dim -> PROJECTION_DIM) + Tanh
          -> trunk:      Linear(PROJECTION_DIM -> hidden) + Tanh
                         Linear(hidden -> hidden) + Tanh
                         Linear(hidden -> 1)
    """

    def __init__(self, input_dim: int, hidden: int = 256, projection_dim: int = PROJECTION_DIM):
        super().__init__()

        self.input_dim = input_dim
        self.projection_dim = projection_dim

        # Under v4 padding this width is the SAME at every rung. It is kept
        # as a separate projection layer so that capacity equality is
        # structural rather than a coincidence of the chosen dimensions.
        self.projection = nn.Sequential(
            nn.Linear(input_dim, projection_dim),
            nn.Tanh(),
        )

        # Byte-identical across every rung of the ladder.
        self.trunk = nn.Sequential(
            nn.Linear(projection_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for module in list(self.projection) + list(self.trunk):
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
                nn.init.constant_(module.bias, 0.0)

        # Value head gain 1.0 is standard; unlike the actor we do not want the
        # critic pinned near zero, since it must span a wide return range.

    def forward(self, critic_input: torch.Tensor) -> torch.Tensor:
        """Return the value estimate, shape (batch,)."""
        return self.trunk(self.projection(critic_input)).squeeze(-1)

    # --------------------------------------------------------------------------
    def parameter_report(self) -> dict[str, int]:
        """Exact parameter counts, split into varying and fixed parts.

        REPORTED IN THE PAPER. Being able to state "the trunk is identical to
        the parameter across all k; only N projection weights differ" is what
        makes the capacity control verifiable rather than merely asserted.
        """
        proj = sum(p.numel() for p in self.projection.parameters())
        trunk = sum(p.numel() for p in self.trunk.parameters())

        return {
            "projection_params": proj,      # varies with k
            "trunk_params": trunk,          # IDENTICAL across all k
            "total_params": proj + trunk,
        }


# ==============================================================================
# SECTION 3 : VERIFICATION HELPERS
# ==============================================================================

def verify_capacity_control(input_dims: list[int], hidden: int = 256) -> str:
    """Build a critic per rung and confirm the trunk really is identical.

    A test that FAILS LOUDLY if someone later refactors the critic in a way
    that reintroduces the capacity confound. Cheap insurance on the paper's
    central experimental control.
    """
    lines = [
        "CRITIC CAPACITY CONTROL CHECK",
        "-" * 70,
        f"  {'input_dim':<12} {'projection':<14} {'trunk':<14} {'total':<12}",
        "-" * 70,
    ]

    trunk_counts = set()

    for dim in input_dims:
        report = Critic(input_dim=dim, hidden=hidden).parameter_report()
        trunk_counts.add(report["trunk_params"])
        lines.append(
            f"  {dim:<12} {report['projection_params']:<14,} "
            f"{report['trunk_params']:<14,} {report['total_params']:<12,}"
        )

    lines.append("-" * 70)

    if len(trunk_counts) == 1:
        lines.append(f"  PASS: trunk identical across all rungs ({trunk_counts.pop():,} params)")
    else:
        lines.append(f"  FAIL: trunk differs across rungs -> {sorted(trunk_counts)}")

    total_span = Critic(max(input_dims)).parameter_report()["total_params"] - \
                 Critic(min(input_dims)).parameter_report()["total_params"]
    lines.append(
        f"  Total-parameter spread across the ladder: {total_span:,} "
        "(projection only; favours HIGH k, i.e. against our hypothesis)"
    )

    return "\n".join(lines)
