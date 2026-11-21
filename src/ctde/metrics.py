"""
================================================================================
 PHYSICAL PERFORMANCE METRICS
================================================================================

Charter §S.11 — adopted regardless of what the diagnostics show.

--------------------------------------------------------------------------------
 WHY RETURN IS NOT ENOUGH
--------------------------------------------------------------------------------
Return is unitless, task-specific, and changes meaning whenever the reward is
retuned (ours has changed three times). A robotics reviewer cannot judge whether
a controller is good from a number like "677".

What they CAN judge:

    RMS formation error 4.2 cm, minimum separation 0.61 m, zero collisions.

So return stays a TRAINING variable; the scientific results are reported in
metres, seconds, and rates.

--------------------------------------------------------------------------------
 DOWNSTREAM ERROR AMPLIFICATION  --  the control-systems contribution
--------------------------------------------------------------------------------
The chain topology raises a question a pure MARL ablation cannot: when the
leader is disturbed, does the error GROW as it propagates downstream?

For gap error

    e_i(t) = p_{i-1}(t) - p_i(t) - Delta*_i

define the amplification ratio

    G_i = RMS(e_i) / RMS(e_{i-1})

    G_i > 1  ->  disturbances AMPLIFY down the chain
    G_i < 1  ->  disturbances ATTENUATE

This matters for the paper because it shows critic selection affects not just
cumulative reward but FORMATION STABILITY -- a property an aerospace-control
reviewer recognises immediately.

NAMING, DELIBERATELY CAUTIOUS
-----------------------------
We call this "empirical downstream error amplification", NOT string stability.

An RMS ratio under stochastic disturbances is a useful diagnostic but is NOT a
formal string-stability proof. A formal claim requires controlled excitation
(chirp or impulse) and a frequency-domain or induced-norm statement such as

    sup_omega | E_i(j*omega) / E_{i-1}(j*omega) | <= 1

Using the formal term on the strength of an RMS ratio would be exactly the kind
of overclaim the September audit was about. See Swaroop & Hedrick for the
formal treatment.
================================================================================
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np


# ==============================================================================
# SECTION 1 : EPISODE RECORD
# ==============================================================================

@dataclass
class EpisodeMetrics:
    """Physical performance over one episode. All quantities in SI units."""

    # ---- Tracking ------------------------------------------------------------
    rms_position_error_m: float          # across all drones
    max_position_error_m: float
    rms_lateral_error_m: float           # sidestep-exploit monitor
    rms_vertical_error_m: float

    # ---- Per-drone (index 0 = leader) ---------------------------------------
    per_drone_rms_error_m: list[float]

    # ---- Safety --------------------------------------------------------------
    min_separation_m: float
    collision: bool
    crashed: bool
    failure_reason: str

    # ---- Chain behaviour -----------------------------------------------------
    # G_i for i = 1..N-1. See the module docstring on naming.
    error_amplification: list[float]

    # ---- Effort --------------------------------------------------------------
    # Proxy for energy: mean |commanded offset|. The true electrical cost would
    # need a motor model; this is monotone in it and needs no extra assumptions.
    mean_control_effort: float
    rms_angular_rate: float

    # ---- Context -------------------------------------------------------------
    episode_length_steps: int
    completed: bool                      # reached the time limit without failure
    mean_coupling_N: float
    mean_gust_speed_ms: float

    def as_dict(self) -> dict:
        return asdict(self)


# ==============================================================================
# SECTION 2 : COLLECTOR
# ==============================================================================

class EpisodeRecorder:
    """Accumulates per-step data and produces an `EpisodeMetrics` at the end.

    Usage
    -----
    >>> rec = EpisodeRecorder(num_agents=4, max_steps=720)
    >>> obs, _ = env.reset(seed=0)
    >>> while True:
    ...     action = policy(obs)
    ...     obs, reward, term, trunc, info = env.step(action)
    ...     rec.record(env, action, info)
    ...     if term or trunc:
    ...         break
    >>> metrics = rec.finalise()

    Design note: the recorder reads through the environment's PUBLIC interface
    (`get_global_state()`, `info`) rather than private attributes -- the same
    rule the correctness tests follow.
    """

    def __init__(self, num_agents: int, max_steps: int):
        self.num_agents = num_agents
        self.max_steps = max_steps

        self._gap_errors: list[np.ndarray] = []      # (N, 3) per step
        self._angular_rates: list[float] = []
        self._efforts: list[float] = []
        self._separations: list[float] = []
        self._couplings: list[float] = []
        self._gusts: list[float] = []

        self._collision = False
        self._crashed = False
        self._failure_reason = ""
        self._steps = 0

    # --------------------------------------------------------------------------
    def record(self, env, action: np.ndarray, info: dict) -> None:
        """Capture one step."""
        state = env.get_global_state()
        positions = state["positions"]

        # Gap error against each drone's reference -- the same quantity the
        # observation and reward use, so the metric matches what was optimised.
        references = env._reference_positions()
        gap = positions - references
        self._gap_errors.append(gap.copy())

        angular = np.array([
            np.linalg.norm(env._getDroneStateVector(i)[13:16])
            for i in range(self.num_agents)
        ])
        self._angular_rates.append(float(angular.mean()))

        self._efforts.append(float(np.abs(np.asarray(action)).mean()))
        self._separations.append(float(info.get("min_separation_m", np.inf)))
        self._couplings.append(float(info.get("total_coupling_N", 0.0)))
        # MEAN PER-DRONE gust SPEED, not the Frobenius norm of the whole array.
        # `gusts` is (N, 3). np.linalg.norm() over that flattens it, giving
        # sqrt(sum over ALL drones), which grows like sqrt(N) and is not a
        # speed. Measured: 5.689 for a field whose true mean per-drone speed was
        # 2.486 -- every logged gust speed was inflated ~2x at N=4.
        gusts = np.atleast_2d(np.asarray(state.get("gusts", np.zeros((1, 3)))))
        self._gusts.append(float(np.linalg.norm(gusts, axis=1).mean()))

        if info.get("collision"):
            self._collision = True
        if info.get("crashed"):
            self._crashed = True
            self._failure_reason = info.get("failure_reason", "")

        self._steps += 1

    # --------------------------------------------------------------------------
    def finalise(self) -> EpisodeMetrics:
        """Compute the episode's metrics."""
        if not self._gap_errors:
            raise ValueError("no steps recorded")

        errors = np.array(self._gap_errors)            # (T, N, 3)
        magnitudes = np.linalg.norm(errors, axis=2)    # (T, N)

        per_drone_rms = [float(np.sqrt(np.mean(magnitudes[:, i] ** 2)))
                         for i in range(self.num_agents)]

        lateral = np.linalg.norm(errors[:, :, 0:2], axis=2)
        vertical = np.abs(errors[:, :, 2])

        return EpisodeMetrics(
            rms_position_error_m=float(np.sqrt(np.mean(magnitudes ** 2))),
            max_position_error_m=float(magnitudes.max()),
            rms_lateral_error_m=float(np.sqrt(np.mean(lateral ** 2))),
            rms_vertical_error_m=float(np.sqrt(np.mean(vertical ** 2))),
            per_drone_rms_error_m=per_drone_rms,
            min_separation_m=float(np.min(self._separations)),
            collision=self._collision,
            crashed=self._crashed,
            failure_reason=self._failure_reason,
            error_amplification=compute_error_amplification(per_drone_rms),
            mean_control_effort=float(np.mean(self._efforts)),
            rms_angular_rate=float(np.sqrt(np.mean(np.array(self._angular_rates) ** 2))),
            episode_length_steps=self._steps,
            completed=(self._steps >= self.max_steps * 0.99) and not self._crashed,
            mean_coupling_N=float(np.mean(self._couplings)),
            mean_gust_speed_ms=float(np.mean(self._gusts)),
        )


# ==============================================================================
# SECTION 3 : DOWNSTREAM ERROR AMPLIFICATION
# ==============================================================================

def compute_error_amplification(per_drone_rms: list[float]) -> list[float]:
    """Ratio of each follower's RMS error to its predecessor's.

    Returns G_i for i = 1..N-1, so the list has length N-1.

    The denominator for the FIRST follower is the leader's own
    trajectory-tracking error -- otherwise there is nothing to divide by, and
    G_1 would be undefined.

    Interpretation
    --------------
        G_i > 1   errors grow downstream  (the chain amplifies disturbances)
        G_i < 1   errors shrink downstream (the chain attenuates them)

    See the module docstring on why this is called EMPIRICAL amplification
    rather than string stability.
    """
    out: list[float] = []
    for i in range(1, len(per_drone_rms)):
        upstream = per_drone_rms[i - 1]
        out.append(float(per_drone_rms[i] / upstream) if upstream > 1e-9 else float("inf"))
    return out


def summarise_amplification(amplifications: list[list[float]]) -> dict[str, float]:
    """Aggregate amplification across episodes or seeds.

    `worst_link` is the headline: the single largest G_i anywhere in the chain.
    A chain is only as stable as its weakest link, so an average would hide the
    failure mode that matters.
    """
    if not amplifications:
        return {}

    arr = np.array(amplifications, dtype=float)          # (episodes, N-1)
    finite = np.isfinite(arr)

    return {
        "mean_amplification": float(arr[finite].mean()) if finite.any() else float("nan"),
        "worst_link": float(arr[finite].max()) if finite.any() else float("nan"),
        "fraction_amplifying": float((arr[finite] > 1.0).mean()) if finite.any() else float("nan"),
        "per_link_mean": [
            float(arr[finite[:, i], i].mean()) if finite[:, i].any() else float("nan")
            for i in range(arr.shape[1])
        ],
    }


# ==============================================================================
# SECTION 4 : REPORTING
# ==============================================================================

def format_metrics_table(metrics: list[EpisodeMetrics], label: str = "") -> str:
    """Human-readable summary in PHYSICAL units.

    This is what belongs in the paper -- centimetres and rates, not return.
    """
    if not metrics:
        return "no episodes recorded"

    def stat(attr: str) -> tuple[float, float]:
        vals = np.array([getattr(m, attr) for m in metrics], dtype=float)
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            return float("nan"), float("nan")
        return float(vals.mean()), float(vals.std(ddof=1)) if len(vals) > 1 else 0.0

    lines = [
        "=" * 70,
        f" PHYSICAL METRICS{(' — ' + label) if label else ''}   (n = {len(metrics)} episodes)",
        "=" * 70,
        "",
        f"  {'metric':<30} {'mean':<12} {'std':<12}",
        "  " + "-" * 56,
    ]

    rows = [
        ("RMS position error (cm)", "rms_position_error_m", 100.0),
        ("max position error (cm)", "max_position_error_m", 100.0),
        ("RMS lateral error (cm)", "rms_lateral_error_m", 100.0),
        ("RMS vertical error (cm)", "rms_vertical_error_m", 100.0),
        ("min separation (m)", "min_separation_m", 1.0),
        ("mean control effort", "mean_control_effort", 1.0),
        ("RMS angular rate (rad/s)", "rms_angular_rate", 1.0),
        ("episode length (steps)", "episode_length_steps", 1.0),
    ]
    for name, attr, scale in rows:
        mean, std = stat(attr)
        lines.append(f"  {name:<30} {mean * scale:<12.3f} {std * scale:<12.3f}")

    collisions = sum(m.collision for m in metrics)
    crashes = sum(m.crashed for m in metrics)
    completed = sum(m.completed for m in metrics)

    lines += [
        "",
        f"  {'completed episodes':<30} {completed}/{len(metrics)}",
        f"  {'crashes':<30} {crashes}/{len(metrics)}",
        f"  {'collisions':<30} {collisions}/{len(metrics)}",
    ]

    amp = summarise_amplification([m.error_amplification for m in metrics])
    if amp:
        lines += [
            "",
            "  DOWNSTREAM ERROR AMPLIFICATION  (G_i = RMS(e_i)/RMS(e_{i-1}))",
            f"    {'mean G':<28} {amp['mean_amplification']:.3f}",
            f"    {'worst link':<28} {amp['worst_link']:.3f}",
            f"    {'fraction of links G > 1':<28} {amp['fraction_amplifying']:.1%}",
            f"    {'per-link mean':<28} "
            + ", ".join(f"{v:.3f}" for v in amp["per_link_mean"]),
            "",
            "    G > 1 means disturbances AMPLIFY downstream.",
            "    (Empirical amplification, not a formal string-stability claim.)",
        ]

    lines += ["", "=" * 70]
    return "\n".join(lines)
