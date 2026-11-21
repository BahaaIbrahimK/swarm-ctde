"""
================================================================================
 DOWNWASH PROBE  --  measuring inter-agent aerodynamic coupling
================================================================================

PHASE 1 EXIT ARTIFACT (Charter §5)

--------------------------------------------------------------------------------
 WHY THIS IS THE FIRST THING WE BUILD
--------------------------------------------------------------------------------
This file contains NO reinforcement learning. That is deliberate.

The entire paper rests on a single physical assumption (Charter §1.5):

    Downwash creates enough inter-agent coupling that a teammate-aware critic
    has something real to explain.

The logic chain of the paper is:

    downwash exists
        -> a drone dips for reasons its OWN observation cannot explain
            -> a critic that sees the teammate ABOVE can explain that dip
                -> variance reduction  (the force pushing performance UP as k grows)
                    -> but it SATURATES once the critic sees the drones that
                       aerodynamically matter
                        -> meanwhile cost keeps growing
                            -> therefore an INTERIOR OPTIMUM  (Claim 1)

Every link after the first depends on the first. If downwash is negligible at
our formation spacings, Claim 1 and Claim 2 are both dead and no amount of good
MAPPO code can rescue them.

So we test the premise FIRST, cheaply, in week 3 -- rather than discovering the
problem expensively in month 5. This is the same "fail fast" logic as the
Phase 3 decision gate in Charter §3.

--------------------------------------------------------------------------------
 WHAT DOWNWASH IS (physical recap)
--------------------------------------------------------------------------------
A rotor produces lift by accelerating air DOWNWARD (Newton's third law: push
air down, get pushed up). So beneath every hovering drone there is a column of
fast-moving descending air.

Put a second drone underneath and two things go wrong at once:

  1. DIRECT FORCE  -- moving air physically pushes down on it.
  2. LOST LIFT     -- its rotors are trying to accelerate air that is ALREADY
                      moving downward. Grabbing air that is already falling
                      gives you less to push against, so the same RPM produces
                      less thrust.

The lower drone sinks, through no fault of its own controller. THAT is the
"unexplained dip" our centralized critic is supposed to explain away.

--------------------------------------------------------------------------------
 WHAT THIS PROBE MEASURES
--------------------------------------------------------------------------------
Two drones, one directly above the other, both commanded to hold hover RPM.
Because both hold the SAME command, any vertical deviation of the LOWER drone
is attributable to aerodynamic interference rather than to its own control.

We sweep:
  * vertical separation   -> how far apart before coupling vanishes?
  * horizontal offset     -> how precisely aligned must drones be to interact?

--------------------------------------------------------------------------------
 THE THREE OUTPUTS AND WHAT EACH IS FOR
--------------------------------------------------------------------------------
  1. PREMISE CHECK        Is coupling real at all? (If no -> project pivots.)
  2. COUPLING HORIZON     The separation beyond which downwash is negligible.
                          This PREDICTS where k* should sit: the critic should
                          see teammates inside the horizon, and no more.
  3. SPACING CALIBRATION  Which formation spacings to use for Claim 2, chosen
                          to span weak -> moderate -> strong coupling.
================================================================================
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np


# ==============================================================================
# SECTION 1 : RESULT RECORD
# ==============================================================================

@dataclass
class ProbeMeasurement:
    """One measurement: two drones at a given relative position.

    Each field is a column in the output CSV.
    """

    # ---- Independent variables (what we set) --------------------------------
    vertical_separation_m: float    # how far BELOW the upper drone
    horizontal_offset_m: float      # lateral offset from directly beneath

    # ---- Dependent variables (what we measure) ------------------------------
    # Vertical drift of the LOWER drone over the measurement window.
    # Negative = it sank. This is the headline number: it is the visible
    # symptom of downwash, expressed in metres.
    lower_drone_z_drift_m: float

    # Mean vertical velocity of the lower drone during measurement (m/s).
    # Negative = descending. More sensitive than drift for weak coupling,
    # because velocity responds before displacement accumulates.
    lower_drone_mean_vz_ms: float

    # Same two quantities for the UPPER drone. These act as a CONTROL:
    # the upper drone is in clean air, so it should show near-zero drift.
    # If the upper drone also sinks, our measurement is contaminated by
    # something else (e.g. hover RPM slightly mis-calibrated) and the
    # lower-drone numbers cannot be trusted.
    upper_drone_z_drift_m: float
    upper_drone_mean_vz_ms: float

    # ---- Derived -------------------------------------------------------------
    # The coupling signal: how much MORE the lower drone sank than the upper.
    # Subtracting the control removes any common-mode bias, isolating the
    # effect actually caused by the neighbour.
    differential_z_drift_m: float

    def as_dict(self) -> dict:
        return asdict(self)


# ==============================================================================
# SECTION 2 : THE PROBE
# ==============================================================================

class DownwashProbe:
    """Measures downwash coupling between two hovering drones.

    Usage
    -----
    >>> probe = DownwashProbe(config)
    >>> measurements = probe.run_sweep()
    >>> # -> list[ProbeMeasurement], one per (separation, offset) pair

    Design note
    -----------
    The environment is created and destroyed for EACH measurement point.
    That is slower than reusing one environment, but it guarantees no state
    leaks between measurements (residual velocities, accumulated drift).
    For a probe of a few dozen points, correctness beats speed.
    """

    def __init__(self, config: dict):
        """
        Parameters
        ----------
        config:
            Resolved config dict (see configs/probe_downwash.yaml).
        """
        self.config = config

        # ---- Read probe settings -------------------------------------------
        probe_cfg = config["probe"]
        self.vertical_separations = probe_cfg["vertical_separations_m"]
        self.horizontal_offsets = probe_cfg["horizontal_offsets_m"]
        self.settle_steps = probe_cfg["settle_steps"]
        self.measure_steps = probe_cfg["measure_steps"]

        # ---- Read physics settings -----------------------------------------
        phys_cfg = config["physics"]
        self.sim_freq_hz = phys_cfg["sim_freq_hz"]
        self.physics_mode = phys_cfg["mode"]

        # Altitude of the UPPER drone. High enough that GROUND EFFECT (a
        # separate near-floor phenomenon) cannot contaminate the measurement --
        # we want downwash in isolation. Read from config so the probe can be
        # retuned without editing code.
        self.upper_altitude_m = phys_cfg.get("probe_upper_altitude_m", 1.5)

        # Guard: the whole point of this probe is downwash. If the physics
        # mode does not include it, every measurement would read zero and we
        # would wrongly conclude the premise is false. Fail loudly instead.
        if "dw" not in self.physics_mode:
            raise ValueError(
                f"physics.mode is '{self.physics_mode}', which does not include "
                "downwash. The probe would measure nothing. Use 'pyb_gnd_drag_dw'."
            )

    # --------------------------------------------------------------------------
    def run_sweep(self, verbose: bool = True) -> list[ProbeMeasurement]:
        """Run the full 2-D sweep over separations and offsets.

        Returns
        -------
        list[ProbeMeasurement]
            One measurement per (vertical_separation, horizontal_offset) pair.
        """
        measurements: list[ProbeMeasurement] = []

        total = len(self.vertical_separations) * len(self.horizontal_offsets)
        idx = 0

        for v_sep in self.vertical_separations:
            for h_off in self.horizontal_offsets:
                idx += 1
                if verbose:
                    print(
                        f"[probe {idx:3d}/{total}] "
                        f"vertical={v_sep:.2f} m  horizontal={h_off:.2f} m",
                        flush=True,
                    )

                m = self.measure_single_point(
                    vertical_separation_m=v_sep,
                    horizontal_offset_m=h_off,
                )
                measurements.append(m)

        return measurements

    # --------------------------------------------------------------------------
    def measure_single_point(
        self,
        vertical_separation_m: float,
        horizontal_offset_m: float,
    ) -> ProbeMeasurement:
        """Measure coupling at one relative drone position.

        PROTOCOL
        --------
        1. Place UPPER drone at a fixed hover altitude.
        2. Place LOWER drone `vertical_separation_m` below it, offset laterally
           by `horizontal_offset_m`.
        3. Command BOTH drones to hold hover RPM (identical command).
        4. Let the wake develop  (`settle_steps`).
        5. Record positions/velocities  (`measure_steps`).
        6. Report how much more the lower drone sank than the upper.

        Because both drones receive the SAME command and differ only in
        position, the difference in their behaviour isolates the aerodynamic
        interaction.
        """
        # Imported here rather than at module top so that this file can be
        # read, linted, and unit-tested without gym-pybullet-drones installed.
        env = self._make_env(
            vertical_separation_m=vertical_separation_m,
            horizontal_offset_m=horizontal_offset_m,
        )

        try:
            # Gymnasium API: reset() returns (obs, info), not just obs.
            env.reset()

            hover_rpm = float(env.HOVER_RPM)
            # Same command for both drones: 4 rotors each, all at hover RPM.
            # Shape (num_drones, 4) matches env.action_space.
            action = np.array([[hover_rpm] * 4, [hover_rpm] * 4], dtype=np.float32)

            # ---- Phase A: settle -------------------------------------------
            # The rotor wake takes time to develop. Measuring immediately would
            # capture a transient rather than the steady-state interaction.
            for _ in range(self.settle_steps):
                env.step(action)

            # Record starting altitudes AFTER settling, so drift is measured
            # from the settled state rather than from the spawn position.
            upper_z_start = float(env._getDroneStateVector(0)[2])
            lower_z_start = float(env._getDroneStateVector(1)[2])

            # ---- Phase B: measure ------------------------------------------
            upper_vz_samples: list[float] = []
            lower_vz_samples: list[float] = []

            for _ in range(self.measure_steps):
                env.step(action)

                upper_state = env._getDroneStateVector(0)
                lower_state = env._getDroneStateVector(1)

                # State layout (gym-pybullet-drones, 20 entries):
                #   [0:3]   position x, y, z
                #   [3:7]   quaternion
                #   [7:10]  roll, pitch, yaw
                #   [10:13] linear velocity vx, vy, vz   <-- index 12 is vz
                #   [13:16] angular velocity
                #   [16:20] last rotor RPMs
                upper_vz_samples.append(float(upper_state[12]))
                lower_vz_samples.append(float(lower_state[12]))

            upper_z_end = float(env._getDroneStateVector(0)[2])
            lower_z_end = float(env._getDroneStateVector(1)[2])

            # ---- Compute results -------------------------------------------
            upper_drift = upper_z_end - upper_z_start
            lower_drift = lower_z_end - lower_z_start

            return ProbeMeasurement(
                vertical_separation_m=vertical_separation_m,
                horizontal_offset_m=horizontal_offset_m,
                lower_drone_z_drift_m=lower_drift,
                lower_drone_mean_vz_ms=float(np.mean(lower_vz_samples)),
                upper_drone_z_drift_m=upper_drift,
                upper_drone_mean_vz_ms=float(np.mean(upper_vz_samples)),
                # Lower minus upper: subtracting the control removes common-mode
                # bias (e.g. both drones slowly sinking because hover RPM is a
                # touch low), leaving only the neighbour-induced effect.
                differential_z_drift_m=lower_drift - upper_drift,
            )

        finally:
            # Always close, even if an exception occurred, so PyBullet does not
            # leak physics clients across the ~54 measurement points.
            env.close()

    # --------------------------------------------------------------------------
    def _make_env(
        self,
        vertical_separation_m: float,
        horizontal_offset_m: float,
    ):
        """Build a 2-drone environment with the requested geometry.

        Isolated into its own method so that the measurement logic above stays
        readable, and so this is the single place to touch if we later change
        simulator versions.

        WHY CtrlAviary AND NOT BaseAviary
        ---------------------------------
        BaseAviary is ABSTRACT: its `_actionSpace()` raises NotImplementedError
        by design. It provides shared machinery but expects a subclass to
        define the action/observation spaces.

        CtrlAviary is the concrete subclass whose action space is direct rotor
        RPM -- exactly what we need, because the probe commands both drones to
        hold hover RPM. It is also what the upstream `examples/downwash.py`
        uses.
        """
        from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
        from gym_pybullet_drones.utils.enums import DroneModel, Physics

        # ---- Geometry -------------------------------------------------------
        # The upper drone sits high enough that GROUND EFFECT (a separate
        # aerodynamic phenomenon near the floor) does not contaminate the
        # measurement. We want downwash in isolation.
        upper_altitude_m = self.upper_altitude_m
        lower_altitude_m = upper_altitude_m - vertical_separation_m

        if lower_altitude_m <= 0.15:
            raise ValueError(
                f"vertical_separation_m={vertical_separation_m} puts the lower "
                f"drone at {lower_altitude_m:.2f} m, low enough for ground "
                "effect to contaminate the downwash measurement. "
                "Raise upper_altitude_m or reduce the separation."
            )

        initial_xyzs = np.array([
            [0.0,                 0.0, upper_altitude_m],   # drone 0 = UPPER
            [horizontal_offset_m, 0.0, lower_altitude_m],   # drone 1 = LOWER
        ])

        # ---- Physics mode ---------------------------------------------------
        physics_map = {
            "pyb": Physics.PYB,
            "pyb_gnd_drag_dw": Physics.PYB_GND_DRAG_DW,
        }
        if self.physics_mode not in physics_map:
            raise ValueError(
                f"Unknown physics.mode '{self.physics_mode}'. "
                f"Expected one of {sorted(physics_map)}."
            )

        drone_model_map = {"cf2x": DroneModel.CF2X, "cf2p": DroneModel.CF2P}
        drone_model = drone_model_map[self.config["drone"]["model"]]

        return CtrlAviary(
            drone_model=drone_model,
            num_drones=2,
            initial_xyzs=initial_xyzs,
            physics=physics_map[self.physics_mode],
            # API NOTE: current gym-pybullet-drones splits the old single
            # `freq` argument into two:
            #   pyb_freq  -- physics integration rate
            #   ctrl_freq -- how often actions are applied
            # We set them EQUAL so that one env.step() == one physics step,
            # giving per-step measurement resolution. (Training will use a
            # lower ctrl_freq for speed; the probe wants maximum resolution.)
            pyb_freq=self.sim_freq_hz,
            ctrl_freq=self.sim_freq_hz,
            gui=False,
            record=False,
        )


# ==============================================================================
# SECTION 3 : ANALYSIS HELPERS
# ==============================================================================

def find_coupling_horizon(
    measurements: list[ProbeMeasurement],
    threshold_fraction: float = 0.1,
) -> float | None:
    """Estimate the separation at which downwash becomes negligible.

    THIS IS THE NUMBER THAT PREDICTS k* (Charter §1.5).

    The mechanism argument says variance reduction SATURATES once the critic
    can see the teammates that aerodynamically matter. "Aerodynamically
    matter" means "inside the coupling horizon". So this number is our
    a-priori prediction of where the performance peak should sit -- which is
    exactly what makes Claim 2 a falsifiable prediction rather than a
    post-hoc story.

    Parameters
    ----------
    measurements:
        Probe output. Only directly-beneath points (horizontal_offset == 0)
        are used, since those show the strongest coupling.
    threshold_fraction:
        The horizon is the separation at which the effect has decayed to this
        fraction of its maximum. 0.1 means "90% of the effect has vanished".

    Returns
    -------
    float | None
        Estimated horizon in metres, or None if coupling was never detected
        (which would mean the project's premise has FAILED -- see kill
        criterion K1-adjacent logic in Charter §3).
    """
    aligned = [m for m in measurements if m.horizontal_offset_m == 0.0]
    if not aligned:
        return None

    aligned.sort(key=lambda m: m.vertical_separation_m)

    # Effect magnitude = how much the lower drone sank relative to the upper.
    # abs() because sinking is negative; we care about magnitude.
    effects = [abs(m.differential_z_drift_m) for m in aligned]
    max_effect = max(effects)

    # If even the tightest spacing produced essentially nothing, the premise
    # fails. 1 mm over a 1-second window is our "indistinguishable from noise"
    # floor.
    if max_effect < 1e-3:
        return None

    threshold = max_effect * threshold_fraction

    # Walk outward and report the first separation where the effect has
    # dropped below threshold.
    for m, effect in zip(aligned, effects):
        if effect < threshold:
            return m.vertical_separation_m

    # Effect never decayed below threshold within the swept range -- the
    # horizon lies beyond our furthest measurement.
    return None


def summarise(measurements: list[ProbeMeasurement]) -> str:
    """Produce a short human-readable verdict on the premise check.

    Printed at the end of the probe run so the answer to "is the premise
    sound?" is immediately visible, without opening the CSV or the plot.
    """
    aligned = [m for m in measurements if m.horizontal_offset_m == 0.0]
    if not aligned:
        return "No aligned (offset=0) measurements found; cannot summarise."

    aligned.sort(key=lambda m: m.vertical_separation_m)

    tightest = aligned[0]
    widest = aligned[-1]
    horizon = find_coupling_horizon(measurements)

    lines = [
        "=" * 74,
        " DOWNWASH PROBE SUMMARY  --  PREMISE CHECK",
        "=" * 74,
        "",
        f"  Tightest spacing tested : {tightest.vertical_separation_m:.2f} m",
        f"    lower-drone drift     : {tightest.lower_drone_z_drift_m:+.4f} m",
        f"    upper-drone drift     : {tightest.upper_drone_z_drift_m:+.4f} m  (control)",
        f"    differential          : {tightest.differential_z_drift_m:+.4f} m  <-- coupling signal",
        "",
        f"  Widest spacing tested   : {widest.vertical_separation_m:.2f} m",
        f"    differential          : {widest.differential_z_drift_m:+.4f} m",
        "",
    ]

    if horizon is None:
        lines += [
            "  COUPLING HORIZON        : NOT DETECTED",
            "",
            "  *** WARNING ***",
            "  Either coupling was never observed (premise FAILS -- Claim 1 and",
            "  Claim 2 are not testable on this setup), or it had not decayed by",
            "  the widest separation tested (extend vertical_separations_m).",
            "  Inspect the plot before proceeding to Phase 2.",
        ]
    else:
        lines += [
            f"  COUPLING HORIZON        : ~{horizon:.2f} m",
            "",
            "  Interpretation: beyond roughly this separation, a teammate exerts",
            "  little aerodynamic influence. Per Charter §1.5, the critic should",
            "  see teammates INSIDE this horizon and gains little from those",
            "  outside it -- this is our a-priori prediction for where k* sits.",
        ]

    lines += ["", "=" * 74]
    return "\n".join(lines)
