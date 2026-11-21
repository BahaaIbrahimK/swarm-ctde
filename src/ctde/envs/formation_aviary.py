"""
================================================================================
 FORMATION AVIARY  --  the task   (TASK VERSION v4-correctness)
================================================================================

Charter §4.2. Rewritten 2026-08-28 after task version v1-raw4d failed.

--------------------------------------------------------------------------------
 WHY v1 FAILED  (read this before changing anything here)
--------------------------------------------------------------------------------
v1 gave the policy direct 4-D rotor RPM control with no assistance. Measured
over 300,000 steps x 3 seeds:

    episode length: 20-100 steps out of 720          (2.8% - 14% of an episode)
    seed 1 got WORSE: 32 steps -> 38 steps after 300k steps
    seed variance: 139% of the mean                  (kill-criterion K3)

Two compounding causes:

  1. THE TASK WAS UNLEARNABLE FROM SCRATCH.
     A quadrotor is underactuated: it cannot translate without first tilting,
     and it falls out of the sky within ~1 s of imperfect thrust. Asking a
     randomly-initialised network to discover attitude stabilisation AND
     formation-keeping simultaneously meant it never cleared step one.

  2. THE REWARD PAID THE AGENT TO CRASH.
     Reward was a per-step penalty and episodes terminated on crash. So dying
     early accumulated FEWER penalties than flying badly for 15 s. Returns
     "improved" 59% while episode length stayed flat -- the policy was learning
     to die faster, not to fly.

Both are fixed below. The lesson generalises: when returns improve but episode
length does not, suspect the termination/reward interaction.

--------------------------------------------------------------------------------
 FIX 1 -- PID-ASSISTED ATTITUDE CONTROL
--------------------------------------------------------------------------------
Real drones do not learn attitude control from scratch; they have an inner
stabilisation loop and an outer guidance layer. We mirror that:

    policy  ->  desired POSITION OFFSET from its formation slot   (3-D)
                        |
                        v
    DSL PID  ->  rotor RPMs that track that setpoint
                        |
                        v
    physics

The policy now learns FORMATION STRATEGY -- "given the wind and my neighbours,
where should I aim?" -- rather than how to keep a quadrotor upright.

WHY THIS DOES NOT WEAKEN THE PAPER, AND ARGUABLY STRENGTHENS IT:

  * The coupling question is untouched. Downwash still pushes drones down; the
    critic's job is still to explain variance the actor cannot observe.
  * It is MORE realistic. Every deployed drone stack works this way, so
    Claim 4 (sim-to-real) becomes more credible, not less.
  * It matches standard practice in quadrotor-swarm RL (Batra et al. 2021,
    CrazyMARL 2025), so reviewers will not find it unusual.

--------------------------------------------------------------------------------
 FIX 2 -- SURVIVAL SHAPING  (remove the incentive to crash)
--------------------------------------------------------------------------------
Two changes make dying strictly worse than flying:

    a) a per-step SURVIVAL BONUS, so existing is better than not existing;
    b) a large TERMINAL PENALTY on crash, scaled by remaining episode time, so
       crashing early is worse than crashing late.

Together these guarantee the return is monotone in survival time -- an agent can
never improve its score by ending the episode sooner.

--------------------------------------------------------------------------------
 FIX 3 -- CLOSE THE SIDESTEP EXPLOIT
--------------------------------------------------------------------------------
Identified during review, and confirmed numerically: at 1 m separation a 10 cm
lateral offset removes 86% of the downwash for a reward cost of only 0.01
(~0.45% of a typical return). The optimal policy would be to step aside --
which would eliminate the very coupling this paper studies, and produce a flat
curve that looks like "no effect" but is actually "the agents engineered the
effect away".

Fix: weight lateral error much more heavily than vertical error, so holding the
slot always beats dodging the wake.
================================================================================
"""

from __future__ import annotations

import numpy as np
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.envs.BaseAviary import BaseAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gymnasium import spaces

from ctde.envs.coupling import coupling_matrix


class FormationAviary(BaseAviary):
    """Vertical-stack formation flight under wind, with PID-assisted control."""

    # ==========================================================================
    # SECTION 1 : CONSTRUCTION
    # ==========================================================================

    def __init__(
        self,
        *,
        num_drones: int = 4,
        formation_spacing_m: float = 1.0,
        episode_len_sec: float = 15.0,
        # ---- Disturbance (v3) ------------------------------------------------
        # Steady wind is nearly useless here: a PID with integral action cancels
        # a constant force exactly, so seed variance measured 0.00 and there was
        # nothing for any critic to explain. Measured, not assumed -- see
        # docs/task_v3_notes.md.
        #
        # The disturbance is therefore STOCHASTIC: an Ornstein-Uhlenbeck gust
        # field. Parameters are given in PHYSICAL units rather than raw process
        # coefficients, so they can be checked against reality:
        #
        #   gust_std_ms      typical gust speed (m/s)
        #   gust_tau_sec     how long a gust persists (seconds)
        #   gust_spatial_corr  how similar the gust is at neighbouring drones
        #
        # Reference air speeds indoors:
        #   0.1-0.3 m/s  still room        2-4 m/s  desk fan at 1-2 m
        #   0.5-1.5 m/s  HVAC draught      5-8 m/s  fan close up / strong vent
        wind_mean_ms: float = 1.0,
        # 3.0 matches RunConfig's recalibrated default (16 Sep 2026). Two
        # different defaults in two files is how a calibration silently drifts,
        # so they are kept equal deliberately -- see trainer.py for the
        # pure-PID sweep that chose this value.
        gust_std_ms: float = 2.5,
        gust_tau_sec: float = 1.0,
        gust_spatial_corr: float = 0.7,
        drone_model: DroneModel = DroneModel.CF2X,
        physics: Physics = Physics.PYB_GND_DRAG_DW,
        pyb_freq: int = 240,
        ctrl_freq: int = 48,
        base_altitude_m: float = 0.8,
        # ---- Reward shaping (see FIX 2 and FIX 3 above) ----------------------
        lateral_weight: float = 10.0,
        survival_bonus: float = 1.0,
        crash_penalty: float = 50.0,
        # ---- Topology (v3) ---------------------------------------------------
        # True  = CHAIN: each follower observes its gap to the drone ABOVE it.
        #         Purely local -- a real drone measures this with UWB or vision.
        # False = the old v2 behaviour, where every drone's observation was
        #         computed against an ANALYTICAL trajectory formula. That is not
        #         local information: a follower implicitly knew the leader's
        #         entire trajectory, so "decentralized execution" was not honest.
        chain_topology: bool = True,

        # ---- Action scaling --------------------------------------------------
        # CALIBRATED 2026-08-28 jointly with the actor's initial log_std.
        # At 0.5 m, exploration noise cost more return than was available to
        # win, so the policy degraded from the PID ceiling instead of improving.
        max_offset_m: float = 0.2,
        rng: np.random.Generator | None = None,
        gui: bool = False,
        record: bool = False,
    ):
        """
        Parameters
        ----------
        formation_spacing_m:
            Vertical gap between drones. THE COUPLING KNOB FOR CLAIM 2.
            Below 0.6875 m the downwash model is extrapolated (a warning is
            emitted) -- see docs/downwash_model_notes.md.
        lateral_weight:
            How much more a lateral error costs than a vertical one. FIX 3.
            Must be large enough that dodging the wake never pays; verified by
            `scripts/verify_task_fixes.py`.
        survival_bonus:
            Added every step the drone is alive. FIX 2.
        crash_penalty:
            Applied on termination, scaled by the fraction of the episode
            remaining, so crashing early is worse than crashing late. FIX 2.
        max_offset_m:
            Action scale. The policy commands a position offset in
            [-max_offset, +max_offset] from its formation slot; the PID tracks
            it. Bounded so the policy cannot command absurd setpoints.
        """
        self.FORMATION_SPACING = formation_spacing_m
        self.EPISODE_LEN_SEC = episode_len_sec

        # Per-step cost charged for every step of the episode NOT flown after a
        # failure. Must exceed the worst per-step reward a LIVING drone can
        # score, or early termination becomes profitable -- see the
        # fixed-horizon note in `_computeReward`. Measured worst survivable
        # rate: -1.07/step at sigma=2.5; 3.0 leaves a wide margin.
        self.FAILURE_STEP_COST = 3.0
        self.CHAIN_TOPOLOGY = chain_topology

        # Disturbance parameters, in physical units.
        self.WIND_MEAN_MS = wind_mean_ms
        self.GUST_STD_MS = gust_std_ms
        self.GUST_TAU_SEC = gust_tau_sec
        self.GUST_SPATIAL_CORR = gust_spatial_corr

        # OU coefficients derived from the physical parameters:
        #   theta = 1/tau                       (decay rate)
        #   sigma = gust_std * sqrt(2*theta)    (so the stationary std matches)
        self._ou_theta = 1.0 / max(gust_tau_sec, 1e-6)
        self._ou_sigma = gust_std_ms * np.sqrt(2.0 * self._ou_theta)

        # Per-drone gust state, initialised in reset().
        self._gusts = np.zeros((num_drones, 3), dtype=np.float64)
        self.LATERAL_WEIGHT = lateral_weight
        self.SURVIVAL_BONUS = survival_bonus
        self.CRASH_PENALTY = crash_penalty
        self.MAX_OFFSET = max_offset_m

        self._rng = rng if rng is not None else np.random.default_rng(0)
        self._wind_vec = np.zeros(3, dtype=np.float64)
        self._crashed = False
        self._failure_reason = ""

        # INTER-DRONE COLLISION THRESHOLD (added 16 Sep 2026).
        # A Crazyflie 2.x is ~92 mm motor-to-motor, so two airframes are in
        # contact at roughly 0.09 m centre-to-centre. We use 0.15 m to also
        # capture the near-miss regime where rotor wakes interact violently and
        # a real flight test would be aborted.
        self.COLLISION_RADIUS_M = 0.15

        # ---- Warn on extrapolated spacing -----------------------------------
        from ctde.envs.coupling import check_spacing_validity

        ok, msg = check_spacing_validity(formation_spacing_m)
        if not ok:
            print(f"[WARN] FormationAviary: {msg}")

        # ---- Geometry: stack built UPWARD from a floor clearance ------------
        # Building downward from a fixed leader altitude would put the lowest
        # drone underground once num_drones * spacing exceeded that altitude.
        self.LOWEST_ALTITUDE = base_altitude_m
        self.LEADER_ALTITUDE = base_altitude_m + (num_drones - 1) * formation_spacing_m

        initial_xyzs = np.array([
            [0.0, 0.0, self.LEADER_ALTITUDE - i * formation_spacing_m]
            for i in range(num_drones)
        ])

        if initial_xyzs[-1, 2] < 0.3:
            raise ValueError(
                f"Lowest drone would sit at {initial_xyzs[-1, 2]:.2f} m, close enough "
                "for GROUND EFFECT to contaminate the downwash coupling. "
                f"Raise base_altitude_m (currently {base_altitude_m} m)."
            )

        super().__init__(
            drone_model=drone_model,
            num_drones=num_drones,
            initial_xyzs=initial_xyzs,
            physics=physics,
            pyb_freq=pyb_freq,
            ctrl_freq=ctrl_freq,
            gui=gui,
            record=record,
        )

        # ---- FIX 1: one inner-loop PID controller per drone ------------------
        # Created AFTER super().__init__ because it needs DRONE_MODEL.
        self._controllers = [DSLPIDControl(drone_model=drone_model) for _ in range(num_drones)]

    # ==========================================================================
    # SECTION 2 : SPACES
    # ==========================================================================

    def _actionSpace(self) -> spaces.Box:
        """Desired position OFFSET from the formation slot, per drone (3-D).

        v1 used raw rotor RPMs (4-D) and was unlearnable. The policy now
        answers a strategic question -- "where should I aim, given the wind and
        my neighbours?" -- while the PID handles the underactuated attitude
        dynamics beneath it.
        """
        return spaces.Box(low=-1.0, high=1.0, shape=(self.NUM_DRONES, 3), dtype=np.float32)

    def _observationSpace(self) -> spaces.Box:
        """Per-agent LOCAL observation, 13-D (v4).

        This is the decentralized-execution half of CTDE: whatever the critic
        receives (the k ladder), the actor always sees exactly these numbers.

        Layout
        ------
            [0:3]   gap error to reference        (relative position)
            [3:6]   roll, pitch, yaw
            [6:9]   velocity RELATIVE to reference
            [9:12]  angular velocity
            [12]    role flag: 1 = leader, 0 = follower

        Changed from 12-D on 16 Sep 2026. Index [6:9] was ABSOLUTE velocity,
        which left the actor unable to compute the gap derivative while the
        k=1 critic received it -- a confound for the whole result (audit #10).
        Index [12] resolves leader/follower aliasing under a shared actor
        (audit #11). Every quantity remains locally measurable on hardware.
        """
        return spaces.Box(low=-np.inf, high=np.inf, shape=(self.NUM_DRONES, 13), dtype=np.float32)

    # ==========================================================================
    # SECTION 3 : ACTION -> RPM  (FIX 1)
    # ==========================================================================

    def _apply_wind(self) -> None:
        """Apply a stochastic, spatially-correlated gust field to every drone.

        WHY STOCHASTIC RATHER THAN STEADY (v3, measured not assumed)
        -----------------------------------------------------------
        Steady wind is nearly useless as a disturbance here. A PID with integral
        action cancels a constant force exactly: the integrator winds up and
        holds the offset. Measured with a pure-PID policy:

            steady 3 m/s   ->  seed std 1.66
            gusty          ->  seed std 14.6      (~9x more)

        A critic can only help by explaining variance the actor cannot predict.
        With a steady disturbance there is none, so every rung of the ladder
        performs identically and the k-sweep is flat for reasons unrelated to
        Claim 1. The same was true of downwash: real force (0.155 N at 0.7 m,
        47% of spare thrust) but perfectly steady, so the PID absorbed it and
        seed variance was 0.00.

        THE MODEL: Ornstein-Uhlenbeck process, per drone
        ------------------------------------------------
            dg = -(1/tau) * g * dt  +  sigma * sqrt(dt) * noise

        A mean-reverting random walk. Produces smooth, temporally correlated
        fluctuation on a timescale of `tau` seconds -- which is what real air
        does, rather than white noise.

        SPATIAL CORRELATION -- the parameter that matters most for this paper
        --------------------------------------------------------------------
        Each drone's gust is a blend of a SHARED component and an INDEPENDENT
        one, mixed by `gust_spatial_corr`:

            corr = 0  ->  every drone gets independent noise. A neighbour's
                          state then tells you NOTHING about your own
                          disturbance, so teammate information is worthless
                          to the critic.
            corr = 1  ->  all drones feel exactly the same gust. A neighbour
                          being pushed PERFECTLY predicts that you are being
                          pushed.

        Real air is in between: a gust front sweeping through the room hits
        nearby drones at correlated times. That correlation is precisely the
        mechanism by which a teammate-aware critic can explain variance a local
        critic cannot -- so this is a knob worth sweeping in its own right,
        much as formation spacing is for Claim 2.
        """
        import pybullet as p

        # Crazyflie 2.x body drag constants.
        #
        # FRONTAL_AREA corrected 2026-08-31: the previous 0.01 m^2 modelled the
        # drone as a solid 10x10 cm plate. A CF2.x is 92 mm across but mostly
        # OPEN AIR (small body, thin arms, tiny motors) -- about 0.0025 m^2.
        # The old value was ~4x too large, which combined with an unrealistic
        # 12 m/s wind to give 3.3x body weight -- requiring ~73 deg of tilt,
        # beyond the aircraft's physical limit of ~63 deg.
        AIR_DENSITY = 1.225        # kg/m^3, sea level
        DRAG_COEFF = 1.0           # bluff body, standard for an open frame
        FRONTAL_AREA = 0.0025      # m^2, Crazyflie 2.x estimated

        # ---- Apply body drag from relative airspeed -------------------------
        #
        # SUBSTEP FIX (16 Sep 2026). This is called once per PHYSICS SUBSTEP,
        # not once per control step.
        #
        # THE BUG: `_apply_wind()` used to be called from `_preprocessAction()`,
        # which runs ONCE per control step. But `BaseAviary.step()` then executes
        # `PYB_STEPS_PER_CTRL` physics substeps, and PyBullet CLEARS accumulated
        # external forces after every `stepSimulation()`. So the wind force acted
        # during only the FIRST substep -- delivering 1/5 of the intended impulse
        # at ctrl_freq=48, and making the disturbance silently dependent on a
        # simulation setting rather than on physics.
        #
        # The OU gust state advances once per CONTROL step (in
        # `_advance_gusts()`), because that is its natural timescale; the FORCE
        # it implies is applied on every substep, because that is how forces work.
        
        for i in range(self.NUM_DRONES):
            state = self._getDroneStateVector(i)
            drone_vel = state[10:13]

            air_velocity = self._wind_vec + self._gusts[i]
            v_rel = air_velocity - drone_vel
            speed = np.linalg.norm(v_rel)
            if speed < 1e-6:
                continue

            force = 0.5 * AIR_DENSITY * DRAG_COEFF * FRONTAL_AREA * speed * v_rel

            # FRAME NOTE (two bugs fixed here, 2026-08-31):
            #
            # 1. The force must be applied in WORLD_FRAME. With LINK_FRAME the
            #    wind direction rotated with the drone, so a tilting drone could
            #    partly "escape" the wind -- which is not how air works.
            #
            # 2. With WORLD_FRAME, `posObj` is ALSO world coordinates. Passing
            #    [0,0,0] applied the force at the world ORIGIN, giving a lever
            #    arm equal to the drone's altitude. Observed: a 0.03 N gust on a
            #    drone at 3.8 m spun it from 0.27 to 1.70 rad of roll in three
            #    control steps. The force was negligible; the moment arm was not.
            #    Passing the drone's own position applies it at the centre of
            #    mass, giving pure translation.
            p.applyExternalForce(
                self.DRONE_IDS[i],
                -1,                                  # base link
                forceObj=force.tolist(),
                posObj=state[0:3].tolist(),
                flags=p.WORLD_FRAME,
                physicsClientId=self.CLIENT,
            )

    def _advance_gusts(self) -> None:
        """Advance the OU gust process by one CONTROL step.

        Separated from force application because the two have different natural
        rates: the gust field evolves on the control timescale (tau ~ 1 s), but
        the force it exerts must be applied on every physics substep or the
        delivered impulse depends on `ctrl_freq`.
        """
        dt = 1.0 / self.CTRL_FREQ

        # One shared draw plus one independent draw per drone, blended so the
        # resulting cross-drone correlation equals GUST_SPATIAL_CORR.
        shared = self._rng.normal(size=3)
        independent = self._rng.normal(size=(self.NUM_DRONES, 3))

        w = np.sqrt(self.GUST_SPATIAL_CORR)
        blended = w * shared[None, :] + np.sqrt(1.0 - self.GUST_SPATIAL_CORR) * independent

        self._gusts += (
            -self._ou_theta * self._gusts * dt
            + self._ou_sigma * np.sqrt(dt) * blended
        )

    def _physics(self, rpm: np.ndarray, nth_drone: int) -> None:
        """Base physics plus our wind force.

        `BaseAviary` calls `_physics()` once per drone per PHYSICS SUBSTEP, so
        this is the correct hook for a force that must act continuously. See the
        substep note in `_apply_wind()`.
        """
        super()._physics(rpm, nth_drone)
        if nth_drone == 0:
            # Apply to all drones once per substep, on the first drone's call.
            self._apply_wind()

    def _preprocessAction(self, action: np.ndarray) -> np.ndarray:
        """Convert commanded offsets into rotor RPMs via the inner PID loop.

        Pipeline:
            action in [-1,1]^3  ->  offset in metres
            slot + offset       ->  target position
            DSL PID             ->  rotor RPMs

        Wind is applied here because `_preprocessAction` is called once per
        control step, immediately before the physics substeps run.
        """
        # Advance the gust field once per CONTROL step; the force it implies is
        # applied on every physics substep via `_physics()`.
        self._advance_gusts()

        action = np.asarray(action, dtype=np.float64).reshape(self.NUM_DRONES, 3)
        action = np.clip(action, -1.0, 1.0)

        # v3: the PID must aim at the CHAIN reference, not the analytical slot.
        # If it aimed at the formula, the inner loop would still be secretly
        # using global trajectory knowledge and the chain topology would be
        # cosmetic.
        slots = self._reference_positions()
        # FEEDFORWARD (added 17 Sep 2026). `computeControl` was previously given
        # target_pos ONLY, so its desired velocity defaulted to ZERO -- while
        # the leader tracks a moving circle and every follower tracks a moving
        # predecessor. The inner loop was therefore asked to hold station on a
        # target it was simultaneously told to chase, which costs avoidable
        # tracking lag proportional to reference speed.
        #
        # This mattered beyond control quality. A one-parameter P-controller on
        # the gap error beat the unaided PID by +109 return / 28% lower RMS, and
        # that gap was read as evidence of an RL EXPLORATION problem. Most of it
        # is the missing feedforward: the P term was substituting for velocity
        # feedback the controller should have had. Diagnosing the learner
        # against a deliberately handicapped baseline would have been wrong.
        #
        # `_reference_velocities()` already existed for the observation; the
        # leader gets its trajectory derivative, each follower the MEASURED
        # velocity of the drone above it (no global knowledge -- see that
        # method's docstring).
        slot_vels = self._reference_velocities()
        rpms = np.zeros((self.NUM_DRONES, 4))

        for i in range(self.NUM_DRONES):
            target_pos = slots[i] + action[i] * self.MAX_OFFSET
            state = self._getDroneStateVector(i)

            rpm, _, _ = self._controllers[i].computeControl(
                control_timestep=self.CTRL_TIMESTEP,
                cur_pos=state[0:3],
                cur_quat=state[3:7],
                cur_vel=state[10:13],
                cur_ang_vel=state[13:16],
                target_pos=target_pos,
                target_vel=slot_vels[i],
            )
            rpms[i, :] = rpm

        return rpms

    # ==========================================================================
    # SECTION 4 : FORMATION GEOMETRY
    # ==========================================================================

    def _leader_target(self, t: float) -> np.ndarray:
        """Leader's commanded position at time `t`.

        A slow circle plus a gentle vertical sinusoid. Followers cannot predict
        it from their own observations, and the stack stays roughly vertical so
        followers remain inside the leader's wake.
        """
        radius, period = 0.5, 12.0
        omega = 2.0 * np.pi / period
        return np.array([
            radius * np.cos(omega * t) - radius,
            radius * np.sin(omega * t),
            self.LEADER_ALTITUDE + 0.2 * np.sin(0.5 * omega * t),
        ])

    def _formation_targets(self, t: float) -> np.ndarray:
        """Commanded slot for every drone: a vertical stack beneath the leader."""
        leader = self._leader_target(t)
        return np.array([
            [leader[0], leader[1], leader[2] - i * self.FORMATION_SPACING]
            for i in range(self.NUM_DRONES)
        ])

    @property
    def _sim_time_sec(self) -> float:
        """Simulation time AFTER the physics just advanced.

        POST-STEP TIME, NOT PRE-STEP  (fixed 17 Sep 2026)
        -------------------------------------------------
        Upstream `BaseAviary.step()` calls `_computeObs()`, `_computeReward()`,
        `_computeTerminated()` and `_computeTruncated()` and only THEN does

            self.step_counter += PYB_STEPS_PER_CTRL

        So during those four calls `step_counter` still holds the time BEFORE
        the step whose results they are describing. Reading it directly meant
        the reference position, the reward and the time limit were all evaluated
        one control step in the past.

        Combined with a strict `>` in the truncation test, a 15 s episode at
        48 Hz ran 722 steps rather than 720 -- which is why "episode completion"
        was reported above 100%.

        Adding one control step's worth of substeps here makes the property mean
        what its callers assume: the time that now holds, after the physics.
        """
        return (self.step_counter + self.PYB_STEPS_PER_CTRL) * self.PYB_TIMESTEP

    # ==========================================================================
    # SECTION 5 : OBSERVATIONS
    # ==========================================================================

    def _reference_positions(self) -> np.ndarray:
        """Where each drone SHOULD be, and against what it is measured.

        THE CHAIN TOPOLOGY (v3). This is the honest-decentralization fix.

        v2 computed every drone's error against `_formation_targets(t)` -- the
        ANALYTICAL leader trajectory. That meant a follower implicitly knew the
        leader's entire future path, which no real drone can. "Decentralized
        execution" was not honest, and the followers were not really following:
        each independently tracked its own slot on a pre-computed curve, so the
        leader could have been switched off entirely.

        v3 CHAIN:
            leader (drone 0)  -> its own absolute trajectory (it has the mission)
            follower i        -> the ACTUAL position of drone i-1, minus the gap

        Every follower quantity is now a RELATIVE measurement to the drone
        immediately above it, ~1 m away -- exactly what UWB ranging or onboard
        vision provides. No global frame, no trajectory formula, no radio link.

        WHY CHAIN RATHER THAN STAR (all followers tracking the leader):
          * sensing range stays constant as the team grows (star needs N x
            spacing, through occluding drones);
          * the drone you SENSE is the drone whose wake hits you, so sensing and
            physics align;
          * it produces ERROR PROPAGATION down the chain -- a lag at drone 1
            perturbs 2, which perturbs 3. That is string-stability behaviour,
            and it is precisely where teammate information becomes valuable to
            the critic. Star topology makes followers independent, leaving far
            less for a centralized critic to explain.
        """
        if not self.CHAIN_TOPOLOGY:
            # v2 behaviour, kept for ablation: everyone against the formula.
            return self._formation_targets(self._sim_time_sec)

        references = np.zeros((self.NUM_DRONES, 3))

        # Leader: its own commanded trajectory.
        references[0] = self._leader_target(self._sim_time_sec)

        # Followers: the ACTUAL position of the drone above, minus the gap.
        for i in range(1, self.NUM_DRONES):
            predecessor = self._getDroneStateVector(i - 1)[0:3]
            references[i] = predecessor - np.array([0.0, 0.0, self.FORMATION_SPACING])

        return references

    def _reference_velocities(self) -> np.ndarray:
        """Velocity of each drone's reference -- the derivative counterpart to
        `_reference_positions()`.

        ADDED 16 Sep 2026 to close an observation asymmetry (audit #10).

        The actor previously received relative POSITION but its own ABSOLUTE
        velocity, so being memoryless it could not compute d(gap)/dt. The k=1
        critic received relative velocity directly -- meaning k=1's measured
        advantage might have been "the critic has a derivative the actor lacks"
        rather than "the critic sees coupling". That is an alternative
        explanation for the entire pilot result, so it had to be removed.

        A real follower with UWB or vision can estimate relative velocity, so
        withholding it was not realistic either.

        leader    -> its commanded trajectory velocity (finite difference)
        follower  -> the ACTUAL velocity of the drone above it
        """
        references = np.zeros((self.NUM_DRONES, 3))

        if not self.CHAIN_TOPOLOGY:
            # v2 behaviour: everyone against the analytical trajectory.
            t = self._sim_time_sec
            dt = 1.0 / self.CTRL_FREQ
            leader_vel = (self._leader_target(t + dt) - self._leader_target(t)) / dt
            references[:] = leader_vel
            return references

        # Leader: derivative of its own commanded trajectory.
        t = self._sim_time_sec
        dt = 1.0 / self.CTRL_FREQ
        references[0] = (self._leader_target(t + dt) - self._leader_target(t)) / dt

        # Followers: the measured velocity of the drone above.
        for i in range(1, self.NUM_DRONES):
            references[i] = self._getDroneStateVector(i - 1)[10:13]

        return references

    def _computeObs(self) -> np.ndarray:
        # v3: chain references -- follower errors are relative to the drone
        # above them, not to an analytical trajectory. See _reference_positions.
        targets = self._reference_positions()
        target_vels = self._reference_velocities()
        obs = np.zeros((self.NUM_DRONES, 13), dtype=np.float32)

        for i in range(self.NUM_DRONES):
            state = self._getDroneStateVector(i)

            obs[i, 0:3] = state[0:3] - targets[i]       # gap error
            obs[i, 3:6] = state[7:10]                   # roll, pitch, yaw

            # RELATIVE velocity (audit #10). Was absolute velocity, which left
            # the actor unable to compute the gap derivative while the k=1
            # critic could -- a confound for the whole result.
            obs[i, 6:9] = state[10:13] - target_vels[i]

            obs[i, 9:12] = state[13:16]                 # angular velocity

            # ROLE FLAG (audit #11). Leader and follower observations MEAN
            # different things: the leader's error is absolute mission tracking,
            # a follower's is a gap to the drone above. With a shared actor and
            # no role indicator the two regimes are aliased, and the network
            # cannot tell which it is in. One bit resolves it.
            obs[i, 12] = 1.0 if i == 0 else 0.0

        return obs

    def get_global_state(self) -> dict:
        """Privileged information for the CENTRALIZED CRITIC only.

        Never given to actors. The observability ladder draws from this:
            positions/velocities -> teammate state for k >= 1
            wind_vec             -> privileged variable for the k=N-1+P rung
            coupling             -> realised downwash, for logging and for
                                    verifying the Claim 2 spacing knob
        """
        positions = np.array([self._getDroneStateVector(i)[0:3] for i in range(self.NUM_DRONES)])
        velocities = np.array([self._getDroneStateVector(i)[10:13] for i in range(self.NUM_DRONES)])

        return {
            "positions": positions,
            "velocities": velocities,
            # The STEADY MEAN wind.
            "wind_vec": self._wind_vec.copy(),
            # THE ACTUAL STOCHASTIC GUST, per drone (added 16 Sep 2026).
            #
            # This was the bug that made the "+P" rung not privileged. Only
            # `wind_vec` was exposed -- the steady mean -- while the OU gust
            # that constitutes essentially the entire disturbance was withheld.
            #
            # Measured before the fix, at gust_std_ms=4.0 after 50 steps:
            #     wind_vec (given)      [ 0.37,  0.87,  0.10]
            #     gusts    (withheld)   [-1.30, -10.37, 3.09]  ...
            #
            # So the rung labelled "maximal centralization" was missing the
            # dominant term, and any conclusion drawn from it was unsound.
            "gusts": self._gusts.copy(),
            # Total air velocity actually acting on each drone: mean + gust.
            # This is what the drag force is computed from, so it is the honest
            # "true disturbance" signal for a privileged critic.
            "air_velocity": self._wind_vec[None, :] + self._gusts,
            # Realised pairwise downwash forces.
            "coupling": coupling_matrix(positions),
        }

    # ==========================================================================
    # SECTION 6 : REWARD  (FIX 2 and FIX 3)
    # ==========================================================================

    def _computeReward(self) -> np.ndarray:
        """Per-agent reward.

            + survival bonus                      (FIX 2: existing beats dying)
            - lateral_weight * lateral_error^2    (FIX 3: closes the sidestep)
            - vertical_error^2
            - small tilt penalty
            - crash penalty scaled by time left   (FIX 2)

        The lateral weight is what makes holding the slot beat dodging the
        wake. Verified numerically by scripts/verify_task_fixes.py.
        """
        # v3: same chain references the observation uses, so the reward is
        # consistent with what the agent can actually perceive. Rewarding a
        # quantity the actor cannot observe would make the task partially
        # unlearnable by construction.
        targets = self._reference_positions()
        rewards = np.zeros(self.NUM_DRONES, dtype=np.float64)

        for i in range(self.NUM_DRONES):
            state = self._getDroneStateVector(i)

            # FIX 2: a per-step bonus makes surviving strictly better than not.
            rewards[i] += self.SURVIVAL_BONUS

            # FIX 3: lateral error is penalised LATERAL_WEIGHT times harder than
            # vertical error, so stepping out of the wake never pays.
            lateral_err = np.linalg.norm(state[0:2] - targets[i][0:2])
            vertical_err = abs(state[2] - targets[i][2])

            rewards[i] -= self.LATERAL_WEIGHT * lateral_err ** 2
            rewards[i] -= vertical_err ** 2

            # Mild attitude penalty. Small on purpose: tilting is REQUIRED to
            # translate, so punishing it hard would fight the task.
            rewards[i] -= 0.05 * np.linalg.norm(state[7:9]) ** 2

        # FIX 2: crashing costs more the earlier it happens, so the return can
        # never be improved by ending the episode sooner.
        #
        # CALL-ORDER BUG, FIXED 16 Sep 2026 -- this is the important part.
        #
        # `BaseAviary.step()` calls, in this order:
        #       line 377   reward     = self._computeReward()
        #       line 378   terminated = self._computeTerminated()
        # and `_computeTerminated()` was what set `self._crashed`. So on the
        # crashing step the reward was computed while the flag was still False,
        # the episode then ended, and the penalty was NEVER DELIVERED.
        #
        # Measured before the fix: a real dive-to-crash episode returned +0.88
        # on its final step -- identical to the step before. A configured
        # penalty of 50.0 had no effect whatsoever.
        #
        # The old verification "passed" because it set `env._crashed = True` by
        # hand and called `_computeReward()` directly, testing the formula while
        # bypassing the ordering that broke it.
        #
        # THE FIX: evaluate the failure condition HERE rather than depending on
        # `_computeTerminated()` having run first. `_check_failure()` is pure --
        # it reads state and returns a verdict without mutating anything -- so
        # calling it from both places is safe and order-independent.
        failed, reason = self._check_failure()
        if failed:
            self._crashed = True
            self._failure_reason = reason

        if self._crashed:
            # PENALTY SCALE, RECALIBRATED 16 Sep 2026.
            #
            # The previous formula scaled by the WORST-CASE per-step return at
            # the 3 m termination boundary, giving penalties of ~58,000 for an
            # early crash against a normal episode return of ~650 -- two orders
            # of magnitude. With no value normalisation, a single failure could
            # dominate the critic's targets and is a plausible cause of the
            # late-training collapses seen in the v3 pilot.
            #
            # The property we actually need is only that return be MONOTONE in
            # survival time: crashing must never beat surviving. That requires
            # forfeiting the remaining survival bonus plus a fixed margin -- not
            # a worst-case-error extrapolation the agent will never occupy.
            remaining_frac = max(0.0, 1.0 - self._sim_time_sec / self.EPISODE_LEN_SEC)
            remaining_steps = remaining_frac * self.EPISODE_LEN_SEC * self.CTRL_FREQ

            # FIXED-HORIZON FAILURE COST  (corrected 17 Sep 2026)
            #
            # The previous formula forfeited SURVIVAL_BONUS (1.0) per remaining
            # step plus a fixed 50. That does NOT guarantee monotonicity, and
            # the gap was measured, not hypothesised: a drone flying badly but
            # legally scores as low as -1.07 per step while ALIVE. Crashing
            # forfeited only 1.00 per step, so a persistently-bad agent could
            # PROFIT by terminating early -- it avoided 1.07/step of tracking
            # cost at a price of 1.00/step. That is the "learn to die" failure
            # mode that killed v1, reachable again through a different door.
            #
            # The fix charges every remaining step at a rate strictly worse than
            # the worst survivable per-step reward. The episode then behaves as
            # if it ran to the full horizon in an absorbing failure state, so
            # ending early is always at least as expensive as continuing, for
            # ANY policy -- no case analysis required.
            #
            # FAILURE_STEP_COST is set below the measured worst survivable rate
            # with margin, so the guarantee does not depend on that measurement
            # staying exactly true as the reward is retuned.
            rewards -= self.CRASH_PENALTY + self.FAILURE_STEP_COST * remaining_steps

        return rewards

    # ==========================================================================
    # SECTION 7 : TERMINATION
    # ==========================================================================

    def _check_failure(self) -> tuple[bool, str]:
        """Evaluate the failure condition WITHOUT mutating state.

        PURE BY DESIGN. Both `_computeReward()` and `_computeTerminated()` call
        this, so the verdict cannot depend on which of them PyBullet happens to
        invoke first. That ordering dependence is exactly what made the crash
        penalty dead code before 16 Sep 2026.

        Returns
        -------
        (failed, reason)
        """
        positions = np.array([self._getDroneStateVector(i)[0:3] for i in range(self.NUM_DRONES)])

        for i in range(self.NUM_DRONES):
            state = self._getDroneStateVector(i)

            if state[2] < 0.1:                                   # hit the floor
                return True, "ground"
            if abs(state[7]) > 1.2 or abs(state[8]) > 1.2:       # tipped past ~70 deg
                return True, "attitude"

        # ---- INTER-DRONE COLLISION (added 16 Sep 2026) ----------------------
        # The charter lists collision as a termination condition, but the
        # implementation checked only floor contact, tilt, and distance from the
        # reference. Two drones could therefore occupy the same space with
        # nothing recorded -- and `crash_rate` was reported as if it measured
        # collisions, which it did not.
        for i in range(self.NUM_DRONES):
            for j in range(i + 1, self.NUM_DRONES):
                if np.linalg.norm(positions[i] - positions[j]) < self.COLLISION_RADIUS_M:
                    return True, "collision"

        targets = self._reference_positions()
        for i in range(self.NUM_DRONES):
            if np.linalg.norm(positions[i] - targets[i]) > 3.0:
                return True, "diverged"

        return False, ""

    def _computeTerminated(self) -> bool:
        """True on failure.

        Delegates to the pure `_check_failure()` so that the reward function
        reaches the identical verdict regardless of call order.
        """
        failed, reason = self._check_failure()
        if failed:
            self._crashed = True
            self._failure_reason = reason
        return failed

    def _computeTruncated(self) -> bool:
        """Time limit reached -- a success, not a failure.

        `>=`, NOT `>`  (fixed 17 Sep 2026)
        ----------------------------------
        With `>`, a 15 s episode at 48 Hz ran 722 steps instead of 720.

        After control step n, `_sim_time_sec` is exactly n/CTRL_FREQ (verified:
        the property tracks the post-step time correctly, and the substep
        counter advances by exactly PYB_STEPS_PER_CTRL each step). At n = 720,
        `1.0 * EPISODE_LEN_SEC > EPISODE_LEN_SEC` is FALSE, so the episode ran
        on. Floating-point accumulation then let it stop two steps later.

        The consequences were small individually and corrosive together:
          * "episode completion" exceeded 100% (reported as 100.28%), which is
            nonsense on its face and was visible in several result tables;
          * `remaining_steps` in the crash penalty was computed against a
            horizon the episode did not actually have;
          * per-episode metrics averaged over a horizon 0.3% longer than the
            one documented everywhere else.
        """
        return self._sim_time_sec >= self.EPISODE_LEN_SEC

    def _computeInfo(self) -> dict:
        """Diagnostics. `mean_lateral_error` is the sidestep-exploit monitor."""
        positions = np.array([self._getDroneStateVector(i)[0:3] for i in range(self.NUM_DRONES)])
        coupling = coupling_matrix(positions)
        targets = self._reference_positions()

        lateral_errors = [
            float(np.linalg.norm(positions[i, 0:2] - targets[i][0:2]))
            for i in range(self.NUM_DRONES)
        ]

        return {
            # Manipulation check for Claim 2: confirms spacing changed the physics.
            "total_coupling_N": float(coupling.sum()),
            "max_coupling_N": float(coupling.max()),
            # FIX 3 monitor: if trained policies drift laterally, the sidestep
            # exploit is back and results are invalid.
            "mean_lateral_error_m": float(np.mean(lateral_errors)),
            "wind_speed_ms": float(np.linalg.norm(self._wind_vec)),
            "crashed": self._crashed,
            # WHY it failed -- "ground", "attitude", "collision", "diverged".
            # Previously `crash_rate` was reported as if it measured collisions;
            # it did not, because collisions were never detected at all.
            "failure_reason": self._failure_reason,
            "collision": self._failure_reason == "collision",
            # Closest approach between any two drones this step. A safety metric
            # in its own right, and reportable in metres for a robotics venue.
            "min_separation_m": float(min(
                (np.linalg.norm(positions[i] - positions[j])
                 for i in range(self.NUM_DRONES)
                 for j in range(i + 1, self.NUM_DRONES)),
                default=np.inf,
            )),
        }

    # ==========================================================================
    # SECTION 8 : RESET
    # ==========================================================================

    def reset(self, seed: int | None = None, options: dict | None = None):
        """Reset, resampling the wind and clearing crash state."""
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._crashed = False
        self._failure_reason = ""
        self._gusts = np.zeros((self.NUM_DRONES, 3), dtype=np.float64)

        # Wind is resampled every episode and observed by NO agent. It is a
        # hidden common cause: it pushes all drones at once, so a teammate's
        # velocity carries information about a disturbance the agent cannot
        # sense directly -- exactly the variance a centralized critic could
        # explain away (Charter §1.5).
        direction = self._rng.normal(size=3)
        direction[2] *= 0.3                       # mostly horizontal
        direction /= max(np.linalg.norm(direction), 1e-9)
        self._wind_vec = direction * self._rng.uniform(0.0, self.WIND_MEAN_MS)

        result = super().reset(seed=seed, options=options)

        # PID controllers hold integral state; it must not leak across episodes.
        for controller in self._controllers:
            controller.reset()

        return result
