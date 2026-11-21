"""
================================================================================
 TRAINING LOOP  --  one run = one (rung, seed) pair
================================================================================

PHASE 3 (Charter §5).

--------------------------------------------------------------------------------
 WHAT ONE "RUN" IS
--------------------------------------------------------------------------------
A run trains a single policy at a single observability rung with a single seed,
under a FIXED sample budget, and records its learning curve.

The full experiment is a grid of runs:

    rungs (k0..k3+P)  x  seeds (>=8)  x  formation spacings (Claim 2)

Charter §7 requires the sample budget to be IDENTICAL across rungs. Otherwise a
high-k run could simply be given more experience, and the comparison would be
meaningless.

--------------------------------------------------------------------------------
 WHY WE EVALUATE SEPARATELY FROM TRAINING
--------------------------------------------------------------------------------
Training episodes use a STOCHASTIC policy (actions sampled for exploration), so
training reward is depressed by exploration noise and is a poor measure of what
the policy has actually learned.

Evaluation runs the DETERMINISTIC policy (distribution mean) on fresh episodes.

--------------------------------------------------------------------------------
 THREE DISJOINT SEED SETS  (v4-correctness)
--------------------------------------------------------------------------------
    TRAINING   per run   what the policy experienced
    PERIODIC   per run   the "is it learning?" curve, used to judge run health
    FINAL      COMMON    the number reported in the paper

Reporting on the periodic seeds would be selecting on the test set.

The FINAL set is deliberately the SAME for every run, rung and baseline (common
random numbers), so a difference between two policies is a difference between
POLICIES rather than between the disturbances they happened to be dealt. Per-run
final seeds previously made one healthy policy look broken -- see
`RunConfig.final_eval_seeds` for that measurement and for what must be reported
separately as a result.

--------------------------------------------------------------------------------
 WHAT GETS REPORTED
--------------------------------------------------------------------------------
Return is a TRAINING variable. Its meaning changed with every reward revision
(three so far), so it cannot be compared across task versions and a reviewer
cannot judge it. Every evaluation episode is therefore recorded by an
`EpisodeRecorder`, and the results carry physical units: RMS formation error in
centimetres, minimum separation in metres, collision rate, and downstream error
amplification G. Charter §S.11; see src/ctde/metrics.py.
================================================================================
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from ctde.envs.formation_aviary import FormationAviary
from ctde.metrics import EpisodeMetrics, EpisodeRecorder, summarise_amplification
from ctde.models.observability import ObservabilityLevel
from ctde.training.mappo import MAPPO, MAPPOConfig
from ctde.utils.run_naming import TASK_VERSION
from ctde.utils.seeding import make_rng, seed_everything

# ==============================================================================
# THE COMMON EVALUATION EPISODE SET
# ==============================================================================
# Every reported number -- every rung, every training seed, every baseline
# controller -- is measured on episodes drawn from this one block, so that
# comparisons differ by POLICY and not by which disturbances were drawn.
#
# Chosen far from the training seeds (0..~50) and from the periodic evaluation
# blocks (seed*1000 + 10_000), so no run can train on or monitor with an episode
# it is later scored on.
FINAL_EVAL_SEED_BASE = 900_000


# ==============================================================================
# SECTION 1 : RUN CONFIGURATION
# ==============================================================================

@dataclass
class RunConfig:
    """Everything defining one run."""

    # ---- The independent variable -------------------------------------------
    level: ObservabilityLevel

    # ---- Experimental conditions --------------------------------------------
    seed: int = 0
    num_agents: int = 4
    formation_spacing_m: float = 1.0        # Claim 2's coupling knob

    # ---- Budget (IDENTICAL across rungs -- Charter §7) -----------------------
    total_env_steps: int = 100_000

    # ---- Environment --------------------------------------------------------
    episode_len_sec: float = 15.0
    # ---- Disturbance (v3) ---------------------------------------------------
    # Steady wind was measured to be useless: a PID cancels a constant force,
    # so seed variance was 0.00 and there was nothing for a critic to explain.
    # v3 uses a stochastic gust field instead. Parameters are in PHYSICAL units.
    #
    # RECALIBRATED TWICE. Current value set 17 Sep 2026.
    #
    # ROUND 1 (16 Sep): sigma was 4.0, chosen when wind reached the drones on
    # 1 of 5 physics substeps (~20% strength). v4 applies it on all 5, so the
    # real disturbance was ~5x the calibration. A PURE PID crashed 40% of
    # episodes -- the first v4 learnability run failed for that reason alone.
    # Lowered to 3.0.
    #
    # ROUND 2 (17 Sep): 3.0 was itself measured on only 12 episodes, and against
    # a PID that was never given target_vel (desired velocity defaulted to zero
    # while tracking moving references). Both were wrong.
    #
    # With feedforward added and 24 common episodes:
    #
    #   sigma   return   seed_std   crashes    RMS
    #   1.5     704.2       3.2      0/24     5.98 cm
    #   2.0     688.0       7.7      0/24     7.32 cm
    #   2.5     656.5      17.7      0/24     9.65 cm   <- chosen
    #   3.0     465.2     295.6      6/24    13.26 cm
    #
    # 2.5 gives ~10x the seed variance of the sigma=0 baseline (17.7 vs 1.73)
    # with ZERO crashes across the full evaluation block. 3.0's apparent
    # seed_std of 295.6 is crash variance: that is DICE, not skill, and a critic
    # cannot learn to predict it.
    #
    # THE REPEATED MISTAKE, RECORDED. Round 1's error was calibrating against a
    # bug. Round 2's was calibrating on too few episodes -- the same small-sample
    # error already diagnosed in the EVALUATION protocol, made again in the
    # CALIBRATION. sigma=3.0 looked crash-free on 8 episodes; all six crashes
    # sit in episodes 8-23. Calibration is a measurement and needs the same
    # sample size discipline as any other.
    #
    # WHAT sigma=2.5 MEANS PHYSICALLY. It is the stationary standard deviation
    # of EACH CARTESIAN COMPONENT, not the gust speed. The vector speed is
    # RMS||g|| = sqrt(3)*sigma ~= 4.3 m/s. Quote the vector speed in the paper:
    # that is what an anemometer reads.
    wind_mean_ms: float = 1.0
    gust_std_ms: float = 2.5
    gust_tau_sec: float = 1.0
    gust_spatial_corr: float = 0.7

    # v3 chain topology: followers observe their gap to the drone ABOVE them,
    # not an analytical trajectory. See FormationAviary._reference_positions.
    chain_topology: bool = True
    base_altitude_m: float = 0.8
    ctrl_freq: int = 48

    # ---- Task shaping (v4-correctness) ----------------------------------------
    # Defaults match the values verified by scripts/verify_task_fixes.py.
    # Changing any of these changes the TASK, not just a hyperparameter --
    # bump TASK_VERSION in utils/run_naming.py if you do.
    lateral_weight: float = 10.0
    survival_bonus: float = 1.0
    crash_penalty: float = 50.0
    max_offset_m: float = 0.2        # calibrated with actor log_std -- see networks.py

    # ---- Evaluation ---------------------------------------------------------
    # EPISODE COUNTS RAISED 16 Sep 2026, after measuring what 3 episodes can
    # actually resolve. THE PROBLEM: a single early crash costs ~-650 return,
    # so one crash in a 3-episode block swings the reported score by ~350 --
    # more than THREE TIMES the entire P-control-vs-PID effect (+109).
    #
    # This was not hypothetical. Training seed 1 scored 130 where seeds 0 and 2
    # scored 450-545, and it was blamed on bad learning. It was not: the PURE
    # PID scores 119.7 on seed 1's eval block, because episode 11000 crashes at
    # step 87 and returns -612. The policy was being graded on a harder exam.
    #
    # Return is BIMODAL (crash vs no-crash), with std ~263 across the mixture,
    # so even n=40 cannot resolve +109 on return alone. RMS position error is
    # not bimodal and discriminates far better:
    #
    #     metric       Cohen's d    n for 80% power
    #     return          1.42            8
    #     RMS error       3.07            2
    #
    # Hence: more episodes AND physical metrics as the headline (Charter §S.11).
    eval_every_steps: int = 10_000
    eval_episodes: int = 8
    # The FINAL reported number uses its own, larger, disjoint seed set. See
    # `final_eval_seeds` below for why the two sets must not overlap. 24 gives a
    # usable CI on return and a tight one on RMS error.
    final_eval_episodes: int = 24

    # ---- Learner ------------------------------------------------------------
    mappo: MAPPOConfig = field(default_factory=MAPPOConfig)

    # ==========================================================================
    # EVALUATION SEED POLICY  (fixed 16 Sep 2026)
    # ==========================================================================
    # Three seed sets, pairwise disjoint:
    #
    #   TRAINING      `seed`                    -- what the policy experienced
    #   PERIODIC      seed*1000 + 10_000 + ep   -- "is it learning?" curve
    #   FINAL         seed*1000 + 50_000 + ep   -- the number that goes in the paper
    #
    # Two bugs this closes:
    #
    #  1. `seed + 10_000 + ep` COLLIDES ACROSS RUNS. Training seed 1 with ep=0
    #     gives 10_001, and training seed 0 with ep=1 also gives 10_001. Seeds
    #     meant to be distinct per run were shared between runs, so the
    #     across-seed spread we report as variance was partly an artefact of
    #     re-scoring on overlapping disturbances. Multiplying by 1000 spaces the
    #     blocks far enough apart that no offset can reach the next one.
    #
    #  2. SELECTING ON THE TEST SET. The periodic curve is what we look at to
    #     decide a run is healthy; reporting the final number on those same
    #     episodes scores the model on data used to judge it.
    #
    # The final set is deliberately larger (10 vs 3): it is the number with a
    # confidence interval attached, and 3 episodes cannot support one.
    # ==========================================================================

    def eval_seeds(self) -> list[int]:
        """Seeds for the PERIODIC (during-training) evaluation.

        PER-RUN, deliberately. These drive "is this run healthy?", and giving
        each run its own episodes keeps that judgement independent across runs.
        They are NEVER reported.
        """
        base = self.seed * 1000 + 10_000
        return [base + ep for ep in range(self.eval_episodes)]

    def final_eval_seeds(self) -> list[int]:
        """Seeds for the FINAL (reported) evaluation.

        COMMON RANDOM NUMBERS  (fixed 17 Sep 2026)
        ------------------------------------------
        The SAME episodes for every policy, every rung, and every baseline --
        deliberately NOT keyed on `self.seed`.

        WHY THIS REPLACED PER-RUN FINAL SEEDS. The previous scheme gave each
        training seed its own final episodes, which quietly added EPISODE
        DIFFICULTY to what we report as between-run variance. The effect was
        large and we measured it: training seed 1 scored 130 where others scored
        450-545, and the cause was that its evaluation block contained an
        episode crashing at step 87 (-612 return). Scoring that same block with
        the PURE PID gives 119.7 -- the policy was fine, the exam was harder.

        With common episodes, every policy faces exactly the same disturbances,
        so a difference between two policies is a difference between POLICIES.
        This is a standard variance-reduction technique (common random numbers);
        it does not understate uncertainty, it removes a nuisance variable from
        the comparison.

        Two sources of variation must then be reported SEPARATELY:
          * across training seeds  -- the algorithm's run-to-run spread, which
            is the primary experimental unit for any claim about k;
          * across evaluation episodes -- deployment variability, reported per
            episode, never pooled into the seed-level uncertainty.

        Per-episode records are saved (`final_eval_episodes.json`) so paired
        tests across policies on identical episodes are possible after the fact.
        """
        return [FINAL_EVAL_SEED_BASE + ep for ep in range(self.final_eval_episodes)]

    def as_dict(self) -> dict:
        return {
            "rung": self.level.label,
            "level_k": self.level.k,
            "level_privileged": self.level.include_privileged,
            "seed": self.seed,
            "num_agents": self.num_agents,
            "formation_spacing_m": self.formation_spacing_m,
            "total_env_steps": self.total_env_steps,
            "episode_len_sec": self.episode_len_sec,
            "wind_mean_ms": self.wind_mean_ms,
            "gust_std_ms": self.gust_std_ms,
            "gust_tau_sec": self.gust_tau_sec,
            "gust_spatial_corr": self.gust_spatial_corr,
            "chain_topology": self.chain_topology,
            # TASK VERSION: results from different task versions must never be
            # pooled into one figure. See utils/run_naming.py.
            "task_version": TASK_VERSION,
            "lateral_weight": self.lateral_weight,
            "survival_bonus": self.survival_bonus,
            "crash_penalty": self.crash_penalty,
            "max_offset_m": self.max_offset_m,
            "mappo": self.mappo.as_dict(),
        }


# ==============================================================================
# SECTION 2 : THE TRAINER
# ==============================================================================

class Trainer:
    """Runs one (rung, seed) training job and records its learning curve."""

    def __init__(self, config: RunConfig, output_dir: Path | None = None):
        self.cfg = config
        self.output_dir = Path(output_dir) if output_dir else None

        # ---- Seed EVERYTHING before constructing anything --------------------
        # Networks are initialised randomly, so seeding must precede creation.
        seed_everything(config.seed, deterministic_torch=True)

        # Environment randomness (wind, initial poses) gets its OWN stream, so
        # that drawing wind samples cannot perturb policy exploration draws.
        # See utils/seeding.py for why this matters.
        self._env_rng = make_rng(config.seed, stream="env")
        self._eval_rng = make_rng(config.seed, stream="eval")

        self.env = self._make_env(self._env_rng)

        # action_dim=3: since v2-pidassist the agent commands a 3-D position OFFSET, not 4
        # rotor RPMs. The inner PID converts it. (v1 used 4.)
        self.learner = MAPPO(
            num_agents=config.num_agents,
            level=config.level,
            obs_dim=13,
            action_dim=3,
            config=config.mappo,
        )

        # ---- Learning curve -------------------------------------------------
        self.history: list[dict] = []
        self.global_step = 0
        # Per-episode physical metrics from the most recent evaluate() call.
        # The sweep reads these to report in metres rather than return.
        self.last_eval_metrics: list[EpisodeMetrics] = []

    # --------------------------------------------------------------------------
    def _make_env(self, rng: np.random.Generator) -> FormationAviary:
        return FormationAviary(
            num_drones=self.cfg.num_agents,
            formation_spacing_m=self.cfg.formation_spacing_m,
            episode_len_sec=self.cfg.episode_len_sec,
            wind_mean_ms=self.cfg.wind_mean_ms,
            gust_std_ms=self.cfg.gust_std_ms,
            gust_tau_sec=self.cfg.gust_tau_sec,
            gust_spatial_corr=self.cfg.gust_spatial_corr,
            chain_topology=self.cfg.chain_topology,
            base_altitude_m=self.cfg.base_altitude_m,
            ctrl_freq=self.cfg.ctrl_freq,
            # v2-pidassist shaping -- see scripts/verify_task_fixes.py
            lateral_weight=self.cfg.lateral_weight,
            survival_bonus=self.cfg.survival_bonus,
            crash_penalty=self.cfg.crash_penalty,
            max_offset_m=self.cfg.max_offset_m,
            rng=rng,
        )

    # ==========================================================================
    # SECTION 3 : MAIN LOOP
    # ==========================================================================

    def train(self, verbose: bool = True) -> list[dict]:
        """Train to the step budget, evaluating periodically.

        Returns
        -------
        list[dict]
            The learning curve: one record per evaluation point.
        """
        cfg = self.cfg
        started = time.time()

        obs, _ = self.env.reset(seed=cfg.seed)
        episode_return = np.zeros(cfg.num_agents)
        episode_len = 0
        recent_returns: list[float] = []
        next_eval_at = cfg.eval_every_steps

        if verbose:
            print(f"[{cfg.level.label}|seed{cfg.seed}] training for "
                  f"{cfg.total_env_steps:,} env steps "
                  f"(critic input {self.learner.critic_obs_dim}-D)")

        while self.global_step < cfg.total_env_steps:
            # ---- Collect one batch ------------------------------------------
            # EXACT BUDGET (fixed 16 Sep 2026). Collecting whole rollouts
            # unconditionally overshot whenever `total_env_steps` was not
            # divisible by `num_steps`: 100,000 became 100,096. Small, but it
            # made the budget we report untrue, and the size of the overshoot
            # depended on the rollout length -- so a rung that changed
            # `num_steps` would silently get a different budget, which is
            # exactly the confound Charter §7 forbids.
            rollout_len = min(cfg.mappo.num_steps,
                              cfg.total_env_steps - self.global_step)

            for _ in range(rollout_len):
                global_state = self.env.get_global_state()
                actions, log_probs, values, critic_obs = self.learner.act(obs, global_state)

                next_obs, rewards, terminated, truncated, info = self.env.step(actions)

                episode_return += rewards
                episode_len += 1
                self.global_step += 1

                # NOTE: only TERMINATION (crash / out of formation) cuts the
                # value bootstrap. Time-limit TRUNCATION does not -- the future
                # still exists, we simply stopped looking. Conflating the two
                # is a classic RL bug that biases value estimates.
                # TWO MASKS, NOT ONE (fixed 16 Sep 2026).
                # Termination kills the bootstrap; truncation does not, but it
                # still breaks the GAE recursion because the env resets.
                # Storing only `terminated` let the recursion run straight
                # through resets, blending unrelated episodes.
                done_flags = np.full(cfg.num_agents, float(terminated), dtype=np.float32)
                trunc_flags = np.full(cfg.num_agents, float(truncated), dtype=np.float32)

                step_index = self.learner.buffer.ptr

                self.learner.buffer.add(
                    obs=obs,
                    critic_obs=critic_obs,
                    actions=actions,
                    log_probs=log_probs,
                    rewards=rewards.astype(np.float32),
                    values=values,
                    dones=done_flags,
                    truncations=trunc_flags,
                )

                # On TRUNCATION we must bootstrap off the episode's TRUE final
                # state. `next_obs` still holds it here -- after the reset below
                # it would belong to a different episode.
                if truncated and not terminated:
                    final_values = self.learner.compute_values(
                        next_obs, self.env.get_global_state()
                    )
                    self.learner.buffer.record_truncation_value(step_index, final_values)

                obs = next_obs

                if terminated or truncated:
                    recent_returns.append(float(episode_return.mean()))
                    episode_return = np.zeros(cfg.num_agents)
                    episode_len = 0
                    obs, _ = self.env.reset()

            # ---- Update ------------------------------------------------------
            last_values = self.learner.compute_values(obs, self.env.get_global_state())
            self.learner.buffer.compute_gae(
                last_values=last_values,
                last_dones=np.zeros(cfg.num_agents, dtype=np.float32),
                gamma=cfg.mappo.gamma,
                gae_lambda=cfg.mappo.gae_lambda,
            )
            stats = self.learner.update()

            # ---- Evaluate ----------------------------------------------------
            if self.global_step >= next_eval_at:
                eval_result = self.evaluate(cfg.eval_episodes)
                next_eval_at += cfg.eval_every_steps

                record = {
                    "step": self.global_step,
                    "eval_return": eval_result["mean_return"],
                    "eval_return_std": eval_result["std_return"],
                    "eval_episode_len": eval_result["mean_length"],
                    # Task-health diagnostics (v2). eval_lateral_error_m is the
                    # sidestep-exploit monitor; eval_crash_rate catches the v1
                    # "learning to die faster" failure mode.
                    "eval_lateral_error_m": eval_result["mean_lateral_error_m"],
                    "eval_coupling_N": eval_result["mean_coupling_N"],
                    "eval_crash_rate": eval_result["crash_rate"],
                    # PHYSICAL metrics (Charter §S.11). Return is a training
                    # variable; these are what the paper reports.
                    "eval_rms_position_error_m": eval_result["rms_position_error_m"],
                    "eval_max_position_error_m": eval_result["max_position_error_m"],
                    "eval_rms_vertical_error_m": eval_result["rms_vertical_error_m"],
                    "eval_min_separation_m": eval_result["min_separation_m"],
                    "eval_collision_rate": eval_result["collision_rate"],
                    "eval_control_effort": eval_result["mean_control_effort"],
                    "eval_worst_error_amplification": eval_result["worst_error_amplification"],
                    "eval_mean_error_amplification": eval_result["mean_error_amplification"],
                    "train_return": float(np.mean(recent_returns[-10:])) if recent_returns else float("nan"),
                    "elapsed_sec": time.time() - started,
                    **stats,
                }
                self.history.append(record)

                if verbose:
                    print(
                        f"  step {self.global_step:>7,}  "
                        f"eval {eval_result['mean_return']:>9.2f}  "
                        f"explained_var {stats['explained_variance']:>6.3f}  "
                        f"value_loss {stats['value_loss']:>8.3f}"
                    )

        if self.output_dir:
            self.save()

        return self.history

    # ==========================================================================
    # SECTION 4 : EVALUATION
    # ==========================================================================

    def evaluate(self, num_episodes: int | None = None,
                 final: bool = False) -> dict[str, float]:
        """Run the DETERMINISTIC policy on fresh, held-out episodes.

        Uses a separate environment instance with its own RNG stream so that
        evaluation never disturbs the training environment's wind sequence --
        otherwise evaluating more often would change training, which would be a
        subtle and very hard-to-find confound.

        Parameters
        ----------
        final
            Use the held-out FINAL seed block instead of the periodic one. See
            `RunConfig.eval_seeds` for why these must be disjoint.

        Notes
        -----
        Every episode is recorded by an `EpisodeRecorder`, so the result carries
        PHYSICAL units -- centimetres, metres of separation, amplification
        ratios -- alongside return. Charter §S.11 makes those the headline
        numbers: a robotics reviewer can judge "RMS error 4.2 cm, zero
        collisions" and cannot judge "return 677".
        """
        seeds = self.cfg.final_eval_seeds() if final else self.cfg.eval_seeds()
        if num_episodes is not None:
            seeds = seeds[:num_episodes]

        eval_env = self._make_env(self._eval_rng)

        returns: list[float] = []
        lengths: list[int] = []
        lateral_errors: list[float] = []
        couplings: list[float] = []
        episode_metrics: list[EpisodeMetrics] = []
        crashes = 0

        max_steps = int(self.cfg.episode_len_sec * self.cfg.ctrl_freq)

        try:
            for seed in seeds:
                obs, _ = eval_env.reset(seed=seed)
                recorder = EpisodeRecorder(self.cfg.num_agents, max_steps)
                ep_return = np.zeros(self.cfg.num_agents)
                steps = 0
                info: dict = {}

                while True:
                    actions, _, _, _ = self.learner.act(
                        obs, eval_env.get_global_state(), deterministic=True
                    )
                    obs, rewards, terminated, truncated, info = eval_env.step(actions)

                    recorder.record(eval_env, actions, info)

                    ep_return += rewards
                    steps += 1

                    # SIDESTEP-EXPLOIT MONITOR. If trained policies drift
                    # laterally, they are dodging the wake -- which deletes the
                    # coupling this paper studies and would make a flat result
                    # look like "no effect" when it is "effect engineered away".
                    lateral_errors.append(info["mean_lateral_error_m"])
                    couplings.append(info["total_coupling_N"])

                    if terminated or truncated:
                        break

                if info.get("crashed"):
                    crashes += 1

                episode_metrics.append(recorder.finalise())
                returns.append(float(ep_return.mean()))
                lengths.append(steps)
        finally:
            eval_env.close()

        self.last_eval_metrics = episode_metrics

        def across(attr: str) -> float:
            vals = np.array([getattr(m, attr) for m in episode_metrics], dtype=float)
            vals = vals[np.isfinite(vals)]
            return float(vals.mean()) if len(vals) else float("nan")

        amp = summarise_amplification(
            [m.error_amplification for m in episode_metrics]
        )

        return {
            "mean_return": float(np.mean(returns)),
            "std_return": float(np.std(returns)),
            "mean_length": float(np.mean(lengths)),
            # Diagnostics for task-health monitoring
            "mean_lateral_error_m": float(np.mean(lateral_errors)),
            "mean_coupling_N": float(np.mean(couplings)),
            "crash_rate": crashes / max(len(seeds), 1),
            # ---- PHYSICAL metrics (Charter §S.11) -- what the paper reports --
            "rms_position_error_m": across("rms_position_error_m"),
            "max_position_error_m": across("max_position_error_m"),
            "rms_vertical_error_m": across("rms_vertical_error_m"),
            "min_separation_m": float(min(
                (m.min_separation_m for m in episode_metrics), default=float("nan")
            )),
            "collision_rate": float(np.mean([m.collision for m in episode_metrics])),
            "mean_control_effort": across("mean_control_effort"),
            "rms_angular_rate": across("rms_angular_rate"),
            # Worst link, not mean: a chain is only as stable as its weakest.
            "worst_error_amplification": amp.get("worst_link", float("nan")),
            "mean_error_amplification": amp.get("mean_amplification", float("nan")),
            "n_eval_episodes": len(seeds),
        }

    # ==========================================================================
    # SECTION 5 : PERSISTENCE
    # ==========================================================================

    def save(self) -> None:
        """Write config, learning curve, and final weights.

        Each run's directory is self-describing (Charter §7): it holds the
        settings, the numbers, and the model that produced them.
        """
        out = self.output_dir
        out.mkdir(parents=True, exist_ok=True)

        with open(out / "run_config.json", "w", encoding="utf-8") as f:
            json.dump(self.cfg.as_dict(), f, indent=2)

        with open(out / "history.json", "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)

        with open(out / "parameter_report.json", "w", encoding="utf-8") as f:
            json.dump(self.learner.parameter_report(), f, indent=2)

        torch.save(self.learner.state_dict(), out / "final_model.pt")

    def close(self) -> None:
        self.env.close()


# ==============================================================================
# SECTION 6 : CONVENIENCE
# ==============================================================================

def run_single(config: RunConfig, output_dir: Path | None = None, verbose: bool = True) -> list[dict]:
    """Run one training job end to end, guaranteeing cleanup."""
    trainer = Trainer(config, output_dir)
    try:
        return trainer.train(verbose=verbose)
    finally:
        trainer.close()
