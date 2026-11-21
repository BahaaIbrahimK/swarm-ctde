"""
================================================================================
 MAPPO  --  Multi-Agent PPO with a k-parameterised centralized critic
================================================================================

PHASE 3 (Charter §5).

--------------------------------------------------------------------------------
 WHY MAPPO AND NOT SAC / TD3 / DDPG
--------------------------------------------------------------------------------
The previous project attempted centralized critics for off-policy algorithms
and the implementation was broken in a way that produced plausible-looking but
meaningless numbers. Two root causes, both structural rather than accidental:

  1. RLlib's SAC/DDPG build their own Q-heads internally; a `custom_model`
     exposing `value_function()` is simply never called by their loss. The
     centralized Q-critic was dead code receiving no gradients.

  2. Actions were written into observations in `on_postprocess_trajectory`,
     then sampled later from a REPLAY BUFFER. Off-policy replay makes those
     stored actions stale, and feeding an agent's own action into Q(s, .)
     alongside the action being evaluated is conceptually wrong.

MAPPO avoids both by construction: it is on-policy, so data is used once while
fresh, and the critic is an explicit V(s) we write ourselves. Charter §9 states
we deliberately do NOT claim algorithm superiority -- algorithm choice is held
FIXED so that k is the only variable. One correct algorithm beats five broken
ones.

--------------------------------------------------------------------------------
 PARAMETER SHARING
--------------------------------------------------------------------------------
All agents share one actor and one critic. Standard in MAPPO, and appropriate
here because our agents are homogeneous in capability (they differ only in
their formation slot, which is already encoded in the observation as position
error).

This also keeps the comparison across k clean: with per-agent networks, total
parameter count would scale with team size and confound the sweep.

Claim 3 (role-asymmetric critics) will deliberately BREAK this sharing later --
that is the point of that experiment, and it is a separate code path.

--------------------------------------------------------------------------------
 THE PPO CLIPPED OBJECTIVE, IN ONE PARAGRAPH
--------------------------------------------------------------------------------
Vanilla policy gradient can take a catastrophically large step and destroy the
policy. PPO computes the probability ratio between the new and old policy,

    ratio = pi_new(a|s) / pi_old(a|s)

and optimises min(ratio * A, clip(ratio, 1-eps, 1+eps) * A). The clip removes
any incentive to move the ratio beyond [1-eps, 1+eps], so updates stay in a
trust region without needing a second-order method.
================================================================================
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn

from ctde.models.networks import Actor, Critic
from ctde.models.observability import (
    ObservabilityLevel, build_critic_input, max_critic_input_dim,
)
from ctde.training.buffer import RolloutBuffer


# ==============================================================================
# SECTION 1 : HYPERPARAMETERS
# ==============================================================================

@dataclass
class MAPPOConfig:
    """PPO hyperparameters.

    CHARTER §4.3 REQUIREMENT -- these are IDENTICAL across every rung of the
    ladder, and were tuned once at the maximal rung (k=N-1), i.e. in the
    BASELINE'S favour. Tuning in favour of the practice we are challenging
    blocks the "you sabotaged the baseline" objection.
    """

    # ---- Rollout ------------------------------------------------------------
    num_steps: int = 256          # env steps per agent per update
    num_epochs: int = 10          # passes over each batch
    minibatch_size: int = 256

    # ---- Optimisation -------------------------------------------------------
    lr_actor: float = 3e-4
    lr_critic: float = 1e-3       # critic can safely learn faster than actor
    max_grad_norm: float = 0.5    # gradient clipping; standard PPO stabiliser

    # ---- PPO objective ------------------------------------------------------
    clip_eps: float = 0.2
    # LOWERED 2026-08-28 from 0.01. The entropy bonus rewards keeping the
    # policy stochastic, which here actively fights the correct solution:
    # near-deterministic commands close to the PID setpoint. With the task's
    # steep lateral penalty, 0.01 was enough to hold the policy in a noisy
    # regime it should have been escaping.
    entropy_coef: float = 0.001
    value_loss_coef: float = 0.5

    # ---- GAE ----------------------------------------------------------------
    gamma: float = 0.99
    gae_lambda: float = 0.95

    # ---- Infrastructure -----------------------------------------------------
    device: str = "cpu"

    def as_dict(self) -> dict:
        """Plain dict, saved alongside results for provenance (Charter §7)."""
        return dict(self.__dict__)


# ==============================================================================
# SECTION 1b : VALUE NORMALISATION
# ==============================================================================

class RunningMeanStd:
    """Welford running mean/variance for value-target normalisation.

    ADDED 16 Sep 2026. MAPPO's authors emphasise value normalisation as one of
    the implementation details that matters most for stability, and we had none.

    WHY IT MATTERS HERE SPECIFICALLY: episode returns sit around 600-700 while a
    crash could cost tens of thousands (now recalibrated, but the scale still
    varies with formation spacing and gust strength, both of which we sweep).
    An unnormalised critic must fit a target whose scale changes across the
    experimental conditions -- so critic accuracy would be confounded with the
    condition itself, which is precisely what this paper measures.

    Normalising the target keeps the critic's learning problem comparable across
    rungs and spacings.
    """

    def __init__(self, epsilon: float = 1e-4):
        self.mean = 0.0
        self.var = 1.0
        self.count = epsilon

    def update(self, x: np.ndarray) -> None:
        batch_mean = float(np.mean(x))
        batch_var = float(np.var(x))
        batch_count = x.size

        delta = batch_mean - self.mean
        total = self.count + batch_count

        self.mean += delta * batch_count / total
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        self.var = (m_a + m_b + delta ** 2 * self.count * batch_count / total) / total
        self.count = total

    @property
    def std(self) -> float:
        return float(np.sqrt(self.var) + 1e-8)

    def normalize(self, x):
        return (x - self.mean) / self.std

    def denormalize(self, x):
        return x * self.std + self.mean


# ==============================================================================
# SECTION 2 : THE LEARNER
# ==============================================================================

class MAPPO:
    """MAPPO with a critic whose observability level is a constructor argument.

    THE ENTIRE EXPERIMENT IS THE `level` ARGUMENT. Everything else is held
    fixed across runs.
    """

    def __init__(
        self,
        num_agents: int,
        level: ObservabilityLevel,
        obs_dim: int = 13,
        # v4: the policy commands a 3-D position OFFSET, not 4 rotor RPMs.
        # The inner PID converts it. Default was stale at 4.
        action_dim: int = 3,
        config: MAPPOConfig | None = None,
    ):
        self.num_agents = num_agents
        self.level = level
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.cfg = config or MAPPOConfig()

        # EXACT CAPACITY CONTROL (wired 16 Sep 2026).
        #
        # Every rung uses the SAME maximum input width; rungs below the maximum
        # have their excluded teammate slots ZEROED rather than absent. The
        # critic is therefore literally the same network at every rung, and only
        # the information content of its input varies.
        #
        # This was implemented in observability.py but NOT CONNECTED here, so
        # the real learners still had rung-dependent widths (13 -> 34) and
        # parameter counts differing by 1,344. The test that "verified" it
        # constructed Critic(input_dim=max_dim) by hand and never instantiated
        # MAPPO, so it could not detect the disconnection.
        self.critic_obs_dim = max_critic_input_dim(num_agents, own_obs_dim=obs_dim)

        # ---- Networks -------------------------------------------------------
        # Actor: IDENTICAL at every rung (confound control 1).
        self.actor = Actor(obs_dim=obs_dim, action_dim=action_dim).to(self.cfg.device)

        # Critic: IDENTICAL at every rung -- same width, same parameters.
        # Only the masked content of its input differs (confound control 2).
        self.critic = Critic(input_dim=self.critic_obs_dim).to(self.cfg.device)

        # Separate optimisers so actor and critic can use different rates.
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.cfg.lr_actor)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=self.cfg.lr_critic)

        # Value normalisation (see RunningMeanStd). Without it, critic accuracy
        # would be confounded with the experimental condition.
        self.value_normalizer = RunningMeanStd()

        self.buffer = RolloutBuffer(
            num_steps=self.cfg.num_steps,
            num_agents=num_agents,
            obs_dim=obs_dim,
            critic_obs_dim=self.critic_obs_dim,
            action_dim=action_dim,
            device=self.cfg.device,
        )

    # ==========================================================================
    # SECTION 3 : ACTING
    # ==========================================================================

    @torch.no_grad()
    def act(
        self,
        obs: np.ndarray,
        global_state: dict,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Select actions for all agents.

        Parameters
        ----------
        obs:
            (num_agents, obs_dim) LOCAL observations -- what actors see.
        global_state:
            From `FormationAviary.get_global_state()`; the critic input is
            assembled from it according to `self.level`.
        deterministic:
            Use the distribution mean (evaluation) rather than sampling
            (training).

        Returns
        -------
        (actions, log_probs, values, critic_obs)
        """
        # ---- Build the k-dependent critic input, per agent -------------------
        critic_obs = np.stack([
            build_critic_input(
                i, obs[i], global_state, self.level,
                padded=True, num_agents=self.num_agents,
            )
            for i in range(self.num_agents)
        ])

        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.cfg.device)
        critic_obs_t = torch.as_tensor(critic_obs, dtype=torch.float32, device=self.cfg.device)

        dist = self.actor.distribution(obs_t)
        # `deterministic` uses the SQUASHED mean -- tanh(mu), not mu -- so the
        # evaluation action is bounded exactly like a sampled one.
        actions = dist.deterministic if deterministic else dist.sample()

        # Sum log-probs over action dimensions: the 4 rotor commands are
        # modelled as independent Gaussians, so the joint log-prob is the sum.
        log_probs = dist.log_prob(actions).sum(dim=-1)

        # The critic predicts in NORMALISED space; convert back so GAE operates
        # on real-scale values.
        values = torch.as_tensor(
            self.value_normalizer.denormalize(self.critic(critic_obs_t).cpu().numpy()),
            dtype=torch.float32,
        )

        return (
            actions.cpu().numpy(),
            log_probs.cpu().numpy(),
            values.cpu().numpy(),
            critic_obs,
        )

    @torch.no_grad()
    def compute_values(self, obs: np.ndarray, global_state: dict) -> np.ndarray:
        """Value estimates only -- used to bootstrap GAE at the batch boundary."""
        critic_obs = np.stack([
            build_critic_input(
                i, obs[i], global_state, self.level,
                padded=True, num_agents=self.num_agents,
            )
            for i in range(self.num_agents)
        ])
        critic_obs_t = torch.as_tensor(critic_obs, dtype=torch.float32, device=self.cfg.device)
        return self.value_normalizer.denormalize(self.critic(critic_obs_t).cpu().numpy())

    # ==========================================================================
    # SECTION 4 : LEARNING
    # ==========================================================================

    def update(self) -> dict[str, float]:
        """Run the PPO update over the collected batch.

        Returns
        -------
        dict
            Diagnostics: losses, KL divergence, clip fraction, and the
            advantage statistics the Phase 7 mechanism study needs.
        """
        batch = self.buffer.get_batch(normalise_advantages=True)

        # Refresh value statistics on this batch BEFORE using them, so the
        # normaliser tracks the current return distribution.
        self.value_normalizer.update(batch["returns"].cpu().numpy())

        num_samples = batch["obs"].shape[0]
        minibatch_size = min(self.cfg.minibatch_size, num_samples)

        stats: dict[str, list[float]] = {
            "policy_loss": [], "value_loss": [], "entropy": [],
            "actor_grad_norm": [], "critic_grad_norm": [],
            "action_saturation": [], "action_abs_mean": [], "action_std": [],
            "approx_kl": [], "clip_fraction": [],
        }

        for _ in range(self.cfg.num_epochs):
            # Fresh shuffle each epoch so minibatches differ between passes.
            indices = torch.randperm(num_samples, device=self.cfg.device)

            for start in range(0, num_samples, minibatch_size):
                mb = indices[start : start + minibatch_size]

                mb_obs = batch["obs"][mb]
                mb_critic_obs = batch["critic_obs"][mb]
                mb_actions = batch["actions"][mb]
                mb_old_log_probs = batch["log_probs"][mb]
                mb_advantages = batch["advantages"][mb]
                mb_returns = batch["returns"][mb]

                # ---- Actor loss ---------------------------------------------
                dist = self.actor.distribution(mb_obs)
                new_log_probs = dist.log_prob(mb_actions).sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1).mean()

                # ratio = pi_new/pi_old, computed in log space for stability.
                ratio = torch.exp(new_log_probs - mb_old_log_probs)

                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.cfg.clip_eps, 1.0 + self.cfg.clip_eps) * mb_advantages

                # Negative because optimisers minimise.
                policy_loss = -torch.min(surr1, surr2).mean()
                actor_loss = policy_loss - self.cfg.entropy_coef * entropy

                self.actor_opt.zero_grad()
                actor_loss.backward()
                # Gradient norm BEFORE clipping. Measured after clipping it is
                # capped at max_grad_norm by construction and tells you nothing;
                # this is the number that reveals a vanishing or exploding
                # actor gradient, which is what "exploration problem" would
                # actually look like.
                actor_grad_norm = nn.utils.clip_grad_norm_(
                    self.actor.parameters(), self.cfg.max_grad_norm
                )
                self.actor_opt.step()

                # ---- Critic loss --------------------------------------------
                # Plain MSE against the GAE returns. THIS is where extra
                # observability is supposed to help: a better-informed critic
                # should fit the returns more accurately, shrinking advantage
                # variance (Charter §1.5).
                # Normalise the TARGET, not the prediction: the critic learns
                # in a standardised space whose scale does not drift with the
                # experimental condition.
                values = self.critic(mb_critic_obs)
                normalised_returns = torch.as_tensor(
                    self.value_normalizer.normalize(mb_returns.cpu().numpy()),
                    dtype=torch.float32, device=values.device,
                )
                value_loss = ((values - normalised_returns) ** 2).mean()

                self.critic_opt.zero_grad()
                (self.cfg.value_loss_coef * value_loss).backward()
                critic_grad_norm = nn.utils.clip_grad_norm_(
                    self.critic.parameters(), self.cfg.max_grad_norm
                )
                self.critic_opt.step()

                # ---- Diagnostics --------------------------------------------
                with torch.no_grad():
                    log_ratio = new_log_probs - mb_old_log_probs
                    # Schulman's low-variance KL estimator.
                    approx_kl = ((torch.exp(log_ratio) - 1) - log_ratio).mean()
                    clip_frac = ((ratio - 1.0).abs() > self.cfg.clip_eps).float().mean()

                    # ---- BEHAVIOURAL diagnostics (added 17 Sep 2026) --------
                    # `entropy` above is the UNSQUASHED Gaussian's entropy. It
                    # cannot fall when tanh saturates, so it cannot detect the
                    # failure mode it looks like it is watching. These can.
                    #
                    # action_saturation: fraction of action components pinned
                    #   near the tanh limits. High values mean the policy is
                    #   effectively bang-bang and gradients through tanh vanish.
                    # action_abs_mean: how hard the policy is actually pushing,
                    #   in normalised units. Rising while tracking error stays
                    #   flat is the "acting more, achieving nothing" signature
                    #   observed in every v4 seed.
                    saturation = (mb_actions.abs() > 0.99).float().mean()
                    action_abs = mb_actions.abs().mean()
                    action_std = mb_actions.std()

                # .detach() before float(): these tensors still carry grad_fn,
                # and converting them directly warns (and would keep the graph
                # alive longer than necessary).
                stats["policy_loss"].append(float(policy_loss.detach()))
                stats["value_loss"].append(float(value_loss.detach()))
                stats["entropy"].append(float(entropy.detach()))
                stats["approx_kl"].append(float(approx_kl))
                stats["clip_fraction"].append(float(clip_frac))
                stats["actor_grad_norm"].append(float(actor_grad_norm))
                stats["critic_grad_norm"].append(float(critic_grad_norm))
                stats["action_saturation"].append(float(saturation))
                stats["action_abs_mean"].append(float(action_abs))
                stats["action_std"].append(float(action_std))

        out = {k: float(np.mean(v)) for k, v in stats.items()}
        out.update(self.buffer.advantage_diagnostics())

        self.buffer.reset()
        return out

    # ==========================================================================
    # SECTION 5 : CHECKPOINTING
    # ==========================================================================

    def state_dict(self) -> dict:
        """Everything needed to REPRODUCE this learner's behaviour.

        THE VALUE NORMALIZER IS PART OF THE MODEL  (added 17 Sep 2026).

        The critic is trained against NORMALISED returns, and `act()` /
        `compute_values()` denormalise its output using running mean and
        variance. Those statistics are learned state. A checkpoint without them
        restores a critic whose outputs are interpreted in the wrong scale --
        it loads cleanly, runs, and is silently wrong, which is the worst
        possible failure mode for a saved result.

        Optimizer state is saved too, so training can be resumed without the
        Adam moment estimates restarting from zero.
        """
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "actor_opt": self.actor_opt.state_dict(),
            "critic_opt": self.critic_opt.state_dict(),
            "value_normalizer": {
                "mean": np.asarray(self.value_normalizer.mean).tolist(),
                "var": np.asarray(self.value_normalizer.var).tolist(),
                "count": float(self.value_normalizer.count),
            },
            "level_k": self.level.k,
            "level_privileged": self.level.include_privileged,
            "critic_obs_dim": self.critic_obs_dim,
        }

    def load_state_dict(self, state: dict, strict_rung: bool = True) -> None:
        """Restore a checkpoint, refusing one from a different rung.

        THE OLD GUARD NEVER FIRED. It compared `critic_obs_dim`, which since the
        padding fix is IDENTICAL at every rung by design -- that is the whole
        point of exact capacity control. So a k0 checkpoint loaded into a k2
        learner succeeded silently, producing a model whose label and whose
        actual information content disagreed.

        The rung identity itself is what must match.
        """
        if strict_rung:
            ck_k = state.get("level_k")
            ck_p = state.get("level_privileged")
            if ck_k != self.level.k or ck_p != self.level.include_privileged:
                ck_label = f"k{ck_k}{'+P' if ck_p else ''}"
                raise ValueError(
                    f"Checkpoint is from rung {ck_label} but this model is "
                    f"{self.level.label}. Checkpoints are rung-specific: every "
                    f"rung has the same critic WIDTH by design, so shapes match "
                    f"and the mistake would otherwise be invisible."
                )

        if state["critic_obs_dim"] != self.critic_obs_dim:
            raise ValueError(
                f"Checkpoint critic input dim {state['critic_obs_dim']} does not match "
                f"this model's {self.critic_obs_dim} (rung {self.level.label})."
            )

        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])

        if "actor_opt" in state:
            self.actor_opt.load_state_dict(state["actor_opt"])
        if "critic_opt" in state:
            self.critic_opt.load_state_dict(state["critic_opt"])

        if "value_normalizer" in state:
            vn = state["value_normalizer"]
            self.value_normalizer.mean = np.asarray(vn["mean"], dtype=np.float64)
            self.value_normalizer.var = np.asarray(vn["var"], dtype=np.float64)
            self.value_normalizer.count = vn["count"]

    def parameter_report(self) -> dict:
        """Parameter counts, reported per rung in the paper."""
        actor_params = sum(p.numel() for p in self.actor.parameters())
        critic = self.critic.parameter_report()
        return {
            "rung": self.level.label,
            "critic_input_dim": self.critic_obs_dim,
            "actor_params": actor_params,
            **critic,
        }
