"""
================================================================================
 ROLLOUT BUFFER + GENERALISED ADVANTAGE ESTIMATION (GAE)
================================================================================

PHASE 3 (Charter §5).

--------------------------------------------------------------------------------
 WHAT A ROLLOUT BUFFER IS
--------------------------------------------------------------------------------
PPO is an ON-POLICY algorithm: it collects a batch of experience with the
CURRENT policy, updates on that batch, then throws it away and collects fresh
experience. The buffer is the temporary store for one such batch.

(Contrast with off-policy methods like SAC/TD3, which keep a large replay
buffer of old experience. That difference is exactly why the previous project's
off-policy centralized critic was broken -- stale actions written into
observations go bad in a replay buffer. On-policy avoids that whole class of
bug, which is one reason we use MAPPO here.)

--------------------------------------------------------------------------------
 WHY THE BUFFER STORES TWO DIFFERENT INPUTS PER STEP
--------------------------------------------------------------------------------
This is where the paper's central mechanism lives:

    obs         -- 12-D LOCAL observation      -> fed to the ACTOR
    critic_obs  -- k-dependent input           -> fed to the CRITIC

The actor input NEVER changes with k (confound control 1). Only `critic_obs`
grows. Everything downstream of that distinction is the experiment.

--------------------------------------------------------------------------------
 WHAT GAE DOES, AND WHY IT IS THE HEART OF THE HYPOTHESIS
--------------------------------------------------------------------------------
The actor needs to know "was that action better or worse than expected?". That
quantity is the ADVANTAGE:

    A(s,a) = (what actually happened) - (what the critic predicted)

The naive estimator uses the full episode return: unbiased but very HIGH
VARIANCE (one lucky gust changes the whole episode's credit). The other extreme
uses a one-step TD error: low variance but BIASED by critic error.

GAE interpolates between them with a parameter lambda:

    delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)          # one-step TD error
    A_t     = delta_t + (gamma*lambda) * A_{t+1}         # exponentially weighted sum

CONNECTION TO CHARTER §1.5 -- read this carefully, it is the whole paper:

  * A BETTER critic makes V(s_t) more accurate, which shrinks delta_t's
    variance. That is the "variance reduction" force pushing performance UP
    as k grows.

  * But a critic that sees MORE than the actor learns a V the actor cannot
    reproduce. It credits outcomes to information the policy cannot act on.
    That component of A_t is noise to the actor. That is the "representational
    mismatch" force pushing performance DOWN as k grows.

Both forces act through THIS function. That is why we instrument advantage
statistics here (Phase 7 mechanism study).
================================================================================
"""

from __future__ import annotations

import numpy as np
import torch


# ==============================================================================
# SECTION 1 : BUFFER
# ==============================================================================

class RolloutBuffer:
    """Fixed-size on-policy storage for one PPO batch.

    Shapes
    ------
    Everything is stored as (num_steps, num_agents, ...). Agents are flattened
    into the batch dimension at update time: with parameter sharing, every
    agent's experience trains the same networks, which is standard MAPPO and
    greatly improves sample efficiency.
    """

    def __init__(
        self,
        num_steps: int,
        num_agents: int,
        obs_dim: int,
        critic_obs_dim: int,
        action_dim: int,
        device: str = "cpu",
    ):
        self.num_steps = num_steps
        self.num_agents = num_agents
        self.device = device

        # ---- Actor input: NEVER varies with k -------------------------------
        self.obs = np.zeros((num_steps, num_agents, obs_dim), dtype=np.float32)

        # ---- Critic input: THE independent variable -------------------------
        self.critic_obs = np.zeros((num_steps, num_agents, critic_obs_dim), dtype=np.float32)

        self.actions = np.zeros((num_steps, num_agents, action_dim), dtype=np.float32)
        self.log_probs = np.zeros((num_steps, num_agents), dtype=np.float32)
        self.rewards = np.zeros((num_steps, num_agents), dtype=np.float32)
        self.values = np.zeros((num_steps, num_agents), dtype=np.float32)

        # TWO MASKS ARE REQUIRED, AND THEY ARE NOT THE SAME (fixed 16 Sep 2026).
        #
        #   dones[t]        TERMINATION -- the episode genuinely ended at t.
        #                   A terminal state has no future, so V(s_{t+1}) must
        #                   be excluded from the bootstrap.
        #
        #   truncations[t]  TIME LIMIT -- we stopped looking, but the future
        #                   still exists. We DO bootstrap (off the true final
        #                   state), but the environment has reset, so the next
        #                   stored transition belongs to a DIFFERENT episode and
        #                   the GAE recursion must not carry across it.
        #
        # Conflating these is a classic RL bug. The previous implementation
        # stored only `dones`, set it from `terminated` alone, and let the
        # recursion run straight through resets.
        self.dones = np.zeros((num_steps, num_agents), dtype=np.float32)
        self.truncations = np.zeros((num_steps, num_agents), dtype=np.float32)

        # V(s_final) at a truncation. The observation stored at t+1 belongs to
        # the RESET episode, so `values[t+1]` cannot be used to bootstrap a
        # truncated step -- we must record the value of the state the episode
        # actually ended in.
        self.truncation_values = np.zeros((num_steps, num_agents), dtype=np.float32)

        # Filled by compute_gae()
        self.advantages = np.zeros((num_steps, num_agents), dtype=np.float32)
        self.returns = np.zeros((num_steps, num_agents), dtype=np.float32)

        self.ptr = 0

    # --------------------------------------------------------------------------
    def add(
        self,
        obs: np.ndarray,
        critic_obs: np.ndarray,
        actions: np.ndarray,
        log_probs: np.ndarray,
        rewards: np.ndarray,
        values: np.ndarray,
        dones: np.ndarray,
        truncations: np.ndarray | None = None,
    ) -> None:
        """Store one timestep of experience for all agents.

        Parameters
        ----------
        dones:
            TERMINATION at this transition (crash, constraint violation).
        truncations:
            TIME LIMIT at this transition. Defaults to zeros for callers that
            have no time limit; the trainer always passes it explicitly.
        """
        if self.ptr >= self.num_steps:
            raise RuntimeError(
                f"RolloutBuffer is full ({self.num_steps} steps). "
                "Call compute_gae() and reset() before adding more."
            )

        i = self.ptr
        self.obs[i] = obs
        self.critic_obs[i] = critic_obs
        self.actions[i] = actions
        self.log_probs[i] = log_probs
        self.rewards[i] = rewards
        self.values[i] = values
        self.dones[i] = dones
        self.truncations[i] = truncations if truncations is not None else 0.0
        self.ptr += 1

    # --------------------------------------------------------------------------
    def compute_gae(
        self,
        last_values: np.ndarray,
        last_dones: np.ndarray,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> None:
        """Compute advantages and returns by backward recursion.

        Parameters
        ----------
        last_values:
            V(s_T) for the state AFTER the final stored step, shape (num_agents,).
            Needed to bootstrap: the batch usually ends mid-episode, and we must
            estimate the value of the remaining future rather than assume zero.
        last_dones:
            Whether each agent's episode had ended at that final state.
        gamma:
            Discount factor. 0.99 at 48 Hz control gives an effective horizon of
            roughly 100 steps (~2 s), which suits formation-keeping.
        gae_lambda:
            Bias-variance knob. 0.95 is the standard PPO default: mostly
            low-variance, slightly biased.

        Notes
        -----
        The recursion runs BACKWARD because A_t depends on A_{t+1}.
        """
        last_gae = np.zeros(self.num_agents, dtype=np.float32)

        for t in reversed(range(self.ptr)):
            # ---- Bootstrap value: V(s_{t+1}) ---------------------------------
            #
            # OFF-BY-ONE FIX (16 Sep 2026). A transition is stored as
            #     (s_t, a_t, r_t, s_{t+1}, done_t)
            # so `dones[t]` answers "did the episode end AS A RESULT of
            # transition t?". The bootstrap for t must therefore read
            # `dones[t]`, NOT `dones[t+1]`.
            #
            # The previous code used `dones[t+1]`, which shifted every mask one
            # step. A mid-rollout terminal then leaked reward and value across
            # the episode boundary. The old test missed this because it placed
            # the terminal on the LAST buffer element -- the one index where the
            # error is invisible, since that step reads `last_dones` instead.
            terminated_here = self.dones[t]
            truncated_here = self.truncations[t]

            if truncated_here.any():
                # TIME LIMIT: the future exists, so bootstrap -- but off the
                # TRUE final state of that episode, not off values[t+1], which
                # belongs to the reset episode.
                bootstrap = np.where(
                    truncated_here > 0,
                    self.truncation_values[t],
                    last_values if t == self.ptr - 1 else self.values[t + 1],
                )
            elif t == self.ptr - 1:
                bootstrap = last_values
            else:
                bootstrap = self.values[t + 1]

            # TERMINATION kills the bootstrap; truncation does not.
            bootstrap_mask = 1.0 - terminated_here

            delta = self.rewards[t] + gamma * bootstrap * bootstrap_mask - self.values[t]

            # ---- Recursion mask ----------------------------------------------
            # The accumulation must stop at ANY episode boundary -- terminal or
            # truncated -- because the next stored transition belongs to a
            # different episode. This is a DIFFERENT mask from the bootstrap
            # one: truncation bootstraps but still breaks the chain.
            boundary = np.maximum(terminated_here, truncated_here)
            continuation_mask = 1.0 - boundary

            last_gae = delta + gamma * gae_lambda * continuation_mask * last_gae
            self.advantages[t] = last_gae

        # Value targets. Adding V(s) back recovers the empirical return, which
        # is what the critic regresses toward.
        self.returns[: self.ptr] = self.advantages[: self.ptr] + self.values[: self.ptr]

    # --------------------------------------------------------------------------
    def set_truncation_values(self, values: np.ndarray) -> None:
        """Record V(s_final) for truncated steps.

        Must be called before `compute_gae()` whenever any stored transition was
        truncated. The trainer computes this from the FINAL observation of the
        episode, captured before the environment resets.

        Parameters
        ----------
        values:
            (num_steps, num_agents) array. Only entries where
            `truncations[t] > 0` are read.
        """
        values = np.asarray(values, dtype=np.float32)
        self.truncation_values[: values.shape[0]] = values.reshape(
            values.shape[0], self.num_agents
        )

    def record_truncation_value(self, step_index: int, values: np.ndarray) -> None:
        """Record V(s_final) for a single truncated step.

        Called by the trainer at the moment of truncation, while the final
        observation is still available. After the environment resets, that state
        is gone and `values[t+1]` would refer to a different episode.
        """
        self.truncation_values[step_index] = np.asarray(values, dtype=np.float32)

    def get_batch(self, normalise_advantages: bool = True) -> dict[str, torch.Tensor]:
        """Return the whole buffer as flat tensors for the update.

        Agents are folded into the batch dimension:
            (num_steps, num_agents, D) -> (num_steps * num_agents, D)

        Parameters
        ----------
        normalise_advantages:
            Standardise advantages to zero mean / unit variance. Standard PPO
            practice: it decouples the policy-gradient step size from the
            reward scale, which matters here because rewards differ across
            formation spacings (Claim 2 sweeps spacing).
        """
        n = self.ptr

        def flat(arr: np.ndarray) -> torch.Tensor:
            shaped = arr[:n].reshape(n * self.num_agents, -1)
            return torch.as_tensor(shaped, dtype=torch.float32, device=self.device)

        def flat1d(arr: np.ndarray) -> torch.Tensor:
            return torch.as_tensor(
                arr[:n].reshape(n * self.num_agents),
                dtype=torch.float32,
                device=self.device,
            )

        advantages = flat1d(self.advantages)
        if normalise_advantages:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return {
            "obs": flat(self.obs),
            "critic_obs": flat(self.critic_obs),
            "actions": flat(self.actions),
            "log_probs": flat1d(self.log_probs),
            "advantages": advantages,
            "returns": flat1d(self.returns),
            "values": flat1d(self.values),
        }

    # --------------------------------------------------------------------------
    def reset(self) -> None:
        """Drop all stored experience (PPO is on-policy; data is single-use)."""
        self.ptr = 0
        # Clear boundary state explicitly: a stale truncation flag from the
        # previous batch would silently break the next one's recursion.
        self.dones.fill(0.0)
        self.truncations.fill(0.0)
        self.truncation_values.fill(0.0)

    # --------------------------------------------------------------------------
    def advantage_diagnostics(self) -> dict[str, float]:
        """Statistics used by the Phase 7 mechanism study.

        The §1.5 argument predicts that as k grows, a rising share of the
        advantage signal becomes unrealizable by the actor -- effectively
        noise. These are the raw numbers for testing that decomposition, so we
        record them from the very first pilot run rather than retrofitting the
        instrumentation later.
        """
        adv = self.advantages[: self.ptr]
        val = self.values[: self.ptr]
        ret = self.returns[: self.ptr]

        # Explained variance: how much of the return variation the critic
        # actually captures. 1.0 = perfect, 0.0 = no better than predicting the
        # mean, negative = worse than the mean.
        var_ret = float(np.var(ret))
        explained_var = (
            float(1.0 - np.var(ret - val) / var_ret) if var_ret > 1e-8 else 0.0
        )

        return {
            "advantage_mean": float(np.mean(adv)),
            "advantage_std": float(np.std(adv)),
            "advantage_abs_mean": float(np.mean(np.abs(adv))),
            "value_mean": float(np.mean(val)),
            "return_mean": float(np.mean(ret)),
            "explained_variance": explained_var,
        }
