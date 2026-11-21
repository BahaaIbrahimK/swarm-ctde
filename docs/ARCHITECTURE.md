# Architecture

How the code is organised, how one training run flows through it, and how results
are evaluated and recorded. For the research rationale see
[`../PROJECT_CHARTER.md`](../PROJECT_CHARTER.md); for definitions of every term see
[`EXPERIMENT_DEFINITION.md`](EXPERIMENT_DEFINITION.md).

---

## 1. Module map

```mermaid
flowchart TB
    subgraph entry["scripts/ · entry points"]
        S1["run_downwash_probe.py"]
        S2["inspect_ladder.py"]
        S3["check_learning.py"]
        S4["run_pilot_sweep.py<br/>run_overnight.py"]
        S5["verify_task_fixes.py"]
    end

    subgraph training["ctde.training"]
        SW["sweep.py<br/>parallel runs · aggregate_by_rung"]
        TR["trainer.py<br/>RunConfig · Trainer · evaluate"]
        MP["mappo.py<br/>MAPPO · value normalization"]
        BF["buffer.py<br/>RolloutBuffer · GAE"]
    end

    subgraph models["ctde.models"]
        OB["observability.py<br/>ladder · padded critic input"]
        NW["networks.py<br/>Actor · Critic · SquashedGaussian"]
    end

    subgraph envs["ctde.envs"]
        FA["formation_aviary.py<br/>FormationAviary task"]
        CP["coupling.py<br/>downwash model · coupling rank"]
    end

    PH["ctde.physics.downwash_probe"]
    BL["ctde.baselines"]
    ME["ctde.metrics"]
    UT["ctde.utils<br/>config · seeding · run_naming"]
    GP[("gym-pybullet-drones<br/>BaseAviary · DSL PID")]
    FG["figures/plot_sweep.py"]

    S1 --> PH --> GP
    S2 --> OB
    S3 --> TR
    S4 --> SW --> TR
    S5 --> FA
    TR --> MP --> BF
    MP --> NW
    MP --> OB --> CP
    TR --> FA --> GP
    FA --> CP
    TR --> ME
    S3 --> BL
    TR --> UT
    SW --> UT
    SW -. "results/ JSON" .-> FG
```

| Module | Responsibility |
|---|---|
| `envs/formation_aviary.py` | The task: chain references, OU gust field, wind force application, reward, crash/collision detection, termination vs truncation |
| `envs/coupling.py` | Analytical downwash proxy used to **rank** teammates by influence; spacing validity checks |
| `models/observability.py` | Builds the ladder (`k0 … k(N−2)`, `+P`) and the zero-padded critic input |
| `models/networks.py` | Actor (13-D → 3-D, tanh-squashed Gaussian) and fixed-capacity Critic |
| `training/buffer.py` | Rollout storage and GAE with correct terminal/truncation handling |
| `training/mappo.py` | PPO update for the shared actor and the centralized critic |
| `training/trainer.py` | One run: environment, learner, periodic and final evaluation, checkpoints |
| `training/sweep.py` | Many runs across rungs × seeds × spacings, process-parallel, and aggregation |
| `baselines.py` | Zero-residual PID, P and PD reference controllers |
| `metrics.py` | Physical metrics (cm, m), crash and collision rates, error amplification down the chain |
| `utils/` | YAML config inheritance, named RNG streams, run directories and manifests |

---

## 2. One environment step

```mermaid
sequenceDiagram
    autonumber
    participant Env as FormationAviary
    participant Act as Actor (shared)
    participant Sel as Ladder selector
    participant Cri as Critic
    participant PID as DSL PID
    participant Phy as PyBullet

    Env->>Act: local obs oᵢ (13-D) for each drone
    Act->>Act: sample aᵢ ~ tanh(N(μ, σ))
    Env->>Sel: global state (positions, velocities, wind)
    Sel->>Cri: own obs + top-k teammates by coupling rank, zero-padded to 28-D
    Cri-->>Act: value V(s | k), stored for GAE
    Act->>PID: target = slot + 0.2 m × aᵢ
    loop 5 physics substeps (240 Hz)
        PID->>Phy: rotor RPMs
        Phy->>Phy: rigid body + ground effect + drag + downwash + gust force
    end
    Phy-->>Env: new state
    Env->>Env: reward · crash / collision check · truncation at 720 steps
```

---

## 3. One training iteration

```mermaid
flowchart LR
    R["Collect rollout<br/>256 steps × N agents"] --> T{"Episode ended?"}
    T -- "terminated (crash)" --> Z["bootstrap = 0"]
    T -- "truncated (time limit)" --> V["bootstrap = V(s_T)"]
    T -- "no" --> G
    Z --> G["GAE · γ = 0.99, λ = 0.95<br/>never crosses an episode reset"]
    V --> G
    G --> U["PPO update · 10 epochs<br/>clip 0.2 · grad-norm 0.5<br/>actor lr 3e-4 · critic lr 1e-3"]
    U --> D["Diagnostics<br/>explained variance · KL · clip fraction<br/>action saturation"]
    D --> E{"Eval due?"}
    E -- "yes" --> PE["Periodic eval · 8 episodes<br/>per-run seeds · health only"]
    E -- "no" --> R
    PE --> R
```

At the end of training the **final evaluation** runs the deterministic policy on
24 held-out episodes. Those are the reported numbers.

---

## 4. Evaluation protocol

| | Periodic evaluation | Final evaluation |
|---|---|---|
| Purpose | "Is this run healthy?" | The reported number |
| Episodes | 8 | 24 |
| Seeds | per run (`seed × 1000 + 10 000 + i`) | **common to every policy, rung and baseline** (from 900 000) |
| Policy | deterministic | deterministic |
| Reported | never | yes, with physical metrics |

**Common random numbers.** Every condition is scored on the same episodes, so
paired comparisons remove episode-difficulty variance. The **training seed** is the
statistical unit; episodes are nested repeated measures.

**Physical metrics first.** Return is bimodal (crash vs. no crash), so each result
also reports:

- RMS tracking error in cm, **conditional on completion** (crashes would otherwise
  censor it downward)
- crash and collision probability
- lateral error (detects the "sidestep the downwash" exploit)
- error amplification down the chain (string stability)

---

## 5. Reproducibility and provenance

```mermaid
flowchart LR
    Y["configs/*.yaml<br/>_base_ inheritance"] --> RC["RunConfig"]
    RC --> RUN["Run"]
    SD["seed + named stream<br/>make_rng(seed, stream)"] --> RUN
    RUN --> EXP["results/&lt;experiment&gt;/&lt;date&gt;_&lt;task_version&gt;/"]
    EXP --> E1["MANIFEST.json · git_commit.txt<br/>sweep_results.json · summary.json"]
    EXP --> DIR["&lt;rung&gt;_seed&lt;n&gt;_sp&lt;Δ&gt;/"]
    DIR --> F1["run_config.json"]
    DIR --> F3["history.json"]
    DIR --> F4["parameter_report.json"]
    DIR --> F5["final_model.pt"]
```

- **The task version is in every result path** (`v4-correctness`), so results from
  different task definitions cannot be pooled by accident.
- **Named RNG streams** isolate subsystems: evaluating more often never changes the
  training environment's gust sequence.
- **Invalid runs are marked, not deleted.** `mark_status()` writes an `INVALID`
  banner with the reason, and the archive is kept as a record of why decisions
  were made.
- **Pinned simulator commit.** `gym-pybullet-drones` is pinned in
  `requirements.cpu.txt`, because a different commit means different physics.
