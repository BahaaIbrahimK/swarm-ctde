"""
================================================================================
 RUN NAMING & RESULT ORGANISATION
================================================================================

WHY THIS FILE EXISTS
--------------------
We will generate roughly 200+ runs across Claims 1-4, over several months, and
then draw figures from them. If a directory is called `k1_seed0_sp1.00` we
cannot answer, six weeks later:

  * which experiment was this -- Claim 1 or Claim 2?
  * which code version produced it?
  * was this before or after the PID-assist fix?
  * is it a pilot, a full sweep, or a throwaway debug run?

Ambiguity here becomes a crisis at figure-drawing time, when results from
different code versions silently get pooled into one plot.

THE NAMING SCHEME
-----------------
    results/
      EXP01_claim1_pilot/                     <- experiment, not a timestamp
        2026-08-28_v2-pidassist/              <- date + CODE VERSION TAG
          MANIFEST.json                       <- what/when/why, machine-readable
          README.txt                          <- same, human-readable
          runs/
            k0_seed00_sp1.00/
            k1_seed00_sp1.00/
            ...
          analysis/
            summary.json
            figures/

KEY DECISIONS
-------------
1. EXPERIMENT ID FIRST (EXP01, EXP02...). Groups by scientific question, not by
   when it happened. All Claim-1 pilots live together regardless of date.

2. CODE VERSION TAG IN THE PATH. `v2-pidassist` means every run beneath it used
   PID-assisted control. This is the single most important field: results from
   different task definitions must NEVER be pooled into one figure, and a tag
   in the path makes that mistake visible instead of silent.

3. ZERO-PADDED SEEDS (seed00, not seed0). Sorts correctly with 10+ seeds.

4. MANIFEST IN EVERY DIRECTORY. Records git commit, task version, date, purpose
   and status. Answers "what is this?" without reading code.

5. STATUS FIELD. Runs get marked `superseded` or `invalid` rather than deleted,
   with a reason. The failed 300k run is evidence of WHY the task changed --
   deleting it loses that.
================================================================================
"""

from __future__ import annotations

import json
import socket
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path

from ctde.utils.seeding import get_git_commit


# ==============================================================================
# SECTION 1 : TASK VERSION  --  the most important field
# ==============================================================================
#
# Bump this whenever the TASK DEFINITION changes in a way that makes results
# incomparable: reward function, action space, termination rules, control mode.
#
# Do NOT bump for hyperparameter changes -- those are recorded in the config.
#
# HISTORY
#   v1-raw4d      Direct 4-D rotor control, no assist. ABANDONED: unlearnable.
#                 Episodes ended after 20-100 of 720 steps across 300k steps of
#                 training; per-step penalties meant crashing early scored
#                 BETTER than flying badly, so the policy learned to die fast.
#
#   v2-pidassist  PID-assisted attitude control + survival shaping + lateral
#                 penalty. Current.
# ==============================================================================

TASK_VERSION = "v4-correctness"

TASK_VERSION_NOTES = {
    "v1-raw4d": (
        "Direct 4-D rotor RPM, no attitude assist. ABANDONED 2026-08-28: task "
        "unlearnable from scratch (episodes 20-100 of 720 steps after 300k "
        "steps) and per-step penalties rewarded early crashing."
    ),
    "v2-pidassist": (
        "PID-assisted control with survival bonus and lateral penalty. "
        "SUPERSEDED 2026-08-31: (a) followers were measured against an "
        "ANALYTICAL leader trajectory, so 'decentralized execution' was not "
        "honest; (b) steady wind and steady downwash are both cancelled by the "
        "PID, leaving seed variance of 0.00 -- nothing for a critic to explain; "
        "(c) wind force used the wrong PyBullet frame and application point."
    ),
    "v3-chain-gust": (
        "Chain topology + stochastic gust field. SUPERSEDED 2026-09-16 by an "
        "external code audit that found five defects invalidating inference: "
        "(1) GAE read dones[t+1] instead of dones[t], so a mid-rollout terminal "
        "leaked reward and value across the episode boundary; (2) GAE recursion "
        "ran through environment resets because truncation was never stored; "
        "(3) the crash penalty was DEAD CODE -- BaseAviary calls _computeReward "
        "before _computeTerminated, so the flag was always False when the reward "
        "was computed; (4) inter-drone collisions were never detected, so the "
        "reported crash_rate did not measure collisions; (5) the privileged "
        "rung received only the steady mean wind, not the stochastic gust that "
        "constitutes the actual disturbance."
    ),
    "v4-correctness": (
        "All five audit defects fixed and covered by tests that exercise the "
        "PUBLIC path rather than bypassing it. Adds: correct termination and "
        "truncation masks in GAE with final-state bootstrapping; order-"
        "independent crash penalty via a pure _check_failure(); inter-drone "
        "collision termination with min_separation_m reporting; privileged "
        "critic receiving the true per-drone air velocity; and EXACT critic "
        "capacity control via a padded fixed-width input with external masking."
    ),
}


# ==============================================================================
# SECTION 2 : EXPERIMENT REGISTRY
# ==============================================================================
#
# Every experiment gets a stable ID. Referenced in the paper, in figure
# captions, and in the charter -- so "EXP03" means one thing forever.
# ==============================================================================

EXPERIMENTS = {
    "EXP00_smoke":          "Pipeline smoke tests. Never reported.",
    "EXP01_learnability":   "Can the agent learn the task at all? Precondition for everything.",
    "EXP02_budget":         "How many steps until convergence? Sizes later sweeps.",
    "EXP03_claim1_pilot":   "Claim 1 pilot: does k matter? DECISION GATE (kill-criterion K1).",
    "EXP04_claim1_full":    "Claim 1 full sweep: performance vs k, >=8 seeds. PAPER FIGURE 1.",
    "EXP05_claim2_coupling": "Claim 2: does k* track coupling strength? PAPER FIGURE 2.",
    "EXP06_claim3_roles":   "Claim 3: role-conditional vs uniform observability.",
    "EXP07_mechanism":      "Phase 7: why the curve bends. PAPER FIGURE 3.",
    "EXP08_hardware":       "Claim 4: sim-to-real transfer on Crazyflies.",
}


# ==============================================================================
# SECTION 3 : MANIFEST
# ==============================================================================

@dataclass
class RunManifest:
    """Self-describing record written into every result directory.

    Answers, without reading any code:
        What is this? When? What code? Is it still valid?
    """

    experiment_id: str
    purpose: str
    task_version: str
    git_commit: str
    created: str
    hostname: str

    # ---- Scientific content --------------------------------------------------
    rungs: list[str] = field(default_factory=list)
    seeds: list[int] = field(default_factory=list)
    spacings: list[float] = field(default_factory=list)
    total_env_steps: int = 0
    num_agents: int = 4

    # ---- Lifecycle -----------------------------------------------------------
    # active      -- current, usable
    # superseded  -- replaced by a later run; kept for the record
    # invalid     -- known-bad; kept as evidence of WHY something changed
    status: str = "active"
    status_reason: str = ""
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    def to_text(self) -> str:
        """Human-readable README written alongside the JSON."""
        lines = [
            "=" * 70,
            f" {self.experiment_id}",
            "=" * 70,
            "",
            f"  PURPOSE      {self.purpose}",
            f"  STATUS       {self.status.upper()}",
        ]
        if self.status_reason:
            lines.append(f"               ({self.status_reason})")
        lines += [
            "",
            f"  created      {self.created}",
            f"  host         {self.hostname}",
            f"  git commit   {self.git_commit}",
            f"  task version {self.task_version}",
            f"               {TASK_VERSION_NOTES.get(self.task_version, '')}",
            "",
            "  CONFIGURATION",
            f"    rungs        {', '.join(self.rungs) if self.rungs else '-'}",
            f"    seeds        {self.seeds if self.seeds else '-'}",
            f"    spacings     {self.spacings if self.spacings else '-'} m",
            f"    agents       {self.num_agents}",
            f"    steps/run    {self.total_env_steps:,}",
        ]
        if self.notes:
            lines += ["", "  NOTES", f"    {self.notes}"]

        lines += [
            "",
            "=" * 70,
            "  WARNING: never pool results across different task versions.",
            "  A different task version means a different experiment.",
            "=" * 70,
        ]
        return "\n".join(lines)


# ==============================================================================
# SECTION 4 : PATH CONSTRUCTION
# ==============================================================================

def make_experiment_dir(
    results_root: Path,
    experiment_id: str,
    task_version: str = TASK_VERSION,
    date: datetime | None = None,
) -> Path:
    """Build (and create) the directory for one experiment run.

    Layout:
        results/<experiment_id>/<YYYY-MM-DD>_<task_version>/

    The date-plus-version directory means re-running the same experiment later
    creates a sibling rather than overwriting -- and if the task version
    changed, that is visible in the path itself.
    """
    if experiment_id not in EXPERIMENTS:
        raise ValueError(
            f"Unknown experiment_id '{experiment_id}'. "
            f"Register it in EXPERIMENTS first. Known: {sorted(EXPERIMENTS)}"
        )

    date = date or datetime.now()
    path = Path(results_root) / experiment_id / f"{date:%Y-%m-%d}_{task_version}"
    path.mkdir(parents=True, exist_ok=True)
    (path / "runs").mkdir(exist_ok=True)
    (path / "analysis").mkdir(exist_ok=True)
    return path


def make_run_name(rung: str, seed: int, spacing: float, num_agents: int | None = None) -> str:
    """Name one individual run directory.

    Format:  k1_seed03_sp1.00           (or  k1_seed03_sp1.00_n6  for non-default N)

    Seeds are zero-padded to two digits so `ls` sorts correctly past seed 9.
    Agent count appears only when it differs from the default, keeping the
    common case readable.
    """
    name = f"{rung}_seed{seed:02d}_sp{spacing:.2f}"
    if num_agents is not None and num_agents != 4:
        name += f"_n{num_agents}"
    return name


def write_manifest(
    directory: Path,
    experiment_id: str,
    *,
    rungs: list[str] | None = None,
    seeds: list[int] | None = None,
    spacings: list[float] | None = None,
    total_env_steps: int = 0,
    num_agents: int = 4,
    task_version: str = TASK_VERSION,
    status: str = "active",
    status_reason: str = "",
    notes: str = "",
) -> RunManifest:
    """Write MANIFEST.json + README.txt into a result directory."""
    manifest = RunManifest(
        experiment_id=experiment_id,
        purpose=EXPERIMENTS.get(experiment_id, "(unregistered)"),
        task_version=task_version,
        git_commit=get_git_commit(),
        created=datetime.now().isoformat(timespec="seconds"),
        hostname=socket.gethostname(),
        rungs=rungs or [],
        seeds=seeds or [],
        spacings=spacings or [],
        total_env_steps=total_env_steps,
        num_agents=num_agents,
        status=status,
        status_reason=status_reason,
        notes=notes,
    )

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    with open(directory / "MANIFEST.json", "w", encoding="utf-8") as f:
        json.dump(manifest.to_dict(), f, indent=2)

    (directory / "README.txt").write_text(manifest.to_text(), encoding="utf-8")
    return manifest


def mark_status(directory: Path, status: str, reason: str) -> None:
    """Re-label an existing result directory.

    Use `superseded` or `invalid` INSTEAD OF DELETING. A failed run is evidence
    of why the design changed; deleting it destroys that record and invites
    repeating the mistake.
    """
    directory = Path(directory)
    manifest_path = directory / "MANIFEST.json"

    if manifest_path.exists():
        with open(manifest_path, encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = {"experiment_id": "(unknown)", "purpose": "(pre-manifest run)"}

    data["status"] = status
    data["status_reason"] = reason
    data["status_changed"] = datetime.now().isoformat(timespec="seconds")

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    banner = (
        f"\n{'=' * 70}\n"
        f"  STATUS: {status.upper()}\n"
        f"  {reason}\n"
        f"  (marked {data['status_changed']})\n"
        f"{'=' * 70}\n"
    )
    readme = directory / "README.txt"
    existing = readme.read_text(encoding="utf-8") if readme.exists() else ""
    readme.write_text(banner + existing, encoding="utf-8")


def find_runs(
    results_root: Path,
    experiment_id: str | None = None,
    task_version: str | None = None,
    status: str = "active",
) -> list[Path]:
    """Locate result directories matching filters.

    Used at figure-drawing time. Filtering by `task_version` is what prevents
    accidentally pooling v1 and v2 results into a single plot.
    """
    root = Path(results_root)
    if not root.exists():
        return []

    found: list[Path] = []
    for manifest_path in root.glob("*/*/MANIFEST.json"):
        with open(manifest_path, encoding="utf-8") as f:
            data = json.load(f)

        if experiment_id and data.get("experiment_id") != experiment_id:
            continue
        if task_version and data.get("task_version") != task_version:
            continue
        if status and data.get("status") != status:
            continue

        found.append(manifest_path.parent)

    return sorted(found)
