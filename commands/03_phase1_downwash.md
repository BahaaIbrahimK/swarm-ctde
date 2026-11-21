# 03 — Phase 1: the downwash probe

**This is the premise check for the entire paper.** Charter §5, Phase 1 exit
artifact.

```bash
cd ~/projects/swarm-ctde
source .venv/bin/activate
```

---

## Why this runs before any learning code

The paper's logic chain (Charter §1.5):

```
downwash exists
  -> a drone dips for reasons its OWN observation cannot explain
     -> a critic that sees the drone ABOVE can explain that dip
        -> variance reduction (the force pushing performance UP as k grows)
           -> but it SATURATES once the critic sees the drones that matter
              -> meanwhile cost keeps growing
                 -> therefore an INTERIOR OPTIMUM  (Claim 1)
```

Every link depends on the first. If downwash is negligible at our spacings,
**Claim 1 and Claim 2 are both dead** and no amount of good MAPPO code rescues
them.

So we test it now — cheaply, in week 3 — rather than discovering the problem in
month 5. Same "fail fast" logic as the Phase 3 decision gate.

---

## Run it

```bash
python scripts/run_downwash_probe.py
```

Takes a few minutes: 9 vertical separations × 6 horizontal offsets = 54
measurement points, each with a settle phase and a measurement phase.

### Options

```bash
# Use a different config
python scripts/run_downwash_probe.py --config configs/probe_downwash.yaml

# Different seed (the probe is near-deterministic, but this proves it)
python scripts/run_downwash_probe.py --seed 3

# Skip plotting (headless machine, or matplotlib trouble)
python scripts/run_downwash_probe.py --no-plot
```

---

## Where the output goes

```
results/phase1_downwash_probe/<timestamp>_seed<N>/
├── downwash_measurements.csv           raw numbers (source of truth)
├── downwash_force_vs_separation.png    the Phase 1 exit figure
├── config.yaml                         exact settings used
└── git_commit.txt                      exact code version
```

The last two exist so that months later any result traces back to the code and
settings that produced it (Charter §7).

```bash
# See the newest run
ls -lt results/phase1_downwash_probe/ | head

# Read the raw numbers
column -s, -t < results/phase1_downwash_probe/*/downwash_measurements.csv | less -S
```

---

## How to read the result

The script prints a summary block. **Read that first** — it answers the premise
question in plain language before you open anything.

### Three possible outcomes

| Outcome | What you see | What it means | Action |
|---|---|---|---|
| ✅ **Premise holds** | Clear differential drift at tight spacing, decaying with distance; a coupling horizon is reported | Downwash coupling is real and local, exactly as the mechanism requires | Proceed to Phase 2 |
| ❌ **Premise fails** | `COUPLING HORIZON: NOT DETECTED`, differential ≈ 0 everywhere | No coupling → nothing for a teammate-aware critic to explain → Claims 1 & 2 untestable | **Stop.** Tighten spacings, or verify downwash physics is on |
| ⚠️ **Range too narrow** | Drift present but never decays across the sweep | The horizon is beyond our furthest measurement | Extend `vertical_separations_m` in the config, re-run |

### The key number: coupling horizon

The separation beyond which downwash becomes negligible.

**This is our a-priori prediction for where `k*` sits.** Per Charter §1.5, the
critic should see teammates *inside* that horizon and gains little from those
outside it. Having this number *before* running the sweep is what makes Claim 2
a genuine prediction rather than a story fitted after the fact — which is
precisely what a reviewer will probe.

### Sanity check to perform yourself

In the summary, compare the two drift numbers at the tightest spacing:

- **lower-drone drift** should be clearly negative (it sank)
- **upper-drone drift** should be near zero (it is in clean air — this is the control)

If the *upper* drone also sank substantially, the measurement is contaminated
(likely hover RPM slightly miscalibrated) and the differential is the number to
trust, not the raw lower-drone figure. That is exactly why the probe reports a
differential at all.

---

## Tuning the sweep

Edit `configs/probe_downwash.yaml` — never hardcode values in Python.

```yaml
probe:
  # Coarser/faster sweep while debugging
  vertical_separations_m: [0.10, 0.30, 1.00]
  horizontal_offsets_m: [0.00, 0.10]

  # Shorter windows = faster but noisier
  settle_steps: 120      # 0.5 s at 240 Hz
  measure_steps: 120
```

A quick coarse run is a good first move to confirm the pipeline works end to
end before committing to the full 54-point sweep.

---

## When it finishes

Send me:

1. The printed summary block
2. The PNG

I will tell you whether the premise holds, and what the coupling horizon
implies for the observability ladder we build in Phase 2.
