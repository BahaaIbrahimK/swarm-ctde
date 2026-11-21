# Figures

Publication-quality plotting for sweep results.

```bash
python figures/plot_sweep.py <results_dir>
```

That is the whole interface — point it at any result directory and it works out
the rungs, seeds, spacings, and task version by itself.

---

## Examples

```bash
# Everything, from the Stage 2 pilot
python figures/plot_sweep.py results/<experiment_id>/<date>_<task_version>

# Just the headline figure
python figures/plot_sweep.py results/<experiment_id>/<date>_<task_version> --only performance

# Custom output location
python figures/plot_sweep.py <dir> --out figures/output/paper_v1

# Final ≥8-seed sweep — drops the "PILOT DATA" stamp
python figures/plot_sweep.py <dir> --not-pilot
```

Output goes to `figures/output/<result-dir-name>/` as **PDF** (for the paper)
and **PNG** (for viewing), plus `statistics.txt`.

---

## The five figures

| # | Figure | What it answers |
|---|---|---|
| 1 | `fig1_performance_vs_k` | **The headline.** Is there an interior optimum? |
| 2 | `fig2_learning_curves` | Did runs converge? How much did seeds disagree? |
| 3 | `fig3_mechanism` | **Why** does the curve bend? |
| 4 | `fig4_task_health` | Is the result an artifact of a broken task? |
| 5 | `fig5_seed_variability` | Is the study powered enough to conclude anything? |

### Figure 1 — Performance vs k

Where Claim 1 lives or dies. Shows mean ± 95% CI **and every individual seed**,
because with 5 seeds the spread *is* the story — plotting only means would
misrepresent how much is noise.

Annotates `k*` only when the interior rung beats **both** endpoints. Beating
only `k0` would merely re-show that centralization helps, known since MADDPG
(2017).

### Figure 3 — Mechanism ← the scientifically interesting one

Two panels side by side, and the argument is in their **divergence**:

- **Left (explained variance):** should rise *monotonically* — more information
  always makes the critic a better predictor.
- **Right (performance):** should *peak and fall* — beyond some `k`, the extra
  accuracy comes from information the actor cannot act on, and becomes noise in
  the policy gradient.

A rising left panel beside a peaked right panel is the signature of
**representational mismatch**. It distinguishes "the critic got worse" (boring)
from "the critic got better but less useful" (the paper's actual claim).

### Figure 4 — Task health

Five bugs were found in this task, four invisible without measurement. This
figure keeps the relevant failure modes visible in the results themselves:

| Panel | Catches |
|---|---|
| Episode completion | v1's "learning to crash faster" |
| Crash rate | termination effects contaminating reward |
| **Lateral error** | **the sidestep exploit** — policies dodging the wake |
| Coupling | whether downwash survived training |

If lateral error grows with training, the exploit is back and the results are
invalid — the coupling the paper studies would have been engineered away.

### Figure 5 — Seed variability

Kill-criterion K3: if seed noise exceeds the between-rung effect, the study is
underpowered and **no conclusion may be drawn**. The figure states that verdict
on itself rather than leaving it to be inferred.

---

## Files

| File | Role |
|---|---|
| `style.py` | Publication style — fonts, colours, sizes, save helpers |
| `loader.py` | Reads result directories into tidy data; statistics |
| `plot_sweep.py` | The five figures + text summary |

### Why a shared `style.py`

Every figure must look like it belongs to the same paper. Changing the look is
a one-file edit rather than a hunt through every script.

Notable choices:

- **Okabe-Ito palette** — colour-blind safe. ~8% of men have red-green colour
  deficiency; reviewers are people.
- **Redundant encoding** — colour is never the only signal; every series also
  has a distinct marker, so figures survive greyscale printing.
- **Fixed colour per rung** — `k1` is the same colour in *every* figure, so a
  reader builds intuition instead of re-reading each legend.
- **Column-width sizing** — `SINGLE_COL = 3.5"`, `DOUBLE_COL = 7.16"`. Figures
  drawn at final size keep their fonts legible; scaling down in LaTeX shrinks
  the text too, which is the most common reason figures look bad in print.

### Statistics: two deliberate choices

**t-distribution, not 1.96.** With n=5 the correct 95% multiplier is 2.78.
Using 1.96 would understate the interval by ~40% — overstating our confidence.

**Welch's t-test, not Student's.** Welch does not assume equal variance, and our
rungs demonstrably differ: in the pilot, `k0`'s std was 104 while `k1`'s was 40.

**Cohen's d is always reported.** With 5 seeds a real effect can easily miss
p<0.05. The effect *size* indicates whether a larger sweep would resolve it —
exactly the decision the pilot exists to inform.

---

## Guards

**Mixed task versions.** Results from `v1-raw4d` and `v2-pidassist` describe
different experiments — different reward, action space, termination. Pooling
them on one axis would be meaningless, so the loader warns loudly when it sees
a mix.

**The pilot stamp.** Pilot figures carry `PILOT DATA (5 seeds) — not a finding`
on the image itself, so the warning travels into slides and emails rather than
living only in a caption that gets separated from the figure.

---

## Adding a figure

Add a function to `plot_sweep.py`:

```python
def fig_my_thing(sweep: Sweep, out_dir: Path, is_pilot: bool = True):
    fig, ax = plt.subplots(figsize=(SINGLE_COL, SINGLE_COL * GOLDEN))
    # ... sweep.rungs, sweep.summary(), sweep.by_rung("k1") ...
    if is_pilot:
        annotate_pilot_warning(fig)
    fig.tight_layout()
    return save(fig, out_dir, "fig6_my_thing")
```

Then register it in the `figures` dict inside `main()`.
