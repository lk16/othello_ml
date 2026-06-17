# Eval Quality — the gating problem

The trained position evaluator is **too inaccurate to be useful in the search**.
This is the single blocker on the two biggest remaining solver speedups, so it is
the project's critical path. This doc states the problem, the evidence, how to
measure it, and the leading hypotheses — enough to resume work cold.

## Why it matters

The Edax comparison (see [speedup-plan.md](speedup-plan.md), "Edax comparison")
showed our exact solver is ~8× slower than Edax almost entirely on **node count**,
and the two levers that close that gap both need a strong eval:

- **Eval-guided move ordering** (Step 34) — the ~3.7× node-count lever.
- **Eval-seeded MTD-f / aspiration** (Step 31) — the ~2× window lever.

Both are *built and wired* but **net-negative or neutral with the current weights**
(eval-guided ordering costs +0.7–2.4% nodes instead of saving). The mechanisms are
correct; they are starved of a good eval. Nothing else in the roadmap matters as
much until this is fixed.

## How to measure it: `eval-check`

```bash
cargo run --release -- eval-check -w ignored/trained_weights.bin -n 18 -m 500 training_data/
```

For each position at `-n` empties, compares `FlatEval::eval_position` to the
**exact** negamax score and reports error **in discs**. This is the direct
eval-quality signal — unlike `bench --weights`, which only sees the downstream
move-ordering effect and conflates accuracy with search depth. Exact solve limits
it to shallow empties (~≤22); `-m` caps the sample. Code: `run_eval_check` in
`src/main.rs`.

Reported metrics: MAE, RMSE, bias (mean `pred − exact`), max abs error, "within ±2
discs" %, and W/D/L sign agreement.

## Current evidence (2026-06, `ignored/trained_weights.bin`)

`eval-check` on `training_data/playok_pgn_75000000.pgn`:

| empties | region | MAE | RMSE | bias | within ±2 | W/D/L sign |
|---|---|---|---|---|---|---|
| 14 | exact-trained base | 5.85 | 7.67 | +0.07 | 25.2% | 93.2% |
| 18 | boot-trained (`train-boot`) | 6.08 | 7.98 | −1.88 | 23.5% | 90.5% |

Reading:
- **MAE ~6 discs, only ~25% of predictions within ±2 discs.** Move ordering must
  rank sibling moves whose exact scores often differ by 2–4 discs; an eval this
  coarse cannot, which is exactly why eval-guided ordering loses the `bench
  --weights` A/B.
- **Sign accuracy is decent (~90–93%)** — the eval knows *who is winning* but not
  *by how much*. Magnitude is what ordering needs.
- **The base (14e) is as weak as the boot region (18e).** So this is **not** a
  bootstrapping artifact — `train-boot` propagated the base's weakness outward
  roughly intact (plus a small negative bias at 18e). The base eval from
  `train-exact` is itself the problem.

## What this is and isn't

- **Not model capacity.** The feature set is Edax's own 47-pattern set
  (`Features::edax()` in `src/training/features.rs`: 3×3 corners, edge/extended-edge
  10-cell patterns, lines, diagonals — trinary indices), and weights are bucketed
  **per empties value** (61 buckets, one per `0..=60`; `EMPTY_RANGE_COUNT` in
  `src/training/weights.rs`). Edax achieves sub-disc accuracy from these *same*
  features, so the representation is sufficient. The gap is **how/what we train**,
  not the model.
- **Not the bootstrap curriculum.** Evidence above: the exact base is equally weak.
  Fix the base first; `train-boot` can only be as good as the eval it boots from.
- **Not search wiring.** Steps 31/34 are built and A/B-able. They are waiting on
  the eval, confirmed by the gate.

## Ruled out by measurement

- **Raw data volume.** The `training_data/` PGN corpus (1.2M games, 1.8 GB; `wthor/`
  dwarfed) holds **~1.1M positions at empties 14** (~0.9–1.2M per bucket from 0–30e
  — ~90% of games reach the deep endgame; counted via `"N. "` move-number tokens).
  Training doesn't dedup (`build_examples`, `cache.rs:365`), so that raw count *is*
  the example count. Plentiful.
- **Label correctness.** The cached label file (`ignored/edax_evals.txt`, ~8M lines,
  Edax-generated) was validated against our exact solver via uniform random sampling
  per bucket: **340/340 bit-exact across empties 4–20** (0 diff), including deep
  17–20e where selective labels would diverge. True exact ground truth. Bucket sizes
  ~460–520k each at 4–16e, ~42k tail at 17–20e.
- **The optimizer.** A from-scratch single-thread retrain (`-t 1`, online SGD) drives
  *in-sample* 14e MAE to **2.82** (within-2 48.5%) — it fits fine.
- **`-t > 1` for training is broken, though.** The parallel clone/merge path (Step
  32) converges far worse: same data/epochs, `-t 16` gave loss 92.8 / MAE 7.24 vs
  `-t 1`'s loss 13.6 / MAE 2.82. **Always train with `-t 1`.** (`-t` also controls
  missing-label solve parallelism, so only raise it when the slice is NOT fully
  cached — and then accept the worse training, or solve labels in a separate pass.)

## The real bottleneck: generalization (overfitting)

Every retrain shows a large **in-sample vs held-out** gap. Held-out 14e MAE on a
common 2000-position set (`760*` files, `-t 1`, 60 epochs), vs the in-sample 2.82:

| training set | ≤16e examples | held-out MAE | within ±2 |
|---|---|---|---|
| 100 files | 0.36M | 11.18 | 12.0% |
| 500 files | 2.14M | 9.10 | 14.3% |
| base `trained_weights.bin` | (more) | **6.04** | 26.3% |

More data monotonically improves held-out (11.18 → 9.10 → base 6.04), confirming
data helps — but with diminishing returns, and the eval *memorizes* (2.82 in-sample)
while generalizing poorly. **Primary structural cause: no symmetry exploitation.**
`Features::edax()` defines **50 independent features** (`feature_weights:
Vec<Vec<Vec<f32>>>`, one table per feature), so the 4 corner features
(`corner_a1/h1/a8/h8`) — the *same* 3×3 pattern under rotation/reflection — learn
*separate* weights, and there is **zero board-symmetry augmentation** in training.
Each physical pattern therefore sees only ~1/4–1/8 of its occurrences, and the eval
isn't symmetry-consistent. Edax mirror-packs weights over the 8-fold symmetry → ~8×
the effective samples per pattern. We get 1×.

## Symmetry handling — DONE (augmentation → weight tying)

Implemented and measured. The feature set first had to be fixed: `Features::edax()`
was hand-transcribed with **bugs** (`corner_a8`, `ext_corner_*`, `diag_4_*` had wrong
cells; a bogus `edge_parity`), surfaced by deriving symmetry orbits. It is now
transcribed **verbatim from Edax `eval.c` `EVAL_F2X`** (46 features = 12 symmetry
shapes; validated by `edax_features_form_clean_symmetry_orbits`).

Two mechanisms tried, same regularization effect:
- **8-fold augmentation** (feed each example's 8 board symmetries): 500f held-out 14e
  **9.10 → 8.11**, collapsed the gap (in-sample 6.70 vs held-out 8.11). Cost: 8×
  examples, 419 s.
- **Weight tying** (`Weights` stores one shared table per symmetry shape — 12 vs 46;
  Edax mirror-packing): 500f held-out **8.13** (≈ augmentation) at **56 s — ~7×
  cheaper**, smaller model, exact symmetry. **This is the committed approach**
  (augmentation removed). `--threads` now only parallelizes missing-label solving;
  weight training is always sequential.

More data on top of tying: 1000 files (4.5M ≤16e) → held-out **7.86**, in-sample
**6.32**. (The base's "6.04" is *not* a clean baseline — it was trained on the whole
corpus incl. the 760* "held-out" set, so it's effectively in-sample; on equal
in-sample footing tied ≈ base.)

**The remaining ceiling: ~6-disc MAE even IN-SAMPLE** (21% within ±2). Symmetry fixed
the *generalization gap* and the feature bugs, but not the absolute accuracy. Edax
reaches <2 MAE from the *same* features, so the gap is now the **training method**,
not symmetry or data quantity.

## Remaining levers (in order of expected value)

1. **Training method / objective** (the ceiling). Our trainer is per-example online
   SGD with inverse-time LR decay and a fixed `gradient = 2·error/n_features` step
   (`trainer.rs`). Edax fits by **batched least-squares regression** over the whole
   corpus. Try: true mini-batch / full-batch gradient accumulation, proper LR tuning,
   more epochs to convergence, L2 regularization, and weight-sharing across adjacent
   empties buckets (Edax-style ply grouping) so rare patterns borrow strength. This
   is now the gating lever for breaking the ~6-MAE ceiling.
2. **More data**, cheap now that tying is 1× cost and `-t N` solves uncached labels:
   train on the full corpus (extend the cache once with `-t 16`). Helps the gap but
   has diminishing returns against the ceiling above.
3. **Ground-truth depth.** Exact labels are cheap only at ≤ ~16e; pushing them
   deeper (via the cache) widens the directly-supervised base for `train-boot`.

Run `eval-check -n 14` on a **held-out** file after each change — target "within ±2"
well above 50% and held-out MAE → low single digits before trusting the eval
downstream. (In-sample numbers flatter; always measure held-out.)

See [speedup-plan.md](speedup-plan.md) Steps 32–34 for the eval-related solver work
(training speedup, `FlatEval`, eval-guided ordering) and the full Edax-gap analysis.
