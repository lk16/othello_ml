# Eval Quality — two bars: move-ordering (met) and score accuracy (open)

The eval is judged against **two distinct bars**, and they have very different status:

1. **Move ordering — MET.** The eval started *too inaccurate to help the search*
   (eval-guided ordering cost +0.7–1.8% nodes). After the fixes below (corrected
   46-feature transcription, symmetry **weight tying**, mini-batch + L2, exact labels
   extended to ≤18e → `ignored/weights_v4.bin`), eval-guided ordering now **cuts ~34%
   of nodes and is ~1.27× faster wall-clock at 18e** (speedup-plan Step 34). Ordering
   only needs the eval to *rank* sibling moves better than mobility, and it now does.

2. **Absolute score accuracy — OPEN, and now understood to be a model-capacity
   ceiling.** The eval is trained against **exact end-of-game scores**, so it *is* an
   estimator of the final disc differential under perfect play. We want to **surface
   that estimate to the user in the future** (e.g. show it during `play`), and for
   that the *absolute* number matters, not just the ranking. It plateaus at **~8-disc
   MAE held-out (~16% within ±2) at empties 14** — poor for a human-facing estimate.
   A conjugate-gradient solver that reaches the **exact least-squares optimum** does
   *no better than SGD* (8.2 vs 8.0 MAE) and is only **7.7 MAE even in-sample**, so
   the gap is **not** the optimizer, **not** overfitting, and **not** data quantity —
   it is the **linear pattern model's capacity** (or the difficulty of the target).
   See [The capacity ceiling](#the-capacity-ceiling-cg-least-squares-experiment).

So the move-ordering bar no longer blocks the solver speedups, but score accuracy
remains the eval's primary unmet goal. This doc records how we got here and what is
still open.

## Why it matters

The Edax comparison (see [speedup-plan.md](speedup-plan.md), "Edax comparison")
showed our exact solver is ~8× slower than Edax almost entirely on **node count**,
and the two levers that close that gap both need a strong eval:

- **Eval-guided move ordering** (Step 34) — the ~3.7× node-count lever.
- **Eval-seeded MTD-f / aspiration** (Step 31) — the ~2× window lever.

Both are *built and wired* but **net-negative or neutral with the current weights**
(eval-guided ordering costs +0.7–2.4% nodes instead of saving). The mechanisms are
correct; they are starved of a good eval. (As of `weights_v4.bin` the move-ordering
lever now pays off — see the status section above.)

Beyond the solver, the eval has a **second, standalone purpose**: it predicts the
perfect-play end score, which we want to **show to the user** (e.g. during `play`) as
a position assessment. That use needs *absolute* accuracy — the open bar above — not
just move ranking.

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

- **Model capacity — now the leading suspect.** The feature set is Edax's own
  46-pattern set (`Features::edax()` in `src/training/features.rs`: 3×3 corners,
  edge/extended-edge 10-cell patterns, lines, diagonals — trinary indices), and
  weights are bucketed **per empties value** (61 buckets, one per `0..=60`;
  `EMPTY_RANGE_COUNT` in `src/training/weights.rs`). We *believed* this was
  sufficient because "Edax achieves sub-disc accuracy from these same features" —
  but the [CG experiment](#the-capacity-ceiling-cg-least-squares-experiment) shows
  the **exact least-squares optimum** of this *linear* model is only ~7.7 MAE
  in-sample at 14e. So either the linear pattern-sum model genuinely cannot do
  better on exact-solve targets, or Edax's reputed accuracy is measured differently
  (post-search at the leaves, on its own self-play distribution, or as RMSE) and
  isn't a like-for-like standalone-MAE comparison. **This needs verifying before
  trusting the "representation is sufficient" assumption.**
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

## Generalization gap — a small-data artifact, not the ceiling

> **Superseded by the [CG experiment](#the-capacity-ceiling-cg-least-squares-experiment).**
> This section described the *small-data* regime, where the model has far more
> weights than examples-per-bucket and **memorizes**. The large in-sample↔held-out
> gap below is real *there*, but it **closes once data is plentiful**, revealing a
> capacity floor (in-sample ≈ held-out ≈ 8 MAE) rather than an overfitting problem.
> Symmetry tying (next section) was still a worthwhile fix. Kept for history.

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

**The remaining ceiling: ~8-disc MAE even IN-SAMPLE** (with plentiful data). Symmetry
fixed the *generalization gap* and the feature bugs, but not the absolute accuracy.
We *hypothesized* the gap was now the **training method** — but the CG experiment
below **refutes that**: switching to the exact least-squares optimum does not help.
See [The capacity ceiling](#the-capacity-ceiling-cg-least-squares-experiment).

## The capacity ceiling (CG least-squares experiment)

To test whether the optimizer was the bottleneck, we replaced SGD with a per-bucket
**conjugate-gradient least-squares solver** (`src/training/cg.rs`, `train -o cg`),
which reaches the *exact global optimum* of the convex per-bucket objective — Edax's
own `eval_builder` method. Ridge is regularization, specified **per example** so a
single value is scale-invariant across data sizes (the data term is an implicit sum
over N examples, so internally `ridge·N` is used; default `0.001`). All runs at
empties 14, exact labels from `ignored/edax_evals.txt`.

| run | train set | MAE | within ±2 | bias | notes |
|---|---|---|---|---|---|
| SGD (60 ep) | 75* (3.9M ex) | **7.99** | 17.1% | +1.09 | held-out `760*` (same-dist, unseen) |
| CG (ridge 1e-3) | 75* (3.9M ex) | 8.19 | 16.0% | **+0.08** | held-out `760*`; **3 s** train vs SGD's 113 s |
| CG (ridge 1e-3) | 75* (3.9M ex) | **7.67** | 16.7% | — | **in-sample** (`7500*`) |

Findings:

- **CG ≈ SGD on held-out** (8.2 vs 8.0 MAE). The exact least-squares optimum is *not*
  better than SGD's early-stopped solution. CG is unbiased by construction (+0.08 vs
  SGD's +1.09) and ~37× faster to fit, but no more accurate.
- **In-sample ≈ held-out for CG** (7.67 vs 8.19). The ~0.5-disc gap means **no
  overfitting** at this data scale, and the global optimum *itself* is only 7.7 MAE.
- Therefore the ceiling is **model capacity / target difficulty**, not the optimizer,
  not overfitting, not data quantity. At empties 14 the exact disc-differential under
  perfect play carries tactical variance a static linear pattern-sum cannot capture.
- The old "2.82 in-sample" was the **small-data overfit regime** (100 files: ~115k
  weights/bucket ≫ examples → memorization, held-out 11.18). With full data the model
  can't memorize and in-sample rises to the true floor.

**Implications:** more data won't help (already at the floor); a better optimizer
won't help (CG is optimal). Breaking ~8 MAE needs a **richer model** (pattern
*interactions* / non-linearity) or a reconsidered target — *or* first confirming
whether Edax's reputed <2 MAE is a like-for-like standalone-MAE figure at all (it may
be measured post-search or on a different distribution). CG stays as a fast,
unbiased, hyperparameter-light default trainer regardless.

## Remaining levers (in order of expected value)

1. **Richer model — the capacity lever.** The CG experiment shows the *linear*
   pattern-sum is at its floor (~8 MAE). Breaking it needs representational power the
   current model lacks: pattern **interactions** / non-linearity (e.g. a small MLP or
   GBDT over the same pattern indices), more/larger patterns, or finer phase
   conditioning. This is now the gating lever for absolute accuracy.
2. **Verify the Edax <2-MAE premise** *before* investing in (1). All "the model is
   sufficient" reasoning rests on Edax reaching sub-disc accuracy from the same
   features. Confirm whether that is a standalone eval-vs-exact MAE at a fixed empties
   (like our `eval-check`), or measured post-search / on self-play / as RMSE. If it
   isn't like-for-like, ~8 MAE standalone may simply be the honest ceiling for a
   static eval at 14e, and the eval should be judged only by the **move-ordering bar
   (already met)**.
3. **Ground-truth depth.** Exact labels are cheap only at ≤ ~16e; pushing them
   deeper (via the cache) widens the directly-supervised base for `train-boot`.

**Ruled out (CG experiment):** the **training method / optimizer** is *not* a lever —
the exact least-squares optimum (CG) matches SGD. Likewise **more data** and
**overfitting** are not the issue at full-corpus scale.

Run `eval-check -n 14` on a **held-out** file after each change — target "within ±2"
well above 50% and held-out MAE → low single digits before trusting the eval
downstream. (In-sample numbers flatter; always measure held-out.)

See [speedup-plan.md](speedup-plan.md) Steps 32–34 for the eval-related solver work
(training speedup, `FlatEval`, eval-guided ordering) and the full Edax-gap analysis.
