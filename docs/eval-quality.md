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

## Leading hypotheses (where to dig next)

Roughly in order of suspected impact:

1. **Training data volume / coverage.** Edax trains on millions of self-play
   positions; we train on a few PlayOK human-game PGNs (~13k positions/file, ~31k
   for 10 files), now spread across **61 per-empties buckets** — likely too few
   examples per bucket to fit ~892K patterns/feature. Try: far more games; verify
   per-bucket example counts; consider sharing/smoothing weights across adjacent
   empties buckets (Edax-style ply grouping) so sparse buckets borrow strength.
2. **Ground-truth availability.** Exact labels (`train-exact`) are only cheap at
   empties ≤ ~16, so the directly-supervised region is thin and everything above is
   bootstrapped from it. More exact labels (deeper, via the cache `src/eval/cache`)
   widen the trustworthy base.
3. **Training method / objective.** Current trainer is online SGD with inverse-time
   LR decay (`src/training/trainer.rs`), one example at a time. Edax uses batched
   regression over a fixed corpus. Check: convergence (does MAE plateau or is it
   undertrained?), LR schedule, regularization, and whether the squared-error loss
   on `[-64,64]` scores is well-scaled. Run `eval-check -n 14` after each change —
   target "within ±2" well above 50% (and MAE → low single digits) before trusting
   the eval downstream.

See [speedup-plan.md](speedup-plan.md) Steps 32–34 for the eval-related solver work
(training speedup, `FlatEval`, eval-guided ordering) and the full Edax-gap analysis.
