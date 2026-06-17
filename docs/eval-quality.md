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

**Raw data volume is NOT the bottleneck — measured.** The `training_data/` PGN
corpus (1.2M games, 1.8 GB; `wthor/` dwarfed) holds **~1.1M positions at empties
14** (and similarly ~0.9–1.2M per bucket from 0–30 empties — ~90% of games reach
the deep endgame; counted via `"N. "` move-number tokens, see git history of this
doc). Training doesn't dedup (`build_examples`, `cache.rs:365`), so that raw count
*is* the example count. ~1.1M exact-labelable examples is plenty for the 10-cell
features (~18 samples/pattern), so a 6-disc MAE points elsewhere.

**Label correctness is also ruled out — measured.** The cached label file
(`ignored/edax_evals.txt`, ~8M lines, Edax-generated) was validated against our own
exact solver via uniform random sampling per empties bucket: **340/340 positions
matched bit-exact across empties 4–20** (0 disc diff), *including* the deep 17–20e
buckets where selective (Edax level < 60) labels would have diverged. So the labels
are true exact ground truth, not approximations. Bucket sizes: ~460–520k labels
each at 4–16e, a thin ~42k tail at 17–20e. The base eval was therefore trained on
*correct, exact, plentiful* ground truth — the weakness is in **training**, not data
or labels. In order of suspected impact:

1. **Was the base actually trained on this much?** `train-exact` must exact-solve
   every ≤16e position (~2 ms at 14e, ~10 ms at 16e), so labelling the full ~1.1M+
   per bucket is multiple hours — the practical gate, not data availability. If the
   base eval was trained on only a few files, it is simply **undertrained** despite
   the large corpus. Check what `ignored/trained_weights.bin` was trained on; if a
   subset, retrain on far more (training itself is fast post-Step-32; the one-time
   exact labels are cached via `--eval-file`, `src/eval/cache`). **Try this first.**
2. **Training method / objective.** If it *was* trained on ~1M examples and still
   sits at ~6-disc MAE, the SGD setup is the culprit. Current trainer is online SGD
   with inverse-time LR decay (`src/training/trainer.rs`), one example at a time.
   Edax uses batched regression over a fixed corpus. Check: convergence (plateau vs
   undertrained?), LR schedule, regularization, and squared-error loss scaling on
   `[-64,64]`. Consider sharing/smoothing weights across adjacent empties buckets
   (Edax-style ply grouping) so rarer patterns borrow strength.
3. **Ground-truth depth.** Exact labels are only cheap at empties ≤ ~16, so the
   directly-supervised region is shallow and everything above is bootstrapped from
   it. Pushing exact labels deeper (via the cache) widens the trustworthy base.

Run `eval-check -n 14` after each change — target "within ±2" well above 50% (and
MAE → low single digits) before trusting the eval downstream.

See [speedup-plan.md](speedup-plan.md) Steps 32–34 for the eval-related solver work
(training speedup, `FlatEval`, eval-guided ordering) and the full Edax-gap analysis.
