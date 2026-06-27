# Alphabeta Exact-Search Speed-Up Plan

Roadmap and design log for the exact endgame solver (`src/eval/alphabeta/`).
Mechanism details live in the code and its doc comments — this file keeps the
*why*: the benchmark recipe, the current baseline, what each step changed, dead
ends, and what's left. Per-constant tuning sweeps live in the constant doc
comments in `alphabeta/`, not here.

## Benchmark

Pulls positions with exactly `--empties` discs free from a fixed set of PGN
files and solves each. `bench` prints to stderr. Node counting is consistent —
one node per visited position across all solvers — so `nodes/s` is meaningful;
`ms/pos` remains the bottom-line measure. (Historical node figures in git
predate the counting fix and aren't comparable.)

```
G="training_data/playok_pgn_7500?000.pgn"   # 10 files
for spec in "14 2000" "16 1000" "18 250" "20 50" "22 8"; do
  set -- $spec
  cargo run --release -q -- bench --empties $1 --max-boards $2 $G
done
```

The whole suite runs in ~90s on the dev box. Board counts are sized so each
level costs ~20–25s; the deep levels (20/22) are small samples, so treat their
`ms/pos` as approximate. When probing heavier counts, wrap a run in `timeout
125` (see best-practices.md) — `bench` only prints at the end.

## Current baseline (after Step 30, sequential)

| empties | boards | nodes/pos | ms/pos | nodes/s |
|---------|--------|-----------|--------|---------|
| 14 | 2000 | 79,106 | 2.0ms | 39.4M |
| 16 | 1000 | 393,323 | 10.5ms | 37.3M |
| 18 | 250 | 1,919,923 | 51.7ms | 37.1M |
| 20 | 50 | 9,714,119 | 271.9ms | 35.7M |
| 22 | 8 | 46,856,150 | 1382.0ms | 33.9M |

Parallel scaling (Step 21/29, 16 threads, AMD 32-core): 20e 311→141 ms, 22e
1601→519 ms. Node counts are non-deterministic under threads (scores stay
exact, guarded by a `ParallelSolver`-vs-`Solver` equality test).

The big sequential wins, in order of impact: richer move ordering (Step 6b,
~1.6×→~2× with depth), the per-square flip table (Step 15, ~1.37×), the
stability cutoff (Step 17, ~1.55× at 20e), the transposition table (Step 10),
the carry-64 flip (Step 11, ~1.13×). Full history is in git.

## Committed steps (branch `speed-up`)

One line each; see the code and `git log` for detail.

- **1** fastest-first move ordering (child stored in the move list, no double `do_move`).
- **2** bitboard flip computation (replaces loop-based `flipped`).
- **4** pass `empties` as a parameter; fast path for `empties == 0`.
- **5b** `solve_1`/`solve_2` leaf solvers — avoid `Vec`/`get_moves`/`Position` at the innermost levels.
- **6** Principal Variation Search in the ordered search.
- **6b** richer move ordering (`order_score`): Edax `movelist_evaluate_fast` weights — `(36 − opp_mobility)·2¹⁵` (dominant) + `corner_stability·2¹¹` + `SQUARE_VALUE` + parity bonus, sorted descending. Corner stability the decisive new signal. Biggest single win (~1.4× at 16e → ~2× at 22e).
- **7** `solve_3`/`solve_4` leaf solvers (fail-soft negamax, natural square order).
- **8** `count_last_flip` table for `solve_1` (compile-time generated).
- **10** transposition table: full-position-keyed `[lower, upper]` + best move, never cleared (exact scores are path-independent), shared across positions. `TT_BITS = 19`.
- **12** deep-search split: leaf cases (≤4 empties) factored into `solve_leaf`.
- **15** per-square flip table: `flip_mask` dispatches through 64 `flip_at::<SQ>` const-generic specializations. The biggest single win.
- **16** Enhanced Transposition Cutoff: probe each child before searching; cut on a stored upper bound that fails high. Cut condition `-upper >= beta` (valid for general windows).
- **17** stability cutoff: opponent stable-disc count `s` bounds our score at `64 − 2s`; cut when `<= alpha`. Full Edax `get_stability`. Biggest win since the flip table.
- **18** dedicated null-window path (`*_nws` taking `alpha` only): lets the compiler fold `beta = alpha + 1`, dropping dead TT-narrowing/PVS-research branches. Identical nodes; ~1.02× at 20e growing with depth.
- **11** flip-computation variants (`src/othello/flip/`): `specialized`, `generic`, `carry64`, `bmi2` (PEXT/PDEP), `avx2` — five behind a fuzz battery, `bench-flip` micro-benchmarks each. **`carry64` is the production default** (fastest on AMD, inlines; SIMD variants kept but not selected — see dead-ends). Cheaper flip then shifted two floors: `TT_MIN_EMPTIES` 7→6, `ETC_MIN_EMPTIES` 8→7 (~7% fewer nodes, flat wall-clock).
- **19** incremental region parity for move ordering. `Search.parity` (Edax `QUADRANT_ID` XOR over empties) toggled per ordered ply (make/undo), ~free; feeds a small odd-parity bonus (minor term under mobility/corner stability). Parity as a *primary* key is a disaster (nodes ~double).
- **21** full recursive YBWC parallel search (`alphabeta/parallel.rs`). Null-window nodes split: eldest child first, then younger siblings fanned across workers sharing the lock-free TT; first to fail high trips a parent-linked `CancelNode`. PV spine stays single-threaded. Threads from nested `std::thread::scope` bounded by a shared budget (`ParCtx`) — no pool, no new dep. Opt-in via `bench --threads`. `SPLIT_MIN_EMPTIES = 14`.
- **23** CPU-specific `count_last_flip` harness (`alphabeta/count_flip/`): `table`, `via_flip`, `bmi2` behind a fuzz battery + `bench-count-flip`. **`table` stays default** (BMI2 microcoded/slow on AMD — see dead-ends).
- **24** CPU-specific `get_moves` harness (`src/othello/get_moves/`): `scalar`, `avx2` behind a correctness battery + `bench-get-moves`. **`scalar` stays default** (see dead-ends).
- **25** stack move buffer (`search::MoveBuf`): per-node heap `Vec` → `MaybeUninit<[(i32,u32,Position); 34]>` + length (Edax's uninit stack `Move move[34]`). One localized `#[allow(unsafe_code)]`. Identical nodes; sequentially neutral but ~2–5% faster in parallel (relieves cross-thread allocator contention).
- **29** lock-free transposition table (`tt::SharedTt`): replaces the Step-21 sharded `Mutex` with Hyatt XOR-validated slots ("lockless hashing") — three `AtomicU64` words tied by XOR, value-based correctness so plain `Relaxed` suffices. Best-effort merge store (a lost race drops a refinement, never correctness). Sequential `Owned` path untouched. Parallel 16t: 20e 179→141 ms (~1.27×), 22e 676→519 ms (~1.30×).
- **30** shallow tier (`Search::shallow`/`shallow_nws`), empties `5 ..= SHALLOW_MAX_EMPTIES = 7`. Edax `search_shallow`: drop the mobility sort, parity-only ordering via a precomputed `PARITY_MASK`, **no move list, no sort, no TT**, only the cheap stability cutoff. Subsumed the old `alphabeta_nosort*` tier + `SORT_MIN_EMPTIES` (both removed; ordered search now starts at empties 8). **+30–40% nodes but neutral-to-faster wall-clock, the win growing with depth and larger in parallel** (`nodes/s` ~24–28M → ~34–39M). The first win in this design space — the pieces each lost alone (Steps 19/22/27); the *combination*, confined to a thin band, pays.

(Steps 3, 13, 14, 20, 22 were subsumed or reverted — see git / dead-ends.)

## Dead ends & decisions (not in code)

- **Step 9 — parity move ordering (reverted).** Odd-parity-first cut ~2% nodes but ran ~2% slower — runtime `quadrant_bit` sort in the hot `solve_3`/`solve_4` costs more than it saves. (Revisited as *secondary* key in Step 19, where it pays.)
- **Step 20 — TT-free `alphabeta_exact` variant (reverted).** A `const USE_TT` split to compile out the below-floor TT code gave identical nodes but **neutral** wall-clock — the skipped branch was already well-predicted, so the second monomorphization only adds bloat. Lesson: a `const bool` split only pays when it removes *executed* work, not a predicted branch.
- **Steps 11/23/24 SIMD variants — kept in-tree, not selected.** On the AMD dev box `bmi2` flip was ~20× *slower* than `carry64` (microcoded PEXT/PDEP), `bmi2` count-flip ~21× slower than `table`, and `avx2` lost to both `carry64` flip and `scalar` get_moves. The trap: `cfg(target_feature = "bmi2")` is true on AMD too, so a cfg auto-select would pick the slow path — no compile-time signal for "fast PEXT". All stay compiled for the micro-benches and future Intel tuning (Step 28). **Re-run on an Intel (Haswell+) box before wiring any per-target override.**
- **Step 22 static presorted order — built, measured, reverted.** An incremental empties `SquareList` (safe index-arena ring, O(1) remove/restore, static `SQUARE_VALUE` order) visited **4.6× nodes at 16e, 7.0× at 18e** for 3–4.6× slower wall-clock, even though cheaper per node. Ordering quality dominates ordering cost. The `SquareList` is not worth it as a move-gen or ordering lever; the safe-arena design is the template if a parallel split later wants O(1) incremental empties.
- **Step 27 — cheaper move-ordering mobility (reverted).** Swapped exact mobility for Edax *potential mobility* (cheap bitboard dilation): visited 1.49×/1.63×/1.78× nodes at 14/16/18e (penalty *grows* with depth) for 30–55% slower wall-clock. Same lesson as Step 22 — the exact mobility key is load-bearing; cheapening the probe is a dead end unless a proxy preserves order (none of static `SQUARE_VALUE`/parity/potential-mobility do).
- **Step 30 empties-list enumeration — built, A/B'd, reverted.** Edax's `EmptyList` walk (flip per empty, `do_move_with_flip`, no `get_moves`), built in the *same* parity order so node counts are identical, ran **~3–4% slower**: it computes a flip for every empty incl. illegal ones, whereas one branchless `get_moves` fill finds all legal moves at once. `get_moves` + parity mask is the shallow tier's final form. (Copy-make vs make/unmake confirmed a non-lever — a `Position` is 16 bytes.)
- **Step 21 root-only YBWC — built, regressed, superseded.** Splitting only the root's siblings got *worse* with threads (+55% nodes at t=8) — with strong ordering the eldest child dominates and is searched sequentially (Amdahl), so almost nothing is left to parallelize at the root. Real scaling needs splitting *inside* the eldest subtree (interior null-window nodes) = the committed full recursive YBWC.

## Edax comparison — where the remaining gap is (evidence)

Benchmarked our sequential solver against Edax to find whether more speed is
available and where. Method: 6 boards of exactly 24 empties, solved
single-threaded by both; exact scores agree on all 6. Edax: `lEdax-x64-modern
-solve <obf> -l 60 -n 1`.

Headline: **Edax is ~8× faster wall-clock** (9.1s vs 72.2s) for the identical
result — almost entirely **node count**, not per-node speed:

| solver / mode | nodes | wall-clock | nodes/s |
|---|---|---|---|
| Ours (cold exact PVS, full window) | 2,463M | 72.2s | 34.1M |
| Edax modern, default (ID + MPC) | 344M | 9.10s | 37.8M |
| Edax modern, `-selectivity 100` (ID, no MPC) | 353M | 9.44s | 37.4M |

- **Per-node speed is a non-issue.** Our 34.1M nodes/s ≈ Edax popcnt (34.2M), ~10% behind the AVX2 modern build (37.8M) — exactly the SIMD-primitive gap (Steps 26/28), the *wrong* target.
- **MPC / selectivity is NOT the lever.** Pure exact (`-selectivity 100`) changed Edax's nodes <3% (344M → 353M).
- **The ~7× is node count from iterative deepening + move ordering** — Edax's shallow eval-guided searches seed the TT with near-perfect moves/bounds before the deep pass. We do a single cold full-window PVS.

Decomposed by prototyping each half against our own solver (same 6 boards, both A/B'd then reverted):

| configuration | nodes | vs our baseline |
|---|---|---|
| Ours, baseline | 2,463M | 1.0× |
| + move-seeding (cheap-eval ID prototype) | 2,461M | 1.0× (no effect) |
| + perfect tight window `[S-1,S+1]` | 1,264M | 1.95× |
| Edax | 344M | 7.2× |

- **Mobility-heuristic move seeding: ~0.** A depth-limited lookahead with our crude eval picks the same moves `order_score` already does — no new information. Improving ordering needs a *positional/pattern* eval, not another mobility signal.
- **Window narrowing: ~2× (perfect-estimate ceiling).** Via MTD-f/aspiration; the smaller lever, and it does *not* need a strong eval — but guess-free it's net-neutral (Step 31).
- **The remaining ~3.7× is move-ordering quality**, reachable only with a real evaluation function. This project trains one (`src/training/`) but it is not yet strong enough.

**Direction (supersedes SIMD Steps 26/28 as the priority):** the eval is the
critical path. **Step 32** make training fast/correct enough to *produce* a
strong eval; **Step 33** an alloc-free flat pattern eval in the solver; **Step
34** wire it into move ordering and eval-seeded MTD-f. Both node-count levers
(~2× window, ~3.7× ordering) are eval-gated. Steps 33/34 are modelled on Edax's
`eval.c`/`move.c`/`midgame.c`.

## Remaining steps

Steps 1–25, 27, 29, 30 are implemented or resolved. **Steps 32–34 are the
current priority phase** (eval-gated node-count levers); Step 31 (MTD-f) is
built but shelved; Steps 26/28 (SIMD) target only the ~10% per-node gap and are
low priority.

### Step 31 — MTD-f / aspiration root — built, net-neutral, shelved
[`Search::solve_mtd`]: repeated exact null-window probes bisecting `[SCORE_MIN,
SCORE_MAX]` to the exact value (~6 probes), reusing the never-cleared TT. Result
stays exact. A/B on 6×24e: **net-neutral, −0.1% nodes** — the ~2× ceiling
assumed a *perfect* estimate; guess-free bisection pays for the journey. MTD-f
only wins with a **good first guess**, which needs an eval, so this lever is
**eval-gated like move ordering**. Kept behind `USE_MTD = false` as the scaffold
for eval-seeded MTD-f; revisit (sequential first, parallel root later) once the
weights are usable.

### Step 32 — training speedup (gating: produce a strong eval) — core done
Training (`src/training/`) gates Steps 33/34. Original problems: per-32-example
batch did `weights.clone()` (107 MB table) + spawn + full-table merge *even
single-threaded*; feature indices recomputed every epoch with a `Vec` alloc.

**Done (model unchanged, verified by bit-identical per-epoch loss):**
`CompiledExample` extracts the 47 feature indices **once** up front;
single-threaded path runs online SGD **in place** (no per-batch clone/merge).
Measured: **356 → ~960K ex/s (~2700×)**; at scale ~4M ex/s/epoch. The speedup
grows with dataset size (old clone/merge cost was O(weights)·n_batches).

**Still open but now low-value:** flatten weights to one `Vec<f32>` (do it as
part of Step 33 — the solver eval needs the flat layout anyway) and a lock-free
parallel path. At ~4M ex/s single-threaded the parallel path is
counter-productive (the clone/merge dwarfs per-example compute; epoch-level
model-averaging converges worse), so `threads = 1` is the right default.

**Superseded:** the SGD trainer described here has since been **removed** in favour
of a per-bucket conjugate-gradient least-squares solver (`src/training/cg.rs`), which
reaches the same accuracy floor at the exact optimum, is faster, and parallelizes
cleanly across buckets with `-t`. See [eval-quality.md](eval-quality.md), "The
capacity ceiling".

### Step 33 — alloc-free flat pattern eval in the solver — core done
`Weights::evaluate` allocates per call and triple-chases `Vec<Vec<Vec<f32>>>` —
unusable in the hot path. Edax's eval (`eval.c`/`eval.h`/`midgame.c`) is the
alloc-free template.

**Done:** [`FlatEval`] (`src/eval/pattern.rs`) copies a trained `Weights` table
**once** into a contiguous range-major `Vec<f32>` (`weights[r * range_stride +
offset[f] + pattern]`). `set(&Position, &mut [u16])` fills the 47 pattern indices
alloc-free; `score(&[u16], empties)` is the straight-line dot product;
`eval_position` does both into a `[u16; 64]` stack scratch. Scores are
**bit-identical** to `Weights::evaluate` (unit-tested via `to_bits()` equality).
Micro-bench (1M positions): `Weights::evaluate` 1.3 M/s → `eval_position` 1.4
M/s (1.12×, dominated by the per-cell index build in `set`) → bare `score` with
precomputed indices **4.0 M/s (3.17×)**. The 3.17× is the real prize, unlocked
**only** by maintaining indices incrementally (Step 34).

**Deferred to Step 34** (used/validated there): `i16` quantization (kept f32 for
exact match now); the **incremental make/unmake update** (`eval_update`,
`eval.c:782`) + its coordinate→feature scatter table (`EVAL_X2F`) — at shallow
ordering/MTD nodes a from-scratch `set` is affordable, so incremental is a later
optimization, not a blocker.

### Step 34 — eval-guided move ordering (~3.7× lever) + eval-seeded MTD-f
Replicate Edax's **two-tier, depth-gated** midgame ordering (`move.c`). Our
`order_score` (6b) + shallow parity tier (30) already match **tier 1**
(`movelist_evaluate_fast`). Missing **tier 2** (`movelist_evaluate`, `move.c:368`):

- **Depth-gated dispatch.** Per node `sort_depth = (depth − 15)/3` (clamped 0..6), gated on `min_depth_table[empties]`. Below the gate, fall back to tier 1.
- **The shallow-search bonus is the new signal** (`move.c:440`): make the move (incremental `eval_update`), add `(SCORE_MAX − shallow_score) · w_eval` where `shallow_score` comes from `search_eval_0/1/2` / `PVS_shallow` by `sort_depth`; restore. `w_eval = 1<<15`, co-dominant with mobility. A *positional/pattern* score — new information (why the mobility-only seeding prototype was ~0).
- **Two structural prerequisites we lack.** (a) **Iterative deepening** — Edax's shallow passes seed the TT with near-perfect best move + bounds, read as the top-2 ordering keys; we do one cold full-window PVS, so no hash moves to promote. (b) **`inc_sort_depth[node_type]`** — PV nodes get a deeper `sort_depth` than cut nodes. Both are part of this step.
- **Same machinery feeds eval-seeded MTD-f (Step 31).** The eval estimate is the MTD-f first guess.

Sequence: (1) iterative-deepening driver + TT hash-move ordering; (2)
`sort_depth`/`min_depth_table` schedule calling the Step-33 eval; (3)
`inc_sort_depth`; (4) feed the estimate to MTD-f. A/B each against 6×24e and the
standard `bench` levels — target is node count, not nodes/s.

**Done (the simplest cut — static-eval ordering term — measured net-negative,
confirming the eval-quality gate).** `Search` carries an optional
`Arc<FlatEval>` (`Solver::with_eval`, wired through `bench --weights` as a
node-count A/B; absent = mobility-only baseline, so the default path is unchanged
by construction). `ordered_moves` adds the `sort_depth = 0` term `(SCORE_MAX −
clamp(eval(child))) · W_EVAL` (`W_EVAL = 1<<13`, `move.c:442`) at `empties >=
EVAL_ORDER_MIN_EMPTIES = 12`. A/B with the existing `trained_weights.bin`:
**+0.7% nodes (14e), +1.0% (16e), +1.8% (18e)** — slightly *worse*, penalty
growing with depth. This is the signature of a *weak* ordering signal displacing
the good mobility order (same shape as reverted Steps 22/27), **not** an inverted
sign (which would ~double nodes). Mechanism correct, gate confirmed: **the eval
is not yet strong enough.**

**UPDATE — the gate flipped (the eval effort paid off).** After fixing the eval
(see [eval-quality.md](eval-quality.md): corrected 46-feature transcription,
symmetry **weight tying**, mini-batch + L2, exact labels extended to ≤18e), the
*same* `sort_depth = 0` term now **cuts ~34 % of nodes** and is **~1.27× faster
wall-clock** at 18e: 1,964,515 → 1,295,312 nodes/pos, 54.5 → 42.9 ms/pos
(`weights_v4.bin`, 211 boards). `nodes/s` drops 36→30 M (the ~17 % per-node eval
cost) but the node cut dominates. So the static-eval ordering term is now a **win**,
not a regression — eval-guided ordering (the ~3.7× lever) is delivering.

**Confirmed across depths (fresh ≤18e CG retrain, 10 `7500?000` files).** The win
**grows monotonically with depth** — the healthy signature (a weak signal would
regress *worse* with depth, like Steps 22/27). Baseline = mobility-only; eval =
same weights via `bench --weights`:

| empties | boards | baseline nodes/pos | eval nodes/pos | node cut | wall-clock |
|---|---|---|---|---|---|
| 14 | 2000 | 79,106 | 71,560 | −9.5% | ~1.04× |
| 18 | 250 | 1,919,923 | 1,489,299 | −22.4% | ~1.11× |
| 20 | 50 | 9,714,119 | 7,087,778 | −27.0% | ~1.12× |

`nodes/s` drops ~14–16% (the per-node eval cost) but the node cut dominates at
every depth, and even 14e — previously where the eval was too weak — is now a net
win. (This retrain is ≤18e exact base; slightly below the doc's earlier
`weights_v4` ~34%/1.27× at 18e, attributable to corpus/sample differences, not a
mechanism change.)

**`W_EVAL` swept (18e, 250 boards, ≤18e weights) — `1<<13` confirmed optimal.**
A sharp minimum: `1<<11` 1,727K, `1<<12` 1,572K, **`1<<13` 1,489K**, `12288`
1,553K, `1<<14` 1,564K nodes/pos. Every neighbour on both sides is worse, so
Edax's `w_eval >> 2` value transfers cleanly to our eval — no retune needed.

**`EVAL_ORDER_MIN_EMPTIES` swept 12 → 14 (a real win).** The gate trades node
cut against per-node eval cost. Lowering it cuts *more* nodes but the
from-scratch `FlatEval::set` on the many shallow nodes dominates (at 8: 1,417K
nodes but 115.7 ms — nodes/s 30→12 M); raising it past the elbow loses ordering.
18e ms/pos by threshold: 8→115.7, 10→65.2, 12→48.3, 13→45.7, **14→44.6**,
15→45.2, 16→46.6 — a clean wall-clock minimum at **14**, confirmed deeper at 20e
(225.3 ms vs 12's 250.2) and neutral at 14e. Adopting 14 lifts the eval-ordering
win to **~1.20× at 18e and ~1.24× at 20e** (was 1.11×/1.12× at 12):

| depth | baseline ms | thr=12 ms | thr=14 ms (adopted) | thr-14 vs baseline |
|---|---|---|---|---|
| 14 | 2.04 | 1.97 | 1.97 | ~1.04× |
| 18 | 53.4 | 48.3 | 44.6 | ~1.20× |
| 20 | 278.9 | 250.2 | 225.3 | ~1.24× |

**Iterative deepening — DONE, the largest ordering win so far.** Edax seeds its
exact endgame pass with a near-perfect hash move at every node by running shallow
heuristic passes first (`iterative_deepening`, `root.c`); the deep pass reads
`hash_data.move[0]` for ordering. Two structural pieces were needed (both Edax):

- **Depth-stamped TT** (prerequisite). Our never-cleared TT was *exact-only* — it
  cut on stored bounds with no depth check, valid only because every entry was
  searched to game end. To let heuristic and exact entries share one table,
  `TtEntry` gained a `depth` stamp (Edax `HashData.depth`); the bound cutoffs now
  require `depth >= empties` (exact-resolved, Edax `search_TC_NWS`) while the
  **move hint is read unconditionally** (Edax `search_guess`). `merge_payload`
  replaces the unconditional bound-intersection with a depth-preferred merge.
  Behavior-neutral on its own (every existing entry is exact, `depth == empties`):
  node counts bit-identical, all tests pass.
- **ID driver** (`Search::solve_id`/`id_pass`). When a trained eval is attached,
  run heuristic passes (`FlatEval` at the horizon, the production `ordered_moves`
  for sort) at depths `start..end step 2`, `end = empties − ITERATIVE_MIN_EMPTIES
  + 2`, each storing a best-move hint stamped with its (shallow) pass depth — so it
  can only reorder the exact pass, never trigger a wrong cutoff. Then the exact
  solve. Exact result unchanged (guarded by an ID-vs-plain equality test over the
  reference positions, parity asserts live).

A/B (10 `7500?000` files, ID seeding on vs off, eval = `trained_weights.bin`), **on
top of the eval-ordering win above**:

| depth | eval no-ID | eval+ID | node cut | no-ID ms | ID ms | speedup |
|---|---|---|---|---|---|---|
| 14 | 76,157 | 62,758 | −17.6% | 1.9 | 1.7 | ~1.12× |
| 16 | 354,318 | 282,582 | −20.2% | 9.5 | 8.1 | ~1.17× |
| 18 | 1,564,721 | 1,144,108 | −26.9% | 45.5 | 38.0 | ~1.20× |
| 20 | 7,425,308 | 5,236,744 | −29.5% | 223.3 | 205.0 | ~1.09× |

**Cumulative vs the mobility-only baseline** (eval ordering + ID): 18e 1,919,923 →
1,144,108 (**−40% nodes, ~1.41×**), 20e 9,714,119 → 5,236,744 (**−46% nodes,
~1.36×**). `ITERATIVE_MIN_EMPTIES` swept 8..14 at 18e — a flat basin, kept at
Edax's **10** (fewest nodes at every depth; 8 over-seeds at wall-clock cost, 12+
under-seed and the shallow node cut collapses; 10 is also fastest at 14e/16e where
bulk ≤16e label-solving runs). `id_pass` stores move hints only (uninformative
bounds), so the depth-stamp's bound-gating is correct-but-dormant — ready for
heuristic-bound storage if it later pays.

**Parallel: eval ordering + parallelized ID (DONE).** `ParallelSolver` carries an
optional `Arc<FlatEval>` (`with_eval`) propagated to every YBWC worker via
`Search::worker`, so the parallel search orders moves like the sequential one, and
the root runs the same `solve_root` (ID seeding + exact pass). **The ID seeding
passes are themselves parallelized** — Edax does this too (`iterative_deepening` →
`PVS_root`/`node_split` run every midgame iteration through YBWC, not just the exact
pass). `id_pass`/`id_pass_nws` mirror the exact PV/NWS split: the heuristic
null-window nodes fan younger siblings across workers through the *same*
`split_nws` (generalized with `Some(depth)`), gated on `ID_SPLIT_MIN_DEPTH`.

A first cut that left seeding **sequential** confirmed why Edax parallelizes it: it
was a serial Amdahl tax that *grew* with depth and went net-negative (16t, eval+ID
vs eval-only: 22e 554 vs 414 ms — even losing to the no-eval baseline). Parallelizing
the seeding flips it. A/B at 16 threads (`ID_SPLIT_MIN_DEPTH = 5`, swept — see the
const; 3 over-splits, 7–9 leave too much serial):

| depth | baseline | eval-only | eval+ID | ID vs eval-only | ID vs baseline |
|---|---|---|---|---|---|
| 20 (100b) | 142.3 ms / 15.4M | 121.7 ms / 12.1M | **106.3 ms / 7.58M** | ~1.14× | ~1.34× |
| 22 (16b) | 467.4 ms / 77.4M | 416.6 ms / 58.9M | **377.4 ms / 40.2M** | ~1.10× | ~1.24× |

So the full sequential stack (eval ordering + ID) now carries over to parallel:
~1.10–1.14× over eval-only and ~1.24–1.34× over the mobility baseline at 16 threads.
Correctness guarded by `parallel_with_eval_matches_sequential` (16 workers vs
sequential plain; the ID split path validated in debug at a lowered threshold so it
fires at 16e). The split machinery is shared with the exact pass, so the tested
exact split covers its synchronization.

**Still open** (now worth doing — the eval is strong enough): the shallow-search
bonus (`sort_depth` 1–6, needs incremental `eval_update` so the per-node cost is
`score` not a full `set`); `inc_sort_depth`; eval-seeded MTD-f (Step 31) — the ID
estimate is now the first guess; storing heuristic bounds in the ID passes (the
depth stamp already supports it); and re-sweeping `ID_SPLIT_MIN_DEPTH`/`SPLIT_MIN_
EMPTIES` jointly at other thread counts. Verify the win across depths and re-bench
after each.

### Step 35 — midgame heuristic search (GUI / self-play): ordering + TT + incremental eval

The *heuristic* depth-limited search (`depth_limited_score`, `alphabeta/depth.rs`)
— used by `gui evaluate`/`pgn` and self-play move selection, distinct from the exact
endgame search — was plain full-width negamax with **no move ordering, no TT, and a
from-scratch leaf eval through the allocating `Weights::evaluate`**. So the GUI felt
slow in the midgame (default depth 6).

Diagnosis (30-empty midgame, trained weights, dev box):

| | |
|---|---|
| depth 6 search | ~0.22 s (and `score_moves` runs one per root move) |
| depth 7 search | ~3 s |
| `Weights::evaluate` (GUI leaf, allocates a `Vec`) | 1.7 M/s |
| `FlatEval::eval_position` (alloc-free) | 2.3 M/s |
| `features.extract` (from-scratch, the bottleneck) | 2.3 M/s |
| `Position::canonical` (the symmetry-fix normalize) | 54.7 M/s (≈3% — not the cost) |

Two independent levers, mirroring the exact-search story: **node count** (ordering +
TT) dominates, **per-eval cost** (alloc + from-scratch features) is second.

**Fix 1 — DONE.** Route the heuristic search through the *same* eval-seeded ordered
PVS the exact solver already uses for ID seeding (`Search::id_pass`): `order_score`
move ordering (mobility + corner + the eval term ≥ `EVAL_ORDER_MIN_EMPTIES`), TT
best-move hints (iterative-deepening passes), `FlatEval` at the horizon (alloc-free,
canonicalized so scores stay symmetry-invariant), and the exact handoff when
`empties ≤ depth`. Exposed as `Solver::heuristic_score`; the GUI builds one
`Solver::with_eval` and reuses its TT across all root moves (still solving root
children exactly below `exact_empties`). Equal to the old negamax value in the clean
midgame (PVS only reorders); near the end it differs slightly — a pass counts as a
ply, the horizon clamps to `[-63,63]`, and `empties ≤ depth` hands off to the *exact*
score (strictly more accurate). Measured ~1.4× at depth 6, ~4.2× at depth 8 (the
ordering win grows with depth). No retrain.

**Fix 2 — DONE (incremental eval, Edax `eval_update`).** [`IncEval`] keeps the 46
feature indices and updates only the moved + flipped squares per move (`FlatEval::
inc_child`, Edax's `EVAL_X2F` scatter), O(flips) instead of a from-scratch
`FlatEval::set`. `Search::heuristic_search` builds the state once at the root
(`inc_root`) and threads it down `id_pass_inc`/`id_pass_nws_inc` (the exact solver's
`id_pass` is untouched).

The tension is resolved without a retrain by combining two pieces:
- **Fixed-perspective weights.** Edax's cheap update needs indices in a *fixed* global
  encoding (digit 1 = the player to move at *even* empties), but our weights are
  trained side-to-move per empties-bucket. So `FlatEval::weights_fixed` color-swaps the
  **odd**-empties buckets once at construction (a per-shape digit permutation); the
  fixed-encoding index then dot-products to the exact side-to-move score (proven
  bit-exact by `inc_root_matches_raw` / `inc_chain_matches_from_scratch`). A pass keeps
  empties parity but flips the mover, so the state is rebuilt (`inc_root`) on a pass.
- **Root canonicalization.** Symmetry is kept not per-leaf but by canonicalizing the
  search root (`Solver::heuristic_score`): symmetric roots share a canonical form →
  identical tree → identical score. Leaves are evaluated raw (orientation relative to
  the canonical root), which is what lets the incremental state flow.

Value semantics therefore shift from fix 1 (per-leaf canonical → root-canonical raw
leaves) — still symmetry-invariant, validated end-to-end against an independent
raw-leaf negamax (`heuristic_score_matches_raw_negamax`). Measured vs fix 1 on a
30-empty midgame: depth 6 ~0.041 → ~0.006 s (~7×), depth 8 ~0.29 → ~0.08 s (~3.5×);
depth 10 ~0.58 s. The from-scratch `FlatEval::set`/`eval_position` (per-leaf
canonical) stays in place for the exact-solver ordering, bootstrap, and eval-check.

Not yet done: incremental eval in the **exact** solver's ordering, and the Step-34
shallow-search ordering bonus it unblocks.

### Steps 26 / 28 — SIMD primitives (low priority, Intel-gated)
From a callgrind profile of the sequential hot path. Self-cost ranking: **flip
~31%** (already maximal — Steps 11/15), **`get_moves` ~17%** (per-node move-gen +
per-child `order_score` mobility), `alphabeta_exact_nws` ~12%,
`solve_1`/`count_last_flip` ~6%, `tt_probe`/`store` ~3%, `malloc`/`free` ~2.7%
(addressed by Step 25), `get_stability` ~1%. These are *instruction counts* —
cache misses, branch mispredicts, and parallel TT contention are not captured;
`perf` is blocked on this box by `perf_event_paranoid = 4`.

- **Step 26** — A/B the `avx2` `get_moves` *inside the search* (not the call-overhead-bound micro-bench of Step 24). At 17% of real work a 10%-faster primitive is ~1.7% overall. `cfg`-gated, Intel-favorable; may still lose on AMD — keep only if it wins.
- **Step 28** — Re-run `bench-flip` / `bench-get-moves` and the search on a Haswell+ box and wire a `cfg`/runtime override only if measured. The single largest latent lever, but gated on hardware we don't have. The Edax comparison shows both target only the ~10% per-node gap, not the ~7× node-count gap — hence low priority behind the eval work.

Minor / opportunistic, not numbered: split PV nodes too (the spine is O(depth)
nodes — small); NUMA-aware shard placement; re-sweep `SPLIT_MIN_EMPTIES` and the
shard count on other hardware.
