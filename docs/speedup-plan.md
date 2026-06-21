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

**Still open** (now worth doing — the eval is strong enough): re-tune `W_EVAL` /
`EVAL_ORDER_MIN_EMPTIES` for the stronger eval; the shallow-search bonus
(`sort_depth` 1–6, needs incremental `eval_update` so the per-node cost is `score`
not a full `set`); iterative deepening + hash-move ordering; `inc_sort_depth`;
eval-seeded MTD-f (Step 31); and wiring the eval into the parallel workers (still
`None`). Verify the win across depths (14/16/20e) and re-bench after each.

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
