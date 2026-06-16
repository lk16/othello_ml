# Alphabeta Exact-Search Speed-Up Plan

Roadmap and design log for the exact endgame solver (`src/eval/alphabeta/`).
Mechanism details live in the code and its doc comments — this file keeps the
*why*: the benchmark recipe, the current baseline, what each step changed, dead
ends, and what's left. Per-constant tuning sweeps live in the constant doc
comments in `alphabeta/`, not here.

## Benchmark

Pulls positions with exactly `--empties` discs free from a fixed set of PGN
files and solves each. `bench` prints to stderr. Node counting is now consistent
— one node per visited position across all solvers — so `nodes/s` is meaningful
(~27M); `ms/pos` remains the bottom-line measure. Historical baseline tables in
git predate the counting fix and undercounted leaf nodes, so their node figures
aren't comparable to these.

```
G="training_data/playok_pgn_7500?000.pgn"   # 10 files
for spec in "14 2000" "16 1000" "18 250" "20 50" "22 8"; do
  set -- $spec
  cargo run --release -q -- bench --empties $1 --max-boards $2 $G
done
```

The whole suite runs in ~90s (under a 2-minute budget) on the dev box. Board
counts are sized so each level costs ~20–25s; the deep levels (20/22) are
necessarily small samples, so treat their `ms/pos` as approximate. When probing
heavier counts, wrap a run in `timeout 125` (see best-practices.md) — `bench`
only prints at the end, so an over-long run is killed with no output.

## Current baseline (after Step 30, sequential)

| empties | boards | nodes/pos | ms/pos | nodes/s |
|---------|--------|-----------|--------|---------|
| 14 | 2000 | 79,106 | 2.0ms | 39.4M |
| 16 | 1000 | 393,323 | 10.5ms | 37.3M |
| 18 | 250 | 1,919,923 | 51.7ms | 37.1M |
| 20 | 50 | 9,714,119 | 271.9ms | 35.7M |
| 22 | 8 | 46,856,150 | 1382.0ms | 33.9M |

Node counts jumped (+30–40%) at Step 30: the shallow tier (empties 5–7) trades a
weaker parity-only move order for a much cheaper per-node cost, so `nodes/s` rose
from ~24–28M to ~34–39M and `ms/pos` is neutral-to-faster (see Step 30). The
pre-Step-30 baseline (56,583 / 281,927 / 1,379,799 / 7,238,915 / 35,303,820
nodes/pos) is in git.

The big wins, in order of impact: the richer move ordering (Step 6b, ~1.6× at
20e growing to ~2× at 22e), the per-square flip table (Step 15, ~1.37×), the
stability cutoff (Step 17, ~1.55× at 20e), the transposition table (Step 10),
and the carry-64 flip (Step 11, ~1.13×). Full history is in git.

## Committed steps (branch `speed-up`)

One line each; see the code and `git log` for detail.

- **1** fastest-first move ordering (child stored in the move list, no double `do_move`).
- **2** bitboard flip computation (replaces loop-based `flipped`).
- **4** pass `empties` as a parameter; fast path for `empties == 0`.
- **5b** `solve_1`/`solve_2` leaf solvers — avoid `Vec`/`get_moves`/`Position`
  at the innermost levels. (Step 3, "special-case last move", is subsumed.)
- **6** Principal Variation Search in the ordered search.
- **7** `solve_3`/`solve_4` leaf solvers (fail-soft negamax, natural square order).
- **8** `count_last_flip` table for `solve_1` (compile-time generated).
- **12** deep-search split: leaf cases (≤4 empties) factored into `solve_leaf`,
  so the `≥5` path never re-tests them.
- **13** skip move ordering below `SORT_MIN_EMPTIES = 6` (subsumed by Step 30: the
  whole 5–7 band is now the shallow tier, and `SORT_MIN_EMPTIES` was removed).
- **14** dedicated `alphabeta_nosort` for the unordered range, no `Vec` (subsumed
  by Step 30; the `alphabeta_nosort*` functions were removed).
- **10** transposition table: full-position-keyed `[lower, upper]` bounds + best
  move, never cleared (exact scores are path-independent), reused across positions
  via a shared `Solver`. `TT_BITS = 19`, `TT_MIN_EMPTIES = 7`.
- **15** per-square flip table: `flip_mask` dispatches through 64 `flip_at::<SQ>`
  const-generic specializations; the compiler folds the move bit and prunes
  off-board rays. The biggest single win.
- **16** Enhanced Transposition Cutoff: probe each child before searching; cut on
  a stored upper bound that fails high. `ETC_MIN_EMPTIES = 8` (= `TT_MIN_EMPTIES
  + 1`, the floor below which children aren't stored). Cut condition is
  `-upper >= beta`, valid for our general windows, not just null windows.
- **17** stability cutoff: opponent stable-disc count `s` bounds our score at
  `64 - 2s`; cut when that `<= alpha`. Full Edax `get_stability` (exact edge
  table + full lines + central spread); `STABILITY_THRESHOLD` kept from Edax
  verbatim. Biggest win since the flip table.
- **18** dedicated null-window path (`*_nws` functions taking `alpha` only): not
  passing `beta` lets the compiler fold it to `alpha + 1`, dropping the dead
  TT-narrowing branches and the PVS re-search. Node counts identical to 17;
  ~1.02× at 20e, growing with depth.
- **11** flip-computation variants (`src/othello/flip/`). Five share one
  signature behind a fuzz battery (every square × patterns + 500k random boards,
  all checked against the proven reference): `specialized` (the Step-15 per-square
  function table), `generic` (inlinable runtime ray-scan), `carry64` (portable
  line gather → `OUTFLANK`/`FLIPPED` lookup → scatter), `bmi2` (PEXT/PDEP) and
  `avx2` (4-lane Kogge-Stone fill), both x86-64-only with localized
  `allow(unsafe_code)`. `bench-flip` micro-benchmarks each. **`carry64` is the
  production default** — fastest on the AMD dev box (~6 ns/flip micro, ~1.13×
  whole-search) and it inlines, unlike the indirect call through the `specialized`
  table. See the decision note below for why the SIMD variants aren't selected.
  The cheaper flip then shifted two coupled floors (re-swept; see the constant
  doc comments): `TT_MIN_EMPTIES` 7 → 6 and `ETC_MIN_EMPTIES` 8 → 7, for ~7%
  fewer nodes at flat wall-clock. `SORT_MIN_EMPTIES` re-swept but stays 6.
- **6b** richer move ordering (`order_score`). The ordered search now scores each
  move with Edax's `movelist_evaluate_fast` weights instead of bare opponent
  mobility: `(36 − opp_mobility)·2¹⁵` (dominant) + `corner_stability·2¹¹` +
  `SQUARE_VALUE[cell]` + parity bonus, sorted descending. Corner stability is the
  decisive new signal (Edax `get_corner_stability`: held corners + corner-anchored
  edge discs). The biggest single win: ~1.4× at 16e growing to ~2× at 22e (e.g.
  20e 12.23M → 7.24M nodes, 460 → 291 ms). `SORT_MIN_EMPTIES` re-swept afterward —
  the costlier ordering still crosses over at 6 (5 cuts more nodes but the per-move
  score doesn't pay that shallow; 7+ explode).
- **19** incremental region parity for move ordering. `Search.parity` (Edax
  `QUADRANT_ID` XOR over empties) is seeded at the root and toggled
  `^= QUADRANT_ID[move]` per ordered ply (make/undo), so it is ~free. Feeds a
  small odd-parity bonus into the move-ordering score (a minor term under mobility
  and corner stability — see Step 6b). On its own, before 6b, it cut ~4–5% of
  nodes at wall-clock-neutral. `n_empties` is already a search parameter (Step 4);
  the empties `SquareList` itself was not added — with `get_moves`-based move
  generation it has no use, and enumeration was never the bottleneck. Parity as a
  *primary* ordering key was tried and is a disaster (it overrides mobility —
  nodes ~doubled).
- **23** CPU-specific `count_last_flip` harness (`src/eval/alphabeta/count_flip/`).
  Three variants share one signature (`fn(pos, player) -> i32`) behind a fuzz
  battery (every square × patterns + 500k random near-full boards, all checked
  against the `table` reference) with a `bench-count-flip` micro-benchmark,
  mirroring the Step 11 flip harness: `table` (the prior per-line `COUNT_FLIP`
  lookup, gathered by shift/multiply — kindergarten), `via_flip` (full flip mask
  via the production flip, then `2×popcount`) and `bmi2` (x86-64 `PEXT` line
  gather, compiled only on x86-64). **`table` stays the production default** —
  fastest on the AMD dev box (2.4 vs `via_flip` 8.7 vs `bmi2` 51 ns/flip micro);
  BMI2 PEXT is microcoded and slow on AMD, same as the Step 11 flip story, so it
  is `cfg`-gated and not auto-selected. See the decision note below.
- **24** CPU-specific `get_moves` harness (`src/othello/get_moves/`). Two
  variants share one signature (`fn(player, opponent) -> u64`) behind a
  correctness battery (structured pattern pairs + 1M random boards, all checked
  against the `scalar` reference) with a `bench-get-moves` micro-benchmark,
  mirroring Step 11: `scalar` (the prior branchless 8-direction Kogge-Stone fill)
  and `avx2` (the four ray axes in parallel across 256-bit lanes, lane shifts
  {1,8,7,9}; Edax `get_moves_avx`, compiled only on x86-64). **`scalar` stays the
  production default** — slightly faster on the AMD dev box (7.24 vs `avx2`
  7.81 ns/call micro) and portable. `Position::get_moves` now delegates to the
  selected variant; search wall-clock unchanged (18e 54.2 vs 54.0 ms baseline).
  See the decision note below.
- **21** full recursive YBWC parallel search (`src/eval/alphabeta/parallel.rs` +
  the split path in `search::alphabeta_exact_nws`). Null-window nodes — the bulk
  of the tree — split: the eldest child is searched first (it usually cuts: young
  brothers wait), then on a fail-low the younger siblings are fanned across worker
  threads, each with its own `Search` but sharing the sharded-mutex transposition
  table; the first to fail high trips a parent-linked `CancelNode`, so the rest
  abort without storing. The PV spine (`alphabeta_exact`) stays single-threaded —
  workers only traverse null-window subtrees — so the dominant eldest subtree
  parallelizes through *its own* null-window descendants (the fix for the failed
  root-only attempt, see the decision note). Threads come from nested
  `std::thread::scope` bounded by a shared budget (`ParCtx`): a split spawns
  helpers only while live threads < cap, else runs the siblings inline — no pool,
  no unbounded spawn, no new dependency. Opt-in via `bench --threads`; the
  sequential `Solver`/owned-table path is untouched (identical node counts, one
  predicted enum branch). Scaling on the AMD 32-core box (ms/pos): 20e 311→179
  (1.74× @16t), 22e 1601→676 (2.37× @16t) — better at depth as larger subtrees
  amortize the speculative-node overhead (+45–83% nodes). `SPLIT_MIN_EMPTIES = 14`
  (swept). Node counts are non-deterministic under threads; scores stay exact
  (a `ParallelSolver`-vs-`Solver` score-equality test guards it).
- **25** stack move buffer (`search::MoveBuf`). The ordered search built its
  `move_list` with a per-node heap `Vec` — ~2.7% of *instructions* in
  `malloc`/`free` (callgrind). Replaced by a `MaybeUninit<[(i32,u32,Position);
  34]>` + length, written/read only over `[..len]` — the model is Edax's
  uninitialized stack `Move move[34]` (`MAX_MOVE = 33`). One localized
  `#[allow(unsafe_code)]` for the init-prefix→`&[T]` cast (write-before-read
  invariant; layout-compatible). Identical node counts. **Sequentially neutral**
  (16e ~10.5ms either way — the allocator fast path is cheap in *cycles* even
  when it is 2.7% of *instructions*; the instruction win does not convert), but
  **~2–5% faster in parallel** (20e: t=8 171 vs 179 ms, t=16 ~160 vs ~164) by
  relieving cross-thread allocator contention under YBWC. A first safe attempt
  with a *zeroed* `[T; 34]` was ~2–3% *slower* (the forced init outweighed the
  cheap malloc) — `MaybeUninit` is what removes the init, exactly as C leaves the
  stack array untouched.
- **29** lock-free transposition table (`tt::SharedTt`). The shared table was a
  sharded `Mutex<Box<[TtEntry]>>` (Step 21); the lock traffic and guard cache line
  were the contention ceiling as threads grow. Replaced by a flat array of Hyatt
  XOR-validated slots ("lockless hashing", as Crafty/Stockfish): the 128-bit key +
  packed payload exceed any single atomic, so the three `AtomicU64` words are tied
  by XOR — `w0 = player ^ data`, `w1 = opponent ^ data`, `w2 = data`. A reader
  recovers the key and accepts only on a full-key match, so a torn read (words from
  two writes) recovers a mismatched key and reads as a miss — correctness is purely
  value-based, so plain `Relaxed` loads/stores suffice (no fence, no per-slot
  lock). Store is a best-effort merge (intersect bounds / keep a real move, same
  policy as the owned table); the RMW isn't atomic but a lost race only drops a
  refinement, never correctness (every written bound is independently valid — the
  Edax trade-off). Same 24 B/slot as the old padded entry, no sharding, no new
  dependency. The sequential `Owned` `Vec` path is **untouched** (16e 281,927
  nodes/pos, ms unchanged). Parallel wall-clock at 16t: **20e 179 → 141 ms
  (~1.27×), 22e 676 → 519 ms (~1.30×)**. The 16-thread `ParallelSolver`-vs-`Solver`
  score-equality test guards the torn-read path under real contention.
- **30** shallow tier (`Search::shallow`/`shallow_nws`), empties
  `5 ..= SHALLOW_MAX_EMPTIES = 7`. Edax's `search_shallow` deliberately drops the
  mobility sort in this band: parity-only ordering (odd-quadrant moves first, via a
  precomputed `PARITY_MASK`), **no move list, no sort, no transposition table**,
  keeping only the cheap stability cutoff. This replaced the prior unordered-5 +
  mobility-ordered-6/7 split (the `alphabeta_nosort*` tier and `SORT_MIN_EMPTIES`
  were thereby subsumed and removed; the ordered search now starts at empties 8).
  A/B'd behind a temporary `const` then hardwired: **+30–40% nodes but
  neutral-to-faster wall-clock, the win growing with depth and larger in parallel**
  — 18e neutral, 22e sequential ~4.5% faster (1356 → 1295 ms), and at 16 threads
  20e ~6.7% (144 → 134) and 22e ~11.5% (463 → 410). `nodes/s` rose ~24–28M → ~34–39M:
  the per-node cost (no per-child `order_score` mobility probe, no TT, no
  list/sort) more than offsets the weaker order. This is the **first win in this
  design space** — the pieces each lost alone (parity-primary ordering Step 19;
  empties `SquareList` and static order Step 22; cheap mobility Step 27); the
  *combination*, confined to a thin band, is what pays. Enumerates moves with
  `get_moves` + the parity mask; Edax's empties-list walk was also tried and lost
  (see the dead-ends note), so this is the final design.

## Dead ends & decisions (not in code)

- **Step 9 — parity move ordering (reverted).** Ordering empties so odd-parity
  regions go first cut ~2% of nodes but ran ~2% *slower* — the runtime
  `quadrant_bit` sort in the hot `solve_3`/`solve_4` costs more than it saves.
  Edax only makes this pay via precomputed `sort3`/`parity_case` lookups plus
  *incremental* parity (one XOR per ply); replicating that for a ~2% node ceiling
  isn't worth it. (Revisited in **Step 19**: incremental parity as a *secondary*
  key in the main ordered search does pay off on node count — the runtime sort in
  the tiny leaf solvers was the original killer.)
- **Step 20 — TT-free `alphabeta_exact` variant (reverted).** Below
  `TT_MIN_EMPTIES` the empties-6 ordered nodes already run `use_tt = false` at
  runtime, with the TT probe/store/ETC/hash-move code compiled in behind a
  branch. A `const USE_TT: bool` split (dispatched by child empties) compiled that
  dead code out for the no-TT body. Node counts identical (as expected), but
  wall-clock was **neutral** (A/B: 18e 84.7→84.4ms, 20e 492.1→489.9ms, both
  within noise) — the skipped branch was already well-predicted, so removing it
  saves nothing while the second monomorphization of two large functions adds
  bloat. Same lesson as the reverted Step 18 const-generic attempt: a `const
  bool` split only pays when it removes *executed* work, not a predicted branch.
- **Step 11 SIMD variants — kept in-tree, not selected.** On the AMD dev box the
  `bmi2` flip was ~20× *slower* than `carry64` (~125 vs ~6 ns/flip): AMD's
  PEXT/PDEP are microcoded. The trap is that `cfg(target_feature = "bmi2")` is
  *true* on AMD too, so a cfg-based auto-select would pick the slow path — there
  is no compile-time signal for "fast PEXT". `avx2` (~10 ns/flip) also lost to
  `carry64` here. Both stay compiled for `bench-flip` and future Intel tuning,
  but production uses the portable `carry64` unconditionally. Re-run `bench-flip`
  on an Intel (Haswell+) box before wiring any per-target override.
- **Step 23 `count_last_flip` variants — same outcome as Step 11.** The `bmi2`
  PEXT gather was ~21× *slower* than the portable `table` lookup on the AMD dev
  box (51 vs 2.4 ns/flip), and `via_flip` (recompute the full flip mask, then
  `2×popcount`) was ~3.6× slower (8.7 ns) — paying for a whole flip mask where a
  per-line count needs only four table reads. The `table` variant (the original
  kindergarten lookup) stays the production default; `bmi2` is kept compiled for
  `bench-count-flip` and future Intel tuning behind the same `cfg` caveat. No
  whole-search change — same node counts, same primitive, just confirmed the
  existing lookup is already the fastest portable option here.
- **Step 24 `get_moves` variants — same outcome as Step 11.** The `avx2`
  four-axes-in-parallel mobility was marginally *slower* than the portable
  `scalar` 8-direction fill on the AMD dev box (7.81 vs 7.24 ns/call): the eight
  scalar shifts already pipeline well, and the `target_feature` boundary blocks
  inlining the SIMD call into the hot loop. `scalar` stays the production default
  (`Position::get_moves` now delegates through the harness; search wall-clock
  unchanged — 18e 54.2 vs 54.0 ms). `avx2` is kept compiled for `bench-get-moves`
  and future tuning on a box where it wins, behind the same `cfg` caveat. Edax's
  AVX2 pays off partly via its *vboard* incremental representation (the move list
  is built once per node from a vector board); a fair re-test there would need
  that, not just a drop-in primitive swap.
- **Step 22 static presorted order — built, measured, reverted.** Edax never
  skips `get_moves` (the empties list only *orders* the walk, legality still comes
  from the bitboard), so the list is purely an ordering lever. Built it anyway to
  measure: an incremental empties `SquareList` as a *safe index arena* (a circular
  doubly-linked ring whose `prev`/`next` are `u8` indices into a fixed inline
  `[EmptyNode; 65]`, giving Edax's O(1) `empty_remove`/`empty_restore` with no
  `unsafe`), built at the root in static `SQUARE_VALUE` priority and
  removed/restored around each child. Behind a `const STATIC_ORDER` switch, A/B'd
  the static walk against the per-node `order_score` + sort: static visits **4.6×
  the nodes at 16e and 7.0× at 18e** (1.30M vs 0.28M; 9.71M vs 1.38M), for 3–4.6×
  slower wall-clock — even though the static walk is *cheaper per node* (≈39M vs
  27M nodes/s; it skips `order_score`'s mobility probe). Ordering quality dominates
  ordering cost by a wide margin. The whole thing (`empties.rs`, the field, the
  gated branches) was **reverted** — dead code with the flag off, and the dynamic
  path is strictly better with it on. Conclusion stands: the `SquareList` is not
  worth it as a move-gen lever (enumeration was never the bottleneck; cf. Step 19)
  nor as an ordering one. If the parallel split (Step 21) later wants an O(1)
  incremental empties structure, the safe-arena design here is the template.
- **Step 27 — cheaper move-ordering mobility (A/B'd, reverted).** `order_score`
  runs a full `get_moves` + popcount per child (the bulk of the ~17% `get_moves`
  hot cost), so the mobility term was swapped for Edax's *potential mobility*
  (empties adjacent to opponent discs — one file-masked bitboard dilation, no
  occluded fill) behind a `const USE_POTENTIAL_MOBILITY`, A/B'd on sequential
  (deterministic) node counts. The proxy visits **1.49× the nodes at 14e, 1.63× at
  16e, 1.78× at 18e** (56.6k→84.5k; 282k→459k; 1.38M→2.46M) — the penalty *grows*
  with depth — for **30–55% slower** wall-clock (16e 10.4→14.5ms, 18e 52.4→81.1ms)
  despite the cheaper per-node key. Same outcome as Step 22's static order:
  potential mobility is a much looser ordering signal than exact mobility, and
  ordering quality dominates ordering cost by a wide margin. **Reverted.** Cheapening
  the mobility probe is a dead end *unless* a proxy can be found that does not
  loosen the order — none of the cheap candidates (static `SQUARE_VALUE`, parity,
  potential mobility) clear that bar. The `get_moves`-in-`order_score` cost is real
  but the exact key is load-bearing.
- **Step 30 empties-list enumeration — built, A/B'd, reverted.** The committed
  shallow tier enumerates moves with `get_moves` + a parity mask. Edax instead
  walks an **empties list** (the safe-arena `EmptyList` ring: `[u8; 65]`
  `next`/`prev`, O(1) `remove`/`restore`, built once at shallow entry), computing a
  flip per empty square (`flip != 0` ⇒ legal) and applying via a flip-reusing
  `do_move_with_flip` — no `get_moves`. Built it in the *same* parity order
  (odd-quadrant pass then even, ascending within each) so **node counts are
  identical** (79,106 / 393,323 / 1,919,923 at 14/16/18e — exact match), making it
  a pure per-node wall-clock test. It ran **~3–4% slower** back-to-back (16e 10.9 vs
  10.45, 18e 56.0 vs 54.15, 20e 271.8 vs 263.7 ms). Cause: the empties walk computes
  a flip for *every* empty (incl. illegal, `flip == 0`) plus list bookkeeping,
  whereas one branchless `get_moves` Kogge-Stone fill finds all legal moves at once
  and only the legal ones are then flipped. Same verdict as the Step 22 `SquareList`
  it reuses: the incremental empties structure is not worth it in our bitboard
  representation. **Reverted** — `get_moves` + parity mask is the shallow tier's
  final form, completing Step 30. (Copy-make vs make/unmake was confirmed a
  non-lever: a `Position` is 16 bytes on the stack.)
- **Step 21 root-only YBWC — built, regressed, superseded.** The first cut split
  only the *root's* siblings (eldest sequential to set alpha, the rest in
  parallel). It regressed and got *worse* with threads (20e ms/pos: 337 / 365 /
  374 at t=1 / 8 / 16; +55% nodes at t=8) — with strong move ordering the eldest
  child dominates the work and is searched sequentially, so there is almost
  nothing left to parallelize at the root (Amdahl), while the speculative sibling
  work and lock traffic only add overhead. Real scaling requires splitting *inside*
  that eldest subtree, i.e. at the interior null-window nodes — which is exactly
  the committed full recursive YBWC (Step 21 above). Root-only is precisely why
  Edax splits recursively rather than at the root.

## Edax comparison — where the remaining gap is (evidence)

Benchmarked our sequential exact solver against Edax (the reference engine) to find
out whether more speed is available and, if so, in which part. Method: 6 boards of
exactly 24 empties from the PGN set (`bench --per-board` emits them as OBF plus our
per-board score/nodes/time), solved single-threaded by both; exact scores agree on
all 6 (confirming both solve the same problem). Edax run from its repo:
`lEdax-x64-modern -solve <obf> -l 60 -n 1` (level 60 = exact). Both cold per board.

Headline: **Edax is ~8× faster wall-clock** (9.1s vs 72.2s) for the identical exact
result — and it is almost entirely **node count**, not per-node speed:

| solver / mode | nodes | wall-clock | nodes/s |
|---|---|---|---|
| Ours (cold exact PVS, full window) | 2,463M | 72.2s | 34.1M |
| Edax modern, default (ID + MPC) | 344M | 9.10s | 37.8M |
| Edax modern, `-selectivity 100` (ID, no MPC) | 353M | 9.44s | 37.4M |
| Edax popcnt, default | 344M | 10.07s | 34.2M |

Each finding measured:
- **Per-node speed is a non-issue.** Our 34.1M nodes/s equals Edax's popcnt build
  (34.2M) and is ~10% behind the AVX2 modern build (37.8M). That ~10% is exactly
  the SIMD-primitive gap (Steps 26/28) — confirmed the *wrong* target. (The matching
  nodes/s also shows the node counts are comparable, not a counting artifact.)
- **MPC / selectivity is NOT the lever.** Forcing Edax to pure exact
  (`-selectivity 100`, no probcut — accepted on the command line, no rebuild)
  changed its node count <3% (344M → 353M).
- **The ~7× is node count from iterative deepening + move ordering** — Edax's
  shallow eval-guided searches seed the TT with near-perfect best moves and bounds
  before the deep exact pass. We do a single cold full-window PVS.

Decomposed the ~7× by prototyping each half against our own solver (same 6 boards;
both prototypes A/B'd behind a `const`, then reverted — throwaway):

| configuration | nodes | vs our baseline |
|---|---|---|
| Ours, baseline | 2,463M | 1.0× |
| + move-seeding (cheap-eval ID prototype) | 2,461M | 1.0× (no effect) |
| + perfect tight window `[S-1,S+1]` (aspiration ceiling) | 1,264M | 1.95× |
| Edax | 344M | 7.2× |

- **Move-ordering seeding with our mobility heuristic: ~0.** A depth-limited
  lookahead with our crude eval picks the same moves `order_score` already does, so
  the TT hint adds no information. Improving ordering needs a *positional/pattern*
  eval, not another mobility signal.
- **Window narrowing: ~2× (ceiling, perfect estimate).** Achievable via
  MTD-f/aspiration; the smaller lever, and it does *not* need a strong eval.
- **The remaining ~3.7× is move-ordering quality**, reachable only with a real
  evaluation function (Edax's `eval.dat`). This project trains one
  (`src/training/`) but it is not yet strong enough, and it is not wired into the
  solver (and `Weights::evaluate` allocates per call — a hot path would need an
  alloc-free eval).

**Direction (supersedes the SIMD steps 26/28 as the priority):**
1. **MTD-f / aspiration** for the ~2× window lever (Step 31). *Built and measured:
   guess-free MTD is net-neutral — the ceiling needs a good score estimate, so this
   lever is eval-gated too. Shelved behind `USE_MTD = false` as the scaffold for
   eval-seeded MTD-f.*
2. Then **eval-guided move ordering** for the ~3.7× — and the eval estimate also
   unblocks (1). Both are deferred until the trained weights are much stronger *and*
   a fast alloc-free eval path exists in the solver. Training the eval is therefore
   the gating next phase.

The eval is now the critical path, so it gets its own steps below, in dependency
order: **Step 32** (make training fast/correct enough to *produce* a strong eval),
**Step 33** (an alloc-free flat pattern eval in the solver), **Step 34** (wire it
into move ordering and MTD-f). Steps 33/34 are modelled directly on Edax's
`eval.c`/`move.c`/`midgame.c` — see those steps for the specific mechanisms.

## Remaining steps

Steps 1–25, 27, 29, 30 are implemented or resolved; Step 31 (MTD-f) is built but
shelved (net-neutral without an eval — see above). **Steps 32–34 below are the
current priority phase** — the eval-gated node-count levers (~2× window, ~3.7×
ordering): Step 32 makes training fast enough to produce a strong eval (**core done:
~2700× single-thread, see below**), Step 33 adds an alloc-free flat pattern eval to
the solver, Step 34 wires it into ordering and MTD-f. Steps 26 and 28 below come from
a **callgrind profile of the sequential hot path**
(release + debug symbols, 16e, 150 boards) — but the Edax comparison above shows
they target the ~10% per-node gap, not the ~7× node-count gap, so they are now low
priority behind the search-algorithm work.
These are *instruction counts*, so cache misses, branch mispredicts, and the
parallel TT contention are not captured — `perf` is the missing wall-clock /
cache view, blocked on this box by `perf_event_paranoid = 4` (needs root to
lower). Self-cost ranking: **flip ~31%** (already maximal — Steps 11/15),
**`get_moves` ~17%** (per-node move-gen *and* per-child `order_score` mobility),
`alphabeta_exact_nws` bookkeeping ~12%, `solve_1`/`count_last_flip` ~6%,
`tt_probe`/`store` ~3%, **`malloc`/`free` ~2.7%** (the per-node move-list `Vec`),
per-node `get_stability` ~1%. (The `find_edge_stable` ~17% seen in a *short* run
is the one-time `OnceLock` edge-table build, not a hot-path cost — it amortizes
to ~0 as boards grow; likewise `memchr`/`CharSearcher` ~1% is PGN parsing at
load.)

### Step 31 — MTD-f / aspiration root (window narrowing) — built, net-neutral, shelved
The Edax comparison shows a ~2× node-count *ceiling* from searching the exact tree
with a tight window instead of the full `[-64, 64]`. Implemented it as
[`Search::solve_mtd`]: repeated exact null-window probes ("score ≥ t?" for odd `t`,
since `S` is always even) bisecting `[SCORE_MIN, SCORE_MAX]` to the exact value
(~6 probes), reusing the never-cleared TT across probes. Result stays exact (every
probe is the exact null-window search). A/B'd behind `USE_MTD` on the 6×24e set:
**net-neutral, −0.1% nodes** (helps on extreme/near-zero scores, hurts on moderate
ones — they cancel; wall-clock identical). The ceiling assumed a *perfect* estimate;
guess-free bisection pays for the journey — the first probe ("S ≥ 1?") is nearly a
full solve, and for scores far from 0 the extra probes outweigh the savings. MTD-f
only wins with a **good first guess**, which needs an evaluation function. So this
lever is **eval-gated, same as move ordering**. Kept behind `USE_MTD = false` as the
scaffold for eval-seeded MTD-f (feed the eval's estimate as the first guess) once
the trained weights are usable; revisit then, sequential first, parallel root later.

### Step 32 — training speedup (gating: produce a strong eval)
Training (`src/training/`) is the gate on Steps 33/34, but the current trainer is
slow *and* its parallel path is mis-designed. Findings from reading
`trainer.rs`/`weights.rs`/`features.rs`:

- **The clone+full-merge runs per 32-example batch — even single-threaded.**
  `train_epochs` never calls the clean in-place `train_batch`; instead, for *every*
  batch it `weights.clone()`s the whole table per worker (`trainer.rs:219`), spawns
  a fresh `std::thread` (no pool, `:222`), and `merge_from_workers` (`:291` →
  `weights.rs:205`) scans **all weights** to average deltas. The table is 30
  empty-ranges × ~892K patterns ≈ **26.7M f32 ≈ 107 MB**, so each batch of 32 pays a
  107 MB copy + a 27M-element scan regardless of the ≤47×32 slots actually touched.
  With `threads = 1` this path still runs (`workers = 1`). **Fix (biggest win, do
  first): single-thread calls `train_batch(&mut weights, …)` in place** — deletes
  the clone/spawn/merge entirely. Likely 1–2 orders of magnitude faster on its own.
- **Feature indices are recomputed every epoch with a `Vec` alloc + per-cell
  `get_cell`.** `features.extract()` allocates a fresh `Vec<u32>` and loops
  cell-by-cell (`features.rs:135`), and is called **twice per example per epoch** —
  in `weights.evaluate` (`weights.rs:68`) and again in `train_batch`
  (`trainer.rs:122`). Positions are fixed across epochs, so **precompute the 47
  indices once at load and store `[u16; 47]` + the range index on
  `TrainingExample`**; every epoch then becomes a pure dot-product + scatter update,
  zero extraction, zero alloc (removes ~95% of feature work over 10+ epochs).
- **Flatten weights to one `Vec<f32>`** with per-(feature,range) base offsets
  (Edax's flat layout, Step 33). Kills the `Vec<Vec<Vec<f32>>>` triple pointer-chase
  in the hot dot-product and makes a *sparse* merge (only touched slots) trivial.
- **Parallel: don't clone/merge the whole table.** SGD over examples is
  embarrassingly parallel — either Hogwild-style shared `&[AtomicU32]` updates, or
  accumulate **sparse** deltas (touched `(feature,range,pattern)` only) and merge
  those, from a persistent pool, not per-batch spawns.

The current merge also averages a minibatch-of-1-per-example; once weights are flat
+ shared this can become proper accumulated-gradient minibatch SGD. None of this
changes the model — only its speed — so it is pure prep for a stronger eval.

**Done (the first two items; model unchanged, verified by identical per-epoch
loss).** `CompiledExample` extracts the 47 feature indices **once** up front and
reuses them every epoch (`evaluate_indices`/`sgd_step_indices` on `Weights` are the
alloc-free, single-`range_idx` accessors). The single-threaded path now runs online
SGD **in place** — no per-batch clone or merge. Measured, identical config (one PGN,
empties ≤ 14, 2624 examples, 2 epochs, threads 1): **356 → ~960K ex/s overall
(~2700×)**, loss bit-identical (epoch 1 499.0375, epoch 2 208.3582). At scale (10
PGNs, 31,777 examples × 10 epochs) the whole run is **~0.1 s at ~4M ex/s/epoch** —
the speedup grows with dataset size because the old per-batch clone/merge cost was
O(weights)·n_batches, independent of batch content. The **third/fourth items
(flatten + lock-free parallel) are still open** but now low-value: at ~4M ex/s
single-threaded the parallel path is *counter-productive* (the 107 MB clone + merge
dwarfs the trivial per-example compute and epoch-level model-averaging converges
worse), so `threads = 1` is the right default. Revisit flattening only as part of
Step 33 (the solver eval needs the flat layout anyway).

### Step 33 — alloc-free flat pattern eval in the solver
`Weights::evaluate` allocates per call (the `extract` `Vec`) and triple-chases
`Vec<Vec<Vec<f32>>>` — unusable in the hot path. Edax's eval (`eval.c`, `eval.h`,
`midgame.c`) is the alloc-free template to copy:

- **Incremental feature state carried in the search.** Edax's `struct Eval`
  (`eval.h:36`) is just `unsigned short feature[48]` (each = a feature's trinary
  index, pre-offset) + `n_empties` + `parity` — ~96 B, lives on the stack/search
  node, never heap-allocated. Add an analogous `[u16; 47]` (+ range index) to the
  solver state.
- **O(touched) make/unmake via a coordinate→feature table.** `EVAL_X2F[square]`
  (`eval.c:104`) lists, per square, the ≤7 features it belongs to and the power-of-3
  weight of that square in each. `eval_update_0/1` (`eval.c:782`) on a move:
  subtract 2× the moved square's contribution, then add/subtract each flipped bit's
  contribution — a short switch, no loop over all 47 features, no alloc.
  `eval_update_leaf` (`:893`) copies parent features first (the copy-make analogue,
  cheap at 96 B). **Build the inverse of `Features` once** (a `[Vec<(feat,pow)>; 64]`
  scatter table) — also reusable to speed up Step 32's one-time extraction.
- **Score = flat dot-product, int16 weights, indexed by ply.** `accumlate_eval`
  (`midgame.c:36`) is literally `sum = w->C9[f[0]] + … + w->S7654[f[45]]` — ~46
  lookups into a flat per-ply weight array, no indirection, no alloc (AVX2 path uses
  gather). Edax stores weights as `short`, mirror-symmetry-**packed** in `eval.dat`,
  unpacked once at load into flat per-ply arrays (`eval.c:660`). Our per-empties
  ranges already mirror Edax's ply = `60 - n_empties` table selection. **Deliverable:
  flatten + (optionally) quantize trained weights to a per-range flat `[i16]`/`[f32]`
  and an `eval(&Eval) -> i32` that is a straight-line sum.** Eval need not run at
  every leaf — only at the shallow nodes where ordering/MTD use it (Step 34).

### Step 34 — eval-guided move ordering (the ~3.7× lever) + eval-seeded MTD-f
With Step 33 in place, replicate Edax's **two-tier, depth-gated** midgame ordering
(`move.c`). Our current `order_score` (Step 6b) + the shallow parity tier (Step 30)
already match **tier 1** (`movelist_evaluate_fast`, `move.c:299`: corner stability +
potential/real mobility + `SQUARE_VALUE` + parity, hash moves shortcut to big
constants). The missing piece is **tier 2** (`movelist_evaluate`, `move.c:368`):

- **Depth-gated dispatch.** Per node Edax computes `sort_depth = (depth - 15) / 3`
  (clamped 0..6) and gates on `min_depth_table[empties]` (`move.c:370`). Below the
  gate it falls back to tier 1; at/above it adds a **shallow-search bonus**.
- **The shallow-search bonus is the new signal** (`move.c:440`): make the move
  (incremental `search_update_midgame` → `eval_update`), then add
  `(SCORE_MAX − shallow_score) · w_eval` where `shallow_score` comes from
  `search_eval_0` (1-ply) / `search_eval_1` (2-ply) / `search_eval_2` (3-ply) /
  `PVS_shallow` (depth 3–6) by `sort_depth`; restore. Weight `w_eval = 1<<15` is
  co-dominant with mobility. This is a *positional/pattern* score — new information —
  which is why the plan's mobility-only seeding prototype was ~0: it duplicated
  `order_score`'s signal. The score layering (move.c:351, wipeout 1<<30 … parity
  1<<0) keeps each term strictly dominating the next.
- **Two structural prerequisites we lack.** (a) **Iterative deepening**: Edax's
  shallow passes seed the TT with near-perfect best move + bounds, and
  `movelist_evaluate` reads `hash_data->move[0/1]` as the top-2 ordering keys — we do
  a single cold full-window PVS, so there are no hash moves to promote. (b)
  **`inc_sort_depth[node_type]`** (`midgame.c:647,791`): PV nodes get a deeper
  `sort_depth` than cut nodes. Both are part of this step, not optional polish.
- **Same machinery feeds eval-seeded MTD-f (Step 31).** The eval's score estimate is
  the MTD-f first guess; the iterative-deepening shallow result is even better. Do
  ordering first (it also produces the estimate), then revisit `USE_MTD`.

Sequence within this step: (1) iterative-deepening driver + TT hash-move ordering;
(2) `sort_depth`/`min_depth_table` schedule calling the Step-33 eval; (3)
`inc_sort_depth`; (4) feed the estimate to MTD-f. A/B each against the 6×24e set and
the standard `bench` levels — the target is node count, not nodes/s.

### Step 26 — AVX2 `get_moves` in the real search
`get_moves` is ~17% of the hot path, yet Step 24 shelved `avx2` on a *micro*-
benchmark that was call-overhead-bound (7.81 vs 7.24 ns/call). At 17% of real
work a 10%-faster primitive is ~1.7% overall, so A/B the `avx2` variant *inside
the search* (not the micro-bench), `cfg`-gated since it is Intel-favorable. May
still lose on this AMD box — measure in `bench`, keep only if it wins.

### Step 28 — SIMD flip / `get_moves` on Intel
flip (~31%) and `get_moves` (~17%) are the two biggest primitives, and their
`avx2`/`bmi2` variants (Steps 11/24, already in-tree) only win on Intel — every
AMD re-bench has lost (microcoded PEXT; `target_feature` boundary blocks
inlining). Re-run `bench-flip` / `bench-get-moves` and the search on a Haswell+
box and wire a `cfg`/runtime override only if measured. The single largest latent
lever, but gated on hardware we do not have here.

Minor / opportunistic, not numbered: split PV nodes too (the spine is O(depth)
nodes — small); NUMA-aware shard placement; re-sweep `SPLIT_MIN_EMPTIES` and the
shard count on other hardware.
