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

## Current baseline (after Step 6b)

| empties | boards | nodes/pos | ms/pos | nodes/s |
|---------|--------|-----------|--------|---------|
| 14 | 2000 | 56,583 | 2.0ms | 28.3M |
| 16 | 1000 | 281,927 | 10.4ms | 27.1M |
| 18 | 250 | 1,379,799 | 54.0ms | 25.6M |
| 20 | 50 | 7,238,915 | 290.5ms | 24.9M |
| 22 | 8 | 35,303,820 | 1521.2ms | 23.2M |

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
- **13** skip move ordering below `SORT_MIN_EMPTIES = 6`.
- **14** dedicated `alphabeta_nosort` for the unordered range (no `Vec`).
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

## Remaining steps

Steps 1–24 are implemented or resolved. The five below come from a **callgrind
profile of the sequential hot path** (release + debug symbols, 16e, 150 boards).
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

### Step 26 — AVX2 `get_moves` in the real search
`get_moves` is ~17% of the hot path, yet Step 24 shelved `avx2` on a *micro*-
benchmark that was call-overhead-bound (7.81 vs 7.24 ns/call). At 17% of real
work a 10%-faster primitive is ~1.7% overall, so A/B the `avx2` variant *inside
the search* (not the micro-bench), `cfg`-gated since it is Intel-favorable. May
still lose on this AMD box — measure in `bench`, keep only if it wins.

### Step 27 — Cheaper move-ordering mobility
Most of the `get_moves` cost is `order_score` running a full `get_moves` +
popcount for *every* child. Edax's "potential mobility" (empties adjacent to
opponent discs — a cheap bitboard dilation, no occluded fill) is a candidate
proxy. Risk is the recurring lesson (Steps 6b/22): ordering *quality* dominates
ordering *cost*, so a weaker key can cost more nodes than it saves. A/B in node
count first; pursue only if mobility can be cheapened without loosening the order.

### Step 28 — SIMD flip / `get_moves` on Intel
flip (~31%) and `get_moves` (~17%) are the two biggest primitives, and their
`avx2`/`bmi2` variants (Steps 11/24, already in-tree) only win on Intel — every
AMD re-bench has lost (microcoded PEXT; `target_feature` boundary blocks
inlining). Re-run `bench-flip` / `bench-get-moves` and the search on a Haswell+
box and wire a `cfg`/runtime override only if measured. The single largest latent
lever, but gated on hardware we do not have here.

### Step 29 — Lock-free transposition table
The shared table is sharded `Mutex<Box<[TtEntry]>>` (Step 21); the lock traffic
and the guard's cache line are the contention ceiling as thread counts grow (the
sequential path also pays one predicted branch plus an uncontended lock via the
`Shared` arm). A lock-free design — Hyatt XOR-validated atomic entries, or a
per-slot seqlock — would cut that, but must stay *exactly* correct (full-key
validation, no torn reads), since the solver is reference-tested. The main lever
left for multi-core scaling beyond what recursive YBWC already gave.

### Step 30 — Edax's shallow-search tier (make/unmake + parity order, no move list)
Edax keeps no heap in its search: the `MoveList` is a stack local (`Move
move[34]`, uninitialized — the model for Step 25's stack buffer). Its make/unmake
machinery — `board_update`/`board_restore`, the empties-list O(1)
`remove`/`restore` (`search->empties[prev].next = …`), and copy-make (`vboard_next`
from a saved parent board) — exists to apply moves *cheaply* (each `Move` carries
its precomputed `flipped`), not to dodge allocation; the stack array already does
that.

What we have never tried is Edax's distinct middle tier, `search_shallow`, for
`n_empties` 5–7 (`DEPTH_TO_SHALLOW_SEARCH = 7`), which deliberately **drops the
mobility sort**: it orders by **parity only** (priority-quadrant moves first, no
`order_score`, no sort), walks the **empties list** with O(1) remove/restore,
applies moves by copy-make, and builds **no move list at all**. We have measured
the *pieces* and each lost on its own — the empties `SquareList` (Step 22,
reverted), parity as a primary key (Step 19, nodes ~doubled), cheap/static vs
dynamic mobility ordering (Step 22 A/B, 4.6–7× nodes), `SORT_MIN_EMPTIES`
(mobility-sort wins from 6). What is *untested* is the **combination** Edax
actually ships: parity ordering + empties-list make/unmake + zero move list,
confined to a thin 5–7 band, where the much cheaper per-node cost might offset the
extra nodes at the shallowest (most numerous) ordered levels. Speculative and
against our piecewise evidence, but it is the one unexplored point in that design
space — worth an A/B in node count and wall-clock if the low-empties ordering /
allocation cost ever looks worth re-opening.

Minor / opportunistic, not numbered: split PV nodes too (the spine is O(depth)
nodes — small); NUMA-aware shard placement; re-sweep `SPLIT_MIN_EMPTIES` and the
shard count on other hardware.
