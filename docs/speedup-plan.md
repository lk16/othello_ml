# Alphabeta Exact-Search Speed-Up Plan

Roadmap and design log for the exact endgame solver (`src/eval/alphabeta/`).
Mechanism details live in the code and its doc comments — this file keeps the
*why*: the benchmark recipe, the current baseline, what each step changed, dead
ends, and what's left. Per-constant tuning sweeps live in the constant doc
comments in `alphabeta.rs`, not here.

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

## Current baseline (after Step 11)

| empties | boards | nodes/pos | ms/pos | nodes/s |
|---------|--------|-----------|--------|---------|
| 14 | 2000 | 75,280 | 2.4ms | 31.4M |
| 16 | 1000 | 406,796 | 13.4ms | 30.4M |
| 18 | 250 | 2,146,052 | 77.2ms | 27.8M |
| 20 | 50 | 12,587,133 | 442.8ms | 28.4M |
| 22 | 8 | 73,398,896 | 2729.6ms | 26.9M |

The big wins, in order of impact: the per-square flip table (Step 15, ~1.37×),
the transposition table (Step 10), the stability cutoff (Step 17, ~1.55× at
20e), and the carry-64 flip (Step 11, ~1.13×). Full history is in git.

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

## Dead ends & decisions (not in code)

- **Step 9 — parity move ordering (reverted).** Ordering empties so odd-parity
  regions go first cut ~2% of nodes but ran ~2% *slower* — the runtime
  `quadrant_bit` sort in the hot `solve_3`/`solve_4` costs more than it saves.
  Edax only makes this pay via precomputed `sort3`/`parity_case` lookups plus
  *incremental* parity (one XOR per ply); replicating that for a ~2% node ceiling
  isn't worth it.
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

## Remaining steps

### Step 6b — Move ordering: Edax tricks
PVS pays off in proportion to ordering quality. Add Edax's other ordering signals
(square-weighted mobility, corner stability) and selectivity tricks to reduce
re-searches. Mobility (the dominant term) is already in place, so expect modest
gains. Re-tune `SORT_MIN_EMPTIES` afterward — richer/costlier ordering shifts its
crossover.

### Step 19 — Incremental empties list (Edax `SquareList`)
Edax never recomputes the set of empty squares: `search_setup` builds a doubly-
linked list (`SquareList empties[66]`, `u8` previous/next per square + `NOMOVE`/
`PASS` sentinels) once, then `empty_remove`/`empty_restore` unlink/relink the
played square in O(1) on make/undo, and `foreach_empty` walks it. The list is
built in a *presorted* square order (corners → edge classes → center), so walking
it yields a static quality move order with no per-node sort, and parity is updated
incrementally (`parity ^= QUADRANT_ID[x]`) alongside.

**Investigated — currently judged low-value; not pursued.** The "eliminate
`trailing_zeros`" motivation doesn't hold up here:
- `trailing_zeros` is a single instruction (~3 cycles); a list walk replaces it
  with serialized pointer-chases. For `solve_leaf`'s ≤4-empty extraction (the most
  frequent case) `tzcnt` + `x &= x-1` is likely *faster* than following 4 links.
- It doesn't remove `get_moves`/`do_move` (the expensive calls). Avoiding
  `get_moves` requires *also* switching move-gen to "walk every empty, flip-test
  it" — trading one branchless 8-direction pass for N per-square flips (worse at
  high empties; our `get_moves` is already branchless).
- We already capture the leaf-solver benefit: `solve_2/3/4` take explicit
  `x1..x4` and flip-test each, exactly like Edax's list-driven leaf solvers.
- Our search is immutable per node (`Position` by value, fresh children); a
  `SquareList` adds make/undo bookkeeping through every child, pass, and PVS
  re-search.

What it *would* unlock is a static presorted move order (replacing the mobility
sort + per-child `get_moves().count_ones()`) and cheap incremental parity — both
**ordering** levers, so their payoff belongs with Step 6b / Step 9, measured in
node count, not enumeration speed. Revisit only as part of an ordering rework.
