# Alphabeta Exact-Search Speed-Up Plan

## Benchmark

Single file (avoids reloading many files per invocation), board counts chosen
so each `--empties` level runs in ~30s. Wrap runs in `timeout` (see
best-practices.md). Example:

```
F=training_data/playok_pgn_75927000.pgn
for spec in "14 1000" "16 350" "18 55" "20 8"; do
  set -- $spec
  timeout 90 cargo run --release -q -- bench --empties $1 --max-boards $2 "$F"
done
```

## Baseline after Step 5b (larger sample)

~12.5–13.5M nodes/s, consistent across depth. The reference point Step 6
improves on.

| empties | boards | nodes/pos | ms/pos | nodes/s |
|---------|--------|-----------|--------|---------|
| 14 | 673 | 116,666 | 9.1ms | 12.85M |
| 16 | 350 | 836,979 | 66.5ms | 12.60M |
| 18 | 55 | 6,243,131 | 498.1ms | 12.53M |
| 20 | 8 | 47,214,268 | 3492.0ms | 13.52M |

## Baseline after Step 6 (PVS)

Null-window (PVS) search. Gain grows with depth (the 20-empty row is only 8
boards, so noisy).

| empties | boards | nodes/pos | ms/pos | nodes/s | vs Step 5b |
|---------|--------|-----------|--------|---------|------------|
| 14 | 673 | 109,412 | 8.6ms | 12.71M | 1.06× |
| 16 | 350 | 707,030 | 57.1ms | 12.39M | 1.16× |
| 18 | 55 | 4,682,275 | 382.5ms | 12.24M | 1.30× |
| 20 | 8 | 40,164,910 | 3036.1ms | 13.23M | 1.15× |

## Baseline after Step 7 (3/4-empty solvers)

Dedicated `solve_3`/`solve_4` leaf solvers, natural square order. ~1.26× at
every depth. The nodes/pos drop is partly a metric change: `solve_3`/`solve_4`
internal nodes aren't counted (only `solve_1` leaves via `solve_2`), which also
lowers reported nodes/s — ms/pos is the honest measure.

| empties | boards | nodes/pos | ms/pos | nodes/s | vs Step 6 |
|---------|--------|-----------|--------|---------|-----------|
| 14 | 673 | 66,306 | 6.8ms | 9.72M | 1.26× |
| 16 | 350 | 431,642 | 45.3ms | 9.52M | 1.26× |
| 18 | 55 | 2,864,005 | 303.1ms | 9.45M | 1.26× |
| 20 | 8 | 23,721,405 | 2388.0ms | 9.93M | 1.27× |

## Baseline after Step 8 (flip table)

`count_last_flip` table for `solve_1`. Identical node counts (search unchanged),
~1.13× faster per node.

| empties | boards | nodes/pos | ms/pos | nodes/s | vs Step 7 |
|---------|--------|-----------|--------|---------|-----------|
| 14 | 673 | 66,306 | 6.0ms | 11.04M | 1.13× |
| 16 | 350 | 431,642 | 40.2ms | 10.75M | 1.13× |
| 18 | 55 | 2,864,005 | 269.5ms | 10.63M | 1.12× |
| 20 | 8 | 23,721,405 | 2079.3ms | 11.41M | 1.15× |

## Baseline after Step 12 (deep-search split)

Leaf cases (≤4 empties) factored into `solve_leaf`; the general `≥5` search no
longer re-tests them per node. Identical node counts; ~2% faster at every depth.

| empties | boards | nodes/pos | ms/pos | nodes/s | vs Step 8 |
|---------|--------|-----------|--------|---------|-----------|
| 14 | 673 | 66,306 | 5.9ms | 11.29M | 1.02× |
| 16 | 350 | 431,642 | 39.5ms | 10.94M | 1.02× |
| 18 | 55 | 2,864,005 | 263.6ms | 10.87M | 1.02× |
| 20 | 8 | 23,721,405 | 2029.3ms | 11.69M | 1.02× |

## Baseline after Step 13 (skip ordering at 5 empties)

Order moves only when `empties >= 6` (`SORT_MIN_EMPTIES`). Node counts rise ~8%
(unordered empties-5 nodes re-search more under PVS) but each node is cheaper
(no `get_moves`-per-child at empties 5), netting ~3% faster.

| empties | boards | nodes/pos | ms/pos | nodes/s | vs Step 12 |
|---------|--------|-----------|--------|---------|------------|
| 14 | 673 | 71,882 | 5.7ms | 12.56M | 1.03× |
| 16 | 350 | 469,604 | 38.3ms | 12.27M | 1.03× |
| 18 | 55 | 3,108,566 | 256.7ms | 12.11M | 1.03× |
| 20 | 8 | 25,561,308 | 1982.8ms | 12.89M | 1.02× |

## Baseline after Step 14 (dedicated no-sort search)

`alphabeta_nosort` handles the unordered range (empties 5 here) iterating the
moves bitboard directly — no move-list `Vec`, no mobility tuples. Identical node
counts to Step 13; ~4–6% faster (empties-5 nodes are frequent, so dropping their
allocation matters).

| empties | boards | nodes/pos | ms/pos | nodes/s | vs Step 13 |
|---------|--------|-----------|--------|---------|------------|
| 14 | 673 | 71,882 | 5.5ms | 13.07M | 1.04× |
| 16 | 350 | 469,604 | 36.3ms | 12.94M | 1.05× |
| 18 | 55 | 3,108,566 | 242.0ms | 12.84M | 1.06× |
| 20 | 8 | 25,561,308 | 1895.7ms | 13.49M | 1.05× |

## Current baseline (after Step 15 — per-square flip table)

`Position::flip_mask` dispatches through a 64-entry table of `flip_at::<SQ>`
const-generic specializations. With the square constant the compiler folds the
move bit and prunes off-board directions, roughly halving the flip work for
edge/corner squares. Identical node counts; ~1.37× faster — flip computation was
a major bottleneck, and the table-lookup cost is dwarfed by the savings.

| empties | boards | nodes/pos | ms/pos | nodes/s | vs Step 14 |
|---------|--------|-----------|--------|---------|------------|
| 14 | 673 | 71,882 | 3.9ms | 18.43M | 1.38× |
| 16 | 350 | 469,604 | 26.3ms | 17.85M | 1.36× |
| 18 | 55 | 3,108,566 | 175.5ms | 17.71M | 1.36× |
| 20 | 8 | 25,561,308 | 1314.1ms | 19.45M | 1.42× |

## Baseline after Step 10 (transposition table)

A position→`[lower, upper]`-bound table (plus best move for ordering), consulted
in the ordered search at `empties >= TT_MIN_EMPTIES`. Because an exact endgame
score is intrinsic to the position, a stored entry never expires — the table is
never cleared, only refined, and is reused across every position a thread
evaluates (warming it). A stored best move seeds PVS ordering; sufficient stored
bounds cut the node off outright. Node counts roughly halve at depth; ~1.30×
(14e) growing to ~1.86× (20e) vs Step 15.

| empties | boards | nodes/pos | ms/pos | nodes/s | vs Step 15 |
|---------|--------|-----------|--------|---------|------------|
| 14 | 673 | 51,414 | 3.0ms | 16.89M | 1.30× |
| 16 | 350 | 293,857 | 18.1ms | 16.28M | 1.45× |
| 18 | 55 | 1,744,944 | 108.0ms | 16.16M | 1.63× |
| 20 | 8 | 12,552,838 | 708.4ms | 17.72M | 1.86× |

Tuning (`TT_BITS`, `TT_MIN_EMPTIES`) — see the swept tables in the constant docs
in `alphabeta.rs`: `TT_BITS = 19` (12 MB) is the cache-locality knee, `2^21+`
adds memory without cutting nodes at these depths; `TT_MIN_EMPTIES = 7` (≈ "on
for the whole ordered search" — the floor is `SORT_MIN_EMPTIES = 6`). Note
`SORT_MIN_EMPTIES` was *not* raised: the hash move only helps levels that probe
the TT (`empties >= 7`), whereas `SORT_MIN_EMPTIES` governs whether to pay the
mobility sort at empties 5 — the TT does not touch that crossover, so it stays
at 6.

## Baseline after Step 16 (Enhanced Transposition Cutoff)

Before searching any child in the ordered search, probe every child in the
transposition table; if one already has a stored upper bound proving the move
fails high (`-upper >= beta`), return that bound without searching. Builds
directly on the Step 10 table — no new state. Gated at `ETC_MIN_EMPTIES = 8`
(the structural floor `TT_MIN_EMPTIES + 1`, since a child at `empties - 1` must
itself be stored to cut). Node savings grow with depth: ~4% (14e) to ~11% (20e),
for ~1.03× (14e) to ~1.12× (20e) vs Step 10.

| empties | boards | nodes/pos | ms/pos | nodes/s | vs Step 10 |
|---------|--------|-----------|--------|---------|------------|
| 14 | 673 | 49,276 | 2.9ms | 16.78M | 1.03× |
| 16 | 350 | 275,165 | 17.1ms | 16.11M | 1.06× |
| 18 | 55 | 1,620,128 | 100.7ms | 16.09M | 1.07× |
| 20 | 8 | 11,210,978 | 631.0ms | 17.77M | 1.12× |

Tuning (`ETC_MIN_EMPTIES`) — see the swept table in the constant doc in
`alphabeta.rs`: swept 7..=12, knee at 8. 7 has *identical* node counts to 8 (its
empties-7 probes hit unstored empties-6 children, so they never cut) but is
slower — confirming the `TT_MIN_EMPTIES + 1` floor. 8 and 9 tie on wall-clock; 8
prunes strictly more nodes, so it wins for deeper-than-bench robustness.

## Current baseline (after Step 17 — stability cutoff)

The opponent's *stable* discs (those that can never be flipped) cap our score at
`64 - 2*stable`; when that upper bound already fails low (`<= alpha`) the node
returns it without searching. Stability is Edax's exact-edge table (built once at
runtime) plus full-line and an iterative central "spread" — a conservative lower
estimate, so the bound is always valid. Gated per-empties by `STABILITY_THRESHOLD`
(only worth computing when alpha is high enough to possibly cut). The biggest win
since the flip table, and it grows sharply with depth: ~1.07× (14e) to ~1.55×
(20e) vs Step 16, node counts nearly halving at 20e.

| empties | boards | nodes/pos | ms/pos | nodes/s | vs Step 16 |
|---------|--------|-----------|--------|---------|------------|
| 14 | 673 | 43,478 | 2.7ms | 16.10M | 1.07× |
| 16 | 350 | 234,963 | 15.0ms | 15.71M | 1.14× |
| 18 | 55 | 1,266,005 | 82.1ms | 15.42M | 1.23× |
| 20 | 8 | 6,407,984 | 408.3ms | 15.69M | 1.55× |

Per-node throughput dips slightly (each gated node now computes stability), but
the node savings dominate. Tuning: `STABILITY_THRESHOLD` is Edax's table kept
verbatim — a swept global offset (−2..+6) changed node counts <0.1% and time
within noise (the gate only decides *when to attempt* a cut; attempts that cannot
cut are skipped regardless), and our scores share Edax's disc-difference units.
Re-tune if move ordering or the stability estimate itself changes.

## History (early benchmark — 14 empties, only 20 boards, noisy)

| Step | nodes/pos | ms/pos | speedup vs baseline |
|------|-----------|--------|---------------------|
| Baseline | 1,644,647 | 94.0ms | 1× |
| Step 1: fastest-first ordering | 141,284 | 15.0ms | 6.3× |
| Step 2: bitboard flips | 141,284 | 11.1ms | 8.5× |
| Step 4: pass empties | 141,284 | 10.6ms | 8.9× |
| Step 5b: 1-empty + 2-empty special cases | 98,313 | 7.8ms | 12.0× |

## Committed steps (all on branch `speed-up`)

- **Step 0** (409004b): bench subcommand (`othello_eval bench --empties N --max-boards N <paths>`)
- **Step 1** (2f42111): fastest-first move ordering (child position stored in vec, no double do_move)
- **Step 2** (58322a0): bitboard flip computation (replaces loop-based flipped())
- **Step 4** (b507509): pass empties as parameter; fast path for empties==0
- **Step 5b** (72a75d2): 1-empty and 2-empty leaf solvers (`solve_1`/`solve_2`),
  avoiding `Vec`, `get_moves()`, and `Position` construction at the innermost
  search levels. Note: Edax's `board_score_1`/`board_solve_2` assume a
  null-window caller, but `alphabeta_exact` passes a general `[alpha, beta]`
  window, so these are reimplemented as `solve_1` (exact, no window) and
  `solve_2` (fail-soft negamax). Step 3 ("special-case last 1 move") is
  subsumed by `solve_1`.
- **Step 6**: Principal Variation Search in `alphabeta_exact` —
  first (best-ordered) move full window, siblings probed with a null window
  `(alpha, alpha+1)` and re-searched only on a fail-high. No empties gate (Edax
  applies PVS at every node). `solve_2` stays general-window because the first
  child along the PV calls it with a full window.
- **Step 7**: `solve_3`/`solve_4` leaf solvers — fail-soft negamax over the 3/4
  empties, recursing into `solve_2`/`solve_3`. Edax's `search_solve_3/4` use a
  fixed-reference min/max convention; reimplemented here as plain negamax for
  consistency with `solve_2`. Natural square order (parity ordering is Step 9).
- **Step 8**: `count_last_flip` table for `solve_1` — the move's four lines are
  full except the move square, so flip counts come from a per-line lookup
  indexed by (line position, player pattern). `COUNT_FLIP` and the diagonal
  masks are generated at compile time with `const` blocks (no hardcoded
  tables). `solve_2`/`solve_3`/`solve_4` keep `flips_for` since they need the
  full flip mask to build child positions.
- **Step 12**: deep-search split — leaf cases (≤4 empties) factored into
  `Search::solve_leaf`; `alphabeta_exact` now handles only `≥5` empties and
  dispatches via `search_exact` at the recursion boundary, so the hot
  internal-node path no longer re-tests the five leaf cases each visit.
  Identical node counts; ~2% faster at every depth.
- **Step 13**: skip move ordering below `SORT_MIN_EMPTIES`. Swept N: N=6 (only
  empties-5 nodes unordered) is best at ~3% over Step 12; N=7 is break-even and
  N=8 is clearly worse (~+40% nodes, +12% time) — ordering pays for itself at
  empties ≥ 6, but at empties 5 the `get_moves`-per-child outweighs the few
  extra nodes from worse PVS ordering. Set `SORT_MIN_EMPTIES = 6`. This crossover
  is empirical: re-tune it after Steps 6b / 10 / 11 (see notes on each), which
  shift the ordering cost/benefit balance.
- **Step 14**: dedicated `alphabeta_nosort` for the unordered range
  (`5 ..< SORT_MIN_EMPTIES`) — iterates the moves bitboard directly with no
  move-list `Vec` or mobility tuples; `search_exact` is now a 3-way dispatch
  (leaf / no-sort / sorted). Identical node counts; ~4–6% faster. The win grows
  if `SORT_MIN_EMPTIES` rises (more levels use this allocation-free path).
- **Step 10**: transposition table. A `[lower, upper]`-bound table keyed on the
  full position (stored in full — a partial key risks returning a wrong score,
  the bug that sank the earlier attempt), with the best move kept for ordering.
  Exact endgame scores are path-independent, so an entry never expires: the
  table is never cleared, only refined (bounds intersected on re-store), and is
  reused across all positions a thread solves. Owned by a reusable `Search`
  exposed via the public `Solver` (one per `bench` run / cache worker / batch),
  so the multi-MB table is allocated once, not per call. Wired only into the
  ordered search (`alphabeta_exact`) at `empties >= TT_MIN_EMPTIES`: a probe
  returns on a sufficient stored bound, narrows the window otherwise, and seeds
  PVS with the stored best move; the fail-hard result is classified against the
  searched window and written back. We deliberately avoid a `thread_local!` for
  the per-thread reuse (less readable) — affordable only because threading an
  owned `Search` through callers is free per node (`&mut self`, never moved);
  reach for `thread_local!` only if an explicit owner ever costs real
  performance. `TT_BITS = 19`, `TT_MIN_EMPTIES = 7` (both swept — see the
  constant docs). ~1.30× (14e) to ~1.86× (20e). `SORT_MIN_EMPTIES` left at 6
  (the hash move helps only TT-probing levels, not the empties-5 sort crossover).
- **Step 15**: per-square flip table. `Position::flip_mask` dispatches through
  a 64-entry `FLIP` table of `flip_at::<SQ>` const-generic specializations; with
  `SQ` constant the compiler folds the move bit and prunes off-board directions.
  Identical node counts; ~1.37× faster — the biggest single win since the early
  steps, confirming flip computation was a major bottleneck. The general 8-ray
  body now lives once, in `flip_at`. Composes with Step 11 (a per-square body
  could itself use SIMD). Verified in the emitted asm: corner specializations
  are ~1/3 the size of the centre (3 vs 8 ray directions), and the table is
  indexed as `FLIP[(mv & 63) as usize]` so no bounds check is emitted (the `&63`
  compiles to a single `and`, dropping the lib's `panic_bounds_check` count from
  36 to 22); perf-neutral since that branch was predicted-not-taken anyway.
- **Step 16**: Enhanced Transposition Cutoff (ETC). In `alphabeta_exact`, before
  searching any child, probe every child in the Step 10 transposition table; if
  one has a stored upper bound proving the move fails high (`-upper >= beta`),
  return that bound without recursing. Reuses the existing table — no new state.
  Gated at `ETC_MIN_EMPTIES = 8`, the structural floor `TT_MIN_EMPTIES + 1` (a
  child at `empties - 1` must be ≥ `TT_MIN_EMPTIES` to have been stored). Swept
  7..=12: 7 matches 8's node counts exactly (empties-7 probes hit unstored
  empties-6 children, never cut) but is slower, confirming the floor; 8 and 9 tie
  on time, 8 prunes more nodes. ~1.03× (14e) to ~1.12× (20e) vs Step 10; node
  savings grow with depth (~4%→~11%). The cut condition uses `>= beta` (not Edax's
  `> alpha`) so it is correct for our general-window nodes, not only null windows.
- **Step 17**: stability cutoff. The opponent's stable-disc count `s` bounds our
  score at `64 - 2s`; if that `<= alpha` the node fails low and returns the bound
  without searching. Stability ported from Edax `board.c`: exact stable edges via
  a 64K `EDGE_STABILITY` table (built once at runtime through a `OnceLock` — the
  recursive fill is impractical to const-evaluate over 65536 entries, and Edax
  likewise builds it at startup), plus full-line detection and an iterative
  central "spread", giving a conservative lower estimate (never overcounts, so the
  bound stays valid). Placed in `alphabeta_exact` after the TT narrowing, gated by
  `STABILITY_THRESHOLD[empties]` (Edax's table, kept verbatim — see the "after
  Step 17" baseline for the offset sweep). Like Edax the cut returns without a TT
  store. ~1.07× (14e) to ~1.55× (20e) vs Step 16 — node counts nearly halve at
  20e, the biggest win since the flip table. Correctness rests on the unchanged
  Edax reference-score test plus stability unit tests.
- **Step 18**: dedicated null-window search path. `search_exact_nws` /
  `alphabeta_exact_nws` / `alphabeta_nosort_nws` take `alpha` only — the window is
  implicitly `[alpha, alpha + 1]`. The PV functions call them for sibling probes;
  the NWS functions call only each other. *Not passing `beta`* (vs the first,
  reverted attempt — a `const PV: bool` generic that still threaded `beta`) is
  what unlocks the win: the compiler folds `beta` to `alpha + 1`, so the TT-probe
  narrowing branches become dead (any entry that would narrow a width-1 window
  already early-returns), ETC's `value >= beta` reduces to `value > alpha`, and
  the PVS first-move/re-search structure collapses to one probe per child that
  cuts on the first fail-high. Node counts identical to Step 17 (a pure
  structural split); ~1.0× at 14–18e, ~1.02× at 20e in a same-session A/B (Step 17
  ~409ms vs NWS ~401ms, every NWS run faster) — the win grows with depth as more
  of the tree is null-window. The leaf solvers stay window-agnostic (NWS-
  specializing them, as the plan floated, is not worth the duplication for the few
  ops it would save).

## Refactors (no perf change)

- **Search struct**: the node counter moved from a thread-local `Cell` global
  into a `Search` struct threaded through the recursion, with the search
  routines as methods. `exact_score_with_nodes` returns the count for `bench`;
  `exact_score` discards it. Measured identical node counts and wall-clock vs
  the Step 8 baseline (a `const`-init thread-local is already a `%fs`-relative
  load/store, the same cost as a struct-field write) — done purely for clearer
  state ownership.
- **Shared flip core**: the 8-direction flip computation, previously duplicated
  between `Position::flipped` and a local `flips_for` in alphabeta.rs, is now a
  single `Position::flip_mask(mv, player, opponent)`. `flipped` = occupied-square
  check + `flip_mask`; the leaf solvers call `flip_mask` directly (they know `mv`
  is empty, so they skip the check). Perf-neutral vs Step 14 (within noise);
  removes the duplication and gives one place for Step 15 to optimize.

## Remaining steps

### Step 6b — Move ordering: Edax tricks
PVS pays off in proportion to ordering quality. Add Edax's other ordering
signals (square-weighted mobility, corner stability) and selectivity tricks to
reduce re-searches. Mobility (the dominant term) is already in place, so expect
modest gains. (Parity ordering is split out into Step 9.) Re-tune
`SORT_MIN_EMPTIES` afterward — richer/costlier ordering shifts its crossover.

### Step 9 — Edax parity move ordering (TRIED, REVERTED)
Order the empties so odd-parity regions are tried first. A quadrant's empties
have odd parity iff the XOR of `quadrant_bit` over them sets that quadrant's bit
(matching Edax's `QUADRANT_ID`/`parity`).

Implemented as a `sort_by_key` reorder in `solve_3`/`solve_4`. Result: ~2% fewer
nodes but ~2% **slower** wall-clock at every depth — the runtime sort (≈12–16
`quadrant_bit` calls per call, in the hottest solvers) costs more than the
pruning saves. Reverted.

Edax avoids this overhead with precomputed `sort3`/`parity_case` table lookups
(O(1) reorder) plus *incremental* parity (one XOR per ply). Replicating that
full machinery is the only way to make parity pay here, for a ~2% node ceiling —
not worth it versus Step 10. Left unimplemented.

### Step 11 — Alternative flip-computation variants
Edax ships many implementations of the same flip primitive (`flip_*.c`,
`count_last_flip_*.c`): portable bitboard (kindergarten / carry), `BMI2`
(PEXT/PDEP), `SSE`/`AVX2`, ARM `NEON`/`SVE`, etc. It selects one at *compile
time* for the target CPU. We currently use one portable bitboard `flips_for`.
Goal: implement a few alternatives, benchmark them, and pick the best per
target. Several are CPU-feature dependent. Faster `flips_for`/`get_moves` lowers
the per-child ordering cost, which pushes `SORT_MIN_EMPTIES` down — re-tune it
afterward.

**Test/config strategy** (no new deps; `std::arch` intrinsics only):
- All variants share one signature and are checked against the existing
  portable reference (`flips_for` / `Position::flipped`) over the deterministic
  square × pattern battery — a single `check_flip_impl(f)` harness.
- Portable variants are always compiled and tested.
- SIMD variants are `#[cfg(target_arch = "...")]`-compiled and written as
  `#[target_feature(enable = "bmi2"|"avx2"|...)]` (so they build even when the
  default target lacks the feature). Their tests guard the call with
  `is_x86_feature_detected!(...)` and no-op (with an `eprintln!` note) when the
  running CPU lacks it — so `cargo test` is correct on any machine; the variant
  is exercised only where supported. Cross-arch variants (NEON/SVE) are only
  testable on ARM/CI, not the x86_64 dev box.
- Production selection: prefer compile-time `cfg(target_feature)` (one impl per
  binary, like Edax) over a runtime function-pointer dispatcher — indirection
  in this hot leaf op would cost more than it saves. `bench` can still exercise
  each compiled variant for comparison.

## Null-window cutoffs (Steps 16–18)

These three steps come from studying how Edax uses null windows. The key
finding, which corrects a common misconception: **Edax does not reserve null
windows for the last few empties — null windows are its default search mode at
essentially every node.** The full `[alpha, beta]` window is the *exception*,
used only along the principal variation (`PVS_midgame`). Every non-PV node, and
the entire exact endgame (`NWS_endgame`), runs at a null window (β = α+1). The
last-4-empties solvers (`board_score_1`/`board_solve_2`/`search_solve_3/4`) are
null-window-only as a *consequence* of being called from that search, not
because those depths are special.

We already match the PV side of this: our `alphabeta_exact`/`alphabeta_nosort`
do textbook PVS (first child full window, siblings probed at `(-alpha-1,
-alpha)` with a re-search on fail-high). Edax layers three extra cutoffs onto its
NWS paths (`USE_TC`/`USE_ETC`/`USE_SC`); we had only the two-bound transposition
narrowing (Step 10). Steps 16 (ETC) and 17 (stability) add the other two —
**and, contrary to the initial framing here, both turned out to be correct in our
general-window search, not just on null windows**: each is a one-sided bound
test (`-upper >= beta` for ETC, `64 - 2*stable <= alpha` for stability) that is a
valid cutoff for any `[alpha, beta]`, firing on the null-window siblings as a
special case. Step 18 (a dedicated null-window path) was therefore never about new
cuts but a leaner per-node path; it landed as a small win (~1.02× at 20e) once
`beta` was dropped from the signature so the compiler could fold it away.

### Step 16 — Enhanced Transposition Cutoff (ETC) [DONE — see Committed steps]
Implemented and committed; `ETC_MIN_EMPTIES = 8` after sweeping 7..=12 (the
`TT_MIN_EMPTIES + 1` structural floor — see the "after Step 16" baseline table
and the constant doc in `alphabeta.rs`). One deviation from Edax worth recording:
Edax cuts on `-upper > alpha` (valid because its NWS callers have `beta =
alpha + 1`); we cut on `-upper >= beta` so the cutoff is correct in our
general-window nodes too, while still firing on the null-window siblings.

### Step 17 — Stability cutoff [DONE — see Committed steps]
Implemented and committed. Ended up the single biggest win of the three (~1.55×
at 20e). Two notes versus the original plan: (1) the cutoff is correct for our
general windows, not "sound only on a null window" — `64 - 2*stable <= alpha` is
a valid fail-low for any window; (2) the threshold table did *not* need re-tuning
to our scale — our scores are already in Edax's disc-difference units, and a swept
global offset moved nodes <0.1% (see the "after Step 17" baseline). Stability uses
the full Edax `get_stability` (exact edge table + full lines + central spread),
not just the corner estimate the plan guessed at.

### Step 18 — Dedicated null-window endgame path [DONE — see Committed steps]
Implemented and committed as `*_nws` functions taking `alpha` only. Two findings
worth recording: (1) a first attempt using a `const PV: bool` generic that still
*passed* `beta` was reverted — it gave no gain (slightly worse, from a second
monomorphization of a large function) because the compiler can't fold a runtime
`beta` to `alpha + 1`; the win only appears when `beta` is dropped from the
signature. (2) `solve_2`/`solve_3`/`solve_4` were *not* NWS-specialized (the plan
floated it): they are already lean fail-soft solvers and the duplication isn't
worth the few ops. Net ~1.02× at 20e, neutral shallower.

**Status:** Steps 16, 17, and 18 are all done (committed). This group is closed.
