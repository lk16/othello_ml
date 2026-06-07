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

## Current baseline (after Step 13 — skip ordering at 5 empties)

Order moves only when `empties >= 6` (`SORT_MIN_EMPTIES`). Node counts rise ~8%
(unordered empties-5 nodes re-search more under PVS) but each node is cheaper
(no `get_moves`-per-child at empties 5), netting ~3% faster.

| empties | boards | nodes/pos | ms/pos | nodes/s | vs Step 12 |
|---------|--------|-----------|--------|---------|------------|
| 14 | 673 | 71,882 | 5.7ms | 12.56M | 1.03× |
| 16 | 350 | 469,604 | 38.3ms | 12.27M | 1.03× |
| 18 | 55 | 3,108,566 | 256.7ms | 12.11M | 1.03× |
| 20 | 8 | 25,561,308 | 1982.8ms | 12.89M | 1.02× |

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

## Refactors (no perf change)

- **Search struct**: the node counter moved from a thread-local `Cell` global
  into a `Search` struct threaded through the recursion, with the search
  routines as methods. `exact_score_with_nodes` returns the count for `bench`;
  `exact_score` discards it. Measured identical node counts and wall-clock vs
  the Step 8 baseline (a `const`-init thread-local is already a `%fs`-relative
  load/store, the same cost as a struct-field write) — done purely for clearer
  state ownership.

## Next up — search-structure experiments (do these before the Remaining steps)

### Step 14 — Dedicated no-sort search function
Step 13 landed with `SORT_MIN_EMPTIES = 6`, so the only unordered general level
is empties = 5. Remove the per-node `if sort` branch (and the unused mobility
tuple) by factoring a separate never-sorts routine for empties = 5, with the
sorting search dispatching into it at the boundary (mirroring Step 12's split).
Expected gain is small (empties-5 nodes are a minority); benchmark to confirm
it's worth the extra code.

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

### Step 10 — Transposition table (was "Step 5"/"Step 9")
Previously attempted but had a correctness bug. Do this once the rest of the
search is in its final shape. A hash move gives cheap good ordering, so this is
the step most likely to let `SORT_MIN_EMPTIES` rise (skip the mobility sort at
more levels) — re-tune it afterward.

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
