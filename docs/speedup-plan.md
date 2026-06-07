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

## Current baseline (after Step 7 — 3/4-empty solvers)

Dedicated `solve_3`/`solve_4` leaf solvers, natural square order (no parity
ordering yet). ~1.26× at every depth. The nodes/pos drop is partly a metric
change: `solve_3`/`solve_4` internal nodes aren't counted (only `solve_1`
leaves via `solve_2`), which also lowers the reported nodes/s — ms/pos is the
honest measure.

| empties | boards | nodes/pos | ms/pos | nodes/s | vs Step 6 |
|---------|--------|-----------|--------|---------|-----------|
| 14 | 673 | 66,306 | 6.8ms | 9.72M | 1.26× |
| 16 | 350 | 431,642 | 45.3ms | 9.52M | 1.26× |
| 18 | 55 | 2,864,005 | 303.1ms | 9.45M | 1.26× |
| 20 | 8 | 23,721,405 | 2388.0ms | 9.93M | 1.27× |

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

## Remaining steps

### Step 6b — Move ordering: Edax tricks
PVS pays off in proportion to ordering quality. Add Edax's other ordering
signals (square-weighted mobility, corner stability) and selectivity tricks to
reduce re-searches. Mobility (the dominant term) is already in place, so expect
modest gains. (Parity ordering is split out into Step 9.)

### Step 8 — Flip table
Edax uses a precomputed `count_last_flip` table indexed by row/col.
Would speed up `solve_1` and `solve_2` (and future 3/4-empty).
Currently using bitboard fallback.

### Step 9 — Edax parity move ordering
Order the empties so odd-parity regions are tried first, in `solve_3`/`solve_4`
(Edax's `sort3` / `parity_case` tables) and the main search. Speed-only;
deferred from Step 7.

### Step 10 — Transposition table (last; was "Step 5"/"Step 9")
Previously attempted but had a correctness bug. Do this last, once the rest of
the search is in its final shape.
