# Alphabeta Exact-Search Speed-Up Plan

## Benchmark target
`cargo run --release -- bench --empties 14 --max-boards 20 training_data/playok_pgn_75927000.pgn`

## Results so far

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

## Remaining steps

### Step 6 — Null-window search + Edax-style tricks
Switch the main search to null-window (PVS): search the first move with a full
window, the rest with `(alpha, alpha+1)`, re-searching only on a fail-high.
This also lets the leaf solvers (`solve_2` and future 3/4-empty) drop back to
the cheaper null-window form Edax uses. Layer on other Edax tricks (e.g.
better move ordering, stability-based cutoffs) as they pay off.

### Step 7 — 3-empty and 4-empty special cases
Port `search_solve_3` / `search_solve_4` from Edax for further speedup.
4-empty adds parity-based move ordering.

### Step 8 — Flip table
Edax uses a precomputed `count_last_flip` table indexed by row/col.
Would speed up `solve_1` and `solve_2` (and future 3/4-empty).
Currently using bitboard fallback.

### Step 9 — Transposition table (last; was "Step 5")
Previously attempted but had a correctness bug. Do this last, once the rest of
the search is in its final shape.
