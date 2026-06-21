# Othello ML Reference

## Architecture

```
othello_eval
├── othello/
│   ├── Position       - bitboard pair (player + opponent discs)
│   ├── Board          - Position + side to move
│   └── Game           - WTHOR/PGN loading; stores positions as Vec<Board>
├── eval/
│   ├── alphabeta      - exact endgame search + depth-limited heuristic search & best_move
│   └── cache          - persistent FEN→score cache for eval files
├── training/
│   ├── Features       - 46 Edax position-pattern features (eval.c EVAL_F2X)
│   ├── Weights        - weight storage, lookup, save/load
│   └── cg             - per-bucket conjugate-gradient least-squares trainer
```

## Key Features

- **Minimal dependencies** — only `ctrlc` for graceful SIGINT handling
- **46 position features** — Edax patterns transcribed from eval.c `EVAL_F2X`, tied by symmetry shape
- **Alpha-beta evaluation** — exact endgame search for training; depth-limited heuristic search for gameplay
- **Compact storage** — single binary file for all weights
- **CG least-squares training** — per-empties-bucket conjugate-gradient fit to the exact global optimum (no learning rate; ridge-regularized)
- **Eval file caching** — `--eval-file` loads cached evaluations or computes & saves
- **Interactive gameplay** — play against the bot via `play` subcommand
- **Search benchmarking** — measure nodes/position and time via `bench` subcommand

## Binary Weight Format

```
[Magic: 0xDEADBEEF (4 bytes)]
[Version: 3 (4 bytes)]          ← f32 weights; older v1/v2 are rejected (re-train)
[N Features: 46 (4 bytes)]
[Feature 0: name_len + name + cells_count + cells...]
...
[Feature 45: name_len + name + cells_count + cells...]
[Weight data: all f32 weights in row-major order]
```

Weights are indexed `[feature][empties-bucket][pattern]`, with **one bucket per
empties value** (61 buckets, empties `0..=60`) — so each ply has its own per-feature
table. (v1/v2 paired adjacent empties into 30 buckets; v3 is incompatible, hence the
re-train.) The table is ~218 MB of f32 (`61 × ~892K patterns`).

## Eval File Format

Each line: `<FEN> <score>`

The FEN is 66 characters (64 board cells + space + side to move).
Uses `X` for black, `O` for white, `-` for empty. The score is a signed integer.

## Building & Running

```bash
# Build
cargo build --release

# Test
cargo test

# Train with exact alpha-beta evaluation (empties <= N). `train` is an alias.
cargo run --release -- train-exact training_data/

# Train with eval file cache (avoids recomputing evaluations)
cargo run --release -- train-exact --eval-file ignored/evals.txt training_data/

# Extend the eval to empties > N via bootstrapped shallow-search labels
# (needs an already exact-trained weights file; see "Bootstrapped training" below)
cargo run --release -- train-boot --exact-empties 16 -n 24 -t 8 -w trained_weights.bin training_data/

# Play against the bot
cargo run --release -- play --weights trained_weights.bin

# Benchmark exact search (nodes/pos and time)
cargo run --release -- bench --empties 14 --max-boards 100 training_data/

# Show all commands
cargo run --release -- --help

# Show options for a subcommand
cargo run --release -- train-exact --help
cargo run --release -- train-boot --help
cargo run --release -- play --help
cargo run --release -- bench --help
```

## Bootstrapped training (`train-boot`)

`train-exact` can only label positions it can solve exactly, i.e. empties ≤ N
(the `-n` flag; exact search is infeasible deeper). `train-boot` extends the eval
to empties > N by **bootstrapping**: each position is labelled with the backed-up
score of a depth-`d` shallow search whose horizon leaves are scored by the *current*
weights (an alloc-free `FlatEval` snapshot). Fitting the eval to its own d-ply
lookahead sharpens it — the standard Logistello/Edax/NNUE-style technique.

To avoid unanchored drift, it works as a **curriculum**: starting from the exact
frontier N, it trains one **band of width `d`** at a time — `(N, N+d]`, then
`(N+d, N+2d]`, … up to `--max-empties`. Because a depth-`d` search from empties
≤ frontier+d bottoms out at empties ≤ frontier (already trained), every label is
anchored to the band below it. Weights are per-empty-range buckets, so each band
updates disjoint buckets (no forgetting). Each band is fit by the same per-bucket CG
least-squares solver as `train`; `-t` parallelizes both the band's label generation
and its bucket solves.

```bash
# Requires an exact-trained weights file first (train-exact, e.g. -n 16).
cargo run --release -- train-boot \
  --exact-empties 16 --max-empties 28 --depth 4 \
  -t 8 -w trained_weights.bin training_data/
```

Key flags: `--exact-empties N` (trusted frontier), `--max-empties M` (train up to),
`--depth d` (shallow-search depth = band width), `-w` (weights file, loaded and
overwritten per band). CG knobs (`--cg-iters`, `--ridge`, `--min-count`) apply per
band, as in `train`.
