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
│   ├── Features       - 47 position features (46 Edax patterns + edge_parity)
│   ├── Weights        - weight storage, lookup, save/load
│   └── Trainer        - SGD optimization with inverse-time LR decay
```

## Key Features

- **Minimal dependencies** — only `ctrlc` for graceful SIGINT handling
- **47 position features** — 46 Edax patterns from eval.c plus one corner-parity feature
- **Alpha-beta evaluation** — exact endgame search for training; depth-limited heuristic search for gameplay
- **Compact storage** — single binary file for all weights
- **Full SGD training** — configurable learning rate, batch size, epochs, LR decay
- **Eval file caching** — `--eval-file` loads cached evaluations or computes & saves
- **Interactive gameplay** — play against the bot via `play` subcommand
- **Search benchmarking** — measure nodes/position and time via `bench` subcommand

## Binary Weight Format

```
[Magic: 0xDEADBEEF (4 bytes)]
[Version: 2 (4 bytes)]          ← f32 weights (v1 = i16, still readable)
[N Features: 47 (4 bytes)]
[Feature 0: name_len + name + cells_count + cells...]
...
[Feature 46: name_len + name + cells_count + cells...]
[Weight data: all f32 weights in row-major order]
```

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

# Train with exact alpha-beta evaluation
cargo run --release -- train training_data/

# Train with eval file cache (avoids recomputing evaluations)
cargo run --release -- train --eval-file ignored/evals.txt training_data/

# Play against the bot
cargo run --release -- play --weights trained_weights.bin

# Benchmark exact search (nodes/pos and time)
cargo run --release -- bench --empties 14 --max-boards 100 training_data/

# Show all commands
cargo run --release -- --help

# Show options for a subcommand
cargo run --release -- train --help
cargo run --release -- play --help
cargo run --release -- bench --help
```
