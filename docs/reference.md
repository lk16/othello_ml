# Othello ML Reference

## Architecture

```
othello_eval
├── othello/
│   ├── Position       - bitboard pair (player + opponent discs)
│   ├── Board          - Position + side to move
│   └── Game           - WTHOR/PGN loading & game replay
├── eval/
│   ├── alphabeta      - exact endgame evaluation via alpha-beta search
│   └── cache          - persistent FEN→score cache for eval files
├── training/
│   ├── Features       - 47 Edax pattern extraction
│   ├── Weights        - weight storage, lookup, save/load
│   └── Trainer        - SGD optimization with inverse-time LR decay
```

## Key Features

- **Minimal dependencies** — only `ctrlc` for graceful SIGINT handling
- **47 Edax features** — exact pattern set extracted from Edax eval.c
- **Alpha-beta evaluation** — exact endgame position scoring via negamax with pruning
- **Compact storage** — single binary file for all weights
- **Full SGD training** — configurable learning rate, batch size, epochs, LR decay
- **Eval file caching** — `--eval-file` loads cached evaluations or computes & saves

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
cargo run --release -- training_data/

# Train with eval file cache (avoids recomputing evaluations)
cargo run --release -- --eval-file ignored/evals.txt training_data/

# Show all options
cargo run --release -- --help
```
