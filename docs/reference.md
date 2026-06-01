# Othello ML Reference

## Architecture

```
othello_eval
├── Board          - 64-bit bitboard representation
├── Features       - 47 Edax pattern extraction
├── Weights        - Weight storage & lookup
├── Training       - SGD optimization
├── EdaxInterface  - Subprocess communication
└── IO             - Binary serialization
```

## Key Features

- **No external dependencies** — pure Rust standard library
- **47 Edax features** — exact pattern set extracted from Edax eval.c
- **Compact storage** — single binary file for all weights
- **Full SGD training** — configurable learning rate, batch size, epochs
- **Edax integration** — subprocess communication for ground truth evaluations
- **Eval file caching** — `--eval-file` loads cached evaluations or computes & saves

## Binary Weight Format

```
[Magic: 0xDEADBEEF (4 bytes)]
[Version: 1 (4 bytes)]
[N Features: 47 (4 bytes)]
[Feature 0: name_len + name + cells_count + cells...]
...
[Feature 46: name_len + name + cells_count + cells...]
[Weight data: all i16 weights in row-major order]
```

## Eval File Format

Each line: `<FEN> <score>`

The FEN is 66 characters (64 board cells + space + side to move). The score is a signed integer.

## Building & Running

```bash
# Build
cargo build --release

# Test
cargo test

# Train with Edax ground truth
EDAX_PATH=/path/to/edax cargo run --release -- training_data/

# Train with eval file cache (avoids recomputing Edax evaluations)
cargo run --release -- --eval-file evals.txt training_data/

# Show all options
cargo run --release -- --help
```
