# Othello Position Evaluator

A Rust implementation of a feature-based Othello position evaluator using the Edax feature system.

## Features

- **47 Edax Features**: Corners (4), edges (4x4), lines (4x4), and diagonals (18)
- **Per-2-Empties Granularity**: 30 separate weight tables (one per 2-empty range from 2 to 60)
- **SGD Training**: Gradient descent optimization against Edax ground truth
- **Binary Serialization**: Single-file persistence for all weights + metadata
- **Minimal Dependencies**: Uses only Rust standard library

## Architecture

```
board.rs      - 64-bit bitboard representation
features.rs   - 47 Edax feature extraction system
weights.rs    - Weight storage and lookup (O(1))
training.rs   - SGD training loop
edax.rs       - Subprocess communication with Edax
io.rs         - Binary save/load functionality
```

## Usage

```rust
use othello_eval::{Board, Features, Weights, Trainer};

// Create initial position
let board = Board::initial();

// Load features and weights
let features = Features::edax();
let mut weights = Weights::new(features);

// Evaluate position
let score = weights.evaluate(&board, &features);

// Train on examples
let trainer = Trainer::new(0.01, 32);
trainer.train_batch(&mut weights, &examples);

// Save/load weights
WeightIO::save(&weights, "weights.bin")?;
let loaded = WeightIO::load("weights.bin")?;
```

## Environment

Set `EDAX_PATH` environment variable to point to Edax binary for ground truth evaluation:
```bash
export EDAX_PATH=/path/to/edax
```

## Building & Testing

```bash
cargo build
cargo test
cargo run
```

All 14 tests pass:
- Board representation tests
- Feature extraction tests
- Weight storage/update tests
- Serialization round-trip tests
- SGD update tests

## Implementation Notes

1. **Cell Indexing**: Standard 0-63 layout (a1=0, h1=7, a8=56, h8=63)
2. **Feature Patterns**: Extracted directly from Edax eval.c EVAL_F2X array
3. **Training**: Uses simple MSE loss with per-feature weight updates
4. **Disc Count**: Rounded to nearest even for table selection (e.g., 31 empties → table 30)
