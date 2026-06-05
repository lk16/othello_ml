# Othello Position Evaluator

A Rust implementation of a feature-based Othello position evaluator using the Edax feature system.

- **47 Edax features** — corners, edges, lines, diagonals
- **Per-2-empties weight tables** — 30 tables for fine-grained evaluation
- **SGD training** — gradient descent against Edax ground truth
- **Binary serialization** — single-file persistence (f32, backward-compatible with i16)
- **Minimal dependencies** — only `ctrlc` for graceful interrupt handling
- **33 tests**, all passing

## Quick start

```bash
export EDAX_PATH=/path/to/edax    # Required: for ground-truth evaluation
cargo build
cargo test
cargo run -- training_data/
```

## Package structure

```
src/
  othello/           # Game logic & file parsing
    position.rs      Position { player, opponent }  — bitboard pair
    board.rs         Board { position, black_to_move }  — position + side to move
    game.rs          Game, WTHOR/PGN loading
  training/          # Evaluation & training
    features.rs      Features — 47 Edax evaluation patterns
    weights.rs       Weights — weight table + save/load
    trainer.rs       Trainer, TrainingConfig, TrainingExample — SGD training
    edax.rs          EdaxInterface — subprocess communication (channel-based progress)
    eval_cache.rs    EvalCache — persistent FEN→score cache
```

## Documentation

- [Reference](docs/reference.md) — architecture, file formats, building & running
- [Claude Code setup](docs/claude-code-setup.md) — project-isolated Claude Code with a custom backend
