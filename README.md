# Othello Position Evaluator

A Rust implementation of a feature-based Othello position evaluator using the Edax feature system.

- **47 Edax features** — corners, edges, lines, diagonals
- **Per-2-empties weight tables** — 30 tables for fine-grained evaluation
- **SGD training** — gradient descent against Edax ground truth
- **Binary serialization** — single-file persistence
- **Zero dependencies** — pure Rust standard library
- **14 tests**, all passing

## Quick start

```bash
export EDAX_PATH=/path/to/edax    # Optional: for ground-truth evaluation
cargo build
cargo test
cargo run
```

## Documentation

- [Implementation summary](docs/implementation-summary.md) — phases, architecture, file format, next steps
- [Claude Code setup](docs/claude-code-setup.md) — project-isolated Claude Code with a custom backend
