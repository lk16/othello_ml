# CLAUDE.md

Othello ML — a Rust feature-based Othello position evaluator trained via SGD against Edax ground truth.

## Build & test

```bash
pre-commit run -a && cargo test
```

Always use `pre-commit run -a` — not `cargo check` directly. It runs check, fmt, and clippy.

## Architecture

- `src/othello/` — game logic: `Position` (bitboards), `Board` (position + side), `Game` (WTHOR/PGN)
- `src/training/` — eval & training: `Features`, `Weights`, `Trainer`, `EdaxInterface`, `EvalCache`
- `test_data/` — sample files for self-contained tests

See [docs/reference.md](docs/reference.md) for detailed architecture and file formats.

## Guidelines

- **Prefer few dependencies.** Currently only `ctrlc`. Avoid pulling in crates for small tasks.
- **When asked a question, suggest solutions in a numbered list.** Point one out as recommended.
- **Consult source code as the main source of truth.** Docstrings may point you in the right direction, but the code is canonical. Never assume — verify.
- **When things are unclear or contradictory, ask immediately.** Don't guess.
- **Prefer commands that require little human approval.** Avoid `sed` or commands with pipes or loops unless it's by far the best solution.
- **Commit with concise one-line messages.**
- **Docs should be grouped and self-contained.** Do not repeat content — cross-link between sections/files instead.
- **The `ignored/` folder** is intentionally not in git. It holds weight files, cached Edax evaluations, and prompt files.
