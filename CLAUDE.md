# CLAUDE.md

Othello ML — a Rust feature-based Othello position evaluator trained via conjugate-gradient least-squares against exact ground truth.

## Build & test

```bash
pre-commit run -a && cargo test
```

Always use `pre-commit run -a` — not `cargo check` directly. It runs check, fmt, and clippy.

## Architecture

- `src/othello/` — game logic: `Position` (bitboards), `Board` (position + side), `Game` (WTHOR/PGN)
- `src/eval/` — exact evaluation: `alphabeta` (negamax), `cache` (FEN→score persistence)
- `src/training/` — training: `Features`, `Weights`, `cg` (CG least-squares trainer)
- `test_data/` — sample files for self-contained tests

See [docs/reference.md](docs/reference.md) for detailed architecture and file formats.
See [docs/best-practices.md](docs/best-practices.md) for project conventions.
See [docs/speedup-plan.md](docs/speedup-plan.md) for the exact-search optimization roadmap.
See [docs/eval-quality.md](docs/eval-quality.md) for the eval-accuracy problem — the current critical path blocking the biggest remaining solver speedups.

## Guidelines

- **Prefer few dependencies.** Currently only `ctrlc`. Avoid pulling in crates for small tasks.
- **When asked a question, suggest solutions in a numbered list.** Point one out as recommended.
- **Consult source code as the main source of truth.** Docstrings may point you in the right direction, but the code is canonical. Never assume — verify.
- **When things are unclear or contradictory, ask immediately.** Don't guess.
- **Prefer commands that require little human approval.** Avoid `sed` or commands with pipes or loops unless it's by far the best solution.
- **Commit with concise one-line messages.**
- **Docs should be grouped and self-contained.** Do not repeat content — cross-link between sections/files instead.
- **The `ignored/` folder** is intentionally not in git. It holds weight files, cached evaluations, and prompt files.
