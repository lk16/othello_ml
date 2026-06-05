# Best Practices

Conventions established during the restructuring of this project.

## Code quality

- **Minimal dependencies.** Currently only `ctrlc`. Avoid pulling in crates for small tasks.
- **Concise docstrings** — specific to the function or module, not redundant with the name. Module-level docs cover cross-cutting concerns.
- **Avoid long functions and deep nesting.** When a function exceeds ~60 lines or takes 5+ parameters, extract helpers or introduce a config struct.
- **Avoid wrapper structs** that only delegate to another type (e.g. `WeightIO`). Put the functions directly on the owning struct.
- **Table-driven tests** where a function has many input/output pairs.

## Commands

- **Always run `pre-commit run -a`**, not `cargo check` directly. It runs check, fmt, and clippy together.
- **Prefer commands that require little human approval.** Avoid `sed` or commands with pipes or loops unless it's by far the best solution.

## Threading

- **Channel-based progress.** Worker threads send progress updates via `mpsc::Sender`; the parent thread aggregates and prints. This avoids glitchy interleaved output.

## Testing

- **Self-contained tests** — don't rely on files outside the repo. Add small sample files to `test_data/` when needed.

## Documentation

- **Cross-link between docs.** Group related content in one file and link from others rather than repeating.
- **`ignored/` is not tracked.** It holds weight files, cached Edax evaluations, and other local-state files.
- **CLAUDE.md** loaded every session — keep it concise with build commands, architecture overview, and these guidelines.

## Project conventions

- Commit with concise one-line messages.
- When asked a question, suggest solutions in a numbered list and point out a recommended one.
- Consult source code as the main source of truth. Docstrings help, but the code is canonical. Never assume — verify.
- When things are unclear or contradictory, ask immediately — don't guess.
