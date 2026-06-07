# Best Practices

Conventions established during the restructuring of this project.
Each item should be descriptive enough that specific examples are unnecessary.

## Code quality

- **Minimal dependencies.** Currently only `ctrlc`. Avoid pulling in crates for small tasks.
- **Never panic.** Avoid `panic!()`, `expect()`, `unwrap()` (outside tests), and `std::process::exit()`. Return `Result` and let the caller handle errors.
- **Concise docstrings** — specific to the function or module, not redundant with the name. Module-level docs cover cross-cutting concerns.
- **Avoid long functions and deep nesting.** When a function exceeds ~60 lines or takes 5+ parameters, extract helpers or introduce a config struct.
- **Prefer `if let Some(x) = opt { ... } else { return ... }`** over `match opt { Some(x) => ..., None => return ... }` when the `None` branch returns, breaks, or continues. This keeps the happy path indented and the early-exit visible. Same for `if let Ok(x)` vs `match` on `Result`.
- **Prefer iterator combinators** (`iter().filter().map().collect()`) over manual `for` loops that accumulate into a collection. The intent is clearer and there's less mutable state.
- **Prefer associated functions** on structs over free functions that take the struct as their first argument.
- **Prefer `Option` or `Result`** over including an error or sentinel variant in an enum.
- **Avoid wrapper structs** that only delegate to another type (e.g. `WeightIO`). Put the functions directly on the owning struct.

## Commands

- **Always run `pre-commit run -a`**, not `cargo check` directly. It runs check, fmt, and clippy together.
- **Prefer commands that require little human approval.** Avoid `sed` or commands with pipes or loops unless it's by far the best solution.
- **Bound long-running commands with a timeout.** Anything whose runtime is uncertain (benchmarks, searches, exact evaluation) should be wrapped in `timeout <secs> …` and/or given an explicit tool-level timeout, so a bad estimate or a hang can't run unbounded. Estimate the cost first (a small calibration run) before launching the full job.
- **The `bench` subcommand reloads all input files on every invocation.** Loading is outside the timed loop but still counts against wall-clock; pass the fewest files that supply enough boards, and prefer one larger file over many small ones when sweeping multiple `--empties` values.

## Threading

- **Channel-based progress.** Worker threads send progress updates via `mpsc::Sender`; the parent thread aggregates and prints. This avoids glitchy interleaved output.

## Testing

- **Add tests when adding or modifying functions.** Correctness is critical — every new `pub fn` or behavior change should have at least a basic test.
- Use **Table-driven tests** where appropriate: a function has many input/output pairs.
- **Self-contained tests** — don't rely on files outside the repo. Add small sample files to `test_data/` when needed.
- **No randomness in tests.** Don't drive tests with a PRNG (even a seeded one) — failures become noise that depends on the seed and the iteration count. For broad coverage, enumerate a deterministic, fixed set of inputs instead: cross every interesting position parameter (e.g. each empty-square index) with a small table of hand-chosen bit patterns. This is reproducible and the failing case is always the same.
- **Cross-check fast paths against a simple reference.** When optimizing a function (e.g. the `solve_1`/`solve_2` endgame leaf solvers), keep an obviously-correct, unoptimized implementation in the test module (e.g. `naive_exact`) and assert the two agree over the enumerated inputs.

## Documentation

- **Cross-link between docs.** Group related content in one file and link from others rather than repeating.
- **`ignored/` is not tracked.** It holds weight files, cached evaluation files, and other local-state files.
- **CLAUDE.md** loaded every session — keep it concise with build commands, architecture overview, and these guidelines.

## Project conventions

- Commit with concise one-line messages.
- **Never commit without asking for review first.** Always show the user what will be committed and wait for approval before running `git commit`.
- When asked a question, suggest solutions in a numbered list and point out a recommended one.
- Consult source code as the main source of truth. Docstrings help, but the code is canonical. Never assume — verify.
- When things are unclear or contradictory, ask immediately — don't guess.
