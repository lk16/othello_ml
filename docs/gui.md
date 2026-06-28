# GUI

A minimal graphical board, ported from [flippy](https://github.com/lk16/flippy)'s
pygame GUI. Built on [macroquad](https://crates.io/crates/macroquad) (pure Rust,
no system libraries). Code lives in `src/gui/`.

## Usage

```bash
othello_eval gui <MODE> [OPTIONS]
```

The mode is the **second positional argument**. Each mode only accepts the flags
it actually uses:

| Mode | What it does | Required flags |
| --- | --- | --- |
| `game` | Play both sides locally. Left-click = move, right-click = undo, click after game over = restart. | — |
| `evaluate` | `game` plus a score on every legal move; the best move is ringed. | `-w/--weights` (defaults to `trained_weights.bin`) |
| `pgn` | Step through a loaded game with a bottom score graph (black's POV). | `-p/--pgn`, `-w/--weights` |
| `play` | Play against the AI. Left-click = your move (ignored while the AI thinks), right-click = undo back to your previous move, click after game over = restart. At game over it shows the final disc counts and who won (on the board and once on stderr). | `-w/--weights` (defaults to `trained_weights.bin`) |

Shared scoring flags (`evaluate`/`pgn`/`play`): `--depth N` (heuristic search depth,
default 6) and `--exact-empties N` (switch to exact search at ≤ N empties,
default 12). `play` adds `-c/--color black|white` (the colour you play; black moves
first, default black). The AI replies on a worker thread, so the UI never blocks
while it searches.

`pgn` keys: ←/→ or right-click navigate, click a square to branch into an
alternative line, `space` = show all move scores (not just the best), `l` = show
the search depth, `f` = flip the board 180°.

```bash
othello_eval gui game
othello_eval gui evaluate -w weights.bin
othello_eval gui pgn -w weights.bin -p game.pgn
othello_eval gui play -w weights.bin -c white -d 8
```

## How it differs from flippy

flippy sources its evaluations from **Edax** and a **remote opening-book / position
database (HTTP API)**. This project has **neither**:

- All per-move scores come from this crate's own search — exact alpha-beta
  ([`Solver`]) once a position is shallow enough, otherwise a depth-limited
  search with the trained pattern eval. A trained weights file is therefore
  required for `evaluate`, `pgn`, and `play`.
- "Level" in flippy is Edax's search level; here it is just our search depth.
- The flippy `frequency` and `watch` modes (which depend on that database) are
  **not implemented**.

An Edax-style backend and a shared position database may be added later; until
then everything is local and self-contained.

## Fonts

On-board text uses a bundled sans-serif (`src/gui/LiberationSans-Regular.ttf`,
SIL OFL 1.1 — see the adjacent `.LICENSE.txt`) instead of macroquad's built-in
pixel font.
