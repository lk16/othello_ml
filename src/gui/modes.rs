//! The three GUI modes: `game`, `evaluate`, `pgn`. Ported from flippy's
//! `mode/{game,evaluate,pgn}.py`, but scored by this crate's own search.

use std::sync::Arc;

use macroquad::prelude::KeyCode;

use super::score::{graph_scores, score_moves, AsyncJob, GraphPoint};
use super::{start_board, Eval, Mode, UiDetails};
use crate::othello::board::Board;
use crate::othello::position::Position;
use crate::training::weights::Weights;
use crate::{FlatEval, Solver};

type MoveJob = (u64, Position);
type MoveOut = (u64, Vec<(u32, i32)>);

/// Apply `cell` to `board`, auto-passing if the opponent has no reply but the
/// game isn't over. Returns `None` for an illegal move.
fn play(board: &Board, cell: u32) -> Option<Board> {
    if board.position.get_moves() & (1u64 << cell) == 0 {
        return None;
    }
    let mut child = Board {
        position: board.position.do_move(cell),
        black_to_move: !board.black_to_move,
    };
    if !child.position.has_moves() {
        let passed = Board {
            position: child.position.pass_move(),
            black_to_move: !child.black_to_move,
        };
        if passed.position.has_moves() {
            child = passed;
        }
    }
    Some(child)
}

// ---------------------------------------------------------------------------
// game
// ---------------------------------------------------------------------------

/// Local hot-seat play: click to move, right-click to undo, click after the
/// game ends to restart.
pub struct GameMode {
    history: Vec<Board>,
}

impl GameMode {
    pub fn new() -> Self {
        GameMode {
            history: vec![start_board()],
        }
    }
}

impl Mode for GameMode {
    fn on_left_click(&mut self, cell: u32) {
        let board = self.board();
        if board.position.is_game_end() {
            self.history = vec![start_board()];
            return;
        }
        if let Some(child) = play(&board, cell) {
            self.history.push(child);
        }
    }

    fn on_right_click(&mut self) {
        if self.history.len() > 1 {
            self.history.pop();
        }
    }

    fn board(&self) -> Board {
        self.history.last().cloned().unwrap_or_else(start_board)
    }
}

// ---------------------------------------------------------------------------
// evaluate
// ---------------------------------------------------------------------------

/// `game` plus per-move scores from our own search, with the best move ringed.
pub struct EvaluateMode {
    game: GameMode,
    engine: AsyncJob<MoveJob, MoveOut>,
    gen: u64,
    last_pos: Option<Position>,
    scores: Vec<(u32, i32)>,
}

impl EvaluateMode {
    pub fn new(weights: Arc<Weights>, depth: u32, exact_empties: u32) -> Self {
        let mut solver = Solver::with_eval(Arc::new(FlatEval::from_weights(&weights)));
        let engine = AsyncJob::new(move |(gen, pos): MoveJob| {
            (gen, score_moves(&pos, depth, exact_empties, &mut solver))
        });
        EvaluateMode {
            game: GameMode::new(),
            engine,
            gen: 0,
            last_pos: None,
            scores: Vec::new(),
        }
    }
}

impl Mode for EvaluateMode {
    fn on_left_click(&mut self, cell: u32) {
        self.game.on_left_click(cell);
    }

    fn on_right_click(&mut self) {
        self.game.on_right_click();
    }

    fn tick(&mut self) {
        let pos = self.game.board().position;
        if self.last_pos != Some(pos) {
            self.last_pos = Some(pos);
            self.gen += 1;
            self.scores.clear();
            self.engine.submit((self.gen, pos));
        }
        while let Some((gen, scores)) = self.engine.poll() {
            if gen == self.gen {
                self.scores = scores;
            }
        }
    }

    fn board(&self) -> Board {
        self.game.board()
    }

    fn ui(&self) -> UiDetails {
        UiDetails {
            evaluations: self
                .scores
                .iter()
                .map(|&(cell, score)| Eval {
                    cell,
                    score,
                    level: None,
                })
                .collect(),
            ..Default::default()
        }
    }
}

// ---------------------------------------------------------------------------
// pgn
// ---------------------------------------------------------------------------

type GraphJob = (u64, Vec<Board>);
type GraphOut = (u64, Vec<GraphPoint>);

/// Review a loaded game: ←/→ (or right-click) navigate, click a square to
/// explore an alternative line, with a black-POV score graph along the bottom.
/// Keys: `space` show all moves, `l` show depth label, `f` flip the board.
pub struct PgnMode {
    boards: Vec<Board>,
    index: usize,
    /// Alternative-line stack; when non-empty we're off the mainline.
    alt: Vec<Board>,
    move_engine: AsyncJob<MoveJob, MoveOut>,
    graph_engine: AsyncJob<GraphJob, GraphOut>,
    gen: u64,
    last_pos: Option<Position>,
    scores: Vec<(u32, i32)>,
    graph_gen: u64,
    graph: Vec<GraphPoint>,
    show_all: bool,
    show_level: bool,
    depth: u32,
}

impl PgnMode {
    pub fn new(game: crate::Game, weights: Arc<Weights>, depth: u32, exact_empties: u32) -> Self {
        // One shared flat eval; each engine thread owns its own solver (and TT).
        let eval = Arc::new(FlatEval::from_weights(&weights));
        let mut move_solver = Solver::with_eval(Arc::clone(&eval));
        let move_engine = AsyncJob::new(move |(gen, pos): MoveJob| {
            (
                gen,
                score_moves(&pos, depth, exact_empties, &mut move_solver),
            )
        });
        let mut graph_solver = Solver::with_eval(eval);
        let graph_engine = AsyncJob::new(move |(gen, boards): GraphJob| {
            (
                gen,
                graph_scores(&boards, depth, exact_empties, &mut graph_solver),
            )
        });

        let mut mode = PgnMode {
            boards: game.positions,
            index: 0,
            alt: Vec::new(),
            move_engine,
            graph_engine,
            gen: 0,
            last_pos: None,
            scores: Vec::new(),
            graph_gen: 0,
            graph: Vec::new(),
            show_all: false,
            show_level: false,
            depth,
        };
        mode.submit_graph();
        mode
    }

    fn submit_graph(&mut self) {
        self.graph_gen += 1;
        self.graph_engine
            .submit((self.graph_gen, self.boards.clone()));
    }

    fn next(&mut self) {
        if !self.alt.is_empty() {
            return;
        }
        if self.index + 1 < self.boards.len() {
            self.index += 1;
        }
    }

    fn prev(&mut self) {
        if !self.alt.is_empty() {
            self.alt.pop();
            return;
        }
        if self.index > 0 {
            self.index -= 1;
        }
    }

    /// The move played in the loaded game at the current mainline board, if any.
    fn played_move(&self) -> Option<u32> {
        if !self.alt.is_empty() {
            return None;
        }
        let cur = self.boards.get(self.index)?;
        let next = self.boards.get(self.index + 1)?;
        let mut remaining = cur.position.get_moves();
        while remaining != 0 {
            let cell = remaining.trailing_zeros();
            remaining &= remaining - 1;
            if cur.position.do_move(cell) == next.position {
                return Some(cell);
            }
        }
        None
    }

    /// Point-reflect every board (flippy's `f`: replace each move with 63-move).
    fn flip(&mut self) {
        for board in &mut self.boards {
            board.position = Position {
                player: board.position.player.reverse_bits(),
                opponent: board.position.opponent.reverse_bits(),
            };
        }
        self.alt.clear();
        self.submit_graph();
    }
}

impl Mode for PgnMode {
    fn on_left_click(&mut self, cell: u32) {
        let board = self.board();
        let Some(child) = play(&board, cell) else {
            return;
        };
        // If this is the move actually played next in the game (and we're on
        // the mainline), just advance instead of branching.
        let next = self.boards.get(self.index + 1);
        let on_mainline = self.alt.is_empty()
            && next.is_some_and(|n| {
                n.position == child.position && n.black_to_move == child.black_to_move
            });
        if on_mainline {
            self.index += 1;
        } else {
            self.alt.push(child);
        }
    }

    fn on_right_click(&mut self) {
        self.prev();
    }

    fn on_key(&mut self, key: KeyCode) {
        match key {
            KeyCode::Right => self.next(),
            KeyCode::Left => self.prev(),
            KeyCode::Space => self.show_all = !self.show_all,
            KeyCode::L => self.show_level = !self.show_level,
            KeyCode::F => self.flip(),
            _ => {}
        }
    }

    fn tick(&mut self) {
        let pos = self.board().position;
        if self.last_pos != Some(pos) {
            self.last_pos = Some(pos);
            self.gen += 1;
            self.scores.clear();
            self.move_engine.submit((self.gen, pos));
        }
        while let Some((gen, scores)) = self.move_engine.poll() {
            if gen == self.gen {
                self.scores = scores;
            }
        }
        while let Some((gen, graph)) = self.graph_engine.poll() {
            if gen == self.graph_gen {
                self.graph = graph;
            }
        }
    }

    fn board(&self) -> Board {
        if let Some(board) = self.alt.last() {
            return board.clone();
        }
        self.boards
            .get(self.index)
            .cloned()
            .unwrap_or_else(start_board)
    }

    fn ui(&self) -> UiDetails {
        let played = self.played_move();
        let level = if self.show_level {
            Some(self.depth)
        } else {
            None
        };
        let mut evaluations: Vec<Eval> = self
            .scores
            .iter()
            .map(|&(cell, score)| Eval { cell, score, level })
            .collect();

        // On the mainline (no alt line), show only the best move(s) plus the
        // one actually played, unless `space` toggled show-all.
        if self.alt.is_empty() && !self.show_all && !evaluations.is_empty() {
            if let Some(best) = evaluations.iter().map(|e| e.score).max() {
                evaluations.retain(|e| e.score == best || Some(e.cell) == played);
            }
        }

        UiDetails {
            evaluations,
            played_move: played,
            graph: self.graph.clone(),
            graph_current: if self.alt.is_empty() {
                Some(self.index)
            } else {
                None
            },
        }
    }
}
