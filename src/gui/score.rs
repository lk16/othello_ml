//! Background scoring for the `evaluate` and `pgn` GUI modes.
//!
//! flippy gets per-move scores from Edax plus a remote opening book; we have
//! neither, so scores come from this crate's own evaluator: exact alpha-beta
//! ([`Solver`]) once a position is shallow enough, otherwise a depth-limited
//! search with the trained [`FlatEval`]-style heuristic ([`depth_limited_score`]).
//! All work runs on a worker thread ([`AsyncJob`]) so the 60 fps UI never blocks.

use std::sync::mpsc::{self, Receiver, Sender};
use std::thread;

use crate::othello::board::Board;
use crate::othello::position::Position;
use crate::Solver;

/// Score every legal move of `pos`, from the side-to-move's perspective
/// (positive = good for the player to move), matching how flippy displays it.
///
/// `solver` must carry an eval ([`Solver::with_eval`]): non-endgame children are
/// scored by the fast eval-seeded ordered search ([`Solver::heuristic_score`]), and
/// reusing one solver across the root moves warms its shared transposition table.
pub fn score_moves(
    pos: &Position,
    depth: u32,
    exact_empties: u32,
    solver: &mut Solver,
) -> Vec<(u32, i32)> {
    let mut out = Vec::new();
    let mut remaining = pos.get_moves();
    while remaining != 0 {
        let cell = remaining.trailing_zeros();
        remaining &= remaining - 1;
        let child = pos.do_move(cell);
        // `child` is from the opponent's perspective; negate to get ours.
        let child_score = if child.empties() <= exact_empties {
            solver.exact_score(&child)
        } else {
            solver.heuristic_score(&child, depth.saturating_sub(1))
        };
        out.push((cell, -child_score));
    }
    out
}

/// Per-board graph point: `(black_to_move, score_from_black_pov)`, or `None`
/// when the board has no legal moves (a pass / game end), mirroring flippy's
/// black-POV evaluation graph.
pub type GraphPoint = Option<(bool, i32)>;

/// Score a whole game's mainline for the bottom evaluation graph.
pub fn graph_scores(
    boards: &[Board],
    depth: u32,
    exact_empties: u32,
    solver: &mut Solver,
) -> Vec<GraphPoint> {
    boards
        .iter()
        .map(|board| {
            if !board.position.has_moves() {
                return None;
            }
            let best = score_moves(&board.position, depth, exact_empties, solver)
                .into_iter()
                .map(|(_, s)| s)
                .max()?;
            // `best` is from the side-to-move's POV; convert to black's POV.
            let black_score = if board.black_to_move { best } else { -best };
            Some((board.black_to_move, black_score))
        })
        .collect()
}

/// A single worker thread running `f` over submitted jobs. Only the most
/// recent pending job is processed (stale jobs are dropped), so rapid board
/// changes never pile up behind an old computation.
pub struct AsyncJob<I, O> {
    tx: Sender<I>,
    rx: Receiver<O>,
}

impl<I: Send + 'static, O: Send + 'static> AsyncJob<I, O> {
    pub fn new<F: FnMut(I) -> O + Send + 'static>(mut f: F) -> Self {
        let (in_tx, in_rx) = mpsc::channel::<I>();
        let (out_tx, out_rx) = mpsc::channel::<O>();
        thread::spawn(move || {
            while let Ok(mut job) = in_rx.recv() {
                // Drop any older queued jobs; keep only the latest.
                while let Ok(next) = in_rx.try_recv() {
                    job = next;
                }
                if out_tx.send(f(job)).is_err() {
                    break;
                }
            }
        });
        AsyncJob {
            tx: in_tx,
            rx: out_rx,
        }
    }

    pub fn submit(&self, job: I) {
        let _ = self.tx.send(job);
    }

    pub fn poll(&self) -> Option<O> {
        self.rx.try_recv().ok()
    }
}
