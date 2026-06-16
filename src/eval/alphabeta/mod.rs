//! Exact Othello position evaluation via alpha-beta search to game end.
//!
//! Submodules: [`search`] (the PVS/null-window core and `Search` state),
//! [`leaf`] (last-few-empties solvers), [`stability`] (the stability cutoff
//! estimate), [`tt`] (transposition table), [`parallel`] (the root-level YBWC
//! parallel solver), and [`depth`] (the depth-limited heuristic search used for
//! gameplay).

mod count_flip;
mod depth;
mod leaf;
mod parallel;
mod search;
mod stability;
mod tt;

#[cfg(test)]
mod testutil;

pub use count_flip::bench_count_flip_variants;
pub use depth::{best_move, depth_limited_score};
pub use parallel::ParallelSolver;

use crate::othello::position::Position;
use search::{board_parity, Search};

/// Score bounds.
pub(crate) const SCORE_MIN: i32 = -64;
pub(crate) const SCORE_MAX: i32 = 64;

/// A/B switch (Step 31): solve the root via MTD / aspiration ([`Search::solve_mtd`],
/// repeated null-window probes converging on the exact score) instead of one
/// full-window search. Same exact result; node count differs. Sequential `Solver`
/// path only. **Off**: guess-free bisection measured net-neutral on 24e (−0.1%
/// nodes — helps on extreme/near-zero scores, hurts on moderate ones) because the
/// ~2× window-narrowing ceiling needs a *good score estimate*, which needs an
/// evaluation function we don't yet have. Kept as the scaffold for eval-seeded
/// MTD-f: once a usable eval exists, feed its estimate as the first guess. See the
/// Edax-comparison section in docs/speedup-plan.md.
const USE_MTD: bool = false;

/// Exact score for `pos` from the side-to-move's perspective, in `[-64, 64]`.
///
/// A one-shot wrapper that allocates a fresh transposition table per call;
/// callers evaluating many positions should reuse a single [`Solver`].
pub fn exact_score(pos: &Position) -> i32 {
    Solver::new().exact_score(pos)
}

/// Exact score together with the number of search nodes visited (for `bench`).
pub fn exact_score_with_nodes(pos: &Position) -> (i32, u64) {
    Solver::new().exact_score_with_nodes(pos)
}

/// Evaluate a batch of positions, one score per position in input order.
pub fn batch_evaluate(positions: &[Position]) -> Vec<i32> {
    let mut solver = Solver::new();
    positions.iter().map(|p| solver.exact_score(p)).collect()
}

/// Reusable exact solver owning a transposition table shared across the
/// positions it evaluates. Exact scores are position-intrinsic, so cross-position
/// reuse is sound and warms the table. Construct once, evaluate many — looping
/// callers hold a single `Solver` so the multi-MB table is allocated only once.
pub struct Solver {
    search: Search,
}

impl Solver {
    /// Allocate a solver with a fresh (empty) transposition table.
    pub fn new() -> Self {
        Solver {
            search: Search::new(),
        }
    }

    /// Exact score for `pos`, reusing this solver's transposition table.
    pub fn exact_score(&mut self, pos: &Position) -> i32 {
        self.search.nodes = 0;
        self.search.parity = board_parity(pos);
        if USE_MTD {
            self.search.solve_mtd(pos, pos.empties())
        } else {
            self.search
                .search_exact(pos, SCORE_MIN, SCORE_MAX, pos.empties())
        }
    }

    /// Exact score plus the number of search nodes visited for this position.
    pub fn exact_score_with_nodes(&mut self, pos: &Position) -> (i32, u64) {
        self.search.nodes = 0;
        self.search.parity = board_parity(pos);
        let score = if USE_MTD {
            self.search.solve_mtd(pos, pos.empties())
        } else {
            self.search
                .search_exact(pos, SCORE_MIN, SCORE_MAX, pos.empties())
        };
        (score, self.search.nodes)
    }
}

impl Default for Solver {
    fn default() -> Self {
        Solver::new()
    }
}

#[cfg(test)]
mod tests {
    use super::testutil::{
        layouts_for, naive_exact, two_empty_layouts, PATTERNS, SQUARES, SQUARES4,
    };
    use super::*;
    use std::fs;

    #[test]
    fn solve_1_and_solve_2_drive_exact_score() {
        // exact_score (dispatching to the 1/2-empty leaf solvers) agrees with the
        // naive solver. One reused solver so the table is allocated once.
        let mut solver = Solver::new();
        for &sq in SQUARES {
            let empty = 1u64 << sq;
            for &pat in PATTERNS {
                let player = pat & !empty;
                let opponent = !player & !empty;
                let pos = Position { player, opponent };
                assert_eq!(solver.exact_score(&pos), naive_exact(&pos));
            }
        }
        for (player, opponent) in two_empty_layouts() {
            let pos = Position { player, opponent };
            assert_eq!(solver.exact_score(&pos), naive_exact(&pos));
        }
    }

    #[test]
    fn solve_3_and_4_drive_exact_score() {
        let mut solver = Solver::new();
        let n = SQUARES4.len();
        for i in 0..n {
            for j in (i + 1)..n {
                for k in (j + 1)..n {
                    for (player, opponent) in layouts_for(&[SQUARES4[i], SQUARES4[j], SQUARES4[k]])
                    {
                        let pos = Position { player, opponent };
                        assert_eq!(solver.exact_score(&pos), naive_exact(&pos));
                    }
                    for l in (k + 1)..n {
                        for (player, opponent) in
                            layouts_for(&[SQUARES4[i], SQUARES4[j], SQUARES4[k], SQUARES4[l]])
                        {
                            let pos = Position { player, opponent };
                            assert_eq!(solver.exact_score(&pos), naive_exact(&pos));
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn test_exact_score_game_end_full() {
        let pos = Position {
            player: u64::MAX,
            opponent: 0,
        };
        assert!(pos.is_game_end());
        assert_eq!(exact_score(&pos), 64);
    }

    #[test]
    fn test_exact_score_game_end_tie() {
        let pos = Position::new();
        assert!(pos.is_game_end());
        assert_eq!(exact_score(&pos), 0);
    }

    #[test]
    fn test_exact_score_game_end_opponent_wins() {
        let pos = Position {
            player: 0,
            opponent: u64::MAX,
        };
        assert!(pos.is_game_end());
        assert_eq!(exact_score(&pos), -64);
    }

    #[test]
    fn test_exact_score_one_empty() {
        let mut player: u64 = 0;
        let mut opponent: u64 = 0;
        for i in 0..32 {
            player |= 1u64 << i;
        }
        for i in 32..63 {
            opponent |= 1u64 << i;
        }
        let pos = Position { player, opponent };
        assert_eq!(pos.empties(), 1);
        let score = exact_score(&pos);
        assert!(score > 0, "black should win, got {score}");
    }

    #[test]
    fn test_batch_evaluate_game_ends() {
        let positions = vec![
            Position {
                player: u64::MAX,
                opponent: 0,
            },
            Position::new(),
        ];
        let scores = batch_evaluate(&positions);
        assert_eq!(scores, vec![64, 0]);
    }

    /// Exact scores must match the Edax reference cache. Reads
    /// `test_data/exact_scores.txt` (FEN + score per line) and checks each.
    #[test]
    fn test_exact_scores_match_reference() {
        let path = "test_data/exact_scores.txt";
        let content = fs::read_to_string(path).expect("Failed to read reference file");

        // One reused solver also checks the shared (warming) table stays correct.
        let mut solver = Solver::new();
        for (line_no, line) in content.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let fen = &line[..66];
            let expected: i32 = line[67..]
                .trim()
                .parse()
                .unwrap_or_else(|_| panic!("Line {}: invalid score", line_no + 1));

            let pos = parse_fen(fen);
            let actual = solver.exact_score(&pos);

            assert_eq!(
                actual,
                expected,
                "Line {}: FEN={fen} expected={expected} actual={actual}",
                line_no + 1,
            );
        }
    }

    /// Parse an Edax FEN (66 chars: 64 board + space + side-to-move) into a
    /// [`Position`] whose `player` is the side to move.
    fn parse_fen(fen: &str) -> Position {
        let board = fen.as_bytes();
        let side = board[65]; // 'X' or 'O'

        let mut x_discs: u64 = 0; // black
        let mut o_discs: u64 = 0; // white

        for i in 0..64 {
            match board[i] {
                b'X' => x_discs |= 1u64 << i,
                b'O' => o_discs |= 1u64 << i,
                b'-' => {}
                _ => panic!(
                    "Invalid FEN character at position {i}: {}",
                    board[i] as char
                ),
            }
        }

        if side == b'X' {
            Position {
                player: x_discs,
                opponent: o_discs,
            }
        } else {
            Position {
                player: o_discs,
                opponent: x_discs,
            }
        }
    }
}
