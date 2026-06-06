//! Exact Othello position evaluation via alpha-beta search to game end.

use crate::othello::position::Position;
use crate::training::features::Features;
use crate::training::weights::Weights;

/// Exact score for `pos` from the perspective of the side to move.
///
/// Searches all legal move sequences to game end with alpha-beta pruning.
/// Handles terminal positions (game over) and passes (no legal moves)
/// directly, matching the semantics of the Edax evaluator.
///
/// The score is bounded to [-64, 64].
pub fn exact_score(pos: &Position) -> i32 {
    alphabeta_exact(pos, SCORE_MIN, SCORE_MAX)
}

/// Score bounds.
const SCORE_MIN: i32 = -64;
const SCORE_MAX: i32 = 64;

/// Negamax with alpha-beta pruning, searching to game end.
fn alphabeta_exact(pos: &Position, mut alpha: i32, beta: i32) -> i32 {
    let moves = pos.get_moves();
    if moves == 0 {
        let passed = pos.pass_move();
        if passed.get_moves() == 0 {
            return pos.final_score();
        }
        return -alphabeta_exact(&passed, -beta, -alpha);
    }

    let mut remaining = moves;
    while remaining != 0 {
        let cell = remaining.trailing_zeros();
        remaining &= remaining - 1;
        let child = pos.do_move(cell);
        let score = -alphabeta_exact(&child, -beta, -alpha);
        if score > alpha {
            alpha = score;
            if alpha >= beta {
                break;
            }
        }
    }

    alpha
}

/// Evaluate a batch of positions, returning one score per position
/// in the same order.  Handles game-end and pass positions without
/// invoking the full search.
pub fn batch_evaluate(positions: &[Position]) -> Vec<i32> {
    positions.iter().map(exact_score).collect()
}

/// Depth-limited evaluation for use in gameplay. Searches `depth` plies
/// and applies a heuristic at the leaves.
pub fn depth_limited_score(
    pos: &Position,
    depth: u32,
    weights: &Weights,
    features: &Features,
) -> i32 {
    alphabeta(pos, depth, weights, features, SCORE_MIN, SCORE_MAX)
}

/// Pick the best legal move for the side to move. Returns `None` when there
/// are no legal moves.
pub fn best_move(
    pos: &Position,
    depth: u32,
    exact_empties: u32,
    weights: &Weights,
    features: &Features,
) -> Option<u32> {
    let moves = pos.get_moves();
    if moves == 0 {
        return None;
    }

    if pos.empties() <= exact_empties {
        return best_move_exact(pos);
    }

    let mut alpha = SCORE_MIN;
    let mut best_cell = 0u32;

    let mut remaining = moves;
    while remaining != 0 {
        let cell = remaining.trailing_zeros();
        remaining &= remaining - 1;
        let child = pos.do_move(cell);
        let score = -alphabeta(
            &child,
            depth.saturating_sub(1),
            weights,
            features,
            -SCORE_MAX,
            -alpha,
        );
        if score > alpha {
            alpha = score;
            best_cell = cell;
        }
    }

    Some(best_cell)
}

/// Pick the best legal move using exact search to game end.
fn best_move_exact(pos: &Position) -> Option<u32> {
    let moves = pos.get_moves();
    if moves == 0 {
        return None;
    }

    let mut alpha = SCORE_MIN;
    let mut best_cell = 0u32;

    let mut remaining = moves;
    while remaining != 0 {
        let cell = remaining.trailing_zeros();
        remaining &= remaining - 1;
        let child = pos.do_move(cell);
        let score = -alphabeta_exact(&child, -SCORE_MAX, -alpha);
        if score > alpha {
            alpha = score;
            best_cell = cell;
        }
    }

    Some(best_cell)
}

/// Negamax with alpha-beta pruning and depth limit.
fn alphabeta(
    pos: &Position,
    depth: u32,
    weights: &Weights,
    features: &Features,
    mut alpha: i32,
    beta: i32,
) -> i32 {
    let moves = pos.get_moves();
    if moves == 0 {
        let passed = pos.pass_move();
        if passed.get_moves() == 0 {
            return pos.final_score();
        }
        return -alphabeta(&passed, depth, weights, features, -beta, -alpha);
    }

    if depth == 0 {
        return heuristic(pos, weights, features);
    }

    let mut remaining = moves;
    while remaining != 0 {
        let cell = remaining.trailing_zeros();
        remaining &= remaining - 1;
        let child = pos.do_move(cell);
        let score = -alphabeta(&child, depth - 1, weights, features, -beta, -alpha);
        if score > alpha {
            alpha = score;
            if alpha >= beta {
                break;
            }
        }
    }

    alpha
}

fn heuristic(pos: &Position, weights: &Weights, features: &Features) -> i32 {
    let score = weights.evaluate(pos, features);
    score.round().clamp(SCORE_MIN as f32, SCORE_MAX as f32) as i32
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

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

    /// Verify alpha-beta exact scores match the Edax reference scores.
    ///
    /// Reads `test_data/exact_scores.txt` (generated by `select_reference.py`
    /// from the Edax eval cache), parses each FEN into a [`Position`], computes
    /// [`exact_score`], and asserts it equals the saved Edax score.
    #[test]
    fn test_exact_scores_match_reference() {
        let path = "test_data/exact_scores.txt";
        let content = fs::read_to_string(path).expect("Failed to read reference file");

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
            let actual = exact_score(&pos);

            assert_eq!(
                actual,
                expected,
                "Line {}: FEN={fen} expected={expected} actual={actual}",
                line_no + 1,
            );
        }
    }

    #[test]
    fn test_depth_limited_score_game_end() {
        let pos = Position {
            player: u64::MAX,
            opponent: 0,
        };
        let features = Features::edax();
        let weights = Weights::new(features.clone());
        assert_eq!(depth_limited_score(&pos, 0, &weights, &features), 64);
    }

    #[test]
    fn test_best_move_uses_exact_for_few_empties() {
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
        let features = Features::edax();
        let weights = Weights::new(features.clone());
        let mv = best_move(&pos, 1, 12, &weights, &features);
        assert!(mv.is_some(), "best_move should return a move with 1 empty");
    }

    #[test]
    fn test_depth_limited_score_bounded() {
        let pos = Position::initial();
        let features = Features::edax();
        let weights = Weights::new(features.clone());
        let score = depth_limited_score(&pos, 4, &weights, &features);
        assert!(
            (SCORE_MIN..=SCORE_MAX).contains(&score),
            "score {score} out of bounds"
        );
    }

    #[test]
    fn test_best_move_returns_legal_move() {
        let pos = Position::initial();
        let features = Features::edax();
        let weights = Weights::new(features.clone());
        let mv = best_move(&pos, 4, 12, &weights, &features);
        assert!(mv.is_some());
        let cell = mv.unwrap_or_else(|| unreachable!());
        let moves = pos.get_moves();
        assert!(
            moves & (1u64 << cell) != 0,
            "best_move returned illegal cell {cell}"
        );
    }

    #[test]
    fn test_best_move_none_when_no_moves() {
        let pos = Position {
            player: u64::MAX,
            opponent: 0,
        };
        let features = Features::edax();
        let weights = Weights::new(features.clone());
        assert!(best_move(&pos, 4, 12, &weights, &features).is_none());
    }

    #[test]
    fn test_heuristic_bounded() {
        let pos = Position::initial();
        let features = Features::edax();
        let weights = Weights::new(features.clone());
        let h = heuristic(&pos, &weights, &features);
        assert!(
            (SCORE_MIN..=SCORE_MAX).contains(&h),
            "heuristic {h} out of bounds"
        );
    }

    /// Parse an Edax FEN (66 chars: 64 board + space + side-to-move) into a
    /// [`Position`] where `player` is the side to move.
    fn parse_fen(fen: &str) -> Position {
        let board = fen.as_bytes();
        let side = board[65]; // 'X' or 'O'

        let mut x_discs: u64 = 0; // black
        let mut o_discs: u64 = 0; // white

        for i in 0..64 {
            match board[i] {
                b'X' => x_discs |= 1u64 << i,
                b'O' => o_discs |= 1u64 << i,
                b'-' => { /* empty */ }
                _ => panic!(
                    "Invalid FEN character at position {i}: {}",
                    board[i] as char
                ),
            }
        }

        if side == b'X' {
            // Black (X) to move → player = black discs
            Position {
                player: x_discs,
                opponent: o_discs,
            }
        } else {
            // White (O) to move → player = white discs
            Position {
                player: o_discs,
                opponent: x_discs,
            }
        }
    }
}
