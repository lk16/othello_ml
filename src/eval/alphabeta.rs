//! Exact Othello position evaluation via alpha-beta search to game end.

use crate::othello::position::Position;

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
    // Terminal position: final score from current player's perspective.
    if pos.is_game_end() {
        return pos.final_score();
    }

    // Pass: no legal moves for the current player.
    if !pos.has_moves() {
        return -alphabeta_exact(&pos.pass_move(), -beta, -alpha);
    }

    let moves = pos.get_moves();
    for cell in 0..64 {
        let bit = 1u64 << cell;
        if moves & bit == 0 {
            continue;
        }
        let child = make_move(pos, cell);
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

/// Apply a move via [`Position::do_move`], which places a disc, flips
/// opponent discs, and swaps sides.
fn make_move(pos: &Position, cell: u32) -> Position {
    pos.do_move(cell)
}

/// Evaluate a batch of positions, returning one score per position
/// in the same order.  Handles game-end and pass positions without
/// invoking the full search.
pub fn batch_evaluate(positions: &[Position]) -> Vec<i32> {
    positions.iter().map(exact_score).collect()
}

/// Depth-limited evaluation for use in gameplay. Searches `depth` plies
/// and applies a heuristic at the leaves.
pub fn depth_limited_score(pos: &Position, depth: u32) -> i32 {
    alphabeta(pos, depth, SCORE_MIN, SCORE_MAX)
}

/// Pick the best legal move for the side to move. Returns `None` when there
/// are no legal moves.
pub fn best_move(pos: &Position, depth: u32, exact_empties: u32) -> Option<u32> {
    let moves = pos.get_moves();
    if moves == 0 {
        return None;
    }

    if pos.empties() <= exact_empties {
        return best_move_exact(pos);
    }

    let mut alpha = SCORE_MIN;
    let mut best_cell = 0u32;

    for cell in 0..64 {
        if moves & (1u64 << cell) == 0 {
            continue;
        }
        let child = pos.do_move(cell);
        let score = -alphabeta(&child, depth.saturating_sub(1), -SCORE_MAX, -alpha);
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

    for cell in 0..64 {
        if moves & (1u64 << cell) == 0 {
            continue;
        }
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
fn alphabeta(pos: &Position, depth: u32, mut alpha: i32, beta: i32) -> i32 {
    if pos.is_game_end() {
        return pos.final_score();
    }

    if !pos.has_moves() {
        return -alphabeta(&pos.pass_move(), depth, -beta, -alpha);
    }

    if depth == 0 {
        return heuristic(pos);
    }

    let moves = pos.get_moves();
    for cell in 0..64 {
        if moves & (1u64 << cell) == 0 {
            continue;
        }
        let child = pos.do_move(cell);
        let score = -alphabeta(&child, depth - 1, -beta, -alpha);
        if score > alpha {
            alpha = score;
            if alpha >= beta {
                break;
            }
        }
    }

    alpha
}

fn heuristic(pos: &Position) -> i32 {
    let disc_diff = pos.player_discs() as i32 - pos.opponent_discs() as i32;
    let mobility = pos.get_moves().count_ones() as i32;

    let corners = [0u32, 7, 56, 63];
    let mut corner_diff = 0i32;
    for &c in &corners {
        let bit = 1u64 << c;
        if pos.player & bit != 0 {
            corner_diff += 1;
        } else if pos.opponent & bit != 0 {
            corner_diff -= 1;
        }
    }

    (disc_diff + 2 * mobility + 10 * corner_diff).clamp(SCORE_MIN, SCORE_MAX)
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
        assert_eq!(depth_limited_score(&pos, 0), 64);
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
        let mv = best_move(&pos, 1, 12);
        assert!(mv.is_some(), "best_move should return a move with 1 empty");
    }

    #[test]
    fn test_depth_limited_score_bounded() {
        let pos = Position::initial();
        let score = depth_limited_score(&pos, 4);
        assert!(
            (SCORE_MIN..=SCORE_MAX).contains(&score),
            "score {score} out of bounds"
        );
    }

    #[test]
    fn test_best_move_returns_legal_move() {
        let pos = Position::initial();
        let mv = best_move(&pos, 4, 12);
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
        assert!(best_move(&pos, 4, 12).is_none());
    }

    #[test]
    fn test_heuristic_bounded() {
        let pos = Position::initial();
        let h = heuristic(&pos);
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
