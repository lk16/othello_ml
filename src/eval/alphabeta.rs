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
    alphabeta(pos, SCORE_MIN, SCORE_MAX)
}

/// Score bounds.  +1 is added to the min bound so that `-beta` / `-alpha`
/// never overflow `i32` in the negamax recursion.
const SCORE_MIN: i32 = -64;
const SCORE_MAX: i32 = 64;

/// Negamax with alpha-beta pruning.
fn alphabeta(pos: &Position, mut alpha: i32, beta: i32) -> i32 {
    // Terminal position: final score from current player's perspective.
    if pos.is_game_end() {
        return pos.final_score();
    }

    // Pass: no legal moves for the current player.
    if !pos.has_moves() {
        return -alphabeta(&pos.pass_move(), -beta, -alpha);
    }

    let moves = pos.get_moves();
    for cell in 0..64 {
        let bit = 1u64 << cell;
        if moves & bit == 0 {
            continue;
        }
        let child = make_move(pos, cell);
        let score = -alphabeta(&child, -beta, -alpha);
        if score > alpha {
            alpha = score;
            if alpha >= beta {
                break;
            }
        }
    }

    alpha
}

/// Apply a move: place disc at `cell`, flip opponent discs, swap sides.
fn make_move(pos: &Position, cell: u32) -> Position {
    let mut child = *pos;
    child.place_disc(cell);
    child.flip_discs(cell);
    // After the move it becomes the opponent's turn.
    std::mem::swap(&mut child.player, &mut child.opponent);
    child
}

/// Evaluate a batch of positions, returning one score per position
/// in the same order.  Handles game-end and pass positions without
/// invoking the full search.
pub fn batch_evaluate(positions: &[Position]) -> Vec<i32> {
    positions.iter().map(exact_score).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exact_score_game_end_full() {
        // Black owns the entire board → 64-0 win
        let pos = Position {
            player: u64::MAX,
            opponent: 0,
        };
        assert!(pos.is_game_end());
        assert_eq!(exact_score(&pos), 64);
    }

    #[test]
    fn test_exact_score_game_end_tie() {
        // Neither side has discs → tie
        let pos = Position::new();
        assert!(pos.is_game_end());
        assert_eq!(exact_score(&pos), 0);
    }

    #[test]
    fn test_exact_score_game_end_opponent_wins() {
        // Opponent owns the entire board
        let pos = Position {
            player: 0,
            opponent: u64::MAX,
        };
        assert!(pos.is_game_end());
        assert_eq!(exact_score(&pos), -64);
    }

    #[test]
    fn test_exact_score_one_empty() {
        // 63 squares filled, 1 empty. Black (player) has 32, White 31.
        // Black to move → places last disc → wins.
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
}
