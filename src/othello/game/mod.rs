// Game loading from WTHOR (.wtb) and PGN (.pgn/.txt) game files.
//
// See sub-modules for format-specific readers and file discovery.

pub mod discovery;
pub mod pgn;
pub mod wthor;

use crate::othello::board::Board;
use crate::othello::position::Position;

pub use discovery::load_games;

/// A complete game loaded from a file.
#[derive(Debug, Clone)]
pub struct Game {
    pub positions: Vec<Board>,
    pub black_name: Option<String>,
    pub white_name: Option<String>,
    pub result_score: Option<String>,
}

/// Flip opponent discs in all 8 directions after placing a disc at `cell`.
pub(crate) fn flip_discs(board: &mut Position, cell: u32) {
    let directions: [(i32, i32); 8] = [
        (-1, -1),
        (0, -1),
        (1, -1),
        (-1, 0),
        (1, 0),
        (-1, 1),
        (0, 1),
        (1, 1),
    ];

    let x = (cell % 8) as i32;
    let y = (cell / 8) as i32;

    for &(dx, dy) in &directions {
        let mut flips: u64 = 0;
        let mut nx = x + dx;
        let mut ny = y + dy;

        while (0..8).contains(&nx) && (0..8).contains(&ny) {
            let idx = (ny * 8 + nx) as u32;
            let bit = 1u64 << idx;

            if board.opponent & bit != 0 {
                flips |= bit;
            } else if board.player & bit != 0 {
                // Found our own disc - flip the captured pieces
                board.player |= flips;
                board.opponent &= !flips;
                break;
            } else {
                // Empty cell - no capture in this direction
                break;
            }

            nx += dx;
            ny += dy;
        }
    }
}
