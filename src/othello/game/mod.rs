// Game loading from WTHOR (.wtb) and PGN (.pgn/.txt) game files.
//
// See sub-modules for format-specific readers and file discovery.

pub mod discovery;
pub mod pgn;
pub mod wthor;

use crate::othello::board::Board;

pub use discovery::load_games;

/// A complete game loaded from a file.
#[derive(Debug, Clone)]
pub struct Game {
    pub positions: Vec<Board>,
    pub black_name: Option<String>,
    pub white_name: Option<String>,
    pub result_score: Option<String>,
}
