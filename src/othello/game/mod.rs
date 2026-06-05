// Game loading from WTHOR (.wtb) and PGN (.pgn/.txt) game files.
//
// See sub-modules for format-specific readers and file discovery.

pub mod discovery;
pub mod pgn;
pub mod wthor;

use crate::othello::board::Board;

pub use discovery::load_games;

/// Outcome of a completed game.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GameResult {
    BlackWin,
    WhiteWin,
    Draw,
}

impl GameResult {
    /// Parse a PlayOK result string like "50-14" or "1/2-1/2".
    pub fn parse(s: &str) -> Option<Self> {
        if s == "1/2-1/2" {
            return Some(GameResult::Draw);
        }
        // Try "score-score" format like "50-14"
        let parts: Vec<&str> = s.split('-').collect();
        if parts.len() == 2 {
            let black: i32 = parts[0].parse().ok()?;
            let white: i32 = parts[1].parse().ok()?;
            if black > white {
                Some(GameResult::BlackWin)
            } else if white > black {
                Some(GameResult::WhiteWin)
            } else {
                Some(GameResult::Draw)
            }
        } else {
            None
        }
    }
}

/// A complete game loaded from a file.
#[derive(Debug, Clone)]
pub struct Game {
    pub positions: Vec<Board>,
    pub black_name: Option<String>,
    pub white_name: Option<String>,
    pub result: Option<GameResult>,
}
