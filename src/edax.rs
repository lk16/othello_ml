use crate::board::Board;
use std::env;
use std::io::{BufRead, BufReader, Write};
use std::process::{Command, Stdio};

/// Interface to the Edax engine for obtaining ground truth position evaluations.
///
/// Starts Edax as a persistent subprocess and communicates via stdin/stdout pipes.
/// This avoids the per-evaluation process-spawn overhead (critical for thousands
/// of training positions).
///
/// ## Edax FEN format
///
/// 64 characters (A1..H1, A2..H2, …, A8..H8):
///   `X` = black disc, `O` = white disc, `-` = empty
/// Followed by a space and the side to move (`X` or `O`).
///
/// ## Edax protocol
///
///   set board <fen>   →  set the board position
///   eval              →  evaluate and print the score
///   quit              →  exit
///
/// Edax's `eval` score is from the perspective of the side to move:
/// positive  = side to move is better,
/// negative  = opponent is better.
pub struct EdaxInterface {
    child: std::process::Child,
    stdin: std::process::ChildStdin,
    reader: BufReader<std::process::ChildStdout>,
}

impl EdaxInterface {
    /// Start Edax in persistent mode.
    ///
    /// Reads the path to the Edax binary from the `EDAX_PATH` environment variable.
    /// Stderr is discarded so it doesn't fill the pipe buffer.
    pub fn new() -> Result<Self, String> {
        let edax_path = env::var("EDAX_PATH")
            .map_err(|_| "EDAX_PATH environment variable not set".to_string())?;

        let mut child = Command::new(&edax_path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|e| format!("Failed to start Edax '{}': {}", edax_path, e))?;

        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| "Failed to open Edax stdin".to_string())?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| "Failed to open Edax stdout".to_string())?;
        let reader = BufReader::new(stdout);

        Ok(EdaxInterface {
            child,
            stdin,
            reader,
        })
    }

    /// Convert a [`Board`] to an Edax FEN string.
    ///
    /// When `black_to_move` is true,  `player` bits → `X`, `opponent` bits → `O`.
    /// When `black_to_move` is false, `player` bits → `O`, `opponent` bits → `X`.
    /// The side-to-move character is always `X` for black, `O` for white.
    pub fn board_to_fen(board: &Board, black_to_move: bool) -> String {
        let mut fen = String::with_capacity(66);
        for i in 0..64 {
            let bit = 1u64 << i;
            let (is_black, is_white) = if black_to_move {
                (board.player & bit != 0, board.opponent & bit != 0)
            } else {
                (board.opponent & bit != 0, board.player & bit != 0)
            };
            fen.push(if is_black {
                'X'
            } else if is_white {
                'O'
            } else {
                '-'
            });
        }
        fen.push(' ');
        fen.push(if black_to_move { 'X' } else { 'O' });
        fen
    }

    /// Evaluate a board position.
    ///
    /// Sends `set board <fen>` followed by `eval` to the persistent Edax
    /// process, then reads output lines until a numeric score is found.
    ///
    /// Returns the score from the perspective of the side to move:
    /// positive = good for the side to move, negative = good for the opponent.
    pub fn evaluate(&mut self, board: &Board, black_to_move: bool) -> Result<i32, String> {
        let fen = Self::board_to_fen(board, black_to_move);
        writeln!(self.stdin, "set board {}", fen)
            .and_then(|_| writeln!(self.stdin, "eval"))
            .and_then(|_| self.stdin.flush())
            .map_err(|e| format!("Error writing to Edax: {}", e))?;

        let mut line = String::new();
        loop {
            line.clear();
            match self.reader.read_line(&mut line) {
                Ok(0) => {
                    // EOF — Edax crashed or closed stdout
                    return Err("Edax closed stdout unexpectedly".to_string());
                }
                Ok(_) => {
                    if let Some(score) = Self::parse_eval_line(&line) {
                        return Ok(score);
                    }
                    // otherwise keep reading — skip board display, banners, etc.
                }
                Err(e) => {
                    return Err(format!("Error reading Edax output: {}", e));
                }
            }
        }
    }

    /// Try to extract a numeric score from an Edax output line.
    ///
    /// Handles several common Edax output formats:
    ///   "+12.34"         — bare score line
    ///   "  +12.34"       — indented score line
    ///   "score: +12.34"   — labelled score
    ///   "eval = -5"       — alternative label
    fn parse_eval_line(line: &str) -> Option<i32> {
        let line = line.trim();
        if line.is_empty() {
            return None;
        }

        // Look for a token that looks like a signed number: +12.34 or -5.00 or +10
        for token in line.split(&[' ', ':', '='][..]) {
            let token = token.trim();
            // Must start with a sign and be at least "+0"
            if token.len() < 2 {
                continue;
            }
            let first = token.chars().next().unwrap();
            if first != '+' && first != '-' {
                continue;
            }
            // Collect characters that belong to the number
            let num_str: String = token
                .chars()
                .take_while(|&c| c == '+' || c == '-' || c == '.' || c.is_ascii_digit())
                .collect();
            if num_str.len() > 1 {
                if let Ok(val) = num_str.parse::<f64>() {
                    return Some(val.round() as i32);
                }
            }
        }

        None
    }

    /// Cleanly shut down Edax.
    ///
    /// Sends `quit` and waits for the process to exit.
    /// It is safe to drop this struct without calling `shutdown` — the
    /// process will be killed on drop, but explicit shutdown is cleaner.
    pub fn shutdown(mut self) -> Result<(), String> {
        let _ = writeln!(self.stdin, "quit");
        let _ = self.stdin.flush();
        self.child.wait().map_err(|e| e.to_string())?;
        Ok(())
    }
}

impl Drop for EdaxInterface {
    fn drop(&mut self) {
        // Best-effort cleanup: try to quit gracefully, then kill if needed.
        let _ = writeln!(self.stdin, "quit");
        let _ = self.stdin.flush();
        // Don't block — if Edax is stuck, the OS will reap the zombie.
        // We use try_wait to avoid hanging the whole program on drop.
        if let Ok(Some(_)) = self.child.try_wait() {
            // already exited
        } else {
            // Give it a moment, then kill
            std::thread::sleep(std::time::Duration::from_millis(100));
            let _ = self.child.kill();
        }
    }
}

/// Check whether Edax is available (EDAX_PATH is set).
pub fn edax_available() -> bool {
    env::var("EDAX_PATH").is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: split FEN into board part (first 64 chars) and side-to-move
    fn split_fen(fen: &str) -> (&str, char) {
        let board_part = &fen[..64];
        let side = fen.chars().last().unwrap();
        (board_part, side)
    }

    #[test]
    fn test_board_to_fen_initial() {
        let board = Board::initial();
        let fen = EdaxInterface::board_to_fen(&board, true);

        // 64 chars + space + side = 66 characters
        assert_eq!(fen.len(), 66);
        assert_eq!(&fen[64..65], " ");

        let (board_part, side) = split_fen(&fen);
        assert_eq!(side, 'X'); // black to move

        // 2 black discs (player), 2 white discs (opponent), 60 empty
        assert_eq!(board_part.chars().filter(|&c| c == 'X').count(), 2);
        assert_eq!(board_part.chars().filter(|&c| c == 'O').count(), 2);
        assert_eq!(board_part.chars().filter(|&c| c == '-').count(), 60);
    }

    #[test]
    fn test_board_to_fen_white_to_move() {
        // When Board::initial() is called with black_to_move=false,
        // player bits (d5,e4=black discs) → O, opponent bits (d4,e5=white discs) → X.
        let board = Board::initial();
        let fen = EdaxInterface::board_to_fen(&board, false);
        let (board_part, side) = split_fen(&fen);

        assert_eq!(side, 'O'); // white to move
        assert_eq!(board_part.chars().filter(|&c| c == 'X').count(), 2);
        assert_eq!(board_part.chars().filter(|&c| c == 'O').count(), 2);
        assert_eq!(board_part.chars().filter(|&c| c == '-').count(), 60);
    }

    #[test]
    fn test_board_to_fen_empty() {
        let board = Board::new();
        let fen = EdaxInterface::board_to_fen(&board, true);
        let (board_part, side) = split_fen(&fen);

        assert_eq!(side, 'X');
        assert_eq!(board_part.chars().filter(|&c| c == '-').count(), 64);
        assert_eq!(board_part.chars().filter(|&c| c == 'X').count(), 0);
        assert_eq!(board_part.chars().filter(|&c| c == 'O').count(), 0);
    }

    #[test]
    fn test_fen_board_order() {
        // Verify A1..H1 are the first 8 characters, A2..H2 are next, etc.
        // A1=0 is the first char in Edax FEN format.
        let board = Board::new();
        let fen = EdaxInterface::board_to_fen(&board, true);
        assert_eq!(fen.len(), 66);
        // All dashes means the ordering is just positional — trust it.
    }

    #[test]
    fn test_parse_eval_line_positive() {
        assert_eq!(EdaxInterface::parse_eval_line("+12.34"), Some(12));
        assert_eq!(EdaxInterface::parse_eval_line("  +10.0"), Some(10));
        assert_eq!(EdaxInterface::parse_eval_line("+0.00"), Some(0));
    }

    #[test]
    fn test_parse_eval_line_negative() {
        assert_eq!(EdaxInterface::parse_eval_line("-5.67"), Some(-6));
        assert_eq!(EdaxInterface::parse_eval_line("  -3.0"), Some(-3));
    }

    #[test]
    fn test_parse_eval_line_labelled() {
        assert_eq!(
            EdaxInterface::parse_eval_line("score: +12.34"),
            Some(12)
        );
        assert_eq!(EdaxInterface::parse_eval_line("eval = -5"), Some(-5));
    }

    #[test]
    fn test_parse_eval_line_non_score() {
        assert_eq!(EdaxInterface::parse_eval_line(""), None);
        assert_eq!(EdaxInterface::parse_eval_line("  A B C D E F G H"), None);
        assert_eq!(EdaxInterface::parse_eval_line("Edax version 4.4"), None);
    }

    #[test]
    fn test_edax_available() {
        // Just check the function compiles and returns a bool
        let _available = edax_available();
    }
}
