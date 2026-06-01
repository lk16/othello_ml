use crate::board::Board;
use std::io::Write;
use std::process::{Command, Stdio};
use std::thread;

/// Interface to the Edax engine for obtaining ground truth position evaluations.
///
/// Uses Edax's `-solve /dev/stdin` batch mode (one process per training run):
///   1. Start Edax with `-solve /dev/stdin -level <N>`
///   2. Write all positions as "problems" to stdin
///   3. Close stdin
///   4. Read stdout, parse scores
///   5. Wait for Edax to exit
///
/// ## Problem format
///
/// Each position is written as:
///   `<64-squares> X;\n`
///
/// Where squares use `X` for the side-to-move discs, `O` for opponent discs,
/// and `-` for empty cells. The side to move is always normalized to `X`.
/// This means evaluations are always from the side-to-move perspective,
/// matching the training target convention.
pub struct EdaxInterface;

impl EdaxInterface {
    /// Evaluate a batch of positions.
    ///
    /// Each board's `player` bitboard is the side to move.
    ///
    /// Handles three cases before sending to Edax:
    ///   1. **Game end** (neither side has moves) → exact final score, no Edax call
    ///   2. **Pass** (current player has no moves, opponent does) → swap sides,
    ///      evaluate the passed position, negate the result
    ///   3. **Normal** → send to Edax as-is
    ///
    /// `edax_path` is the path to the Edax binary.
    /// `level` is the search level (0–60, must be even).
    ///
    /// Returns a `Vec<i32>` of scores, one per position, in the same order.
    /// Scores are from the side-to-move perspective.
    pub fn batch_evaluate(
        positions: &[Board],
        level: u32,
        edax_path: &str,
        edax_threads: usize,
    ) -> Result<Vec<i32>, String> {
        let n = positions.len();
        if n == 0 {
            return Ok(Vec::new());
        }

        // Classify each position: game-end, pass, or normal
        enum Action {
            Normal(Board),
            Pass(Board),    // passed board (swapped sides)
            GameEnd(i32),   // exact final score
        }

        let actions: Vec<Action> = positions
            .iter()
            .map(|board| {
                if board.is_game_end() {
                    Action::GameEnd(board.final_score())
                } else if !board.has_moves() {
                    Action::Pass(board.pass_move())
                } else {
                    Action::Normal(*board)
                }
            })
            .collect();

        // Collect only the positions that need Edax evaluation
        let edax_boards: Vec<Board> = actions
            .iter()
            .filter_map(|a| match a {
                Action::Normal(b) | Action::Pass(b) => Some(*b),
                Action::GameEnd(_) => None,
            })
            .collect();

        let edax_scores = if edax_boards.is_empty() {
            Vec::new()
        } else if edax_threads <= 1 || edax_boards.len() < edax_threads * 2 {
            Self::run_edax_solve(&edax_boards, level, edax_path)?
        } else {
            // Split boards across threads, each with its own Edax processes
            let chunk_size = (edax_boards.len() + edax_threads - 1) / edax_threads;
            let edax_path = edax_path.to_string();
            let mut handles = Vec::with_capacity(edax_threads);

            for thread_idx in 0..edax_threads {
                let start = thread_idx * chunk_size;
                if start >= edax_boards.len() {
                    break;
                }
                let end = usize::min(start + chunk_size, edax_boards.len());
                let subset: Vec<Board> = edax_boards[start..end].to_vec();
                let path = edax_path.clone();

                handles.push((
                    thread_idx,
                    thread::spawn(move || {
                        EdaxInterface::run_edax_solve(&subset, level, &path)
                    }),
                ));
            }

            eprintln!(
                "  {} Edax threads processing {} positions ({} chunks/thread)",
                handles.len(),
                edax_boards.len(),
                (chunk_size + EdaxInterface::CHUNK_SIZE - 1) / EdaxInterface::CHUNK_SIZE,
            );

            let n_threads = handles.len();
            let mut results: Vec<Option<Vec<i32>>> = vec![None; n_threads];
            for (idx, handle) in handles {
                match handle.join() {
                    Ok(Ok(scores)) => results[idx] = Some(scores),
                    Ok(Err(e)) => {
                        return Err(format!("Edax thread {} failed: {}", idx, e));
                    }
                    Err(_) => {
                        return Err(format!("Edax thread {} panicked", idx));
                    }
                }
            }

            results.into_iter().flatten().flatten().collect()
        };

        // Map scores back to the original order
        let mut score_iter = edax_scores.into_iter();
        let scores: Vec<i32> = actions
            .iter()
            .map(|action| match action {
                Action::GameEnd(score) => *score,
                Action::Normal(_) => score_iter.next().expect("score count mismatch"),
                Action::Pass(_) => -score_iter.next().expect("score count mismatch"),
            })
            .collect();

        Ok(scores)
    }

    /// Default number of positions per Edax process chunk.
    ///
    /// Each chunk spawns a separate Edax process, writes positions to stdin,
    /// closes stdin, reads results, and exits. This avoids overflowing OS pipe
    /// buffers on large datasets (2M+ positions).
    const CHUNK_SIZE: usize = 100;

    /// Run Edax -solve on a list of boards, chunking into separate processes.
    ///
    /// All boards must have legal moves for the side to move — game-ends and
    /// passes must be handled by the caller before reaching this function.
    fn run_edax_solve(
        boards: &[Board],
        level: u32,
        edax_path: &str,
    ) -> Result<Vec<i32>, String> {
        let n = boards.len();
        if n == 0 {
            return Ok(Vec::new());
        }

        let num_chunks = (n + Self::CHUNK_SIZE - 1) / Self::CHUNK_SIZE;
        let mut all_scores = Vec::with_capacity(n);
        let chunk_start = std::time::Instant::now();

        for chunk_idx in 0..num_chunks {
            let start = chunk_idx * Self::CHUNK_SIZE;
            let end = usize::min(start + Self::CHUNK_SIZE, n);
            let chunk = &boards[start..end];

            let scores = Self::run_edax_solve_chunk(chunk, level, edax_path)
                .map_err(|e| format!("Chunk {}/{} (positions {}-{}): {}", chunk_idx + 1, num_chunks, start + 1, end, e))?;

            all_scores.extend(scores);

            // Progress with ETA
            if num_chunks > 1 {
                let done = chunk_idx + 1;
                let elapsed = chunk_start.elapsed().as_secs_f64().max(0.001);
                let avg_per_chunk = elapsed / done as f64;
                let remaining = avg_per_chunk * (num_chunks - done) as f64;
                eprint!(
                    "\r  [{:3}%] chunk {}/{} | {}/{} pos | ETA: {:.0}s          ",
                    done * 100 / num_chunks, done, num_chunks, end, n, remaining
                );
                let _ = std::io::stderr().flush();
            }
        }

        if num_chunks > 1 {
            eprintln!();
        }

        Ok(all_scores)
    }

    /// Run a single Edax -solve process on a small chunk of boards.
    fn run_edax_solve_chunk(
        boards: &[Board],
        level: u32,
        edax_path: &str,
    ) -> Result<Vec<i32>, String> {
        use std::path::Path;

        // Edax needs to run from the directory above its binary so it can
        // find data/ (book, evaluation weights, etc.). Matches flippy:
        //   cwd = self.edax_path.parent.parent
        let edax_bin = Path::new(edax_path);
        let cwd = edax_bin
            .parent()
            .and_then(|p| p.parent())
            .unwrap_or_else(|| Path::new("."));

        let mut child = Command::new(edax_path)
            .arg("-solve")
            .arg("/dev/stdin")
            .arg("-level")
            .arg(level.to_string())
            .arg("-verbose")
            .arg("3")
            .current_dir(cwd)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| format!("Failed to start Edax '{}': {}", edax_path, e))?;

        {
            let stdin = child
                .stdin
                .as_mut()
                .ok_or_else(|| "Failed to open Edax stdin".to_string())?;

            let mut input = String::with_capacity(boards.len() * 70);
            for board in boards {
                input.push_str(&Self::board_to_problem(board));
            }
            stdin
                .write_all(input.as_bytes())
                .map_err(|e| format!("Error writing to Edax: {}", e))?;
            // stdin is dropped here → closed, signalling EOF to Edax
        }

        let output = child
            .wait_with_output()
            .map_err(|e| format!("Failed to wait for Edax: {}", e))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(format!(
                "Edax exited with error (status: {}): {}",
                output.status, stderr.trim()
            ));
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        Self::parse_solve_output(&stdout, boards.len())
    }

    /// Convert a board to Edax problem format.
    ///
    /// Always normalizes so `X` is the side to move, `O` is the opponent.
    /// `board.player` = side to move → `X`, `board.opponent` → `O`.
    fn board_to_problem(board: &Board) -> String {
        let mut squares = String::with_capacity(70); // 64 squares + " X;\n" + margin
        for i in 0..64 {
            let bit = 1u64 << i;
            if board.player & bit != 0 {
                squares.push('X');
            } else if board.opponent & bit != 0 {
                squares.push('O');
            } else {
                squares.push('-');
            }
        }
        squares.push_str(" X;\n");
        squares
    }

    /// Parse the stdout from `edax -solve`.
    ///
    /// With `-verbose 3`, Edax outputs a block per position separated by
    /// `*** problem ***` markers. Within each block, it prints a score line
    /// for each search depth iteration. We take the **last** score in each
    /// block (the deepest/final evaluation), matching the Python flippy
    /// parser behaviour.
    fn parse_solve_output(stdout: &str, expected_count: usize) -> Result<Vec<i32>, String> {
        let mut scores = Vec::with_capacity(expected_count);

        // Split on "*** problem" markers — each block is one position.
        // The first element is text before the first marker (banner or empty).
        let blocks: Vec<&str> = stdout.split("*** problem").collect();
        let problems = if blocks.len() > 1 { &blocks[1..] } else { &[] };

        for block in problems {
            let mut last_score: Option<i32> = None;

            for line in block.lines() {
                let line = line.trim();
                if line.is_empty()
                    || line.starts_with("----")
                    || line.contains("positions;")
                    || line.contains("/dev/stdin")
                {
                    continue;
                }

                if let Some(score) = Self::parse_solve_line(line) {
                    last_score = Some(score);
                }
            }

            match last_score {
                Some(score) => scores.push(score),
                None => {
                    return Err(format!(
                        "No score found in problem block {}. Block content:\n{}",
                        scores.len() + 1,
                        &block[..block.len().min(500)]
                    ));
                }
            }
        }

        if scores.len() != expected_count {
            return Err(format!(
                "Expected {} scores from Edax but parsed {}.\n\
                 Make sure all input positions have at least one legal move.",
                expected_count,
                scores.len()
            ));
        }

        Ok(scores)
    }

    /// Parse one output line from Edax's -solve output.
    ///
    /// Expected format: `<depth>@<confidence>%  <+score>  <moves...>`
    /// Examples: ` 10@100%  +12  f5  g6  ...`
    ///           `  5@73%   -3  d3  c4  ...`
    fn parse_solve_line(line: &str) -> Option<i32> {
        let columns: Vec<&str> = line.split_whitespace().collect();
        if columns.len() < 2 {
            return None;
        }

        // First column must be depth[@confidence%]
        if !columns[0].contains('@') && !columns[0].ends_with('%') {
            // Could be just a depth number, but -verbose 3 includes @confidence
        }

        // Check that the first column looks like "N@M%" or "N"
        let first = columns[0];
        let has_at = first.contains('@');
        let has_pct = first.ends_with('%') || first.contains('%');

        if !has_at && !has_pct && first.parse::<u32>().is_err() {
            return None; // Not a depth column — not a score line
        }

        // Second column is the score: +12, -5, etc.
        let score_str = columns[1];
        // Strip angle brackets sometimes used for exact scores: <+10>
        let score_str = score_str.trim_matches(|c: char| c == '<' || c == '>');
        score_str.parse::<i32>().ok()
    }
}

/// Convert a board to an Edax FEN string (used for eval file persistence).
///
/// 64 characters (A1..H1, A2..H2, …, A8..H8) using:
///   `X` = black disc, `O` = white disc, `-` = empty
/// Followed by a space and the side to move (`X` for black, `O` for white).
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

/// Check whether Edax is available (EDAX_PATH is set).
pub fn edax_available() -> bool {
    std::env::var("EDAX_PATH").is_ok()
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
    fn test_board_to_problem_initial() {
        let board = Board::initial();
        let problem = EdaxInterface::board_to_problem(&board);

        // Should end with " X;\n"
        assert!(problem.ends_with(" X;\n"));
        // 64 squares + " X;\n" = 68 chars
        assert_eq!(problem.len(), 68);

        let squares = &problem[..64];
        assert_eq!(squares.chars().filter(|&c| c == 'X').count(), 2);
        assert_eq!(squares.chars().filter(|&c| c == 'O').count(), 2);
        assert_eq!(squares.chars().filter(|&c| c == '-').count(), 60);
    }

    #[test]
    fn test_board_to_problem_normalizes_to_x() {
        // Even with black_to_move=false, board_to_problem always uses X
        // for the side to move (player)
        let board = Board::initial();
        let problem = EdaxInterface::board_to_problem(&board);
        // Player = side to move → X, always
        assert!(problem.ends_with(" X;\n"));
    }

    #[test]
    fn test_board_to_fen_initial() {
        let board = Board::initial();
        let fen = board_to_fen(&board, true);

        assert_eq!(fen.len(), 66);
        assert_eq!(&fen[64..65], " ");

        let (board_part, side) = split_fen(&fen);
        assert_eq!(side, 'X');

        assert_eq!(board_part.chars().filter(|&c| c == 'X').count(), 2);
        assert_eq!(board_part.chars().filter(|&c| c == 'O').count(), 2);
        assert_eq!(board_part.chars().filter(|&c| c == '-').count(), 60);
    }

    #[test]
    fn test_board_to_fen_white_to_move() {
        let board = Board::initial();
        let fen = board_to_fen(&board, false);
        let (board_part, side) = split_fen(&fen);

        assert_eq!(side, 'O');
        assert_eq!(board_part.chars().filter(|&c| c == 'X').count(), 2);
        assert_eq!(board_part.chars().filter(|&c| c == 'O').count(), 2);
        assert_eq!(board_part.chars().filter(|&c| c == '-').count(), 60);
    }

    #[test]
    fn test_board_to_fen_empty() {
        let board = Board::new();
        let fen = board_to_fen(&board, true);
        let (board_part, side) = split_fen(&fen);

        assert_eq!(side, 'X');
        assert_eq!(board_part.chars().filter(|&c| c == '-').count(), 64);
        assert_eq!(board_part.chars().filter(|&c| c == 'X').count(), 0);
        assert_eq!(board_part.chars().filter(|&c| c == 'O').count(), 0);
    }

    #[test]
    fn test_fen_board_order() {
        let board = Board::new();
        let fen = board_to_fen(&board, true);
        assert_eq!(fen.len(), 66);
    }

    #[test]
    fn test_parse_solve_line_positive() {
        assert_eq!(
            EdaxInterface::parse_solve_line(" 10@100%  +12  f5  g6  d6"),
            Some(12)
        );
        assert_eq!(
            EdaxInterface::parse_solve_line("  5@73%   -3  d3  c4"),
            Some(-3)
        );
    }

    #[test]
    fn test_parse_solve_line_exact() {
        // Exact scores may be wrapped in angle brackets
        assert_eq!(
            EdaxInterface::parse_solve_line(" 24@100%  <+10>  c4  d3"),
            Some(10)
        );
        assert_eq!(
            EdaxInterface::parse_solve_line(" 60@100%  <-64>  a1"),
            Some(-64)
        );
    }

    #[test]
    fn test_parse_solve_line_non_score() {
        assert_eq!(EdaxInterface::parse_solve_line(""), None);
        assert_eq!(EdaxInterface::parse_solve_line("  A B C D E F G H"), None);
        assert_eq!(
            EdaxInterface::parse_solve_line("Edax version 4.4"),
            None
        );
    }

    #[test]
    fn test_edax_available() {
        let _available = edax_available();
    }
}
