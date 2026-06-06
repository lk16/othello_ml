// PGN (.pgn / .txt) text format reader.
//
// Moves are 2-character fields (a1-h8). PlayOK variant uses "Black"/"White"
// tags and scores like "50-14" as the Result.

use crate::othello::board::Board;
use crate::othello::game::GameResult;
use crate::othello::position::Position;
use std::collections::HashMap;
use std::fs;
use std::path::Path;

/// Convert a PlayOK-style field (e.g., "E6", "f4") to a cell index (0-63).
fn field_to_index(field: &str) -> Option<u8> {
    if field.len() != 2 {
        return None;
    }
    let chars: Vec<char> = field.chars().collect();
    let col = chars[0].to_ascii_lowercase() as i32 - 'a' as i32;
    let row = chars[1] as i32 - '1' as i32;

    if !(0..=7).contains(&col) || !(0..=7).contains(&row) {
        return None;
    }

    Some((row * 8 + col) as u8)
}

/// Parse a single PGN game from an iterator over lines.
/// Returns `Ok(None)` if no game could be parsed (e.g., empty input).
/// Returns `Err` if the game contains corrupted/illegal moves.
fn parse_pgn_game<'a, I>(lines: &mut I) -> Result<Option<super::Game>, String>
where
    I: Iterator<Item = &'a str>,
{
    let mut metadata: HashMap<String, String> = HashMap::new();
    let mut move_lines: Vec<String> = Vec::new();
    let mut header_done = false;

    for line in lines.by_ref() {
        let line = line.trim();

        if line.is_empty() {
            if !header_done && !metadata.is_empty() {
                // Empty line after headers - moves follow
                header_done = true;
                continue;
            }
            if header_done && !move_lines.is_empty() {
                // Empty line after moves - end of game
                break;
            }
            continue;
        }

        if !header_done && line.starts_with('[') {
            // Parse header [Key "Value"]
            if let Some(rest) = line.strip_prefix('[') {
                if let Some(rest) = rest.strip_suffix(']') {
                    let parts: Vec<&str> = rest.splitn(2, ' ').collect();
                    if parts.len() == 2 {
                        let key = parts[0].trim().to_string();
                        let value = parts[1].trim().trim_matches('"').to_string();
                        metadata.insert(key, value);
                    }
                }
            }
        } else {
            // Move text line
            header_done = true;
            move_lines.push(line.to_string());
        }
    }

    if metadata.is_empty() && move_lines.is_empty() {
        return Ok(None);
    }

    // Replay moves to generate board positions
    let mut positions = Vec::new();
    let mut board = Position::initial();
    let mut black_to_move = true;

    for line in &move_lines {
        for word in line.split_whitespace() {
            // Skip move numbers (e.g., "1.", "30.")
            let word = word.trim_end_matches('.');
            if word.chars().all(|c| c.is_ascii_digit()) {
                continue;
            }
            // Skip result strings like "50-14", "0-64", "1/2-1/2"
            if word.contains('-') && word.len() >= 3 && !word.contains('/') {
                break;
            }
            if word == "1/2-1/2" {
                break;
            }

            if let Some(cell) = field_to_index(word) {
                // Check if the move is legal (flips at least one disc).
                // This covers both occupied cells and empty cells with no flips.
                if board.flipped(cell as u32) == 0 {
                    // Illegal move — assume a pass (current player has no moves).
                    // Swap sides and retry the move for the other player.
                    std::mem::swap(&mut board.player, &mut board.opponent);
                    black_to_move = !black_to_move;

                    if board.flipped(cell as u32) == 0 {
                        // Still illegal after swap — game is corrupted.
                        return Err(format!(
                            "Illegal move '{}' at position with {} empties",
                            word,
                            board.empties()
                        ));
                    }
                }

                // Passed positions are not recorded: the passing player has
                // no legal moves, so there is no position to train on.

                // Record position BEFORE the move
                let faced = Board {
                    position: Position {
                        player: if black_to_move {
                            board.player
                        } else {
                            board.opponent
                        },
                        opponent: if black_to_move {
                            board.opponent
                        } else {
                            board.player
                        },
                    },
                    black_to_move,
                };
                positions.push(faced);

                // Apply the move and switch sides
                board = board.do_move(cell as u32);
                black_to_move = !black_to_move;
            }
        }
    }

    if positions.is_empty() {
        return Ok(None);
    }

    Ok(Some(super::Game {
        positions,
        black_name: metadata.get("Black").cloned(),
        white_name: metadata.get("White").cloned(),
        result: metadata.get("Result").and_then(|s| GameResult::parse(s)),
    }))
}

/// Parse PGN content with potentially multiple games.
pub fn parse_pgn_multi(content: &str) -> Result<Vec<super::Game>, String> {
    let lines: Vec<&str> = content.lines().collect();
    let mut games = Vec::new();
    let mut iter = lines.iter().copied().peekable();

    loop {
        // Skip leading blank lines
        while iter.peek().is_some_and(|l| l.trim().is_empty()) {
            iter.next();
        }

        if iter.peek().is_none() {
            break;
        }

        match parse_pgn_game(&mut iter)? {
            Some(game) => games.push(game),
            None => break,
        }
    }

    Ok(games)
}

/// Read a PGN file (possibly with multiple games).
pub fn read_pgn_file(path: &Path) -> Result<Vec<super::Game>, String> {
    let content = fs::read_to_string(path)
        .map_err(|e| format!("Failed to read {}: {}", path.display(), e))?;
    parse_pgn_multi(&content)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_field_to_index() {
        let cases = [
            ("a1", Some(0)),
            ("h1", Some(7)),
            ("A2", Some(8)),
            ("H8", Some(63)),
            ("E6", Some(44)), // e=4, 6=5 → 44
        ];
        for (input, expected) in cases {
            assert_eq!(field_to_index(input), expected, "field_to_index({input:?})");
        }
    }

    #[test]
    fn test_parse_pgn_single() {
        let pgn = r#"[Event "?"]
[Black "hz36"]
[White "lk16"]
[Result "50-14"]

1. E6 f4 2. E3 d6 3. C5 f3 50-14
"#;
        let games = parse_pgn_multi(pgn).expect("valid game");
        assert!(!games.is_empty());
        assert_eq!(games[0].black_name.as_deref(), Some("hz36"));
        assert_eq!(games[0].white_name.as_deref(), Some("lk16"));
        assert_eq!(games[0].result, Some(GameResult::BlackWin));
        assert!(!games[0].positions.is_empty());
    }

    #[test]
    fn test_parse_pgn_empty() {
        let pgn = "";
        let games = parse_pgn_multi(pgn).expect("empty input");
        assert!(games.is_empty());
    }

    #[test]
    fn test_parse_pgn_corrupted_move() {
        // A1 is occupied at the start, so playing there is illegal for both sides.
        let pgn = r#"[Event "?"]
[Black "test"]
[White "test"]
[Result "0-0"]

1. A1 0-0
"#;
        let result = parse_pgn_multi(pgn);
        assert!(result.is_err(), "corrupted game should return error");
        assert!(
            result.unwrap_err().contains("Illegal move"),
            "error should mention illegal move"
        );
    }
}
