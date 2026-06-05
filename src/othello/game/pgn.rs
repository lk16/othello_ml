// PGN (.pgn / .txt) text format reader.
//
// Moves are 2-character fields (a1-h8). PlayOK variant uses "Black"/"White"
// tags and scores like "50-14" as the Result.

use crate::othello::board::Board;
use crate::othello::position::{Cell, Position};
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
/// Returns None if no game could be parsed (e.g., empty input).
fn parse_pgn_game<'a, I>(lines: &mut I) -> Option<super::Game>
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
        return None;
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
                if board.get_cell(cell as u32) != Cell::Empty {
                    // Could be a pass - swap sides
                    std::mem::swap(&mut board.player, &mut board.opponent);
                    black_to_move = !black_to_move;

                    if board.get_cell(cell as u32) != Cell::Empty {
                        // Still invalid, revert and skip
                        std::mem::swap(&mut board.player, &mut board.opponent);
                        black_to_move = !black_to_move;
                        continue;
                    }
                }

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

                // Apply the move
                board.player |= 1u64 << cell;
                board.flip_discs(cell as u32);

                // Switch sides
                std::mem::swap(&mut board.player, &mut board.opponent);
                black_to_move = !black_to_move;
            }
        }
    }

    if positions.is_empty() {
        return None;
    }

    Some(super::Game {
        positions,
        black_name: metadata.get("Black").cloned(),
        white_name: metadata.get("White").cloned(),
        result_score: metadata.get("Result").cloned(),
    })
}

/// Parse PGN content with potentially multiple games.
pub fn parse_pgn_multi(content: &str) -> Vec<super::Game> {
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

        if let Some(game) = parse_pgn_game(&mut iter) {
            games.push(game);
        } else {
            break;
        }
    }

    games
}

/// Read a PGN file (possibly with multiple games).
pub fn read_pgn_file(path: &Path) -> Result<Vec<super::Game>, String> {
    let content = fs::read_to_string(path)
        .map_err(|e| format!("Failed to read {}: {}", path.display(), e))?;
    Ok(parse_pgn_multi(&content))
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
        let games = parse_pgn_multi(pgn);
        assert!(!games.is_empty());
        assert_eq!(games[0].black_name.as_deref(), Some("hz36"));
        assert_eq!(games[0].white_name.as_deref(), Some("lk16"));
        assert_eq!(games[0].result_score.as_deref(), Some("50-14"));
        assert!(!games[0].positions.is_empty());
    }

    #[test]
    fn test_parse_pgn_empty() {
        let pgn = "";
        let games = parse_pgn_multi(pgn);
        assert!(games.is_empty());
    }
}
