// Game loading from WTHOR (.wtb) and PGN (.pgn/.txt) game files.
//
// WTHOR format: Binary format with a 16-byte header followed by 68-byte
// game records. Each record contains metadata and up to 60 moves.
// Game count = (file_size - 16) / 68.
// Move encoding: edax_index = 8 * ((wthor_value - 11) / 10) + ((wthor_value - 11) % 10)
//
// PGN format: Text format with metadata headers [Key "Value"] followed by move text.
// Moves are 2-character fields (a1-h8). PlayOK variant uses "Black"/"White" tags
// and scores like "50-14" as the Result.
//
// Supports loading individual files or recursively scanning directories.

use crate::othello::board::Board;
use crate::othello::position::{Cell, Position};
use std::fs;
use std::path::Path;

/// A complete game loaded from a file.
#[derive(Debug, Clone)]
pub struct Game {
    pub positions: Vec<Board>,
    pub black_name: Option<String>,
    pub white_name: Option<String>,
    pub result_score: Option<String>,
}

// ─── WTHOR (.wtb) reader ──────────────────────────────────────────

/// Convert a WTHOR move byte to an Edax cell index (0-63).
///
/// WTHOR encodes moves as `10 * row + col + 1` with rows/cols 1-8.
/// Edax expects a linear cell index: `8 * ((x - 11) / 10) + ((x - 11) % 10)`.
/// This maps:
///   WTHOR 11 -> 0 (A1), WTHOR 18 -> 7 (H1)
///   WTHOR 21 -> 8 (A2), WTHOR 28 -> 15 (H2)
///   ...
///   WTHOR 81 -> 56 (A8), WTHOR 88 -> 63 (H8)
fn move_from_wthor(x: u8) -> u8 {
    if !(11..=88).contains(&x) {
        return 255; // invalid/NOMOVE indicator
    }
    let adjusted = x - 11;
    8 * (adjusted / 10) + (adjusted % 10)
}

/// Read all games from a WTHOR (.wtb) file.
pub fn read_wthor_file(path: &Path) -> Result<Vec<Game>, String> {
    let data = fs::read(path).map_err(|e| format!("Failed to read {}: {}", path.display(), e))?;

    // WTHOR files have a 16-byte header (not a game count).
    // Number of games is derived from the file size.
    // Each WthorGame is 68 bytes:
    //   tournament: i16 (2)
    //   black: i16 (2)
    //   white: i16 (2)
    //   score: i8 (1)
    //   theoric_score: i8 (1)
    //   x[60]: [u8; 60]
    const WTHOR_HEADER_SIZE: usize = 16;
    const WTHOR_GAME_SIZE: usize = 68;

    if data.len() < WTHOR_HEADER_SIZE {
        return Err(format!(
            "File {} too small for WTHOR header (need at least {} bytes, got {})",
            path.display(),
            WTHOR_HEADER_SIZE,
            data.len()
        ));
    }

    let body_size = data.len() - WTHOR_HEADER_SIZE;
    if body_size % WTHOR_GAME_SIZE != 0 {
        return Err(format!(
            "File {}: body size {} is not a multiple of game record size {}",
            path.display(),
            body_size,
            WTHOR_GAME_SIZE
        ));
    }
    let num_games = body_size / WTHOR_GAME_SIZE;

    let mut games = Vec::with_capacity(num_games);
    let mut offset = WTHOR_HEADER_SIZE; // skip 16-byte header

    for _ in 0..num_games {
        if offset + WTHOR_GAME_SIZE > data.len() {
            break;
        }

        let game_data = &data[offset..offset + WTHOR_GAME_SIZE];
        offset += WTHOR_GAME_SIZE;

        // Parse WTHOR fields (all little-endian)
        let _tournament = i16::from_le_bytes([game_data[0], game_data[1]]);
        let _black_id = i16::from_le_bytes([game_data[2], game_data[3]]);
        let _white_id = i16::from_le_bytes([game_data[4], game_data[5]]);
        let _score = game_data[6] as i8;
        let _theoric_score = game_data[7] as i8;
        let moves = &game_data[8..68]; // 60 move bytes

        // Replay the game to generate board positions
        let mut positions = Vec::new();
        let mut board = Position::initial();
        let mut black_to_move = true;

        for &mv in moves.iter() {
            if mv == 0 {
                // NOMOVE - game ended
                break;
            }

            let cell = move_from_wthor(mv);
            if cell > 63 {
                break;
            }

            // Check if the move is valid by seeing if the cell is empty
            if board.get_cell(cell as u32) != Cell::Empty {
                // Skip invalid moves (could be a pass that's not recorded)
                // Swap sides and try again
                std::mem::swap(&mut board.player, &mut board.opponent);
                black_to_move = !black_to_move;

                if board.get_cell(cell as u32) != Cell::Empty {
                    // Still occupied, game is corrupted - break
                    std::mem::swap(&mut board.player, &mut board.opponent);
                    break;
                }
            }

            // Place the disc and flip captured pieces
            board.player |= 1u64 << cell;
            flip_discs(&mut board, cell as u32);

            // Record the position BEFORE the move (the position the player faced)
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

            // Switch sides
            std::mem::swap(&mut board.player, &mut board.opponent);
            black_to_move = !black_to_move;
        }

        if !positions.is_empty() {
            games.push(Game {
                positions,
                black_name: None,
                white_name: None,
                result_score: None,
            });
        }
    }

    Ok(games)
}

/// Flip opponent discs in all 8 directions after placing a disc at `cell`.
fn flip_discs(board: &mut Position, cell: u32) {
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

// ─── PGN (.pgn / .txt) reader ─────────────────────────────────────

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
fn parse_pgn_game<'a, I>(lines: &mut I) -> Option<Game>
where
    I: Iterator<Item = &'a str>,
{
    let mut metadata: std::collections::HashMap<String, String> = std::collections::HashMap::new();
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
                flip_discs(&mut board, cell as u32);

                // Switch sides
                std::mem::swap(&mut board.player, &mut board.opponent);
                black_to_move = !black_to_move;
            }
        }
    }

    if positions.is_empty() {
        return None;
    }

    Some(Game {
        positions,
        black_name: metadata.get("Black").cloned(),
        white_name: metadata.get("White").cloned(),
        result_score: metadata.get("Result").cloned(),
    })
}

/// Parse PGN content with potentially multiple games.
pub fn parse_pgn_multi(content: &str) -> Vec<Game> {
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

        let game = parse_pgn_game(&mut iter);
        if let Some(game) = game {
            games.push(game);
        } else {
            break;
        }
    }

    games
}

/// Read a PGN file (possibly with multiple games).
pub fn read_pgn_file(path: &Path) -> Result<Vec<Game>, String> {
    let content = fs::read_to_string(path)
        .map_err(|e| format!("Failed to read {}: {}", path.display(), e))?;
    Ok(parse_pgn_multi(&content))
}

// ─── File discovery ───────────────────────────────────────────────

/// Determine file type from extension.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FileType {
    Wthor,
    Pgn,
    Unknown,
}

fn file_type(path: &Path) -> FileType {
    match path.extension().and_then(|e| e.to_str()) {
        Some("wtb") | Some("WTH") => FileType::Wthor,
        Some("pgn") | Some("PGN") | Some("txt") | Some("TXT") => FileType::Pgn,
        _ => FileType::Unknown,
    }
}

/// Recursively collect all .wtb and .pgn/.txt files from a directory.
fn collect_game_files(dir: &Path) -> Result<Vec<std::path::PathBuf>, String> {
    let mut files = Vec::new();

    let entries = fs::read_dir(dir)
        .map_err(|e| format!("Failed to read directory {}: {}", dir.display(), e))?;

    for entry in entries {
        let entry = entry.map_err(|e| e.to_string())?;
        let path = entry.path();

        if path.is_dir() {
            files.extend(collect_game_files(&path)?);
        } else if matches!(file_type(&path), FileType::Wthor | FileType::Pgn) {
            files.push(path);
        }
    }

    Ok(files)
}

/// Load games from a list of paths (files or directories).
/// Directories are scanned recursively for .wtb, .pgn, .txt files.
pub fn load_games(paths: &[String]) -> Result<Vec<Game>, String> {
    let mut all_file_paths: Vec<std::path::PathBuf> = Vec::new();

    for path_str in paths {
        let path = Path::new(path_str);

        if path.is_dir() {
            all_file_paths.extend(collect_game_files(path)?);
        } else if path.is_file() {
            all_file_paths.push(path.to_path_buf());
        } else {
            eprintln!("Warning: {} does not exist, skipping", path.display());
        }
    }

    if all_file_paths.is_empty() {
        return Err("No game files found".to_string());
    }

    let mut all_games = Vec::new();
    for file_path in &all_file_paths {
        let ft = file_type(file_path);
        eprintln!("Loading {} ({:?})...", file_path.display(), ft);

        let games = match ft {
            FileType::Wthor => read_wthor_file(file_path)?,
            FileType::Pgn => read_pgn_file(file_path)?,
            FileType::Unknown => {
                eprintln!("  Unknown file type, skipping");
                continue;
            }
        };

        eprintln!(
            "  Loaded {} games, {} positions",
            games.len(),
            games.iter().map(|g| g.positions.len()).sum::<usize>()
        );
        all_games.extend(games);
    }

    eprintln!(
        "Total: {} games from {} files",
        all_games.len(),
        all_file_paths.len()
    );
    Ok(all_games)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_move_from_wthor() {
        let cases = [
            (11, 0),  // A1
            (18, 7),  // H1
            (21, 8),  // A2
            (28, 15), // H2
            (81, 56), // A8
            (88, 63), // H8
        ];
        for (input, expected) in cases {
            assert_eq!(move_from_wthor(input), expected, "move_from_wthor({input})");
        }
    }

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
