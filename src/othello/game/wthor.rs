// WTHOR (.wtb) binary format reader.
//
// Each game record is 68 bytes: 8 bytes of metadata + 60 move bytes.
// Moves are encoded as 10×row + col + 1 (rows/cols 1-8), converted to
// Edax cell indices (0-63) for replay.

use crate::othello::board::Board;
use crate::othello::position::{Cell, Position};
use std::fs;
use std::path::Path;

use super::flip_discs;

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
pub fn read_wthor_file(path: &Path) -> Result<Vec<super::Game>, String> {
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
            games.push(super::Game {
                positions,
                black_name: None,
                white_name: None,
                result_score: None,
            });
        }
    }

    Ok(games)
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
    fn test_read_wthor_file_sample() {
        let path = std::path::Path::new("test_data/sample.wtb");
        let games = read_wthor_file(path).expect("failed to read sample WTB");
        assert!(
            !games.is_empty(),
            "sample WTB should contain at least one game"
        );
        assert!(
            !games[0].positions.is_empty(),
            "game should have generated positions"
        );
    }
}
