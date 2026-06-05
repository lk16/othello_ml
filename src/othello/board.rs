// Board type for Othello board positions extracted from games.

use crate::othello::position::Position;

/// A single board position extracted from a game, with side-to-move information.
#[derive(Debug, Clone)]
pub struct Board {
    pub position: Position,
    /// Which player's turn it is (true = black/player side to move)
    pub black_to_move: bool,
}

impl Board {
    /// Count empty cells on this board.
    pub fn empties(&self) -> u32 {
        self.position.empties()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_board_empties_initial() {
        let board = Board {
            position: Position::initial(),
            black_to_move: true,
        };
        assert_eq!(board.empties(), 60);
    }

    #[test]
    fn test_board_empties_empty_board() {
        let board = Board {
            position: Position::new(),
            black_to_move: true,
        };
        assert_eq!(board.empties(), 64);
    }

    #[test]
    fn test_board_empties_full_board() {
        let mut pos = Position::new();
        // Fill the board: place discs in all 64 cells (alternating sides just to fill)
        for i in 0..64 {
            pos.place_disc(i);
        }
        let board = Board {
            position: pos,
            black_to_move: true,
        };
        assert_eq!(board.empties(), 0);
    }
}
