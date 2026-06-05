// Board representation for Othello (8x8 board)
// Uses two 64-bit integers: one for each player's discs
// This is a standard bitboard representation used in many game engines.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Board {
    // Bitboards: bit position i represents cell i (0-63)
    // Cell mapping: a1=0, b1=1, ..., h1=7 (rank 1)
    //              a2=8, b2=9, ..., h2=15 (rank 2)
    //              ...
    //              a8=56, b8=57, ..., h8=63 (rank 8)
    pub player: u64,   // Current player's discs
    pub opponent: u64, // Opponent's discs
}

impl Board {
    /// Create an empty board
    pub fn new() -> Self {
        Board {
            player: 0,
            opponent: 0,
        }
    }

    /// Create initial Othello position
    pub fn initial() -> Self {
        // d4=27, e4=28, d5=35, e5=36
        // Initially: player (black) at d5=35, e4=28
        // opponent (white) at e5=36, d4=27
        Board {
            player: (1u64 << 35) | (1u64 << 28),
            opponent: (1u64 << 36) | (1u64 << 27),
        }
    }

    /// Get piece at cell (0-63)
    pub fn get_cell(&self, cell: u32) -> Cell {
        let bit = 1u64 << cell;
        if self.player & bit != 0 {
            Cell::Player
        } else if self.opponent & bit != 0 {
            Cell::Opponent
        } else {
            Cell::Empty
        }
    }

    /// Count discs for current player
    pub fn player_discs(&self) -> u32 {
        self.player.count_ones()
    }

    /// Count discs for opponent
    pub fn opponent_discs(&self) -> u32 {
        self.opponent.count_ones()
    }

    /// Count empty cells
    pub fn empties(&self) -> u32 {
        64 - self.player_discs() - self.opponent_discs()
    }

    /// Place a disc for current player at cell
    pub fn place_disc(&mut self, cell: u32) {
        let bit = 1u64 << cell;
        self.player |= bit;
    }

    /// Get all occupied cells
    pub fn occupied(&self) -> u64 {
        self.player | self.opponent
    }

    /// Get all empty cells
    pub fn empty_cells(&self) -> u64 {
        !self.occupied()
    }

    /// Convert cell index (0-63) to x8 board representation (same as Edax)
    /// For compatibility with Edax patterns
    pub fn cell_to_coords(cell: u32) -> (u32, u32) {
        (cell % 8, cell / 8)
    }

    /// Compute a bitboard of all legal moves for the current player.
    ///
    /// A move is legal at an empty cell if placing a disc there would flip
    /// at least one opponent disc in any of the 8 directions.
    pub fn get_moves(&self) -> u64 {
        // Mask: exclude the rightmost column to prevent horizontal wrapping
        // 0x7E7E7E7E7E7E7E7E = all columns except 'H'
        let mask = self.opponent & 0x7E7E7E7E7E7E7E7E;
        let mut moves: u64 = 0;

        // Horizontal / vertical / diagonal shift amounts
        for &shift in &[1, 7, 9, 8] {
            // Direction: positive shift (left/up)
            let mut flip = mask & (self.player << shift);
            flip |= mask & (flip << shift);
            let mask_dir = mask & (mask << shift);
            flip |= mask_dir & (flip << (2 * shift));
            flip |= mask_dir & (flip << (2 * shift));
            moves |= flip << shift;

            // Direction: negative shift (right/down)
            let mut flip = mask & (self.player >> shift);
            flip |= mask & (flip >> shift);
            let mask_dir = mask & (mask >> shift);
            flip |= mask_dir & (flip >> (2 * shift));
            flip |= mask_dir & (flip >> (2 * shift));
            moves |= flip >> shift;
        }

        // Only empty cells are legal moves, mask to 64 bits
        moves & !(self.player | self.opponent)
    }

    /// Check whether the current player has any legal moves.
    pub fn has_moves(&self) -> bool {
        self.get_moves() != 0
    }

    /// Check whether the game is over (neither player has legal moves).
    pub fn is_game_end(&self) -> bool {
        !self.has_moves() && !self.pass_move().has_moves()
    }

    /// Return the board after a pass (swap sides).
    ///
    /// The current player has no legal moves so they pass;
    /// the opponent becomes the new side to move.
    pub fn pass_move(&self) -> Board {
        Board {
            player: self.opponent,
            opponent: self.player,
        }
    }

    /// Exact final score from the perspective of the side to move.
    ///
    /// At game end, all remaining empty squares go to the winner.
    /// - If side to move wins: 64 - 2×opponent_discs  (positive)
    /// - If opponent wins:    2×player_discs - 64   (negative)
    /// - Tie: 0
    pub fn final_score(&self) -> i32 {
        let me = self.player_discs() as i32;
        let opp = self.opponent_discs() as i32;

        if me > opp {
            64 - 2 * opp
        } else if opp > me {
            -64 + 2 * me
        } else {
            0
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cell {
    Player,
    Opponent,
    Empty,
}

impl Default for Board {
    fn default() -> Self {
        Board::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_initial_board() {
        let board = Board::initial();
        assert_eq!(board.player_discs(), 2);
        assert_eq!(board.opponent_discs(), 2);
        assert_eq!(board.empties(), 60);
    }

    #[test]
    fn test_get_cell() {
        let board = Board::initial();
        assert_eq!(board.get_cell(35), Cell::Player);
        assert_eq!(board.get_cell(28), Cell::Player);
        assert_eq!(board.get_cell(36), Cell::Opponent);
        assert_eq!(board.get_cell(27), Cell::Opponent);
        assert_eq!(board.get_cell(0), Cell::Empty);
    }

    #[test]
    fn test_place_disc() {
        let mut board = Board::new();
        board.place_disc(0);
        assert_eq!(board.get_cell(0), Cell::Player);
        assert_eq!(board.player_discs(), 1);
    }

    #[test]
    fn test_initial_has_moves() {
        let board = Board::initial();
        assert!(board.has_moves());
        // The initial position has exactly 4 legal moves for black
        assert_eq!(board.get_moves().count_ones(), 4);
    }

    #[test]
    fn test_pass_move() {
        let board = Board::initial();
        let passed = board.pass_move();
        // After pass, former opponent becomes player
        assert_eq!(passed.player, board.opponent);
        assert_eq!(passed.opponent, board.player);
    }

    #[test]
    fn test_is_not_game_end_initially() {
        let board = Board::initial();
        assert!(!board.is_game_end());
    }

    #[test]
    fn test_game_end_and_final_score() {
        // Create a game-end position: black controls entire board
        let board = Board {
            player: 0xFFFFFFFFFFFFFFFF, // all discs
            opponent: 0,
        };
        assert!(!board.has_moves()); // no empty squares
        assert!(board.is_game_end());
        // Player wins: 64 - 2*0 = 64
        assert_eq!(board.final_score(), 64);

        // Opponent wins
        let board = Board {
            player: 0,
            opponent: 0xFFFFFFFFFFFFFFFF,
        };
        assert_eq!(board.final_score(), -64);

        // Tie: 32 each
        let board = Board {
            player: 0x00000000FFFFFFFF,
            opponent: 0xFFFFFFFF00000000,
        };
        assert_eq!(board.final_score(), 0);
    }

    #[test]
    fn test_get_moves_empty_board() {
        // An empty board: no opponent discs to flip → no moves
        let board = Board::new();
        assert!(!board.has_moves());
        assert_eq!(board.get_moves(), 0);
    }
}
