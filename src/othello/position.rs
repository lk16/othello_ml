//! Position representation for Othello — a pair of bitboards (player + opponent).

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Position {
    // Bitboards: bit position i represents cell i (0-63)
    // Cell mapping: a1=0, b1=1, ..., h1=7 (rank 1)
    //              a2=8, b2=9, ..., h2=15 (rank 2)
    //              ...
    //              a8=56, b8=57, ..., h8=63 (rank 8)
    pub player: u64,   // Current player's discs
    pub opponent: u64, // Opponent's discs
}

impl Position {
    /// Create an empty position.
    pub fn new() -> Self {
        Position {
            player: 0,
            opponent: 0,
        }
    }

    /// Create initial Othello position.
    pub fn initial() -> Self {
        // d4=27, e4=28, d5=35, e5=36
        // Initially: player (black) at d5=35, e4=28
        // opponent (white) at e5=36, d4=27
        Position {
            player: (1u64 << 35) | (1u64 << 28),
            opponent: (1u64 << 36) | (1u64 << 27),
        }
    }

    /// Get piece at cell (0-63).
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

    /// Count discs for current player.
    pub fn player_discs(&self) -> u32 {
        self.player.count_ones()
    }

    /// Count discs for opponent.
    pub fn opponent_discs(&self) -> u32 {
        self.opponent.count_ones()
    }

    /// Count empty cells.
    pub fn empties(&self) -> u32 {
        64 - self.player_discs() - self.opponent_discs()
    }

    /// Count total occupied cells.
    pub fn count_discs(&self) -> u32 {
        (self.player | self.opponent).count_ones()
    }

    /// Bitboard of all opponent discs that would be flipped by playing at `mv`.
    ///
    /// Returns 0 if `mv` is occupied or would not flip any discs.
    /// Adapted from Edax / flippy.
    pub fn flipped(&self, mv: u32) -> u64 {
        if (self.player | self.opponent) & (1u64 << mv) != 0 {
            return 0;
        }
        Self::flip_mask(mv, self.player, self.opponent)
    }

    /// Discs flipped by `player` playing at `mv`, with `opponent` the other
    /// discs. The single source of the 8-direction flip computation; `flipped`
    /// wraps it with the occupied-square check. Assumes `mv` is empty (no such
    /// check), which lets the endgame leaf solvers skip it on the hot path.
    /// Adapted from Edax / flippy.
    pub(crate) fn flip_mask(mv: u32, player: u64, opponent: u64) -> u64 {
        let move_bit = 1u64 << mv;

        // Horizontal and diagonal shifts need col B-G mask to prevent wrap-around.
        const MIDDLE_COLUMNS: u64 = 0x7E7E7E7E7E7E7E7E;

        let opp_h = opponent & MIDDLE_COLUMNS;
        let opp_v = opponent;
        let opp_d = opponent & MIDDLE_COLUMNS;

        let mut flipped: u64 = 0;

        // Horizontal: shift left (<<1) and right (>>1)
        let mut f = opp_h & (move_bit << 1);
        f |= opp_h & (f << 1);
        f |= opp_h & (f << 1);
        f |= opp_h & (f << 1);
        f |= opp_h & (f << 1);
        f |= opp_h & (f << 1);
        if player & (f << 1) != 0 {
            flipped |= f;
        }

        let mut f = opp_h & (move_bit >> 1);
        f |= opp_h & (f >> 1);
        f |= opp_h & (f >> 1);
        f |= opp_h & (f >> 1);
        f |= opp_h & (f >> 1);
        f |= opp_h & (f >> 1);
        if player & (f >> 1) != 0 {
            flipped |= f;
        }

        // Vertical: shift up (<<8) and down (>>8)
        let mut f = opp_v & (move_bit << 8);
        f |= opp_v & (f << 8);
        f |= opp_v & (f << 8);
        f |= opp_v & (f << 8);
        f |= opp_v & (f << 8);
        f |= opp_v & (f << 8);
        if player & (f << 8) != 0 {
            flipped |= f;
        }

        let mut f = opp_v & (move_bit >> 8);
        f |= opp_v & (f >> 8);
        f |= opp_v & (f >> 8);
        f |= opp_v & (f >> 8);
        f |= opp_v & (f >> 8);
        f |= opp_v & (f >> 8);
        if player & (f >> 8) != 0 {
            flipped |= f;
        }

        // Diagonal /: shift <<7 and >>7
        let mut f = opp_d & (move_bit << 7);
        f |= opp_d & (f << 7);
        f |= opp_d & (f << 7);
        f |= opp_d & (f << 7);
        f |= opp_d & (f << 7);
        f |= opp_d & (f << 7);
        if player & (f << 7) != 0 {
            flipped |= f;
        }

        let mut f = opp_d & (move_bit >> 7);
        f |= opp_d & (f >> 7);
        f |= opp_d & (f >> 7);
        f |= opp_d & (f >> 7);
        f |= opp_d & (f >> 7);
        f |= opp_d & (f >> 7);
        if player & (f >> 7) != 0 {
            flipped |= f;
        }

        // Diagonal \: shift <<9 and >>9
        let mut f = opp_d & (move_bit << 9);
        f |= opp_d & (f << 9);
        f |= opp_d & (f << 9);
        f |= opp_d & (f << 9);
        f |= opp_d & (f << 9);
        f |= opp_d & (f << 9);
        if player & (f << 9) != 0 {
            flipped |= f;
        }

        let mut f = opp_d & (move_bit >> 9);
        f |= opp_d & (f >> 9);
        f |= opp_d & (f >> 9);
        f |= opp_d & (f >> 9);
        f |= opp_d & (f >> 9);
        f |= opp_d & (f >> 9);
        if player & (f >> 9) != 0 {
            flipped |= f;
        }

        flipped
    }

    /// Apply a move, returning the resulting position (opponent to move next).
    ///
    /// If the move is invalid (occupied cell or no flips), returns `*self`
    /// unchanged.
    /// Adapted from Edax / flippy.
    pub fn do_move(&self, mv: u32) -> Position {
        let flipped = self.flipped(mv);
        if flipped == 0 {
            return *self;
        }

        let opp = self.player | flipped | (1u64 << mv);
        let me = self.opponent & !opp;

        Position {
            player: me,
            opponent: opp,
        }
    }

    /// Get all occupied cells.
    pub fn occupied(&self) -> u64 {
        self.player | self.opponent
    }

    /// Get all empty cells.
    pub fn empty_cells(&self) -> u64 {
        !self.occupied()
    }

    /// Convert cell index (0-63) to x8 board representation.
    pub fn cell_to_coords(cell: u32) -> (u32, u32) {
        (cell % 8, cell / 8)
    }

    /// Compute a bitboard of all legal moves for the current player.
    ///
    /// A move is legal at an empty cell if placing a disc there would flip
    /// at least one opponent disc in any of the 8 directions.
    ///
    /// Adapted from Edax / flippy.
    pub fn get_moves(&self) -> u64 {
        // Middle columns mask (excludes A and H) prevents horizontal and
        // diagonal bitboard wrapping.  Vertical shifts need no mask.
        const MIDDLE_COLUMNS: u64 = 0x7E7E7E7E7E7E7E7E;
        let mask = self.opponent & MIDDLE_COLUMNS;
        let mut moves: u64 = 0;

        // Horizontal (shift 1)
        let mut flip_l = mask & (self.player << 1);
        flip_l |= mask & (flip_l << 1);
        let mask_l = mask & (mask << 1);
        flip_l |= mask_l & (flip_l << 2);
        flip_l |= mask_l & (flip_l << 2);
        let mut flip_r = mask & (self.player >> 1);
        flip_r |= mask & (flip_r >> 1);
        let mask_r = mask & (mask >> 1);
        flip_r |= mask_r & (flip_r >> 2);
        flip_r |= mask_r & (flip_r >> 2);
        moves |= (flip_l << 1) | (flip_r >> 1);

        // Diagonal / (shift 7)
        let mut flip_l = mask & (self.player << 7);
        flip_l |= mask & (flip_l << 7);
        let mask_l = mask & (mask << 7);
        flip_l |= mask_l & (flip_l << 14);
        flip_l |= mask_l & (flip_l << 14);
        let mut flip_r = mask & (self.player >> 7);
        flip_r |= mask & (flip_r >> 7);
        let mask_r = mask & (mask >> 7);
        flip_r |= mask_r & (flip_r >> 14);
        flip_r |= mask_r & (flip_r >> 14);
        moves |= (flip_l << 7) | (flip_r >> 7);

        // Diagonal \ (shift 9)
        let mut flip_l = mask & (self.player << 9);
        flip_l |= mask & (flip_l << 9);
        let mask_l = mask & (mask << 9);
        flip_l |= mask_l & (flip_l << 18);
        flip_l |= mask_l & (flip_l << 18);
        let mut flip_r = mask & (self.player >> 9);
        flip_r |= mask & (flip_r >> 9);
        let mask_r = mask & (mask >> 9);
        flip_r |= mask_r & (flip_r >> 18);
        flip_r |= mask_r & (flip_r >> 18);
        moves |= (flip_l << 9) | (flip_r >> 9);

        // Vertical (shift 8) — no column masking needed
        let mut flip_l = self.opponent & (self.player << 8);
        flip_l |= self.opponent & (flip_l << 8);
        let mask_l = self.opponent & (self.opponent << 8);
        flip_l |= mask_l & (flip_l << 16);
        flip_l |= mask_l & (flip_l << 16);
        let mut flip_r = self.opponent & (self.player >> 8);
        flip_r |= self.opponent & (flip_r >> 8);
        let mask_r = self.opponent & (self.opponent >> 8);
        flip_r |= mask_r & (flip_r >> 16);
        flip_r |= mask_r & (flip_r >> 16);
        moves |= (flip_l << 8) | (flip_r >> 8);

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
    pub fn pass_move(&self) -> Position {
        Position {
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

    /// Convert to an Edax FEN string.
    ///
    /// 64 characters (A1..H1, A2..H2, …, A8..H8) using:
    ///   `X` = black disc, `O` = white disc, `-` = empty
    /// Followed by a space and the side to move (`X` for black, `O` for white).
    pub fn to_fen(&self, black_to_move: bool) -> String {
        let mut fen = String::with_capacity(66);
        for i in 0..64 {
            let bit = 1u64 << i;
            let (is_black, is_white) = if black_to_move {
                (self.player & bit != 0, self.opponent & bit != 0)
            } else {
                (self.opponent & bit != 0, self.player & bit != 0)
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
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cell {
    Player,
    Opponent,
    Empty,
}

impl Default for Position {
    fn default() -> Self {
        Position::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_initial_board() {
        let board = Position::initial();
        assert_eq!(board.player_discs(), 2);
        assert_eq!(board.opponent_discs(), 2);
        assert_eq!(board.empties(), 60);
    }

    #[test]
    fn test_get_cell() {
        let board = Position::initial();
        assert_eq!(board.get_cell(35), Cell::Player);
        assert_eq!(board.get_cell(28), Cell::Player);
        assert_eq!(board.get_cell(36), Cell::Opponent);
        assert_eq!(board.get_cell(27), Cell::Opponent);
        assert_eq!(board.get_cell(0), Cell::Empty);
    }

    #[test]
    fn test_initial_has_moves() {
        let board = Position::initial();
        assert!(board.has_moves());
        // The initial position has exactly 4 legal moves for black
        assert_eq!(board.get_moves().count_ones(), 4);
    }

    #[test]
    fn test_pass_move() {
        let board = Position::initial();
        let passed = board.pass_move();
        // After pass, former opponent becomes player
        assert_eq!(passed.player, board.opponent);
        assert_eq!(passed.opponent, board.player);
    }

    #[test]
    fn test_is_not_game_end_initially() {
        let board = Position::initial();
        assert!(!board.is_game_end());
    }

    #[test]
    fn test_game_end_and_final_score() {
        let board = Position {
            player: u64::MAX,
            opponent: 0,
        };
        assert!(!board.has_moves());
        assert!(board.is_game_end());
        assert_eq!(board.final_score(), 64);

        let board = Position {
            player: 0,
            opponent: u64::MAX,
        };
        assert_eq!(board.final_score(), -64);

        let board = Position {
            player: 0x00000000FFFFFFFF,
            opponent: 0xFFFFFFFF00000000,
        };
        assert_eq!(board.final_score(), 0);
    }

    #[test]
    fn test_get_moves_empty_board() {
        let board = Position::new();
        assert!(!board.has_moves());
        assert_eq!(board.get_moves(), 0);
    }

    #[test]
    fn test_do_move_initial() {
        // Black plays f5 (cell 37) in the initial position
        let pos = Position::initial();
        let next = pos.do_move(37);
        // Black places f5, flips e5.  White loses e5.
        // After swap: player=white(1), opponent=black(4)
        assert_eq!(next.player_discs(), 1); // white: 2 original, lost 1 (e5 flipped)
        assert_eq!(next.opponent_discs(), 4); // black: 2 original + 1 placed + 1 flipped
        assert!(next.has_moves());
    }

    #[test]
    fn test_do_move_invalid_occupied() {
        let pos = Position::initial();
        // d5 (cell 35) already has a black disc
        let result = pos.do_move(35);
        assert_eq!(result, pos); // unchanged
    }

    #[test]
    fn test_do_move_invalid_no_flips() {
        let pos = Position::new();
        // No opponent discs → no flips possible
        let result = pos.do_move(0);
        assert_eq!(result, pos); // unchanged
    }

    #[test]
    fn test_flipped_initial() {
        let pos = Position::initial();
        // Black playing f5 (cell 37) should flip e5 (cell 36)
        let flipped = pos.flipped(37);
        assert_eq!(flipped, 1u64 << 36);
    }

    #[test]
    fn test_flipped_invalid_occupied() {
        let pos = Position::initial();
        assert_eq!(pos.flipped(35), 0); // already occupied
    }

    #[test]
    fn test_flipped_invalid_no_flips() {
        let pos = Position::new();
        assert_eq!(pos.flipped(0), 0);
    }

    #[test]
    fn test_get_moves_corner_capture() {
        // Regression: column A disc should be capturable by a vertical move.
        // White to move, can play A8 capturing A7.
        // Build position: white=O=player, black=X=opponent
        // a8 empty, a7 black, a6 white
        let mut pos = Position::new();
        pos.player |= 1u64 << 40; // a6 white
        pos.opponent |= 1u64 << 48; // a7 black
        let moves = pos.get_moves();
        assert!(moves & (1u64 << 56) != 0, "A8 should be a legal move");
    }

    #[test]
    fn test_get_moves_no_phantom_wrap() {
        // Regression: player disc at A8 should not wrap to H7 via horizontal shift.
        // This position has X to move, but X has no legal moves (must pass).
        // If get_moves is buggy, it finds a phantom move at g7 (cell 54).
        let fen = "OOOOOOOOOXXXXOXOOXXXOXXOOXOOXXOOOXOXXXOOOOXOXXXOOOOXXX-OXXXXXXXO X";
        // Parse FEN manually
        let board = fen.as_bytes();
        let mut x_discs: u64 = 0;
        let mut o_discs: u64 = 0;
        for i in 0..64 {
            match board[i] {
                b'X' => x_discs |= 1u64 << i,
                b'O' => o_discs |= 1u64 << i,
                _ => {}
            }
        }
        let pos = Position {
            player: x_discs,
            opponent: o_discs,
        };
        assert!(!pos.has_moves(), "X should have no legal moves (must pass)");
        assert_eq!(pos.get_moves(), 0);
    }

    #[test]
    fn test_do_move_sequence() {
        // Play a known opening and verify disc counts.
        let pos = Position::initial();
        // Black f5 (cell 37) — captures e5
        let next = pos.do_move(37);
        assert_eq!((next.opponent_discs(), next.player_discs()), (4, 1));
        // White d6 (cell 43) — captures d5
        let next = next.do_move(43);
        assert_eq!((next.opponent_discs(), next.player_discs()), (3, 3));
    }
}
