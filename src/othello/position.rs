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

    /// The 8 board symmetries (dihedral group D4) of this position, applied to
    /// both bitboards. The exact game value is invariant under every one of them,
    /// so a training example and its 7 images share the same target score — this is
    /// what `train-exact`'s 8-fold augmentation exploits (see docs/eval-quality.md).
    /// Element 0 is the identity (`== *self`).
    pub fn symmetries(&self) -> [Position; 8] {
        std::array::from_fn(|k| Position {
            player: board_symmetry(self.player, k),
            opponent: board_symmetry(self.opponent, k),
        })
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
    /// discs. The single source of the flip computation; `flipped` wraps it with
    /// the occupied-square check. Assumes `mv` is empty (no such check), which
    /// lets the endgame leaf solvers skip it on the hot path.
    ///
    /// Delegates to the target-selected variant in [`crate::othello::flip`].
    #[inline]
    pub(crate) fn flip_mask(mv: u32, player: u64, opponent: u64) -> u64 {
        crate::othello::flip::flip(mv, player, opponent)
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
    /// Delegates to the target-selected variant in [`crate::othello::get_moves`].
    #[inline]
    pub fn get_moves(&self) -> u64 {
        crate::othello::get_moves::get_moves(self.player, self.opponent)
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

/// Apply the `k`-th board symmetry (0..8, dihedral group D4) to a bitboard.
/// `k == 0` is the identity. Composes the standard LERF routines `flip_vertical`
/// (`swap_bytes`), `rotate180` (`reverse_bits`), `mirror_files`, `flip_diag_a1h8`.
/// Shared by [`Position::symmetries`] and the feature-symmetry grouping
/// (`Features::symmetry_shapes`), which needs the induced cell permutation.
pub fn board_symmetry(b: u64, k: usize) -> u64 {
    match k {
        0 => b,                                // identity
        1 => b.swap_bytes(),                   // flip vertical (mirror ranks)
        2 => mirror_files(b),                  // flip horizontal (mirror files)
        3 => b.reverse_bits(),                 // rotate 180
        4 => flip_diag_a1h8(b),                // transpose (a1-h8 diagonal)
        5 => flip_diag_a1h8(b.reverse_bits()), // anti-diagonal (a8-h1)
        6 => flip_diag_a1h8(b).swap_bytes(),   // rotate 90
        7 => mirror_files(flip_diag_a1h8(b)),  // rotate 270
        _ => panic!("board symmetry index out of range: {k}"),
    }
}

/// Mirror a bitboard horizontally (file a <-> h): reverse the bits within each
/// rank byte. Standard Chess-Programming-Wiki `mirrorHorizontal` for LERF boards.
fn mirror_files(b: u64) -> u64 {
    const K1: u64 = 0x5555555555555555;
    const K2: u64 = 0x3333333333333333;
    const K4: u64 = 0x0f0f0f0f0f0f0f0f;
    let b = ((b >> 1) & K1) | ((b & K1) << 1);
    let b = ((b >> 2) & K2) | ((b & K2) << 2);
    ((b >> 4) & K4) | ((b & K4) << 4)
}

/// Transpose a bitboard about the a1-h8 diagonal (swap ranks and files). Standard
/// Chess-Programming-Wiki `flipDiagA1H8` for LERF boards.
fn flip_diag_a1h8(b: u64) -> u64 {
    const K1: u64 = 0x5500550055005500;
    const K2: u64 = 0x3333000033330000;
    const K4: u64 = 0x0f0f0f0f00000000;
    let t = K4 & (b ^ (b << 28));
    let b = b ^ t ^ (t >> 28);
    let t = K2 & (b ^ (b << 14));
    let b = b ^ t ^ (t >> 14);
    let t = K1 & (b ^ (b << 7));
    b ^ t ^ (t >> 7)
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
    fn symmetries_are_structurally_consistent() {
        // An asymmetric position: a1 (player) + b1 (opponent) only.
        let p = Position {
            player: 1 << 0,
            opponent: 1 << 1,
        };
        let syms = p.symmetries();
        // Element 0 is the identity.
        assert_eq!(syms[0], p);
        // All 8 images are distinct for an asymmetric board, and each preserves the
        // disc counts and legal-move count (legality is symmetry-invariant) — a
        // strong check that every transform is a genuine board symmetry.
        for (i, a) in syms.iter().enumerate() {
            assert_eq!(a.player_discs(), p.player_discs());
            assert_eq!(a.opponent_discs(), p.opponent_discs());
            assert_eq!(a.get_moves().count_ones(), p.get_moves().count_ones());
            for (j, b) in syms.iter().enumerate() {
                if i != j {
                    assert_ne!(a, b, "symmetries {i} and {j} collide");
                }
            }
        }
    }

    #[test]
    fn symmetries_preserve_exact_score() {
        use crate::eval::alphabeta::exact_score;
        // A near-full board (8 empties → cheap exact) with an asymmetric disc split.
        let empties_mask: u64 = (1 << 0)
            | (1 << 1)
            | (1 << 2)
            | (1 << 3)
            | (1 << 9)
            | (1 << 10)
            | (1 << 18)
            | (1 << 27);
        let occ = !empties_mask;
        let p = Position {
            player: occ & 0xAAAA_AAAA_AAAA_AAAA,
            opponent: occ & 0x5555_5555_5555_5555,
        };
        let base = exact_score(&p);
        for (i, s) in p.symmetries().iter().enumerate() {
            assert_eq!(s.empties(), p.empties());
            assert_eq!(exact_score(s), base, "symmetry {i} changed the exact score");
        }
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
