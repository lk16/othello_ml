//! Scalar branchless mobility: a Kogge-Stone occluded fill in each of the eight
//! ray directions, one direction at a time. The portable production default.

/// Bitboard of all legal moves for `player` against `opponent`.
#[inline]
pub(crate) fn get_moves(player: u64, opponent: u64) -> u64 {
    // Middle columns mask (excludes A and H) prevents horizontal and diagonal
    // bitboard wrapping. Vertical shifts need no mask.
    const MIDDLE_COLUMNS: u64 = 0x7E7E7E7E7E7E7E7E;
    let mask = opponent & MIDDLE_COLUMNS;
    let mut moves: u64 = 0;

    // Horizontal (shift 1)
    let mut flip_l = mask & (player << 1);
    flip_l |= mask & (flip_l << 1);
    let mask_l = mask & (mask << 1);
    flip_l |= mask_l & (flip_l << 2);
    flip_l |= mask_l & (flip_l << 2);
    let mut flip_r = mask & (player >> 1);
    flip_r |= mask & (flip_r >> 1);
    let mask_r = mask & (mask >> 1);
    flip_r |= mask_r & (flip_r >> 2);
    flip_r |= mask_r & (flip_r >> 2);
    moves |= (flip_l << 1) | (flip_r >> 1);

    // Diagonal / (shift 7)
    let mut flip_l = mask & (player << 7);
    flip_l |= mask & (flip_l << 7);
    let mask_l = mask & (mask << 7);
    flip_l |= mask_l & (flip_l << 14);
    flip_l |= mask_l & (flip_l << 14);
    let mut flip_r = mask & (player >> 7);
    flip_r |= mask & (flip_r >> 7);
    let mask_r = mask & (mask >> 7);
    flip_r |= mask_r & (flip_r >> 14);
    flip_r |= mask_r & (flip_r >> 14);
    moves |= (flip_l << 7) | (flip_r >> 7);

    // Diagonal \ (shift 9)
    let mut flip_l = mask & (player << 9);
    flip_l |= mask & (flip_l << 9);
    let mask_l = mask & (mask << 9);
    flip_l |= mask_l & (flip_l << 18);
    flip_l |= mask_l & (flip_l << 18);
    let mut flip_r = mask & (player >> 9);
    flip_r |= mask & (flip_r >> 9);
    let mask_r = mask & (mask >> 9);
    flip_r |= mask_r & (flip_r >> 18);
    flip_r |= mask_r & (flip_r >> 18);
    moves |= (flip_l << 9) | (flip_r >> 9);

    // Vertical (shift 8) — no column masking needed
    let mut flip_l = opponent & (player << 8);
    flip_l |= opponent & (flip_l << 8);
    let mask_l = opponent & (opponent << 8);
    flip_l |= mask_l & (flip_l << 16);
    flip_l |= mask_l & (flip_l << 16);
    let mut flip_r = opponent & (player >> 8);
    flip_r |= opponent & (flip_r >> 8);
    let mask_r = opponent & (opponent >> 8);
    flip_r |= mask_r & (flip_r >> 16);
    flip_r |= mask_r & (flip_r >> 16);
    moves |= (flip_l << 8) | (flip_r >> 8);

    moves & !(player | opponent)
}
