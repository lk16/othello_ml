//! Count-last-flip by reusing the production flip primitive: compute the full
//! flip mask and double its popcount. Recomputes the whole mask rather than
//! summing per-line counts; a portable baseline for the benchmark. The
//! reconstructed opponent (`!player & !pos`) relies on the 1-empty full-board
//! invariant.

use crate::othello::flip::flip;

/// 2× the discs `player` flips by playing at `pos`, valid only when `pos` is the
/// board's only empty square.
#[allow(dead_code)]
#[inline]
pub(crate) fn count_last_flip(pos: u32, player: u64) -> i32 {
    let opponent = !player & !(1u64 << pos);
    2 * flip(pos, player, opponent).count_ones() as i32
}
