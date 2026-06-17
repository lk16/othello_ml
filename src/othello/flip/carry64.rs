//! Portable line-table flip (the kindergarten/carry technique). Each of the
//! four lines is gathered into an 8-bit value — by shift (row), by the
//! file-gather multiply (column), or by the diagonal replicate-multiply
//! (diagonals) — the in-line flip is looked up in the shared tables, then
//! scattered back the same way. No SIMD, no per-square function table.

use super::line::{inline_flip, LINE_MASKS};

const FILE_A: u64 = 0x0101_0101_0101_0101;
const DIAG_REPLICATE: u64 = 0x0101_0101_0101_0101;
const FILE_GATHER: u64 = 0x0102_0408_1020_4080;

/// Spread 8 bits to board positions 0, 8, …, 56 (inverse of the column gather).
#[inline]
fn expand_to_file(b: u8) -> u64 {
    let mut x = b as u64;
    x = (x | (x << 28)) & 0x0000_000F_0000_000F;
    x = (x | (x << 14)) & 0x0003_0003_0003_0003;
    x = (x | (x << 7)) & FILE_A;
    x
}

/// Discs flipped by `player` playing at `mv` (assumed empty).
///
/// A benchmark/selection alternative to the production [`super::flip`].
#[allow(dead_code)]
#[inline]
pub(crate) fn flip(mv: u32, player: u64, opponent: u64) -> u64 {
    let sq = (mv & 63) as usize;
    let r = (sq / 8) as u32;
    let c = (sq % 8) as u32;
    let [_, _, back, fwd] = LINE_MASKS[sq];
    let mut flipped = 0u64;

    // Row: already contiguous; move at column c.
    let p = (player >> (8 * r)) & 0xFF;
    let o = (opponent >> (8 * r)) & 0xFF;
    flipped |= (inline_flip(c, p as u8, o as u8) as u64) << (8 * r);

    // Column: gather to a byte (bit = row); move at row r.
    let p = (((player >> c) & FILE_A).wrapping_mul(FILE_GATHER)) >> 56;
    let o = (((opponent >> c) & FILE_A).wrapping_mul(FILE_GATHER)) >> 56;
    flipped |= expand_to_file(inline_flip(r, p as u8, o as u8)) << c;

    // Diagonal ╲: gather by replicate (bit = column); move at column c.
    let p = (player & back).wrapping_mul(DIAG_REPLICATE) >> 56;
    let o = (opponent & back).wrapping_mul(DIAG_REPLICATE) >> 56;
    flipped |= (inline_flip(c, p as u8, o as u8) as u64).wrapping_mul(DIAG_REPLICATE) & back;

    // Diagonal ╱: same, other diagonal.
    let p = (player & fwd).wrapping_mul(DIAG_REPLICATE) >> 56;
    let o = (opponent & fwd).wrapping_mul(DIAG_REPLICATE) >> 56;
    flipped |= (inline_flip(c, p as u8, o as u8) as u64).wrapping_mul(DIAG_REPLICATE) & fwd;

    flipped
}
