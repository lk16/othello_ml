//! Portable count-last-flip: one `COUNT_FLIP` lookup per line, gathered by
//! shift (row), file-gather multiply (column), or diagonal replicate-multiply
//! (diagonals). The kindergarten technique — no SIMD, no per-square table. The
//! production default (portable; see the SIMD note in [`super`]).

use super::tables::COUNT_FLIP;

/// Diagonal masks per square: `[0]` = the ╲ diagonal, `[1]` = the ╱ diagonal.
const MASK_DIAG: [[u64; 64]; 2] = {
    let mut m = [[0u64; 64]; 2];
    let mut pos = 0;
    while pos < 64 {
        let x = (pos % 8) as i32;
        let y = (pos / 8) as i32;
        let mut sq = 0;
        while sq < 64 {
            let sx = sq % 8;
            let sy = sq / 8;
            if sx - sy == x - y {
                m[0][pos] |= 1u64 << sq;
            }
            if sx + sy == x + y {
                m[1][pos] |= 1u64 << sq;
            }
            sq += 1;
        }
        pos += 1;
    }
    m
};

/// Gather column `x` (bits x, x+8, …, x+56) into 8 contiguous bits, bit r = row r.
#[inline]
fn pack_v(p: u64, x: u32) -> usize {
    (((p >> x) & 0x0101_0101_0101_0101).wrapping_mul(0x0102_0408_1020_4080) >> 56) as usize
}

/// Gather a diagonal-masked bitboard into 8 bits, bit c = column c.
#[inline]
fn pack_d(pm: u64) -> usize {
    (pm.wrapping_mul(0x0101_0101_0101_0101) >> 56) as usize
}

/// 2× the discs `player` flips by playing at `pos`, valid only when `pos` is the
/// board's only empty square (the 1-empty leaf invariant).
#[inline]
pub(crate) fn count_last_flip(pos: u32, player: u64) -> i32 {
    let x = (pos & 7) as usize;
    let y = (pos >> 3) as usize;
    let mut n = COUNT_FLIP[x][((player >> (y * 8)) & 0xFF) as usize]; // row
    n += COUNT_FLIP[y][pack_v(player, x as u32)]; // column
    n += COUNT_FLIP[x][pack_d(player & MASK_DIAG[0][pos as usize])]; // ╲
    n += COUNT_FLIP[x][pack_d(player & MASK_DIAG[1][pos as usize])]; // ╱
    n as i32
}
