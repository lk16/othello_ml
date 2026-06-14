//! BMI2 count-last-flip: gather each line with `PEXT` and sum its `COUNT_FLIP`
//! entry. `PEXT(player, mask)` compacts the line's player discs to a contiguous
//! field, so the move sits at compacted index `popcount(mask & (move_bit - 1))`
//! and the shared table applies directly — no per-line shift/multiply gather.
//!
//! Always compiled on x86-64 (the `target_feature` attribute enables `PEXT` for
//! this function alone); callers must ensure BMI2 is present (static
//! `cfg(target_feature)` in production, runtime detection in tests/bench).
//!
//! SIMD intrinsics require `unsafe`; the crate otherwise denies it, so it is
//! allowed locally. `dead_code` is allowed because the variant is unused unless
//! selected by `cfg(target_feature)` or benchmarked.
#![allow(unsafe_code, dead_code)]

use super::tables::COUNT_FLIP;
use core::arch::x86_64::_pext_u64;

/// Per-square full-line masks `[row, column, diagonal-╲, diagonal-╱]`, each
/// including the square itself.
const LINE_MASKS: [[u64; 4]; 64] = {
    let mut masks = [[0u64; 4]; 64];
    let mut sq = 0usize;
    while sq < 64 {
        let r = (sq / 8) as i32;
        let c = (sq % 8) as i32;
        let mut row = 0u64;
        let mut col = 0u64;
        let mut back = 0u64; // ╲ : col - row constant
        let mut fwd = 0u64; // ╱ : col + row constant
        let mut s = 0usize;
        while s < 64 {
            let sr = (s / 8) as i32;
            let sc = (s % 8) as i32;
            if sr == r {
                row |= 1 << s;
            }
            if sc == c {
                col |= 1 << s;
            }
            if sc - sr == c - r {
                back |= 1 << s;
            }
            if sc + sr == c + r {
                fwd |= 1 << s;
            }
            s += 1;
        }
        masks[sq] = [row, col, back, fwd];
        sq += 1;
    }
    masks
};

/// 2× the discs `player` flips by playing at `pos`, valid only when `pos` is the
/// board's only empty square.
///
/// # Safety
/// The CPU must support the BMI2 instruction set.
#[target_feature(enable = "bmi2")]
pub(crate) unsafe fn count_last_flip(pos: u32, player: u64) -> i32 {
    let move_bit = 1u64 << (pos & 63);
    let masks = &LINE_MASKS[(pos & 63) as usize];
    let mut n = 0u32;
    for &m in masks {
        let p = _pext_u64(player, m) as u8;
        let t = (m & (move_bit - 1)).count_ones() as usize;
        n += COUNT_FLIP[t][p as usize] as u32;
    }
    n as i32
}
