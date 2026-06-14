//! BMI2 flip: gather each line with `PEXT`, look up the in-line flip, scatter
//! back with `PDEP`. `PEXT(bb, mask)` compacts the masked bits to a contiguous
//! field, so the move sits at compacted index `popcount(mask & (move_bit - 1))`
//! and the line tables apply directly; `PDEP` is the exact inverse.
//!
//! Always compiled on x86-64 (the `target_feature` attribute enables the
//! intrinsics for this function alone); callers must ensure BMI2 is present
//! (static `cfg(target_feature)` in production, runtime detection in tests/bench).
//!
//! SIMD intrinsics require `unsafe`; the crate otherwise denies it, so it is
//! allowed locally for this module only. `dead_code` is allowed because the
//! variant is unused unless selected by `cfg(target_feature)` or benchmarked.
#![allow(unsafe_code, dead_code)]

use super::line::{inline_flip, LINE_MASKS};
use core::arch::x86_64::{_pdep_u64, _pext_u64};

/// Discs flipped by `player` playing at `mv` (assumed empty).
///
/// # Safety
/// The CPU must support the BMI2 instruction set.
#[target_feature(enable = "bmi2")]
pub(crate) unsafe fn flip(mv: u32, player: u64, opponent: u64) -> u64 {
    let move_bit = 1u64 << (mv & 63);
    let masks = &LINE_MASKS[(mv & 63) as usize];
    let mut flipped = 0u64;
    for &m in masks {
        let p = _pext_u64(player, m);
        let o = _pext_u64(opponent, m);
        let t = (m & (move_bit - 1)).count_ones();
        let fl = inline_flip(t, p as u8, o as u8) as u64;
        flipped |= _pdep_u64(fl, m);
    }
    flipped
}
