//! AVX2 vectorized mobility: the four ray *axes* run in parallel across the four
//! 64-bit lanes of a 256-bit register, using per-lane variable shifts
//! (`sllv`/`srlv`). Lane order (low→high) is shift {1, 8, 7, 9} = horizontal,
//! vertical, the two diagonals. One pass does the four forward (left) Kogge-Stone
//! occluded fills, one the four backward (right) fills; each propagates the
//! player's reach through opponent runs, and the legal move sits one step beyond.
//! Mirrors Edax `get_moves_avx`.
//!
//! SIMD intrinsics require `unsafe` (allowed locally; the crate otherwise denies
//! it) and the variant is unused unless selected by `cfg(target_feature)` or
//! benchmarked, hence `dead_code`.
#![allow(unsafe_code, dead_code)]

use core::arch::x86_64::*;

const MASK_EDGE: i64 = 0x7E7E_7E7E_7E7E_7E7Eu64 as i64; // B–G columns (no wrap)
const MASK_ALL: i64 = -1; // 0xFFFF... for the vertical lane

/// Bitboard of all legal moves for `player` against `opponent`.
///
/// # Safety
/// The CPU must support the AVX2 instruction set.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn get_moves(player: u64, opponent: u64) -> u64 {
    let pp = _mm256_set1_epi64x(player as i64);
    // Lane order (low→high): shift 1, 8, 7, 9.
    let shift = _mm256_set_epi64x(9, 7, 8, 1);
    let shift2 = _mm256_add_epi64(shift, shift);
    // Vertical (shift 8) needs no edge mask; the others mask off A/H columns.
    let moo = _mm256_and_si256(
        _mm256_set1_epi64x(opponent as i64),
        _mm256_set_epi64x(MASK_EDGE, MASK_EDGE, MASK_ALL, MASK_EDGE),
    );

    // Forward fill (left shifts).
    let mut fl = _mm256_and_si256(moo, _mm256_sllv_epi64(pp, shift));
    fl = _mm256_or_si256(fl, _mm256_and_si256(moo, _mm256_sllv_epi64(fl, shift)));
    let pre_l = _mm256_and_si256(moo, _mm256_sllv_epi64(moo, shift));
    fl = _mm256_or_si256(fl, _mm256_and_si256(pre_l, _mm256_sllv_epi64(fl, shift2)));
    fl = _mm256_or_si256(fl, _mm256_and_si256(pre_l, _mm256_sllv_epi64(fl, shift2)));
    let mut mm = _mm256_sllv_epi64(fl, shift);

    // Backward fill (right shifts).
    let mut fr = _mm256_and_si256(moo, _mm256_srlv_epi64(pp, shift));
    fr = _mm256_or_si256(fr, _mm256_and_si256(moo, _mm256_srlv_epi64(fr, shift)));
    let pre_r = _mm256_and_si256(moo, _mm256_srlv_epi64(moo, shift));
    fr = _mm256_or_si256(fr, _mm256_and_si256(pre_r, _mm256_srlv_epi64(fr, shift2)));
    fr = _mm256_or_si256(fr, _mm256_and_si256(pre_r, _mm256_srlv_epi64(fr, shift2)));
    mm = _mm256_or_si256(mm, _mm256_srlv_epi64(fr, shift));

    // Fold the four lanes down to one u64, then drop occupied squares.
    let lo = _mm256_castsi256_si128(mm);
    let hi = _mm256_extracti128_si256(mm, 1);
    let or2 = _mm_or_si128(lo, hi);
    let or1 = _mm_or_si128(or2, _mm_unpackhi_epi64(or2, or2));
    (_mm_cvtsi128_si64(or1) as u64) & !(player | opponent)
}
