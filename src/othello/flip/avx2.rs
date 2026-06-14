//! AVX2 vectorized flip: the four line directions run in parallel across the
//! four 64-bit lanes of a 256-bit register, using per-lane variable shifts
//! (`sllv`/`srlv`). One pass does the four forward shifts {1, 8, 7, 9}, one the
//! four backward shifts; each is a Kogge-Stone occluded fill of the move bit
//! through opponent discs, kept only where a player disc closes the run.
//!
//! SIMD intrinsics require `unsafe` (allowed locally; the crate otherwise denies
//! it) and the variant is unused unless selected by `cfg(target_feature)` or
//! benchmarked, hence `dead_code`.
#![allow(unsafe_code, dead_code)]

use core::arch::x86_64::*;

const MASK_EDGE: i64 = 0x7E7E_7E7E_7E7E_7E7Eu64 as i64; // B–G columns (no wrap)
const MASK_ALL: i64 = -1; // 0xFFFF... for the vertical lane

/// Discs flipped by `player` playing at `mv` (assumed empty).
///
/// # Safety
/// The CPU must support the AVX2 instruction set.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn flip(mv: u32, player: u64, opponent: u64) -> u64 {
    let move_bit = 1u64 << (mv & 63);
    let move_v = _mm256_set1_epi64x(move_bit as i64);
    let player_v = _mm256_set1_epi64x(player as i64);
    // Lane order (low→high): shift 1, 8, 7, 9.
    let shifts = _mm256_set_epi64x(9, 7, 8, 1);
    // Vertical (shift 8) needs no edge mask; the others mask off A/H columns.
    let opp_v = _mm256_and_si256(
        _mm256_set1_epi64x(opponent as i64),
        _mm256_set_epi64x(MASK_EDGE, MASK_EDGE, MASK_ALL, MASK_EDGE),
    );
    let zero = _mm256_setzero_si256();

    // Forward fill (left shifts).
    let mut f = _mm256_and_si256(opp_v, _mm256_sllv_epi64(move_v, shifts));
    for _ in 0..5 {
        f = _mm256_or_si256(f, _mm256_and_si256(opp_v, _mm256_sllv_epi64(f, shifts)));
    }
    let bound = _mm256_and_si256(player_v, _mm256_sllv_epi64(f, shifts));
    let fwd = _mm256_andnot_si256(_mm256_cmpeq_epi64(bound, zero), f);

    // Backward fill (right shifts).
    let mut f = _mm256_and_si256(opp_v, _mm256_srlv_epi64(move_v, shifts));
    for _ in 0..5 {
        f = _mm256_or_si256(f, _mm256_and_si256(opp_v, _mm256_srlv_epi64(f, shifts)));
    }
    let bound = _mm256_and_si256(player_v, _mm256_srlv_epi64(f, shifts));
    let bwd = _mm256_andnot_si256(_mm256_cmpeq_epi64(bound, zero), f);

    // OR the eight direction results: combine the two vectors, then fold the
    // four lanes down to one u64.
    let total = _mm256_or_si256(fwd, bwd);
    let lo = _mm256_castsi256_si128(total);
    let hi = _mm256_extracti128_si256(total, 1);
    let or2 = _mm_or_si128(lo, hi);
    let or1 = _mm_or_si128(or2, _mm_unpackhi_epi64(or2, or2));
    _mm_cvtsi128_si64(or1) as u64
}
