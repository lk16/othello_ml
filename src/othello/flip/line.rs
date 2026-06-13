//! Shared line-flip tables for the table-driven variants (`carry64`, `bmi2`).
//!
//! A flip is computed one line at a time. Each of the four lines through the
//! move square is gathered into an 8-bit value with the move at bit position
//! `t`; [`inline_flip`] returns the flipped bits on that line, which the caller
//! scatters back to the board. The two small tables factor the in-line flip:
//!   `flipped_bits = FLIPPED[t][OUTFLANK[t][o] & p]`
//! where `OUTFLANK[t][o]` marks the first non-opponent cell beyond the
//! contiguous opponent run on each side of `t`, and `FLIPPED[t][outflank]` is
//! the set of cells strictly between `t` and each bound that is a player disc.
//!
//! Consumers (`carry64`, `bmi2`) are selected by `cfg(target_feature)` or only
//! reached from the variant benchmark, so the items are unused on some builds.
#![allow(dead_code)]

/// `OUTFLANK[t][o]`: for opponent line-pattern `o` and move at `t`, the bit at
/// the first non-`o` cell on each side of `t` (the candidate bounding squares).
const OUTFLANK: [[u8; 256]; 8] = {
    let mut table = [[0u8; 256]; 8];
    let mut t = 0usize;
    while t < 8 {
        let mut o = 0usize;
        while o < 256 {
            let mut of = 0u32;
            // Leftward (lower bits): first clear cell ends the run.
            let mut j = t as i32 - 1;
            while j >= 0 {
                if o & (1 << j) == 0 {
                    of |= 1 << j;
                    break;
                }
                j -= 1;
            }
            // Rightward (higher bits).
            let mut j = t + 1;
            while j < 8 {
                if o & (1 << j) == 0 {
                    of |= 1 << j;
                    break;
                }
                j += 1;
            }
            table[t][o] = of as u8;
            o += 1;
        }
        t += 1;
    }
    table
};

/// `FLIPPED[t][outflank]`: cells strictly between `t` and each bound bit set in
/// `outflank` (a player disc), i.e. the discs flipped on this line.
const FLIPPED: [[u8; 256]; 8] = {
    let mut table = [[0u8; 256]; 8];
    let mut t = 0usize;
    while t < 8 {
        let mut of = 0usize;
        while of < 256 {
            let mut fl = 0u32;
            let mut b = 0usize;
            while b < 8 {
                if of & (1 << b) != 0 {
                    if b < t {
                        let mut k = b + 1;
                        while k < t {
                            fl |= 1 << k;
                            k += 1;
                        }
                    } else if b > t {
                        let mut k = t + 1;
                        while k < b {
                            fl |= 1 << k;
                            k += 1;
                        }
                    }
                }
                b += 1;
            }
            table[t][of] = fl as u8;
            of += 1;
        }
        t += 1;
    }
    table
};

/// Flipped bits on one 8-bit line: move at position `t`, player bits `p`,
/// opponent bits `o` (bit `t` clear in both).
#[inline]
pub(super) fn inline_flip(t: u32, p: u8, o: u8) -> u8 {
    let outflank = OUTFLANK[t as usize][o as usize] & p;
    FLIPPED[t as usize][outflank as usize]
}

/// Per-square full-line masks `[row, column, diagonal-╲, diagonal-╱]`. Each mask
/// includes the square itself. Used by `bmi2` (PEXT/PDEP) and `carry64` (the two
/// diagonal masks; row/column are handled by shifts).
pub(super) const LINE_MASKS: [[u64; 4]; 64] = {
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
