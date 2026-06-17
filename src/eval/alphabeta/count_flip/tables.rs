//! Shared `COUNT_FLIP` table for the count-last-flip variants (`table`, `bmi2`).
//!
//! `COUNT_FLIP[i][pattern]` = 2× discs flipped by playing at line-position `i`,
//! where `pattern` bit `j` is set iff the player holds line-cell `j` (every
//! other cell being an opponent disc — the full-line invariant of the 1-empty
//! leaf). Doubled to ease disc-difference arithmetic, matching Edax.
//!
//! Both consumers are selected by `cfg(target_feature)` or only reached from the
//! variant benchmark, so the table is unused on some builds.
#![allow(dead_code)]

pub(super) const COUNT_FLIP: [[u8; 256]; 8] = {
    let mut table = [[0u8; 256]; 8];
    let mut i = 0;
    while i < 8 {
        let mut p = 0usize;
        while p < 256 {
            let mut flips = 0u32;
            let mut run = 0u32;
            let mut j = i + 1;
            while j < 8 {
                if p & (1 << j) != 0 {
                    flips += run;
                    break;
                }
                run += 1;
                j += 1;
            }
            run = 0;
            let mut j = i;
            while j > 0 {
                j -= 1;
                if p & (1 << j) != 0 {
                    flips += run;
                    break;
                }
                run += 1;
            }
            table[i][p] = (2 * flips) as u8;
            p += 1;
        }
        i += 1;
    }
    table
};
