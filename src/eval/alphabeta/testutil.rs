//! Shared fixtures for the exact-search unit tests.

use super::SCORE_MIN;
use crate::othello::position::Position;

/// Independent exact negamax that never uses the `solve_1`..`solve_4` fast
/// paths, so it serves as ground truth for them.
pub(crate) fn naive_exact(pos: &Position) -> i32 {
    if pos.empties() == 0 {
        return pos.final_score();
    }
    let moves = pos.get_moves();
    if moves == 0 {
        let passed = pos.pass_move();
        if passed.get_moves() == 0 {
            return pos.final_score();
        }
        return -naive_exact(&passed);
    }
    let mut best = SCORE_MIN - 1;
    let mut remaining = moves;
    while remaining != 0 {
        let cell = remaining.trailing_zeros();
        remaining &= remaining - 1;
        let s = -naive_exact(&pos.do_move(cell));
        if s > best {
            best = s;
        }
    }
    best
}

/// Empty-square indices exercising corners, edges, centre and both board halves.
pub(crate) const SQUARES: &[u32] = &[0, 1, 7, 8, 9, 14, 27, 28, 35, 36, 49, 55, 56, 62, 63];

/// Smaller square set (includes all four corners) bounding the 4-empty count.
pub(crate) const SQUARES4: &[u32] = &[0, 7, 9, 28, 35, 49, 56, 63];

/// Disc-layout patterns assigned to the player (the rest of the board becomes
/// the opponent): empty, full, alternating, row/column/diagonal stripes.
pub(crate) const PATTERNS: &[u64] = &[
    0x0000_0000_0000_0000,
    0xFFFF_FFFF_FFFF_FFFF,
    0xAAAA_AAAA_AAAA_AAAA,
    0x5555_5555_5555_5555,
    0xFF00_FF00_FF00_FF00,
    0x00FF_00FF_00FF_00FF,
    0xF0F0_F0F0_F0F0_F0F0,
    0x0F0F_0F0F_0F0F_0F0F,
    0x8040_2010_0804_0201,
    0x0102_0408_1020_4080,
    0xC3C3_C3C3_C3C3_C3C3,
    0x1234_5678_9ABC_DEF0,
];

/// Every ordered pair of distinct empty squares from [`SQUARES`], crossed with
/// every pattern, as (player, opponent) layouts.
pub(crate) fn two_empty_layouts() -> impl Iterator<Item = (u64, u64)> {
    SQUARES.iter().enumerate().flat_map(|(i, &s1)| {
        SQUARES[i + 1..].iter().flat_map(move |&s2| {
            let empty = (1u64 << s1) | (1u64 << s2);
            PATTERNS.iter().map(move |&pat| {
                let player = pat & !empty;
                (player, !player & !empty)
            })
        })
    })
}

/// (player, opponent) layouts for a fixed empty set, one per pattern. The
/// empties become the board's only empty cells.
pub(crate) fn layouts_for(empties: &[u32]) -> impl Iterator<Item = (u64, u64)> + '_ {
    let mask = empties.iter().fold(0u64, |m, &s| m | (1u64 << s));
    PATTERNS.iter().map(move |&pat| {
        let player = pat & !mask;
        (player, !player & !mask)
    })
}
