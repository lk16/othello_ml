//! Per-square specialization (Edax's `flip[]`): one `flip_at::<SQ>` per square,
//! dispatched through a function-pointer table. With `SQ` constant the compiler
//! folds the move bit and prunes the rays that run off the board for that square.

/// Discs flipped by `player` playing at the compile-time square `SQ`. The same
/// 8-direction scan as [`super::generic::flip`], specialized per square.
fn flip_at<const SQ: usize>(player: u64, opponent: u64) -> u64 {
    // Horizontal and diagonal shifts need the col B-G mask to prevent wrap.
    const MIDDLE_COLUMNS: u64 = 0x7E7E7E7E7E7E7E7E;
    let move_bit = 1u64 << SQ;
    let opp_h = opponent & MIDDLE_COLUMNS;
    let opp_v = opponent;
    let opp_d = opponent & MIDDLE_COLUMNS;
    let mut flipped: u64 = 0;

    // Each block walks one direction: gather the run of opponent discs, then
    // keep it only if a player disc closes the run.
    macro_rules! ray {
        ($opp:expr, $shift:tt, $n:literal) => {{
            let mut f = $opp & (move_bit $shift $n);
            f |= $opp & (f $shift $n);
            f |= $opp & (f $shift $n);
            f |= $opp & (f $shift $n);
            f |= $opp & (f $shift $n);
            f |= $opp & (f $shift $n);
            if player & (f $shift $n) != 0 {
                flipped |= f;
            }
        }};
    }

    ray!(opp_h, <<, 1); // horizontal
    ray!(opp_h, >>, 1);
    ray!(opp_v, <<, 8); // vertical
    ray!(opp_v, >>, 8);
    ray!(opp_d, <<, 7); // diagonal /
    ray!(opp_d, >>, 7);
    ray!(opp_d, <<, 9); // diagonal \
    ray!(opp_d, >>, 9);

    flipped
}

/// Build a 64-entry array of `flip_at::<SQ>` function pointers.
macro_rules! flip_table {
    ($($sq:literal),+ $(,)?) => {
        [$(flip_at::<$sq> as fn(u64, u64) -> u64),+]
    };
}

/// One specialized flip function per square. Indexed by the move square.
#[rustfmt::skip]
const FLIP: [fn(u64, u64) -> u64; 64] = flip_table!(
     0,  1,  2,  3,  4,  5,  6,  7,
     8,  9, 10, 11, 12, 13, 14, 15,
    16, 17, 18, 19, 20, 21, 22, 23,
    24, 25, 26, 27, 28, 29, 30, 31,
    32, 33, 34, 35, 36, 37, 38, 39,
    40, 41, 42, 43, 44, 45, 46, 47,
    48, 49, 50, 51, 52, 53, 54, 55,
    56, 57, 58, 59, 60, 61, 62, 63,
);

/// Discs flipped by `player` playing at `mv` (assumed empty). The `& 63` proves
/// the index is in range so no bounds check is emitted.
#[inline]
pub(crate) fn flip(mv: u32, player: u64, opponent: u64) -> u64 {
    FLIP[(mv & 63) as usize](player, opponent)
}
