//! Portable runtime ray-scan: the same 8-direction SWAR fill as
//! [`super::specialized`], but with the move square a runtime value. No
//! per-square function table, so it carries no indirect call and inlines into
//! its callers.

/// Discs flipped by `player` playing at `mv` (assumed empty).
///
/// A benchmark/selection alternative to the production [`super::flip`]; only
/// reached from the variant benchmark and the correctness battery.
#[allow(dead_code)]
#[inline]
pub(crate) fn flip(mv: u32, player: u64, opponent: u64) -> u64 {
    const MIDDLE_COLUMNS: u64 = 0x7E7E7E7E7E7E7E7E;
    let move_bit = 1u64 << (mv & 63);
    let opp_h = opponent & MIDDLE_COLUMNS;
    let opp_v = opponent;
    let opp_d = opponent & MIDDLE_COLUMNS;
    let mut flipped: u64 = 0;

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

    ray!(opp_h, <<, 1);
    ray!(opp_h, >>, 1);
    ray!(opp_v, <<, 8);
    ray!(opp_v, >>, 8);
    ray!(opp_d, <<, 7);
    ray!(opp_d, >>, 7);
    ray!(opp_d, <<, 9);
    ray!(opp_d, >>, 9);

    flipped
}
