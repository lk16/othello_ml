//! Disc stability estimate (ported from Edax `board.c`).
//!
//! A *stable* disc can never be flipped for the rest of the game. The opponent's
//! stable discs bound our best result: if the opponent is guaranteed `s` discs we
//! can finish with at most `64 - 2*s`. [`get_stability`] returns a lower estimate
//! of the count (it may undercount, never overcount), so that bound is always
//! valid — undercounting only makes the stability cutoff fire less often.

use std::sync::OnceLock;

/// Per-edge stable-square table: `EDGE_STABILITY[p << 8 | o]` masks the squares
/// stable for `p` on an 8-cell edge. Built once at runtime (the recursive fill is
/// impractical to const-evaluate over all 65536 entries; Edax also builds it at
/// startup).
static EDGE_STABILITY: OnceLock<[u8; 256 * 256]> = OnceLock::new();

pub(super) fn edge_stability_table() -> &'static [u8; 256 * 256] {
    EDGE_STABILITY.get_or_init(build_edge_stability)
}

/// `other`-discs flipped by a disc placed at line position `i` on an 8-cell line:
/// a run of `other` discs flips only when closed by a `mover` disc.
fn edge_flips(mover: u32, other: u32, i: i32) -> u32 {
    let mut flip = 0;
    let mut run = 0;
    let mut j = i - 1;
    while j >= 0 {
        let b = 1u32 << j;
        if other & b != 0 {
            run |= b;
        } else {
            if mover & b != 0 {
                flip |= run;
            }
            break;
        }
        j -= 1;
    }
    run = 0;
    let mut j = i + 1;
    while j < 8 {
        let b = 1u32 << j;
        if other & b != 0 {
            run |= b;
        } else {
            if mover & b != 0 {
                flip |= run;
            }
            break;
        }
        j += 1;
    }
    flip
}

/// Squares that stay `old_p`'s in every continuation of edge play. `stable` is
/// the candidate set carried down the recursion (initially `old_p`); a square
/// drops out the moment some line of play flips it. Like Edax, an empty edge
/// square may be played by either side and even with no flip — a superset of
/// legal play that keeps the estimate conservative.
fn find_edge_stable(old_p: u32, old_o: u32, mut stable: u32) -> u32 {
    let empties = !(old_p | old_o) & 0xff;
    stable &= old_p;
    if stable == 0 || empties == 0 {
        return stable;
    }
    let mut x = 0;
    while x < 8 {
        let bit = 1u32 << x;
        if empties & bit != 0 {
            let f = edge_flips(old_p, old_o, x); // old_p plays at x
            stable = find_edge_stable(old_p | bit | f, old_o & !f, stable);
            if stable == 0 {
                return 0;
            }
            let f = edge_flips(old_o, old_p, x); // old_o plays at x
            stable = find_edge_stable(old_p & !f, old_o | bit | f, stable);
            if stable == 0 {
                return 0;
            }
        }
        x += 1;
    }
    stable
}

fn build_edge_stability() -> [u8; 256 * 256] {
    let mut table = [0u8; 256 * 256];
    let mut po = 0;
    while po < 256 * 256 {
        let p = (po >> 8) as u32;
        let o = (po & 0xff) as u32;
        if p & o == 0 {
            table[po] = find_edge_stable(p, o, p) as u8;
        }
        po += 1;
    }
    table
}

#[inline]
fn pack_a1a8(x: u64) -> usize {
    ((x & 0x0101_0101_0101_0101).wrapping_mul(0x0102_0408_1020_4080) >> 56) as usize
}
#[inline]
fn pack_h1h8(x: u64) -> usize {
    ((x & 0x8080_8080_8080_8080).wrapping_mul(0x0002_0408_1020_4081) >> 56) as usize
}
#[inline]
fn unpack_a2a7(x: u64) -> u64 {
    (x & 0x7e).wrapping_mul(0x0000_0408_1020_4080) & 0x0001_0101_0101_0100
}
#[inline]
fn unpack_h2h7(x: u64) -> u64 {
    (x & 0x7e).wrapping_mul(0x0002_0408_1020_4000) & 0x0080_8080_8080_8000
}

/// Exact stable-edge mask for `p`: the four edges looked up in `EDGE_STABILITY`
/// and reassembled into board coordinates.
#[inline]
fn get_stable_edge(table: &[u8; 256 * 256], p: u64, o: u64) -> u64 {
    let top = table[((p & 0xff) as usize) << 8 | (o & 0xff) as usize] as u64;
    let bottom = (table[((p >> 56) as usize) << 8 | (o >> 56) as usize] as u64) << 56;
    let left = unpack_a2a7(table[pack_a1a8(p) << 8 | pack_a1a8(o)] as u64);
    let right = unpack_h2h7(table[pack_h1h8(p) << 8 | pack_h1h8(o)] as u64);
    top | bottom | left | right
}

/// For each line direction, a mask of squares whose entire line is full:
/// `[horizontal, vertical, diagonal-↗, diagonal-↘]`. A disc full in all four
/// directions cannot be flipped, hence stable.
#[inline]
fn get_full_lines(disc: u64) -> [u64; 4] {
    let mut h = disc;
    h &= h >> 1;
    h &= h >> 2;
    h &= h >> 4;
    let full_h = (h & 0x0101_0101_0101_0101).wrapping_mul(0xff);

    let mut v = disc;
    v &= v.rotate_right(8);
    v &= v.rotate_right(16);
    v &= v.rotate_right(32);
    let full_v = v;

    let (mut l7, mut r7) = (disc, disc);
    l7 &= 0xff01_0101_0101_0101 | (l7 >> 7);
    r7 &= 0x8080_8080_8080_80ff | (r7 << 7);
    l7 &= 0xffff_0303_0303_0303 | (l7 >> 14);
    r7 &= 0xc0c0_c0c0_c0c0_ffff | (r7 << 14);
    l7 &= 0xffff_ffff_0f0f_0f0f | (l7 >> 28);
    r7 &= 0xf0f0_f0f0_ffff_ffff | (r7 << 28);
    let full_d7 = l7 & r7;

    let (mut l9, mut r9) = (disc, disc);
    l9 &= 0xff80_8080_8080_8080 | (l9 >> 9);
    r9 &= 0x0101_0101_0101_01ff | (r9 << 9);
    l9 &= 0xffff_c0c0_c0c0_c0c0 | (l9 >> 18);
    r9 &= 0x0303_0303_0303_ffff | (r9 << 18);
    let full_d9 = l9 & r9 & (0x0f0f_0f0f_f0f0_f0f0 | (l9 >> 36) | (r9 << 36));

    [full_h, full_v, full_d9, full_d7]
}

/// Lower estimate of the number of `p` discs that can never be flipped. Stable
/// edges and full-line-bound central discs seed the set; it then spreads to any
/// central `p` disc that is, in every direction, full or adjacent to a stable
/// disc — iterated to a fixpoint.
pub(super) fn get_stability(table: &[u8; 256 * 256], p: u64, o: u64) -> u32 {
    const CENTRAL: u64 = 0x007e_7e7e_7e7e_7e00; // squares off all four edges
    let p_central = p & CENTRAL;
    let full = get_full_lines(p | o);
    let mut stable =
        get_stable_edge(table, p, o) | (p_central & full[0] & full[1] & full[2] & full[3]);
    if stable == 0 {
        return 0;
    }
    loop {
        let old = stable;
        let h = (stable >> 1) | (stable << 1) | full[0];
        let v = (stable >> 8) | (stable << 8) | full[1];
        let d9 = (stable >> 9) | (stable << 9) | full[2];
        let d7 = (stable >> 7) | (stable << 7) | full[3];
        stable |= h & v & d9 & d7 & p_central;
        if stable == old {
            break;
        }
    }
    stable.count_ones()
}

/// Per-empties alpha gate for the stability cutoff. The cutoff can only fire when
/// `alpha` is high enough that `64 - 2*stable` may fall at or below it, and
/// stable discs are scarce early, so below the threshold the stability
/// computation is pure overhead. `99` means "never try". Ported verbatim from
/// Edax's `NWS_STABILITY_THRESHOLD` (same disc-difference units; a swept ±offset
/// moved node counts <0.1%). Indexed 0..=64 empties; trailing entries are `99`.
#[rustfmt::skip]
pub(super) const STABILITY_THRESHOLD: [i32; 65] = [
    99, 99, 99, 99,  6,  8, 10, 12,
     8, 10, 20, 22, 24, 26, 28, 30,
    32, 34, 36, 38, 40, 42, 44, 46,
    48, 48, 50, 50, 52, 52, 54, 54,
    56, 56, 58, 58, 60, 60, 62, 62,
    64, 64, 64, 64, 64, 64, 64, 64,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
    99,
];

#[cfg(test)]
mod tests {
    use super::super::testutil::two_empty_layouts;
    use super::*;

    #[test]
    fn stability_empty_board_is_zero() {
        let t = edge_stability_table();
        assert_eq!(get_stability(t, 0, 0), 0);
        // No discs for `p` → nothing stable, regardless of opponent fill.
        assert_eq!(get_stability(t, 0, 0xFFFF_FFFF_FFFF_FFFF), 0);
    }

    #[test]
    fn stability_full_board_all_p() {
        // Whole board `p`: every disc is stable (no empties, all lines full).
        let t = edge_stability_table();
        assert_eq!(get_stability(t, 0xFFFF_FFFF_FFFF_FFFF, 0), 64);
    }

    #[test]
    fn stability_lone_corners_are_stable() {
        let t = edge_stability_table();
        for &c in &[0u32, 7, 56, 63] {
            let p = 1u64 << c;
            assert!(get_stability(t, p, 0) >= 1, "corner {c} should be stable");
        }
        // A lone centre disc (d4) is not stable.
        assert_eq!(get_stability(t, 1u64 << 27, 0), 0);
    }

    #[test]
    fn stability_is_a_lower_bound_on_true_stability() {
        // On a full board every `p` disc is trivially stable; on a non-full board
        // the estimate must never exceed the disc count.
        let t = edge_stability_table();
        for (player, opponent) in two_empty_layouts() {
            let full_player = player | !(player | opponent);
            assert_eq!(
                get_stability(t, full_player, opponent),
                full_player.count_ones(),
                "full board: every disc stable; player={full_player:#x}"
            );
            assert!(get_stability(t, player, opponent) <= player.count_ones());
        }
    }
}
