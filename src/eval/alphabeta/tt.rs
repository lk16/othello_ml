//! Transposition table for the exact search.
//!
//! Maps a position to a `[lower, upper]` bound on its exact score plus the best
//! move found. An exact endgame score is intrinsic to the position (never
//! path-dependent), so an entry stays valid for the table's lifetime: the table
//! is never cleared, only refined, and a warm table speeds up later positions.
//! A single [`Search`] owns it and is reused across positions.

use super::search::Search;
use crate::othello::position::Position;

/// Table size as a power of two. Swept 17..23: node counts barely move above
/// ~2^19 (collisions already rare at benchmarked depths), so the win is cache
/// locality. 2^19 (a 12 MB table) is the knee — almost all of the locality win,
/// with 2× the headroom of 2^18 for solves deeper than the benchmark.
const TT_BITS: u32 = 19;
pub(super) const TT_SIZE: usize = 1 << TT_BITS;
const TT_MASK: u64 = TT_SIZE as u64 - 1;

/// Minimum empties at which the table is consulted. The TT is only wired into
/// the ordered search, so the effective floor is `max(SORT_MIN_EMPTIES, this)`.
/// Swept 6..10: 6 and 7 are within noise of each other and both beat 8/10; 7
/// edges ahead at the less-noisy 16/18-empty levels (skipping the very numerous
/// empties-6 probes, where overhead roughly cancels node savings). Re-swept
/// after the stability + null-window steps: unchanged.
pub(super) const TT_MIN_EMPTIES: u32 = 7;

/// Minimum empties at which the ordered search runs Enhanced Transposition
/// Cutoff. ETC reads *children's* entries, so the structural floor is
/// `TT_MIN_EMPTIES + 1` (= 8): below it the probes hit unstored children and
/// never cut. Swept 7..=12 — 8 ties 9 on wall-clock and prunes strictly more
/// nodes, so it is preferred for robustness on deeper solves.
pub(super) const ETC_MIN_EMPTIES: u32 = 8;

/// Sentinel "no move" square (a real square is 0..64).
pub(super) const NO_MOVE: u8 = 64;

/// One table slot: the full position (for exact collision detection — a partial
/// key risks a wrong score), a `[lower, upper]` score bound, and the best move
/// for ordering. A slot is empty iff `player | opponent == 0`.
#[derive(Clone, Copy)]
pub(super) struct TtEntry {
    pub(super) player: u64,
    pub(super) opponent: u64,
    pub(super) lower: i8,
    pub(super) upper: i8,
    pub(super) best_move: u8,
}

impl Default for TtEntry {
    fn default() -> Self {
        TtEntry {
            player: 0,
            opponent: 0,
            lower: 0,
            upper: 0,
            best_move: NO_MOVE,
        }
    }
}

/// Hash both bitboards to a table slot.
#[inline]
fn tt_index(player: u64, opponent: u64) -> usize {
    let mut h = player.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    h ^= opponent.wrapping_mul(0xC2B2_AE3D_27D4_EB4F).rotate_left(32);
    h ^= h >> 29;
    h = h.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    h ^= h >> 32;
    (h & TT_MASK) as usize
}

impl Search {
    /// Probe `pos`; returns a copy of the entry on an exact (full-position) hit.
    #[inline]
    pub(super) fn tt_probe(&self, pos: &Position) -> Option<TtEntry> {
        let e = self.tt[tt_index(pos.player, pos.opponent)];
        if e.player == pos.player && e.opponent == pos.opponent && (e.player | e.opponent) != 0 {
            Some(e)
        } else {
            None
        }
    }

    /// Record a `[lower, upper]` bound (and best move) for `pos`. On a slot that
    /// already holds this position the bounds are intersected (both are valid for
    /// the intrinsic score) and a real best move kept; otherwise always-replace.
    #[inline]
    pub(super) fn tt_store(&mut self, pos: &Position, lower: i8, upper: i8, best_move: u8) {
        let e = &mut self.tt[tt_index(pos.player, pos.opponent)];
        if e.player == pos.player && e.opponent == pos.opponent && (e.player | e.opponent) != 0 {
            if lower > e.lower {
                e.lower = lower;
            }
            if upper < e.upper {
                e.upper = upper;
            }
            if best_move != NO_MOVE {
                e.best_move = best_move;
            }
        } else {
            *e = TtEntry {
                player: pos.player,
                opponent: pos.opponent,
                lower,
                upper,
                best_move,
            };
        }
    }
}
