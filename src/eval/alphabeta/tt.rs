//! Transposition table for the exact search.
//!
//! Maps a position to a `[lower, upper]` bound on its exact score plus the best
//! move found. An exact endgame score is intrinsic to the position (never
//! path-dependent), so an entry stays valid for the table's lifetime: the table
//! is never cleared, only refined, and a warm table speeds up later positions.
//! A single [`Search`] owns it and is reused across positions.

use super::search::Search;
use crate::othello::position::Position;
use std::sync::Mutex;

/// Table size as a power of two. Swept 17..23: node counts barely move above
/// ~2^19 (collisions already rare at benchmarked depths), so the win is cache
/// locality. 2^19 (a 12 MB table) is the knee — almost all of the locality win,
/// with 2× the headroom of 2^18 for solves deeper than the benchmark.
const TT_BITS: u32 = 19;
pub(super) const TT_SIZE: usize = 1 << TT_BITS;
const TT_MASK: u64 = TT_SIZE as u64 - 1;

/// Minimum empties at which the table is consulted. The TT is only wired into
/// the ordered search, so the effective floor is `max(SORT_MIN_EMPTIES, this)`.
/// Swept 6..10 (8/10 clearly lose). Originally 6 and 7 tied so 7 was kept;
/// re-swept after the carry-64 flip (Step 11) made nodes cheaper, 6 now wins
/// reproducibly (16e 13.7 vs 14.1ms, 18e 76.5 vs 78.6) and visits ~6% fewer
/// nodes (it probes/stores the numerous empties-6 nodes, enabling more cuts).
/// Re-swept jointly with `ETC_MIN_EMPTIES` after the Step 6b ordering change:
/// (6,7) keeps the fewest nodes at neutral wall-clock (raising either floor cuts
/// no time but adds nodes), so unchanged.
pub(super) const TT_MIN_EMPTIES: u32 = 6;

/// Minimum empties at which the ordered search runs Enhanced Transposition
/// Cutoff. ETC reads *children's* entries, so the structural floor is
/// `TT_MIN_EMPTIES + 1` (= 7): below it the probes hit unstored children and
/// never cut. Swept 7..=9 at `TT_MIN_EMPTIES = 6`: 7 ties 8 on wall-clock and
/// prunes strictly more nodes (its empties-7 probes now cut, since children at
/// empties 6 are stored), so it is preferred. 9 is slightly slower.
pub(super) const ETC_MIN_EMPTIES: u32 = 7;

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

/// Number of mutex shards for the shared (parallel) table, as a power of two.
/// Probes scatter across slots, so with a handful of worker threads contention
/// on any one shard is rare; 1024 keeps lock traffic spread while the per-shard
/// allocation overhead stays trivial.
const TT_SHARD_BITS: u32 = 10;
const TT_SHARDS: usize = 1 << TT_SHARD_BITS;
/// Slots per shard. The high `TT_SHARD_BITS` of a slot index pick the shard.
const TT_SHARD_SLOTS: usize = TT_SIZE >> TT_SHARD_BITS;

/// Transposition table backing for a [`Search`]: either a private `Vec` (the
/// single-threaded path — no synchronization) or a handle to a sharded,
/// mutex-guarded table shared by parallel YBWC workers (Step 21). A given
/// `Search` is always one variant, so the dispatch branch is well-predicted.
pub(super) enum TtBackend {
    Owned(Vec<TtEntry>),
    Shared(std::sync::Arc<SharedTt>),
}

/// Sharded concurrent transposition table for parallel search. The slot array is
/// split into [`TT_SHARDS`] contiguous mutex-guarded ranges; an entry access
/// locks only its shard. Full-position-keyed like the owned table, so it stays
/// exactly correct (no torn reads, no partial-key collisions); a stored bound is
/// valid for the position's intrinsic score regardless of which worker wrote it.
pub(super) struct SharedTt {
    shards: Vec<Mutex<Box<[TtEntry]>>>,
}

impl SharedTt {
    pub(super) fn new() -> Self {
        let shards = (0..TT_SHARDS)
            .map(|_| Mutex::new(vec![TtEntry::default(); TT_SHARD_SLOTS].into_boxed_slice()))
            .collect();
        SharedTt { shards }
    }

    /// Probe `pos`; returns the entry on an exact (full-position) hit.
    #[inline]
    fn probe(&self, pos: &Position) -> Option<TtEntry> {
        let idx = tt_index(pos.player, pos.opponent);
        let shard = self.shards[idx >> (TT_BITS - TT_SHARD_BITS)]
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let e = shard[idx & (TT_SHARD_SLOTS - 1)];
        if e.player == pos.player && e.opponent == pos.opponent && (e.player | e.opponent) != 0 {
            Some(e)
        } else {
            None
        }
    }

    /// Record a `[lower, upper]` bound (and best move) for `pos`, with the same
    /// intersect-or-replace policy as the owned table.
    #[inline]
    fn store(&self, pos: &Position, lower: i8, upper: i8, best_move: u8) {
        let idx = tt_index(pos.player, pos.opponent);
        let mut shard = self.shards[idx >> (TT_BITS - TT_SHARD_BITS)]
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        merge_entry(
            &mut shard[idx & (TT_SHARD_SLOTS - 1)],
            pos,
            lower,
            upper,
            best_move,
        );
    }
}

/// Update slot `e` with a bound for `pos`: intersect bounds and keep a real best
/// move when the slot already holds `pos`, else always-replace. Shared by both
/// table backings.
#[inline]
fn merge_entry(e: &mut TtEntry, pos: &Position, lower: i8, upper: i8, best_move: u8) {
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
        match &self.tt {
            TtBackend::Owned(v) => {
                let e = v[tt_index(pos.player, pos.opponent)];
                if e.player == pos.player
                    && e.opponent == pos.opponent
                    && (e.player | e.opponent) != 0
                {
                    Some(e)
                } else {
                    None
                }
            }
            TtBackend::Shared(s) => s.probe(pos),
        }
    }

    /// Record a `[lower, upper]` bound (and best move) for `pos`. On a slot that
    /// already holds this position the bounds are intersected (both are valid for
    /// the intrinsic score) and a real best move kept; otherwise always-replace.
    #[inline]
    pub(super) fn tt_store(&mut self, pos: &Position, lower: i8, upper: i8, best_move: u8) {
        match &mut self.tt {
            TtBackend::Owned(v) => {
                let e = &mut v[tt_index(pos.player, pos.opponent)];
                merge_entry(e, pos, lower, upper, best_move);
            }
            TtBackend::Shared(s) => s.store(pos, lower, upper, best_move),
        }
    }
}
