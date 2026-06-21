//! Transposition table for the exact search.
//!
//! Maps a position to a `[lower, upper]` bound on its exact score plus the best
//! move found. An exact endgame score is intrinsic to the position (never
//! path-dependent), so an entry stays valid for the table's lifetime: the table
//! is never cleared, only refined, and a warm table speeds up later positions.
//! A single [`Search`] owns it and is reused across positions.

use super::search::Search;
use crate::othello::position::Position;
use std::sync::atomic::{AtomicU64, Ordering};

/// Table size as a power of two. Swept 17..23: node counts barely move above
/// ~2^19 (collisions already rare at benchmarked depths), so the win is cache
/// locality. 2^19 (a 12 MB table) is the knee — almost all of the locality win,
/// with 2× the headroom of 2^18 for solves deeper than the benchmark.
const TT_BITS: u32 = 19;
pub(super) const TT_SIZE: usize = 1 << TT_BITS;
const TT_MASK: u64 = TT_SIZE as u64 - 1;

/// Minimum empties at which the table is consulted. The TT is only wired into the
/// ordered search, which since the shallow tier (Step 30) runs at `empties >
/// SHALLOW_MAX_EMPTIES` (≥ 8), so the *effective* floor is now 8 — values of this
/// const below 8 no longer change behavior (empties 6–7 are searched without the
/// TT). The historical sweep below predates Step 30, when 6/7 were TT-ordered
/// nodes: swept 6..10 (8/10 clearly lose); 6 then beat 7 (16e 13.7 vs 14.1ms, 18e
/// 76.5 vs 78.6, ~6% fewer nodes by probing/storing the numerous empties-6 nodes),
/// re-swept jointly with `ETC_MIN_EMPTIES` after Step 6b. Kept at 6; re-tune
/// alongside `SHALLOW_MAX_EMPTIES` if the shallow band moves.
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
/// key risks a wrong score), a `[lower, upper]` score bound, the best move for
/// ordering, and the search `depth` that produced the bound (Edax `HashData.depth`).
/// A slot is empty iff `player | opponent == 0`.
///
/// `depth` is the number of plies searched below this node before the bound was
/// established: for the exact search (always to game end) that is the node's
/// `empties`, so an exact entry is stamped `depth == empties`. The
/// iterative-deepening heuristic passes (Step 34) stamp the shallower pass depth.
/// A stored bound is only valid for a *cutoff* when `depth >= empties` (i.e. the
/// subtree was fully resolved — see [`Search::tt_probe`] callers); shallower
/// entries still supply the `best_move` hint, which is always usable for ordering.
/// This is the mechanism that lets heuristic and exact entries share one table
/// without a heuristic bound corrupting an exact cutoff (Edax `search_TC_NWS`).
#[derive(Clone, Copy)]
pub(super) struct TtEntry {
    pub(super) player: u64,
    pub(super) opponent: u64,
    pub(super) lower: i8,
    pub(super) upper: i8,
    pub(super) best_move: u8,
    pub(super) depth: u8,
}

impl Default for TtEntry {
    fn default() -> Self {
        TtEntry {
            player: 0,
            opponent: 0,
            lower: 0,
            upper: 0,
            best_move: NO_MOVE,
            depth: 0,
        }
    }
}

/// Transposition table backing for a [`Search`]: either a private `Vec` (the
/// single-threaded path — no synchronization) or a handle to a lock-free table
/// shared by parallel YBWC workers (Step 21/29). A given `Search` is always one
/// variant, so the dispatch branch is well-predicted.
pub(super) enum TtBackend {
    Owned(Vec<TtEntry>),
    Shared(std::sync::Arc<SharedTt>),
}

/// One lock-free slot, Hyatt XOR-validated ("lockless hashing"). The 128-bit key
/// (`player`, `opponent`) plus the packed payload exceed any single atomic, so
/// instead of a lock the three words are tied together by XOR: `w0 = player ^
/// data`, `w1 = opponent ^ data`, `w2 = data`. A reader recovers `data = w2`,
/// `player = w0 ^ data`, `opponent = w1 ^ data` and accepts the entry only on a
/// full-key match. A torn read — words observed from two different writes — yields
/// at least one mismatched word, so the recovered key fails the compare and the
/// slot reads as a miss (the searcher just recomputes). Correctness is purely
/// value-based, so plain `Relaxed` atomics suffice: no fence, no per-slot lock.
struct Slot {
    w0: AtomicU64,
    w1: AtomicU64,
    w2: AtomicU64,
}

impl Slot {
    fn empty() -> Self {
        Slot {
            w0: AtomicU64::new(0),
            w1: AtomicU64::new(0),
            w2: AtomicU64::new(0),
        }
    }
}

/// Pack the payload (score bounds + best move + depth stamp) into one word.
#[inline]
fn pack_data(lower: i8, upper: i8, best_move: u8, depth: u8) -> u64 {
    (lower as u8 as u64)
        | ((upper as u8 as u64) << 8)
        | ((best_move as u64) << 16)
        | ((depth as u64) << 24)
}

/// Lock-free concurrent transposition table for parallel search: a flat array of
/// XOR-validated [`Slot`]s, no sharding and no mutex. Full-position-keyed like the
/// owned table, so a stored bound is valid for the position's intrinsic score
/// regardless of which worker wrote it.
pub(super) struct SharedTt {
    slots: Box<[Slot]>,
}

impl SharedTt {
    pub(super) fn new() -> Self {
        let slots = (0..TT_SIZE).map(|_| Slot::empty()).collect();
        SharedTt { slots }
    }

    /// Probe `pos`; returns the entry on an exact (full-position) hit. A torn or
    /// foreign slot recovers a non-matching key and returns `None`.
    #[inline]
    fn probe(&self, pos: &Position) -> Option<TtEntry> {
        let slot = &self.slots[tt_index(pos.player, pos.opponent)];
        let data = slot.w2.load(Ordering::Relaxed);
        let player = slot.w0.load(Ordering::Relaxed) ^ data;
        let opponent = slot.w1.load(Ordering::Relaxed) ^ data;
        if player == pos.player && opponent == pos.opponent && (player | opponent) != 0 {
            Some(TtEntry {
                player,
                opponent,
                lower: data as u8 as i8,
                upper: (data >> 8) as u8 as i8,
                best_move: (data >> 16) as u8,
                depth: (data >> 24) as u8,
            })
        } else {
            None
        }
    }

    /// Record a `[lower, upper]` bound (best move + depth) for `pos`. Best-effort
    /// depth-preferred merge (same policy as the owned table, [`merge_payload`]): if
    /// the slot still holds this position, combine by depth, else replace. The
    /// read-modify-write is not atomic, but a lost race only drops a refinement —
    /// every written bound is independently valid — so the table stays exact, the
    /// same trade-off Edax makes for its lockless table.
    #[inline]
    fn store(&self, pos: &Position, lower: i8, upper: i8, best_move: u8, depth: u8) {
        let slot = &self.slots[tt_index(pos.player, pos.opponent)];
        let cur_data = slot.w2.load(Ordering::Relaxed);
        let cur_player = slot.w0.load(Ordering::Relaxed) ^ cur_data;
        let cur_opponent = slot.w1.load(Ordering::Relaxed) ^ cur_data;
        let (lo, hi, bm, d) = if cur_player == pos.player
            && cur_opponent == pos.opponent
            && (cur_player | cur_opponent) != 0
        {
            let cur = (
                cur_data as u8 as i8,
                (cur_data >> 8) as u8 as i8,
                (cur_data >> 16) as u8,
                (cur_data >> 24) as u8,
            );
            merge_payload(cur, (lower, upper, best_move, depth))
        } else {
            (lower, upper, best_move, depth)
        };
        let data = pack_data(lo, hi, bm, d);
        slot.w0.store(pos.player ^ data, Ordering::Relaxed);
        slot.w1.store(pos.opponent ^ data, Ordering::Relaxed);
        slot.w2.store(data, Ordering::Relaxed);
    }
}

/// Combine an incoming bound with whatever the slot currently holds for the **same**
/// position, by depth — Edax's depth-preferred replacement. A deeper result
/// supersedes (its bound targets a different, more-resolved value, so it must not
/// be intersected with a shallower one); equal depth intersects the bounds (both
/// valid for that depth's minimax value) and keeps a real best move; a shallower
/// result is ignored, never degrading a deeper/exact bound. The best move falls
/// back to the existing one only when the incoming entry carries none. Returns the
/// `(lower, upper, best_move, depth)` payload to write.
#[inline]
fn merge_payload(cur: (i8, i8, u8, u8), new: (i8, i8, u8, u8)) -> (i8, i8, u8, u8) {
    let (clo, chi, cbm, cd) = cur;
    let (mut lo, mut hi, mut bm, d) = new;
    if d > cd {
        if bm == NO_MOVE {
            bm = cbm;
        }
        (lo, hi, bm, d)
    } else if d == cd {
        if clo > lo {
            lo = clo;
        }
        if chi < hi {
            hi = chi;
        }
        if bm == NO_MOVE {
            bm = cbm;
        }
        (lo, hi, bm, d)
    } else {
        (clo, chi, cbm, cd)
    }
}

/// Update slot `e` with a bound for `pos`: depth-preferred merge ([`merge_payload`])
/// when the slot already holds `pos`, else always-replace.
#[inline]
fn merge_entry(e: &mut TtEntry, pos: &Position, lower: i8, upper: i8, best_move: u8, depth: u8) {
    if e.player == pos.player && e.opponent == pos.opponent && (e.player | e.opponent) != 0 {
        let (lo, hi, bm, d) = merge_payload(
            (e.lower, e.upper, e.best_move, e.depth),
            (lower, upper, best_move, depth),
        );
        e.lower = lo;
        e.upper = hi;
        e.best_move = bm;
        e.depth = d;
    } else {
        *e = TtEntry {
            player: pos.player,
            opponent: pos.opponent,
            lower,
            upper,
            best_move,
            depth,
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

    /// Record a `[lower, upper]` bound (best move + `depth` stamp) for `pos`. On a
    /// slot that already holds this position the payload is merged by depth
    /// ([`merge_payload`]); otherwise always-replace. `depth` is the plies searched
    /// below `pos` — `empties` for an exact (to-game-end) store, the pass depth for
    /// an iterative-deepening heuristic store.
    #[inline]
    pub(super) fn tt_store(
        &mut self,
        pos: &Position,
        lower: i8,
        upper: i8,
        best_move: u8,
        depth: u8,
    ) {
        match &mut self.tt {
            TtBackend::Owned(v) => {
                let e = &mut v[tt_index(pos.player, pos.opponent)];
                merge_entry(e, pos, lower, upper, best_move, depth);
            }
            TtBackend::Shared(s) => s.store(pos, lower, upper, best_move, depth),
        }
    }
}
