//! The exact endgame search: negamax with PVS, move ordering, a transposition
//! table, and Enhanced Transposition / stability cutoffs. Full-window (PV) and
//! null-window (NWS) variants are kept separate so the compiler can fold `beta`
//! to `alpha + 1` on the NWS path — the bulk of the tree — and drop the PVS
//! re-search machinery there.

use super::stability::{edge_stability_table, get_stability, STABILITY_THRESHOLD};
use super::tt::{TtEntry, ETC_MIN_EMPTIES, NO_MOVE, TT_MIN_EMPTIES, TT_SIZE};
use super::{SCORE_MAX, SCORE_MIN};
use crate::othello::position::Position;

/// Minimum empties at which the deep search orders moves (by opponent mobility).
/// Below this, moves are searched in natural order — near the leaves the ordering
/// work costs more than the pruning it buys. Swept 4..10: lower values cut nodes
/// but run slower (ordering low-empty nodes costs more than it saves), higher
/// values explode nodes. Re-swept after the carry-64 flip (Step 11): the cheaper
/// flip nudged the crossover toward 7, but 6 and 7 are within noise while 6
/// visits fewer nodes, so 6 stays. Re-tune when the ordering cost/benefit shifts.
const SORT_MIN_EMPTIES: u32 = 6;

/// Region id per square: one of the four board quadrants, as a single bit
/// (top-left = 1, top-right = 2, bottom-left = 4, bottom-right = 8). Edax's
/// `QUADRANT_ID`. The board's parity is the XOR of these over its empty squares
/// — bit `q` set iff quadrant `q` holds an odd number of empties.
const QUADRANT_ID: [u8; 64] = {
    let mut q = [0u8; 64];
    let mut sq = 0;
    while sq < 64 {
        let hb = if sq % 8 >= 4 { 1 } else { 0 };
        let vb = if sq / 8 >= 4 { 2 } else { 0 };
        q[sq] = 1u8 << (hb + vb);
        sq += 1;
    }
    q
};

/// Region parity of `pos`: XOR of [`QUADRANT_ID`] over its empty squares. Used to
/// seed [`Search::parity`] at the root; maintained incrementally during search.
pub(super) fn board_parity(pos: &Position) -> u8 {
    let mut p = 0u8;
    let mut e = !pos.occupied();
    while e != 0 {
        p ^= QUADRANT_ID[e.trailing_zeros() as usize];
        e &= e - 1;
    }
    p
}

/// Mutable state for one exact search: nodes visited, the transposition table,
/// a borrowed handle to the shared edge-stability table, and the running region
/// parity (maintained incrementally in the ordered search for move ordering).
pub(super) struct Search {
    pub(super) nodes: u64,
    pub(super) tt: Vec<TtEntry>,
    edge_stability: &'static [u8; 256 * 256],
    /// Region parity of the current node's empties (Edax `parity`). Seeded at the
    /// root via [`board_parity`], then toggled `^= QUADRANT_ID[move]` per ply.
    pub(super) parity: u8,
}

impl Search {
    pub(super) fn new() -> Self {
        Search {
            nodes: 0,
            tt: vec![TtEntry::default(); TT_SIZE],
            edge_stability: edge_stability_table(),
            parity: 0,
        }
    }

    /// Full-window dispatch by `empties`: leaf solver (≤4), unordered search
    /// (below `SORT_MIN_EMPTIES`), else the ordered PVS search.
    #[inline]
    pub(super) fn search_exact(
        &mut self,
        pos: &Position,
        alpha: i32,
        beta: i32,
        empties: u32,
    ) -> i32 {
        if empties <= 4 {
            self.solve_leaf(pos, alpha, beta, empties)
        } else if empties < SORT_MIN_EMPTIES {
            self.alphabeta_nosort(pos, alpha, beta, empties)
        } else {
            self.alphabeta_exact(pos, alpha, beta, empties)
        }
    }

    /// Null-window dispatch: the window is implicitly `[alpha, alpha + 1]`, so
    /// `beta` is not passed. Used for every non-PV node. The leaf solvers are
    /// window-agnostic, so the leaf case reuses them with an explicit `alpha + 1`.
    #[inline]
    fn search_exact_nws(&mut self, pos: &Position, alpha: i32, empties: u32) -> i32 {
        if empties <= 4 {
            self.solve_leaf(pos, alpha, alpha + 1, empties)
        } else if empties < SORT_MIN_EMPTIES {
            self.alphabeta_nosort_nws(pos, alpha, empties)
        } else {
            self.alphabeta_exact_nws(pos, alpha, empties)
        }
    }

    /// Ordered negamax with PVS for `empties >= SORT_MIN_EMPTIES`. Consults the
    /// transposition table (at `empties >= TT_MIN_EMPTIES`) for a cutoff bound,
    /// hash move, and bound write-back.
    fn alphabeta_exact(
        &mut self,
        pos: &Position,
        mut alpha: i32,
        mut beta: i32,
        empties: u32,
    ) -> i32 {
        self.nodes += 1;
        debug_assert_eq!(self.parity, board_parity(pos), "parity desync (exact)");

        let use_tt = empties >= TT_MIN_EMPTIES;
        let mut hash_move = NO_MOVE;
        if use_tt {
            if let Some(e) = self.tt_probe(pos) {
                let (lo, hi) = (e.lower as i32, e.upper as i32);
                if lo >= beta {
                    return lo;
                }
                if hi <= alpha {
                    return hi;
                }
                if lo == hi {
                    return lo;
                }
                if lo > alpha {
                    alpha = lo;
                }
                if hi < beta {
                    beta = hi;
                }
                hash_move = e.best_move;
            }
        }
        // Window actually searched (after TT narrowing); used to classify the
        // result as a bound or exact.
        let search_alpha = alpha;
        let search_beta = beta;

        // Stability cutoff: the opponent's stable discs cap our score at
        // `64 - 2*stable`; if that already fails low the node can't beat alpha.
        // Gated per-empties; returns without a TT store (cheap to recompute).
        if alpha >= STABILITY_THRESHOLD[empties as usize] {
            let stable = get_stability(self.edge_stability, pos.opponent, pos.player) as i32;
            let bound = SCORE_MAX - 2 * stable;
            if bound <= alpha {
                return bound;
            }
        }

        let moves = pos.get_moves();
        if moves == 0 {
            let passed = pos.pass_move();
            if passed.get_moves() == 0 {
                return pos.final_score();
            }
            return -self.alphabeta_exact(&passed, -beta, -alpha, empties);
        }

        let mut move_list: Vec<(u32, u32, Position)> =
            Vec::with_capacity(moves.count_ones() as usize);
        let mut remaining = moves;
        while remaining != 0 {
            let cell = remaining.trailing_zeros();
            remaining &= remaining - 1;
            let child = pos.do_move(cell);
            move_list.push((child.get_moves().count_ones(), cell, child));
        }

        // Enhanced Transposition Cutoff: if a child has a stored upper bound, our
        // value for that move is at least `-upper`; if that meets beta the node
        // fails high. Gated at ETC_MIN_EMPTIES so children are deep enough to be
        // stored and the per-child probe pays for itself.
        if empties >= ETC_MIN_EMPTIES {
            for &(_, cell, ref child) in &move_list {
                if let Some(e) = self.tt_probe(child) {
                    let value = -(e.upper as i32);
                    if value >= beta {
                        self.tt_store(pos, value as i8, SCORE_MAX as i8, cell as u8);
                        return value;
                    }
                }
            }
        }

        // Order odd-parity quadrants first (you tend to get the last move in a
        // region with an odd empty count), then by opponent mobility.
        let parity = self.parity;
        move_list.sort_unstable_by_key(|&(mobility, cell, _)| {
            let even = (parity & QUADRANT_ID[cell as usize] == 0) as u32;
            (mobility, even)
        });

        // A hash best move is the strongest ordering signal: pull it to the front.
        if hash_move != NO_MOVE {
            if let Some(i) = move_list.iter().position(|&(_, c, _)| c as u8 == hash_move) {
                move_list[..=i].rotate_right(1);
            }
        }

        // PVS: full-window search the best-ordered move, then null-window probe
        // each sibling and re-search only on a fail-high.
        let mut best_cell = NO_MOVE;
        let mut first = true;
        for &(_, cell, ref child) in &move_list {
            self.parity ^= QUADRANT_ID[cell as usize];
            let score = if first {
                -self.search_exact(child, -beta, -alpha, empties - 1)
            } else {
                let probe = -self.search_exact_nws(child, -alpha - 1, empties - 1);
                if probe > alpha && probe < beta {
                    -self.search_exact(child, -beta, -alpha, empties - 1)
                } else {
                    probe
                }
            };
            self.parity ^= QUADRANT_ID[cell as usize];
            first = false;
            if score > alpha {
                alpha = score;
                best_cell = cell as u8;
                if alpha >= beta {
                    break;
                }
            }
        }

        if use_tt {
            // Classify against the searched window: at/below `search_alpha` an
            // upper bound; at/above `search_beta` a lower bound; else exact.
            let (lower, upper) = if alpha <= search_alpha {
                (SCORE_MIN as i8, alpha as i8)
            } else if alpha >= search_beta {
                (alpha as i8, SCORE_MAX as i8)
            } else {
                (alpha as i8, alpha as i8)
            };
            self.tt_store(pos, lower, upper, best_cell);
        }

        alpha
    }

    /// Negamax with PVS but no move ordering, for the `5 ..< SORT_MIN_EMPTIES`
    /// range. Moves are tried in natural board order with no move-list allocation.
    fn alphabeta_nosort(&mut self, pos: &Position, mut alpha: i32, beta: i32, empties: u32) -> i32 {
        self.nodes += 1;

        let moves = pos.get_moves();
        if moves == 0 {
            let passed = pos.pass_move();
            if passed.get_moves() == 0 {
                return pos.final_score();
            }
            return -self.alphabeta_nosort(&passed, -beta, -alpha, empties);
        }

        let mut first = true;
        let mut remaining = moves;
        while remaining != 0 {
            let cell = remaining.trailing_zeros();
            remaining &= remaining - 1;
            let child = pos.do_move(cell);
            let score = if first {
                -self.search_exact(&child, -beta, -alpha, empties - 1)
            } else {
                let probe = -self.search_exact_nws(&child, -alpha - 1, empties - 1);
                if probe > alpha && probe < beta {
                    -self.search_exact(&child, -beta, -alpha, empties - 1)
                } else {
                    probe
                }
            };
            first = false;
            if score > alpha {
                alpha = score;
                if alpha >= beta {
                    break;
                }
            }
        }

        alpha
    }

    /// Null-window counterpart of [`Search::alphabeta_exact`] (window
    /// `[alpha, alpha + 1]`). Dropping `beta` lets the compiler fold it: the TT
    /// probe never narrows, ETC's `value >= beta` becomes `value > alpha`, and PVS
    /// collapses to a single null-window probe per child that cuts on the first
    /// fail-high. Node count is identical to `alphabeta_exact` with `beta = alpha + 1`.
    fn alphabeta_exact_nws(&mut self, pos: &Position, alpha: i32, empties: u32) -> i32 {
        self.nodes += 1;
        debug_assert_eq!(self.parity, board_parity(pos), "parity desync (nws)");
        let beta = alpha + 1;

        let use_tt = empties >= TT_MIN_EMPTIES;
        let mut hash_move = NO_MOVE;
        if use_tt {
            if let Some(e) = self.tt_probe(pos) {
                let (lo, hi) = (e.lower as i32, e.upper as i32);
                if lo >= beta {
                    return lo;
                }
                if hi <= alpha {
                    return hi;
                }
                if lo == hi {
                    return lo;
                }
                // A width-1 window that survives the checks above is already as
                // narrow as the stored bounds permit — no narrowing.
                hash_move = e.best_move;
            }
        }

        if alpha >= STABILITY_THRESHOLD[empties as usize] {
            let stable = get_stability(self.edge_stability, pos.opponent, pos.player) as i32;
            let bound = SCORE_MAX - 2 * stable;
            if bound <= alpha {
                return bound;
            }
        }

        let moves = pos.get_moves();
        if moves == 0 {
            let passed = pos.pass_move();
            if passed.get_moves() == 0 {
                return pos.final_score();
            }
            return -self.alphabeta_exact_nws(&passed, -beta, empties);
        }

        let mut move_list: Vec<(u32, u32, Position)> =
            Vec::with_capacity(moves.count_ones() as usize);
        let mut remaining = moves;
        while remaining != 0 {
            let cell = remaining.trailing_zeros();
            remaining &= remaining - 1;
            let child = pos.do_move(cell);
            move_list.push((child.get_moves().count_ones(), cell, child));
        }

        if empties >= ETC_MIN_EMPTIES {
            for &(_, cell, ref child) in &move_list {
                if let Some(e) = self.tt_probe(child) {
                    let value = -(e.upper as i32);
                    if value >= beta {
                        self.tt_store(pos, value as i8, SCORE_MAX as i8, cell as u8);
                        return value;
                    }
                }
            }
        }

        let parity = self.parity;
        move_list.sort_unstable_by_key(|&(mobility, cell, _)| {
            let even = (parity & QUADRANT_ID[cell as usize] == 0) as u32;
            (mobility, even)
        });
        if hash_move != NO_MOVE {
            if let Some(i) = move_list.iter().position(|&(_, c, _)| c as u8 == hash_move) {
                move_list[..=i].rotate_right(1);
            }
        }

        // Every child shares the null window; the first fail-high cuts the node.
        let mut best = alpha;
        let mut best_cell = NO_MOVE;
        for &(_, cell, ref child) in &move_list {
            self.parity ^= QUADRANT_ID[cell as usize];
            let score = -self.search_exact_nws(child, -beta, empties - 1);
            self.parity ^= QUADRANT_ID[cell as usize];
            if score > best {
                best = score;
                best_cell = cell as u8;
                break;
            }
        }

        if use_tt {
            // A null-window result is never exact: a lower bound on a fail-high,
            // else an upper bound.
            let (lower, upper) = if best > alpha {
                (best as i8, SCORE_MAX as i8)
            } else {
                (SCORE_MIN as i8, best as i8)
            };
            self.tt_store(pos, lower, upper, best_cell);
        }

        best
    }

    /// Null-window counterpart of [`Search::alphabeta_nosort`]: natural order, no
    /// re-search, first fail-high cuts.
    fn alphabeta_nosort_nws(&mut self, pos: &Position, alpha: i32, empties: u32) -> i32 {
        self.nodes += 1;
        let beta = alpha + 1;

        let moves = pos.get_moves();
        if moves == 0 {
            let passed = pos.pass_move();
            if passed.get_moves() == 0 {
                return pos.final_score();
            }
            return -self.alphabeta_nosort_nws(&passed, -beta, empties);
        }

        let mut best = alpha;
        let mut remaining = moves;
        while remaining != 0 {
            let cell = remaining.trailing_zeros();
            remaining &= remaining - 1;
            let child = pos.do_move(cell);
            let score = -self.search_exact_nws(&child, -beta, empties - 1);
            if score > best {
                best = score;
                break;
            }
        }

        best
    }
}
