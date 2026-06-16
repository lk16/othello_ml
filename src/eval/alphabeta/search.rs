//! The exact endgame search: negamax with PVS, move ordering, a transposition
//! table, and Enhanced Transposition / stability cutoffs. Full-window (PV) and
//! null-window (NWS) variants are kept separate so the compiler can fold `beta`
//! to `alpha + 1` on the NWS path — the bulk of the tree — and drop the PVS
//! re-search machinery there.

use super::parallel::{CancelNode, ParCtx};
use super::stability::{edge_stability_table, get_stability, STABILITY_THRESHOLD};
use super::tt::{SharedTt, TtBackend, ETC_MIN_EMPTIES, NO_MOVE, TT_MIN_EMPTIES, TT_SIZE};
use super::{SCORE_MAX, SCORE_MIN};
use crate::eval::pattern::FlatEval;
use crate::othello::position::Position;
use std::mem::MaybeUninit;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

/// Minimum empties at which a null-window node may split its younger siblings
/// across worker threads (full recursive YBWC, Step 21). Below this a subtree is
/// too small to amortize the spawn and shared-table lock traffic, so it runs
/// sequentially. Only reached when the search carries a parallel context
/// ([`Search::par`]); the sequential solver never splits. Re-swept after the
/// lock-free TT (Step 29) with larger samples and a thread sweep: at 16 threads 14
/// and 15 are within ~3% (14 wins 20e, 15 wins 22e/24e), but 15 is the robust
/// choice — it beats 14 by 10–18% at 24/32 threads, where 14's more aggressive
/// splitting inflates speculative-node and TT-contention overhead. Best wall-clock
/// is always at 16 threads (= physical cores on the dev box); hyperthreading (32)
/// is ~12–18% slower — this search is ALU/cache-bound, so SMT siblings only
/// contend. Below 14 the subtrees are too small (spawn + speculative nodes
/// dominate); the earlier pre-Step-29 16-thread-only sweep is in git history.
const SPLIT_MIN_EMPTIES: u32 = 15;

/// Top of the shallow-tier band (Edax `DEPTH_TO_SHALLOW_SEARCH`): empties
/// `5 ..= SHALLOW_MAX_EMPTIES` use the cheap [`Search::shallow`] path (parity-only
/// ordering, no move list, no mobility sort, no transposition table) in place of
/// the mobility-sorted ordered search. A/B'd against the prior unordered-5 +
/// ordered-6/7 split (Step 30): +30–40% nodes but neutral-to-faster wall-clock,
/// the win growing with depth and larger in parallel (22e: ~4.5% sequential,
/// ~11% at 16 threads). Enumerates moves with `get_moves` + this parity mask;
/// Edax's empties-list walk (one flip per empty) was also implemented and A/B'd
/// (identical node counts) but ran ~3–4% slower here, so it was reverted — see the
/// dead-ends note in docs/speedup-plan.md.
const SHALLOW_MAX_EMPTIES: u32 = 7;

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

/// For each 4-bit region-parity value, the board mask of squares lying in an
/// odd-parity quadrant (the quadrants whose [`QUADRANT_ID`] bit is set in the
/// parity). The shallow tier ([`Search::shallow`]) ANDs this with the legal moves
/// to visit odd-region moves first — parity ordering with no list, no sort.
const PARITY_MASK: [u64; 16] = {
    let mut table = [0u64; 16];
    let mut p = 0usize;
    while p < 16 {
        let mut m = 0u64;
        let mut sq = 0usize;
        while sq < 64 {
            if (QUADRANT_ID[sq] as usize) & p != 0 {
                m |= 1u64 << sq;
            }
            sq += 1;
        }
        table[p] = m;
        p += 1;
    }
    table
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

/// Static per-square ordering value (Edax `SQUARE_VALUE`, "JCW's score"):
/// corners high, C/X squares low. A minor tie-break in the move-ordering score.
#[rustfmt::skip]
const SQUARE_VALUE: [i32; 64] = [
    18,  4, 16, 12, 12, 16,  4, 18,
     4,  2,  6,  8,  8,  6,  2,  4,
    16,  6, 14, 10, 10, 14,  6, 16,
    12,  8, 10,  0,  0, 10,  8, 12,
    12,  8, 10,  0,  0, 10,  8, 12,
    16,  6, 14, 10, 10, 14,  6, 16,
     4,  2,  6,  8,  8,  6,  2,  4,
    18,  4, 16, 12, 12, 16,  4, 18,
];

// Move-ordering weights (Edax `w_*`), in descending importance. Real (opponent)
// mobility dominates; corner stability is the next signal; square value and
// parity are minor tie-breaks.
const W_MOBILITY: i32 = 1 << 15;
const W_CORNER: i32 = 1 << 11;

/// Weight of the eval-guided ordering term (Step 34), mirroring Edax's static-eval
/// sort bonus `(SCORE_MAX − search_eval_0) * (w_eval >> 2)` (`move.c:442`).
/// Co-dominant with mobility (`w_eval >> 2 = 1 << 13` vs `w_mobility = 1 << 15`, and
/// the eval spans 0..128 vs mobility's 0..36). Only applied when a [`FlatEval`] is
/// attached to the search and only at `empties >= EVAL_ORDER_MIN_EMPTIES`.
const W_EVAL: i32 = 1 << 13;

/// Minimum empties at which the eval-guided ordering term is added. The eval is the
/// expensive ordering signal (a from-scratch `FlatEval::set` per child), so it is
/// confined to the upper plies where ordering quality matters most and such nodes
/// are few — the analogue of Edax's `min_depth_table` gate (`move.c:370`). Tunable.
const EVAL_ORDER_MIN_EMPTIES: u32 = 12;

/// Quick corner-anchored stability count for `p`: held corners plus their edge
/// neighbours anchored by a held corner. Edax `get_corner_stability` — a cheap
/// lower bound used only for move ordering.
#[inline]
fn corner_stability(p: u64) -> u32 {
    let stable = ((0x0100_0000_0000_0001 & p) << 1)
        | ((0x8000_0000_0000_0080 & p) >> 1)
        | ((0x0000_0000_0000_0081 & p) << 8)
        | ((0x8100_0000_0000_0000 & p) >> 8)
        | 0x8100_0000_0000_0081;
    (stable & p).count_ones()
}

/// Move-ordering score for playing `cell` (higher first). `child` is the
/// resulting position (opponent to move); `parity`/`parity_weight` add the
/// odd-parity bonus. Mirrors Edax's `movelist_evaluate_fast`.
#[inline]
fn order_score(child: &Position, cell: u32, parity: u8, parity_weight: i32) -> i32 {
    let opp_mobility = child.get_moves().count_ones() as i32;
    // `child.opponent` is the mover's resulting discs (corners we just secured).
    let mut score = (36 - opp_mobility) * W_MOBILITY;
    score += corner_stability(child.opponent) as i32 * W_CORNER;
    score += SQUARE_VALUE[cell as usize];
    if parity & QUADRANT_ID[cell as usize] != 0 {
        score += parity_weight;
    }
    score
}

/// Capacity of the stack move buffer. An Othello position has at most 33 legal
/// moves; 34 leaves margin. A position with more would panic on `push` (checked
/// index) rather than corrupt memory — but that cannot happen in real play.
const MAX_MOVES: usize = 34;

/// Stack-allocated ordered-move buffer of `(order_score, cell, child)` triples,
/// replacing the per-node heap `Vec` (Step 25). Backed by [`MaybeUninit`] so the
/// unused tail is never written — the model is Edax's uninitialized stack
/// `Move move[34]`, which is what makes the per-node buffer free of both heap
/// traffic and init cost. Invariant: `push` writes indices `0..len` in order and
/// nothing else, and only `[..len]` is ever read, so the prefix is always a fully
/// initialized — hence valid — `&[(i32, u32, Position)]`.
struct MoveBuf {
    moves: [MaybeUninit<(i32, u32, Position)>; MAX_MOVES],
    len: usize,
}

impl MoveBuf {
    #[inline]
    fn new() -> Self {
        // `[MaybeUninit; N]` needs no initialization; the array is left untouched
        // until `push` writes into it.
        MoveBuf {
            moves: [MaybeUninit::uninit(); MAX_MOVES],
            len: 0,
        }
    }

    #[inline]
    fn push(&mut self, entry: (i32, u32, Position)) {
        // `write` initializes slot `len`; the checked index panics rather than
        // overflowing if a position ever exceeds `MAX_MOVES` (it cannot).
        self.moves[self.len].write(entry);
        self.len += 1;
    }

    #[inline]
    fn len(&self) -> usize {
        self.len
    }

    #[inline]
    #[allow(unsafe_code)]
    fn as_slice(&self) -> &[(i32, u32, Position)] {
        let init = &self.moves[..self.len];
        // SAFETY: `push` initialized every index in `0..len`, and `MaybeUninit<T>`
        // has the same layout as `T`, so the prefix is a valid `&[T]`.
        unsafe {
            &*(init as *const [MaybeUninit<(i32, u32, Position)>]
                as *const [(i32, u32, Position)])
        }
    }

    #[inline]
    #[allow(unsafe_code)]
    fn as_mut_slice(&mut self) -> &mut [(i32, u32, Position)] {
        let init = &mut self.moves[..self.len];
        // SAFETY: as in `as_slice` — every index in `0..len` is initialized.
        unsafe {
            &mut *(init as *mut [MaybeUninit<(i32, u32, Position)>]
                as *mut [(i32, u32, Position)])
        }
    }
}

/// Shared state of one null-window split point (full recursive YBWC, Step 21),
/// borrowed by every participating worker. `work` is the cursor over the younger
/// siblings (`move_list`); `best` collects the fail-high `(score, cell)`; `cancel`
/// is tripped on the cutoff; `parent_parity` is the split node's parity (a child's
/// is `parent_parity ^ QUADRANT_ID[cell]`).
struct SplitCtx<'a> {
    move_list: &'a [(i32, u32, Position)],
    work: &'a AtomicUsize,
    best: &'a Mutex<(i32, u8)>,
    cancel: &'a CancelNode,
    parent_parity: u8,
    beta: i32,
    empties: u32,
}

/// Worker loop for a null-window split point. Each participating thread pulls the
/// next younger sibling from the shared cursor and null-window-searches it; the
/// first to fail high records the cut and trips `cancel`, unwinding the rest.
fn worker_loop_nws(w: &mut Search, ctx: &SplitCtx) {
    let alpha = ctx.beta - 1;
    loop {
        if ctx.cancel.cancelled() {
            return;
        }
        let i = ctx.work.fetch_add(1, Ordering::Relaxed);
        if i >= ctx.move_list.len() {
            return;
        }
        let (_, cell, ref child) = ctx.move_list[i];
        w.parity = ctx.parent_parity ^ QUADRANT_ID[cell as usize];
        let score = -w.search_exact_nws(child, -ctx.beta, ctx.empties - 1);
        if ctx.cancel.cancelled() {
            return; // aborted mid-search: discard this (meaningless) result
        }
        if score > alpha {
            // Fail-high: this null-window node cuts. Record and stop the siblings.
            let mut b = ctx.best.lock().unwrap_or_else(|e| e.into_inner());
            if score > b.0 {
                *b = (score, cell as u8);
            }
            drop(b);
            ctx.cancel.cancel();
            return;
        }
    }
}

/// Mutable state for one exact search: nodes visited, the transposition table,
/// a borrowed handle to the shared edge-stability table, and the running region
/// parity (maintained incrementally in the ordered search for move ordering).
pub(super) struct Search {
    pub(super) nodes: u64,
    pub(super) tt: TtBackend,
    edge_stability: &'static [u8; 256 * 256],
    /// Region parity of the current node's empties (Edax `parity`). Seeded at the
    /// root via [`board_parity`], then toggled `^= QUADRANT_ID[move]` per ply.
    pub(super) parity: u8,
    /// Parallel context: the thread budget shared by all workers of one parallel
    /// solve (Step 21). `None` for the single-threaded search, which never splits.
    par: Option<Arc<ParCtx>>,
    /// Cancellation handle for the nearest enclosing split: a beta-cutoff trips it
    /// to abort the sibling subtrees. `None` (or an untripped root) when not under
    /// a split. Checked only on the parallel path.
    cancel: Option<Arc<CancelNode>>,
    /// Optional trained pattern eval for eval-guided move ordering (Step 34). When
    /// `Some`, the ordered search adds the eval term to each move's order score at
    /// `empties >= EVAL_ORDER_MIN_EMPTIES`. `None` = the mobility-only baseline
    /// ordering (Steps 6b/30). Sequential `Solver` only; parallel workers run `None`.
    eval: Option<Arc<FlatEval>>,
}

impl Search {
    /// A search with a private (single-threaded) transposition table; never
    /// splits.
    pub(super) fn new() -> Self {
        Search {
            nodes: 0,
            tt: TtBackend::Owned(vec![Default::default(); TT_SIZE]),
            edge_stability: edge_stability_table(),
            parity: 0,
            par: None,
            cancel: None,
            eval: None,
        }
    }

    /// Attach a trained pattern eval for eval-guided move ordering (Step 34).
    pub(super) fn set_eval(&mut self, eval: Arc<FlatEval>) {
        self.eval = Some(eval);
    }

    /// A worker search for parallel YBWC (Step 21): shares the sharded table and
    /// the thread budget, and carries a cancellation handle. Per-worker state
    /// (nodes, parity) is fresh.
    pub(super) fn worker(tt: Arc<SharedTt>, par: Arc<ParCtx>, cancel: Arc<CancelNode>) -> Self {
        Search {
            nodes: 0,
            tt: TtBackend::Shared(tt),
            edge_stability: edge_stability_table(),
            parity: 0,
            par: Some(par),
            cancel: Some(cancel),
            eval: None,
        }
    }

    /// Whether the nearest enclosing split (or an ancestor) has been cancelled.
    #[inline]
    fn is_cancelled(&self) -> bool {
        self.cancel.as_deref().is_some_and(CancelNode::cancelled)
    }

    /// The shared table handle (only valid on a worker; the single-threaded search
    /// never reaches the split path).
    fn shared_tt(&self) -> Arc<SharedTt> {
        match &self.tt {
            TtBackend::Shared(t) => Arc::clone(t),
            TtBackend::Owned(_) => unreachable!("split requested on a private table"),
        }
    }

    /// Full-window dispatch by `empties`: leaf solver (≤4), shallow tier
    /// (`5 ..= SHALLOW_MAX_EMPTIES`), else the ordered PVS search.
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
        } else if empties <= SHALLOW_MAX_EMPTIES {
            self.shallow(pos, alpha, beta, empties)
        } else {
            self.alphabeta_exact(pos, alpha, beta, empties)
        }
    }

    /// Solve `pos` exactly via repeated null-window searches (MTD / bisection,
    /// Step 31) rather than one full-window search. Each probe tests "score ≥ t"
    /// for an odd threshold `t` (the exact score is always even), bisecting
    /// `[SCORE_MIN, SCORE_MAX]` down to the exact value in ~6 probes. The
    /// transposition table (never cleared) carries bounds across probes, so
    /// successive probes are cheap. Exact: every probe is the exact null-window
    /// search, so each "score ≥ t" answer is exact. The first probe (`t = 1`) is
    /// the most informative split (sign of the score), MTD-f-style from a guess of 0.
    pub(super) fn solve_mtd(&mut self, pos: &Position, empties: u32) -> i32 {
        let mut lo = SCORE_MIN; // proven lower bound: score >= lo
        let mut hi = SCORE_MAX; // proven upper bound: score <= hi
        while lo < hi {
            // `lo`/`hi` stay even, so `mid` is exact; pick the odd threshold
            // strictly inside `(lo, hi)`.
            let mid = (lo + hi) / 2;
            let t = if mid & 1 != 0 { mid } else { mid + 1 };
            if self.search_exact_nws(pos, t - 1, empties) >= t {
                lo = t + 1; // score >= t (odd) => score >= t + 1 (even)
            } else {
                hi = t - 1; // score <  t        => score <= t - 1 (even)
            }
        }
        lo
    }

    /// Null-window dispatch: the window is implicitly `[alpha, alpha + 1]`, so
    /// `beta` is not passed. Used for every non-PV node. The leaf solvers are
    /// window-agnostic, so the leaf case reuses them with an explicit `alpha + 1`.
    #[inline]
    fn search_exact_nws(&mut self, pos: &Position, alpha: i32, empties: u32) -> i32 {
        if empties <= 4 {
            self.solve_leaf(pos, alpha, alpha + 1, empties)
        } else if empties <= SHALLOW_MAX_EMPTIES {
            self.shallow_nws(pos, alpha, empties)
        } else {
            self.alphabeta_exact_nws(pos, alpha, empties)
        }
    }

    /// Build the move list for `pos` (legal `moves`) into a stack buffer, scoring
    /// each with [`order_score`]. Shared by the PV and null-window ordered nodes.
    #[inline]
    fn ordered_moves(&self, pos: &Position, moves: u64, empties: u32) -> MoveBuf {
        let parity = self.parity;
        let parity_weight = if empties < 12 { 1 << 3 } else { 1 << 2 };
        // Eval-guided ordering (Step 34): attach the pattern-eval term only when a
        // `FlatEval` is present and high enough in the tree to amortize its cost.
        let eval = if empties >= EVAL_ORDER_MIN_EMPTIES {
            self.eval.as_deref()
        } else {
            None
        };
        let mut buf = MoveBuf::new();
        let mut remaining = moves;
        while remaining != 0 {
            let cell = remaining.trailing_zeros();
            remaining &= remaining - 1;
            let child = pos.do_move(cell);
            let mut score = order_score(&child, cell, parity, parity_weight);
            if let Some(ev) = eval {
                // `child` is opponent-to-move, so its eval is from the opponent's
                // perspective: a low child score is good for us. Mirrors Edax's
                // `(SCORE_MAX − search_eval_0(child)) * (w_eval >> 2)` (move.c:442).
                let child_eval = ev
                    .eval_position(&child)
                    .round()
                    .clamp((SCORE_MIN + 1) as f32, (SCORE_MAX - 1) as f32)
                    as i32;
                score += (SCORE_MAX - child_eval) * W_EVAL;
            }
            buf.push((score, cell, child));
        }
        buf
    }

    /// Ordered negamax with PVS for `empties > SHALLOW_MAX_EMPTIES`. Consults the
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

        let mut move_list = self.ordered_moves(pos, moves, empties);

        // Enhanced Transposition Cutoff: if a child has a stored upper bound, our
        // value for that move is at least `-upper`; if that meets beta the node
        // fails high. Gated at ETC_MIN_EMPTIES so children are deep enough to be
        // stored and the per-child probe pays for itself.
        if empties >= ETC_MIN_EMPTIES {
            for &(_, cell, ref child) in move_list.as_slice() {
                if let Some(e) = self.tt_probe(child) {
                    let value = -(e.upper as i32);
                    if value >= beta {
                        self.tt_store(pos, value as i8, SCORE_MAX as i8, cell as u8);
                        return value;
                    }
                }
            }
        }

        // Best ordering score first (mobility-dominant; see `order_score`).
        move_list
            .as_mut_slice()
            .sort_unstable_by_key(|&(score, _, _)| core::cmp::Reverse(score));

        // A hash best move is the strongest ordering signal: pull it to the front.
        if hash_move != NO_MOVE {
            if let Some(i) = move_list
                .as_slice()
                .iter()
                .position(|&(_, c, _)| c as u8 == hash_move)
            {
                move_list.as_mut_slice()[..=i].rotate_right(1);
            }
        }

        // PVS: full-window search the best-ordered move, then null-window probe
        // each sibling and re-search only on a fail-high.
        let mut best_cell = NO_MOVE;
        let mut first = true;
        for &(_, cell, ref child) in move_list.as_slice() {
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

    /// Null-window counterpart of [`Search::alphabeta_exact`] (window
    /// `[alpha, alpha + 1]`). Dropping `beta` lets the compiler fold it: the TT
    /// probe never narrows, ETC's `value >= beta` becomes `value > alpha`, and PVS
    /// collapses to a single null-window probe per child that cuts on the first
    /// fail-high. Node count is identical to `alphabeta_exact` with `beta = alpha + 1`.
    fn alphabeta_exact_nws(&mut self, pos: &Position, alpha: i32, empties: u32) -> i32 {
        self.nodes += 1;
        debug_assert_eq!(self.parity, board_parity(pos), "parity desync (nws)");
        // Aborted by a sibling's cutoff at an enclosing split: bail without storing
        // (the returned value is discarded by the split owner). Only the parallel
        // path carries a cancel handle; the sequential search skips this.
        if self.is_cancelled() {
            return alpha;
        }
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

        let mut move_list = self.ordered_moves(pos, moves, empties);

        if empties >= ETC_MIN_EMPTIES {
            for &(_, cell, ref child) in move_list.as_slice() {
                if let Some(e) = self.tt_probe(child) {
                    let value = -(e.upper as i32);
                    if value >= beta {
                        self.tt_store(pos, value as i8, SCORE_MAX as i8, cell as u8);
                        return value;
                    }
                }
            }
        }

        move_list
            .as_mut_slice()
            .sort_unstable_by_key(|&(score, _, _)| core::cmp::Reverse(score));
        if hash_move != NO_MOVE {
            if let Some(i) = move_list
                .as_slice()
                .iter()
                .position(|&(_, c, _)| c as u8 == hash_move)
            {
                move_list.as_mut_slice()[..=i].rotate_right(1);
            }
        }

        // Every child shares the null window; the first fail-high cuts the node.
        // YBW: search the eldest child sequentially (it usually cuts), and only on
        // a fail-low fan the younger siblings across worker threads — when deep
        // enough and a parallel context is present (full recursive YBWC, Step 21).
        let mut best = alpha;
        let mut best_cell = NO_MOVE;

        let (_, c0, ref ch0) = move_list.as_slice()[0];
        self.parity ^= QUADRANT_ID[c0 as usize];
        let s0 = -self.search_exact_nws(ch0, -beta, empties - 1);
        self.parity ^= QUADRANT_ID[c0 as usize];
        if self.is_cancelled() {
            return best; // aborted by an ancestor split: discard, do not store
        }
        if s0 > best {
            best = s0;
            best_cell = c0 as u8;
        }

        if best <= alpha && move_list.len() > 1 {
            if self.par.is_some() && empties >= SPLIT_MIN_EMPTIES {
                let (pb, pc) = self.split_nws(move_list.as_slice(), beta, empties);
                if self.is_cancelled() {
                    return best;
                }
                if pb > best {
                    best = pb;
                    best_cell = pc;
                }
            } else {
                for &(_, cell, ref child) in &move_list.as_slice()[1..] {
                    self.parity ^= QUADRANT_ID[cell as usize];
                    let score = -self.search_exact_nws(child, -beta, empties - 1);
                    self.parity ^= QUADRANT_ID[cell as usize];
                    if self.is_cancelled() {
                        return best;
                    }
                    if score > best {
                        best = score;
                        best_cell = cell as u8;
                        break;
                    }
                }
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

    /// Fan the younger siblings (`move_list[1..]`) of a null-window node across
    /// worker threads (full recursive YBWC, Step 21). Returns `(best, best_cell)`:
    /// the fail-high `(score, cell)` if a sibling cut, else `(alpha, NO_MOVE)`.
    /// Acquires up to the remaining thread budget and participates on this thread
    /// too, so it always finishes the siblings even with no budget. Restores
    /// `self`'s parity and cancel handle before returning.
    fn split_nws(
        &mut self,
        move_list: &[(i32, u32, Position)],
        beta: i32,
        empties: u32,
    ) -> (i32, u8) {
        let alpha = beta - 1;
        let par = match &self.par {
            Some(p) => Arc::clone(p),
            None => return (alpha, NO_MOVE),
        };
        let n = move_list.len();

        let tt = self.shared_tt();
        let child_cancel = CancelNode::child(self.cancel.clone());
        let parent_parity = self.parity;
        let work = AtomicUsize::new(1); // index 0 (the eldest) is already searched
        let best = Mutex::new((alpha, NO_MOVE));
        let helper_nodes = AtomicU64::new(0);
        let ctx = SplitCtx {
            move_list,
            work: &work,
            best: &best,
            cancel: &child_cancel,
            parent_parity,
            beta,
            empties,
        };

        // Acquire helper slots; this thread participates too, so at most one fewer
        // than the number of remaining siblings.
        let mut helpers = 0;
        while helpers + 1 < n && par.try_acquire() {
            helpers += 1;
        }

        std::thread::scope(|scope| {
            for _ in 0..helpers {
                let tt = tt.clone();
                let par = par.clone();
                let cancel = child_cancel.clone();
                let ctx = &ctx;
                let helper_nodes = &helper_nodes;
                scope.spawn(move || {
                    let mut w = Search::worker(tt, par.clone(), cancel);
                    worker_loop_nws(&mut w, ctx);
                    par.release();
                    helper_nodes.fetch_add(w.nodes, Ordering::Relaxed);
                });
            }
            // This thread participates under the same cancel handle.
            let saved = self.cancel.replace(child_cancel.clone());
            worker_loop_nws(self, &ctx);
            self.cancel = saved;
        });

        self.nodes += helper_nodes.load(Ordering::Relaxed);
        self.parity = parent_parity;
        let outcome = *best.lock().unwrap_or_else(|e| e.into_inner());
        outcome
    }

    /// Shallow tier (Step 30) for `5 ..= SHALLOW_MAX_EMPTIES`: PVS with
    /// parity-only move ordering (odd-parity quadrants first) — no move list, no
    /// mobility sort, no transposition table — keeping only the cheap stability
    /// cutoff. Recurses through [`Search::search_exact`], which dispatches the leaf
    /// solvers at `<= 4` empties.
    fn shallow(&mut self, pos: &Position, mut alpha: i32, beta: i32, empties: u32) -> i32 {
        self.nodes += 1;
        debug_assert_eq!(self.parity, board_parity(pos), "parity desync (shallow)");

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
            return -self.shallow(&passed, -beta, -alpha, empties);
        }

        // Parity ordering: odd-region moves first, then the rest. No list, no sort.
        let odd = moves & PARITY_MASK[self.parity as usize];
        let mut first = true;
        for group in [odd, moves & !odd] {
            let mut remaining = group;
            while remaining != 0 {
                let cell = remaining.trailing_zeros();
                remaining &= remaining - 1;
                let child = pos.do_move(cell);
                self.parity ^= QUADRANT_ID[cell as usize];
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
                self.parity ^= QUADRANT_ID[cell as usize];
                first = false;
                if score > alpha {
                    alpha = score;
                    if alpha >= beta {
                        return alpha;
                    }
                }
            }
        }
        alpha
    }

    /// Null-window counterpart of [`Search::shallow`] (window `[alpha, alpha + 1]`):
    /// parity-ordered, no re-search, first fail-high cuts the node.
    fn shallow_nws(&mut self, pos: &Position, alpha: i32, empties: u32) -> i32 {
        self.nodes += 1;
        debug_assert_eq!(
            self.parity,
            board_parity(pos),
            "parity desync (shallow nws)"
        );
        let beta = alpha + 1;

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
            return -self.shallow_nws(&passed, -beta, empties);
        }

        let odd = moves & PARITY_MASK[self.parity as usize];
        let mut best = alpha;
        'search: for group in [odd, moves & !odd] {
            let mut remaining = group;
            while remaining != 0 {
                let cell = remaining.trailing_zeros();
                remaining &= remaining - 1;
                let child = pos.do_move(cell);
                self.parity ^= QUADRANT_ID[cell as usize];
                let score = -self.search_exact_nws(&child, -beta, empties - 1);
                self.parity ^= QUADRANT_ID[cell as usize];
                if score > best {
                    best = score;
                    break 'search;
                }
            }
        }

        best
    }
}
