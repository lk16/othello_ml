//! Exact Othello position evaluation via alpha-beta search to game end.

use crate::othello::position::Position;
use crate::training::features::Features;
use crate::training::weights::Weights;

/// Exact score for `pos` from the perspective of the side to move.
///
/// Searches all legal move sequences to game end with alpha-beta pruning.
/// Handles terminal positions (game over) and passes (no legal moves)
/// directly, matching the semantics of the Edax evaluator.
///
/// The score is bounded to [-64, 64]. A one-shot convenience wrapper: it
/// allocates a fresh transposition table per call, so callers that evaluate
/// many positions should instead reuse a single [`Solver`] (see its docs).
pub fn exact_score(pos: &Position) -> i32 {
    Solver::new().exact_score(pos)
}

/// Exact score together with the number of search nodes visited. Used by the
/// `bench` subcommand; `exact_score` runs the identical search without exposing
/// the node count.
pub fn exact_score_with_nodes(pos: &Position) -> (i32, u64) {
    Solver::new().exact_score_with_nodes(pos)
}

/// Score bounds.
const SCORE_MIN: i32 = -64;
const SCORE_MAX: i32 = 64;

// ---------------------------------------------------------------------------
// Transposition table
// ---------------------------------------------------------------------------
//
// Maps a position to a [lower, upper] bound on its exact score (plus the best
// move found, used for ordering). Because an exact endgame score is intrinsic
// to the position — it never depends on the path taken to reach it — a stored
// entry stays valid for the lifetime of the table, across every search. So the
// table is never invalidated or cleared: it only accumulates and refines bounds
// (and a warm table from earlier positions speeds up later ones).
//
// A single shared [`Search`]/[`Solver`] owns the table and is reused across
// positions (see [`Solver`]); the table is allocated once and only the node
// counter resets per search. We deliberately avoid a `thread_local!` for the
// reuse — it reads worse than an explicit owner — and only because threading an
// owned `Search` through the callers costs nothing per node (it is borrowed as
// `&mut self`, never moved). Reach for `thread_local!` only if an explicit
// owner ever turns out to carry a real performance penalty.

/// Transposition table size as a power of two (number of slots). Swept 17..23:
/// node counts barely move above ~2^19 (collisions are already rare at the
/// benchmarked depths), so the win is cache locality — smaller is faster until
/// collisions start adding nodes on the deepest searches (2^17 regressed the
/// 20-empty level). 2^19 (a 12 MB table) is the knee: ~all of the locality win
/// while keeping 2× the headroom of 2^18 for solves deeper than the benchmark.
const TT_BITS: u32 = 19;
const TT_SIZE: usize = 1 << TT_BITS;
const TT_MASK: u64 = TT_SIZE as u64 - 1;

/// Minimum empties at which the transposition table is consulted. Below this,
/// transpositions are too rare for the probe/store to pay for itself, so the
/// search runs without touching the table. (The TT is only wired into the
/// ordered search, so the effective floor is `max(SORT_MIN_EMPTIES, this)`.)
/// Swept 6..10: 6 and 7 are within noise of each other and both clearly beat
/// 8/10; 7 (don't probe the very numerous empties-6 nodes, where the probe's
/// overhead roughly cancels its node savings) edges ahead at the less-noisy
/// 16/18-empty levels. Re-sweep when the search shape changes.
const TT_MIN_EMPTIES: u32 = 7;

/// Sentinel "no move" square (a real square is 0..64).
const NO_MOVE: u8 = 64;

/// One transposition-table slot: the position (stored in full for exact
/// collision detection — a partial key risks returning a wrong score) and a
/// `[lower, upper]` bound on its exact score, plus the best move for ordering.
/// A slot is empty iff `player | opponent == 0` (the disc-free board never
/// occurs as a real interior search node).
#[derive(Clone, Copy)]
struct TtEntry {
    player: u64,
    opponent: u64,
    lower: i8,
    upper: i8,
    best_move: u8,
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

/// Hash a position to a table slot. A 64-bit mix of both bitboards, masked to
/// the table size (a power of two).
#[inline]
fn tt_index(player: u64, opponent: u64) -> usize {
    let mut h = player.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    h ^= opponent.wrapping_mul(0xC2B2_AE3D_27D4_EB4F).rotate_left(32);
    h ^= h >> 29;
    h = h.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    h ^= h >> 32;
    (h & TT_MASK) as usize
}

/// Reusable exact solver: owns a transposition table shared across the
/// positions it evaluates. Exact scores are position-intrinsic, so cross-
/// position reuse is sound and warms the table. Construct once, evaluate many —
/// callers that loop (the `bench` subcommand, cache workers, `batch_evaluate`)
/// hold a single `Solver` so the multi-MB table is allocated only once.
pub struct Solver {
    search: Search,
}

impl Solver {
    /// Allocate a solver with a fresh (empty) transposition table.
    pub fn new() -> Self {
        Solver {
            search: Search::new(),
        }
    }

    /// Exact score for `pos`, reusing this solver's transposition table.
    pub fn exact_score(&mut self, pos: &Position) -> i32 {
        self.search.nodes = 0;
        self.search
            .search_exact(pos, SCORE_MIN, SCORE_MAX, pos.empties())
    }

    /// Exact score plus the number of search nodes visited for this position.
    pub fn exact_score_with_nodes(&mut self, pos: &Position) -> (i32, u64) {
        self.search.nodes = 0;
        let score = self
            .search
            .search_exact(pos, SCORE_MIN, SCORE_MAX, pos.empties());
        (score, self.search.nodes)
    }
}

impl Default for Solver {
    fn default() -> Self {
        Solver::new()
    }
}

/// Minimum empties at which the deep search orders moves (by opponent mobility).
/// Below this, moves are searched in natural order — near the leaves the
/// ordering work costs more than the pruning it buys.
///
/// This is an empirically tuned crossover (ordering cost vs. extra nodes from
/// worse PVS ordering); re-tune it when anything shifts that balance — a
/// transposition table's hash move could allow a higher value, cheaper
/// flip/`get_moves` a lower one, richer ordering either way.
const SORT_MIN_EMPTIES: u32 = 6;

// ---------------------------------------------------------------------------
// Leaf solvers (ported from Edax endgame.c)
// ---------------------------------------------------------------------------

/// Game-over score when no moves remain: winner gets all empties.
/// Port of `board_solve` from Edax.
fn solve_game_over(player: u64, n_empties: u32) -> i32 {
    let n = n_empties as i32;
    let diff = 2 * player.count_ones() as i32 - 64 + n; // player_discs - opponent_discs
    if diff > 0 {
        diff + n
    } else if diff < 0 {
        diff - n
    } else {
        0
    }
}

// --- last-move flip count (Edax `count_last_flip`) -------------------------
//
// When a move is the board's *only* empty square, the four lines through it are
// otherwise full, so every non-player cell on a line is an opponent disc. The
// number of discs the move flips on one 8-cell line then depends only on the
// player-disc pattern and the move's position in the line — a value we look up
// in `COUNT_FLIP`. Tables are generated at compile time rather than hardcoded.

/// `COUNT_FLIP[i][pattern]` = 2× discs flipped by playing at line-position `i`,
/// where `pattern` bit j is set iff the player holds line-cell j and every
/// other (non-`i`) cell is an opponent disc. Doubled to ease disc-difference
/// arithmetic, matching Edax.
const COUNT_FLIP: [[u8; 256]; 8] = {
    let mut table = [[0u8; 256]; 8];
    let mut i = 0;
    while i < 8 {
        let mut p = 0usize;
        while p < 256 {
            let mut flips = 0u32;
            // Walk right of i: opponent cells flip once a player cell closes them.
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
            // Walk left of i.
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

/// Diagonal masks per square: `[0]` = the ╲ diagonal, `[1]` = the ╱ diagonal.
const MASK_DIAG: [[u64; 64]; 2] = {
    let mut m = [[0u64; 64]; 2];
    let mut pos = 0;
    while pos < 64 {
        let x = (pos % 8) as i32;
        let y = (pos / 8) as i32;
        let mut sq = 0;
        while sq < 64 {
            let sx = sq % 8;
            let sy = sq / 8;
            if sx - sy == x - y {
                m[0][pos] |= 1u64 << sq;
            }
            if sx + sy == x + y {
                m[1][pos] |= 1u64 << sq;
            }
            sq += 1;
        }
        pos += 1;
    }
    m
};

/// Gather column `x` (bits x, x+8, …, x+56) into a contiguous 8-bit value,
/// bit r = row r.
#[inline]
fn pack_v(p: u64, x: u32) -> usize {
    (((p >> x) & 0x0101_0101_0101_0101).wrapping_mul(0x0102_0408_1020_4080) >> 56) as usize
}

/// Gather a diagonal-masked bitboard into 8 bits, bit c = column c. Each
/// diagonal cell sits in a distinct row/column, so the bytes don't collide.
#[inline]
fn pack_d(pm: u64) -> usize {
    (pm.wrapping_mul(0x0101_0101_0101_0101) >> 56) as usize
}

/// 2× the number of discs `player` flips by playing at `pos`, valid only when
/// `pos` is the board's only empty square (the [`solve_1`] invariant).
fn count_last_flip(pos: u32, player: u64) -> i32 {
    let x = (pos & 7) as usize;
    let y = (pos >> 3) as usize;
    let mut n = COUNT_FLIP[x][((player >> (y * 8)) & 0xFF) as usize]; // row
    n += COUNT_FLIP[y][pack_v(player, x as u32)]; // column
    n += COUNT_FLIP[x][pack_d(player & MASK_DIAG[0][pos as usize])]; // ╲
    n += COUNT_FLIP[x][pack_d(player & MASK_DIAG[1][pos as usize])]; // ╱
    n as i32
}

/// 1-empty leaf solver. Returns the exact score from `player`'s perspective
/// for a full board with a single empty square at `sq`.
///
/// With one empty, every move is forced (the lone empty is the only candidate),
/// so no search window is needed. A pure function: the calling `Search` method
/// accounts for this node.
fn solve_1(player: u64, sq: u32) -> i32 {
    // Differential after `player` places at `sq` before flips: +1 disc for the
    // player. Each flipped disc then shifts the differential by a further 2.
    let score_base = 2 * player.count_ones() as i32 - 62;

    // The side to move is forced to play at `sq` whenever that move is legal.
    let n_flips = count_last_flip(sq, player);
    if n_flips != 0 {
        return score_base + n_flips;
    }

    // Player cannot play, so it passes; the opponent is then forced to play.
    let opponent = !player & !(1u64 << sq);
    let n_flips_opp = count_last_flip(sq, opponent);
    if n_flips_opp != 0 {
        return score_base - 2 - n_flips_opp;
    }

    // Neither side can play: game over with one empty square.
    solve_game_over(player, 1)
}

/// Mutable state for one exact search: the running count of nodes visited and
/// the transposition table. The recursive search routines are methods so this
/// state is explicit rather than a thread-local global.
struct Search {
    nodes: u64,
    tt: Vec<TtEntry>,
}

impl Search {
    /// A searcher with a freshly allocated transposition table.
    fn new() -> Self {
        Search {
            nodes: 0,
            tt: vec![TtEntry::default(); TT_SIZE],
        }
    }

    /// Look up `pos` in the table. Returns a copy of the entry on an exact hit
    /// (full position match), `None` otherwise. Caller must ensure the table is
    /// non-empty.
    #[inline]
    fn tt_probe(&self, pos: &Position) -> Option<TtEntry> {
        let e = self.tt[tt_index(pos.player, pos.opponent)];
        if e.player == pos.player && e.opponent == pos.opponent && (e.player | e.opponent) != 0 {
            Some(e)
        } else {
            None
        }
    }

    /// Record a `[lower, upper]` bound (and best move) for `pos`. If the slot
    /// already holds this position, the bounds are refined (intersected) and a
    /// real best move is kept; otherwise the slot is overwritten (always-
    /// replace). Both old and new bounds are valid for the intrinsic exact
    /// score, so intersecting them is sound.
    #[inline]
    fn tt_store(&mut self, pos: &Position, lower: i8, upper: i8, best_move: u8) {
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

    /// 2-empty leaf solver: a negamax alpha-beta search over a full board with
    /// empties at `x1` and `x2`. Returns the score from `player`'s perspective.
    /// Counts one node per 1-empty child visited.
    fn solve_2(
        &mut self,
        player: u64,
        opponent: u64,
        mut alpha: i32,
        beta: i32,
        x1: u32,
        x2: u32,
    ) -> i32 {
        const NONE: i32 = SCORE_MIN - 1; // below any real score
        let mut best = NONE;

        // Player tries x1.
        let f1 = Position::flip_mask(x1, player, opponent);
        if f1 != 0 {
            let moved = player | f1 | (1u64 << x1);
            let child_player = opponent & !moved; // opponent to move, only x2 empty
            best = -solve_1(child_player, x2);
            self.nodes += 1;
            if best > alpha {
                alpha = best;
            }
        }

        // Player tries x2 unless x1 already caused a beta cutoff.
        if alpha < beta {
            let f2 = Position::flip_mask(x2, player, opponent);
            if f2 != 0 {
                let moved = player | f2 | (1u64 << x2);
                let child_player = opponent & !moved; // opponent to move, only x1 empty
                let s = -solve_1(child_player, x1);
                self.nodes += 1;
                if s > best {
                    best = s;
                }
            }
        }

        if best != NONE {
            return best;
        }

        // Player has no move and passes. If the opponent also has no move the game
        // is over; otherwise search the opponent's reply and negate.
        if Position::flip_mask(x1, opponent, player) == 0
            && Position::flip_mask(x2, opponent, player) == 0
        {
            return solve_game_over(player, 2);
        }
        -self.solve_2(opponent, player, -beta, -alpha, x1, x2)
    }

    /// 3-empty leaf solver: a fail-soft negamax alpha-beta over the three empties
    /// at `x1`, `x2`, `x3`, recursing into [`solve_2`]. Returns the score from
    /// `player`'s perspective. Leaf node accounting happens in `solve_2`.
    fn solve_3(
        &mut self,
        player: u64,
        opponent: u64,
        mut alpha: i32,
        beta: i32,
        x1: u32,
        x2: u32,
        x3: u32,
    ) -> i32 {
        const NONE: i32 = SCORE_MIN - 1;
        let mut best = NONE;

        // (sq, the other two empties) for each candidate move.
        for &(sq, a, b) in &[(x1, x2, x3), (x2, x1, x3), (x3, x1, x2)] {
            if alpha >= beta {
                break;
            }
            let f = Position::flip_mask(sq, player, opponent);
            if f != 0 {
                let moved = player | f | (1u64 << sq); // child opponent (just-moved discs)
                let child_player = opponent & !moved; // opponent to move, empties a and b
                let s = -self.solve_2(child_player, moved, -beta, -alpha, a, b);
                if s > best {
                    best = s;
                    if best > alpha {
                        alpha = best;
                    }
                }
            }
        }

        if best != NONE {
            return best;
        }

        // Player passes. Game over if the opponent also has no move here.
        if Position::flip_mask(x1, opponent, player) == 0
            && Position::flip_mask(x2, opponent, player) == 0
            && Position::flip_mask(x3, opponent, player) == 0
        {
            return solve_game_over(player, 3);
        }
        -self.solve_3(opponent, player, -beta, -alpha, x1, x2, x3)
    }

    /// 4-empty leaf solver: a fail-soft negamax alpha-beta over the four empties,
    /// recursing into [`solve_3`]. Returns the score from `player`'s perspective.
    fn solve_4(
        &mut self,
        player: u64,
        opponent: u64,
        mut alpha: i32,
        beta: i32,
        x1: u32,
        x2: u32,
        x3: u32,
        x4: u32,
    ) -> i32 {
        const NONE: i32 = SCORE_MIN - 1;
        let mut best = NONE;

        // (sq, the other three empties) for each candidate move.
        for &(sq, a, b, c) in &[
            (x1, x2, x3, x4),
            (x2, x1, x3, x4),
            (x3, x1, x2, x4),
            (x4, x1, x2, x3),
        ] {
            if alpha >= beta {
                break;
            }
            let f = Position::flip_mask(sq, player, opponent);
            if f != 0 {
                let moved = player | f | (1u64 << sq);
                let child_player = opponent & !moved;
                let s = -self.solve_3(child_player, moved, -beta, -alpha, a, b, c);
                if s > best {
                    best = s;
                    if best > alpha {
                        alpha = best;
                    }
                }
            }
        }

        if best != NONE {
            return best;
        }

        // Player passes. Game over if the opponent also has no move here.
        if Position::flip_mask(x1, opponent, player) == 0
            && Position::flip_mask(x2, opponent, player) == 0
            && Position::flip_mask(x3, opponent, player) == 0
            && Position::flip_mask(x4, opponent, player) == 0
        {
            return solve_game_over(player, 4);
        }
        -self.solve_4(opponent, player, -beta, -alpha, x1, x2, x3, x4)
    }

    /// Dispatch to the right routine for `empties`: the leaf solver for four or
    /// fewer empties, the unordered PVS search below `SORT_MIN_EMPTIES`, the
    /// ordered PVS search above it.
    #[inline]
    fn search_exact(&mut self, pos: &Position, alpha: i32, beta: i32, empties: u32) -> i32 {
        if empties <= 4 {
            self.solve_leaf(pos, alpha, beta, empties)
        } else if empties < SORT_MIN_EMPTIES {
            self.alphabeta_nosort(pos, alpha, beta, empties)
        } else {
            self.alphabeta_exact(pos, alpha, beta, empties)
        }
    }

    /// Leaf dispatcher for positions with at most four empties, routing to the
    /// dedicated `solve_1`..`solve_4` solvers (or `final_score` at game end).
    fn solve_leaf(&mut self, pos: &Position, alpha: i32, beta: i32, empties: u32) -> i32 {
        self.nodes += 1;
        match empties {
            0 => pos.final_score(),
            1 => {
                let sq = (!pos.occupied()).trailing_zeros();
                solve_1(pos.player, sq)
            }
            2 => {
                let mut empty = !pos.occupied();
                let x1 = empty.trailing_zeros();
                empty &= empty - 1;
                let x2 = empty.trailing_zeros();
                self.solve_2(pos.player, pos.opponent, alpha, beta, x1, x2)
            }
            3 => {
                let mut empty = !pos.occupied();
                let x1 = empty.trailing_zeros();
                empty &= empty - 1;
                let x2 = empty.trailing_zeros();
                empty &= empty - 1;
                let x3 = empty.trailing_zeros();
                self.solve_3(pos.player, pos.opponent, alpha, beta, x1, x2, x3)
            }
            _ => {
                let mut empty = !pos.occupied();
                let x1 = empty.trailing_zeros();
                empty &= empty - 1;
                let x2 = empty.trailing_zeros();
                empty &= empty - 1;
                let x3 = empty.trailing_zeros();
                empty &= empty - 1;
                let x4 = empty.trailing_zeros();
                self.solve_4(pos.player, pos.opponent, alpha, beta, x1, x2, x3, x4)
            }
        }
    }

    /// Negamax with PVS and move ordering, for `empties >= SORT_MIN_EMPTIES`.
    /// Children dispatch through [`Search::search_exact`] at the recursion
    /// boundary, so this hot path never re-tests the leaf or unordered cases.
    ///
    /// At `empties >= TT_MIN_EMPTIES` (and when a table is present) the node is
    /// looked up in the transposition table first: a sufficient stored bound
    /// returns immediately, otherwise the stored best move seeds the ordering
    /// and the final bound is written back.
    fn alphabeta_exact(
        &mut self,
        pos: &Position,
        mut alpha: i32,
        mut beta: i32,
        empties: u32,
    ) -> i32 {
        self.nodes += 1;

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
                // Narrow the search window to the still-unknown interval.
                if lo > alpha {
                    alpha = lo;
                }
                if hi < beta {
                    beta = hi;
                }
                hash_move = e.best_move;
            }
        }
        // Window actually searched (after any TT narrowing); the result is
        // classified against this to decide whether it is a bound or exact.
        let search_alpha = alpha;
        let search_beta = beta;

        let moves = pos.get_moves();
        if moves == 0 {
            let passed = pos.pass_move();
            if passed.get_moves() == 0 {
                return pos.final_score();
            }
            return -self.alphabeta_exact(&passed, -beta, -alpha, empties);
        }

        // Order children by opponent mobility (fewest replies first).
        let mut move_list: Vec<(u32, u32, Position)> =
            Vec::with_capacity(moves.count_ones() as usize);
        let mut remaining = moves;
        while remaining != 0 {
            let cell = remaining.trailing_zeros();
            remaining &= remaining - 1;
            let child = pos.do_move(cell);
            move_list.push((child.get_moves().count_ones(), cell, child));
        }
        move_list.sort_unstable_by_key(|&(mobility, _, _)| mobility);

        // A transposition-table best move is the strongest ordering signal:
        // pull it to the front, keeping the mobility order of the rest.
        if hash_move != NO_MOVE {
            if let Some(i) = move_list.iter().position(|&(_, c, _)| c as u8 == hash_move) {
                move_list[..=i].rotate_right(1);
            }
        }

        // Principal Variation Search: search the first (best-ordered) move with the
        // full window, then probe each sibling with a null window and re-search only
        // on a fail-high. No empties gate — Edax applies this at every node.
        let mut best_cell = NO_MOVE;
        let mut first = true;
        for &(_, cell, ref child) in &move_list {
            let score = if first {
                -self.search_exact(child, -beta, -alpha, empties - 1)
            } else {
                let probe = -self.search_exact(child, -alpha - 1, -alpha, empties - 1);
                if probe > alpha && probe < beta {
                    -self.search_exact(child, -beta, -alpha, empties - 1)
                } else {
                    probe
                }
            };
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
            // Classify the fail-hard result against the searched window: at or
            // below `search_alpha` it is only an upper bound; at or above
            // `search_beta` only a lower bound; strictly inside, exact.
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

    /// Negamax with PVS but no move ordering, for the few-empties range
    /// (`5 ..< SORT_MIN_EMPTIES`) where ordering costs more than it saves.
    /// Moves are tried in natural board order with no move-list allocation.
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
                let probe = -self.search_exact(&child, -alpha - 1, -alpha, empties - 1);
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
}

/// Evaluate a batch of positions, returning one score per position
/// in the same order.  Handles game-end and pass positions without
/// invoking the full search.
pub fn batch_evaluate(positions: &[Position]) -> Vec<i32> {
    let mut solver = Solver::new();
    positions.iter().map(|p| solver.exact_score(p)).collect()
}

/// Depth-limited evaluation for use in gameplay. Searches `depth` plies
/// and applies a heuristic at the leaves.
pub fn depth_limited_score(
    pos: &Position,
    depth: u32,
    weights: &Weights,
    features: &Features,
) -> i32 {
    alphabeta(pos, depth, weights, features, SCORE_MIN, SCORE_MAX)
}

/// Pick the best legal move for the side to move. Returns `None` when there
/// are no legal moves.
pub fn best_move(
    pos: &Position,
    depth: u32,
    exact_empties: u32,
    weights: &Weights,
    features: &Features,
) -> Option<u32> {
    let moves = pos.get_moves();
    if moves == 0 {
        return None;
    }

    if pos.empties() <= exact_empties {
        return best_move_exact(pos);
    }

    let mut alpha = SCORE_MIN;
    let mut best_cell = 0u32;

    let mut remaining = moves;
    while remaining != 0 {
        let cell = remaining.trailing_zeros();
        remaining &= remaining - 1;
        let child = pos.do_move(cell);
        let score = -alphabeta(
            &child,
            depth.saturating_sub(1),
            weights,
            features,
            -SCORE_MAX,
            -alpha,
        );
        if score > alpha {
            alpha = score;
            best_cell = cell;
        }
    }

    Some(best_cell)
}

/// Pick the best legal move using exact search to game end.
fn best_move_exact(pos: &Position) -> Option<u32> {
    let moves = pos.get_moves();
    if moves == 0 {
        return None;
    }

    let mut alpha = SCORE_MIN;
    let mut best_cell = 0u32;
    let empties = pos.empties();
    let mut searcher = Search::new();

    let mut remaining = moves;
    while remaining != 0 {
        let cell = remaining.trailing_zeros();
        remaining &= remaining - 1;
        let child = pos.do_move(cell);
        let score = -searcher.search_exact(&child, -SCORE_MAX, -alpha, empties - 1);
        if score > alpha {
            alpha = score;
            best_cell = cell;
        }
    }

    Some(best_cell)
}

/// Negamax with alpha-beta pruning and depth limit.
fn alphabeta(
    pos: &Position,
    depth: u32,
    weights: &Weights,
    features: &Features,
    mut alpha: i32,
    beta: i32,
) -> i32 {
    let moves = pos.get_moves();
    if moves == 0 {
        let passed = pos.pass_move();
        if passed.get_moves() == 0 {
            return pos.final_score();
        }
        return -alphabeta(&passed, depth, weights, features, -beta, -alpha);
    }

    if depth == 0 {
        return heuristic(pos, weights, features);
    }

    let mut remaining = moves;
    while remaining != 0 {
        let cell = remaining.trailing_zeros();
        remaining &= remaining - 1;
        let child = pos.do_move(cell);
        let score = -alphabeta(&child, depth - 1, weights, features, -beta, -alpha);
        if score > alpha {
            alpha = score;
            if alpha >= beta {
                break;
            }
        }
    }

    alpha
}

fn heuristic(pos: &Position, weights: &Weights, features: &Features) -> i32 {
    let score = weights.evaluate(pos, features);
    score.round().clamp(SCORE_MIN as f32, SCORE_MAX as f32) as i32
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    // --- helpers -----------------------------------------------------------

    /// Flip mask for `player` playing at `sq` on a board whose other cells are
    /// `opponent` — the reference the leaf solvers use directly.
    fn flips_for(sq: u32, player: u64, opponent: u64) -> u64 {
        Position::flip_mask(sq, player, opponent)
    }

    // --- solve_game_over ---------------------------------------------------

    #[test]
    fn solve_game_over_player_wins() {
        // 32 player discs, 30 opponent discs, 2 empties → player wins, gets empties
        // score = (32+2) - 30 = 4  → diff+n = (32-30)+2 + 2 = 6
        // player=32, opp=30, diff = 2*32-64+2=2, score = 2+2=4
        let player: u64 = 0x00000000FFFFFFFF; // 32 bits
                                              // opponent: 30 bits in upper half, leave 2 empty
        let opponent: u64 = 0x3FFFFFFF00000000; // 30 bits
        assert_eq!(player.count_ones(), 32);
        assert_eq!(opponent.count_ones(), 30);
        let score = solve_game_over(player, 2);
        assert_eq!(score, 4); // diff=2, score=diff+n=4
    }

    #[test]
    fn solve_game_over_opponent_wins() {
        // 30 player, 32 opponent, 2 empties → opponent wins
        // diff = 2*30-64+2 = -2, score = diff-n = -4
        let player: u64 = 0x000000003FFFFFFF; // 30 bits
        let opponent: u64 = 0xFFFFFFFF00000000; // 32 bits
        assert_eq!(player.count_ones(), 30);
        assert_eq!(opponent.count_ones(), 32);
        let score = solve_game_over(player, 2);
        assert_eq!(score, -4);
    }

    #[test]
    fn solve_game_over_tie() {
        // 31 each, 2 empties → tie
        let player: u64 = 0x000000007FFFFFFF; // 31 bits
        let opponent: u64 = 0x7FFFFFFF00000000; // 31 bits
        assert_eq!(player.count_ones(), 31);
        assert_eq!(opponent.count_ones(), 31);
        let score = solve_game_over(player, 2);
        assert_eq!(score, 0);
    }

    #[test]
    fn solve_game_over_zero_empties_matches_final_score() {
        // With 0 empties, solve_game_over should match Position::final_score.
        let player: u64 = 0x00000000FFFFFFFF;
        let opponent: u64 = 0xFFFFFFFF00000000;
        let pos = Position { player, opponent };
        assert_eq!(solve_game_over(player, 0), pos.final_score());
    }

    // --- leaf solver references --------------------------------------------

    /// Independent exact negamax that never uses the `solve_1`/`solve_2`
    /// fast paths, so it serves as ground truth for them.
    fn naive_exact(pos: &Position) -> i32 {
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

    /// Empty-square indices exercising corners, edges, centre and both
    /// board halves.
    const SQUARES: &[u32] = &[0, 1, 7, 8, 9, 14, 27, 28, 35, 36, 49, 55, 56, 62, 63];

    /// Deterministic disc-layout patterns (assigned to the player; the rest of
    /// the board becomes the opponent). Chosen to vary density and adjacency:
    /// empty, full, alternating bits, row/column/diagonal stripes.
    const PATTERNS: &[u64] = &[
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

    // --- solve_1 -----------------------------------------------------------

    #[test]
    fn solve_1_player_forced_play() {
        // Full board, empty at a1 (0); play a1 flips b1 (opponent) anchored on c1.
        // player = {c1}, everything else opponent. After play: player 3, opp 61 → -58.
        let sq = 0u32;
        let player = 1u64 << 2; // c1
        let pos = Position {
            player,
            opponent: !player & !(1u64 << sq),
        };
        assert_eq!(pos.empties(), 1);
        assert_eq!(solve_1(player, sq), -58);
        assert_eq!(solve_1(player, sq), naive_exact(&pos));
    }

    #[test]
    fn solve_1_game_over_player_wins() {
        // Full board, lone empty, opponent absent: nobody can flip → game over.
        let sq = 0u32;
        let player = !(1u64 << sq); // 63 discs
        assert_eq!(solve_1(player, sq), 64);
    }

    #[test]
    fn solve_1_matches_naive() {
        // Cross every empty square with every disc pattern.
        for &sq in SQUARES {
            let empty = 1u64 << sq;
            for &pat in PATTERNS {
                let player = pat & !empty;
                let opponent = !player & !empty;
                let pos = Position { player, opponent };
                assert_eq!(
                    solve_1(player, sq),
                    naive_exact(&pos),
                    "player={player:#x} sq={sq}"
                );
            }
        }
    }

    #[test]
    fn count_last_flip_matches_flips_for() {
        // On a full board with one empty, the table lookup must equal 2× the
        // popcount of the full flip mask, for every square and pattern.
        for &sq in SQUARES {
            let empty = 1u64 << sq;
            for &pat in PATTERNS {
                let player = pat & !empty;
                let opponent = !player & !empty;
                let expected = 2 * flips_for(sq, player, opponent).count_ones() as i32;
                assert_eq!(
                    count_last_flip(sq, player),
                    expected,
                    "player={player:#x} sq={sq}"
                );
            }
        }
    }

    // --- solve_2 -----------------------------------------------------------

    // The solve_* helpers take a reused `Search`: the solver never touches the
    // transposition table, but constructing a fresh `Search` allocates it, so
    // reuse keeps these tight test loops from re-allocating it thousands of
    // times.
    fn run_solve_2(s: &mut Search, player: u64, opponent: u64) -> i32 {
        let mut empty = !(player | opponent);
        let x1 = empty.trailing_zeros();
        empty &= empty - 1;
        let x2 = empty.trailing_zeros();
        s.solve_2(player, opponent, SCORE_MIN, SCORE_MAX, x1, x2)
    }

    /// Every ordered pair of distinct empty squares from [`SQUARES`], crossed
    /// with every disc pattern. Returns the (player, opponent) layouts.
    fn two_empty_layouts() -> impl Iterator<Item = (u64, u64)> {
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

    #[test]
    fn solve_2_matches_naive() {
        let mut s = Search::new();
        for (player, opponent) in two_empty_layouts() {
            let pos = Position { player, opponent };
            assert_eq!(pos.empties(), 2);
            assert_eq!(
                run_solve_2(&mut s, player, opponent),
                naive_exact(&pos),
                "player={player:#x} opponent={opponent:#x}"
            );
        }
    }

    #[test]
    fn solve_2_respects_window() {
        // Full window equals the truth (checked above); narrow windows must
        // fail soft on the correct side of the bound.
        let mut s = Search::new();
        for (player, opponent) in two_empty_layouts() {
            let truth = naive_exact(&Position { player, opponent });
            let mut e = !(player | opponent);
            let x1 = e.trailing_zeros();
            e &= e - 1;
            let x2 = e.trailing_zeros();

            // Window entirely above the true score → fail low (result <= alpha).
            let lo = truth + 1;
            let r = s.solve_2(player, opponent, lo, lo + 1, x1, x2);
            assert!(r <= lo, "fail-low: r={r} alpha={lo} truth={truth}");
            // Window entirely below the true score → fail high (result >= beta).
            let hi = truth - 1;
            let r = s.solve_2(player, opponent, hi - 1, hi, x1, x2);
            assert!(r >= hi, "fail-high: r={r} beta={hi} truth={truth}");
        }
    }

    #[test]
    fn solve_1_and_solve_2_drive_exact_score() {
        // End-to-end: exact_score (which dispatches to the leaf solvers at
        // 1 and 2 empties) agrees with the independent naive solver. One reused
        // solver so the table is allocated once across the whole battery.
        let mut solver = Solver::new();
        for &sq in SQUARES {
            let empty = 1u64 << sq;
            for &pat in PATTERNS {
                let player = pat & !empty;
                let opponent = !player & !empty;
                let pos = Position { player, opponent };
                assert_eq!(solver.exact_score(&pos), naive_exact(&pos));
            }
        }
        for (player, opponent) in two_empty_layouts() {
            let pos = Position { player, opponent };
            assert_eq!(solver.exact_score(&pos), naive_exact(&pos));
        }
    }

    // --- solve_3 / solve_4 -------------------------------------------------

    /// Player/opponent layouts for a fixed set of empty squares, one per
    /// pattern. The empties become the board's only empty cells.
    fn layouts_for(empties: &[u32]) -> impl Iterator<Item = (u64, u64)> + '_ {
        let mask = empties.iter().fold(0u64, |m, &s| m | (1u64 << s));
        PATTERNS.iter().map(move |&pat| {
            let player = pat & !mask;
            (player, !player & !mask)
        })
    }

    fn run_solve_3(s: &mut Search, player: u64, opponent: u64) -> i32 {
        let mut e = !(player | opponent);
        let x1 = e.trailing_zeros();
        e &= e - 1;
        let x2 = e.trailing_zeros();
        e &= e - 1;
        let x3 = e.trailing_zeros();
        s.solve_3(player, opponent, SCORE_MIN, SCORE_MAX, x1, x2, x3)
    }

    fn run_solve_4(s: &mut Search, player: u64, opponent: u64) -> i32 {
        let mut e = !(player | opponent);
        let x1 = e.trailing_zeros();
        e &= e - 1;
        let x2 = e.trailing_zeros();
        e &= e - 1;
        let x3 = e.trailing_zeros();
        e &= e - 1;
        let x4 = e.trailing_zeros();
        s.solve_4(player, opponent, SCORE_MIN, SCORE_MAX, x1, x2, x3, x4)
    }

    /// Smaller square set (includes all four corners) to bound the 4-empty
    /// combination count.
    const SQUARES4: &[u32] = &[0, 7, 9, 28, 35, 49, 56, 63];

    #[test]
    fn solve_3_matches_naive() {
        let mut s = Search::new();
        let n = SQUARES.len();
        for i in 0..n {
            for j in (i + 1)..n {
                for k in (j + 1)..n {
                    let empties = [SQUARES[i], SQUARES[j], SQUARES[k]];
                    for (player, opponent) in layouts_for(&empties) {
                        let pos = Position { player, opponent };
                        assert_eq!(pos.empties(), 3);
                        assert_eq!(
                            run_solve_3(&mut s, player, opponent),
                            naive_exact(&pos),
                            "player={player:#x} opponent={opponent:#x}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn solve_4_matches_naive() {
        let mut s = Search::new();
        let n = SQUARES4.len();
        for i in 0..n {
            for j in (i + 1)..n {
                for k in (j + 1)..n {
                    for l in (k + 1)..n {
                        let empties = [SQUARES4[i], SQUARES4[j], SQUARES4[k], SQUARES4[l]];
                        for (player, opponent) in layouts_for(&empties) {
                            let pos = Position { player, opponent };
                            assert_eq!(pos.empties(), 4);
                            assert_eq!(
                                run_solve_4(&mut s, player, opponent),
                                naive_exact(&pos),
                                "player={player:#x} opponent={opponent:#x}"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn solve_3_and_4_respect_window() {
        // Fail-soft: tight windows must land on the correct side of the bound.
        let check = |player: u64, opponent: u64, solve: &dyn Fn(u64, u64, i32, i32) -> i32| {
            let truth = naive_exact(&Position { player, opponent });
            let lo = truth + 1;
            assert!(solve(player, opponent, lo, lo + 1) <= lo, "fail-low");
            let hi = truth - 1;
            assert!(solve(player, opponent, hi - 1, hi) >= hi, "fail-high");
        };
        // `RefCell` so the `&dyn Fn` closure can reuse one `Search` (and its
        // table) rather than allocate per call.
        let s = std::cell::RefCell::new(Search::new());
        let solve3 = |p: u64, o: u64, a: i32, b: i32| {
            let mut e = !(p | o);
            let x1 = e.trailing_zeros();
            e &= e - 1;
            let x2 = e.trailing_zeros();
            e &= e - 1;
            let x3 = e.trailing_zeros();
            s.borrow_mut().solve_3(p, o, a, b, x1, x2, x3)
        };
        let n = SQUARES4.len();
        for i in 0..n {
            for j in (i + 1)..n {
                for k in (j + 1)..n {
                    for (player, opponent) in layouts_for(&[SQUARES4[i], SQUARES4[j], SQUARES4[k]])
                    {
                        check(player, opponent, &solve3);
                    }
                }
            }
        }
    }

    #[test]
    fn solve_3_and_4_drive_exact_score() {
        // End-to-end through exact_score's empties==3 / ==4 dispatch. One reused
        // solver so the table is allocated once across the whole battery.
        let mut solver = Solver::new();
        let n = SQUARES4.len();
        for i in 0..n {
            for j in (i + 1)..n {
                for k in (j + 1)..n {
                    for (player, opponent) in layouts_for(&[SQUARES4[i], SQUARES4[j], SQUARES4[k]])
                    {
                        let pos = Position { player, opponent };
                        assert_eq!(solver.exact_score(&pos), naive_exact(&pos));
                    }
                    for l in (k + 1)..n {
                        for (player, opponent) in
                            layouts_for(&[SQUARES4[i], SQUARES4[j], SQUARES4[k], SQUARES4[l]])
                        {
                            let pos = Position { player, opponent };
                            assert_eq!(solver.exact_score(&pos), naive_exact(&pos));
                        }
                    }
                }
            }
        }
    }

    // --- existing tests below ----------------------------------------------

    #[test]
    fn test_exact_score_game_end_full() {
        let pos = Position {
            player: u64::MAX,
            opponent: 0,
        };
        assert!(pos.is_game_end());
        assert_eq!(exact_score(&pos), 64);
    }

    #[test]
    fn test_exact_score_game_end_tie() {
        let pos = Position::new();
        assert!(pos.is_game_end());
        assert_eq!(exact_score(&pos), 0);
    }

    #[test]
    fn test_exact_score_game_end_opponent_wins() {
        let pos = Position {
            player: 0,
            opponent: u64::MAX,
        };
        assert!(pos.is_game_end());
        assert_eq!(exact_score(&pos), -64);
    }

    #[test]
    fn test_exact_score_one_empty() {
        let mut player: u64 = 0;
        let mut opponent: u64 = 0;
        for i in 0..32 {
            player |= 1u64 << i;
        }
        for i in 32..63 {
            opponent |= 1u64 << i;
        }
        let pos = Position { player, opponent };
        assert_eq!(pos.empties(), 1);
        let score = exact_score(&pos);
        assert!(score > 0, "black should win, got {score}");
    }

    #[test]
    fn test_batch_evaluate_game_ends() {
        let positions = vec![
            Position {
                player: u64::MAX,
                opponent: 0,
            },
            Position::new(),
        ];
        let scores = batch_evaluate(&positions);
        assert_eq!(scores, vec![64, 0]);
    }

    /// Verify alpha-beta exact scores match the Edax reference scores.
    ///
    /// Reads `test_data/exact_scores.txt` (generated by `select_reference.py`
    /// from the Edax eval cache), parses each FEN into a [`Position`], computes
    /// [`exact_score`], and asserts it equals the saved Edax score.
    #[test]
    fn test_exact_scores_match_reference() {
        let path = "test_data/exact_scores.txt";
        let content = fs::read_to_string(path).expect("Failed to read reference file");

        // One reused solver: also checks that a shared (warming) transposition
        // table stays correct across many independent positions.
        let mut solver = Solver::new();
        for (line_no, line) in content.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let fen = &line[..66];
            let expected: i32 = line[67..]
                .trim()
                .parse()
                .unwrap_or_else(|_| panic!("Line {}: invalid score", line_no + 1));

            let pos = parse_fen(fen);
            let actual = solver.exact_score(&pos);

            assert_eq!(
                actual,
                expected,
                "Line {}: FEN={fen} expected={expected} actual={actual}",
                line_no + 1,
            );
        }
    }

    #[test]
    fn test_depth_limited_score_game_end() {
        let pos = Position {
            player: u64::MAX,
            opponent: 0,
        };
        let features = Features::edax();
        let weights = Weights::new(features.clone());
        assert_eq!(depth_limited_score(&pos, 0, &weights, &features), 64);
    }

    #[test]
    fn test_best_move_uses_exact_for_few_empties() {
        let mut player: u64 = 0;
        let mut opponent: u64 = 0;
        for i in 0..32 {
            player |= 1u64 << i;
        }
        for i in 32..63 {
            opponent |= 1u64 << i;
        }
        let pos = Position { player, opponent };
        assert_eq!(pos.empties(), 1);
        let features = Features::edax();
        let weights = Weights::new(features.clone());
        let mv = best_move(&pos, 1, 12, &weights, &features);
        assert!(mv.is_some(), "best_move should return a move with 1 empty");
    }

    #[test]
    fn test_depth_limited_score_bounded() {
        let pos = Position::initial();
        let features = Features::edax();
        let weights = Weights::new(features.clone());
        let score = depth_limited_score(&pos, 4, &weights, &features);
        assert!(
            (SCORE_MIN..=SCORE_MAX).contains(&score),
            "score {score} out of bounds"
        );
    }

    #[test]
    fn test_best_move_returns_legal_move() {
        let pos = Position::initial();
        let features = Features::edax();
        let weights = Weights::new(features.clone());
        let mv = best_move(&pos, 4, 12, &weights, &features);
        assert!(mv.is_some());
        let cell = mv.unwrap_or_else(|| unreachable!());
        let moves = pos.get_moves();
        assert!(
            moves & (1u64 << cell) != 0,
            "best_move returned illegal cell {cell}"
        );
    }

    #[test]
    fn test_best_move_none_when_no_moves() {
        let pos = Position {
            player: u64::MAX,
            opponent: 0,
        };
        let features = Features::edax();
        let weights = Weights::new(features.clone());
        assert!(best_move(&pos, 4, 12, &weights, &features).is_none());
    }

    #[test]
    fn test_heuristic_bounded() {
        let pos = Position::initial();
        let features = Features::edax();
        let weights = Weights::new(features.clone());
        let h = heuristic(&pos, &weights, &features);
        assert!(
            (SCORE_MIN..=SCORE_MAX).contains(&h),
            "heuristic {h} out of bounds"
        );
    }

    /// Parse an Edax FEN (66 chars: 64 board + space + side-to-move) into a
    /// [`Position`] where `player` is the side to move.
    fn parse_fen(fen: &str) -> Position {
        let board = fen.as_bytes();
        let side = board[65]; // 'X' or 'O'

        let mut x_discs: u64 = 0; // black
        let mut o_discs: u64 = 0; // white

        for i in 0..64 {
            match board[i] {
                b'X' => x_discs |= 1u64 << i,
                b'O' => o_discs |= 1u64 << i,
                b'-' => { /* empty */ }
                _ => panic!(
                    "Invalid FEN character at position {i}: {}",
                    board[i] as char
                ),
            }
        }

        if side == b'X' {
            // Black (X) to move → player = black discs
            Position {
                player: x_discs,
                opponent: o_discs,
            }
        } else {
            // White (O) to move → player = white discs
            Position {
                player: o_discs,
                opponent: x_discs,
            }
        }
    }
}
