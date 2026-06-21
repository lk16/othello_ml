//! Full recursive Young Brothers Wait Concept (YBWC) parallel search (Step 21).
//!
//! Parallelism lives in the null-window nodes (`alphabeta_exact_nws`), which are
//! the bulk of the tree. At such a node the eldest child is searched first (it
//! usually cuts — "young brothers wait"); only on a fail-low are the younger
//! siblings fanned across worker threads, each searching with the same null
//! window. The first sibling to fail high trips the node's [`CancelNode`], so the
//! others unwind. The principal-variation spine (`alphabeta_exact`) stays
//! single-threaded — workers only ever traverse null-window subtrees — which is
//! what lets the dominant eldest subtree parallelize through *its* null-window
//! descendants rather than leaving one core to carry it (the root-only failure).
//!
//! Threads come from nested `std::thread::scope` calls, bounded by a shared
//! budget ([`ParCtx`]): a split spawns helpers only while the live-thread count
//! is under the cap, and otherwise runs the siblings on the splitting thread —
//! so no thread pool and no unbounded spawning. Workers share one sharded,
//! mutex-guarded transposition table ([`super::tt::SharedTt`]); exact scores are
//! position-intrinsic, so a bound written by any worker is valid for all. No
//! external dependency.

use super::search::{board_parity, Search};
use super::tt::SharedTt;
use crate::eval::pattern::FlatEval;
use crate::othello::position::Position;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;

/// Shared thread budget for one parallel solve: caps the number of live worker
/// threads. Workers acquire a slot before splitting and release it when done.
pub(super) struct ParCtx {
    active: AtomicUsize,
    max: usize,
}

impl ParCtx {
    /// A budget for `max` total worker threads. The initial (root) worker counts
    /// as one, so `active` starts at 1.
    pub(super) fn new(max: usize) -> Self {
        ParCtx {
            active: AtomicUsize::new(1),
            max: max.max(1),
        }
    }

    /// Try to claim one worker slot; returns whether a slot was free.
    pub(super) fn try_acquire(&self) -> bool {
        let mut cur = self.active.load(Ordering::Relaxed);
        loop {
            if cur >= self.max {
                return false;
            }
            match self.active.compare_exchange_weak(
                cur,
                cur + 1,
                Ordering::AcqRel,
                Ordering::Relaxed,
            ) {
                Ok(_) => return true,
                Err(c) => cur = c,
            }
        }
    }

    /// Release a previously acquired worker slot.
    pub(super) fn release(&self) {
        self.active.fetch_sub(1, Ordering::Release);
    }
}

/// A node in the cancellation chain: one per split point. A beta-cutoff trips
/// `stop`; [`CancelNode::cancelled`] walks up the (short) chain so a worker deep
/// in a sub-split also sees an ancestor's cutoff. Shared by `Arc`.
pub(super) struct CancelNode {
    stop: AtomicBool,
    parent: Option<Arc<CancelNode>>,
}

impl CancelNode {
    /// The root of the chain (never tripped from above).
    pub(super) fn root() -> Arc<Self> {
        Arc::new(CancelNode {
            stop: AtomicBool::new(false),
            parent: None,
        })
    }

    /// A child split point under `parent`.
    pub(super) fn child(parent: Option<Arc<CancelNode>>) -> Arc<Self> {
        Arc::new(CancelNode {
            stop: AtomicBool::new(false),
            parent,
        })
    }

    /// Trip this split point (a sibling cut here).
    pub(super) fn cancel(&self) {
        self.stop.store(true, Ordering::Relaxed);
    }

    /// Whether this split point or any ancestor has been tripped.
    pub(super) fn cancelled(&self) -> bool {
        self.stop.load(Ordering::Relaxed)
            || self.parent.as_deref().is_some_and(CancelNode::cancelled)
    }
}

/// Parallel exact solver (full recursive YBWC). Owns the shared transposition
/// table — reused and warmed across the positions it solves, like the sequential
/// [`super::Solver`] — and the worker-thread count.
pub struct ParallelSolver {
    tt: Arc<SharedTt>,
    threads: usize,
    /// Optional trained pattern eval, propagated to every worker for eval-guided
    /// move ordering (Step 34). `None` = the mobility-only baseline ordering.
    /// Iterative-deepening seeding is *not* used on the parallel path (its
    /// sequential passes would be a serial bottleneck — see `Search::solve_root`).
    eval: Option<Arc<FlatEval>>,
}

impl ParallelSolver {
    /// A solver with a fresh shared table and `threads` workers (clamped to ≥ 1).
    pub fn new(threads: usize) -> Self {
        ParallelSolver {
            tt: Arc::new(SharedTt::new()),
            threads: threads.max(1),
            eval: None,
        }
    }

    /// A parallel solver whose every worker uses a trained pattern eval for
    /// eval-guided move ordering (Step 34). The eval is shared (`Arc`); each worker
    /// clones the handle. (Iterative deepening is sequential-only — see
    /// `Search::solve_root` — so the parallel win is ordering, not seeding.)
    pub fn with_eval(threads: usize, eval: Arc<FlatEval>) -> Self {
        ParallelSolver {
            tt: Arc::new(SharedTt::new()),
            threads: threads.max(1),
            eval: Some(eval),
        }
    }

    /// Exact score for `pos` plus the number of search nodes visited. Node counts
    /// are non-deterministic under parallelism (they depend on worker timing and
    /// shared-table contents); the score is always exact.
    pub fn exact_score_with_nodes(&self, pos: &Position) -> (i32, u64) {
        let par = Arc::new(ParCtx::new(self.threads));
        let mut root = Search::worker(
            Arc::clone(&self.tt),
            par,
            CancelNode::root(),
            self.eval.clone(),
        );
        root.parity = board_parity(pos);
        // The root is a PV node whose null-window descendants split across workers;
        // with an eval attached, every worker carries it for eval-guided ordering.
        // `solve_root` skips the (sequential) iterative-deepening seeding on the
        // parallel path — it would be an unparallelized serial bottleneck — so the
        // parallel win here is eval ordering, not ID. See `Search::solve_root`.
        let score = root.solve_root(pos, pos.empties());
        (score, root.nodes)
    }

    /// Exact score for `pos`.
    pub fn exact_score(&self, pos: &Position) -> i32 {
        self.exact_score_with_nodes(pos).0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::alphabeta::Solver;

    /// Parallel YBWC must return the same exact scores as the sequential solver.
    /// Uses a few near-endgame positions deep enough to exercise the split
    /// (empties ≥ `SPLIT_MIN_EMPTIES`), reached by random self-play. Kept small —
    /// this runs in the debug `cargo test` build, where each solve is ~15× slower.
    #[test]
    fn parallel_matches_sequential() {
        let mut rng: u64 = 0xABCD_1234_5678_9F01;
        let mut next = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        };

        // 16 workers stresses the lock-free table's torn-read path under real
        // contention — the failure mode a low thread count would hide (Step 29).
        let par = ParallelSolver::new(16);
        let mut seq = Solver::new();
        let mut checked = 0;
        for _ in 0..12 {
            // Play random moves down to ~16 empties from the initial position.
            let mut pos = Position::initial();
            while pos.empties() > 16 {
                let moves = pos.get_moves();
                if moves == 0 {
                    let passed = pos.pass_move();
                    if passed.get_moves() == 0 {
                        break;
                    }
                    pos = passed;
                    continue;
                }
                let pick = next() % moves.count_ones() as u64;
                let mut m = moves;
                for _ in 0..pick {
                    m &= m - 1;
                }
                pos = pos.do_move(m.trailing_zeros());
            }
            if pos.get_moves() == 0 {
                continue;
            }
            assert_eq!(
                par.exact_score(&pos),
                seq.exact_score(&pos),
                "parallel/sequential score mismatch at player={:#x} opponent={:#x}",
                pos.player,
                pos.opponent
            );
            checked += 1;
        }
        assert!(checked > 0, "no positions exercised the parallel path");
    }

    /// The parallel solver with eval-guided ordering must return the same exact
    /// scores as the plain sequential solver — the eval only reorders moves, so it
    /// cannot change an exact result, on any number of workers. (ID seeding is not
    /// used on the parallel path; see `Search::solve_root`.) Same positions as
    /// `parallel_matches_sequential`.
    #[test]
    fn parallel_with_eval_matches_sequential() {
        use crate::eval::pattern::FlatEval;
        use crate::training::{Features, Weights};

        let weights = Weights::new(Features::edax());
        let flat = Arc::new(FlatEval::from_weights(&weights));
        let par = ParallelSolver::with_eval(16, flat);
        let mut seq = Solver::new();

        let mut rng: u64 = 0xABCD_1234_5678_9F01;
        let mut next = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        };

        let mut checked = 0;
        for _ in 0..12 {
            let mut pos = Position::initial();
            while pos.empties() > 16 {
                let moves = pos.get_moves();
                if moves == 0 {
                    let passed = pos.pass_move();
                    if passed.get_moves() == 0 {
                        break;
                    }
                    pos = passed;
                    continue;
                }
                let pick = next() % moves.count_ones() as u64;
                let mut m = moves;
                for _ in 0..pick {
                    m &= m - 1;
                }
                pos = pos.do_move(m.trailing_zeros());
            }
            if pos.get_moves() == 0 {
                continue;
            }
            assert_eq!(
                par.exact_score(&pos),
                seq.exact_score(&pos),
                "parallel-eval/sequential mismatch at player={:#x} opponent={:#x}",
                pos.player,
                pos.opponent
            );
            checked += 1;
        }
        assert!(checked > 0, "no positions exercised the parallel eval path");
    }
}
