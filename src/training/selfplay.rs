//! Self-play training-data generation (TD(λ) targets).
//!
//! `train-boot` labels a *fixed* corpus of human-game positions with a shallow
//! search whose leaves use the current weights. This module instead **generates
//! its own positions** by letting the current eval play itself, then labels every
//! visited position with a **TD(λ) return** that blends the eval's own bootstrap
//! estimates along the trajectory with the game's **exact terminal** disc
//! differential. Two wins over `train-boot`:
//!
//! - **On-distribution data** — positions are the ones this eval actually reaches,
//!   not human games.
//! - **Better deep targets** — the exact terminal anchors the return, and λ
//!   propagates it inward, instead of a single shallow look at the weak eval.
//!
//! It does **not** break the linear model's ~8-disc capacity ceiling (see
//! `docs/eval-quality.md`); judge it by held-out bias / sign / move-ordering at
//! deep empties, not by chasing sub-disc absolute accuracy.
//!
//! ## TD(λ) target (fixed-perspective formulation)
//!
//! Two-player zero-sum makes a *side-to-move* value alternate sign every ply, which
//! is awkward, so the return is computed from a **fixed reference side (Black)** and
//! converted back at the end. Let a game visit decision states `s_0 … s_{k-1}` then
//! a terminal state. With Black-perspective bootstrap estimates `U(s)` and the
//! exact terminal Black differential `z` (no discount, zero intermediate reward),
//! the forward-view λ-return has the standard backward recursion
//!
//! ```text
//!   G_k = z                                   (terminal)
//!   G_i = (1−λ)·U(s_{i+1}) + λ·G_{i+1}         (i = k−1 … 0)
//! ```
//!
//! The stored label for `s_i` is `G_i` flipped back to side-to-move perspective.
//! `λ→1` is pure Monte-Carlo (label = terminal outcome); `λ→0` is one-ply
//! bootstrapping off the next state's eval.

use crate::eval::pattern::FlatEval;
use crate::othello::position::Position;
use crate::training::weights::Weights;
use crate::training::TrainingExample;
use crate::Solver;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;

/// Knobs for self-play game generation and TD(λ) labelling.
#[derive(Debug, Clone)]
pub struct SelfPlayConfig {
    /// Heuristic search depth for move selection **and** the bootstrap estimates.
    pub depth: u32,
    /// Trust frontier: at empties ≤ this, move selection switches to exact search
    /// and these buckets are **not** emitted as examples (the exact-trained anchor
    /// is left untouched, exactly as `train-boot`).
    pub exact_empties: u32,
    /// Highest empties bucket to emit examples for.
    pub max_empties: u32,
    /// TD(λ) blend in `[0, 1]` (see module docs).
    pub lambda: f64,
    /// Number of opening plies played uniformly at random, to diversify games
    /// (greedy self-play alone is deterministic and would replay one game).
    pub random_plies: u32,
    /// Threads for game generation (games are independent).
    pub threads: usize,
}

/// Tiny SplitMix64 PRNG — avoids a `rand` dependency (project keeps deps minimal).
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `0..n` (`n > 0`).
    fn below(&mut self, n: u32) -> u32 {
        (self.next_u64() % n as u64) as u32
    }
}

/// Pick a uniformly random legal move from a `get_moves` bitboard (must be nonzero).
fn random_move(moves: u64, rng: &mut Rng) -> u32 {
    let count = moves.count_ones();
    let pick = rng.below(count);
    let mut remaining = moves;
    for _ in 0..pick {
        remaining &= remaining - 1;
    }
    remaining.trailing_zeros()
}

/// Best legal move for `pos` via the fast eval-seeded search (Step 35): exact below
/// the trust frontier, the ordered+TT+incremental heuristic search above. Mirrors the
/// GUI's `score_moves` argmax and reuses the solver's transposition table across the
/// sibling moves (and across the game), so it is far cheaper than the un-ordered
/// negamax `best_move` it replaces. `None` only when there are no legal moves.
fn fast_best_move(
    solver: &mut Solver,
    pos: &Position,
    depth: u32,
    exact_empties: u32,
) -> Option<u32> {
    let mut remaining = pos.get_moves();
    if remaining == 0 {
        return None;
    }
    // First legal move as the default, so the return is always a legal cell.
    let mut best_cell = remaining.trailing_zeros();
    let mut best_score = i32::MIN;
    while remaining != 0 {
        let cell = remaining.trailing_zeros();
        remaining &= remaining - 1;
        let child = pos.do_move(cell);
        // `child` is opponent-to-move; negate to get our score. Solve exactly once the
        // child is within the frontier (also warms the shared TT), else search heuristically.
        let child_score = if child.empties() <= exact_empties {
            solver.exact_score(&child)
        } else {
            solver.heuristic_score(&child, depth.saturating_sub(1))
        };
        if -child_score > best_score {
            best_score = -child_score;
            best_cell = cell;
        }
    }
    Some(best_cell)
}

/// Play one self-play game and return its TD(λ)-labelled training examples
/// (only positions with empties in `(exact_empties, max_empties]`). `solver` carries
/// the current eval ([`Solver::with_eval`]) and is reused across moves and the U(s)
/// bootstrap so its transposition table stays warm.
fn play_one_game(
    solver: &mut Solver,
    config: &SelfPlayConfig,
    rng: &mut Rng,
) -> Vec<TrainingExample> {
    // Walk the game, recording each *decision* state (side to move has a move).
    let mut pos = Position::initial();
    let mut black_to_move = true;
    let mut nodes: Vec<(Position, bool)> = Vec::new();
    let mut ply = 0u32;

    loop {
        if pos.is_game_end() {
            break;
        }
        if !pos.has_moves() {
            pos = pos.pass_move();
            black_to_move = !black_to_move;
            continue;
        }

        nodes.push((pos, black_to_move));

        let mv = if ply < config.random_plies {
            random_move(pos.get_moves(), rng)
        } else {
            // `fast_best_move` solves exactly within the frontier, so the endgame is
            // played (near-)perfectly — strengthening the terminal `z`.
            fast_best_move(solver, &pos, config.depth, config.exact_empties)
                .unwrap_or_else(|| random_move(pos.get_moves(), rng))
        };
        // Both move sources return a *legal* cell drawn from `get_moves()`, so the
        // move always fills a square: empties strictly decrease and the game ends in
        // <= 60 plies. (The top-of-loop `is_game_end`/`has_moves` checks handle the
        // terminal/pass cases.)
        pos = pos.do_move(mv);
        black_to_move = !black_to_move;
        ply += 1;
    }

    let k = nodes.len();
    if k == 0 {
        return Vec::new();
    }

    // Exact terminal disc differential, in Black's fixed perspective.
    let z_stm = pos.final_score() as f64;
    let z_black = if black_to_move { z_stm } else { -z_stm };

    // Black-perspective bootstrap estimate U(s) for each decision state, via the same
    // fast eval-seeded search used for move selection (Step 35 `heuristic_score`).
    let mut u_black: Vec<f64> = Vec::with_capacity(k);
    for &(p, b) in &nodes {
        let v = solver.heuristic_score(&p, config.depth) as f64;
        u_black.push(if b { v } else { -v });
    }

    // Backward forward-view λ-return (Black perspective). g[k] is the terminal.
    let mut g = vec![0f64; k + 1];
    g[k] = z_black;
    for i in (0..k).rev() {
        let u_next = if i + 1 == k { z_black } else { u_black[i + 1] };
        g[i] = (1.0 - config.lambda) * u_next + config.lambda * g[i + 1];
    }

    // Emit labels for the trainable band, converted back to side-to-move sign.
    let mut out = Vec::new();
    for (i, &(p, b)) in nodes.iter().enumerate() {
        let e = p.empties();
        if e > config.exact_empties && e <= config.max_empties {
            let target_stm = if b { g[i] } else { -g[i] };
            out.push(TrainingExample {
                position: p,
                target_score: target_stm.round().clamp(-64.0, 64.0) as i32,
            });
        }
    }
    out
}

/// Generate `games` self-play games and return their pooled TD(λ) examples.
///
/// `base_seed` seeds the per-game PRNGs (game `g` uses `base_seed ^ mix(g)`), so
/// each iteration passes a fresh seed for fresh games. `progress` is bumped once
/// per finished game (a monitor thread reads it for the live count); `interrupted`
/// is polled between games so Ctrl+C stops generation promptly with a partial batch.
pub fn generate_examples(
    weights: &Weights,
    config: &SelfPlayConfig,
    games: usize,
    base_seed: u64,
    progress: &AtomicUsize,
    interrupted: &AtomicBool,
) -> Vec<TrainingExample> {
    // Build the flat eval once and share it (read-only) across the per-thread solvers.
    let flat = Arc::new(FlatEval::from_weights(weights));
    let threads = config.threads.max(1);

    // One reused solver per thread: its transposition table warms across the chunk's
    // games (entries are position-keyed and path-independent, so cross-game reuse is
    // sound and only speeds ordering).
    let play_chunk = |flat: Arc<FlatEval>, lo: usize, hi: usize| -> Vec<TrainingExample> {
        let mut solver = Solver::with_eval(flat);
        let mut out = Vec::new();
        for gi in lo..hi {
            if interrupted.load(Ordering::Relaxed) {
                break;
            }
            let mut rng = Rng(base_seed ^ (gi as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
            out.extend(play_one_game(&mut solver, config, &mut rng));
            progress.fetch_add(1, Ordering::Relaxed);
        }
        out
    };

    if threads <= 1 {
        return play_chunk(flat, 0, games);
    }

    // Split the game indices into `threads` contiguous chunks; each thread owns one.
    let chunk = games.div_ceil(threads);
    std::thread::scope(|s| {
        let play_chunk = &play_chunk;
        let handles: Vec<_> = (0..threads)
            .map(|t| {
                let lo = (t * chunk).min(games);
                let hi = ((t + 1) * chunk).min(games);
                let flat = Arc::clone(&flat);
                s.spawn(move || play_chunk(flat, lo, hi))
            })
            .collect();
        handles
            .into_iter()
            .flat_map(|h| h.join().unwrap_or_default())
            .collect()
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::training::features::Features;

    /// An eval-backed solver over a fresh (zero) weight table — enough to drive the
    /// self-play machinery in tests (label structure does not depend on eval quality).
    fn solver_for(weights: &Weights) -> Solver {
        Solver::with_eval(Arc::new(FlatEval::from_weights(weights)))
    }

    fn cfg() -> SelfPlayConfig {
        SelfPlayConfig {
            depth: 2,
            exact_empties: 12,
            max_empties: 60,
            lambda: 0.7,
            random_plies: 4,
            threads: 1,
        }
    }

    #[test]
    fn random_move_is_legal() {
        let pos = Position::initial();
        let moves = pos.get_moves();
        let mut rng = Rng(12345);
        for _ in 0..50 {
            let mv = random_move(moves, &mut rng);
            assert!(moves & (1u64 << mv) != 0, "illegal move {mv}");
        }
    }

    #[test]
    fn one_game_labels_are_bounded_and_in_band() {
        let weights = Weights::new(Features::edax());
        let mut solver = solver_for(&weights);
        let config = cfg();
        let mut rng = Rng(999);
        let examples = play_one_game(&mut solver, &config, &mut rng);
        assert!(!examples.is_empty(), "a full game should yield examples");
        for ex in &examples {
            let e = ex.position.empties();
            assert!(
                e > config.exact_empties && e <= config.max_empties,
                "empties {e} outside trainable band"
            );
            assert!(
                (-64..=64).contains(&ex.target_score),
                "target {} out of range",
                ex.target_score
            );
        }
    }

    #[test]
    fn lambda_one_labels_match_terminal_outcome() {
        // With λ=1 every state's Black-return is the exact terminal differential,
        // so each label equals ±z (sign = side to move). Verifiable without the
        // recursion: just replay and compare to final_score.
        let weights = Weights::new(Features::edax());
        let mut solver = solver_for(&weights);
        let mut config = cfg();
        config.lambda = 1.0;
        config.random_plies = 0; // deterministic greedy game
        let mut rng = Rng(7);
        let examples = play_one_game(&mut solver, &config, &mut rng);
        // All labels must have equal magnitude (|z|), differing only in sign.
        let mag = examples[0].target_score.abs();
        for ex in &examples {
            assert_eq!(ex.target_score.abs(), mag, "λ=1 label magnitude drifted");
        }
    }

    #[test]
    fn generate_examples_respects_interrupt() {
        let weights = Weights::new(Features::edax());
        let config = cfg();
        let progress = AtomicUsize::new(0);
        let interrupted = AtomicBool::new(true); // already set → no games played
        let out = generate_examples(&weights, &config, 100, 1, &progress, &interrupted);
        assert!(out.is_empty());
        assert_eq!(progress.load(Ordering::Relaxed), 0);
    }
}
