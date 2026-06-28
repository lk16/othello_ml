//! Depth-limited heuristic search for gameplay (distinct from the exact endgame
//! search): negamax to a fixed depth, applying the learned evaluation at the
//! leaves, plus exact search once the position is shallow enough.

use super::search::Search;
use super::{SCORE_MAX, SCORE_MIN};
use crate::eval::pattern::FlatEval;
use crate::othello::position::Position;
use crate::training::features::Features;
use crate::training::weights::Weights;

/// Depth-limited evaluation for gameplay: search `depth` plies, heuristic leaves.
pub fn depth_limited_score(
    pos: &Position,
    depth: u32,
    weights: &Weights,
    features: &Features,
) -> i32 {
    alphabeta(pos, depth, weights, features, SCORE_MIN, SCORE_MAX)
}

/// Bootstrapped training label for `pos`: the backed-up score of a depth-`depth`
/// negamax whose horizon leaves are scored by the alloc-free trained pattern eval
/// [`FlatEval`] (Step 33). Side-to-move perspective, in `[-64, 64]`.
///
/// Used by `train-boot` to label positions with empties > N, where exact search
/// is infeasible. The eval is exact-trained at empties ≤ N, so a shallow search
/// from a position just above N bottoms out in that well-trained band — anchoring
/// the label to (near-)ground-truth. Identical in structure to [`alphabeta`], but
/// with `FlatEval::eval_position` at the leaves instead of [`heuristic`].
pub fn bootstrap_score(pos: &Position, flat: &FlatEval, depth: u32) -> i32 {
    boot_ab(pos, depth, flat, SCORE_MIN, SCORE_MAX)
}

/// Negamax with alpha-beta and a depth limit, [`FlatEval`] at the horizon.
fn boot_ab(pos: &Position, depth: u32, flat: &FlatEval, mut alpha: i32, beta: i32) -> i32 {
    let moves = pos.get_moves();
    if moves == 0 {
        let passed = pos.pass_move();
        if passed.get_moves() == 0 {
            return pos.final_score();
        }
        return -boot_ab(&passed, depth, flat, -beta, -alpha);
    }

    if depth == 0 {
        return flat
            .eval_position(pos)
            .round()
            .clamp(SCORE_MIN as f32, SCORE_MAX as f32) as i32;
    }

    let mut remaining = moves;
    while remaining != 0 {
        let cell = remaining.trailing_zeros();
        remaining &= remaining - 1;
        let child = pos.do_move(cell);
        let score = -boot_ab(&child, depth - 1, flat, -beta, -alpha);
        if score > alpha {
            alpha = score;
            if alpha >= beta {
                break;
            }
        }
    }
    alpha
}

/// Best legal move for the side to move, or `None` when there are no moves.
/// Switches to exact search at or below `exact_empties`.
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
    // Default to the first legal move: if every move scores <= the initial alpha
    // (e.g. a clamped boundary eval), best_cell must still be a *legal* cell, never
    // a stale 0 — a `do_move` on an illegal cell is a no-op and would loop callers.
    let mut best_cell = moves.trailing_zeros();
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
        // A proven maximal win can't be beaten — stop. This also keeps the next
        // child's window `(-SCORE_MAX, -alpha)` non-degenerate: at alpha == SCORE_MAX
        // it would collapse to `alpha == beta`, which the exact leaf solvers
        // (`solve_2..solve_4`) mishandle as an infinite pass loop.
        if alpha >= SCORE_MAX {
            break;
        }
    }

    Some(best_cell)
}

/// Best legal move using exact search to game end.
fn best_move_exact(pos: &Position) -> Option<u32> {
    let moves = pos.get_moves();
    if moves == 0 {
        return None;
    }

    let mut alpha = SCORE_MIN;
    // First legal move as the default (see `best_move`): guarantees a legal return
    // even when no child improves on the initial alpha.
    let mut best_cell = moves.trailing_zeros();
    let empties = pos.empties();
    let mut searcher = Search::new();

    let mut remaining = moves;
    while remaining != 0 {
        let cell = remaining.trailing_zeros();
        remaining &= remaining - 1;
        let child = pos.do_move(cell);
        searcher.parity = super::search::board_parity(&child);
        let score = -searcher.search_exact(&child, -SCORE_MAX, -alpha, empties - 1, None);
        if score > alpha {
            alpha = score;
            best_cell = cell;
        }
        // Stop at a proven maximal win, and keep the next child's window
        // `(-SCORE_MAX, -alpha)` non-degenerate (see `best_move`): a collapsed
        // `alpha == beta` window drives `solve_2..solve_4` into an infinite pass loop.
        if alpha >= SCORE_MAX {
            break;
        }
    }

    Some(best_cell)
}

/// Negamax with alpha-beta pruning and a depth limit.
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

pub(super) fn heuristic(pos: &Position, weights: &Weights, features: &Features) -> i32 {
    let score = weights.evaluate(pos, features);
    score.round().clamp(SCORE_MIN as f32, SCORE_MAX as f32) as i32
}

#[cfg(test)]
mod tests {
    use super::*;

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

    /// Regression for the GUI `evaluate` bug: the four opening moves are board
    /// symmetries of one another, so a depth-limited search must score them
    /// identically. They previously showed 2,1,1,1 because the learned eval was not
    /// symmetry-invariant (fixed by normalizing to the canonical form in `evaluate`).
    #[test]
    fn opening_moves_score_equally() {
        let features = Features::edax();
        let mut weights = Weights::new(features.clone());

        // Non-trivial, position-dependent weights, filled over the symmetries of a
        // random game so a non-normalized eval would make the four moves diverge.
        let mut s = 0x9E37_79B9u32;
        let rnd = |s: &mut u32| {
            *s ^= *s << 13;
            *s ^= *s >> 17;
            *s ^= *s << 5;
            ((*s % 10_000) as f32) / 100.0 - 50.0
        };
        let mut pos = Position::initial();
        for _ in 0..50 {
            for sym in pos.symmetries() {
                let empties = sym.empties();
                for (f, &p) in features.extract(&sym).iter().enumerate() {
                    weights.set_weight(f, p, empties, rnd(&mut s));
                }
            }
            let moves = pos.get_moves();
            if moves == 0 {
                pos = pos.pass_move();
                if pos.get_moves() == 0 {
                    break;
                }
                continue;
            }
            pos = pos.do_move(moves.trailing_zeros());
        }

        let start = Position::initial();
        let mut scores = Vec::new();
        let mut remaining = start.get_moves();
        while remaining != 0 {
            let cell = remaining.trailing_zeros();
            remaining &= remaining - 1;
            let child = start.do_move(cell);
            scores.push(-depth_limited_score(&child, 4, &weights, &features));
        }
        assert_eq!(scores.len(), 4);
        assert!(
            scores.iter().all(|&s| s == scores[0]),
            "opening moves scored differently: {scores:?}"
        );
    }

    /// `heuristic_score` must keep the symmetry-invariance fix: the four equivalent
    /// opening moves score equally through the fast search too.
    #[test]
    fn heuristic_score_opening_moves_equal() {
        use crate::{FlatEval, Solver};
        use std::sync::Arc;

        let features = Features::edax();
        let mut weights = Weights::new(features.clone());
        let mut s = 0x9E37_79B9u32;
        let rnd = |s: &mut u32| {
            *s ^= *s << 13;
            *s ^= *s >> 17;
            *s ^= *s << 5;
            ((*s % 10_000) as f32) / 100.0 - 50.0
        };
        let mut pos = Position::initial();
        for _ in 0..40 {
            for sym in pos.symmetries() {
                let e = sym.empties();
                for (f, &p) in features.extract(&sym).iter().enumerate() {
                    weights.set_weight(f, p, e, rnd(&mut s));
                }
            }
            let moves = pos.get_moves();
            if moves == 0 {
                break;
            }
            pos = pos.do_move(moves.trailing_zeros());
        }

        let eval = Arc::new(FlatEval::from_weights(&weights));
        let start = Position::initial();
        let mut scores = Vec::new();
        let mut remaining = start.get_moves();
        while remaining != 0 {
            let cell = remaining.trailing_zeros();
            remaining &= remaining - 1;
            let mut solver = Solver::with_eval(Arc::clone(&eval));
            scores.push(-solver.heuristic_score(&start.do_move(cell), 4));
        }
        assert_eq!(scores.len(), 4);
        assert!(
            scores.iter().all(|&s| s == scores[0]),
            "opening moves diverged: {scores:?}"
        );
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
    fn test_bootstrap_score_game_end_is_exact() {
        // At game end, the search returns the true final score regardless of eval.
        let pos = Position {
            player: u64::MAX,
            opponent: 0,
        };
        let weights = Weights::new(Features::edax());
        let flat = crate::eval::pattern::FlatEval::from_weights(&weights);
        assert_eq!(bootstrap_score(&pos, &flat, 4), 64);
    }

    #[test]
    fn test_bootstrap_score_depth0_is_leaf_eval() {
        // With depth 0 the score is just the clamped leaf eval of the position.
        let pos = Position::initial();
        let weights = Weights::new(Features::edax());
        let flat = crate::eval::pattern::FlatEval::from_weights(&weights);
        let expected = flat
            .eval_position(&pos)
            .round()
            .clamp(SCORE_MIN as f32, SCORE_MAX as f32) as i32;
        assert_eq!(bootstrap_score(&pos, &flat, 0), expected);
    }

    #[test]
    fn test_bootstrap_score_bounded() {
        let pos = Position::initial();
        let weights = Weights::new(Features::edax());
        let flat = crate::eval::pattern::FlatEval::from_weights(&weights);
        let s = bootstrap_score(&pos, &flat, 4);
        assert!(
            (SCORE_MIN..=SCORE_MAX).contains(&s),
            "score {s} out of bounds"
        );
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
}
