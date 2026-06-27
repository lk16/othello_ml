//! Alloc-free flat pattern evaluation (Step 33).
//!
//! [`Weights`] stores its table as `Vec<Vec<Vec<f32>>>` and
//! [`Weights::evaluate`] allocates a `Vec` per call (via `Features::extract`),
//! so it is unusable on a search hot path. [`FlatEval`] is the alloc-free
//! counterpart, modelled on Edax's `accumlate_eval` (`midgame.c:36`): the whole
//! table is copied **once** into a single contiguous `Vec<f32>`, and a position
//! is scored by a straight-line dot product over the active feature patterns —
//! no per-call allocation, no triple pointer-chase.
//!
//! Layout is range-major: for a given empty-range `r` and feature `f`, the
//! weight for pattern `p` lives at `weights[r * range_stride + offset[f] + p]`,
//! where `offset[f]` is the prefix sum of the per-feature pattern counts. This
//! mirrors Edax's per-ply flat `Eval_weight` block (`eval.h:47`); our 61
//! per-empties buckets play the role of Edax's per-ply tables (`ply = 60 - empties`).
//!
//! Scores are bit-identical to [`Weights::evaluate`] (same f32 values summed in
//! the same feature order) — see the equality test. Quantization to `i16` (the
//! Edax `short` packing) and the incremental make/unmake update
//! (`eval_update`, `eval.c:782`) are deferred to Step 34, where the eval is
//! wired into move ordering / MTD-f and only runs at shallow nodes.

use crate::othello::position::{Cell, Position};
use crate::training::weights::{empty_range_index, Weights};

/// Maximum features we support inline (the Edax set is 47); the scratch index
/// buffer is sized to this so `set` never allocates.
const MAX_FEATURES: usize = 64;

/// A flattened, alloc-free view of a trained [`Weights`] table.
#[derive(Clone)]
pub struct FlatEval {
    /// All weights, range-major: `weights[r * range_stride + offset[f] + p]`.
    weights: Vec<f32>,
    /// Per-feature base offset within one range block (prefix sum of `len`).
    offset: Vec<u32>,
    /// Cells of each feature, for building the pattern index from a position.
    cells: Vec<Vec<u8>>,
    /// Patterns per range block (`sum_f 3^cells[f]`).
    range_stride: usize,
    /// Number of features (47 for the Edax set).
    n_features: usize,

    // ─── Incremental-eval support (Edax `eval_update`, Step 35 fix 2) ───────
    //
    /// `weights` re-expressed in the **fixed global perspective** the incremental
    /// state uses (digit 0 = the player to move at *even* empties). Identical to
    /// `weights` for even-empties buckets; for odd buckets each shape's block is
    /// remapped by the player↔opponent digit swap (see `from_weights`). Lets the
    /// O(flips) [`IncEval`] update feed our side-to-move-trained weights without
    /// re-deriving the side-to-move index each ply. See [`IncEval`].
    weights_fixed: Vec<f32>,
    /// Per square (0..64): the `(feature, power = 3^position)` pairs of every
    /// feature containing it — the scatter [`IncEval::child`] adds/subtracts on a
    /// move. Edax's `EVAL_X2F` (`eval.c`).
    x2f: Vec<Vec<(u32, u16)>>,
}

/// Incremental evaluation state: one trinary pattern index per feature, in the
/// fixed global perspective of [`FlatEval::weights_fixed`], plus the empties count.
///
/// Maintained à la Edax `Eval` (`eval.c`): built once at a search root
/// ([`FlatEval::inc_root`]) and advanced per move by touching only the moved and
/// flipped squares ([`FlatEval::inc_child`]) — O(flips), not a from-scratch scan of
/// all features. [`FlatEval::inc_score`] dot-products it against `weights_fixed`.
///
/// The fixed perspective (vs the per-leaf side-to-move scan of [`FlatEval::set`]) is
/// what makes the update cheap: only the changed cells move, never the whole board
/// across the per-ply side flip. Scores are *not* per-leaf symmetry-normalized; the
/// search canonicalizes its root instead (symmetric roots share a canonical form, so
/// the whole tree — and the score — matches).
#[derive(Clone)]
pub struct IncEval {
    idx: [u16; MAX_FEATURES],
    empties: u32,
}

impl FlatEval {
    /// Flatten a trained [`Weights`] table into the contiguous layout.
    ///
    /// Copies every weight verbatim (f32, lossless), so scores match
    /// [`Weights::evaluate`] exactly.
    pub fn from_weights(weights: &Weights) -> Self {
        let features = weights.features();
        let n_features = features.count();
        assert!(
            n_features <= MAX_FEATURES,
            "FlatEval supports at most {MAX_FEATURES} features, got {n_features}"
        );

        // Weights are tied per symmetry shape; the flat block holds one slot range
        // per *shape*, and every feature points (via `offset`) at its shape's block.
        // Symmetric features therefore share weights, exactly as in `Weights`.
        let table = weights.shape_weights();
        let feature_to_shape = weights.feature_to_shape();
        let n_shapes = table.len();

        // Offset of each shape's block within one range, = prefix sum of shape sizes.
        let mut shape_offset = vec![0u32; n_shapes];
        let mut acc = 0u32;
        for (s, shape_weights) in table.iter().enumerate() {
            shape_offset[s] = acc;
            acc += shape_weights[0].len() as u32;
        }
        let range_stride = acc as usize;

        // Per-feature: offset = its shape's block; cells for index extraction.
        let mut offset = Vec::with_capacity(n_features);
        let mut cells = Vec::with_capacity(n_features);
        for (f, feature) in features.all().iter().enumerate() {
            offset.push(shape_offset[feature_to_shape[f]]);
            cells.push(feature.cells.iter().map(|&c| c as u8).collect());
        }

        let n_ranges = weights.empty_range_count();
        let mut flat = vec![0.0f32; n_ranges * range_stride];
        for (s, shape_weights) in table.iter().enumerate() {
            for (r, range_weights) in shape_weights.iter().enumerate() {
                let base = r * range_stride + shape_offset[s] as usize;
                flat[base..base + range_weights.len()].copy_from_slice(range_weights);
            }
        }

        // Fixed-perspective weights: even buckets verbatim, odd buckets with each
        // shape's pattern weights remapped by the player↔opponent digit swap, so the
        // O(flips) `IncEval` (which stores indices in the fixed encoding) reproduces
        // the side-to-move score (see `IncEval`). The swap is a per-cell-count
        // permutation `weights_fixed[swap(v)] = weights[v]`.
        let mut flat_fixed = flat.clone();
        for (s, shape_weights) in table.iter().enumerate() {
            let n_cells = cells_from_pattern_count(shape_weights[0].len());
            for (r, range_weights) in shape_weights.iter().enumerate() {
                if r % 2 == 0 {
                    continue; // even bucket: identical to `flat`
                }
                let base = r * range_stride + shape_offset[s] as usize;
                for (v, &w) in range_weights.iter().enumerate() {
                    flat_fixed[base + swap_index(v as u16, n_cells) as usize] = w;
                }
            }
        }

        // Coordinate → features scatter: for each square, the (feature, 3^position)
        // pairs of every feature that contains it.
        let mut x2f: Vec<Vec<(u32, u16)>> = vec![Vec::new(); 64];
        for (f, feature_cells) in cells.iter().enumerate() {
            let mut pow = 1u16;
            for &cell in feature_cells {
                x2f[cell as usize].push((f as u32, pow));
                pow = pow.saturating_mul(3);
            }
        }

        FlatEval {
            weights: flat,
            offset,
            cells,
            range_stride,
            n_features,
            weights_fixed: flat_fixed,
            x2f,
        }
    }

    /// Number of features (length of the index buffer `set` fills).
    pub fn n_features(&self) -> usize {
        self.n_features
    }

    /// Fill `out[0..n_features]` with this position's feature-pattern indices.
    ///
    /// The position is normalized to its symmetry [`canonical`](Position::canonical)
    /// form first, so the resulting score is invariant across all 8 board symmetries
    /// — matching [`Weights::evaluate`], which does the same. (The learned eval is not
    /// symmetry-invariant otherwise; see [`Position::canonical`].)
    ///
    /// Alloc-free. `out` must have length `>= n_features`. The index of feature
    /// `f` is `sum_c value(cell_c) * 3^c`, with `value` = 0 empty / 1 player /
    /// 2 opponent — identical to `Feature::extract_index`.
    #[inline]
    pub fn set(&self, pos: &Position, out: &mut [u16]) {
        let pos = &pos.canonical();
        for (f, feature_cells) in self.cells.iter().enumerate() {
            let mut idx = 0u16;
            let mut pow = 1u16;
            for &cell in feature_cells {
                let value = match pos.get_cell(cell as u32) {
                    Cell::Empty => 0,
                    Cell::Player => 1,
                    Cell::Opponent => 2,
                };
                idx += value * pow;
                pow *= 3;
            }
            out[f] = idx;
        }
    }

    /// Straight-line dot product over precomputed feature-pattern `indices`.
    ///
    /// Alloc-free. `indices[0..n_features]` are pattern indices (e.g. from
    /// [`FlatEval::set`] or maintained incrementally). `empties` selects the
    /// range block.
    #[inline]
    pub fn score(&self, indices: &[u16], empties: u32) -> f32 {
        let base = empty_range_index(empties) * self.range_stride;
        let mut sum = 0.0f32;
        for (&off, &pattern) in self.offset.iter().zip(&indices[..self.n_features]) {
            sum += self.weights[base + off as usize + pattern as usize];
        }
        sum
    }

    /// Evaluate a position alloc-free (full `set` + `score`).
    ///
    /// Bit-identical to [`Weights::evaluate`]. Suitable for shallow ordering /
    /// MTD nodes where a from-scratch eval is affordable.
    #[inline]
    pub fn eval_position(&self, pos: &Position) -> f32 {
        let mut indices = [0u16; MAX_FEATURES];
        self.set(pos, &mut indices);
        self.score(&indices, pos.empties())
    }

    // ─── Incremental eval (Edax `eval_update`, Step 35 fix 2) ──────────────

    /// Build the incremental state for a position from scratch (Edax `eval_set`).
    ///
    /// Indices are in the fixed global perspective: digit 1 = the disc of the player
    /// to move at *even* empties (`A`), digit 2 = the other player (`B`), 0 = empty.
    /// So at even empties this equals the side-to-move index of [`FlatEval::set`]
    /// (un-canonicalized); at odd empties it is its player↔opponent swap — which
    /// [`FlatEval::weights_fixed`] undoes at score time.
    pub fn inc_root(&self, pos: &Position) -> IncEval {
        let empties = pos.empties();
        // `A` = the player to move when empties is even.
        let (a_board, b_board) = if empties % 2 == 0 {
            (pos.player, pos.opponent)
        } else {
            (pos.opponent, pos.player)
        };
        let mut idx = [0u16; MAX_FEATURES];
        for (f, feature_cells) in self.cells.iter().enumerate() {
            let mut v = 0u16;
            let mut pow = 1u16;
            for &cell in feature_cells {
                let bit = 1u64 << cell;
                let digit = if a_board & bit != 0 {
                    1
                } else if b_board & bit != 0 {
                    2
                } else {
                    0
                };
                v += digit * pow;
                pow = pow.saturating_mul(3);
            }
            idx[f] = v;
        }
        IncEval { idx, empties }
    }

    /// Advance the state by one move (Edax `eval_update`): play `cell`, flipping the
    /// `flipped` mask, returning the child's state. Touches only the moved and
    /// flipped squares — O(flips). `parent` is the state of the position to move.
    ///
    /// In the fixed encoding the mover at even empties is `A`, at odd empties `B`, so
    /// the per-cell digit deltas are: `A` moves → move +1, flips −1 (B→A); `B` moves
    /// → move +2, flips +1 (A→B). `pow` carries the `3^position` weight per feature.
    pub fn inc_child(&self, parent: &IncEval, cell: u32, flipped: u64) -> IncEval {
        let mut idx = parent.idx;
        let a_moves = parent.empties % 2 == 0;
        // Moved square: empty (0) → mover's disc.
        for &(f, pow) in &self.x2f[cell as usize] {
            let d = if a_moves { pow } else { 2 * pow };
            idx[f as usize] = idx[f as usize].wrapping_add(d);
        }
        // Flipped squares change owner.
        let mut rem = flipped;
        while rem != 0 {
            let c = rem.trailing_zeros() as usize;
            rem &= rem - 1;
            for &(f, pow) in &self.x2f[c] {
                idx[f as usize] = if a_moves {
                    idx[f as usize].wrapping_sub(pow) // B(2) → A(1)
                } else {
                    idx[f as usize].wrapping_add(pow) // A(1) → B(2)
                };
            }
        }
        IncEval {
            idx,
            empties: parent.empties - 1,
        }
    }

    /// Score an incremental state (side-to-move perspective, matching an
    /// un-canonicalized [`FlatEval::set`] + [`FlatEval::score`]). Dot-products the
    /// fixed-perspective indices against [`FlatEval::weights_fixed`].
    #[inline]
    pub fn inc_score(&self, e: &IncEval) -> f32 {
        let base = empty_range_index(e.empties) * self.range_stride;
        let mut sum = 0.0f32;
        for (&off, &pattern) in self.offset.iter().zip(&e.idx[..self.n_features]) {
            sum += self.weights_fixed[base + off as usize + pattern as usize];
        }
        sum
    }
}

/// Number of cells of a shape whose pattern table has `count = 3^cells` entries.
fn cells_from_pattern_count(count: usize) -> u32 {
    let mut n = 0u32;
    let mut c = 1usize;
    while c < count {
        c *= 3;
        n += 1;
    }
    debug_assert_eq!(3usize.pow(n), count, "pattern count must be a power of 3");
    n
}

/// The player↔opponent digit swap of a trinary pattern `v` over `n_cells` cells:
/// each digit 1↔2, 0 fixed. An involution; used to remap odd-empties weight buckets
/// into the fixed perspective (see [`FlatEval::weights_fixed`]).
fn swap_index(v: u16, n_cells: u32) -> u16 {
    let mut out = 0u16;
    let mut pow = 1u16;
    let mut rest = v;
    for _ in 0..n_cells {
        let digit = rest % 3;
        rest /= 3;
        let swapped = match digit {
            1 => 2,
            2 => 1,
            other => other,
        };
        out += swapped * pow;
        pow = pow.saturating_mul(3);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::training::features::Features;

    /// xorshift32 → f32 in roughly [-50, 50], for filling weights deterministically.
    fn rnd(state: &mut u32) -> f32 {
        let mut x = *state;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        *state = x;
        ((x % 10_000) as f32) / 100.0 - 50.0
    }

    /// Generate a spread of positions by walking random legal moves from the start.
    fn sample_positions() -> Vec<Position> {
        let mut positions = vec![Position::initial()];
        let mut state = 0x1234_5678u32;
        let mut pos = Position::initial();
        for _ in 0..60 {
            let moves = pos.get_moves();
            if moves == 0 {
                pos = pos.pass_move();
                if pos.get_moves() == 0 {
                    break;
                }
                continue;
            }
            // pick a pseudo-random legal move
            let n = moves.count_ones();
            let mut pick = (rnd(&mut state).abs() as u32) % n;
            let mut m = moves;
            let cell = loop {
                let cell = m.trailing_zeros();
                m &= m - 1;
                if pick == 0 {
                    break cell;
                }
                pick -= 1;
            };
            pos = pos.do_move(cell);
            positions.push(pos);
        }
        positions
    }

    #[test]
    fn flat_eval_matches_weights_evaluate() {
        let features = Features::edax();
        let mut weights = Weights::new(features.clone());

        let positions = sample_positions();

        // Assign a distinct nonzero weight to every (feature, pattern, empties)
        // slot that the sample positions actually touch, so a mis-indexed flat
        // lookup would surface as a wrong sum.
        let mut state = 0x9E37_79B9u32;
        for pos in &positions {
            let empties = pos.empties();
            let indices = features.extract(pos);
            for (f, &p) in indices.iter().enumerate() {
                weights.set_weight(f, p, empties, rnd(&mut state));
            }
        }

        let flat = FlatEval::from_weights(&weights);

        for pos in &positions {
            let expected = weights.evaluate(pos, &features);
            let got = flat.eval_position(pos);
            assert_eq!(
                got.to_bits(),
                expected.to_bits(),
                "mismatch at empties={}: flat={got} weights={expected}",
                pos.empties()
            );
        }
    }

    /// Regression: `FlatEval` must score all 8 board symmetries identically, the
    /// same way [`Weights::evaluate`] does (both normalize to the canonical form).
    /// Mirrors `weights::tests::evaluate_is_symmetry_invariant`; guards the GUI bug
    /// where the 4 equivalent opening moves scored 2,1,1,1.
    #[test]
    fn flat_eval_is_symmetry_invariant() {
        let features = Features::edax();
        let mut weights = Weights::new(features.clone());

        // Distinct nonzero weights on every slot touched by the samples and their
        // symmetries, so a non-normalized eval would diverge across orientations.
        let mut state = 0x9E37_79B9u32;
        let positions = sample_positions();
        for pos in &positions {
            for sym in pos.symmetries() {
                let empties = sym.empties();
                for (f, &p) in features.extract(&sym).iter().enumerate() {
                    weights.set_weight(f, p, empties, rnd(&mut state));
                }
            }
        }

        let flat = FlatEval::from_weights(&weights);
        for pos in &positions {
            let base = flat.eval_position(pos);
            for (k, sym) in pos.symmetries().iter().enumerate() {
                assert_eq!(
                    flat.eval_position(sym).to_bits(),
                    base.to_bits(),
                    "symmetry {k} differs at empties {}",
                    pos.empties()
                );
            }
        }
    }

    #[test]
    fn set_then_score_matches_eval_position() {
        let features = Features::edax();
        let mut weights = Weights::new(features.clone());
        let mut state = 0x0BAD_F00Du32;
        let pos = Position::initial().do_move(19); // some opening move
        let indices = features.extract(&pos);
        for (f, &p) in indices.iter().enumerate() {
            weights.set_weight(f, p, pos.empties(), rnd(&mut state));
        }
        let flat = FlatEval::from_weights(&weights);

        let mut buf = vec![0u16; flat.n_features()];
        flat.set(&pos, &mut buf);
        let via_parts = flat.score(&buf, pos.empties());
        let via_full = flat.eval_position(&pos);
        assert_eq!(via_parts.to_bits(), via_full.to_bits());
    }

    /// Un-canonicalized side-to-move eval (the reference the incremental state must
    /// reproduce): the raw `FlatEval::set`+`score` *without* the symmetry
    /// canonicalization, summed in feature order.
    fn raw_score(flat: &FlatEval, pos: &Position) -> f32 {
        let base = empty_range_index(pos.empties()) * flat.range_stride;
        let mut sum = 0.0f32;
        for (f, feature_cells) in flat.cells.iter().enumerate() {
            let mut idx = 0u16;
            let mut pow = 1u16;
            for &cell in feature_cells {
                let v = match pos.get_cell(cell as u32) {
                    Cell::Empty => 0,
                    Cell::Player => 1,
                    Cell::Opponent => 2,
                };
                idx += v * pow;
                pow = pow.saturating_mul(3);
            }
            sum += flat.weights[base + flat.offset[f] as usize + idx as usize];
        }
        sum
    }

    /// Distinct nonzero weights on every slot the sample positions touch, so the
    /// fixed-perspective remap and the update deltas are actually exercised.
    fn flat_with_random_weights(positions: &[Position]) -> FlatEval {
        let features = Features::edax();
        let mut weights = Weights::new(features.clone());
        let mut state = 0x9E37_79B9u32;
        for pos in positions {
            let empties = pos.empties();
            for (f, &p) in features.extract(pos).iter().enumerate() {
                weights.set_weight(f, p, empties, rnd(&mut state));
            }
        }
        FlatEval::from_weights(&weights)
    }

    /// `inc_root` + `inc_score` reproduces the un-canonicalized side-to-move eval for
    /// any position (both empties parities, via the fixed-perspective weight remap).
    #[test]
    fn inc_root_matches_raw() {
        let positions = sample_positions();
        let flat = flat_with_random_weights(&positions);
        for pos in &positions {
            assert_eq!(
                flat.inc_score(&flat.inc_root(pos)).to_bits(),
                raw_score(&flat, pos).to_bits(),
                "empties {}",
                pos.empties()
            );
        }
    }

    /// Full-window raw-leaf negamax reference (no ordering/TT/PVS): the value the
    /// incremental heuristic search must reproduce. Horizon clamps to `[-63, 63]`, a
    /// pass counts as a ply — matching `Search::id_pass_inc`. Assumes empties > depth
    /// (no exact handoff), so callers keep depth small from the opening.
    fn raw_negamax(flat: &FlatEval, pos: &Position, depth: u32, mut alpha: i32, beta: i32) -> i32 {
        let moves = pos.get_moves();
        if moves == 0 {
            let passed = pos.pass_move();
            if passed.get_moves() == 0 {
                return pos.final_score();
            }
            return -raw_negamax(flat, &passed, depth.saturating_sub(1), -beta, -alpha);
        }
        if depth == 0 {
            return raw_score(flat, pos).round().clamp(-63.0, 63.0) as i32;
        }
        let mut rem = moves;
        while rem != 0 {
            let cell = rem.trailing_zeros();
            rem &= rem - 1;
            let score = -raw_negamax(flat, &pos.do_move(cell), depth - 1, -beta, -alpha);
            if score > alpha {
                alpha = score;
                if alpha >= beta {
                    break;
                }
            }
        }
        alpha
    }

    /// End-to-end: the incremental eval-seeded ordered search (`heuristic_score`)
    /// reproduces the independent raw-leaf negamax (root-canonicalized) — validating
    /// the `inc`-threaded PVS/TT/ordering against a from-scratch reference.
    #[test]
    fn heuristic_score_matches_raw_negamax() {
        use crate::Solver;
        use std::sync::Arc;

        let positions = sample_positions();
        // Opening positions only (empties > depth, so no exact handoff in the search).
        let openings: Vec<&Position> = positions.iter().filter(|p| p.empties() > 50).collect();
        let flat = flat_with_random_weights(&positions);
        let eval = Arc::new(flat.clone());

        for p in &openings {
            for d in 1..=4u32 {
                let canon = p.canonical();
                let expected = raw_negamax(&flat, &canon, d, -64, 64);
                let mut solver = Solver::with_eval(Arc::clone(&eval));
                assert_eq!(
                    solver.heuristic_score(p, d),
                    expected,
                    "depth {d}, empties {}",
                    p.empties()
                );
            }
        }
    }

    /// The O(flips) `inc_child` update, chained across a whole game, stays bit-exact
    /// vs a from-scratch `inc_root` — and rebuilds correctly across passes.
    #[test]
    fn inc_chain_matches_from_scratch() {
        let positions = sample_positions();
        let flat = flat_with_random_weights(&positions);

        let mut inc = flat.inc_root(&positions[0]);
        for i in 0..positions.len() {
            let pos = &positions[i];
            // Maintained state equals a fresh build at every node.
            assert_eq!(inc.idx, flat.inc_root(pos).idx, "idx drift at step {i}");
            assert_eq!(
                flat.inc_score(&inc).to_bits(),
                raw_score(&flat, pos).to_bits(),
                "score drift at step {i}"
            );
            let Some(next) = positions.get(i + 1) else {
                break;
            };
            // Find the move linking pos → next; absence means a pass (rebuild).
            let mut moved = false;
            let mut rem = pos.get_moves();
            while rem != 0 {
                let cell = rem.trailing_zeros();
                rem &= rem - 1;
                if &pos.do_move(cell) == next {
                    inc = flat.inc_child(&inc, cell, pos.flipped(cell));
                    moved = true;
                    break;
                }
            }
            if !moved {
                inc = flat.inc_root(next); // pass: empties unchanged, rebuild
            }
        }
    }
}
