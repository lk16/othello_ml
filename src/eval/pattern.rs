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

        FlatEval {
            weights: flat,
            offset,
            cells,
            range_stride,
            n_features,
        }
    }

    /// Number of features (length of the index buffer `set` fills).
    pub fn n_features(&self) -> usize {
        self.n_features
    }

    /// Fill `out[0..n_features]` with this position's feature-pattern indices.
    ///
    /// Alloc-free. `out` must have length `>= n_features`. The index of feature
    /// `f` is `sum_c value(cell_c) * 3^c`, with `value` = 0 empty / 1 player /
    /// 2 opponent — identical to `Feature::extract_index`.
    #[inline]
    pub fn set(&self, pos: &Position, out: &mut [u16]) {
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
}
