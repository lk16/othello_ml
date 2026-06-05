use crate::features::Features;

/// Weight table for evaluating Othello positions.
///
/// Stores evaluation weights indexed by:
/// - Feature index: which of the 47 features (0-46)
/// - Empty count range: which of the 30 disc-count tables (indices 0-29 for empties 2,4,6,...,60)
/// - Pattern index: which pattern configuration within that feature (0 to 3^cells-1)
///
/// Weights are stored as f32 internally for precise SGD updates.
/// They are rounded to i16 only at serialization time.
///
/// Position evaluation = sum of all feature weights for the current board state.
pub struct Weights {
    // Feature scores: [feature][empty_range][pattern] = score (f32 for SGD precision)
    feature_weights: Vec<Vec<Vec<f32>>>,
    empty_ranges: Vec<u32>, // 2, 4, 6, 8, ... 60
    features: Features,
}

impl Weights {
    /// Create a new weight table with Edax features and SGD training
    pub fn new(features: Features) -> Self {
        let n_features = features.count();
        let n_empty_ranges = 30; // empties: 2, 4, 6, ..., 60

        // Initialize with empty ranges: 2, 4, 6, ..., 60
        let mut empty_ranges = Vec::with_capacity(n_empty_ranges);
        for i in 1..=30 {
            empty_ranges.push(i * 2);
        }

        // Pre-allocate weight tables
        let mut feature_weights = Vec::with_capacity(n_features);
        for feature in features.all() {
            let max_pattern = (feature.max_index() + 1) as usize;
            let mut feature_scores = Vec::with_capacity(n_empty_ranges);

            for _ in 0..n_empty_ranges {
                // Initialize all weights to 0.0
                feature_scores.push(vec![0.0f32; max_pattern]);
            }

            feature_weights.push(feature_scores);
        }

        Weights {
            feature_weights,
            empty_ranges,
            features,
        }
    }

    /// Get the appropriate empty range index for a given number of empties
    /// Rounds down to nearest even number
    fn empty_range_index(&self, empties: u32) -> usize {
        let clamped = empties.clamp(2, 60);

        // Round down to nearest even, then convert to index (0-29)
        let even = (clamped / 2) * 2;
        ((even / 2) - 1) as usize
    }

    /// Evaluate a board position by summing contributions from all features
    pub fn evaluate(&self, board: &crate::board::Board, features: &Features) -> f32 {
        let empties = board.empties();
        let range_idx = self.empty_range_index(empties);
        let feature_indices = features.extract(board);

        let mut score = 0.0f32;
        for (feat_idx, &pattern_idx) in feature_indices.iter().enumerate() {
            if feat_idx < self.feature_weights.len() {
                let pattern_idx = pattern_idx as usize;
                if pattern_idx < self.feature_weights[feat_idx][range_idx].len() {
                    score += self.feature_weights[feat_idx][range_idx][pattern_idx];
                }
            }
        }

        score
    }

    /// Get weight for a specific feature, pattern, and empty range
    pub fn get_weight(&self, feature_idx: usize, pattern_idx: u32, empties: u32) -> f32 {
        if feature_idx >= self.feature_weights.len() {
            return 0.0;
        }

        let range_idx = self.empty_range_index(empties);
        let pattern_idx = pattern_idx as usize;

        if pattern_idx < self.feature_weights[feature_idx][range_idx].len() {
            self.feature_weights[feature_idx][range_idx][pattern_idx]
        } else {
            0.0
        }
    }

    /// Set weight for a specific feature, pattern, and empty range
    pub fn set_weight(&mut self, feature_idx: usize, pattern_idx: u32, empties: u32, weight: f32) {
        if feature_idx >= self.feature_weights.len() {
            return;
        }

        let range_idx = self.empty_range_index(empties);
        let pattern_idx = pattern_idx as usize;

        if pattern_idx < self.feature_weights[feature_idx][range_idx].len() {
            self.feature_weights[feature_idx][range_idx][pattern_idx] = weight;
        }
    }

    /// Update weight using SGD
    /// weight += learning_rate * gradient
    /// Weights are clipped to [-MAX_WEIGHT, MAX_WEIGHT] to prevent explosion.
    pub fn update_weight_sgd(
        &mut self,
        feature_idx: usize,
        pattern_idx: u32,
        empties: u32,
        learning_rate: f32,
        gradient: f32,
    ) {
        const MAX_WEIGHT: f32 = 100.0;
        let current = self.get_weight(feature_idx, pattern_idx, empties);
        let new_weight = (current + learning_rate * gradient).clamp(-MAX_WEIGHT, MAX_WEIGHT);
        self.set_weight(feature_idx, pattern_idx, empties, new_weight);
    }

    /// Get the features struct
    pub fn features(&self) -> &Features {
        &self.features
    }

    /// Get feature count
    pub fn feature_count(&self) -> usize {
        self.feature_weights.len()
    }

    /// Get number of empty ranges
    pub fn empty_range_count(&self) -> usize {
        self.empty_ranges.len()
    }

    /// Get all weights as f32 (for serialization: round to i16 when writing)
    pub fn get_all_weights(&self) -> &Vec<Vec<Vec<f32>>> {
        &self.feature_weights
    }

    /// Set all weights from f32 (for deserialization)
    pub fn set_all_weights(&mut self, weights: Vec<Vec<Vec<f32>>>) {
        self.feature_weights = weights;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_weights_creation() {
        let features = Features::edax();
        let weights = Weights::new(features);
        assert!(weights.feature_count() > 0);
        assert_eq!(weights.empty_range_count(), 30);
    }

    #[test]
    fn test_empty_range_index() {
        let features = Features::edax();
        let weights = Weights::new(features);

        assert_eq!(weights.empty_range_index(2), 0);
        assert_eq!(weights.empty_range_index(3), 0); // rounds down to 2
        assert_eq!(weights.empty_range_index(4), 1);
        assert_eq!(weights.empty_range_index(60), 29);
    }

    #[test]
    fn test_weight_get_set() {
        let features = Features::edax();
        let mut weights = Weights::new(features);

        weights.set_weight(0, 0, 30, 42.0);
        assert!((weights.get_weight(0, 0, 30) - 42.0).abs() < 0.001);
    }

    #[test]
    fn test_sgd_update() {
        let features = Features::edax();
        let mut weights = Weights::new(features);

        weights.set_weight(0, 0, 30, 0.0);
        weights.update_weight_sgd(0, 0, 30, 0.01, 100.0);
        let w = weights.get_weight(0, 0, 30);
        assert!(w > 0.0); // weight should increase
        assert!((w - 1.0).abs() < 0.001); // 0.01 * 100 = 1.0 (was truncated to 0 as i16!)
    }
}
