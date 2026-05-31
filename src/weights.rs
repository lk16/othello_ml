use crate::features::Features;

/// Weight table for evaluating Othello positions.
///
/// Stores evaluation weights indexed by:
/// - Feature index: which of the 47 features (0-46)
/// - Empty count range: which of the 30 disc-count tables (indices 0-29 for empties 2,4,6,...,60)
/// - Pattern index: which pattern configuration within that feature (0 to 3^cells-1)
///
/// Each weight is an i16 representing the evaluation contribution of that feature configuration.
/// Position evaluation = sum of all feature weights for the current board state.
pub struct Weights {
    // Feature scores: [feature][empty_range][pattern] = score
    feature_weights: Vec<Vec<Vec<i16>>>,
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
                // Initialize all weights to 0
                feature_scores.push(vec![0i16; max_pattern]);
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
        let clamped = if empties < 2 {
            2
        } else if empties > 60 {
            60
        } else {
            empties
        };

        // Round down to nearest even, then convert to index (0-29)
        let even = (clamped / 2) * 2;
        ((even / 2) - 1) as usize
    }

    /// Evaluate a board position by summing contributions from all features
    pub fn evaluate(&self, board: &crate::board::Board, features: &Features) -> i16 {
        let empties = board.empties();
        let range_idx = self.empty_range_index(empties);
        let feature_indices = features.extract(board);

        let mut score = 0i32;
        for (feat_idx, &pattern_idx) in feature_indices.iter().enumerate() {
            if feat_idx < self.feature_weights.len() {
                let pattern_idx = pattern_idx as usize;
                if pattern_idx < self.feature_weights[feat_idx][range_idx].len() {
                    score += self.feature_weights[feat_idx][range_idx][pattern_idx] as i32;
                }
            }
        }

        // Clamp to i16 range
        score.max(i16::MIN as i32).min(i16::MAX as i32) as i16
    }

    /// Get weight for a specific feature, pattern, and empty range
    pub fn get_weight(&self, feature_idx: usize, pattern_idx: u32, empties: u32) -> i16 {
        if feature_idx >= self.feature_weights.len() {
            return 0;
        }

        let range_idx = self.empty_range_index(empties);
        let pattern_idx = pattern_idx as usize;

        if pattern_idx < self.feature_weights[feature_idx][range_idx].len() {
            self.feature_weights[feature_idx][range_idx][pattern_idx]
        } else {
            0
        }
    }

    /// Set weight for a specific feature, pattern, and empty range
    pub fn set_weight(&mut self, feature_idx: usize, pattern_idx: u32, empties: u32, weight: i16) {
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
    pub fn update_weight_sgd(
        &mut self,
        feature_idx: usize,
        pattern_idx: u32,
        empties: u32,
        learning_rate: f32,
        gradient: f32,
    ) {
        let current = self.get_weight(feature_idx, pattern_idx, empties);
        let delta = (learning_rate * gradient) as i16;
        let new_weight = (current as i32 + delta as i32)
            .max(i16::MIN as i32)
            .min(i16::MAX as i32) as i16;
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

    /// Get all weights (for serialization)
    pub fn get_all_weights(&self) -> &Vec<Vec<Vec<i16>>> {
        &self.feature_weights
    }

    /// Set all weights (for deserialization)
    pub fn set_all_weights(&mut self, weights: Vec<Vec<Vec<i16>>>) {
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

        weights.set_weight(0, 0, 30, 42);
        assert_eq!(weights.get_weight(0, 0, 30), 42);
    }

    #[test]
    fn test_sgd_update() {
        let features = Features::edax();
        let mut weights = Weights::new(features);

        weights.set_weight(0, 0, 30, 0);
        weights.update_weight_sgd(0, 0, 30, 0.1, 100.0);
        let w = weights.get_weight(0, 0, 30);
        assert!(w > 0); // weight should increase
    }
}
