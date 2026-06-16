use crate::training::features::Features;

/// Number of empty-count buckets: **one weight table per empties value** `0..=60`.
/// (Previously 30 buckets paired adjacent empties; now each empties value has its
/// own table for finer-grained, per-ply weights.)
pub(crate) const EMPTY_RANGE_COUNT: usize = 61;

/// Weight-table bucket index for a position with `empties` empty squares: one
/// bucket per value, clamped to `60`. Single source of truth shared by [`Weights`]
/// and the flat solver eval (`crate::eval::pattern::FlatEval`) so the two cannot
/// drift apart.
pub(crate) fn empty_range_index(empties: u32) -> usize {
    empties.min(60) as usize
}

/// Weight table for evaluating Othello positions.
///
/// Stores evaluation weights indexed by:
/// - Feature index: which of the 47 features (0-46)
/// - Empty count: which of the 61 per-empties tables (one per empties value 0..=60)
/// - Pattern index: which pattern configuration within that feature (0 to 3^cells-1)
///
/// Weights are stored as f32 internally for precise SGD updates.
///
/// Position evaluation = sum of all feature weights for the current board state.
#[derive(Clone)]
pub struct Weights {
    // Feature scores: [feature][empty_range][pattern] = score (f32 for SGD precision)
    feature_weights: Vec<Vec<Vec<f32>>>,
    empty_ranges: Vec<u32>, // one entry per empties value 0..=60
    features: Features,
}

impl Weights {
    /// Create a new weight table with Edax features and SGD training
    pub fn new(features: Features) -> Self {
        let n_features = features.count();
        let n_empty_ranges = EMPTY_RANGE_COUNT; // one per empties value 0..=60

        // One bucket per empties value 0..=60.
        let empty_ranges: Vec<u32> = (0..EMPTY_RANGE_COUNT as u32).collect();

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

    /// Weight-table bucket index for `empties` — one bucket per empties value.
    /// Delegates to the shared [`empty_range_index`] (kept in sync with `FlatEval`).
    fn empty_range_index(&self, empties: u32) -> usize {
        empty_range_index(empties)
    }

    /// Evaluate a board position by summing contributions from all features
    pub fn evaluate(&self, board: &crate::othello::position::Position, features: &Features) -> f32 {
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

    /// Evaluate from precomputed feature-pattern indices (one per feature).
    ///
    /// Alloc-free hot path used by training: the caller extracts the indices
    /// once (positions are fixed across epochs) and reuses them, so this skips
    /// the per-call `Features::extract` `Vec` and the per-cell `get_cell` loop.
    /// The `range_idx` is computed once instead of per feature.
    pub fn evaluate_indices(&self, indices: &[u32], empties: u32) -> f32 {
        let range_idx = self.empty_range_index(empties);
        let mut score = 0.0f32;
        for (feat_idx, &pattern_idx) in indices.iter().enumerate() {
            let row = &self.feature_weights[feat_idx][range_idx];
            let p = pattern_idx as usize;
            if p < row.len() {
                score += row[p];
            }
        }
        score
    }

    /// In-place SGD step over precomputed feature-pattern indices.
    ///
    /// Equivalent to calling [`Weights::update_weight_sgd`] for every feature,
    /// but computes `range_idx` once and indexes directly (no per-feature
    /// `get`/`set` round-trip). Same `MAX_WEIGHT` clamp.
    pub fn sgd_step_indices(
        &mut self,
        indices: &[u32],
        empties: u32,
        learning_rate: f32,
        gradient: f32,
    ) {
        const MAX_WEIGHT: f32 = 100.0;
        let range_idx = self.empty_range_index(empties);
        let step = learning_rate * gradient;
        for (feat_idx, &pattern_idx) in indices.iter().enumerate() {
            let row = &mut self.feature_weights[feat_idx][range_idx];
            let p = pattern_idx as usize;
            if p < row.len() {
                row[p] = (row[p] + step).clamp(-MAX_WEIGHT, MAX_WEIGHT);
            }
        }
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

    /// The feature set used by this weight table.
    pub fn features(&self) -> &Features {
        &self.features
    }

    /// Number of features in the weight table.
    pub fn feature_count(&self) -> usize {
        self.feature_weights.len()
    }

    /// Number of empty-count buckets (61: one per empties value 0..=60).
    pub fn empty_range_count(&self) -> usize {
        self.empty_ranges.len()
    }

    /// Load weights from `path` or create a fresh table if the file doesn't exist
    /// or fails to load. Logs progress to stderr.
    pub fn load_or_create(path: &str, features: &Features) -> Self {
        if std::path::Path::new(path).exists() {
            eprintln!("Loading weights from {path} ...");
            match Weights::load(path) {
                Ok(w) => {
                    eprintln!(
                        "Loaded weights: {} features x {} empty ranges",
                        w.feature_count(),
                        w.empty_range_count()
                    );
                    w
                }
                Err(e) => {
                    eprintln!("Error loading weights (starting fresh): {e}");
                    Weights::new(features.clone())
                }
            }
        } else {
            let w = Weights::new(features.clone());
            eprintln!(
                "Weight table: {} features x {} empty ranges",
                w.feature_count(),
                w.empty_range_count()
            );
            w
        }
    }

    /// Print a sample of learned weights to stderr for diagnostics.
    pub fn print_sample(&self, features: &Features, n: usize) {
        use crate::othello::position::Position;

        eprintln!("\n--- Sample learned weights (feature 0 = A1 corner, empty=60) ---");
        let board = Position::initial();
        let feature_indices = features.extract(&board);
        for (feat_idx, &pattern_idx) in feature_indices.iter().enumerate().take(n) {
            let w = self.get_weight(feat_idx, pattern_idx, 60);
            if w != 0.0 {
                eprintln!("  Feature {feat_idx} pattern {pattern_idx}: weight = {w}");
            }
        }
    }

    /// Get all weights as f32.
    pub fn get_all_weights(&self) -> &Vec<Vec<Vec<f32>>> {
        &self.feature_weights
    }

    /// Replace all weights (used during deserialization).
    pub fn set_all_weights(&mut self, weights: Vec<Vec<Vec<f32>>>) {
        self.feature_weights = weights;
    }

    /// Merge weight deltas from parallel workers.
    ///
    /// Each worker cloned the weights before training, so
    /// `workers[i] - self` is the delta from worker i.  We apply the
    /// average delta: `self += sum(workers[i] - self) / n_workers`.
    pub fn merge_from_workers(&mut self, workers: &[Weights]) {
        let n = workers.len() as f32;
        for f in 0..self.feature_weights.len() {
            for e in 0..self.feature_weights[f].len() {
                for p in 0..self.feature_weights[f][e].len() {
                    let original = self.feature_weights[f][e][p];
                    let delta: f32 = workers
                        .iter()
                        .map(|w| w.feature_weights[f][e][p] - original)
                        .sum();
                    self.feature_weights[f][e][p] = original + delta / n;
                }
            }
        }
    }

    // ─── Serialization ──────────────────────────────────────────────

    const MAGIC_NUMBER: u32 = 0xDEADBEEF;
    // v3: one weight table per empties value 0..=60 (61 buckets). v1/v2 used 30
    // paired-empties buckets and are no longer loadable — re-train from scratch.
    const FORMAT_VERSION: u32 = 3;

    /// Save weights to a file as f32 (lossless).
    pub fn save(&self, path: &str) -> Result<(), String> {
        use std::io::{BufWriter, Write};

        let mut file = std::fs::File::create(path)
            .map(BufWriter::new)
            .map_err(|e| format!("Failed to create file: {e}"))?;

        // Write header
        file.write_all(&Self::MAGIC_NUMBER.to_le_bytes())
            .map_err(|e| e.to_string())?;
        file.write_all(&Self::FORMAT_VERSION.to_le_bytes())
            .map_err(|e| e.to_string())?;

        let n_features = self.feature_count() as u32;
        file.write_all(&n_features.to_le_bytes())
            .map_err(|e| e.to_string())?;

        // Write features metadata
        for feature in self.features.all() {
            let name_bytes = feature.name.as_bytes();
            file.write_all(&(name_bytes.len() as u32).to_le_bytes())
                .map_err(|e| e.to_string())?;
            file.write_all(name_bytes).map_err(|e| e.to_string())?;

            file.write_all(&(feature.cells.len() as u32).to_le_bytes())
                .map_err(|e| e.to_string())?;
            for &cell in &feature.cells {
                file.write_all(&cell.to_le_bytes())
                    .map_err(|e| e.to_string())?;
            }
        }

        // Write weight data as f32 (lossless, no rounding)
        for feature_weights in &self.feature_weights {
            for empty_range_weights in feature_weights {
                for &weight in empty_range_weights {
                    file.write_all(&weight.to_le_bytes())
                        .map_err(|e| e.to_string())?;
                }
            }
        }

        file.flush().map_err(|e| e.to_string())?;
        Ok(())
    }

    /// Load weights from a file (format v3: f32 weights, 61 per-empties buckets).
    /// Older v1/v2 files use a different bucketing and are rejected — re-train.
    pub fn load(path: &str) -> Result<Weights, String> {
        use std::io::{BufReader, Read};

        let mut file = std::fs::File::open(path)
            .map(BufReader::new)
            .map_err(|e| format!("Failed to open file: {e}"))?;

        // Read header
        let mut header = [0u8; 12];
        file.read_exact(&mut header)
            .map_err(|e| format!("Failed to read header: {e}"))?;

        let magic = u32::from_le_bytes([header[0], header[1], header[2], header[3]]);
        if magic != Self::MAGIC_NUMBER {
            return Err("Invalid magic number".to_string());
        }

        let version = u32::from_le_bytes([header[4], header[5], header[6], header[7]]);
        if version != Self::FORMAT_VERSION {
            return Err(format!(
                "Unsupported format version: {version} (this build expects v{}; the \
                 empties-bucketing changed — re-train from scratch)",
                Self::FORMAT_VERSION
            ));
        }

        let n_features =
            u32::from_le_bytes([header[8], header[9], header[10], header[11]]) as usize;

        // Read features metadata
        let mut features_vec = Vec::new();
        for _ in 0..n_features {
            let mut name_len_bytes = [0u8; 4];
            file.read_exact(&mut name_len_bytes)
                .map_err(|e| e.to_string())?;
            let name_len = u32::from_le_bytes(name_len_bytes) as usize;

            let mut name_bytes = vec![0u8; name_len];
            file.read_exact(&mut name_bytes)
                .map_err(|e| e.to_string())?;
            let name = String::from_utf8(name_bytes)
                .map_err(|e| format!("Invalid UTF-8 in feature name: {e}"))?;

            let mut n_cells_bytes = [0u8; 4];
            file.read_exact(&mut n_cells_bytes)
                .map_err(|e| e.to_string())?;
            let n_cells = u32::from_le_bytes(n_cells_bytes) as usize;

            let mut cells = Vec::new();
            for _ in 0..n_cells {
                let mut cell_bytes = [0u8; 4];
                file.read_exact(&mut cell_bytes)
                    .map_err(|e| e.to_string())?;
                cells.push(u32::from_le_bytes(cell_bytes));
            }

            features_vec.push((name, cells));
        }

        // Reconstruct features from loaded data
        let features = Features::edax();

        // Verify loaded features match expected
        if features.count() != n_features {
            return Err(format!(
                "Feature count mismatch: expected {}, got {}",
                n_features,
                features.count()
            ));
        }

        // Create weights structure
        let mut weights = Weights::new(features);

        // Read weight data
        let mut all_weights = Vec::new();
        for feature_idx in 0..n_features {
            let feature = weights
                .features()
                .get(feature_idx)
                .ok_or("Feature index out of range")?;
            let max_pattern = (feature.max_index() + 1) as usize;
            let n_empty_ranges = weights.empty_range_count();

            let mut feature_weights = Vec::new();
            for _ in 0..n_empty_ranges {
                let mut empty_range_weights = Vec::new();
                for _ in 0..max_pattern {
                    let mut weight_bytes = [0u8; 4];
                    file.read_exact(&mut weight_bytes)
                        .map_err(|e| e.to_string())?;
                    empty_range_weights.push(f32::from_le_bytes(weight_bytes));
                }
                feature_weights.push(empty_range_weights);
            }
            all_weights.push(feature_weights);
        }

        weights.set_all_weights(all_weights);
        Ok(weights)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn test_weights_creation() {
        let features = Features::edax();
        let weights = Weights::new(features);
        assert!(weights.feature_count() > 0);
        assert_eq!(weights.empty_range_count(), 61);
    }

    #[test]
    fn test_empty_range_index() {
        let features = Features::edax();
        let weights = Weights::new(features);

        // One bucket per empties value: index == empties (clamped to 60).
        assert_eq!(weights.empty_range_index(0), 0);
        assert_eq!(weights.empty_range_index(2), 2);
        assert_eq!(weights.empty_range_index(3), 3);
        assert_eq!(weights.empty_range_index(4), 4);
        assert_eq!(weights.empty_range_index(60), 60);
        assert_eq!(weights.empty_range_index(64), 60); // clamps
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
        assert!((w - 1.0).abs() < 0.001); // 0.01 * 100 = 1.0
    }

    #[test]
    fn test_save_and_load() {
        let features = Features::edax();
        let mut weights = Weights::new(features);

        // Set some test weights with fractional parts
        weights.set_weight(0, 0, 2, 42.7);
        weights.set_weight(1, 5, 10, 99.3);

        let path = "/tmp/test_weights.bin";

        // Save
        assert!(weights.save(path).is_ok());

        // Verify file exists and has content
        assert!(fs::metadata(path).is_ok());

        // Load
        let loaded = Weights::load(path);
        assert!(loaded.is_ok());

        let loaded_weights = loaded.unwrap();
        // Fractional parts should be preserved exactly with f32 storage
        assert!((loaded_weights.get_weight(0, 0, 2) - 42.7).abs() < 0.001);
        assert!((loaded_weights.get_weight(1, 5, 10) - 99.3).abs() < 0.001);

        // Clean up
        let _ = fs::remove_file(path);
    }
}
