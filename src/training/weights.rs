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
    // Weights are **tied by symmetry shape**: features that are board-symmetry
    // images of one another (e.g. the 4 corners) share one weight table, so the
    // same physical pattern always updates the same weights (Edax mirror-packing;
    // see `Features::symmetry_shapes`). Storage is per shape, not per feature:
    //   shape_weights[shape][empty_range][pattern] = score (f32 for SGD precision)
    shape_weights: Vec<Vec<Vec<f32>>>,
    // feature index -> shape id (the shared table it reads/updates).
    feature_to_shape: Vec<usize>,
    features: Features,
}

impl Weights {
    /// Create a new weight table with Edax features and SGD training.
    ///
    /// Symmetric features are tied: one shared `[empty_range][pattern]` table per
    /// symmetry shape (12 shapes for the 46 Edax features).
    pub fn new(features: Features) -> Self {
        let n_empty_ranges = EMPTY_RANGE_COUNT; // one per empties value 0..=60
        let (feature_to_shape, n_shapes) = features.symmetry_shapes();

        // Pattern count of each shape = 3^cells of any of its member features.
        let mut shape_pattern_count = vec![0usize; n_shapes];
        for (f, feature) in features.all().iter().enumerate() {
            shape_pattern_count[feature_to_shape[f]] = (feature.max_index() + 1) as usize;
        }

        let shape_weights = shape_pattern_count
            .iter()
            .map(|&pc| vec![vec![0.0f32; pc]; n_empty_ranges])
            .collect();

        Weights {
            shape_weights,
            feature_to_shape,
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
            if let Some(&shape) = self.feature_to_shape.get(feat_idx) {
                let row = &self.shape_weights[shape][range_idx];
                let pattern_idx = pattern_idx as usize;
                if pattern_idx < row.len() {
                    score += row[pattern_idx];
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
            let row = &self.shape_weights[self.feature_to_shape[feat_idx]][range_idx];
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
            let row = &mut self.shape_weights[self.feature_to_shape[feat_idx]][range_idx];
            let p = pattern_idx as usize;
            if p < row.len() {
                row[p] = (row[p] + step).clamp(-MAX_WEIGHT, MAX_WEIGHT);
            }
        }
    }

    /// Get weight for a specific feature, pattern, and empty range.
    /// Reads the feature's tied shape table.
    pub fn get_weight(&self, feature_idx: usize, pattern_idx: u32, empties: u32) -> f32 {
        let Some(&shape) = self.feature_to_shape.get(feature_idx) else {
            return 0.0;
        };
        let range_idx = self.empty_range_index(empties);
        let pattern_idx = pattern_idx as usize;
        let row = &self.shape_weights[shape][range_idx];
        if pattern_idx < row.len() {
            row[pattern_idx]
        } else {
            0.0
        }
    }

    /// Set weight for a specific feature, pattern, and empty range.
    /// Writes the feature's tied shape table (shared with symmetric features).
    pub fn set_weight(&mut self, feature_idx: usize, pattern_idx: u32, empties: u32, weight: f32) {
        let Some(&shape) = self.feature_to_shape.get(feature_idx) else {
            return;
        };
        let range_idx = self.empty_range_index(empties);
        let pattern_idx = pattern_idx as usize;
        let row = &mut self.shape_weights[shape][range_idx];
        if pattern_idx < row.len() {
            row[pattern_idx] = weight;
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
        self.feature_to_shape.len()
    }

    /// Number of symmetry shapes (tied weight tables; 12 for the Edax set).
    pub fn shape_count(&self) -> usize {
        self.shape_weights.len()
    }

    /// Map from feature index to its tied shape id.
    pub fn feature_to_shape(&self) -> &[usize] {
        &self.feature_to_shape
    }

    /// Number of empty-count buckets (61: one per empties value 0..=60).
    pub fn empty_range_count(&self) -> usize {
        EMPTY_RANGE_COUNT
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

    /// All weights as f32, per **shape**: `[shape][empty_range][pattern]`.
    pub fn shape_weights(&self) -> &Vec<Vec<Vec<f32>>> {
        &self.shape_weights
    }

    /// Merge weight deltas from parallel workers.
    ///
    /// Each worker cloned the weights before training, so
    /// `workers[i] - self` is the delta from worker i.  We apply the
    /// average delta: `self += sum(workers[i] - self) / n_workers`.
    pub fn merge_from_workers(&mut self, workers: &[Weights]) {
        let n = workers.len() as f32;
        for s in 0..self.shape_weights.len() {
            for e in 0..self.shape_weights[s].len() {
                for p in 0..self.shape_weights[s][e].len() {
                    let original = self.shape_weights[s][e][p];
                    let delta: f32 = workers
                        .iter()
                        .map(|w| w.shape_weights[s][e][p] - original)
                        .sum();
                    self.shape_weights[s][e][p] = original + delta / n;
                }
            }
        }
    }

    // ─── Serialization ──────────────────────────────────────────────

    const MAGIC_NUMBER: u32 = 0xDEADBEEF;
    // v4: weights tied by symmetry shape (one table per shape, 12 for the Edax set)
    // and the corrected 46-feature set. Weight data is written per shape in
    // `Features::symmetry_shapes` order. v1/v2/v3 are no longer loadable — re-train.
    const FORMAT_VERSION: u32 = 4;

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

        // Write weight data as f32 (lossless, no rounding), per shape. The shape
        // structure is re-derived from the features on load (deterministic), so it
        // need not be stored.
        for shape_weights in &self.shape_weights {
            for empty_range_weights in shape_weights {
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

        // Create weights structure (allocates the per-shape tables) and fill them
        // in the same shape/range/pattern order they were written.
        let mut weights = Weights::new(features);
        for s in 0..weights.shape_weights.len() {
            for r in 0..weights.shape_weights[s].len() {
                for p in 0..weights.shape_weights[s][r].len() {
                    let mut weight_bytes = [0u8; 4];
                    file.read_exact(&mut weight_bytes)
                        .map_err(|e| e.to_string())?;
                    weights.shape_weights[s][r][p] = f32::from_le_bytes(weight_bytes);
                }
            }
        }
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
        assert_eq!(weights.feature_count(), 46);
        assert_eq!(weights.shape_count(), 12); // tied: 46 features → 12 shape tables
        assert_eq!(weights.empty_range_count(), 61);
    }

    #[test]
    fn symmetric_features_share_storage() {
        let features = Features::edax();
        let (map, _) = features.symmetry_shapes();
        // Features 0 and 2 are both corners → same shape.
        assert_eq!(map[0], map[2]);

        let mut w = Weights::new(features);
        // Writing via one corner is visible via another at the same (pattern, range):
        // they are tied to one shared table.
        w.set_weight(0, 7, 30, 3.5);
        assert_eq!(w.get_weight(2, 7, 30), 3.5);
        // A feature in a different shape is untouched.
        let other = (0..map.len()).find(|&f| map[f] != map[0]).unwrap();
        assert_eq!(w.get_weight(other, 7, 30), 0.0);
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
