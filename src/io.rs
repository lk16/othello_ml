use crate::features::Features;
use crate::weights::Weights;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};

/// Binary serialization format for Othello position weights.
///
/// File format:
/// - Header (12 bytes):
///   - Magic number: 0xDEADBEEF (4 bytes) - file validation
///   - Format version: 2 (4 bytes) - compatibility checking
///   - Number of features: N (4 bytes)
/// - Feature metadata (variable):
///   - For each feature: name_length + name + cells_count + cell_indices
/// - Weight data (variable):
///   - All weights in row-major order: [feature][empty_range][pattern]
///   - Weights are stored as f32 (little-endian) for lossless precision.
///
/// The single file contains everything needed to reconstruct weights from disk.
const MAGIC_NUMBER: u32 = 0xDEADBEEF;
const FORMAT_VERSION: u32 = 2;

pub struct WeightIO;

impl WeightIO {
    /// Save weights to a file as f32 (lossless).
    pub fn save(weights: &Weights, path: &str) -> Result<(), String> {
        let mut file =
            BufWriter::new(File::create(path).map_err(|e| format!("Failed to create file: {e}"))?);

        // Write header
        file.write_all(&MAGIC_NUMBER.to_le_bytes())
            .map_err(|e| e.to_string())?;
        file.write_all(&FORMAT_VERSION.to_le_bytes())
            .map_err(|e| e.to_string())?;

        let n_features = weights.feature_count() as u32;
        file.write_all(&n_features.to_le_bytes())
            .map_err(|e| e.to_string())?;

        // Write features metadata
        let features = weights.features();
        for feature in features.all() {
            // Write feature name length and name
            let name_bytes = feature.name.as_bytes();
            file.write_all(&(name_bytes.len() as u32).to_le_bytes())
                .map_err(|e| e.to_string())?;
            file.write_all(name_bytes).map_err(|e| e.to_string())?;

            // Write cells
            file.write_all(&(feature.cells.len() as u32).to_le_bytes())
                .map_err(|e| e.to_string())?;
            for &cell in &feature.cells {
                file.write_all(&cell.to_le_bytes())
                    .map_err(|e| e.to_string())?;
            }
        }

        // Write weight data as f32 (lossless, no rounding)
        let all_weights = weights.get_all_weights();
        for feature_weights in all_weights {
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

    /// Load weights from a file.
    ///
    /// Supports both format versions:
    /// - Version 2: weights stored as f32 (4 bytes each)
    /// - Version 1: weights stored as i16 (2 bytes each) — for backward compatibility
    pub fn load(path: &str) -> Result<Weights, String> {
        let mut file =
            BufReader::new(File::open(path).map_err(|e| format!("Failed to open file: {e}"))?);

        // Read header
        let mut header = [0u8; 12];
        file.read_exact(&mut header)
            .map_err(|e| format!("Failed to read header: {e}"))?;

        let magic = u32::from_le_bytes([header[0], header[1], header[2], header[3]]);
        if magic != MAGIC_NUMBER {
            return Err("Invalid magic number".to_string());
        }

        let version = u32::from_le_bytes([header[4], header[5], header[6], header[7]]);
        let is_v1 = version == 1;
        if version != FORMAT_VERSION && !is_v1 {
            return Err(format!("Unsupported format version: {version}"));
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
                    if is_v1 {
                        // Version 1: i16 (2 bytes)
                        let mut weight_bytes = [0u8; 2];
                        file.read_exact(&mut weight_bytes)
                            .map_err(|e| e.to_string())?;
                        empty_range_weights.push(i16::from_le_bytes(weight_bytes) as f32);
                    } else {
                        // Version 2: f32 (4 bytes)
                        let mut weight_bytes = [0u8; 4];
                        file.read_exact(&mut weight_bytes)
                            .map_err(|e| e.to_string())?;
                        empty_range_weights.push(f32::from_le_bytes(weight_bytes));
                    }
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
    fn test_save_and_load() {
        let features = Features::edax();
        let mut weights = Weights::new(features);

        // Set some test weights with fractional parts
        weights.set_weight(0, 0, 2, 42.7);
        weights.set_weight(1, 5, 10, 99.3);

        let path = "/tmp/test_weights.bin";

        // Save
        assert!(WeightIO::save(&weights, path).is_ok());

        // Verify file exists and has content
        assert!(fs::metadata(path).is_ok());

        // Load
        let loaded = WeightIO::load(path);
        assert!(loaded.is_ok());

        let loaded_weights = loaded.unwrap();
        // Fractional parts should be preserved exactly with f32 storage
        assert!((loaded_weights.get_weight(0, 0, 2) - 42.7).abs() < 0.001);
        assert!((loaded_weights.get_weight(1, 5, 10) - 99.3).abs() < 0.001);

        // Clean up
        let _ = fs::remove_file(path);
    }
}
