//! Persistent cache of exact position evaluations, stored as `<FEN> <score>` lines.
//!
//! Avoids re-evaluating the same positions across training runs by
//! loading known evaluations from disk and only computing missing ones.

use crate::eval::alphabeta;
use crate::othello::board::Board;
use crate::othello::position::Position;
use crate::training::trainer::TrainingExample;
use std::collections::HashMap;
use std::fs;
use std::io::Write;

/// Cached exact evaluations backed by a text file.
pub struct EvalCache {
    path: String,
}

impl EvalCache {
    /// Create a cache handle. The backing file does not need to exist yet.
    pub fn new(path: String) -> Self {
        EvalCache { path }
    }

    /// True if the cache file already exists on disk.
    pub fn exists(&self) -> bool {
        std::path::Path::new(&self.path).exists()
    }

    /// Load all cached evaluations into a FEN → score map.
    ///
    /// Format: one `<FEN> <score>` pair per line. The FEN is 66 characters
    /// (64 board + space + side to move).
    pub fn load_map(&self) -> Result<HashMap<String, i32>, String> {
        let content = fs::read_to_string(&self.path)
            .map_err(|e| format!("Failed to read {}: {}", self.path, e))?;
        let mut map = HashMap::new();

        for (line_no, line) in content.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            if line.len() < 68 {
                return Err(format!(
                    "{}:{}: line too short (expected '<66-char FEN> <score>')",
                    self.path,
                    line_no + 1
                ));
            }
            let fen = &line[..66];
            let score_str = line[67..].trim();
            let score = score_str.parse::<i32>().map_err(|e| {
                format!(
                    "{}:{}: invalid score '{}': {}",
                    self.path,
                    line_no + 1,
                    score_str,
                    e
                )
            })?;

            if fen.as_bytes()[64] != b' ' {
                return Err(format!(
                    "{}:{}: FEN missing space at position 65",
                    self.path,
                    line_no + 1
                ));
            }

            map.insert(fen.to_string(), score);
        }

        Ok(map)
    }

    /// Append evaluations for a subset of positions to the cache file.
    pub fn append(&self, positions: &[&Board], scores: &[i32]) -> Result<(), String> {
        let mut file = fs::OpenOptions::new()
            .append(true)
            .open(&self.path)
            .map_err(|e| format!("Failed to open {} for appending: {}", self.path, e))?;
        for (pos, &score) in positions.iter().zip(scores.iter()) {
            let fen = pos.position.to_fen(pos.black_to_move);
            writeln!(file, "{fen} {score}")
                .map_err(|e| format!("Failed to write eval file: {e}"))?;
        }
        Ok(())
    }

    /// Create (or overwrite) the cache file with evaluations for all positions.
    pub fn save_all(
        &self,
        positions: &[Board],
        examples: &[TrainingExample],
    ) -> Result<(), String> {
        let mut file = fs::File::create(&self.path)
            .map_err(|e| format!("Failed to create {}: {}", self.path, e))?;
        for (pos, ex) in positions.iter().zip(examples.iter()) {
            let fen = pos.position.to_fen(pos.black_to_move);
            writeln!(file, "{} {}", fen, ex.target_score)
                .map_err(|e| format!("Failed to write eval file: {e}"))?;
        }
        Ok(())
    }

    /// Build training examples from positions, using cached evaluations when
    /// available and computing the rest via alpha-beta.
    ///
    /// If the cache file exists, known positions are loaded from it and only
    /// missing ones are computed (appended after). If the cache doesn't
    /// exist yet, all positions are evaluated and the result is saved.
    pub fn build_examples(&self, positions: &[Board]) -> Result<Vec<TrainingExample>, String> {
        if self.exists() {
            self.build_from_existing(positions)
        } else {
            self.build_fresh(positions)
        }
    }

    /// Load cached evaluations and compute only missing positions.
    fn build_from_existing(&self, positions: &[Board]) -> Result<Vec<TrainingExample>, String> {
        eprintln!("\n--- Loading evaluations from {} ---", self.path);
        let eval_map = self.load_map()?;
        eprintln!("Loaded {} evaluations", eval_map.len());

        let mut examples = Vec::with_capacity(positions.len());
        let mut missing: Vec<&Board> = Vec::new();
        for pos in positions {
            let fen = pos.position.to_fen(pos.black_to_move);
            match eval_map.get(&fen) {
                Some(&score) => examples.push(TrainingExample {
                    position: pos.position,
                    target_score: score,
                }),
                None => missing.push(pos),
            }
        }

        if !missing.is_empty() {
            let n = missing.len();
            eprintln!("Computing {n} missing positions with alpha-beta...");
            let boards: Vec<Position> = missing.iter().map(|p| p.position).collect();
            let scores = alphabeta::batch_evaluate(&boards);

            self.append(&missing, &scores)?;
            eprintln!("Appended {n} new evaluations to {}", self.path);

            for (pos, &score) in missing.iter().zip(scores.iter()) {
                examples.push(TrainingExample {
                    position: pos.position,
                    target_score: score,
                });
            }
        }
        Ok(examples)
    }

    /// Evaluate all positions and create a new cache file.
    fn build_fresh(&self, positions: &[Board]) -> Result<Vec<TrainingExample>, String> {
        let n = positions.len();
        eprintln!(
            "\n--- Evaluating {n} positions with alpha-beta → saving to {} ---",
            self.path
        );

        let eval_start = std::time::Instant::now();
        let boards: Vec<Position> = positions.iter().map(|p| p.position).collect();
        let scores = alphabeta::batch_evaluate(&boards);

        let elapsed = eval_start.elapsed();
        eprintln!(
            "  Done in {:.1}s ({:.0} pos/s)",
            elapsed.as_secs_f64(),
            n as f64 / elapsed.as_secs_f64().max(0.001)
        );

        let examples: Vec<TrainingExample> = positions
            .iter()
            .zip(scores.iter())
            .map(|(pos, &score)| TrainingExample {
                position: pos.position,
                target_score: score,
            })
            .collect();

        eprintln!("Saving evaluations to {} ...", self.path);
        self.save_all(positions, &examples)?;
        eprintln!("Saved {} evaluations", examples.len());
        Ok(examples)
    }
}

/// Build training examples from positions, either via an eval-file cache or
/// by evaluating all positions with alpha-beta directly.
///
/// When `eval_file` is `Some`, positions are looked up in the cache (computing
/// and appending any missing ones). When `None`, all positions are evaluated
/// in one batch with no caching.
pub fn build_examples(
    eval_file: &Option<String>,
    positions: &[Board],
) -> Result<Vec<TrainingExample>, String> {
    if let Some(ref path) = eval_file {
        let cache = EvalCache::new(path.clone());
        cache.build_examples(positions)
    } else {
        let n = positions.len();
        eprintln!("\n--- Evaluating {n} positions with alpha-beta ---");
        let eval_start = std::time::Instant::now();
        let boards: Vec<Position> = positions.iter().map(|p| p.position).collect();
        let scores = alphabeta::batch_evaluate(&boards);
        let elapsed = eval_start.elapsed();
        eprintln!(
            "  Done in {:.1}s ({:.0} pos/s)",
            elapsed.as_secs_f64(),
            n as f64 / elapsed.as_secs_f64().max(0.001)
        );
        let examples = positions
            .iter()
            .zip(scores.iter())
            .map(|(pos, &score)| TrainingExample {
                position: pos.position,
                target_score: score,
            })
            .collect();
        Ok(examples)
    }
}
