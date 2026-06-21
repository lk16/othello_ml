//! Persistent cache of exact position evaluations, stored as `<FEN> <score>` lines.
//!
//! Avoids re-evaluating the same positions across training runs by
//! loading known evaluations from disk and only computing missing ones.

use crate::eval::alphabeta;
use crate::othello::board::Board;
use crate::othello::position::Position;
use crate::training::TrainingExample;
use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

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

    /// Append evaluations for specific indices of the missing positions.
    fn append_indices(
        &self,
        missing: &[&Board],
        scores: &[i32],
        indices: &[usize],
    ) -> Result<(), String> {
        let mut file = fs::OpenOptions::new()
            .append(true)
            .open(&self.path)
            .map_err(|e| format!("Failed to open {} for appending: {}", self.path, e))?;
        for &idx in indices {
            let fen = missing[idx].position.to_fen(missing[idx].black_to_move);
            writeln!(file, "{fen} {}", scores[idx])
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

    /// Save evaluations for specific indices of positions.
    fn save_indices(
        &self,
        positions: &[Board],
        scores: &[i32],
        indices: &[usize],
    ) -> Result<(), String> {
        let mut file = fs::File::create(&self.path)
            .map_err(|e| format!("Failed to create {}: {}", self.path, e))?;
        for &idx in indices {
            let fen = positions[idx].position.to_fen(positions[idx].black_to_move);
            writeln!(file, "{fen} {}", scores[idx])
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
    pub fn build_examples(
        &self,
        positions: &[Board],
        interrupt: &Arc<AtomicBool>,
        threads: usize,
    ) -> Result<Vec<TrainingExample>, String> {
        if self.exists() {
            self.build_from_existing(positions, interrupt, threads)
        } else {
            self.build_fresh(positions, interrupt, threads)
        }
    }

    /// Load cached evaluations and compute only missing positions.
    fn build_from_existing(
        &self,
        positions: &[Board],
        interrupt: &Arc<AtomicBool>,
        threads: usize,
    ) -> Result<Vec<TrainingExample>, String> {
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

            let missing_positions: Vec<Position> = missing.iter().map(|b| b.position).collect();
            let (scores, completed) =
                evaluate_positions_parallel(&missing_positions, threads, interrupt);

            if interrupt.load(Ordering::Relaxed) {
                eprintln!(
                    "\nInterrupted! Saving {} evaluated positions...",
                    completed.len()
                );
                self.append_indices(&missing, &scores, &completed)?;
                eprintln!("Saved {} evaluations to {}", completed.len(), self.path);
                return Err("Interrupted by user".to_string());
            }

            self.append_indices(&missing, &scores, &completed)?;
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
    fn build_fresh(
        &self,
        positions: &[Board],
        interrupt: &Arc<AtomicBool>,
        threads: usize,
    ) -> Result<Vec<TrainingExample>, String> {
        let n = positions.len();
        eprintln!(
            "\n--- Evaluating {n} positions with alpha-beta → saving to {} ---",
            self.path
        );

        let all_positions: Vec<Position> = positions.iter().map(|b| b.position).collect();
        let (scores, completed) = evaluate_positions_parallel(&all_positions, threads, interrupt);

        if interrupt.load(Ordering::Relaxed) {
            eprintln!(
                "\nInterrupted! Saving {} evaluated positions...",
                completed.len()
            );
            self.save_indices(positions, &scores, &completed)?;
            eprintln!("Saved {} evaluations to {}", completed.len(), self.path);
            return Err("Interrupted by user".to_string());
        }

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

/// Evaluate positions in parallel using worker threads and channels.
///
/// Returns `(scores, completed_indices)`. If interrupted, `completed_indices`
/// contains only the indices that were evaluated before the interrupt.
/// Otherwise, `completed_indices.len() == positions.len()`.
fn evaluate_positions_parallel(
    positions: &[Position],
    threads: usize,
    interrupt: &Arc<AtomicBool>,
) -> (Vec<i32>, Vec<usize>) {
    use std::sync::mpsc;

    let n = positions.len();
    if n == 0 {
        return (Vec::new(), Vec::new());
    }

    let n_threads = threads.max(1).min(n);
    let (tx, rx) = mpsc::channel::<(usize, i32)>();

    let chunk_size = n.div_ceil(n_threads);
    for (chunk_idx, chunk) in positions.chunks(chunk_size).enumerate() {
        let tx = tx.clone();
        let chunk: Vec<Position> = chunk.to_vec();
        let start = chunk_idx * chunk_size;
        let interrupt = Arc::clone(interrupt);
        std::thread::spawn(move || {
            // One solver (and transposition table) per worker, reused across the
            // whole chunk so the table is allocated once and warms up.
            let mut solver = alphabeta::Solver::new();
            for (i, pos) in chunk.iter().enumerate() {
                if interrupt.load(Ordering::Relaxed) {
                    return;
                }
                let score = solver.exact_score(pos);
                if tx.send((start + i, score)).is_err() {
                    return;
                }
            }
        });
    }
    drop(tx);

    let mut scores = vec![0i32; n];
    let mut completed = Vec::with_capacity(n);
    let eval_start = std::time::Instant::now();
    // Refresh the progress line at most ~5x/sec. Printing + flushing on every
    // received result floods the terminal and throttles throughput on large
    // batches (this loop runs once per evaluated board).
    let mut last_print = std::time::Instant::now();

    while let Ok((idx, score)) = rx.recv() {
        scores[idx] = score;
        completed.push(idx);
        let done = completed.len();
        if last_print.elapsed() < std::time::Duration::from_millis(200) && done < n {
            continue;
        }
        last_print = std::time::Instant::now();
        let elapsed = eval_start.elapsed().as_secs_f64();
        let remaining = n - done;
        let rate = done as f64 / elapsed.max(0.001);
        let eta = remaining as f64 / rate;
        eprint!("\r  {done}/{n} evaluated, {remaining} remaining, ETA: {eta:.0}s          ");
        let _ = std::io::stderr().flush();
    }

    let elapsed = eval_start.elapsed();
    eprintln!(
        "\r  Done in {:.1}s ({:.0} pos/s)          ",
        elapsed.as_secs_f64(),
        n as f64 / elapsed.as_secs_f64().max(0.001)
    );

    (scores, completed)
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
    interrupt: &Arc<AtomicBool>,
    threads: usize,
) -> Result<Vec<TrainingExample>, String> {
    if let Some(ref path) = eval_file {
        let cache = EvalCache::new(path.clone());
        cache.build_examples(positions, interrupt, threads)
    } else {
        let n = positions.len();
        eprintln!("\n--- Evaluating {n} positions with alpha-beta ---");

        let all_positions: Vec<Position> = positions.iter().map(|b| b.position).collect();
        let (scores, completed) = evaluate_positions_parallel(&all_positions, threads, interrupt);

        if interrupt.load(Ordering::Relaxed) {
            eprintln!(
                "\nInterrupted! Evaluated {}/{} positions.",
                completed.len(),
                n
            );
            return Err("Interrupted by user".to_string());
        }

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
