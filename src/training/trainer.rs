use crate::othello::position::Position;
use crate::training::weights::Weights;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

/// Message sent from worker threads back to the main thread.
enum WorkerMsg {
    Done { weights: Weights, loss: f64 },
}

/// Training data point: a board position paired with its ground truth evaluation.
#[derive(Debug, Clone)]
pub struct TrainingExample {
    pub position: Position,
    pub target_score: i32, // Ground truth score from Edax
}

/// SGD (Stochastic Gradient Descent) trainer for optimizing Othello position weights.
///
/// Uses inverse-time learning rate decay:
///   effective_lr = learning_rate / (1.0 + lr_decay × epoch)
///
/// Training process:
/// 1. Forward pass: evaluate board with current weights
/// 2. Compute error: target_score - predicted_score
/// 3. Backward pass: for each feature contributing to the prediction,
///    update its weight: w = w - effective_lr × gradient
/// 4. Repeat for multiple epochs over training data
pub struct Trainer {
    learning_rate: f32, // Initial step size (before decay)
    lr_decay: f32,      // Inverse-time decay factor (0 = no decay)
    batch_size: usize,  // Number of examples per training batch
}

/// Simple xorshift32 PRNG for deterministic shuffling (no external crates needed).
struct XorShift32(u32);

impl XorShift32 {
    fn new(seed: u32) -> Self {
        XorShift32(seed.wrapping_add(1)) // seed of 0 would break the generator
    }

    fn next(&mut self) -> u32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.0 = x;
        x
    }

    /// Random usize in [0, n)
    fn gen_range(&mut self, n: usize) -> usize {
        (self.next() as usize) % n
    }
}

/// Fisher-Yates shuffle using the given PRNG.
fn shuffle<T>(slice: &mut [T], rng: &mut XorShift32) {
    for i in (1..slice.len()).rev() {
        let j = rng.gen_range(i + 1);
        slice.swap(i, j);
    }
}

/// Configuration for a training run.
pub struct TrainingConfig {
    /// Number of epochs to train.
    pub epochs: usize,
    /// Offset added to epoch number for LR decay schedule (0 for fresh start).
    pub epoch_offset: usize,
    /// Optional interrupt flag for graceful early stopping.
    pub interrupt: Option<Arc<AtomicBool>>,
    /// Number of threads for parallel batch processing (1 = single-threaded).
    pub threads: usize,
}

impl Trainer {
    pub fn new(learning_rate: f32, batch_size: usize, lr_decay: f32) -> Self {
        Trainer {
            learning_rate,
            lr_decay,
            batch_size,
        }
    }

    /// Effective learning rate for a given epoch using inverse-time decay.
    fn effective_lr(&self, epoch: usize) -> f32 {
        self.learning_rate / (1.0 + self.lr_decay * epoch as f32)
    }

    /// Train weights on a batch of examples, returning the accumulated squared error.
    pub fn train_batch(
        &self,
        weights: &mut Weights,
        examples: &[TrainingExample],
        effective_lr: f32,
    ) -> f64 {
        let features = weights.features().clone();
        let n_features = features.count() as f32;
        let mut loss: f64 = 0.0;

        for example in examples {
            // Forward pass: compute prediction
            let predicted = weights.evaluate(&example.position, &features);
            let error = example.target_score as f32 - predicted;
            loss += (error as f64) * (error as f64);

            // Backward pass: update each feature weight.
            //
            // Model: prediction = sum of N feature weights (all features active, coef = 1).
            // MSE loss: L = (target - predicted)²
            // dL/dw = dL/dp × dp/dw = -2×(target-predicted) × 1 = -2×error
            //
            // Without normalization, the effective prediction correction per example is
            // N × lr × 2 × error, which with N=47 and lr=0.01 is 0.94×error — far too
            // aggressive and causes oscillatory divergence.  Dividing by N keeps each
            // weight's contribution sensible and prevents overshoot.
            let gradient = 2.0 * error / n_features;

            let feature_indices = features.extract(&example.position);
            for (feat_idx, &pattern_idx) in feature_indices.iter().enumerate() {
                weights.update_weight_sgd(
                    feat_idx,
                    pattern_idx,
                    example.position.empties(),
                    effective_lr,
                    gradient,
                );
            }
        }
        loss
    }

    /// Train for multiple epochs with progress logging.
    ///
    /// Uses [`TrainingConfig`] to control epochs, LR schedule offset,
    /// optional early stopping via an interrupt flag, and thread count.
    ///
    /// When `threads > 1`, each batch is processed in parallel: the batch
    /// is split across worker threads, each with a cloned copy of the
    /// weights.  Results are sent back via channels and merged by
    /// averaging the weight deltas.
    pub fn train_epochs(
        &self,
        weights: &mut Weights,
        examples: &mut [TrainingExample],
        config: &TrainingConfig,
    ) {
        use std::io::{self, Write};
        use std::sync::mpsc;

        let epochs = config.epochs;
        let epoch_offset = config.epoch_offset;
        let n_examples = examples.len();
        let n_batches = n_examples.div_ceil(self.batch_size);
        let total_updates = n_examples * weights.feature_count() * epochs;
        let n_threads = config.threads.max(1);

        eprintln!(
            "Training: {} epochs × {} examples ({} batches/epoch, batch_size={})",
            epochs, n_examples, n_batches, self.batch_size
        );
        eprintln!(
            "  weight updates: {} examples × {} features × {} epochs ≈ {:.1}M total",
            n_examples,
            weights.feature_count(),
            epochs,
            total_updates as f64 / 1_000_000.0
        );
        eprintln!(
            "  lr schedule: {:.4} (epoch {}) → {:.4} (epoch {}) | decay={}",
            self.effective_lr(epoch_offset),
            epoch_offset,
            self.effective_lr(epoch_offset + epochs.saturating_sub(1)),
            epoch_offset + epochs.saturating_sub(1),
            self.lr_decay
        );
        if n_threads > 1 {
            eprintln!("  threads: {n_threads}");
        }
        eprintln!();

        let total_start = Instant::now();

        let mut completed: usize = 0;
        let mut last_loss: f64 = 0.0;

        for epoch in 0..epochs {
            if let Some(ref flag) = config.interrupt {
                if flag.load(Ordering::Relaxed) {
                    eprintln!(
                        "\nInterrupted after {} epochs (global epoch {}) — keeping weights from last completed epoch.",
                        completed, epoch_offset + completed
                    );
                    break;
                }
            }

            let global_epoch = epoch_offset + epoch;
            let mut rng = XorShift32::new(global_epoch as u32);
            shuffle(examples, &mut rng);

            let current_lr = self.effective_lr(global_epoch);

            let epoch_start = Instant::now();
            let mut loss: f64 = 0.0;

            for chunk in examples.chunks(self.batch_size) {
                let workers = n_threads.min(chunk.len());
                let chunk: Vec<TrainingExample> = chunk.to_vec();
                let chunk_size = chunk.len();

                let (tx, rx) = mpsc::channel::<WorkerMsg>();

                for part in chunk.chunks(chunk_size.div_ceil(workers)) {
                    let tx = tx.clone();
                    let mut w = weights.clone();
                    let part = part.to_vec();
                    let lr = current_lr;
                    std::thread::spawn(move || {
                        let loss = {
                            let features = w.features().clone();
                            let n_features = features.count() as f32;
                            let mut loss: f64 = 0.0;
                            for example in &part {
                                let predicted = w.evaluate(&example.position, &features);
                                let error = example.target_score as f32 - predicted;
                                loss += (error as f64) * (error as f64);
                                let gradient = 2.0 * error / n_features;
                                let feature_indices = features.extract(&example.position);
                                for (feat_idx, &pattern_idx) in feature_indices.iter().enumerate() {
                                    w.update_weight_sgd(
                                        feat_idx,
                                        pattern_idx,
                                        example.position.empties(),
                                        lr,
                                        gradient,
                                    );
                                }
                            }
                            loss
                        };
                        let _ = tx.send(WorkerMsg::Done { weights: w, loss });
                    });
                }
                drop(tx);

                let mut worker_weights = Vec::with_capacity(workers);
                let mut done = 0usize;
                let spin = b"|/-\\";

                while done < workers {
                    match rx.recv() {
                        Ok(WorkerMsg::Done {
                            weights: w,
                            loss: l,
                        }) => {
                            done += 1;
                            loss += l;
                            worker_weights.push(w);
                            let elapsed = epoch_start.elapsed().as_secs_f64();
                            let throughput = if elapsed > 0.01 {
                                (done * chunk_size / workers) as f64 / elapsed
                            } else {
                                0.0
                            };
                            eprint!(
                                "\r  {} {}/{} workers  {:.0} ex/s          ",
                                spin[done % spin.len()] as char,
                                done,
                                workers,
                                throughput,
                            );
                            let _ = io::stderr().flush();
                        }
                        Err(_) => break,
                    }
                }

                if let Some(ref flag) = config.interrupt {
                    if flag.load(Ordering::Relaxed) {
                        eprintln!(
                            "\nInterrupted during epoch {global_epoch} — keeping weights from last completed epoch."
                        );
                        return;
                    }
                }

                weights.merge_from_workers(&worker_weights);
            }

            let epoch_elapsed = epoch_start.elapsed();
            let total_elapsed = total_start.elapsed();
            let avg_loss = loss / n_examples as f64;
            let throughput = n_examples as f64 / epoch_elapsed.as_secs_f64().max(0.001);

            let avg_epoch_secs = total_elapsed.as_secs_f64() / (epoch + 1) as f64;
            let remaining_secs = avg_epoch_secs * (epochs - epoch - 1) as f64;

            eprintln!(
                "\rEpoch {}/{} (global {}) | loss: {:.4} | lr: {:.4} | time: {:.1}s | {:.0} ex/s | ETA: {:.0}s   ",
                epoch + 1, epochs, global_epoch + 1, avg_loss, current_lr, epoch_elapsed.as_secs_f64(), throughput, remaining_secs
            );

            last_loss = avg_loss;
            completed = epoch + 1;
        }

        let total_secs = total_start.elapsed().as_secs_f64();
        let overall_throughput = (n_examples * completed) as f64 / total_secs.max(0.001);
        eprintln!(
            "\rComplete        | loss: {last_loss:.4} | time: {total_secs:.1}s | {overall_throughput:.0} ex/s   "
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trainer_creation() {
        let trainer = Trainer::new(0.1, 32, 0.01);
        assert!((trainer.learning_rate - 0.1).abs() < 0.001);
        assert_eq!(trainer.batch_size, 32);
        assert!((trainer.lr_decay - 0.01).abs() < 0.001);
    }

    #[test]
    fn test_effective_lr_no_decay() {
        let trainer = Trainer::new(0.1, 32, 0.0);
        assert!((trainer.effective_lr(0) - 0.1).abs() < 0.001);
        assert!((trainer.effective_lr(100) - 0.1).abs() < 0.001);
        assert!((trainer.effective_lr(1000) - 0.1).abs() < 0.001);
    }

    #[test]
    fn test_effective_lr_with_decay() {
        let trainer = Trainer::new(0.1, 32, 0.01);
        // epoch 0: 0.1 / 1.0 = 0.1
        assert!((trainer.effective_lr(0) - 0.1).abs() < 0.001);
        // epoch 100: 0.1 / 2.0 = 0.05
        assert!((trainer.effective_lr(100) - 0.05).abs() < 0.001);
        // epoch 900: 0.1 / 10.0 = 0.01
        assert!((trainer.effective_lr(900) - 0.01).abs() < 0.001);
    }

    #[test]
    fn test_training_example() {
        let position = Position::initial();
        let example = TrainingExample {
            position,
            target_score: 10,
        };
        assert_eq!(example.target_score, 10);
    }
}
