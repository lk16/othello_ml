use crate::othello::position::Position;
use crate::training::weights::Weights;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

/// A training example "compiled" once for the epoch loop: the feature-pattern
/// indices are extracted a single time (positions are fixed across epochs) and
/// reused every epoch, eliminating the per-epoch `Features::extract` `Vec`
/// alloc and per-cell `get_cell` loop. Shuffled together so indices stay paired
/// with their target.
struct CompiledExample {
    indices: Vec<u32>,
    empties: u32,
    target: f32,
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
    /// Feature indices are extracted **once** up front ([`CompiledExample`]) and
    /// reused every epoch — positions are fixed, so re-extracting per epoch (and
    /// the `Vec`/`get_cell` cost) is pure waste.
    ///
    /// Single-threaded (`threads <= 1`) runs online SGD **in place** over the
    /// compiled examples — no per-batch weight clone or full-table merge (the
    /// old path cloned the ~107 MB table and scanned all weights per 32-example
    /// batch, even single-threaded). With `threads > 1`, each thread trains its
    /// shard on a cloned copy and the weight deltas are averaged **once per
    /// epoch** (model-averaging SGD), so the clone/merge cost is paid `threads`
    /// times per epoch instead of once per batch.
    pub fn train_epochs(
        &self,
        weights: &mut Weights,
        examples: &[TrainingExample],
        config: &TrainingConfig,
    ) {
        let epochs = config.epochs;
        let epoch_offset = config.epoch_offset;
        let n_examples = examples.len();
        let total_updates = n_examples * weights.feature_count() * epochs;
        let n_threads = config.threads.max(1);

        // Compile once: extract feature indices for every example a single time.
        let features = weights.features().clone();
        let n_features = features.count() as f32;
        let mut compiled: Vec<CompiledExample> = examples
            .iter()
            .map(|ex| CompiledExample {
                indices: features.extract(&ex.position),
                empties: ex.position.empties(),
                target: ex.target_score as f32,
            })
            .collect();

        eprintln!(
            "Training: {} epochs × {} examples (online SGD, batch_size={})",
            epochs, n_examples, self.batch_size
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
            shuffle(&mut compiled, &mut rng);

            let current_lr = self.effective_lr(global_epoch);

            let epoch_start = Instant::now();

            let loss: f64 = if n_threads <= 1 {
                train_shard_in_place(weights, &compiled, n_features, current_lr)
            } else {
                // Model-averaging SGD: split the epoch across threads, each
                // training a shard on its own clone; average the deltas once.
                let shard = n_examples.div_ceil(n_threads);
                let results: Vec<(Weights, f64)> = std::thread::scope(|s| {
                    let handles: Vec<_> = compiled
                        .chunks(shard)
                        .map(|part| {
                            let mut w = weights.clone();
                            s.spawn(move || {
                                let loss =
                                    train_shard_in_place(&mut w, part, n_features, current_lr);
                                (w, loss)
                            })
                        })
                        .collect();
                    handles.into_iter().map(|h| h.join().unwrap()).collect()
                });

                let mut loss = 0.0;
                let mut worker_weights = Vec::with_capacity(results.len());
                for (w, l) in results {
                    loss += l;
                    worker_weights.push(w);
                }
                weights.merge_from_workers(&worker_weights);
                loss
            };

            let epoch_elapsed = epoch_start.elapsed();
            let total_elapsed = total_start.elapsed();
            let avg_loss = loss / n_examples as f64;
            let throughput = n_examples as f64 / epoch_elapsed.as_secs_f64().max(0.001);

            let avg_epoch_secs = total_elapsed.as_secs_f64() / (epoch + 1) as f64;
            let remaining_secs = avg_epoch_secs * (epochs - epoch - 1) as f64;

            eprintln!(
                "Epoch {}/{} (global {}) | loss: {:.4} | lr: {:.4} | time: {:.1}s | {:.0} ex/s | ETA: {:.0}s",
                epoch + 1, epochs, global_epoch + 1, avg_loss, current_lr, epoch_elapsed.as_secs_f64(), throughput, remaining_secs
            );

            last_loss = avg_loss;
            completed = epoch + 1;
        }

        let total_secs = total_start.elapsed().as_secs_f64();
        let overall_throughput = (n_examples * completed) as f64 / total_secs.max(0.001);
        eprintln!(
            "Complete | loss: {last_loss:.4} | time: {total_secs:.1}s | {overall_throughput:.0} ex/s"
        );
    }
}

/// Run online SGD over one shard of compiled examples, mutating `weights` in
/// place and returning the accumulated squared error. Shared by the
/// single-threaded path and the per-thread workers.
fn train_shard_in_place(
    weights: &mut Weights,
    shard: &[CompiledExample],
    n_features: f32,
    lr: f32,
) -> f64 {
    let mut loss = 0.0f64;
    for ce in shard {
        let predicted = weights.evaluate_indices(&ce.indices, ce.empties);
        let error = ce.target - predicted;
        loss += (error as f64) * (error as f64);
        let gradient = 2.0 * error / n_features;
        weights.sgd_step_indices(&ce.indices, ce.empties, lr, gradient);
    }
    loss
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
