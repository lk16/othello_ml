use crate::board::Board;
use crate::weights::Weights;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

/// Training data point: a board position paired with its ground truth evaluation.
#[derive(Debug, Clone)]
pub struct TrainingExample {
    pub board: Board,
    pub target_score: i32, // Ground truth score from Edax
}

/// SGD (Stochastic Gradient Descent) trainer for optimizing Othello position weights.
///
/// Training process:
/// 1. Forward pass: evaluate board with current weights
/// 2. Compute error: target_score - predicted_score
/// 3. Backward pass: for each feature contributing to the prediction,
///    update its weight: w = w - learning_rate * gradient
/// 4. Repeat for multiple epochs over training data
pub struct Trainer {
    learning_rate: f32,  // Step size for weight updates
    batch_size: usize,   // Number of examples per training batch
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

impl Trainer {
    pub fn new(learning_rate: f32, batch_size: usize) -> Self {
        Trainer {
            learning_rate,
            batch_size,
        }
    }

    /// Train weights on a batch of examples, returning the accumulated squared error.
    pub fn train_batch(&self, weights: &mut Weights, examples: &[TrainingExample]) -> f64 {
        let features = weights.features().clone();
        let n_features = features.count() as f32;
        let mut loss: f64 = 0.0;

        for example in examples {
            // Forward pass: compute prediction
            let predicted = weights.evaluate(&example.board, &features);
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

            let feature_indices = features.extract(&example.board);
            for (feat_idx, &pattern_idx) in feature_indices.iter().enumerate() {
                weights.update_weight_sgd(
                    feat_idx,
                    pattern_idx,
                    example.board.empties(),
                    self.learning_rate,
                    gradient,
                );
            }
        }
        loss
    }

    /// Train for multiple epochs with progress logging.
    ///
    /// Examples are shuffled at the start of each epoch to avoid systematic
    /// ordering biases.  If `interrupt` is provided, the loop checks the flag
    /// at the start of each epoch and returns early when it is set, preserving
    /// weights from the last fully completed epoch.
    pub fn train_epochs(
        &self,
        weights: &mut Weights,
        examples: &mut [TrainingExample],
        epochs: usize,
        interrupt: Option<&AtomicBool>,
    ) {
        use std::io::{self, Write};

        let n_examples = examples.len();
        let n_batches = (n_examples + self.batch_size - 1) / self.batch_size;
        let total_updates = n_examples * weights.feature_count() * epochs;

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
        eprintln!();

        let total_start = Instant::now();

        let mut completed: usize = 0;
        let mut last_loss: f64 = 0.0;

        for epoch in 0..epochs {
            if let Some(flag) = interrupt {
                if flag.load(Ordering::Relaxed) {
                    eprintln!(
                        "\nInterrupted after {} epochs — keeping weights from last completed epoch.",
                        completed
                    );
                    break;
                }
            }

            // Shuffle examples at the start of each epoch to break ordering
            // biases.  Use a deterministic seed per epoch so runs are reproducible.
            let mut rng = XorShift32::new(epoch as u32);
            shuffle(examples, &mut rng);

            let epoch_start = Instant::now();
            let mut loss: f64 = 0.0;

            // Process in mini-batches
            for (batch_idx, chunk) in examples.chunks(self.batch_size).enumerate() {
                loss += self.train_batch(weights, chunk);

                // Intra-epoch progress: show every 10% of batches
                let progress_pct = (batch_idx + 1) * 100 / n_batches;
                if progress_pct % 10 == 0 || batch_idx == n_batches - 1 {
                    let elapsed = epoch_start.elapsed();
                    let done = (batch_idx + 1) * self.batch_size;
                    let throughput = done as f64 / elapsed.as_secs_f64().max(0.001);
                    eprint!(
                        "\r  [{:3}%] batch {}/{} ({:.0} ex/s)          ",
                        progress_pct, batch_idx + 1, n_batches, throughput
                    );
                    let _ = io::stderr().flush();
                }
            }

            let epoch_elapsed = epoch_start.elapsed();
            let total_elapsed = total_start.elapsed();
            let avg_loss = loss / n_examples as f64;
            let throughput = n_examples as f64 / epoch_elapsed.as_secs_f64().max(0.001);

            // Estimate time remaining
            let avg_epoch_secs = total_elapsed.as_secs_f64() / (epoch + 1) as f64;
            let remaining_secs = avg_epoch_secs * (epochs - epoch - 1) as f64;

            eprintln!(
                "\rEpoch {}/{} | loss: {:.4} | time: {:.1}s | {:.0} ex/s | ETA: {:.0}s   ",
                epoch + 1, epochs, avg_loss, epoch_elapsed.as_secs_f64(), throughput, remaining_secs
            );

            last_loss = avg_loss;
            completed = epoch + 1;
        }

        let total_secs = total_start.elapsed().as_secs_f64();
        let overall_throughput = (n_examples * completed) as f64 / total_secs.max(0.001);
        eprintln!(
            "\rComplete        | loss: {:.4} | time: {:.1}s | {:.0} ex/s   ",
            last_loss, total_secs, overall_throughput
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trainer_creation() {
        let trainer = Trainer::new(0.01, 32);
        assert_eq!(trainer.learning_rate, 0.01);
        assert_eq!(trainer.batch_size, 32);
    }

    #[test]
    fn test_training_example() {
        let board = Board::initial();
        let example = TrainingExample {
            board,
            target_score: 10,
        };
        assert_eq!(example.target_score, 10);
    }
}
