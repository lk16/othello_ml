use crate::board::Board;
use crate::weights::Weights;

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

impl Trainer {
    pub fn new(learning_rate: f32, batch_size: usize) -> Self {
        Trainer {
            learning_rate,
            batch_size,
        }
    }

    /// Train weights on a batch of examples
    pub fn train_batch(&self, weights: &mut Weights, examples: &[TrainingExample]) {
        let features = weights.features().clone();

        for example in examples {
            // Forward pass: compute prediction
            let predicted = weights.evaluate(&example.board, &features) as i32;
            let error = example.target_score - predicted;

            // Backward pass: update each feature weight
            let feature_indices = features.extract(&example.board);
            for (feat_idx, &pattern_idx) in feature_indices.iter().enumerate() {
                // Simple SGD: gradient is the error
                // We want to minimize (predicted - target)^2
                // So dL/dw = 2 * (predicted - target) * 1 = 2 * error
                let gradient = 2.0 * error as f32;

                weights.update_weight_sgd(
                    feat_idx,
                    pattern_idx,
                    example.board.empties(),
                    self.learning_rate,
                    -gradient, // Negative because we want to reduce error
                );
            }
        }
    }

    /// Train for multiple epochs
    pub fn train_epochs(
        &self,
        weights: &mut Weights,
        examples: &[TrainingExample],
        epochs: usize,
    ) {
        for epoch in 0..epochs {
            let mut loss = 0.0;

            // Process in mini-batches
            for chunk in examples.chunks(self.batch_size) {
                self.train_batch(weights, chunk);

                // Compute loss for this batch
                let features = weights.features();
                for example in chunk {
                    let predicted = weights.evaluate(&example.board, features) as i32;
                    let error = example.target_score - predicted;
                    loss += (error * error) as f32;
                }
            }

            if epoch % 10 == 0 {
                eprintln!("Epoch {}: loss = {}", epoch, loss / examples.len() as f32);
            }
        }
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
