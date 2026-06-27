pub mod cg;
pub mod features;
pub mod selfplay;
pub mod weights;

pub use cg::{train_least_squares, CgConfig};
pub use features::Features;
pub use selfplay::{generate_examples, SelfPlayConfig};
pub use weights::Weights;

use crate::othello::position::Position;

/// Training data point: a board position paired with its ground truth evaluation.
#[derive(Debug, Clone)]
pub struct TrainingExample {
    pub position: Position,
    pub target_score: i32, // Ground truth score (exact solve / Edax)
}
