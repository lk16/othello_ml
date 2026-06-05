pub mod eval;
pub mod othello;
pub mod training;

pub use eval::alphabeta::{batch_evaluate, exact_score};
pub use eval::cache::{build_examples, EvalCache};
pub use othello::board::Board;
pub use othello::game::{load_games, Game, GameResult};
pub use othello::position::Position;
pub use training::features::Features;
pub use training::trainer::{Trainer, TrainingConfig, TrainingExample};
pub use training::weights::Weights;
