pub mod othello;
pub mod training;

pub use othello::board::Board;
pub use othello::game::{load_games, Game};
pub use othello::position::Position;
pub use training::edax::{edax_available, EdaxInterface};
pub use training::eval_cache::{build_examples, EvalCache};
pub use training::features::Features;
pub use training::trainer::{Trainer, TrainingConfig, TrainingExample};
pub use training::weights::Weights;
