pub mod othello;
pub mod training;

pub use othello::board::Board;
pub use othello::game::{load_games, Game};
pub use othello::positions::{extract_positions, Position};
pub use training::edax::{board_to_fen, edax_available, EdaxInterface};
pub use training::features::Features;
pub use training::training::{Trainer, TrainingExample};
pub use training::weights::Weights;
