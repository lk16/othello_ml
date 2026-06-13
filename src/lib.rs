pub mod eval;
pub mod othello;
pub mod training;

pub use eval::alphabeta::{
    batch_evaluate, best_move, depth_limited_score, exact_score, exact_score_with_nodes, Solver,
};
pub use eval::cache::{build_examples, EvalCache};

/// Micro-benchmark the flip-computation variants (Step 11). See
/// [`othello::flip`].
pub fn bench_flip_variants() {
    othello::flip::bench_variants();
}
pub use othello::board::Board;
pub use othello::game::{load_games, Game, GameResult};
pub use othello::position::Position;
pub use training::features::Features;
pub use training::trainer::{Trainer, TrainingConfig, TrainingExample};
pub use training::weights::Weights;
