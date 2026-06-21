pub mod eval;
pub mod gui;
pub mod othello;
pub mod training;

pub use gui::{run as run_gui, GuiArgs, GuiMode};

pub use eval::alphabeta::{
    batch_evaluate, best_move, bootstrap_score, depth_limited_score, exact_score,
    exact_score_with_nodes, ParallelSolver, Solver,
};
pub use eval::cache::{build_examples, EvalCache};
pub use eval::pattern::FlatEval;

/// Micro-benchmark the flip-computation variants (Step 11). See
/// [`othello::flip`].
pub fn bench_flip_variants() {
    othello::flip::bench_variants();
}

/// Micro-benchmark the count-last-flip variants (Step 23). See
/// [`eval::alphabeta::bench_count_flip_variants`].
pub fn bench_count_flip_variants() {
    eval::alphabeta::bench_count_flip_variants();
}

/// Micro-benchmark the mobility (`get_moves`) variants (Step 24). See
/// [`othello::get_moves`].
pub fn bench_get_moves_variants() {
    othello::get_moves::bench_get_moves_variants();
}
pub use othello::board::Board;
pub use othello::game::{load_games, Game, GameResult};
pub use othello::position::Position;
pub use training::cg::{train_least_squares, CgConfig};
pub use training::features::Features;
pub use training::weights::Weights;
pub use training::TrainingExample;
