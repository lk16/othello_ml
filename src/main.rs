use othello_eval::{Board, Features, Weights};

fn main() {
    println!("Othello Evaluator - Weight Training System");

    // Test basic board operations
    let board = Board::initial();
    println!("Initial board - Player: {}, Opponent: {}, Empties: {}",
        board.player_discs(),
        board.opponent_discs(),
        board.empties()
    );

    // Test features
    let features = Features::edax();
    println!("Loaded {} Edax features", features.count());

    // Test weights
    let weights = Weights::new(features);
    println!("Created weight table with {} features and {} empty ranges",
        weights.feature_count(),
        weights.empty_range_count()
    );

    println!("Ready for training!");
}
