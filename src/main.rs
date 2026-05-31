use othello_eval::{
    edax_available, extract_positions, load_games, Board, EdaxInterface, Features, Trainer,
    TrainingExample, Weights,
};
use std::env;
use std::io::Write;

fn main() {
    let args: Vec<String> = env::args().collect();

    // Parse arguments
    // Usage: othello_eval [--max-empties N] <path1> [path2 ...]
    //   paths can be .wtb, .pgn, .txt files or directories (scanned recursively)

    let mut max_empties: u32 = 60; // default: train on all positions (up to 60 empties)
    let mut paths: Vec<String> = Vec::new();
    let mut i = 1;
    while i < args.len() {
        if args[i] == "--max-empties" || args[i] == "-n" {
            i += 1;
            if i < args.len() {
                max_empties = args[i].parse::<u32>().unwrap_or(60);
            }
        } else if args[i] == "--help" || args[i] == "-h" {
            print_usage(&args[0]);
            return;
        } else {
            paths.push(args[i].clone());
        }
        i += 1;
    }

    if paths.is_empty() {
        eprintln!("Error: No input files or directories specified.\n");
        print_usage(&args[0]);
        return;
    }

    println!("=== Othello ML Training ===");
    println!("Max empties: {}", max_empties);
    println!("Input paths: {:?}", paths);

    // Load games from all specified paths
    eprintln!("\n--- Loading games ---");
    let games = match load_games(&paths) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("Error loading games: {}", e);
            return;
        }
    };

    // Extract positions with empties <= max_empties
    eprintln!(
        "\n--- Extracting positions (empties <= {}) ---",
        max_empties
    );
    let positions = extract_positions(&games, max_empties);
    eprintln!("Extracted {} positions", positions.len());

    if positions.is_empty() {
        eprintln!("No positions match the criteria. Exiting.");
        return;
    }

    // Initialize features and weights
    let features = Features::edax();
    eprintln!("Features: {}", features.count());

    let mut weights = Weights::new(features.clone());
    eprintln!(
        "Weight table: {} features x {} empty ranges",
        weights.feature_count(),
        weights.empty_range_count()
    );

    // Create training examples with ground truth evaluations.
    // Uses Edax if available, otherwise falls back to a disc-difference heuristic.
    let use_edax = edax_available();
    let examples: Vec<TrainingExample> = if use_edax {
        eprintln!("\n--- Evaluating positions with Edax ---");
        let mut edax = EdaxInterface::new()
            .expect("EDAX_PATH is set but failed to start Edax");
        let n = positions.len();
        let mut examples = Vec::with_capacity(n);

        for (i, pos) in positions.iter().enumerate() {
            match edax.evaluate(&pos.board, pos.black_to_move) {
                Ok(score) => {
                    examples.push(TrainingExample {
                        board: pos.board,
                        target_score: score,
                    });
                }
                Err(e) => {
                    eprintln!("\nEdax error at position {}: {}", i + 1, e);
                    eprintln!("Falling back to disc-difference heuristic for remaining positions.");
                    // Fall back to heuristic for remaining positions
                    for p in positions.iter().skip(i) {
                        let disc_diff: i32 = if p.black_to_move {
                            p.board.player.count_ones() as i32 - p.board.opponent.count_ones() as i32
                        } else {
                            p.board.opponent.count_ones() as i32 - p.board.player.count_ones() as i32
                        };
                        examples.push(TrainingExample {
                            board: p.board,
                            target_score: disc_diff,
                        });
                    }
                    break;
                }
            }

            // Progress indicator
            if (i + 1) % 100 == 0 || i + 1 == n {
                eprint!("\r  {}/{} positions evaluated", i + 1, n);
                let _ = std::io::stderr().flush();
            }
        }
        let _ = edax.shutdown();
        eprintln!();
        examples
    } else {
        eprintln!("\n--- Using disc-difference heuristic (set EDAX_PATH to use Edax) ---");
        positions
            .iter()
            .map(|pos| {
                let disc_diff: i32 = if pos.black_to_move {
                    pos.board.player.count_ones() as i32 - pos.board.opponent.count_ones() as i32
                } else {
                    pos.board.opponent.count_ones() as i32 - pos.board.player.count_ones() as i32
                };
                TrainingExample {
                    board: pos.board,
                    target_score: disc_diff,
                }
            })
            .collect()
    };

    eprintln!("Training examples: {}", examples.len());

    // Train
    eprintln!("\n--- Training ---");
    let trainer = Trainer::new(0.01, 32);
    trainer.train_epochs(&mut weights, &examples, 50);

    // Show some learned weights for corner features
    eprintln!("\n--- Sample learned weights (feature 0 = A1 corner, empty=60) ---");
    let board = Board::initial();
    let feature_indices = features.extract(&board);
    for (feat_idx, &pattern_idx) in feature_indices.iter().enumerate().take(10) {
        let w = weights.get_weight(feat_idx, pattern_idx, 60);
        if w != 0.0 {
            eprintln!(
                "  Feature {} pattern {}: weight = {}",
                feat_idx, pattern_idx, w
            );
        }
    }

    // Save weights
    eprintln!("\n--- Saving weights ---");
    match othello_eval::io::WeightIO::save(&weights, "trained_weights.bin") {
        Ok(()) => eprintln!("Weights saved to trained_weights.bin"),
        Err(e) => eprintln!("Error saving weights: {}", e),
    }

    eprintln!("\nDone!");
}

fn print_usage(program: &str) {
    eprintln!("Usage: {} [OPTIONS] <path>...", program);
    eprintln!();
    eprintln!("Train Othello evaluation weights from game files.");
    eprintln!();
    eprintln!("OPTIONS:");
    eprintln!(
        "  -n, --max-empties N   Only train on positions with <= N empty cells (default: 60)"
    );
    eprintln!("  -h, --help            Show this help message");
    eprintln!();
    eprintln!("INPUT:");
    eprintln!("  One or more paths to:");
    eprintln!("    - .wtb files (WTHOR binary format)");
    eprintln!("    - .pgn / .txt files (PGN text format, PlayOK variant)");
    eprintln!("    - directories (scanned recursively for game files)");
    eprintln!();
    eprintln!("EXAMPLES:");
    eprintln!("  {} training_data/wthor/", program);
    eprintln!("  {} --max-empties 20 game.txt training_data/", program);
    eprintln!("  {} -n 30 ~/Downloads/lk16.txt", program);
}
