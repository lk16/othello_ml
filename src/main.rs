use othello_eval::{
    board_to_fen, edax_available, extract_positions, load_games, Board, EdaxInterface, Features,
    Trainer, TrainingExample, Weights,
};
use std::collections::HashMap;
use std::env;
use std::fs;
use std::io::Write;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

fn main() {
    let args: Vec<String> = env::args().collect();

    let mut max_empties: u32 = 60; // default: train on all positions (up to 60 empties)
    let mut epochs: usize = 10; // default: 10 training epochs
    let mut eval_file: Option<String> = None;
    let mut weights_file: String = String::from("trained_weights.bin");
    let mut edax_level: u32 = 10; // default: Edax search level (0-60, even)
    let mut edax_threads: usize = 1; // default: single Edax process
    let mut paths: Vec<String> = Vec::new();
    let mut i = 1;
    while i < args.len() {
        if args[i] == "--max-empties" || args[i] == "-n" {
            i += 1;
            if i < args.len() {
                max_empties = args[i].parse::<u32>().unwrap_or(60);
            }
        } else if args[i] == "--epochs" || args[i] == "-e" {
            i += 1;
            if i < args.len() {
                epochs = args[i].parse::<usize>().unwrap_or(10);
            }
        } else if args[i] == "--eval-file" || args[i] == "-f" {
            i += 1;
            if i < args.len() {
                eval_file = Some(args[i].clone());
            }
        } else if args[i] == "--level" || args[i] == "-l" {
            i += 1;
            if i < args.len() {
                edax_level = args[i].parse::<u32>().unwrap_or(10);
            }
        } else if args[i] == "--edax-threads" || args[i] == "-t" {
            i += 1;
            if i < args.len() {
                edax_threads = args[i].parse::<usize>().unwrap_or(1);
            }
        } else if args[i] == "--weights" || args[i] == "-w" {
            i += 1;
            if i < args.len() {
                weights_file = args[i].clone();
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
    println!("Epochs: {}", epochs);
    if edax_available() || eval_file.is_some() {
        println!("Edax level: {}", edax_level);
    }
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

    // Initialize features and weights (load from file if present)
    let features = Features::edax();
    eprintln!("Features: {}", features.count());

    let mut weights = if std::path::Path::new(&weights_file).exists() {
        eprintln!("Loading weights from {} ...", weights_file);
        match othello_eval::io::WeightIO::load(&weights_file) {
            Ok(w) => {
                eprintln!(
                    "Loaded weights: {} features x {} empty ranges",
                    w.feature_count(),
                    w.empty_range_count()
                );
                w
            }
            Err(e) => {
                eprintln!("Error loading weights (starting fresh): {}", e);
                Weights::new(features.clone())
            }
        }
    } else {
        let w = Weights::new(features.clone());
        eprintln!(
            "Weight table: {} features x {} empty ranges",
            w.feature_count(),
            w.empty_range_count()
        );
        w
    };

    // Require Edax — all evaluations use it for ground truth.
    if !edax_available() {
        eprintln!("Error: Edax is required. Set EDAX_PATH to the Edax binary.");
        std::process::exit(1);
    }
    let edax_path =
        env::var("EDAX_PATH").expect("EDAX_PATH should be set (checked by edax_available)");

    // Create training examples.
    // --eval-file acts as a cache: load if present, compute+append missing, create if not.
    let examples: Vec<TrainingExample> = if let Some(ref eval_path) = eval_file {
        if std::path::Path::new(eval_path).exists() {
            // Cache hit — load, compute missing with Edax, append to file
            eprintln!("\n--- Loading evaluations from {} ---", eval_path);
            let eval_map = load_eval_file(eval_path).unwrap_or_else(|e| {
                eprintln!("Error loading eval file: {}", e);
                std::process::exit(1);
            });
            eprintln!("Loaded {} evaluations", eval_map.len());

            let mut examples = Vec::with_capacity(positions.len());
            let mut missing_positions: Vec<&othello_eval::Position> = Vec::new();
            for pos in &positions {
                let fen = board_to_fen(&pos.board, pos.black_to_move);
                match eval_map.get(&fen) {
                    Some(&score) => {
                        examples.push(TrainingExample {
                            board: pos.board,
                            target_score: score,
                        });
                    }
                    None => {
                        missing_positions.push(pos);
                    }
                }
            }

            if !missing_positions.is_empty() {
                let n_missing = missing_positions.len();
                eprintln!(
                    "Computing {} missing positions with Edax (level {})...",
                    n_missing, edax_level
                );
                let missing_boards: Vec<Board> =
                    missing_positions.iter().map(|p| p.board).collect();
                let scores = EdaxInterface::batch_evaluate(
                    &missing_boards,
                    edax_level,
                    &edax_path,
                    edax_threads,
                )
                .unwrap_or_else(|e| {
                    eprintln!("Edax evaluation failed: {}", e);
                    std::process::exit(1);
                });

                append_eval_file(eval_path, &missing_positions, &scores)
                    .unwrap_or_else(|e| {
                        eprintln!("Error appending to eval file: {}", e);
                        std::process::exit(1);
                    });
                eprintln!("Appended {} new evaluations to {}", n_missing, eval_path);

                for (pos, &score) in missing_positions.iter().zip(scores.iter()) {
                    examples.push(TrainingExample {
                        board: pos.board,
                        target_score: score,
                    });
                }
            }
            examples
        } else {
            // Cache miss — compute everything and save
            eprintln!(
                "\n--- Evaluating positions with Edax (level {}) → saving to {} ---",
                edax_level, eval_path
            );
            let n = positions.len();
            eprintln!("Submitting {} positions to Edax...", n);

            let eval_start = std::time::Instant::now();
            let boards: Vec<Board> = positions.iter().map(|p| p.board).collect();
            let scores =
                EdaxInterface::batch_evaluate(&boards, edax_level, &edax_path, edax_threads).unwrap_or_else(
                    |e| {
                        eprintln!("Edax evaluation failed: {}", e);
                        std::process::exit(1);
                    },
                );

            let elapsed = eval_start.elapsed();
            eprintln!(
                "  Done in {:.1}s ({:.0} pos/s)",
                elapsed.as_secs_f64(),
                n as f64 / elapsed.as_secs_f64().max(0.001)
            );

            let examples: Vec<TrainingExample> = positions
                .iter()
                .zip(scores.iter())
                .map(|(pos, &score)| TrainingExample {
                    board: pos.board,
                    target_score: score,
                })
                .collect();

            eprintln!("Saving evaluations to {} ...", eval_path);
            save_eval_from_positions(eval_path, &positions, &examples).unwrap_or_else(|e| {
                eprintln!("Error saving eval file: {}", e);
                std::process::exit(1);
            });
            eprintln!("Saved {} evaluations", examples.len());
            examples
        }
    } else {
        // No eval file — compute with Edax (no caching)
        eprintln!("\n--- Evaluating positions with Edax (level {}) ---", edax_level);
        let n = positions.len();
        eprintln!("Submitting {} positions to Edax...", n);

        let eval_start = std::time::Instant::now();
        let boards: Vec<Board> = positions.iter().map(|p| p.board).collect();
        let scores =
            EdaxInterface::batch_evaluate(&boards, edax_level, &edax_path, edax_threads).unwrap_or_else(|e| {
                eprintln!("Edax evaluation failed: {}", e);
                std::process::exit(1);
            });

        let elapsed = eval_start.elapsed();
        eprintln!(
            "  Done in {:.1}s ({:.0} pos/s)",
            elapsed.as_secs_f64(),
            n as f64 / elapsed.as_secs_f64().max(0.001)
        );

        positions
            .iter()
            .zip(scores.iter())
            .map(|(pos, &score)| TrainingExample {
                board: pos.board,
                target_score: score,
            })
            .collect()
    };

    eprintln!("Training examples: {}", examples.len());

    // Train
    eprintln!("\n--- Training ---");
    eprintln!("(press Ctrl+C to stop early and save weights)");
    let interrupted = Arc::new(AtomicBool::new(false));
    {
        let interrupted = Arc::clone(&interrupted);
        ctrlc::set_handler(move || {
            eprintln!("\nInterrupt received — finishing current epoch...");
            interrupted.store(true, Ordering::Relaxed);
        })
        .expect("Failed to set Ctrl+C handler");
    }
    let trainer = Trainer::new(0.01, 32);
    trainer.train_epochs(&mut weights, &examples, epochs, Some(&interrupted));

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
    match othello_eval::io::WeightIO::save(&weights, &weights_file) {
        Ok(()) => eprintln!("Weights saved to {}", weights_file),
        Err(e) => eprintln!("Error saving weights: {}", e),
    }

    eprintln!("\nDone!");
}

fn print_usage(program: &str) {
    eprintln!("Usage: {} [OPTIONS] <path>...", program);
    eprintln!();
    eprintln!("Train Othello evaluation weights from game files.");
    eprintln!();
    eprintln!("Requires Edax (set EDAX_PATH) for ground truth evaluations.");
    eprintln!();
    eprintln!("OPTIONS:");
    eprintln!(
        "  -n, --max-empties N   Only train on positions with <= N empty cells (default: 60)"
    );
    eprintln!(
        "  -e, --epochs N        Number of training epochs (default: 10)"
    );
    eprintln!(
        "  -l, --level N         Edax search level, 0-60 even (default: 10)"
    );
    eprintln!(
        "  -t, --edax-threads N  Parallel Edax processes (default: 1)"
    );
    eprintln!(
        "  -f, --eval-file PATH  Eval cache (load if exists, compute+append missing, create if not)"
    );
    eprintln!(
        "  -w, --weights PATH    Weights output file (default: trained_weights.bin)"
    );
    eprintln!("  -h, --help            Show this help message");
    eprintln!();
    eprintln!("EVAL FILE FORMAT:");
    eprintln!("  Each line: <Edax FEN> <score>");
    eprintln!("  FEN is 66 chars (64 board cells + space + side to move).");
    eprintln!();
    eprintln!("INPUT:");
    eprintln!("  One or more paths to:");
    eprintln!("    - .wtb files (WTHOR binary format)");
    eprintln!("    - .pgn / .txt files (PGN text format, PlayOK variant)");
    eprintln!("    - directories (scanned recursively for game files)");
    eprintln!();
    eprintln!("EXAMPLES:");
    eprintln!(
        "  EDAX_PATH=../edax {} training_data/",
        program
    );
    eprintln!(
        "  EDAX_PATH=../edax {} --max-empties 20 --epochs 50 training_data/",
        program
    );
    eprintln!(
        "  EDAX_PATH=../edax {} --eval-file ignored/evals.txt --epochs 30 training_data/",
        program
    );
}

// ─── Eval file load / save ──────────────────────────────────────────

/// Load an eval file into a map from FEN string to score.
///
/// Format: one `<FEN> <score>` pair per line. The FEN is 66 characters
/// (64 board + space + side to move), the score is a signed integer.
fn load_eval_file(path: &str) -> Result<HashMap<String, i32>, String> {
    let content = fs::read_to_string(path)
        .map_err(|e| format!("Failed to read {}: {}", path, e))?;
    let mut map = HashMap::new();

    for (line_no, line) in content.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        // FEN is exactly 66 chars; score is the remainder
        if line.len() < 68 {
            return Err(format!(
                "{}:{}: line too short (expected '<66-char FEN> <score>')",
                path,
                line_no + 1
            ));
        }
        let fen = &line[..66];
        let score_str = line[67..].trim();
        let score = score_str
            .parse::<i32>()
            .map_err(|e| format!("{}:{}: invalid score '{}': {}", path, line_no + 1, score_str, e))?;

        if fen.as_bytes()[64] != b' ' {
            return Err(format!(
                "{}:{}: FEN missing space at position 65",
                path,
                line_no + 1
            ));
        }

        map.insert(fen.to_string(), score);
    }

    Ok(map)
}

/// Append evaluations for a subset of positions to an existing eval file.
fn append_eval_file(
    path: &str,
    positions: &[&othello_eval::Position],
    scores: &[i32],
) -> Result<(), String> {
    let mut file = fs::OpenOptions::new()
        .append(true)
        .open(path)
        .map_err(|e| format!("Failed to open {} for appending: {}", path, e))?;
    for (pos, &score) in positions.iter().zip(scores.iter()) {
        let fen = board_to_fen(&pos.board, pos.black_to_move);
        writeln!(file, "{} {}", fen, score)
            .map_err(|e| format!("Failed to write eval file: {}", e))?;
    }
    Ok(())
}

/// Save evaluations to a file from positions and examples.
///
/// Uses positions to recover the `black_to_move` flag needed for FEN generation.
fn save_eval_from_positions(
    path: &str,
    positions: &[othello_eval::Position],
    examples: &[TrainingExample],
) -> Result<(), String> {
    let mut file = fs::File::create(path)
        .map_err(|e| format!("Failed to create {}: {}", path, e))?;
    for (pos, ex) in positions.iter().zip(examples.iter()) {
        let fen = board_to_fen(&pos.board, pos.black_to_move);
        writeln!(file, "{} {}", fen, ex.target_score)
            .map_err(|e| format!("Failed to write eval file: {}", e))?;
    }
    Ok(())
}

