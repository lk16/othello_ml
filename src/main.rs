use othello_eval::{
    edax_available, extract_positions, load_games, EdaxInterface, EvalCache, Features, Position,
    Trainer, TrainingConfig, TrainingExample, Weights,
};
use std::env;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

fn main() {
    let args: Vec<String> = env::args().collect();

    let mut max_empties: u32 = 60; // default: train on all positions (up to 60 empties)
    let mut epochs: usize = 10; // default: 10 training epochs
    let mut lr_decay: f32 = 0.01; // default: inverse-time decay (0 = no decay)
    let mut resume_epoch: usize = 0; // default: start from epoch 0 for LR schedule
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
        } else if args[i] == "--lr-decay" || args[i] == "-d" {
            i += 1;
            if i < args.len() {
                lr_decay = args[i].parse::<f32>().unwrap_or(0.01);
            }
        } else if args[i] == "--resume-epoch" || args[i] == "-r" {
            i += 1;
            if i < args.len() {
                resume_epoch = args[i].parse::<usize>().unwrap_or(0);
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
    println!("Max empties: {max_empties}");
    println!("Epochs: {epochs}");
    if resume_epoch > 0 {
        println!("Resume epoch: {resume_epoch} (LR schedule continues from here)");
    }
    if edax_available() || eval_file.is_some() {
        println!("Edax level: {edax_level}");
    }
    println!("Input paths: {paths:?}");

    // Load games from all specified paths
    eprintln!("\n--- Loading games ---");
    let games = match load_games(&paths) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("Error loading games: {e}");
            return;
        }
    };

    // Extract positions with empties <= max_empties
    eprintln!("\n--- Extracting positions (empties <= {max_empties}) ---");
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
        eprintln!("Loading weights from {weights_file} ...");
        match othello_eval::Weights::load(&weights_file) {
            Ok(w) => {
                eprintln!(
                    "Loaded weights: {} features x {} empty ranges",
                    w.feature_count(),
                    w.empty_range_count()
                );
                w
            }
            Err(e) => {
                eprintln!("Error loading weights (starting fresh): {e}");
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

    let mut examples = if let Some(ref path) = eval_file {
        let cache = EvalCache::new(path.clone());
        cache.build_examples(&positions, edax_level, &edax_path, edax_threads)
    } else {
        eprintln!("\n--- Evaluating positions with Edax (level {edax_level}) ---");
        let n = positions.len();
        eprintln!("Submitting {n} positions to Edax...");
        let eval_start = std::time::Instant::now();
        let boards: Vec<Position> = positions.iter().map(|p| p.position).collect();
        let scores = EdaxInterface::batch_evaluate(&boards, edax_level, &edax_path, edax_threads)
            .unwrap_or_else(|e| {
                eprintln!("Edax evaluation failed: {e}");
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
                position: pos.position,
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
    // Learning rate = 0.1 with gradient normalization (gradient / N_features).
    // Effective per-example prediction correction ≈ lr × 2 = 20%.
    // Inverse-time decay: effective_lr = lr / (1 + decay × epoch).
    let trainer = Trainer::new(0.1, 32, lr_decay);
    let train_config = TrainingConfig {
        epochs,
        epoch_offset: resume_epoch,
        interrupt: Some(interrupted),
    };
    trainer.train_epochs(&mut weights, &mut examples, &train_config);

    // Show some learned weights for corner features
    eprintln!("\n--- Sample learned weights (feature 0 = A1 corner, empty=60) ---");
    let board = Position::initial();
    let feature_indices = features.extract(&board);
    for (feat_idx, &pattern_idx) in feature_indices.iter().enumerate().take(10) {
        let w = weights.get_weight(feat_idx, pattern_idx, 60);
        if w != 0.0 {
            eprintln!("  Feature {feat_idx} pattern {pattern_idx}: weight = {w}");
        }
    }

    // Save weights
    eprintln!("\n--- Saving weights ---");
    match weights.save(&weights_file) {
        Ok(()) => eprintln!("Weights saved to {weights_file}"),
        Err(e) => eprintln!("Error saving weights: {e}"),
    }

    eprintln!("\nDone!");
}

fn print_usage(program: &str) {
    eprintln!("Usage: {program} [OPTIONS] <path>...");
    eprintln!();
    eprintln!("Train Othello evaluation weights from game files.");
    eprintln!();
    eprintln!("Requires Edax (set EDAX_PATH) for ground truth evaluations.");
    eprintln!();
    eprintln!("OPTIONS:");
    eprintln!(
        "  -n, --max-empties N   Only train on positions with <= N empty cells (default: 60)"
    );
    eprintln!("  -e, --epochs N        Number of training epochs (default: 10)");
    eprintln!("  -l, --level N         Edax search level, 0-60 even (default: 10)");
    eprintln!("  -t, --edax-threads N  Parallel Edax processes (default: 1)");
    eprintln!(
        "  -f, --eval-file PATH  Eval cache (load if exists, compute+append missing, create if not)"
    );
    eprintln!("  -w, --weights PATH    Weights output file (default: trained_weights.bin)");
    eprintln!("  -d, --lr-decay F      Inverse-time LR decay factor (default: 0.01, 0 = no decay)");
    eprintln!("  -r, --resume-epoch N  Resume LR schedule from this epoch (default: 0)");
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
    eprintln!("  EDAX_PATH=../edax {program} training_data/");
    eprintln!("  EDAX_PATH=../edax {program} --max-empties 20 --epochs 50 training_data/");
    eprintln!(
        "  EDAX_PATH=../edax {program} --eval-file ignored/evals.txt --epochs 30 training_data/"
    );
    eprintln!("  EDAX_PATH=../edax {program} -e 1000 -d 0.001 -f evals.txt training_data/");
    eprintln!("  EDAX_PATH=../edax {program} -e 500 -r 1000 -w trained_weights.bin training_data/");
    eprintln!("    (resume from epoch 1000, LR schedule continues from there)");
}
