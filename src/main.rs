use othello_eval::{build_examples, load_games, Features, Trainer, TrainingConfig, Weights};
use std::env;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

/// Parsed command-line arguments.
struct CliArgs {
    max_empties: u32,
    epochs: usize,
    lr_decay: f32,
    resume_epoch: usize,
    eval_file: Option<String>,
    weights_file: String,
    paths: Vec<String>,
}

/// Parse CLI arguments. Returns `None` if `--help` was shown.
fn parse_args() -> Option<CliArgs> {
    let args: Vec<String> = env::args().collect();

    let mut max_empties: u32 = 60;
    let mut epochs: usize = 10;
    let mut lr_decay: f32 = 0.01;
    let mut resume_epoch: usize = 0;
    let mut eval_file: Option<String> = None;
    let mut weights_file: String = String::from("trained_weights.bin");
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
            return None;
        } else {
            paths.push(args[i].clone());
        }
        i += 1;
    }

    Some(CliArgs {
        max_empties,
        epochs,
        lr_decay,
        resume_epoch,
        eval_file,
        weights_file,
        paths,
    })
}

fn main() {
    let args = if let Some(a) = parse_args() {
        a
    } else {
        return;
    };

    if args.paths.is_empty() {
        eprintln!("Error: No input files or directories specified.\n");
        print_usage(&env::args().collect::<Vec<_>>()[0]);
        return;
    }

    eprintln!("=== Othello ML Training ===");
    eprintln!("Max empties: {}", args.max_empties);
    eprintln!("Epochs: {}", args.epochs);
    if args.resume_epoch > 0 {
        eprintln!(
            "Resume epoch: {} (LR schedule continues from here)",
            args.resume_epoch
        );
    }
    eprintln!("Input paths: {:?}", args.paths);

    // Load games
    eprintln!("\n--- Loading games ---");
    let games = match load_games(&args.paths) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("Error loading games: {e}");
            return;
        }
    };

    // Extract positions
    eprintln!(
        "\n--- Extracting positions (empties <= {}) ---",
        args.max_empties
    );
    let positions: Vec<othello_eval::Board> = games
        .iter()
        .flat_map(|game| game.positions.iter())
        .filter(|pos| pos.empties() <= args.max_empties)
        .cloned()
        .collect();
    eprintln!("Extracted {} positions", positions.len());

    if positions.is_empty() {
        eprintln!("No positions match the criteria. Exiting.");
        return;
    }

    // Initialize features and weights
    let features = Features::edax();
    eprintln!("Features: {}", features.count());

    let mut weights = Weights::load_or_create(&args.weights_file, &features);

    let mut examples = match build_examples(&args.eval_file, &positions) {
        Ok(ex) => ex,
        Err(e) => {
            eprintln!("Error: {e}");
            return;
        }
    };
    eprintln!("Training examples: {}", examples.len());

    // Train
    eprintln!("\n--- Training ---");
    eprintln!("(press Ctrl+C to stop early and save weights)");
    let interrupted = Arc::new(AtomicBool::new(false));
    {
        let interrupted = Arc::clone(&interrupted);
        if let Err(e) = ctrlc::set_handler(move || {
            eprintln!("\nInterrupt received — finishing current epoch...");
            interrupted.store(true, Ordering::Relaxed);
        }) {
            eprintln!("Warning: Failed to set Ctrl+C handler: {e}");
        }
    }

    let trainer = Trainer::new(0.1, 32, args.lr_decay);
    let train_config = TrainingConfig {
        epochs: args.epochs,
        epoch_offset: args.resume_epoch,
        interrupt: Some(interrupted),
    };
    trainer.train_epochs(&mut weights, &mut examples, &train_config);

    // Show sample weights and save
    weights.print_sample(&features, 10);

    eprintln!("\n--- Saving weights ---");
    match weights.save(&args.weights_file) {
        Ok(()) => eprintln!("Weights saved to {}", args.weights_file),
        Err(e) => eprintln!("Error saving weights: {e}"),
    }

    eprintln!("\nDone!");
}

fn print_usage(program: &str) {
    eprintln!("Usage: {program} [OPTIONS] <path>...");
    eprintln!();
    eprintln!("Train Othello evaluation weights from game files.");
    eprintln!();
    eprintln!("Uses exact alpha-beta evaluation for ground truth scores.");
    eprintln!();
    eprintln!("OPTIONS:");
    eprintln!(
        "  -n, --max-empties N   Only train on positions with <= N empty cells (default: 60)"
    );
    eprintln!("  -e, --epochs N        Number of training epochs (default: 10)");
    eprintln!(
        "  -f, --eval-file PATH  Eval cache (load if exists, compute+append missing, create if not)"
    );
    eprintln!("  -w, --weights PATH    Weights output file (default: trained_weights.bin)");
    eprintln!("  -d, --lr-decay F      Inverse-time LR decay factor (default: 0.01, 0 = no decay)");
    eprintln!("  -r, --resume-epoch N  Resume LR schedule from this epoch (default: 0)");
    eprintln!("  -h, --help            Show this help message");
    eprintln!();
    eprintln!("EVAL FILE FORMAT:");
    eprintln!("  Each line: <FEN> <score>");
    eprintln!("  FEN is 66 chars (64 board cells + space + side to move).");
    eprintln!();
    eprintln!("INPUT:");
    eprintln!("  One or more paths to:");
    eprintln!("    - .wtb files (WTHOR binary format)");
    eprintln!("    - .pgn / .txt files (PGN text format, PlayOK variant)");
    eprintln!("    - directories (scanned recursively for game files)");
    eprintln!();
    eprintln!("EXAMPLES:");
    eprintln!("  {program} training_data/");
    eprintln!("  {program} --max-empties 20 --epochs 50 training_data/");
    eprintln!("  {program} --eval-file ignored/evals.txt --epochs 30 training_data/");
    eprintln!("  {program} -e 1000 -d 0.001 -f evals.txt training_data/");
}
