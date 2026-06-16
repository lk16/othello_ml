use othello_eval::{
    best_move, bootstrap_score, build_examples, load_games, Board, Features, FlatEval,
    ParallelSolver, Position, Solver, Trainer, TrainingConfig, TrainingExample, Weights,
};
use std::env;
use std::io::{self, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

enum Command {
    Train(TrainArgs),
    TrainBoot(TrainBootArgs),
    Play(PlayArgs),
    Bench(BenchArgs),
    BenchFlip,
    BenchCountFlip,
    BenchGetMoves,
}

struct TrainArgs {
    max_empties: u32,
    epochs: usize,
    lr_decay: f32,
    resume_epoch: usize,
    eval_file: Option<String>,
    weights_file: String,
    threads: usize,
    paths: Vec<String>,
}

/// Args for `train-boot`: bootstrapped training of empties > N against shallow-search
/// estimates that use the current weights at their leaves (see `run_train_boot`).
struct TrainBootArgs {
    /// Exact-trained frontier N: the eval is trusted at empties ≤ N (anchor).
    exact_empties: u32,
    /// Train bands up to this many empties (M).
    max_empties: u32,
    /// Shallow-search depth for labels — also the curriculum band width, so each
    /// band's leaves land in the already-trained band below it.
    depth: u32,
    /// SGD epochs per band.
    epochs: usize,
    /// Inverse-time LR decay (per band; each band restarts the schedule).
    lr_decay: f32,
    /// Weights file: loaded for the starting eval (must already be exact-trained)
    /// and overwritten after each band.
    weights_file: String,
    /// Threads for label generation (training itself is single-threaded online SGD).
    threads: usize,
    paths: Vec<String>,
}

struct PlayArgs {
    depth: u32,
    exact_empties: u32,
    player_color: Option<PlayerColor>,
    weights_file: Option<String>,
}

struct BenchArgs {
    empties: u32,
    max_boards: Option<usize>,
    threads: usize,
    /// When set (sequential only), print one OBF line per board to stdout (for
    /// cross-checking against another solver) and per-board score/nodes/time to
    /// stderr, instead of only the aggregate summary.
    per_board: bool,
    /// Optional trained-weights file: when set, the sequential solver uses
    /// eval-guided move ordering (Step 34). Absence = the mobility-only baseline,
    /// so `bench` vs `bench --weights` is a node-count A/B.
    weights: Option<String>,
    paths: Vec<String>,
}

enum PlayerColor {
    Black,
    White,
}

fn parse_args() -> Option<Command> {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        print_usage(&args[0]);
        return None;
    }

    match args[1].as_str() {
        // `train` is kept as an alias for `train-exact` (the original behaviour).
        "train" | "train-exact" => parse_train_args(&args[0], &args[2..]),
        "train-boot" => parse_train_boot_args(&args[0], &args[2..]),
        "play" => parse_play_args(&args[0], &args[2..]),
        "bench" => parse_bench_args(&args[0], &args[2..]),
        "bench-flip" => Some(Command::BenchFlip),
        "bench-count-flip" => Some(Command::BenchCountFlip),
        "bench-get-moves" => Some(Command::BenchGetMoves),
        "--help" | "-h" => {
            print_usage(&args[0]);
            None
        }
        other => {
            eprintln!("Unknown command: {other}\n");
            print_usage(&args[0]);
            None
        }
    }
}

fn parse_train_args(program: &str, args: &[String]) -> Option<Command> {
    let mut max_empties: u32 = 60;
    let mut epochs: usize = 10;
    let mut lr_decay: f32 = 0.01;
    let mut resume_epoch: usize = 0;
    let mut eval_file: Option<String> = None;
    let mut weights_file: String = String::from("trained_weights.bin");
    let mut threads: usize = 1;
    let mut paths: Vec<String> = Vec::new();
    let mut i = 0;
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
        } else if args[i] == "--threads" || args[i] == "-t" {
            i += 1;
            if i < args.len() {
                threads = args[i].parse::<usize>().unwrap_or(1).max(1);
            }
        } else if args[i] == "--help" || args[i] == "-h" {
            print_train_usage(program);
            return None;
        } else {
            paths.push(args[i].clone());
        }
        i += 1;
    }

    Some(Command::Train(TrainArgs {
        max_empties,
        epochs,
        lr_decay,
        resume_epoch,
        eval_file,
        weights_file,
        threads,
        paths,
    }))
}

fn parse_train_boot_args(program: &str, args: &[String]) -> Option<Command> {
    let mut exact_empties: u32 = 16;
    let mut max_empties: u32 = 24;
    let mut depth: u32 = 4;
    let mut epochs: usize = 100;
    let mut lr_decay: f32 = 0.01;
    let mut weights_file: String = String::from("trained_weights.bin");
    let mut threads: usize = 1;
    let mut paths: Vec<String> = Vec::new();
    let mut i = 0;
    while i < args.len() {
        if args[i] == "--exact-empties" {
            i += 1;
            if i < args.len() {
                exact_empties = args[i].parse::<u32>().unwrap_or(16);
            }
        } else if args[i] == "--max-empties" || args[i] == "-n" {
            i += 1;
            if i < args.len() {
                max_empties = args[i].parse::<u32>().unwrap_or(24);
            }
        } else if args[i] == "--depth" {
            i += 1;
            if i < args.len() {
                depth = args[i].parse::<u32>().unwrap_or(4);
            }
        } else if args[i] == "--epochs" || args[i] == "-e" {
            i += 1;
            if i < args.len() {
                epochs = args[i].parse::<usize>().unwrap_or(100);
            }
        } else if args[i] == "--lr-decay" || args[i] == "-d" {
            i += 1;
            if i < args.len() {
                lr_decay = args[i].parse::<f32>().unwrap_or(0.01);
            }
        } else if args[i] == "--weights" || args[i] == "-w" {
            i += 1;
            if i < args.len() {
                weights_file = args[i].clone();
            }
        } else if args[i] == "--threads" || args[i] == "-t" {
            i += 1;
            if i < args.len() {
                threads = args[i].parse::<usize>().unwrap_or(1).max(1);
            }
        } else if args[i] == "--help" || args[i] == "-h" {
            print_train_boot_usage(program);
            return None;
        } else {
            paths.push(args[i].clone());
        }
        i += 1;
    }

    Some(Command::TrainBoot(TrainBootArgs {
        exact_empties,
        max_empties,
        depth,
        epochs,
        lr_decay,
        weights_file,
        threads,
        paths,
    }))
}

fn parse_play_args(program: &str, args: &[String]) -> Option<Command> {
    let mut player_color = None;
    let mut depth: u32 = 6;
    let mut exact_empties: u32 = 12;
    let mut weights_file: Option<String> = None;
    let mut i = 0;
    while i < args.len() {
        if args[i] == "--player" || args[i] == "-p" {
            i += 1;
            if i < args.len() {
                if let Some(c) = parse_player_color(&args[i]) {
                    player_color = Some(c);
                } else {
                    eprintln!("Invalid player color: {}. Use b/w/black/white.", args[i]);
                    return None;
                }
            }
        } else if args[i] == "--depth" {
            i += 1;
            if i < args.len() {
                depth = args[i].parse::<u32>().unwrap_or(6);
            }
        } else if args[i] == "--exact-empties" {
            i += 1;
            if i < args.len() {
                exact_empties = args[i].parse::<u32>().unwrap_or(12);
            }
        } else if args[i] == "--weights" || args[i] == "-w" {
            i += 1;
            if i < args.len() {
                weights_file = Some(args[i].clone());
            }
        } else if args[i] == "--help" || args[i] == "-h" {
            print_play_usage(program);
            return None;
        } else {
            eprintln!("Unknown option for play: {}\n", args[i]);
            print_play_usage(program);
            return None;
        }
        i += 1;
    }

    Some(Command::Play(PlayArgs {
        depth,
        exact_empties,
        player_color,
        weights_file,
    }))
}

fn parse_bench_args(program: &str, args: &[String]) -> Option<Command> {
    let mut empties: u32 = 20;
    let mut max_boards: Option<usize> = None;
    let mut threads: usize = 1;
    let mut per_board = false;
    let mut weights: Option<String> = None;
    let mut paths: Vec<String> = Vec::new();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--per-board" => {
                per_board = true;
            }
            "--weights" | "-w" => {
                i += 1;
                if i < args.len() {
                    weights = Some(args[i].clone());
                }
            }
            "--empties" | "-n" => {
                i += 1;
                if i < args.len() {
                    empties = args[i].parse().unwrap_or(20);
                }
            }
            "--max-boards" | "-m" => {
                i += 1;
                if i < args.len() {
                    max_boards = args[i].parse().ok();
                }
            }
            "--threads" | "-t" => {
                i += 1;
                if i < args.len() {
                    threads = args[i].parse::<usize>().unwrap_or(1).max(1);
                }
            }
            "--help" | "-h" => {
                print_bench_usage(program);
                return None;
            }
            _ => paths.push(args[i].clone()),
        }
        i += 1;
    }
    if paths.is_empty() {
        print_bench_usage(program);
        return None;
    }
    Some(Command::Bench(BenchArgs {
        empties,
        max_boards,
        threads,
        per_board,
        weights,
        paths,
    }))
}

fn parse_player_color(s: &str) -> Option<PlayerColor> {
    match s.to_lowercase().as_str() {
        "b" | "black" => Some(PlayerColor::Black),
        "w" | "white" => Some(PlayerColor::White),
        _ => None,
    }
}

fn main() {
    let cmd = if let Some(c) = parse_args() {
        c
    } else {
        return;
    };

    match cmd {
        Command::Train(args) => run_train(args),
        Command::TrainBoot(args) => run_train_boot(args),
        Command::Play(args) => run_play(args),
        Command::Bench(args) => run_bench(args),
        Command::BenchFlip => othello_eval::bench_flip_variants(),
        Command::BenchCountFlip => othello_eval::bench_count_flip_variants(),
        Command::BenchGetMoves => othello_eval::bench_get_moves_variants(),
    }
}

fn run_train(args: TrainArgs) {
    if args.paths.is_empty() {
        eprintln!("Error: No input files or directories specified.\n");
        let program = env::args().next().unwrap_or_default();
        print_train_usage(&program);
        return;
    }

    eprintln!("=== Othello ML Training ===");
    eprintln!("Max empties: {}", args.max_empties);
    eprintln!("Epochs: {}", args.epochs);
    eprintln!("Threads: {}", args.threads);
    if args.resume_epoch > 0 {
        eprintln!(
            "Resume epoch: {} (LR schedule continues from here)",
            args.resume_epoch
        );
    }
    eprintln!("Input paths: {:?}", args.paths);

    let games = match load_games(&args.paths) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("Error loading games: {e}");
            return;
        }
    };

    eprintln!(
        "\n--- Extracting positions (empties <= {}) ---",
        args.max_empties
    );
    let positions: Vec<Board> = games
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

    let features = Features::edax();
    eprintln!("Features: {}", features.count());

    let mut weights = Weights::load_or_create(&args.weights_file, &features);

    eprintln!("\n--- Building training examples ---");
    eprintln!("(press Ctrl+C to stop early and save progress)");
    let interrupted = Arc::new(AtomicBool::new(false));
    {
        let interrupted = Arc::clone(&interrupted);
        if let Err(e) = ctrlc::set_handler(move || {
            eprintln!("\nInterrupt received — finishing current operation...");
            interrupted.store(true, Ordering::Relaxed);
        }) {
            eprintln!("Warning: Failed to set Ctrl+C handler: {e}");
        }
    }

    let examples = match build_examples(&args.eval_file, &positions, &interrupted, args.threads) {
        Ok(ex) => ex,
        Err(e) => {
            eprintln!("Error: {e}");
            return;
        }
    };
    eprintln!("Training examples: {}", examples.len());

    eprintln!("\n--- Training ---");
    eprintln!("(press Ctrl+C to stop early and save weights)");

    let trainer = Trainer::new(0.1, 32, args.lr_decay);
    let train_config = TrainingConfig {
        epochs: args.epochs,
        epoch_offset: args.resume_epoch,
        interrupt: Some(interrupted),
        threads: args.threads,
    };
    trainer.train_epochs(&mut weights, &examples, &train_config);

    weights.print_sample(&features, 10);

    eprintln!("\n--- Saving weights ---");
    match weights.save(&args.weights_file) {
        Ok(()) => eprintln!("Weights saved to {}", args.weights_file),
        Err(e) => eprintln!("Error saving weights: {e}"),
    }

    eprintln!("\nDone!");
}

/// Bootstrapped training of deeper positions (empties > N).
///
/// The eval is exact-trained at empties ≤ N (the `train-exact` output). This
/// command extends it outward, one **band of width `depth`** at a time: each band
/// `(frontier, frontier+depth]` is labelled by a depth-`depth` shallow search whose
/// leaves are scored by the *current* weights (frozen `FlatEval` snapshot), then
/// trained. Because a depth-`depth` search from empties ≤ frontier+depth bottoms out
/// at empties ≤ frontier — already trained — every label is anchored to the band
/// below it, expanding the well-trained frontier upward without unanchored drift.
/// Weights are per-empty-range buckets, so each band updates disjoint buckets (no
/// forgetting of lower bands). Training is single-threaded online SGD; `--threads`
/// parallelises only the (independent) label generation.
fn run_train_boot(args: TrainBootArgs) {
    if args.paths.is_empty() {
        eprintln!("Error: No input files or directories specified.\n");
        let program = env::args().next().unwrap_or_default();
        print_train_boot_usage(&program);
        return;
    }

    eprintln!("=== Othello ML Bootstrapped Training ===");
    eprintln!("Exact frontier N : {}", args.exact_empties);
    eprintln!("Train up to      : {} empties", args.max_empties);
    eprintln!("Search depth/band: {}", args.depth);
    eprintln!("Epochs per band  : {}", args.epochs);
    eprintln!("Label threads    : {}", args.threads);
    eprintln!("Weights file     : {}", args.weights_file);

    if args.depth == 0 {
        eprintln!("Error: --depth must be >= 1 (depth 0 = bare eval, no bootstrapping).");
        return;
    }

    // The starting eval must already be exact-trained; a fresh (all-zero) table
    // produces meaningless labels, so require the file to load.
    let mut weights = match Weights::load(&args.weights_file) {
        Ok(w) => w,
        Err(e) => {
            eprintln!(
                "Error: could not load weights from {} ({e}).\n\
                 train-boot needs an exact-trained eval to bootstrap from — run `train-exact` first.",
                args.weights_file
            );
            return;
        }
    };

    let games = match load_games(&args.paths) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("Error loading games: {e}");
            return;
        }
    };
    let positions: Vec<Board> = games
        .iter()
        .flat_map(|game| game.positions.iter())
        .filter(|p| p.empties() > args.exact_empties && p.empties() <= args.max_empties)
        .cloned()
        .collect();
    eprintln!(
        "Extracted {} positions with empties in ({}, {}]",
        positions.len(),
        args.exact_empties,
        args.max_empties
    );
    if positions.is_empty() {
        eprintln!("No positions in the bootstrap range. Exiting.");
        return;
    }

    let interrupted = Arc::new(AtomicBool::new(false));
    {
        let interrupted = Arc::clone(&interrupted);
        if let Err(e) = ctrlc::set_handler(move || {
            eprintln!("\nInterrupt received — finishing current band, then stopping...");
            interrupted.store(true, Ordering::Relaxed);
        }) {
            eprintln!("Warning: Failed to set Ctrl+C handler: {e}");
        }
    }

    let trainer = Trainer::new(0.1, 32, args.lr_decay);

    let mut frontier = args.exact_empties;
    while frontier < args.max_empties {
        if interrupted.load(Ordering::Relaxed) {
            break;
        }
        let hi = (frontier + args.depth).min(args.max_empties);
        let band: Vec<Board> = positions
            .iter()
            .filter(|p| p.empties() > frontier && p.empties() <= hi)
            .cloned()
            .collect();

        eprintln!(
            "\n--- Band empties ({frontier}, {hi}]: {} positions ---",
            band.len()
        );
        if band.is_empty() {
            frontier = hi;
            continue;
        }

        // Freeze the current weights as the leaf eval for this band's labels.
        let flat = Arc::new(FlatEval::from_weights(&weights));
        let t = Instant::now();
        let examples = bootstrap_label(&band, &flat, args.depth, args.threads);
        eprintln!(
            "Labelled {} positions in {:.1}s (depth-{} shallow search)",
            examples.len(),
            t.elapsed().as_secs_f64(),
            args.depth
        );

        let train_config = TrainingConfig {
            epochs: args.epochs,
            epoch_offset: 0, // each band is its own disjoint weight region
            interrupt: Some(Arc::clone(&interrupted)),
            threads: 1, // single-threaded online SGD (best convergence)
        };
        trainer.train_epochs(&mut weights, &examples, &train_config);

        match weights.save(&args.weights_file) {
            Ok(()) => eprintln!("Saved weights after band ({frontier}, {hi}]"),
            Err(e) => eprintln!("Error saving weights: {e}"),
        }
        frontier = hi;
    }

    eprintln!("\nDone (trained up to empties {frontier}).");
}

/// Label a band of positions by bootstrapped shallow search, parallelised across
/// `threads` (the labels are independent). Training stays single-threaded.
fn bootstrap_label(
    band: &[Board],
    flat: &Arc<FlatEval>,
    depth: u32,
    threads: usize,
) -> Vec<TrainingExample> {
    let label = |b: &Board| TrainingExample {
        position: b.position,
        target_score: bootstrap_score(&b.position, flat, depth),
    };

    if threads <= 1 {
        return band.iter().map(label).collect();
    }

    let chunk = band.len().div_ceil(threads);
    std::thread::scope(|s| {
        let handles: Vec<_> = band
            .chunks(chunk)
            .map(|part| {
                let flat = Arc::clone(flat);
                s.spawn(move || {
                    part.iter()
                        .map(|b| TrainingExample {
                            position: b.position,
                            target_score: bootstrap_score(&b.position, &flat, depth),
                        })
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        handles
            .into_iter()
            .flat_map(|h| h.join().unwrap_or_default())
            .collect()
    })
}

fn run_play(args: PlayArgs) {
    let weights_file = if let Some(w) = args.weights_file {
        w
    } else {
        eprintln!("Error: --weights/-w is required for play command");
        return;
    };

    let features = Features::edax();
    let weights = Weights::load_or_create(&weights_file, &features);

    let player_is_black = match args.player_color {
        Some(PlayerColor::Black) => true,
        Some(PlayerColor::White) => false,
        None => {
            let nanos = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .subsec_nanos();
            nanos % 2 == 0
        }
    };

    eprintln!(
        "You play {}",
        if player_is_black {
            "black (●)"
        } else {
            "white (○)"
        }
    );

    let mut position = Position::initial();
    let mut black_to_move = true;

    loop {
        if position.is_game_end() {
            let board = Board {
                position,
                black_to_move,
            };
            println!("{}", board.show(false));
            print_game_result(&position, black_to_move, player_is_black);
            return;
        }

        if !position.has_moves() {
            let side = if black_to_move { "Black" } else { "White" };
            eprintln!("{side} has no moves, passing.");
            position = position.pass_move();
            black_to_move = !black_to_move;
            continue;
        }

        let is_player_turn = player_is_black == black_to_move;
        let board = Board {
            position,
            black_to_move,
        };

        if is_player_turn {
            println!("{}", board.show(true));
            let moves = position.get_moves();
            let cell = prompt_for_move(moves);
            eprintln!("You play: {}", cell_to_field(cell));
            position = position.do_move(cell);
        } else {
            println!("{}", board.show(false));
            eprintln!("Bot is thinking...");
            if let Some(cell) = best_move(
                &position,
                args.depth,
                args.exact_empties,
                &weights,
                &features,
            ) {
                eprintln!("Bot plays: {}", cell_to_field(cell));
                position = position.do_move(cell);
            } else {
                position = position.pass_move();
            }
        }

        black_to_move = !black_to_move;
    }
}

fn run_bench(args: BenchArgs) {
    let games = match load_games(&args.paths) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("Error loading games: {e}");
            return;
        }
    };

    let iter = games
        .iter()
        .flat_map(|g| g.positions.iter())
        .filter(|b| b.empties() == args.empties);
    let positions: Vec<Board> = if let Some(limit) = args.max_boards {
        iter.take(limit).cloned().collect()
    } else {
        iter.cloned().collect()
    };

    if positions.is_empty() {
        eprintln!("No positions found with exactly {} empties.", args.empties);
        eprintln!("Try a different --empties value.");
        return;
    }

    eprintln!(
        "Benchmarking exact_score: {} positions, {} empties each, {} thread(s)",
        positions.len(),
        args.empties,
        args.threads,
    );

    // Step 34: optional eval-guided ordering. Build the flat eval once and share it.
    let eval: Option<Arc<FlatEval>> = match &args.weights {
        Some(path) => match Weights::load(path) {
            Ok(w) => {
                eprintln!("Eval-guided ordering: loaded weights from {path}");
                Some(Arc::new(FlatEval::from_weights(&w)))
            }
            Err(e) => {
                eprintln!("Error loading weights from {path}: {e}");
                return;
            }
        },
        None => None,
    };
    if eval.is_some() && args.threads > 1 {
        eprintln!("Note: eval-guided ordering is sequential-only; ignored with --threads > 1.");
    }
    let make_solver = || match &eval {
        Some(e) => Solver::with_eval(Arc::clone(e)),
        None => Solver::new(),
    };

    // Per-board mode (sequential only): one OBF line per board to stdout, plus
    // per-board score/nodes/time to stderr. A fresh `Solver` per board avoids the
    // warm shared TT skewing later boards — each is solved cold, matching how a
    // one-shot external solver sees it.
    if args.per_board {
        let mut total_nodes: u64 = 0;
        let mut total_time = 0.0;
        for (idx, board) in positions.iter().enumerate() {
            let mut solver = make_solver();
            let t = Instant::now();
            let (score, nodes) = solver.exact_score_with_nodes(&board.position);
            let secs = t.elapsed().as_secs_f64();
            total_nodes += nodes;
            total_time += secs;
            // OBF line for the external solver (player = X, X to move).
            println!("{};", board.position.to_fen(true));
            eprintln!(
                "board {idx}: score={score} nodes={nodes} time={:.1}ms",
                secs * 1000.0
            );
        }
        let n = positions.len() as f64;
        eprintln!("Per-board totals:");
        eprintln!("  Total time   : {total_time:.3}s");
        eprintln!("  Time/position: {:.1}ms", total_time / n * 1000.0);
        eprintln!("  Total nodes  : {total_nodes}");
        eprintln!("  Nodes/pos    : {:.0}", total_nodes as f64 / n);
        eprintln!(
            "  Nodes/s      : {:.2}M",
            total_nodes as f64 / total_time / 1_000_000.0
        );
        return;
    }

    // threads == 1 uses the sequential solver (private, lock-free table);
    // threads > 1 solves each position with root-level YBWC (Step 21).
    let mut total_nodes: u64 = 0;
    let start = Instant::now();

    if args.threads > 1 {
        let solver = ParallelSolver::new(args.threads);
        for board in &positions {
            let (_, nodes) = solver.exact_score_with_nodes(&board.position);
            total_nodes += nodes;
        }
    } else {
        let mut solver = make_solver();
        for board in &positions {
            let (_, nodes) = solver.exact_score_with_nodes(&board.position);
            total_nodes += nodes;
        }
    }

    let elapsed = start.elapsed().as_secs_f64();
    let n = positions.len() as f64;

    eprintln!("Results:");
    eprintln!("  Positions    : {}", positions.len());
    eprintln!("  Total time   : {elapsed:.3}s");
    eprintln!("  Time/position: {:.1}ms", elapsed / n * 1000.0);
    eprintln!("  Positions/s  : {:.0}", n / elapsed);
    eprintln!("  Total nodes  : {total_nodes}");
    eprintln!("  Nodes/pos    : {:.0}", total_nodes as f64 / n);
    eprintln!(
        "  Nodes/s      : {:.2}M",
        total_nodes as f64 / elapsed / 1_000_000.0
    );
}

fn prompt_for_move(moves: u64) -> u32 {
    loop {
        print!("Your move: ");
        let _ = io::stdout().flush();

        let mut input = String::new();
        if io::stdin().read_line(&mut input).is_err() {
            continue;
        }

        if let Some(cell) = parse_move_input(&input, moves) {
            return cell;
        }

        eprintln!("Invalid move. Use the letter shown on the board.");
    }
}

fn parse_move_input(input: &str, moves: u64) -> Option<u32> {
    let input = input.trim();
    if input.is_empty() {
        return None;
    }
    let ch = input.chars().next()?;
    if !ch.is_ascii_lowercase() {
        return None;
    }
    let target = (ch as u8).wrapping_sub(b'a');
    let mut label = 0u8;
    for cell in 0..64u32 {
        if moves & (1u64 << cell) != 0 {
            if label == target {
                return Some(cell);
            }
            label += 1;
        }
    }
    None
}

fn cell_to_field(cell: u32) -> String {
    let col = (b'a' + (cell % 8) as u8) as char;
    let row = (cell / 8) + 1;
    format!("{col}{row}")
}

fn print_game_result(position: &Position, black_to_move: bool, player_is_black: bool) {
    let (black_discs, white_discs) = if black_to_move {
        (position.player_discs(), position.opponent_discs())
    } else {
        (position.opponent_discs(), position.player_discs())
    };

    eprintln!("Game over!");
    eprintln!("Black: {black_discs} discs");
    eprintln!("White: {white_discs} discs");

    let player_discs = if player_is_black {
        black_discs
    } else {
        white_discs
    };
    let bot_discs = if player_is_black {
        white_discs
    } else {
        black_discs
    };

    if player_discs > bot_discs {
        eprintln!("You win!");
    } else if bot_discs > player_discs {
        eprintln!("Bot wins!");
    } else {
        eprintln!("Draw!");
    }
}

fn print_usage(program: &str) {
    eprintln!("Usage: {program} <COMMAND> [OPTIONS]");
    eprintln!();
    eprintln!("Othello ML training and playing.");
    eprintln!();
    eprintln!("Commands:");
    eprintln!("  train-exact  Train weights on exact-search labels (empties <= N). Alias: train");
    eprintln!(
        "  train-boot   Extend the eval to empties > N via bootstrapped shallow-search labels"
    );
    eprintln!("  play     Play a game against the CLI");
    eprintln!("  bench    Benchmark exact alpha-beta search speed");
    eprintln!("  bench-flip  Micro-benchmark the flip-computation variants");
    eprintln!("  bench-count-flip  Micro-benchmark the count-last-flip variants");
    eprintln!("  bench-get-moves  Micro-benchmark the mobility variants");
    eprintln!();
    eprintln!("Use \"{program} <command> --help\" for more information about a command.");
}

fn print_train_usage(program: &str) {
    eprintln!("Usage: {program} train [OPTIONS] <path>...");
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
    eprintln!("  -t, --threads N       Number of threads for parallel training (default: 1)");
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
    eprintln!("  {program} train training_data/");
    eprintln!("  {program} train --max-empties 20 --epochs 50 training_data/");
    eprintln!("  {program} train --eval-file ignored/evals.txt --epochs 30 training_data/");
    eprintln!("  {program} train -e 1000 -d 0.001 -f evals.txt training_data/");
}

fn print_train_boot_usage(program: &str) {
    eprintln!("Usage: {program} train-boot [OPTIONS] <path>...");
    eprintln!();
    eprintln!("Extend a trained eval to positions with empties > N using bootstrapped labels:");
    eprintln!("each position is labelled by a shallow search whose leaves use the current");
    eprintln!("weights, expanding the well-trained frontier outward one band at a time.");
    eprintln!();
    eprintln!("Requires an already exact-trained weights file (run `train-exact` first).");
    eprintln!();
    eprintln!("OPTIONS:");
    eprintln!("      --exact-empties N Exact-trained frontier to bootstrap from (default: 16)");
    eprintln!("  -n, --max-empties N   Train bands up to N empties (default: 24)");
    eprintln!("      --depth N         Shallow-search depth = curriculum band width (default: 4)");
    eprintln!("  -e, --epochs N        SGD epochs per band (default: 100)");
    eprintln!("  -d, --lr-decay F      Inverse-time LR decay factor (default: 0.01)");
    eprintln!("  -w, --weights PATH    Weights file: loaded and overwritten (default: trained_weights.bin)");
    eprintln!("  -t, --threads N       Threads for label generation (training is single-threaded)");
    eprintln!("  -h, --help            Show this help message");
    eprintln!();
    eprintln!("INPUT:");
    eprintln!("  Same game-file paths as train-exact (.wtb/.pgn/.txt/directories).");
    eprintln!();
    eprintln!("EXAMPLES:");
    eprintln!("  {program} train-boot -w ignored/trained_weights.bin training_data/");
    eprintln!(
        "  {program} train-boot --exact-empties 16 -n 28 --depth 4 -t 8 -w w.bin training_data/"
    );
}

fn print_bench_usage(program: &str) {
    eprintln!("Usage: {program} bench [OPTIONS] <path>...");
    eprintln!();
    eprintln!("Benchmark exact alpha-beta search over positions from game files.");
    eprintln!();
    eprintln!("OPTIONS:");
    eprintln!("  -n, --empties N    Only use positions with exactly N empty cells (default: 20)");
    eprintln!("  -m, --max-boards N Cap the number of positions benchmarked (default: all)");
    eprintln!(
        "  -t, --threads N    Workers for root-level YBWC per position (default: 1 = sequential)"
    );
    eprintln!(
        "  -w, --weights PATH Use eval-guided move ordering from a trained-weights file (Step 34;"
    );
    eprintln!("                     sequential only). Omit for the mobility-only baseline.");
    eprintln!("  -h, --help         Show this help");
    eprintln!();
    eprintln!("INPUT:");
    eprintln!("  One or more paths to .wtb/.pgn files or directories (same as train subcommand).");
}

fn print_play_usage(program: &str) {
    eprintln!("Usage: {program} play [OPTIONS]");
    eprintln!();
    eprintln!("Play a game against the CLI bot.");
    eprintln!();
    eprintln!("OPTIONS:");
    eprintln!("  -p, --player COLOR    Player color: b/black or w/white (default: random)");
    eprintln!("      --depth N         Bot search depth (default: 6)");
    eprintln!("      --exact-empties N Use exact search when <= N empties remain (default: 12)");
    eprintln!("  -w, --weights PATH    Weights file (required)");
    eprintln!("  -h, --help            Show this help message");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_move_input_valid() {
        let pos = Position::initial();
        let moves = pos.get_moves();
        let first_cell = moves.trailing_zeros();
        assert_eq!(parse_move_input("a", moves), Some(first_cell));
    }

    #[test]
    fn test_parse_move_input_invalid_letter() {
        let pos = Position::initial();
        let moves = pos.get_moves();
        assert_eq!(parse_move_input("z", moves), None);
    }

    #[test]
    fn test_parse_move_input_empty() {
        let pos = Position::initial();
        let moves = pos.get_moves();
        assert_eq!(parse_move_input("", moves), None);
        assert_eq!(parse_move_input("  ", moves), None);
    }

    #[test]
    fn test_parse_move_input_uppercase() {
        let pos = Position::initial();
        let moves = pos.get_moves();
        assert_eq!(parse_move_input("A", moves), None);
    }

    #[test]
    fn test_parse_player_color() {
        assert!(matches!(parse_player_color("b"), Some(PlayerColor::Black)));
        assert!(matches!(
            parse_player_color("black"),
            Some(PlayerColor::Black)
        ));
        assert!(matches!(parse_player_color("w"), Some(PlayerColor::White)));
        assert!(matches!(
            parse_player_color("white"),
            Some(PlayerColor::White)
        ));
        assert!(matches!(
            parse_player_color("Black"),
            Some(PlayerColor::Black)
        ));
        assert!(parse_player_color("x").is_none());
        assert!(parse_player_color("").is_none());
    }

    #[test]
    fn test_cell_to_field() {
        assert_eq!(cell_to_field(0), "a1");
        assert_eq!(cell_to_field(7), "h1");
        assert_eq!(cell_to_field(56), "a8");
        assert_eq!(cell_to_field(63), "h8");
        assert_eq!(cell_to_field(27), "d4");
    }
}
