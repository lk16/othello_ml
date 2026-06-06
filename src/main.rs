use othello_eval::{
    best_move, build_examples, load_games, Board, Features, Position, Trainer, TrainingConfig,
    Weights,
};
use std::env;
use std::io::{self, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

enum Command {
    Train(TrainArgs),
    Play(PlayArgs),
}

struct TrainArgs {
    max_empties: u32,
    epochs: usize,
    lr_decay: f32,
    resume_epoch: usize,
    eval_file: Option<String>,
    weights_file: String,
    paths: Vec<String>,
}

struct PlayArgs {
    depth: u32,
    exact_empties: u32,
    player_color: Option<PlayerColor>,
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
        "train" => parse_train_args(&args[0], &args[2..]),
        "play" => parse_play_args(&args[0], &args[2..]),
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
        paths,
    }))
}

fn parse_play_args(program: &str, args: &[String]) -> Option<Command> {
    let mut player_color = None;
    let mut depth: u32 = 6;
    let mut exact_empties: u32 = 12;
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
        Command::Play(args) => run_play(args),
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

    let mut examples = match build_examples(&args.eval_file, &positions) {
        Ok(ex) => ex,
        Err(e) => {
            eprintln!("Error: {e}");
            return;
        }
    };
    eprintln!("Training examples: {}", examples.len());

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

    weights.print_sample(&features, 10);

    eprintln!("\n--- Saving weights ---");
    match weights.save(&args.weights_file) {
        Ok(()) => eprintln!("Weights saved to {}", args.weights_file),
        Err(e) => eprintln!("Error saving weights: {e}"),
    }

    eprintln!("\nDone!");
}

fn run_play(args: PlayArgs) {
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
            "black (○ when your turn)"
        } else {
            "white (○ when your turn)"
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
            if let Some(cell) = best_move(&position, args.depth, args.exact_empties) {
                eprintln!("Bot plays: {}", cell_to_field(cell));
                position = position.do_move(cell);
            } else {
                position = position.pass_move();
            }
        }

        black_to_move = !black_to_move;
    }
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
    eprintln!("  train    Train evaluation weights from game files");
    eprintln!("  play     Play a game against the CLI");
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

fn print_play_usage(program: &str) {
    eprintln!("Usage: {program} play [OPTIONS]");
    eprintln!();
    eprintln!("Play a game against the CLI bot.");
    eprintln!();
    eprintln!("OPTIONS:");
    eprintln!("  -p, --player COLOR    Player color: b/black or w/white (default: random)");
    eprintln!("      --depth N         Bot search depth (default: 6)");
    eprintln!("      --exact-empties N Use exact search when <= N empties remain (default: 12)");
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
