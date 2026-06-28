//! Minimal macroquad GUI: `gui game | evaluate | pgn`.
//!
//! A small port of flippy's pygame GUI (game / evaluate / pgn modes). Unlike
//! flippy, all evaluations come from this crate's own search (no Edax, no remote
//! opening-book API) — see `docs/gui.md`.

mod modes;
mod render;
mod score;

use std::sync::Arc;

use macroquad::prelude::*;

use crate::othello::board::Board;
use crate::othello::position::Position;
use crate::training::weights::Weights;
use crate::{load_games, Game};
use modes::{EvaluateMode, GameMode, PgnMode, PlayMode};
use score::GraphPoint;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GuiMode {
    Game,
    Evaluate,
    Pgn,
    Play,
}

impl GuiMode {
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "game" => Some(GuiMode::Game),
            "evaluate" => Some(GuiMode::Evaluate),
            "pgn" => Some(GuiMode::Pgn),
            "play" => Some(GuiMode::Play),
            _ => None,
        }
    }
}

/// Arguments for the `gui` subcommand. Only flags each mode actually uses are
/// parsed (see `main.rs`).
pub struct GuiArgs {
    pub mode: GuiMode,
    /// Trained weights (required for evaluate/pgn — they need scores).
    pub weights_file: Option<String>,
    /// PGN/game file (required for pgn mode).
    pub pgn_file: Option<String>,
    /// Heuristic search depth for move scores / the graph.
    pub depth: u32,
    /// Switch to exact search at/below this many empties.
    pub exact_empties: u32,
    /// `play` mode: the colour the human plays (the engine plays the other).
    pub human_black: bool,
}

/// A single move's evaluation shown on the board.
pub struct Eval {
    pub cell: u32,
    pub score: i32,
    /// Optional search-depth label (pgn `l` toggle).
    pub level: Option<u32>,
}

/// Everything the renderer needs that isn't the raw board.
#[derive(Default)]
pub struct UiDetails {
    pub evaluations: Vec<Eval>,
    /// The move actually played here in a loaded game (pgn mode).
    pub played_move: Option<u32>,
    /// Black-POV score per mainline board (pgn mode); empty otherwise.
    pub graph: Vec<GraphPoint>,
    pub graph_current: Option<usize>,
}

/// A GUI mode: handles input and produces a board + UI overlay each frame.
pub trait Mode {
    fn on_left_click(&mut self, _cell: u32) {}
    fn on_right_click(&mut self) {}
    fn on_key(&mut self, _key: KeyCode) {}
    /// Called every frame: poll background workers, submit new jobs.
    fn tick(&mut self) {}
    fn board(&self) -> Board;
    fn ui(&self) -> UiDetails {
        UiDetails::default()
    }
}

pub fn start_board() -> Board {
    Board {
        position: Position::initial(),
        black_to_move: true,
    }
}

/// Entry point for the `gui` subcommand. Loads weights / a game as needed,
/// then runs the macroquad event loop.
pub fn run(args: GuiArgs) {
    // Evaluate, pgn and play need a trained eval; game does not.
    let eval = match args.mode {
        GuiMode::Game => None,
        GuiMode::Evaluate | GuiMode::Pgn | GuiMode::Play => {
            let Some(path) = args.weights_file.as_deref() else {
                eprintln!(
                    "Error: --weights/-w is required for `gui {}`.",
                    mode_name(args.mode)
                );
                return;
            };
            match Weights::load(path) {
                Ok(w) => Some(Arc::new(w)),
                Err(e) => {
                    eprintln!("Error loading weights from {path}: {e}");
                    return;
                }
            }
        }
    };

    let mode: Box<dyn Mode> = match args.mode {
        GuiMode::Game => Box::new(GameMode::new()),
        GuiMode::Evaluate => {
            let Some(weights) = eval else { return };
            Box::new(EvaluateMode::new(weights, args.depth, args.exact_empties))
        }
        GuiMode::Pgn => {
            let Some(weights) = eval else { return };
            let Some(path) = args.pgn_file.as_deref() else {
                eprintln!("Error: --pgn/-p is required for `gui pgn`.");
                return;
            };
            let game = match load_first_game(path) {
                Ok(g) => g,
                Err(e) => {
                    eprintln!("Error loading PGN {path}: {e}");
                    return;
                }
            };
            Box::new(PgnMode::new(game, weights, args.depth, args.exact_empties))
        }
        GuiMode::Play => {
            let Some(weights) = eval else { return };
            Box::new(PlayMode::new(
                weights,
                args.depth,
                args.exact_empties,
                args.human_black,
            ))
        }
    };

    let height = if args.mode == GuiMode::Pgn {
        render::BOARD_PX + render::GRAPH_PX
    } else {
        render::BOARD_PX
    };
    let conf = macroquad::conf::Conf {
        miniquad_conf: macroquad::miniquad::conf::Conf {
            window_title: "Othello".to_string(),
            window_width: render::BOARD_PX as i32,
            window_height: height as i32,
            window_resizable: false,
            ..Default::default()
        },
        ..Default::default()
    };
    macroquad::Window::from_config(conf, amain(mode));
}

fn mode_name(mode: GuiMode) -> &'static str {
    match mode {
        GuiMode::Game => "game",
        GuiMode::Evaluate => "evaluate",
        GuiMode::Pgn => "pgn",
        GuiMode::Play => "play",
    }
}

fn load_first_game(path: &str) -> Result<Game, String> {
    let games = load_games(&[path.to_string()]).map_err(|e| e.to_string())?;
    games
        .into_iter()
        .next()
        .ok_or_else(|| "no games found in file".to_string())
}

async fn amain(mut mode: Box<dyn Mode>) {
    // A clean sans-serif for on-board text (macroquad's built-in font is a
    // pixel font). Bundled; falls back to the default if it fails to load.
    let font =
        macroquad::text::load_ttf_font_from_bytes(include_bytes!("LiberationSans-Regular.ttf"))
            .ok();

    loop {
        if is_mouse_button_pressed(MouseButton::Left) {
            let (x, y) = mouse_position();
            if let Some(cell) = render::pixel_to_cell(x, y) {
                mode.on_left_click(cell);
            }
        }
        if is_mouse_button_pressed(MouseButton::Right) {
            mode.on_right_click();
        }
        for key in [
            KeyCode::Left,
            KeyCode::Right,
            KeyCode::Space,
            KeyCode::L,
            KeyCode::F,
        ] {
            if is_key_pressed(key) {
                mode.on_key(key);
            }
        }

        mode.tick();

        let board = mode.board();
        let ui = mode.ui();
        render::render(&board, &ui, font.as_ref());

        next_frame().await;
    }
}
