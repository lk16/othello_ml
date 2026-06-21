//! macroquad drawing for the GUI: the 8x8 board, discs, legal-move dots,
//! per-move score numbers + best-move ring, and the pgn evaluation graph.
//! Ported from flippy's `window.py` (same colours and layout).

use macroquad::prelude::*;

use super::{Eval, UiDetails};
use crate::othello::board::Board;

pub const BOARD_PX: f32 = 600.0;
pub const GRAPH_PX: f32 = 200.0;
const SQUARE: f32 = BOARD_PX / 8.0;
const DISC_R: f32 = SQUARE / 2.0 - 5.0;
const MOVE_R: f32 = SQUARE / 8.0;

fn rgb(r: u8, g: u8, b: u8) -> Color {
    Color::from_rgba(r, g, b, 255)
}

fn col_white() -> Color {
    rgb(255, 255, 255)
}
fn col_black() -> Color {
    rgb(0, 0, 0)
}
fn col_gray() -> Color {
    rgb(128, 128, 128)
}
fn col_bg() -> Color {
    rgb(0, 128, 0)
}
fn col_played() -> Color {
    rgb(0, 96, 0)
}
fn col_score_line() -> Color {
    rgb(96, 96, 96)
}

fn center(index: u32) -> (f32, f32) {
    let col = (index % 8) as f32;
    let row = (index / 8) as f32;
    (col * SQUARE + SQUARE / 2.0, row * SQUARE + SQUARE / 2.0)
}

/// Map a screen pixel to a board cell index, or `None` if outside the board.
pub fn pixel_to_cell(x: f32, y: f32) -> Option<u32> {
    if x < 0.0 || y < 0.0 || x >= BOARD_PX || y >= BOARD_PX {
        return None;
    }
    let col = (x / SQUARE) as u32;
    let row = (y / SQUARE) as u32;
    Some(row * 8 + col)
}

/// Draw `text` centred (horizontally) on `(cx, cy)` using the bundled font.
fn draw_centered(text: &str, cx: f32, cy: f32, size: u16, color: Color, font: Option<&Font>) {
    let dim = measure_text(text, font, size, 1.0);
    draw_text_ex(
        text,
        cx - dim.width / 2.0,
        cy + dim.height / 2.0,
        TextParams {
            font,
            font_size: size,
            color,
            ..Default::default()
        },
    );
}

pub fn render(board: &Board, ui: &UiDetails, font: Option<&Font>) {
    clear_background(col_bg());

    let turn_color = if board.black_to_move {
        col_black()
    } else {
        col_white()
    };

    // Side-to-move sees its own discs as `player`; map to absolute colours.
    let (black_bits, white_bits) = if board.black_to_move {
        (board.position.player, board.position.opponent)
    } else {
        (board.position.opponent, board.position.player)
    };
    let moves = board.position.get_moves();

    let best_score = ui.evaluations.iter().map(|e| e.score).max();

    for index in 0..64u32 {
        let bit = 1u64 << index;
        let (cx, cy) = center(index);

        if black_bits & bit != 0 {
            draw_circle(cx, cy, DISC_R, col_black());
        } else if white_bits & bit != 0 {
            draw_circle(cx, cy, DISC_R, col_white());
        }

        if ui.played_move == Some(index) {
            draw_circle(cx, cy, DISC_R, col_played());
        }

        if let Some(eval) = ui.evaluations.iter().find(|e| e.cell == index) {
            draw_eval(
                cx,
                cy,
                turn_color,
                eval,
                best_score == Some(eval.score),
                font,
            );
        } else if moves & bit != 0 {
            draw_circle(cx, cy, MOVE_R, turn_color);
        }
    }

    draw_graph(ui, font);
}

fn draw_eval(cx: f32, cy: f32, color: Color, eval: &Eval, is_best: bool, font: Option<&Font>) {
    let size: u16 = if eval.score.abs() < 100 {
        26
    } else if eval.score.abs() < 1000 {
        20
    } else {
        14
    };
    draw_centered(&eval.score.to_string(), cx, cy, size, color, font);
    if is_best {
        draw_circle_lines(cx, cy, SQUARE / 2.0 - 8.0, 1.0, color);
    }
    if let Some(level) = eval.level {
        draw_text_ex(
            level.to_string(),
            cx + SQUARE / 8.0,
            cy + SQUARE / 4.0,
            TextParams {
                font,
                font_size: 11,
                color,
                ..Default::default()
            },
        );
    }
}

/// Black-POV evaluation graph along the bottom (pgn mode only).
fn draw_graph(ui: &UiDetails, font: Option<&Font>) {
    if ui.graph.is_empty() {
        return;
    }

    let big_margin = 40.0;
    let small_margin = 10.0;
    let x_min = big_margin;
    let y_min = BOARD_PX + small_margin;
    let x_max = BOARD_PX - small_margin;
    let y_max = BOARD_PX + GRAPH_PX - small_margin;

    draw_rectangle(x_min, y_min, x_max - x_min, y_max - y_min, col_gray());

    let valid: Vec<i32> = ui.graph.iter().filter_map(|p| p.map(|(_, s)| s)).collect();
    if valid.is_empty() {
        return;
    }

    let min_score = valid.iter().copied().min().unwrap_or(-4).min(-4) - 2;
    let max_score = valid.iter().copied().max().unwrap_or(4).max(4) + 2;
    let range = (max_score - min_score) as f32;

    let interval = match max_score - min_score {
        r if r <= 20 => 4,
        r if r <= 40 => 8,
        r if r <= 80 => 16,
        _ => 32,
    };

    let mut score = -64;
    while score <= 64 {
        if score >= min_score && score <= max_score {
            let y = y_min + (y_max - y_min) * (max_score - score) as f32 / range;
            draw_line(x_min, y, x_max, y, 1.0, col_score_line());

            let (text, text_color) = if score == 0 {
                ("0".to_string(), col_gray())
            } else if score < 0 {
                (format!("+{}", score.abs()), col_white())
            } else {
                (format!("+{score}"), col_black())
            };
            draw_centered(&text, x_min / 2.0, y, 11, text_color, font);
        }
        score += interval;
    }

    let n = ui.graph.len();
    let mut prev: Option<(f32, f32)> = None;
    let mut dots: Vec<(Color, f32, f32, f32)> = Vec::new();
    for (offset, point) in ui.graph.iter().enumerate() {
        let Some((black_to_move, black_score)) = *point else {
            continue;
        };
        let denom = (n.saturating_sub(1)).max(1) as f32;
        let x = x_min + (x_max - x_min) * (offset as f32 / denom);
        let y = y_min + (y_max - y_min) * (max_score - black_score) as f32 / range;
        let radius = if Some(offset) == ui.graph_current {
            5.0
        } else {
            3.0
        };
        let dot_color = if black_to_move {
            col_black()
        } else {
            col_white()
        };
        if let Some((px, py)) = prev {
            draw_line(px, py, x, y, 1.0, col_black());
        }
        prev = Some((x, y));
        dots.push((dot_color, x, y, radius));
    }
    for (color, x, y, radius) in dots {
        draw_circle(x, y, radius, color);
    }
}
