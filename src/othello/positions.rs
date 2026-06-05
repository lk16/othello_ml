// Position type for Othello board positions extracted from games.

use crate::othello::board::Board;
use crate::othello::game::Game;

/// A single board position extracted from a game, with side-to-move information.
#[derive(Debug, Clone)]
pub struct Position {
    pub board: Board,
    /// Which player's turn it is (true = black/player side to move)
    pub black_to_move: bool,
}

/// Extract all positions from loaded games, filtered to those with empties <= max_empties.
pub fn extract_positions(games: &[Game], max_empties: u32) -> Vec<Position> {
    let mut positions = Vec::new();

    for game in games {
        for pos in &game.positions {
            let empties = 64_u32
                .saturating_sub(pos.board.player.count_ones())
                .saturating_sub(pos.board.opponent.count_ones());
            if empties <= max_empties {
                positions.push(pos.clone());
            }
        }
    }

    positions
}
