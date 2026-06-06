use crate::othello::position::Position;

#[derive(Debug, Clone)]
pub struct Board {
    pub position: Position,
    pub black_to_move: bool,
}

impl Board {
    pub fn empties(&self) -> u32 {
        self.position.empties()
    }

    /// Render the board as a string. When `show_move_labels` is true, legal
    /// moves are labelled a, b, c… for user input; otherwise they appear as ·.
    pub fn show(&self, show_move_labels: bool) -> String {
        let moves = self.position.get_moves();
        let mut out = String::new();

        out.push_str("+-a-b-c-d-e-f-g-h-+\n");

        let mut label = b'a';
        for y in 0..8u32 {
            out.push_str(&format!("{} ", y + 1));
            for x in 0..8u32 {
                let cell = y * 8 + x;
                let bit = 1u64 << cell;

                if self.position.player & bit != 0 {
                    out.push_str("○ ");
                } else if self.position.opponent & bit != 0 {
                    out.push_str("● ");
                } else if moves & bit != 0 {
                    if show_move_labels {
                        out.push(label as char);
                        out.push(' ');
                        label += 1;
                    } else {
                        out.push_str("· ");
                    }
                } else {
                    out.push_str("  ");
                }
            }
            out.push_str("|\n");
        }

        out.push_str("+-----------------+");
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_board_empties_initial() {
        let board = Board {
            position: Position::initial(),
            black_to_move: true,
        };
        assert_eq!(board.empties(), 60);
    }

    #[test]
    fn test_board_empties_empty_board() {
        let board = Board {
            position: Position::new(),
            black_to_move: true,
        };
        assert_eq!(board.empties(), 64);
    }

    #[test]
    fn test_board_empties_full_board() {
        let pos = Position {
            player: u64::MAX,
            opponent: 0,
        };
        let board = Board {
            position: pos,
            black_to_move: true,
        };
        assert_eq!(board.empties(), 0);
    }

    #[test]
    fn test_show_initial_no_labels() {
        let board = Board {
            position: Position::initial(),
            black_to_move: true,
        };
        let out = board.show(false);
        assert!(out.starts_with("+-a-b-c-d-e-f-g-h-+\n"));
        assert!(out.ends_with("+-----------------+"));
        assert!(out.contains("· "), "legal moves should appear as ·");
    }

    #[test]
    fn test_show_initial_with_labels() {
        let board = Board {
            position: Position::initial(),
            black_to_move: true,
        };
        let out = board.show(true);
        assert!(out.contains("a "), "first legal move should be labelled a");
        assert!(out.contains("b "), "second legal move should be labelled b");
    }

    #[test]
    fn test_show_label_count_matches_moves() {
        let board = Board {
            position: Position::initial(),
            black_to_move: true,
        };
        let out = board.show(true);
        let move_count = board.position.get_moves().count_ones() as usize;
        assert_eq!(move_count, 4, "initial position has 4 legal moves");
        for label in b'a'..(b'a' + move_count as u8) {
            let s = format!("{} ", label as char);
            assert!(out.contains(&s), "missing label {}", label as char);
        }
    }
}
