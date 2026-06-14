//! Leaf solvers for the last few empties (ported from Edax `endgame.c`).
//!
//! `solve_1`..`solve_4` search a near-full board directly, avoiding the `Vec` /
//! `get_moves` / `Position` machinery of the general search. `solve_1` is a pure
//! function; `solve_2`..`solve_4` and the `solve_leaf` dispatcher are [`Search`]
//! methods so they can count nodes. Every visited position counts exactly once.

use super::count_flip::count_last_flip;
use super::search::Search;
use super::SCORE_MIN;
use crate::othello::position::Position;

/// Game-over score when no moves remain: winner gets all empties.
pub(super) fn solve_game_over(player: u64, n_empties: u32) -> i32 {
    let n = n_empties as i32;
    let diff = 2 * player.count_ones() as i32 - 64 + n; // player_discs - opponent_discs
    if diff > 0 {
        diff + n
    } else if diff < 0 {
        diff - n
    } else {
        0
    }
}

/// 1-empty leaf solver: exact score from `player`'s perspective for a full board
/// with a single empty at `sq`. The move is forced, so no window is needed; a
/// pure function (the calling [`Search`] method counts this node).
pub(super) fn solve_1(player: u64, sq: u32) -> i32 {
    // Differential after `player` plays `sq` before flips: +1 disc; each flip
    // then shifts it by a further 2.
    let score_base = 2 * player.count_ones() as i32 - 62;

    let n_flips = count_last_flip(sq, player);
    if n_flips != 0 {
        return score_base + n_flips;
    }

    // Player cannot play and passes; the opponent is then forced to play.
    let opponent = !player & !(1u64 << sq);
    let n_flips_opp = count_last_flip(sq, opponent);
    if n_flips_opp != 0 {
        return score_base - 2 - n_flips_opp;
    }

    solve_game_over(player, 1)
}

impl Search {
    /// 2-empty leaf solver: negamax alpha-beta over empties `x1`, `x2`. Counts
    /// itself plus one per `solve_1` child visited.
    pub(super) fn solve_2(
        &mut self,
        player: u64,
        opponent: u64,
        mut alpha: i32,
        beta: i32,
        x1: u32,
        x2: u32,
    ) -> i32 {
        self.nodes += 1;
        const NONE: i32 = SCORE_MIN - 1; // below any real score
        let mut best = NONE;

        let f1 = Position::flip_mask(x1, player, opponent);
        if f1 != 0 {
            let moved = player | f1 | (1u64 << x1);
            let child_player = opponent & !moved; // opponent to move, only x2 empty
            self.nodes += 1; // solve_1 leaf (x2)
            best = -solve_1(child_player, x2);
            if best > alpha {
                alpha = best;
            }
        }

        if alpha < beta {
            let f2 = Position::flip_mask(x2, player, opponent);
            if f2 != 0 {
                let moved = player | f2 | (1u64 << x2);
                let child_player = opponent & !moved; // opponent to move, only x1 empty
                self.nodes += 1; // solve_1 leaf (x1)
                let s = -solve_1(child_player, x1);
                if s > best {
                    best = s;
                }
            }
        }

        if best != NONE {
            return best;
        }

        // Player has no move and passes; game over if the opponent also cannot.
        if Position::flip_mask(x1, opponent, player) == 0
            && Position::flip_mask(x2, opponent, player) == 0
        {
            return solve_game_over(player, 2);
        }
        -self.solve_2(opponent, player, -beta, -alpha, x1, x2)
    }

    /// 3-empty leaf solver: fail-soft negamax over `x1`, `x2`, `x3`, recursing
    /// into [`Search::solve_2`].
    pub(super) fn solve_3(
        &mut self,
        player: u64,
        opponent: u64,
        mut alpha: i32,
        beta: i32,
        x1: u32,
        x2: u32,
        x3: u32,
    ) -> i32 {
        self.nodes += 1;
        const NONE: i32 = SCORE_MIN - 1;
        let mut best = NONE;

        // (sq, the other two empties) per candidate move.
        for &(sq, a, b) in &[(x1, x2, x3), (x2, x1, x3), (x3, x1, x2)] {
            if alpha >= beta {
                break;
            }
            let f = Position::flip_mask(sq, player, opponent);
            if f != 0 {
                let moved = player | f | (1u64 << sq);
                let child_player = opponent & !moved;
                let s = -self.solve_2(child_player, moved, -beta, -alpha, a, b);
                if s > best {
                    best = s;
                    if best > alpha {
                        alpha = best;
                    }
                }
            }
        }

        if best != NONE {
            return best;
        }

        if Position::flip_mask(x1, opponent, player) == 0
            && Position::flip_mask(x2, opponent, player) == 0
            && Position::flip_mask(x3, opponent, player) == 0
        {
            return solve_game_over(player, 3);
        }
        -self.solve_3(opponent, player, -beta, -alpha, x1, x2, x3)
    }

    /// 4-empty leaf solver: fail-soft negamax over the four empties, recursing
    /// into [`Search::solve_3`].
    pub(super) fn solve_4(
        &mut self,
        player: u64,
        opponent: u64,
        mut alpha: i32,
        beta: i32,
        x1: u32,
        x2: u32,
        x3: u32,
        x4: u32,
    ) -> i32 {
        self.nodes += 1;
        const NONE: i32 = SCORE_MIN - 1;
        let mut best = NONE;

        // (sq, the other three empties) per candidate move.
        for &(sq, a, b, c) in &[
            (x1, x2, x3, x4),
            (x2, x1, x3, x4),
            (x3, x1, x2, x4),
            (x4, x1, x2, x3),
        ] {
            if alpha >= beta {
                break;
            }
            let f = Position::flip_mask(sq, player, opponent);
            if f != 0 {
                let moved = player | f | (1u64 << sq);
                let child_player = opponent & !moved;
                let s = -self.solve_3(child_player, moved, -beta, -alpha, a, b, c);
                if s > best {
                    best = s;
                    if best > alpha {
                        alpha = best;
                    }
                }
            }
        }

        if best != NONE {
            return best;
        }

        if Position::flip_mask(x1, opponent, player) == 0
            && Position::flip_mask(x2, opponent, player) == 0
            && Position::flip_mask(x3, opponent, player) == 0
            && Position::flip_mask(x4, opponent, player) == 0
        {
            return solve_game_over(player, 4);
        }
        -self.solve_4(opponent, player, -beta, -alpha, x1, x2, x3, x4)
    }

    /// Leaf dispatcher for at most four empties, routing to `solve_1`..`solve_4`
    /// (or `final_score` at game end). The 0/1-empty arms count their node here;
    /// the others count in the solver they call.
    pub(super) fn solve_leaf(
        &mut self,
        pos: &Position,
        alpha: i32,
        beta: i32,
        empties: u32,
    ) -> i32 {
        match empties {
            0 => {
                self.nodes += 1;
                pos.final_score()
            }
            1 => {
                self.nodes += 1;
                let sq = (!pos.occupied()).trailing_zeros();
                solve_1(pos.player, sq)
            }
            2 => {
                let mut empty = !pos.occupied();
                let x1 = empty.trailing_zeros();
                empty &= empty - 1;
                let x2 = empty.trailing_zeros();
                self.solve_2(pos.player, pos.opponent, alpha, beta, x1, x2)
            }
            3 => {
                let mut empty = !pos.occupied();
                let x1 = empty.trailing_zeros();
                empty &= empty - 1;
                let x2 = empty.trailing_zeros();
                empty &= empty - 1;
                let x3 = empty.trailing_zeros();
                self.solve_3(pos.player, pos.opponent, alpha, beta, x1, x2, x3)
            }
            _ => {
                let mut empty = !pos.occupied();
                let x1 = empty.trailing_zeros();
                empty &= empty - 1;
                let x2 = empty.trailing_zeros();
                empty &= empty - 1;
                let x3 = empty.trailing_zeros();
                empty &= empty - 1;
                let x4 = empty.trailing_zeros();
                self.solve_4(pos.player, pos.opponent, alpha, beta, x1, x2, x3, x4)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::testutil::{
        layouts_for, naive_exact, two_empty_layouts, PATTERNS, SQUARES, SQUARES4,
    };
    use super::super::SCORE_MAX;
    use super::*;

    /// Flip mask for `player` playing at `sq` against `opponent` — the reference
    /// the leaf solvers use directly.
    fn flips_for(sq: u32, player: u64, opponent: u64) -> u64 {
        Position::flip_mask(sq, player, opponent)
    }

    // The solve_* helpers take a reused `Search`: it never touches the table, but
    // a fresh `Search` allocates one, so reuse keeps these loops from
    // re-allocating it thousands of times.
    fn run_solve_2(s: &mut Search, player: u64, opponent: u64) -> i32 {
        let mut empty = !(player | opponent);
        let x1 = empty.trailing_zeros();
        empty &= empty - 1;
        let x2 = empty.trailing_zeros();
        s.solve_2(player, opponent, SCORE_MIN, SCORE_MAX, x1, x2)
    }

    fn run_solve_3(s: &mut Search, player: u64, opponent: u64) -> i32 {
        let mut e = !(player | opponent);
        let x1 = e.trailing_zeros();
        e &= e - 1;
        let x2 = e.trailing_zeros();
        e &= e - 1;
        let x3 = e.trailing_zeros();
        s.solve_3(player, opponent, SCORE_MIN, SCORE_MAX, x1, x2, x3)
    }

    fn run_solve_4(s: &mut Search, player: u64, opponent: u64) -> i32 {
        let mut e = !(player | opponent);
        let x1 = e.trailing_zeros();
        e &= e - 1;
        let x2 = e.trailing_zeros();
        e &= e - 1;
        let x3 = e.trailing_zeros();
        e &= e - 1;
        let x4 = e.trailing_zeros();
        s.solve_4(player, opponent, SCORE_MIN, SCORE_MAX, x1, x2, x3, x4)
    }

    #[test]
    fn solve_game_over_player_wins() {
        // 32 player, 30 opponent, 2 empties → player wins, gets the empties.
        let player: u64 = 0x00000000FFFFFFFF;
        let opponent: u64 = 0x3FFFFFFF00000000;
        assert_eq!(player.count_ones(), 32);
        assert_eq!(opponent.count_ones(), 30);
        assert_eq!(solve_game_over(player, 2), 4);
    }

    #[test]
    fn solve_game_over_opponent_wins() {
        // 30 player, 32 opponent, 2 empties → opponent wins.
        let player: u64 = 0x000000003FFFFFFF;
        let opponent: u64 = 0xFFFFFFFF00000000;
        assert_eq!(player.count_ones(), 30);
        assert_eq!(opponent.count_ones(), 32);
        assert_eq!(solve_game_over(player, 2), -4);
    }

    #[test]
    fn solve_game_over_tie() {
        let player: u64 = 0x000000007FFFFFFF;
        let opponent: u64 = 0x7FFFFFFF00000000;
        assert_eq!(player.count_ones(), 31);
        assert_eq!(opponent.count_ones(), 31);
        assert_eq!(solve_game_over(player, 2), 0);
    }

    #[test]
    fn solve_game_over_zero_empties_matches_final_score() {
        let player: u64 = 0x00000000FFFFFFFF;
        let opponent: u64 = 0xFFFFFFFF00000000;
        let pos = Position { player, opponent };
        assert_eq!(solve_game_over(player, 0), pos.final_score());
    }

    #[test]
    fn solve_1_player_forced_play() {
        // Full board, empty at a1; playing a1 flips b1 anchored on c1.
        let sq = 0u32;
        let player = 1u64 << 2; // c1
        let pos = Position {
            player,
            opponent: !player & !(1u64 << sq),
        };
        assert_eq!(pos.empties(), 1);
        assert_eq!(solve_1(player, sq), -58);
        assert_eq!(solve_1(player, sq), naive_exact(&pos));
    }

    #[test]
    fn solve_1_game_over_player_wins() {
        // Lone empty, opponent absent: nobody can flip → game over.
        let sq = 0u32;
        let player = !(1u64 << sq); // 63 discs
        assert_eq!(solve_1(player, sq), 64);
    }

    #[test]
    fn solve_1_matches_naive() {
        for &sq in SQUARES {
            let empty = 1u64 << sq;
            for &pat in PATTERNS {
                let player = pat & !empty;
                let opponent = !player & !empty;
                let pos = Position { player, opponent };
                assert_eq!(
                    solve_1(player, sq),
                    naive_exact(&pos),
                    "player={player:#x} sq={sq}"
                );
            }
        }
    }

    #[test]
    fn count_last_flip_matches_flips_for() {
        // On a full board with one empty, the lookup equals 2× the flip popcount.
        for &sq in SQUARES {
            let empty = 1u64 << sq;
            for &pat in PATTERNS {
                let player = pat & !empty;
                let opponent = !player & !empty;
                let expected = 2 * flips_for(sq, player, opponent).count_ones() as i32;
                assert_eq!(
                    count_last_flip(sq, player),
                    expected,
                    "player={player:#x} sq={sq}"
                );
            }
        }
    }

    #[test]
    fn solve_2_matches_naive() {
        let mut s = Search::new();
        for (player, opponent) in two_empty_layouts() {
            let pos = Position { player, opponent };
            assert_eq!(pos.empties(), 2);
            assert_eq!(
                run_solve_2(&mut s, player, opponent),
                naive_exact(&pos),
                "player={player:#x} opponent={opponent:#x}"
            );
        }
    }

    #[test]
    fn solve_2_respects_window() {
        // Narrow windows must fail soft on the correct side of the bound.
        let mut s = Search::new();
        for (player, opponent) in two_empty_layouts() {
            let truth = naive_exact(&Position { player, opponent });
            let mut e = !(player | opponent);
            let x1 = e.trailing_zeros();
            e &= e - 1;
            let x2 = e.trailing_zeros();

            let lo = truth + 1;
            let r = s.solve_2(player, opponent, lo, lo + 1, x1, x2);
            assert!(r <= lo, "fail-low: r={r} alpha={lo} truth={truth}");
            let hi = truth - 1;
            let r = s.solve_2(player, opponent, hi - 1, hi, x1, x2);
            assert!(r >= hi, "fail-high: r={r} beta={hi} truth={truth}");
        }
    }

    #[test]
    fn solve_3_matches_naive() {
        let mut s = Search::new();
        let n = SQUARES.len();
        for i in 0..n {
            for j in (i + 1)..n {
                for k in (j + 1)..n {
                    let empties = [SQUARES[i], SQUARES[j], SQUARES[k]];
                    for (player, opponent) in layouts_for(&empties) {
                        let pos = Position { player, opponent };
                        assert_eq!(pos.empties(), 3);
                        assert_eq!(
                            run_solve_3(&mut s, player, opponent),
                            naive_exact(&pos),
                            "player={player:#x} opponent={opponent:#x}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn solve_4_matches_naive() {
        let mut s = Search::new();
        let n = SQUARES4.len();
        for i in 0..n {
            for j in (i + 1)..n {
                for k in (j + 1)..n {
                    for l in (k + 1)..n {
                        let empties = [SQUARES4[i], SQUARES4[j], SQUARES4[k], SQUARES4[l]];
                        for (player, opponent) in layouts_for(&empties) {
                            let pos = Position { player, opponent };
                            assert_eq!(pos.empties(), 4);
                            assert_eq!(
                                run_solve_4(&mut s, player, opponent),
                                naive_exact(&pos),
                                "player={player:#x} opponent={opponent:#x}"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn solve_3_and_4_respect_window() {
        // Tight windows must land on the correct side of the bound.
        let check = |player: u64, opponent: u64, solve: &dyn Fn(u64, u64, i32, i32) -> i32| {
            let truth = naive_exact(&Position { player, opponent });
            let lo = truth + 1;
            assert!(solve(player, opponent, lo, lo + 1) <= lo, "fail-low");
            let hi = truth - 1;
            assert!(solve(player, opponent, hi - 1, hi) >= hi, "fail-high");
        };
        // `RefCell` so the `&dyn Fn` closure can reuse one `Search`.
        let s = std::cell::RefCell::new(Search::new());
        let solve3 = |p: u64, o: u64, a: i32, b: i32| {
            let mut e = !(p | o);
            let x1 = e.trailing_zeros();
            e &= e - 1;
            let x2 = e.trailing_zeros();
            e &= e - 1;
            let x3 = e.trailing_zeros();
            s.borrow_mut().solve_3(p, o, a, b, x1, x2, x3)
        };
        let n = SQUARES4.len();
        for i in 0..n {
            for j in (i + 1)..n {
                for k in (j + 1)..n {
                    for (player, opponent) in layouts_for(&[SQUARES4[i], SQUARES4[j], SQUARES4[k]])
                    {
                        check(player, opponent, &solve3);
                    }
                }
            }
        }
    }
}
