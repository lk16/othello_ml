use crate::othello::position::Position;

/// Represents the 47 Edax features used for position evaluation.
/// Each feature is a subset of board cells that forms a pattern.
/// The pattern can be any subset: corners, edges, diagonals, rows, columns, etc.
///
/// Features work by converting the pattern of discs in a subset to an index:
/// - Each cell can be empty (0), player disc (1), or opponent disc (2)
/// - Index = sum of (value * 3^position) for each cell in the feature
/// - This gives a trinary index from 0 to 3^N-1 where N is the number of cells
#[derive(Debug, Clone)]
pub struct Features {
    features: Vec<Feature>,
}

#[derive(Debug, Clone)]
pub struct Feature {
    pub cells: Vec<u32>, // Cell indices (0-63)
    pub name: String,
}

impl Features {
    /// The exact 46 Edax pattern features, transcribed verbatim from
    /// `EVAL_F2X` in Edax's `eval.c` (each group of 4 — or 2 for the long
    /// diagonal — is the symmetry orbit of one pattern shape). Cell indexing
    /// matches Edax: `A1=0, B1=1, …, H1=7, A2=8, …, H8=63` (LERF).
    ///
    /// Each group's instances are exact board-symmetry images of one another, so
    /// the same physical pattern always yields the same trinary index across the
    /// group — which lets their weights be tied (see [`Self::symmetry_shapes`]).
    /// That property is checked by the `symmetry_shapes_*` tests, which double as a
    /// transcription guard.
    pub fn edax() -> Self {
        // Coordinate groups copied directly from eval.c `EVAL_F2X` (blank-line
        // groups preserved). `sq("C2")` → cell index.
        let groups: [(&str, &[&[&str]]); 12] = [
            (
                "corner",
                &[
                    &["A1", "B1", "A2", "B2", "C1", "A3", "C2", "B3", "C3"],
                    &["H1", "G1", "H2", "G2", "F1", "H3", "F2", "G3", "F3"],
                    &["A8", "A7", "B8", "B7", "A6", "C8", "B6", "C7", "C6"],
                    &["H8", "H7", "G8", "G7", "H6", "F8", "G6", "F7", "F6"],
                ],
            ),
            (
                "edge_a",
                &[
                    &["A5", "A4", "A3", "A2", "A1", "B2", "B1", "C1", "D1", "E1"],
                    &["H5", "H4", "H3", "H2", "H1", "G2", "G1", "F1", "E1", "D1"],
                    &["A4", "A5", "A6", "A7", "A8", "B7", "B8", "C8", "D8", "E8"],
                    &["H4", "H5", "H6", "H7", "H8", "G7", "G8", "F8", "E8", "D8"],
                ],
            ),
            (
                "edge_2x",
                &[
                    &["B2", "A1", "B1", "C1", "D1", "E1", "F1", "G1", "H1", "G2"],
                    &["B7", "A8", "B8", "C8", "D8", "E8", "F8", "G8", "H8", "G7"],
                    &["B2", "A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "B7"],
                    &["G2", "H1", "H2", "H3", "H4", "H5", "H6", "H7", "H8", "G7"],
                ],
            ),
            (
                "edge_x",
                &[
                    &["A1", "C1", "D1", "C2", "D2", "E2", "F2", "E1", "F1", "H1"],
                    &["A8", "C8", "D8", "C7", "D7", "E7", "F7", "E8", "F8", "H8"],
                    &["A1", "A3", "A4", "B3", "B4", "B5", "B6", "A5", "A6", "A8"],
                    &["H1", "H3", "H4", "G3", "G4", "G5", "G6", "H5", "H6", "H8"],
                ],
            ),
            (
                "line2",
                &[
                    &["A2", "B2", "C2", "D2", "E2", "F2", "G2", "H2"],
                    &["A7", "B7", "C7", "D7", "E7", "F7", "G7", "H7"],
                    &["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8"],
                    &["G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8"],
                ],
            ),
            (
                "line3",
                &[
                    &["A3", "B3", "C3", "D3", "E3", "F3", "G3", "H3"],
                    &["A6", "B6", "C6", "D6", "E6", "F6", "G6", "H6"],
                    &["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8"],
                    &["F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8"],
                ],
            ),
            (
                "line4",
                &[
                    &["A4", "B4", "C4", "D4", "E4", "F4", "G4", "H4"],
                    &["A5", "B5", "C5", "D5", "E5", "F5", "G5", "H5"],
                    &["D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8"],
                    &["E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8"],
                ],
            ),
            (
                "diag8",
                &[
                    &["A1", "B2", "C3", "D4", "E5", "F6", "G7", "H8"],
                    &["A8", "B7", "C6", "D5", "E4", "F3", "G2", "H1"],
                ],
            ),
            (
                "diag7",
                &[
                    &["B1", "C2", "D3", "E4", "F5", "G6", "H7"],
                    &["H2", "G3", "F4", "E5", "D6", "C7", "B8"],
                    &["A2", "B3", "C4", "D5", "E6", "F7", "G8"],
                    &["G1", "F2", "E3", "D4", "C5", "B6", "A7"],
                ],
            ),
            (
                "diag6",
                &[
                    &["C1", "D2", "E3", "F4", "G5", "H6"],
                    &["A3", "B4", "C5", "D6", "E7", "F8"],
                    &["F1", "E2", "D3", "C4", "B5", "A6"],
                    &["H3", "G4", "F5", "E6", "D7", "C8"],
                ],
            ),
            (
                "diag5",
                &[
                    &["D1", "E2", "F3", "G4", "H5"],
                    &["A4", "B5", "C6", "D7", "E8"],
                    &["E1", "D2", "C3", "B4", "A5"],
                    &["H4", "G5", "F6", "E7", "D8"],
                ],
            ),
            (
                "diag4",
                &[
                    &["D1", "C2", "B3", "A4"],
                    &["A5", "B6", "C7", "D8"],
                    &["E1", "F2", "G3", "H4"],
                    &["H5", "G6", "F7", "E8"],
                ],
            ),
        ];

        let mut features = Vec::new();
        for (name, instances) in groups {
            for (i, coords) in instances.iter().enumerate() {
                features.push(Feature::from_coords(&format!("{name}_{i}"), coords));
            }
        }
        Features { features }
    }

    /// Extract feature indices from a board position
    /// Returns vector of indices for each feature, where each index represents
    /// the configuration of that feature's cells
    pub fn extract(&self, board: &Position) -> Vec<u32> {
        self.features
            .iter()
            .map(|feature| feature.extract_index(board))
            .collect()
    }

    /// Number of features (normally 47 for the Edax set).
    pub fn count(&self) -> usize {
        self.features.len()
    }

    /// Group features into symmetry **shapes** for weight tying.
    ///
    /// Returns `feature_to_shape` (one shape id per feature) and the shape count.
    /// Two features share a shape iff their *ordered* cell-lists are images of one
    /// another under one of the 8 board symmetries (dihedral group D4). When that
    /// holds, `index(f_j, transform_k(P)) == index(f_i, P)` for the mapping symmetry
    /// `k` (the cells line up position-for-position), so the same physical pattern
    /// always lands on the same trinary index — which is exactly what lets their
    /// weights be tied into one shared table (Edax-style mirror-packing).
    ///
    /// Shapes are numbered by first appearance in feature order, so shape 0 contains
    /// feature 0. Derived deterministically, so save/load can reconstruct it without
    /// storing the mapping.
    pub fn symmetry_shapes(&self) -> (Vec<usize>, usize) {
        // Induced cell permutations: cell_perm[k][c] = image of cell c under the
        // k-th board symmetry (read off by transforming the single-bit board 1<<c).
        let cell_perm: [[u32; 64]; 8] = std::array::from_fn(|k| {
            std::array::from_fn(|c| {
                crate::othello::position::board_symmetry(1u64 << c, k).trailing_zeros()
            })
        });

        // Canonical key of a feature = the lexicographically smallest image of its
        // ordered cell-list over the 8 symmetries. Equal keys ⇒ same shape.
        let canon = |cells: &[u32]| -> Vec<u32> {
            (0..8)
                .map(|k| {
                    cells
                        .iter()
                        .map(|&c| cell_perm[k][c as usize])
                        .collect::<Vec<u32>>()
                })
                .min()
                .unwrap_or_default()
        };

        let mut keys: Vec<Vec<u32>> = Vec::new();
        let mut feature_to_shape = Vec::with_capacity(self.features.len());
        for feature in &self.features {
            let key = canon(&feature.cells);
            let id = keys.iter().position(|k| *k == key).unwrap_or_else(|| {
                keys.push(key);
                keys.len() - 1
            });
            feature_to_shape.push(id);
        }
        (feature_to_shape, keys.len())
    }

    /// Look up a feature by index.
    pub fn get(&self, idx: usize) -> Option<&Feature> {
        self.features.get(idx)
    }

    /// All features.
    pub fn all(&self) -> &[Feature] {
        &self.features
    }
}

/// Parse an algebraic square (e.g. `"C2"`) into a cell index (LERF: `A1=0`,
/// `H1=7`, `A2=8`, …, `H8=63`). Panics on malformed input — only used for the
/// compile-time-constant feature table.
fn sq(coord: &str) -> u32 {
    let b = coord.as_bytes();
    assert!(b.len() == 2, "bad square {coord:?}");
    let file = (b[0].to_ascii_uppercase() - b'A') as u32;
    let rank = (b[1] - b'1') as u32;
    assert!(file < 8 && rank < 8, "square out of range {coord:?}");
    rank * 8 + file
}

impl Feature {
    pub fn new(name: &str, cells: Vec<u32>) -> Self {
        Feature {
            cells,
            name: name.to_string(),
        }
    }

    /// Build a feature from algebraic coordinates (as written in Edax `eval.c`).
    pub fn from_coords(name: &str, coords: &[&str]) -> Self {
        Feature::new(name, coords.iter().map(|c| sq(c)).collect())
    }

    /// Extract this feature's index from the board
    /// Each cell contributes a trinary value: 0 (empty), 1 (player), 2 (opponent)
    /// Index = sum of (value * 3^position) for each cell
    pub fn extract_index(&self, board: &Position) -> u32 {
        let mut index = 0u32;
        let mut power = 1u32;

        for &cell in &self.cells {
            let value = match board.get_cell(cell) {
                crate::othello::position::Cell::Empty => 0,
                crate::othello::position::Cell::Player => 1,
                crate::othello::position::Cell::Opponent => 2,
            };
            index += value * power;
            power *= 3;
        }

        index
    }

    /// Maximum index for this feature (3^num_cells - 1)
    pub fn max_index(&self) -> u32 {
        3u32.pow(self.cells.len() as u32) - 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feature_extraction() {
        let board = Position::initial();
        let feature = Feature::new("test", vec![0, 1, 2]);

        // Position has cells 0,1,2 empty initially
        let idx = feature.extract_index(&board);
        assert_eq!(idx, 0); // All empty = index 0
    }

    #[test]
    fn test_feature_max_index() {
        let feature = Feature::new("test3", vec![0, 1, 2]);
        assert_eq!(feature.max_index(), 26); // 3^3 - 1 = 26

        let feature = Feature::new("test2", vec![0, 1]);
        assert_eq!(feature.max_index(), 8); // 3^2 - 1 = 8
    }

    #[test]
    fn test_edax_features() {
        let features = Features::edax();
        assert!(features.count() > 0);
    }

    /// Transcription guard: the 46 Edax features must form exactly 12 clean
    /// symmetry orbits (one per pattern shape), and each contiguous group of 4 (or
    /// 2 for the long diagonal) defined in `edax()` must be exactly one shape. A
    /// mis-transcribed cell would break an orbit and trip this.
    #[test]
    fn edax_features_form_clean_symmetry_orbits() {
        let features = Features::edax();
        assert_eq!(features.count(), 46);
        let (map, n_shapes) = features.symmetry_shapes();
        assert_eq!(n_shapes, 12, "expected 12 pattern shapes");

        // Group sizes in edax() order: 11 groups of 4, plus diag8 (size 2).
        let expected_sizes = [4, 4, 4, 4, 4, 4, 4, 2, 4, 4, 4, 4];
        let mut f = 0;
        for (shape, size) in expected_sizes.iter().enumerate() {
            for _ in 0..*size {
                assert_eq!(map[f], shape, "feature {f} not in expected shape {shape}");
                f += 1;
            }
        }
        assert_eq!(f, features.count());
    }

    /// Every feature in a derived shape must share the same cell count (so they can
    /// share one `3^cells` weight table).
    #[test]
    fn symmetry_shapes_consistent_cell_counts() {
        let features = Features::edax();
        let (map, n_shapes) = features.symmetry_shapes();
        assert_eq!(map.len(), features.count());
        for s in 0..n_shapes {
            let members: Vec<usize> = (0..map.len()).filter(|&f| map[f] == s).collect();
            let len0 = features.get(members[0]).unwrap().cells.len();
            for &m in &members {
                assert_eq!(features.get(m).unwrap().cells.len(), len0);
            }
        }
    }

    /// Tying is only correct if, for two features in the same shape, the same
    /// physical pattern produces the same trinary index. Concretely: there is a
    /// board symmetry `k` with `index(f_i, P) == index(f_j, transform_k(P))` for
    /// all `P`. Verify it directly over a spread of positions.
    #[test]
    fn symmetric_features_share_pattern_indices() {
        use crate::othello::position::board_symmetry;
        let features = Features::edax();
        let (map, n_shapes) = features.symmetry_shapes();

        // A spread of positions via random legal play.
        let mut positions = vec![Position::initial()];
        let mut pos = Position::initial();
        let mut s = 0x1234_5678u32;
        for _ in 0..40 {
            let moves = pos.get_moves();
            if moves == 0 {
                pos = pos.pass_move();
                if pos.get_moves() == 0 {
                    break;
                }
                continue;
            }
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            let mut pick = s % moves.count_ones();
            let mut m = moves;
            let cell = loop {
                let c = m.trailing_zeros();
                m &= m - 1;
                if pick == 0 {
                    break c;
                }
                pick -= 1;
            };
            pos = pos.do_move(cell);
            positions.push(pos);
        }

        for s in 0..n_shapes {
            let members: Vec<usize> = (0..map.len()).filter(|&f| map[f] == s).collect();
            let rep = features.get(members[0]).unwrap();
            for &m in &members[1..] {
                let f = features.get(m).unwrap();
                // Find the symmetry k mapping the representative's cells to f's.
                let k = (0..8)
                    .find(|&k| {
                        rep.cells
                            .iter()
                            .zip(&f.cells)
                            .all(|(&c, &d)| board_symmetry(1u64 << c, k).trailing_zeros() == d)
                    })
                    .expect("shape members must be cell-wise symmetry images");
                for p in &positions {
                    let tp = Position {
                        player: board_symmetry(p.player, k),
                        opponent: board_symmetry(p.opponent, k),
                    };
                    assert_eq!(
                        rep.extract_index(p),
                        f.extract_index(&tp),
                        "shape {s}: {} vs {} disagree under symmetry {k}",
                        rep.name,
                        f.name
                    );
                }
            }
        }
    }
}
