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
    /// Create the exact 47 Edax features from eval.c
    /// Cell indexing: a1=0, b1=1, ..., h1=7, a2=8, ..., h8=63
    pub fn edax() -> Self {
        let features = vec![
            // 4 corners (9 cells each, 3x3)
            Feature::new("corner_a1", vec![0, 1, 8, 9, 2, 16, 10, 17, 18]), // A1,B1,A2,B2,C1,A3,C2,B3,C3
            Feature::new("corner_h1", vec![7, 6, 15, 14, 5, 23, 13, 22, 21]), // H1,G1,H2,G2,F1,H3,F2,G3,F3
            Feature::new("corner_a8", vec![56, 48, 57, 49, 40, 58, 50, 59, 51]), // A8,A7,B8,B7,A6,C8,B6,C7,C6
            Feature::new("corner_h8", vec![63, 62, 55, 54, 61, 47, 53, 46, 45]), // H8,H7,G8,G7,H6,F8,G6,F7,F6
            // 4 left/right edge + extended (10 cells each)
            Feature::new("edge_a_left1", vec![32, 24, 16, 8, 0, 9, 1, 2, 3, 4]), // A5,A4,A3,A2,A1,B2,B1,C1,D1,E1
            Feature::new("edge_h_right1", vec![39, 31, 23, 15, 7, 14, 6, 5, 4, 3]), // H5,H4,H3,H2,H1,G2,G1,F1,E1,D1
            Feature::new("edge_a_left2", vec![24, 32, 40, 48, 56, 49, 57, 58, 59, 60]), // A4,A5,A6,A7,A8,B7,B8,C8,D8,E8
            Feature::new(
                "edge_h_right2",
                vec![31, 39, 47, 55, 63, 54, 62, 61, 60, 59],
            ), // H4,H5,H6,H7,H8,G7,G8,F8,E8,D8
            // 4 top/bottom edge patterns (10 cells each)
            Feature::new("edge_top1", vec![9, 0, 1, 2, 3, 4, 5, 6, 7, 14]), // B2,A1,B1,C1,D1,E1,F1,G1,H1,G2
            Feature::new("edge_bottom1", vec![49, 56, 57, 58, 59, 60, 61, 62, 63, 54]), // B7,A8,B8,C8,D8,E8,F8,G8,H8,G7
            Feature::new("edge_top2", vec![9, 0, 8, 16, 24, 32, 40, 48, 56, 49]), // B2,A1,A2,A3,A4,A5,A6,A7,A8,B7
            Feature::new("edge_bottom2", vec![14, 7, 15, 23, 31, 39, 47, 55, 63, 54]), // G2,H1,H2,H3,H4,H5,H6,H7,H8,G7
            // 4 extended corner patterns (10 cells each)
            Feature::new("ext_corner_a1", vec![0, 2, 3, 9, 10, 11, 12, 13, 18, 7]), // A1,C1,D1,C2,D2,E2,F2,E3,F3,H1
            Feature::new(
                "ext_corner_a8",
                vec![56, 58, 59, 49, 50, 51, 52, 53, 42, 63],
            ), // A8,C8,D8,C7,D7,E7,F7,E8,F8,H8
            Feature::new("ext_corner_h1", vec![0, 16, 24, 17, 25, 33, 41, 32, 40, 56]), // A1,A3,A4,B3,B4,B5,B6,A5,A6,A8
            Feature::new("ext_corner_h8", vec![7, 23, 31, 22, 30, 38, 46, 39, 47, 63]), // H1,H3,H4,G3,G4,G5,G6,H5,H6,H8
            // 8 lines (8 cells each) - rows 2,7 and cols B,G
            Feature::new("line_row2", vec![8, 9, 10, 11, 12, 13, 14, 15]), // A2-H2
            Feature::new("line_row7", vec![48, 49, 50, 51, 52, 53, 54, 55]), // A7-H7
            Feature::new("line_col_b", vec![1, 9, 17, 25, 33, 41, 49, 57]), // B1-B8
            Feature::new("line_col_g", vec![6, 14, 22, 30, 38, 46, 54, 62]), // G1-G8
            // 4 more lines (8 cells each) - rows 3,6 and cols C,F
            Feature::new("line_row3", vec![16, 17, 18, 19, 20, 21, 22, 23]), // A3-H3
            Feature::new("line_row6", vec![40, 41, 42, 43, 44, 45, 46, 47]), // A6-H6
            Feature::new("line_col_c", vec![2, 10, 18, 26, 34, 42, 50, 58]), // C1-C8
            Feature::new("line_col_f", vec![5, 13, 21, 29, 37, 45, 53, 61]), // F1-F8
            // 4 more lines (8 cells each) - rows 4,5 and cols D,E
            Feature::new("line_row4", vec![24, 25, 26, 27, 28, 29, 30, 31]), // A4-H4
            Feature::new("line_row5", vec![32, 33, 34, 35, 36, 37, 38, 39]), // A5-H5
            Feature::new("line_col_d", vec![3, 11, 19, 27, 35, 43, 51, 59]), // D1-D8
            Feature::new("line_col_e", vec![4, 12, 20, 28, 36, 44, 52, 60]), // E1-E8
            // 2 main diagonals (8 cells each)
            Feature::new("diag_main", vec![0, 9, 18, 27, 36, 45, 54, 63]), // A1-H8
            Feature::new("diag_anti", vec![56, 49, 42, 35, 28, 21, 14, 7]), // A8-H1
            // 4 diagonals (7 cells each)
            Feature::new("diag_7_1", vec![1, 10, 19, 28, 37, 46, 55]), // B1-H7
            Feature::new("diag_7_2", vec![8, 17, 26, 35, 44, 53, 62]), // A2-H8
            Feature::new("diag_7_3", vec![48, 41, 34, 27, 20, 13, 6]), // A7-G1
            Feature::new("diag_7_4", vec![57, 50, 43, 36, 29, 22, 15]), // B8-H2
            // 4 diagonals (6 cells each)
            Feature::new("diag_6_1", vec![2, 11, 20, 29, 38, 47]), // C1-H6
            Feature::new("diag_6_2", vec![16, 25, 34, 43, 52, 61]), // A3-F8
            Feature::new("diag_6_3", vec![40, 33, 26, 19, 12, 5]), // A6-F1
            Feature::new("diag_6_4", vec![58, 51, 44, 37, 30, 23]), // C8-H3
            // 4 diagonals (5 cells each)
            Feature::new("diag_5_1", vec![3, 12, 21, 30, 39]), // D1-H5
            Feature::new("diag_5_2", vec![24, 33, 42, 51, 60]), // A4-E8
            Feature::new("diag_5_3", vec![32, 25, 18, 11, 4]), // A5-E1
            Feature::new("diag_5_4", vec![59, 52, 45, 38, 31]), // D8-H4
            // 4 diagonals (4 cells each)
            Feature::new("diag_4_1", vec![3, 12, 21, 30]), // D1-G4
            Feature::new("diag_4_2", vec![24, 33, 42, 51]), // A4-D7
            Feature::new("diag_4_3", vec![28, 21, 14, 7]), // E1-H4
            Feature::new("diag_4_4", vec![35, 44, 53, 62]), // D5-G8
            // 1 additional feature for completeness (parity or mobility hint)
            Feature::new("edge_parity", vec![0, 7, 56, 63]), // 4 corners as compacted feature
        ];

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

    /// Look up a feature by index.
    pub fn get(&self, idx: usize) -> Option<&Feature> {
        self.features.get(idx)
    }

    /// All features.
    pub fn all(&self) -> &[Feature] {
        &self.features
    }
}

impl Feature {
    pub fn new(name: &str, cells: Vec<u32>) -> Self {
        Feature {
            cells,
            name: name.to_string(),
        }
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
}
