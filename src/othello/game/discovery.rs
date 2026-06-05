// File type detection, recursive directory scanning, and top-level game loading.

use std::fs;
use std::path::Path;

use super::pgn::read_pgn_file;
use super::wthor::read_wthor_file;

/// Recognised game file formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FileType {
    Wthor,
    Pgn,
}

/// Determine file type from extension. Returns `None` for unrecognised extensions.
fn file_type(path: &Path) -> Option<FileType> {
    match path.extension().and_then(|e| e.to_str()) {
        Some("wtb") | Some("WTH") => Some(FileType::Wthor),
        Some("pgn") | Some("PGN") | Some("txt") | Some("TXT") => Some(FileType::Pgn),
        _ => None,
    }
}

/// Recursively collect all .wtb and .pgn/.txt files from a directory.
fn collect_game_files(dir: &Path) -> Result<Vec<std::path::PathBuf>, String> {
    let mut files = Vec::new();

    let entries = fs::read_dir(dir)
        .map_err(|e| format!("Failed to read directory {}: {}", dir.display(), e))?;

    for entry in entries {
        let entry = entry.map_err(|e| e.to_string())?;
        let path = entry.path();

        if path.is_dir() {
            files.extend(collect_game_files(&path)?);
        } else if file_type(&path).is_some() {
            files.push(path);
        }
    }

    Ok(files)
}

/// Load games from a list of paths (files or directories).
/// Directories are scanned recursively for .wtb, .pgn, .txt files.
pub fn load_games(paths: &[String]) -> Result<Vec<super::Game>, String> {
    let mut all_file_paths: Vec<std::path::PathBuf> = Vec::new();

    for path_str in paths {
        let path = Path::new(path_str);

        if path.is_dir() {
            all_file_paths.extend(collect_game_files(path)?);
        } else if path.is_file() {
            all_file_paths.push(path.to_path_buf());
        } else {
            eprintln!("Warning: {} does not exist, skipping", path.display());
        }
    }

    if all_file_paths.is_empty() {
        return Err("No game files found".to_string());
    }

    let mut all_games = Vec::new();
    for file_path in &all_file_paths {
        if let Some(ft) = file_type(file_path) {
            eprintln!("Loading {} ({:?})...", file_path.display(), ft);

            let games = match ft {
                FileType::Wthor => read_wthor_file(file_path)?,
                FileType::Pgn => read_pgn_file(file_path)?,
            };

            eprintln!(
                "  Loaded {} games, {} positions",
                games.len(),
                games.iter().map(|g| g.positions.len()).sum::<usize>()
            );
            all_games.extend(games);
        } else {
            eprintln!("  Unknown file type, skipping");
        }
    }

    eprintln!(
        "Total: {} games from {} files",
        all_games.len(),
        all_file_paths.len()
    );
    Ok(all_games)
}
