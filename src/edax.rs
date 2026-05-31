use std::process::{Command, Stdio};
use std::io::Write;
use std::env;

/// Interface to Edax for getting ground truth evaluations
pub struct EdaxInterface {
    edax_path: String,
}

impl EdaxInterface {
    /// Create a new Edax interface from the EDAX_PATH environment variable
    pub fn new() -> Result<Self, String> {
        let edax_path = env::var("EDAX_PATH")
            .map_err(|_| "EDAX_PATH environment variable not set".to_string())?;

        Ok(EdaxInterface { edax_path })
    }

    /// Get the evaluation score for a position from Edax
    /// board_str should be in Edax FEN format or similar
    /// Returns the score as an i32
    pub fn evaluate(&self, board_str: &str) -> Result<i32, String> {
        // Start Edax in batch mode
        let mut child = Command::new(&self.edax_path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| format!("Failed to start Edax: {}", e))?;

        {
            let stdin = child
                .stdin
                .as_mut()
                .ok_or("Failed to open stdin".to_string())?;

            // Send board string and eval command
            writeln!(stdin, "{}", board_str).map_err(|e| e.to_string())?;
            writeln!(stdin, "eval").map_err(|e| e.to_string())?;
            writeln!(stdin, "quit").map_err(|e| e.to_string())?;
        }

        let output = child
            .wait_with_output()
            .map_err(|e| format!("Failed to wait for Edax: {}", e))?;

        let stdout = String::from_utf8_lossy(&output.stdout);

        // Parse score from output (simplified - Edax format may vary)
        for line in stdout.lines() {
            if line.contains("=") || line.contains("score") {
                if let Some(score_str) = line.split('=').nth(1) {
                    if let Ok(score) = score_str.trim().parse::<i32>() {
                        return Ok(score);
                    }
                }
            }
        }

        Err("Could not parse Edax output".to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_edax_interface_creation() {
        // This will only pass if EDAX_PATH is set
        if env::var("EDAX_PATH").is_ok() {
            let edax = EdaxInterface::new();
            assert!(edax.is_ok());
        }
    }
}
