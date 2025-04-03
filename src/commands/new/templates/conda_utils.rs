use std::path::Path;
use std::process::Command;

use crate::error::Result;
use crate::utils::cli_ui;
use crate::utils::validation::exports::sanitize_for_conda_env;

/// Check if conda is available and get its version
pub fn ensure_conda_available() -> Result<Option<String>> {
    let output = Command::new("conda").arg("--version").output();

    match output {
        Ok(output) if output.status.success() => {
            let version_string = String::from_utf8_lossy(&output.stdout);
            let version = version_string
                .trim()
                .split_whitespace()
                .last()
                .unwrap_or("")
                .to_string();
            cli_ui::display_success(&format!("Found conda: {}", version));
            Ok(Some(version))
        }
        _ => {
            cli_ui::display_warning("Conda is not available on this system!");
            Ok(None)
        }
    }
}

/// Check conda version and warn if outdated
pub fn check_conda_version(version: &str) -> Result<()> {
    // Parse semver from version string (format: "conda X.Y.Z")
    let version_parts: Vec<&str> = version.split('.').collect();
    if version_parts.len() >= 3 {
        let major: u32 = match version_parts[0].parse() {
            Ok(m) => m,
            Err(_) => return Ok(()),
        };
        if major < 4 {
            cli_ui::display_warning(
                "Your conda version is outdated. Consider upgrading to conda 4.0+",
            );
        }
    }
    Ok(())
}

/// Install poetry in conda environment
pub fn install_poetry_in_conda_env(env_name: &str) -> Result<()> {
    cli_ui::display_info("Installing Poetry in Conda environment...");
    let conda_cmd = Command::new("conda")
        .args(&["run", "-n", env_name, "pip", "install", "poetry"])
        .status();

    match conda_cmd {
        Ok(status) if status.success() => {
            cli_ui::display_success("Poetry installed successfully in conda environment");
            Ok(())
        }
        _ => {
            cli_ui::display_warning("Failed to install Poetry in conda environment.");
            cli_ui::display_info(
                "You can install it manually later with: conda run -n <env> pip install poetry",
            );
            Ok(())
        }
    }
}

/// Install uv in conda environment
pub fn install_uv_in_conda_env(env_name: &str) -> Result<()> {
    cli_ui::display_info("Installing UV in Conda environment...");
    let conda_cmd = Command::new("conda")
        .args(&["run", "-n", env_name, "pip", "install", "uv"])
        .status();

    match conda_cmd {
        Ok(status) if status.success() => {
            cli_ui::display_success("UV installed successfully in conda environment");
            Ok(())
        }
        _ => {
            cli_ui::display_warning("Failed to install UV in conda environment.");
            cli_ui::display_info(
                "You can install it manually later with: conda run -n <env> pip install uv",
            );
            Ok(())
        }
    }
}

/// Create conda environment from environment file
pub fn create_conda_environment(project_dir: &Path, environment_file: &str) -> Result<bool> {
    cli_ui::display_info("Creating conda environment...");

    // Use Command::output instead of status to capture output
    let conda_cmd = Command::new("conda")
        .args(&["env", "create", "-f", environment_file])
        .current_dir(project_dir)
        .output();

    match conda_cmd {
        Ok(output) if output.status.success() => {
            // Get project name for conda environment
            let raw_project_name = project_dir
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("my-project");

            // Sanitize name for conda environment
            let conda_env_name = sanitize_for_conda_env(raw_project_name);

            cli_ui::display_success(&format!(
                "Conda environment '{}' created successfully!",
                conda_env_name
            ));
            cli_ui::display_info(&format!("To activate: conda activate {}", conda_env_name));
            Ok(true)
        }
        Ok(output) => {
            // Show detailed error information
            cli_ui::display_warning("Failed to create conda environment.");

            // Display command output which contains error information
            let stderr = String::from_utf8_lossy(&output.stderr);
            if !stderr.is_empty() {
                cli_ui::display_error("Error details:");
                for line in stderr.lines().take(5) {
                    // Limit to first 5 lines to avoid flooding
                    cli_ui::display_warning(line);
                }
            }

            cli_ui::display_info(&format!(
                "You can create it manually later with: conda env create -f {}",
                environment_file
            ));
            Ok(false)
        }
        Err(e) => {
            cli_ui::display_warning(&format!("Failed to execute conda command: {}", e));
            cli_ui::display_info(&format!(
                "You can create it manually later with: conda env create -f {}",
                environment_file
            ));
            Ok(false)
        }
    }
}

/// Generate base conda environment.yml content
pub fn generate_base_environment_yml(
    project_name: &str,
    channels: &[&str],
    dependencies: &[&str],
) -> String {
    // Sanitize project name for conda environment
    let conda_env_name = sanitize_for_conda_env(project_name);

    // If the name was sanitized, show a warning
    if conda_env_name != project_name {
        cli_ui::display_warning(&format!(
            "Project name '{}' contains characters not allowed in conda environment names.",
            project_name
        ));
        cli_ui::display_info(&format!(
            "Using '{}' as the conda environment name instead.",
            conda_env_name
        ));
    }

    let mut content = format!("name: {}\nchannels:\n", conda_env_name);

    // Add channels
    for channel in channels {
        content.push_str(&format!("  - {}\n", channel));
    }

    // Add dependencies
    content.push_str("\ndependencies:\n");
    for dependency in dependencies {
        content.push_str(&format!("  - {}\n", dependency));
    }

    content
}
