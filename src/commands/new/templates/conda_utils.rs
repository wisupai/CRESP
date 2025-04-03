use crate::error::Result;
use crate::utils::cli_ui;
use std::path::Path;
use std::process::Command;

/// Ensure conda is available and return conda version if available
pub fn ensure_conda_available() -> Result<Option<String>> {
    let output = Command::new("conda").arg("--version").output();

    match output {
        Ok(output) if output.status.success() => {
            let version_str = String::from_utf8_lossy(&output.stdout);
            if let Some(version) = version_str.split_whitespace().nth(1) {
                cli_ui::display_success(&format!("Found conda version: {}", version));
                return Ok(Some(version.to_string()));
            }
            Ok(Some("unknown".to_string()))
        }
        _ => {
            cli_ui::display_error(
                "Conda is required for CRESP projects but not found on your system.",
            );
            cli_ui::display_info("Please install Conda (Miniconda or Anaconda) first:");
            cli_ui::display_info(
                "https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html",
            );
            Ok(None)
        }
    }
}

/// Check conda version and display update recommendation if needed
pub fn check_conda_version(version: &str) -> Result<()> {
    // Simple version comparison, only compare major version
    let parts: Vec<&str> = version.split('.').collect();
    if parts.len() >= 2 {
        if let Ok(major) = parts[0].parse::<u32>() {
            if major < 23 {
                cli_ui::display_warning(&format!(
                    "You are using an older version of conda ({}). Consider updating it for better performance and compatibility.",
                    version
                ));
                cli_ui::display_info(
                    "To update conda, run: conda update -n base -c defaults conda",
                );
            }
        }
    }
    Ok(())
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

/// Sanitize project name for use as conda environment name
/// Conda environment names cannot contain spaces or certain special characters
pub fn sanitize_for_conda_env(name: &str) -> String {
    // Replace spaces and invalid characters with underscores
    let sanitized = name
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '_' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect::<String>();

    // If name begins with a non-alphanumeric character, prefix with 'env_'
    if !sanitized.is_empty() && !sanitized.chars().next().unwrap().is_alphanumeric() {
        format!("env_{}", sanitized)
    } else {
        sanitized
    }
}

/// Validate if a project name is suitable for conda environment
/// Returns true if valid, along with a message if invalid
pub fn validate_conda_env_name(name: &str) -> (bool, String) {
    // Check if name contains spaces
    if name.contains(' ') {
        return (
            false,
            "Project name cannot contain spaces when using Conda (spaces will be replaced with underscores)".to_string()
        );
    }

    // Check for other invalid characters
    let has_invalid_chars = name
        .chars()
        .any(|c| !c.is_alphanumeric() && c != '_' && c != '-');
    if has_invalid_chars {
        return (
            false,
            "Project name contains invalid characters for Conda environment (only alphanumeric, underscore, and hyphen are allowed)".to_string()
        );
    }

    // Check if name starts with a valid character
    if !name.is_empty() && !name.chars().next().unwrap().is_alphanumeric() {
        return (
            false,
            "Project name must start with an alphanumeric character for Conda environment"
                .to_string(),
        );
    }

    (
        true,
        "Project name is valid for Conda environment".to_string(),
    )
}
