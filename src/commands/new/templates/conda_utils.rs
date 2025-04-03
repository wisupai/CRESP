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
    let conda_cmd = Command::new("conda")
        .args(&["env", "create", "-f", environment_file])
        .current_dir(project_dir)
        .status();

    match conda_cmd {
        Ok(status) if status.success() => {
            // Get project name for conda environment
            let project_name = project_dir
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("my-project");

            cli_ui::display_success(&format!(
                "Conda environment '{}' created successfully!",
                project_name
            ));
            cli_ui::display_info(&format!("To activate: conda activate {}", project_name));
            Ok(true)
        }
        _ => {
            cli_ui::display_warning("Failed to create conda environment.");
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
    let mut content = format!("name: {}\nchannels:\n", project_name);

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
