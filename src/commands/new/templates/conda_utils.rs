use std::path::Path;
use std::process::{Command, Stdio};

use crate::error::Result;
use crate::utils::cli_ui;
use crate::utils::validation::exports::sanitize_for_conda_env;

/// Check if conda is available and get its version
pub fn ensure_conda_available() -> Result<Option<String>> {
    // Try to find conda executable
    let conda_path = match find_conda_executable() {
        Ok(path) => path,
        Err(_) => {
            cli_ui::display_warning("Conda is not available on this system!");
            return Ok(None);
        }
    };

    // Get conda version
    let output = Command::new(&conda_path).arg("--version").output();

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

            // Check if conda version is suitable
            check_conda_version(&version)?;

            Ok(Some(version))
        }
        _ => {
            cli_ui::display_warning("Conda is not working properly on this system!");
            cli_ui::display_info(
                "Please ensure that conda is correctly installed and in your PATH.",
            );
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

/// Check if conda supports faster solver
fn check_faster_solver_support(conda_path: &str) -> bool {
    // Check if conda version supports the libmamba solver
    let output = Command::new(conda_path)
        .args(&["--version"])
        .output();
    
    if let Ok(output) = output {
        let version_str = String::from_utf8_lossy(&output.stdout);
        // Parse version from string like "conda 23.1.0"
        if let Some(version) = version_str.split_whitespace().nth(1) {
            // Extract major version
            if let Some(major_str) = version.split('.').next() {
                if let Ok(major) = major_str.parse::<u32>() {
                    // libmamba solver is supported in conda 22.11.0+
                    return major >= 23 || (major == 22 && version.contains("22.11.") || version.contains("22.12."));
                }
            }
        }
    }
    
    // Also check if mamba is installed separately
    let mamba_check = Command::new("mamba")
        .args(&["--version"])
        .output();
    
    mamba_check.is_ok() && mamba_check.unwrap().status.success()
}

/// Install poetry in conda environment
pub fn install_poetry_in_conda_env(env_name: &str) -> Result<()> {
    cli_ui::display_info("Installing Poetry in Conda environment...");

    // Find conda executable path
    let conda_path = match find_conda_executable() {
        Ok(path) => path,
        Err(e) => {
            cli_ui::display_warning(&format!("Failed to find conda executable: {}", e));
            cli_ui::display_info(
                "You can install Poetry manually later with: conda run -n <env> pip install poetry",
            );
            return Ok(());
        }
    };

    let conda_cmd = Command::new(&conda_path)
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

    // Find conda executable path
    let conda_path = match find_conda_executable() {
        Ok(path) => path,
        Err(e) => {
            cli_ui::display_warning(&format!("Failed to find conda executable: {}", e));
            cli_ui::display_info(
                "You can install UV manually later with: conda run -n <env> pip install uv",
            );
            return Ok(());
        }
    };

    let conda_cmd = Command::new(&conda_path)
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

    // Add delay after file write operations
    std::thread::sleep(std::time::Duration::from_secs(1));

    // Find conda executable path - need to use full path instead of simple "conda" command
    let conda_path = match find_conda_executable() {
        Ok(path) => path,
        Err(e) => {
            cli_ui::display_error(&format!("Failed to find conda executable: {}", e));
            cli_ui::display_info(
                "Please make sure conda is correctly installed and available in your PATH.",
            );
            return Ok(false);
        }
    };

    // Check if faster solver is available
    let use_faster_solver = check_faster_solver_support(&conda_path);
    if use_faster_solver {
        cli_ui::display_info("Using faster dependency solver (libmamba) for conda");
    } else {
        cli_ui::display_info("Using standard conda solver (this might take some time)");
        cli_ui::display_info("For faster environment creation, consider installing conda 22.11.0+ or mamba");
    }

    // Ensure environment file exists
    let env_file_path = project_dir.join(environment_file);
    if !env_file_path.exists() {
        cli_ui::display_error(&format!("Environment file not found: {:?}", env_file_path));

        // Check if directory exists
        if !project_dir.exists() {
            cli_ui::display_error(&format!(
                "Project directory does not exist: {:?}",
                project_dir
            ));
        } else {
            // List directory contents for debugging
            cli_ui::display_info(&format!("Project directory contents at {:?}:", project_dir));
            if let Ok(entries) = std::fs::read_dir(project_dir) {
                for entry in entries {
                    if let Ok(entry) = entry {
                        cli_ui::display_info(&format!("  - {:?}", entry.path()));
                    }
                }
            }
        }

        return Ok(false);
    }

    // Log environment file contents for debugging
    cli_ui::display_info(&format!("Using environment file: {:?}", env_file_path));
    if let Ok(content) = std::fs::read_to_string(&env_file_path) {
        cli_ui::display_info(&format!("Environment file content:\n{}", content));
    }

    // Force flush filesystem cache to ensure file is written to disk
    #[cfg(unix)]
    {
        use std::process::Command;
        let _ = Command::new("sync").status();
    }

    // Wait again to ensure filesystem is fully synced
    std::thread::sleep(std::time::Duration::from_secs(1));

    // Use absolute path
    let abs_project_dir = if project_dir.is_absolute() {
        project_dir.to_path_buf()
    } else {
        std::env::current_dir()?.join(project_dir)
    };

    // Absolute environment file path
    let abs_env_file_path = if env_file_path.is_absolute() {
        env_file_path.clone()
    } else {
        abs_project_dir.join(environment_file)
    };

    // Ensure working directory is correct
    let original_dir = std::env::current_dir()?;
    cli_ui::display_info(&format!("Current directory before: {:?}", original_dir));

    if std::env::current_dir()? != abs_project_dir {
        cli_ui::display_info(&format!(
            "Changing working directory to: {:?}",
            abs_project_dir
        ));
        std::env::set_current_dir(&abs_project_dir)?;
    }

    // Execute conda command with full path and inherited stdio to show real-time progress
    let command_str = if use_faster_solver {
        format!("{} env create --solver=libmamba -f {}", conda_path, environment_file)
    } else {
        format!("{} env create -f {}", conda_path, environment_file)
    };
    
    cli_ui::display_info(&format!("Running: {}", command_str));

    // Use relative path of environment file (relative to working directory)
    let relative_env_file = environment_file;

    // Execute conda command with full path and inherit stdio to show real-time output
    cli_ui::display_info(&format!("Using conda executable: {}", conda_path));
    cli_ui::display_info("Starting conda environment creation (showing real-time progress):");
    
    // Build the command with appropriate solver options
    let mut cmd = Command::new(&conda_path);
    cmd.arg("env")
       .arg("create");
    
    // Add faster solver if available
    if use_faster_solver {
        cmd.arg("--solver=libmamba");
    }
    
    // Add other options to potentially speed up environment creation
    cmd.arg("--yes")  // Auto-confirm prompts
       .arg("-f")
       .arg(relative_env_file)
       .current_dir(&abs_project_dir)
       .stdin(Stdio::inherit())
       .stdout(Stdio::inherit())
       .stderr(Stdio::inherit());
    
    // Use spawn and wait with stdio inheritance to display real-time conda output
    let conda_status = cmd.spawn().and_then(|mut child| child.wait());

    // Display current working directory
    cli_ui::display_info(&format!(
        "Current directory during command: {:?}",
        std::env::current_dir()?
    ));

    // Restore original working directory
    std::env::set_current_dir(original_dir)?;
    cli_ui::display_info(&format!(
        "Current directory after: {:?}",
        std::env::current_dir()?
    ));

    match conda_status {
        Ok(status) if status.success() => {
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
        Ok(status) => {
            // Show error information
            cli_ui::display_warning(&format!(
                "Failed to create conda environment. Exit code: {:?}",
                status.code()
            ));

            // Provide alternative solution
            cli_ui::display_info("You can create the environment manually with this command:");
            if use_faster_solver {
                cli_ui::display_info(&format!(
                    "cd {:?} && {} env create --solver=libmamba -f {}",
                    abs_project_dir, conda_path, environment_file
                ));
            } else {
                cli_ui::display_info(&format!(
                    "cd {:?} && {} env create -f {}",
                    abs_project_dir, conda_path, environment_file
                ));
                cli_ui::display_info("For faster environment solving, consider:");
                cli_ui::display_info("1. Installing mamba: conda install -n base -c conda-forge mamba");
                cli_ui::display_info("2. Upgrading to conda 22.11.0+ for libmamba solver");
            }
            Ok(false)
        }
        Err(e) => {
            cli_ui::display_warning(&format!("Failed to execute conda command: {}", e));
            cli_ui::display_error(&format!("Error type: {:?}", e.kind()));

            // Print more debug information
            cli_ui::display_info(&format!("Conda path: {}", conda_path));
            cli_ui::display_info(&format!("Working directory: {:?}", abs_project_dir));
            cli_ui::display_info(&format!("Environment file: {}", environment_file));
            cli_ui::display_info(&format!(
                "Absolute environment file path: {:?}",
                abs_env_file_path
            ));

            // Check file permissions
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                if let Ok(metadata) = std::fs::metadata(&abs_env_file_path) {
                    let permissions = metadata.permissions();
                    cli_ui::display_info(&format!("File permissions: {:o}", permissions.mode()));
                }
            }

            // Check conda executable
            if let Ok(output) = Command::new(&conda_path).arg("--version").output() {
                if output.status.success() {
                    let version = String::from_utf8_lossy(&output.stdout);
                    cli_ui::display_info(&format!("Conda version check: {}", version.trim()));
                } else {
                    cli_ui::display_warning(
                        "Conda executable exists but could not run version check",
                    );
                }
            } else {
                cli_ui::display_error(
                    "Could not execute conda version check. The executable may not be valid.",
                );
            }

            // Provide alternative solution with tips for faster solving
            cli_ui::display_info("You can create the environment manually with this command:");
            if use_faster_solver {
                cli_ui::display_info(&format!(
                    "cd {:?} && {} env create --solver=libmamba -f {}",
                    abs_project_dir, conda_path, environment_file
                ));
            } else {
                cli_ui::display_info(&format!(
                    "cd {:?} && {} env create -f {}",
                    abs_project_dir, conda_path, environment_file
                ));
                cli_ui::display_info("For faster environment solving, consider:");
                cli_ui::display_info("1. Installing mamba: conda install -n base -c conda-forge mamba");
                cli_ui::display_info("2. Upgrading to conda 22.11.0+ for libmamba solver");
            }
            Ok(false)
        }
    }
}

/// Find conda executable path
pub fn find_conda_executable() -> Result<String> {
    // Try to find conda executable
    let possible_conda_commands = &["conda", "micromamba", "mamba"];

    // Try standard command first (which uses PATH environment variable)
    for cmd in possible_conda_commands {
        if let Ok(output) = Command::new(cmd).arg("--version").output() {
            if output.status.success() {
                cli_ui::display_info(&format!("Found conda command: {}", cmd));
                return Ok(cmd.to_string());
            }
        }
    }

    // If not found in PATH, look in common installation directories
    let common_paths = if cfg!(target_os = "windows") {
        vec![
            r"C:\ProgramData\Anaconda3\Scripts\conda.exe",
            r"C:\ProgramData\miniconda3\Scripts\conda.exe",
            r"C:\ProgramData\Miniconda3\Scripts\conda.exe",
            r"C:\ProgramData\anaconda3\Scripts\conda.exe",
            r"C:\ProgramData\miniforge3\Scripts\conda.exe",
            r"C:\ProgramData\Miniforge3\Scripts\conda.exe",
            r"C:\tools\miniconda3\Scripts\conda.exe",
            r"C:\tools\Miniconda3\Scripts\conda.exe",
            r"C:\tools\anaconda3\Scripts\conda.exe",
            r"C:\tools\Anaconda3\Scripts\conda.exe",
        ]
    } else if cfg!(target_os = "macos") {
        vec![
            "/opt/anaconda3/bin/conda",
            "/opt/miniconda3/bin/conda",
            "/opt/miniforge3/bin/conda",
            "/usr/local/anaconda3/bin/conda",
            "/usr/local/miniconda3/bin/conda",
            "/usr/local/miniforge3/bin/conda",
            "/usr/local/opt/conda/bin/conda",
            "/usr/local/Caskroom/miniconda/base/bin/conda",
            "/usr/local/Caskroom/miniforge/base/bin/conda",
            "/Applications/anaconda3/bin/conda",
        ]
    } else {
        vec![
            "/opt/anaconda3/bin/conda",
            "/opt/miniconda3/bin/conda",
            "/opt/miniforge3/bin/conda",
            "/usr/local/anaconda3/bin/conda",
            "/usr/local/miniconda3/bin/conda",
            "/usr/local/miniforge3/bin/conda",
            "/usr/bin/conda",
            "/usr/local/bin/conda",
        ]
    };

    // Check the common paths first
    for path in &common_paths {
        if std::path::Path::new(path).exists() {
            // Verify the path is executable by running version command
            if let Ok(output) = Command::new(path).arg("--version").output() {
                if output.status.success() {
                    cli_ui::display_info(&format!("Found conda at: {}", path));
                    return Ok(path.to_string());
                }
            }
        }
    }

    // Add user home directory paths
    let mut home_paths = Vec::new();
    if let Ok(home) = std::env::var("HOME").or_else(|_| std::env::var("USERPROFILE")) {
        if cfg!(target_os = "windows") {
            home_paths.push(format!("{}\\Anaconda3\\Scripts\\conda.exe", home));
            home_paths.push(format!("{}\\miniconda3\\Scripts\\conda.exe", home));
            home_paths.push(format!("{}\\Miniconda3\\Scripts\\conda.exe", home));
            home_paths.push(format!("{}\\miniforge3\\Scripts\\conda.exe", home));
            home_paths.push(format!("{}\\Miniforge3\\Scripts\\conda.exe", home));
        } else {
            home_paths.push(format!("{}/anaconda3/bin/conda", home));
            home_paths.push(format!("{}/miniconda3/bin/conda", home));
            home_paths.push(format!("{}/miniforge3/bin/conda", home));
            home_paths.push(format!("{}/.conda/bin/conda", home));
        }
    }

    // Check home directory paths
    for path_string in &home_paths {
        let path = path_string.as_str();
        if std::path::Path::new(path).exists() {
            // Verify the path is executable by running version command
            if let Ok(output) = Command::new(path).arg("--version").output() {
                if output.status.success() {
                    cli_ui::display_info(&format!("Found conda at: {}", path));
                    return Ok(path.to_string());
                }
            }
        }
    }

    // If still not found, throw an error with helpful message
    cli_ui::display_warning("Could not find conda executable in common locations.");
    cli_ui::display_info("Please make sure conda is installed and in your PATH.");
    cli_ui::display_info(
        "You can install conda from: https://docs.conda.io/en/latest/miniconda.html",
    );

    // Return error with informative message
    Err(crate::error::Error::Command(
        "Conda executable not found".into(),
    ))
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
