use std::path::Path;
use std::process::{Command, Stdio};

use crate::error::Result;
use crate::utils::cli_ui;
use crate::utils::validation::exports::sanitize_for_conda_env;
use log::{debug, info, trace};

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
            info!("Please ensure that conda is correctly installed and in your PATH.");
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
    let output = Command::new(conda_path).args(&["--version"]).output();

    if let Ok(output) = output {
        let version_str = String::from_utf8_lossy(&output.stdout);
        // Parse version from string like "conda 23.1.0"
        if let Some(version) = version_str.split_whitespace().nth(1) {
            // Extract major version
            if let Some(major_str) = version.split('.').next() {
                if let Ok(major) = major_str.parse::<u32>() {
                    // libmamba solver is supported in conda 22.11.0+
                    return major >= 23
                        || (major == 22 && version.contains("22.11.")
                            || version.contains("22.12."));
                }
            }
        }
    }

    // Also check if mamba is installed separately
    let mamba_check = Command::new("mamba").args(&["--version"]).output();

    mamba_check.is_ok() && mamba_check.unwrap().status.success()
}

/// Install poetry in conda environment
pub fn install_poetry_in_conda_env(env_name: &str) -> Result<()> {
    info!("Installing Poetry in Conda environment...");

    // Find conda executable path
    let conda_path = match find_conda_executable() {
        Ok(path) => path,
        Err(e) => {
            cli_ui::display_warning(&format!("Failed to find conda executable: {}", e));
            info!(
                "You can install Poetry manually later with: conda run -n <env> pip install poetry"
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
            info!("You can install it manually later with: conda run -n <env> pip install poetry");
            Ok(())
        }
    }
}

/// Install uv in conda environment
pub fn install_uv_in_conda_env(env_name: &str) -> Result<()> {
    info!("Installing UV in Conda environment...");

    // Find conda executable path
    let conda_path = match find_conda_executable() {
        Ok(path) => path,
        Err(e) => {
            cli_ui::display_warning(&format!("Failed to find conda executable: {}", e));
            info!("You can install UV manually later with: conda run -n <env> pip install uv");
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
            info!("You can install it manually later with: conda run -n <env> pip install uv");
            Ok(())
        }
    }
}

/// Create conda environment from environment file
pub fn create_conda_environment(project_dir: &Path, environment_file: &str) -> Result<bool> {
    info!("Creating conda environment...");

    // Add delay after file write operations (reduced from 2s to 1s)
    std::thread::sleep(std::time::Duration::from_secs(1));

    // Find conda executable path - need to use full path instead of simple "conda" command
    let conda_path = match find_conda_executable() {
        Ok(path) => path,
        Err(e) => {
            cli_ui::display_error(&format!("Failed to find conda executable: {}", e));
            cli_ui::display_message(
                "Please make sure conda is correctly installed and available in your PATH.",
            );
            return Ok(false);
        }
    };

    // Check if faster solver is available
    let use_faster_solver = check_faster_solver_support(&conda_path);
    if use_faster_solver {
        debug!("Using faster dependency solver (libmamba) for conda");
    } else {
        debug!("Using standard conda solver (this might take some time)");
        debug!("For faster environment creation, consider installing conda 22.11.0+ or mamba");
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
            // List directory contents for debugging - moved to trace level
            trace!("Project directory contents at {:?}:", project_dir);
            if let Ok(entries) = std::fs::read_dir(project_dir) {
                for entry in entries {
                    if let Ok(entry) = entry {
                        trace!("  - {:?}", entry.path());
                    }
                }
            }
        }

        return Ok(false);
    }

    // Log environment file contents for debugging - moved to trace level
    debug!("Using environment file: {:?}", env_file_path);
    if let Ok(content) = std::fs::read_to_string(&env_file_path) {
        trace!("Environment file content:\n{}", content);
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
    trace!("Current directory before: {:?}", original_dir);

    if std::env::current_dir()? != abs_project_dir {
        debug!("Changing working directory to: {:?}", abs_project_dir);
        std::env::set_current_dir(&abs_project_dir)?;
    }

    // Read the current environment name
    let mut env_name = String::new();
    if let Ok(content) = std::fs::read_to_string(&abs_env_file_path) {
        if let Some(name_line) = content.lines().find(|line| line.starts_with("name:")) {
            if let Some(name) = name_line.split(':').nth(1) {
                env_name = name.trim().to_string();
            }
        }
    }

    // 尝试创建环境，支持多次尝试
    let mut attempt_count = 0;
    let max_attempts = 3;
    
    while attempt_count < max_attempts {
        attempt_count += 1;
        
        // Execute conda command with full path and inherited stdio to show real-time progress
        let command_str = if use_faster_solver {
            format!(
                "{} env create --solver=libmamba -f {}",
                conda_path, environment_file
            )
        } else {
            format!("{} env create -f {}", conda_path, environment_file)
        };

        debug!("Running: {}", command_str);

        // Use relative path of environment file (relative to working directory)
        let relative_env_file = environment_file;

        // Execute conda command with full path and inherit stdio to show real-time output
        debug!("Using conda executable: {}", conda_path);
        cli_ui::display_progress("Creating environment", "This may take a while...");

        // Build the command with appropriate solver options
        let mut cmd = Command::new(&conda_path);
        cmd.arg("env").arg("create");

        // Add faster solver if available
        if use_faster_solver {
            cmd.arg("--solver=libmamba");
        }

        // Add options for environment creation
        cmd.arg("-f")
            .arg(relative_env_file)
            .current_dir(&abs_project_dir)
            .stdin(Stdio::inherit())
            .stdout(Stdio::inherit())
            .stderr(Stdio::inherit());

        // Use spawn and wait with stdio inheritance to display real-time conda output
        let conda_status = cmd.spawn().and_then(|mut child| child.wait());

        // Display current working directory - moved to trace level
        trace!(
            "Current directory during command: {:?}",
            std::env::current_dir()?
        );

        match conda_status {
            Ok(status) if status.success() => {
                // Restore original working directory
                std::env::set_current_dir(original_dir)?;
                
                cli_ui::display_success(&format!(
                    "Conda environment '{}' created successfully!",
                    env_name
                ));
                cli_ui::display_message(&format!("To activate: conda activate {}", env_name));
                return Ok(true);
            }
            Ok(status) => {
                // check if the environment already exists
                cli_ui::display_warning(&format!(
                    "Failed to create conda environment. Exit code: {:?}",
                    status.code()
                ));
                
                // ask the user if they want to try a different name
                if attempt_count < max_attempts {
                    let retry = cli_ui::prompt_confirm("Environment already exists. Would you like to try a different name?", true)?;
                    
                    if retry {
                        // let the user input a new environment name
                        let new_name: String = cli_ui::prompt_input(&format!("Enter new environment name (current: {}):", env_name), None)?;
                        
                        if !new_name.is_empty() {
                            // update the name in the environment file
                            if let Ok(content) = std::fs::read_to_string(&abs_env_file_path) {
                                let new_content = content.replacen(&format!("name: {}", env_name), &format!("name: {}", new_name), 1);
                                std::fs::write(&abs_env_file_path, new_content)?;
                                env_name = new_name;
                                cli_ui::display_info(&format!("Retrying with new environment name: {}", env_name));
                                continue;  // 重试创建
                            }
                        }
                    }
                }
                
                // recover the original working directory
                std::env::set_current_dir(original_dir)?;
                
                // provide the command to create the environment manually
                cli_ui::display_message("You can create the environment manually with this command:");
                if use_faster_solver {
                    cli_ui::display_message(&format!(
                        "cd {:?} && {} env create --solver=libmamba -f {}",
                        abs_project_dir, conda_path, environment_file
                    ));
                } else {
                    cli_ui::display_message(&format!(
                        "cd {:?} && {} env create -f {}",
                        abs_project_dir, conda_path, environment_file
                    ));
                }
                return Ok(false);
            }
            Err(e) => {
                // recover the original working directory
                std::env::set_current_dir(original_dir)?;
                
                cli_ui::display_warning(&format!("Failed to execute conda command: {}", e));
                debug!("Error type: {:?}", e.kind());
                
                // provide the command to create the environment manually
                cli_ui::display_message("You can create the environment manually with this command:");
                if use_faster_solver {
                    cli_ui::display_message(&format!(
                        "cd {:?} && {} env create --solver=libmamba -f {}",
                        abs_project_dir, conda_path, environment_file
                    ));
                } else {
                    cli_ui::display_message(&format!(
                        "cd {:?} && {} env create -f {}",
                        abs_project_dir, conda_path, environment_file
                    ));
                }
                return Ok(false);
            }
        }
    }
    
    // Exceeded max attempts
    cli_ui::display_warning(&format!("Failed to create conda environment after {} attempts", max_attempts));
    std::env::set_current_dir(original_dir)?;
    Ok(false)
}

/// Find conda executable path
pub fn find_conda_executable() -> Result<String> {
    // Try to find conda executable
    let possible_conda_commands = &["conda", "micromamba", "mamba"];

    // Try standard command first (which uses PATH environment variable)
    for cmd in possible_conda_commands {
        if let Ok(output) = Command::new(cmd).arg("--version").output() {
            if output.status.success() {
                debug!("Found conda command: {}", cmd);
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
            "/opt/mambaforge3/bin/conda",
            "/usr/local/anaconda3/bin/conda",
            "/usr/local/miniconda3/bin/conda",
            "/usr/local/miniforge3/bin/conda",
            "/usr/local/mambaforge/bin/conda",
            "/usr/local/opt/conda/bin/conda",
            "/usr/local/Caskroom/miniconda/base/bin/conda",
            "/usr/local/Caskroom/miniforge/base/bin/conda",
            "/usr/local/Caskroom/mambaforge/base/bin/conda",
            "/Applications/anaconda3/bin/conda",
            "/Applications/miniconda3/bin/conda",
            "/Applications/miniforge3/bin/conda",
            "/Applications/mambaforge/bin/conda",
        ]
    } else {
        vec![
            "/opt/anaconda3/bin/conda",
            "/opt/miniconda3/bin/conda",
            "/opt/miniforge3/bin/conda",
            "/opt/mambaforge/bin/conda",
            "/usr/local/anaconda3/bin/conda",
            "/usr/local/miniconda3/bin/conda",
            "/usr/local/miniforge3/bin/conda",
            "/usr/local/mambaforge/bin/conda",
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
                    debug!("Found conda at: {}", path);
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
            home_paths.push(format!("{}\\mambaforge\\Scripts\\conda.exe", home));
            home_paths.push(format!("{}\\Mambaforge\\Scripts\\conda.exe", home));
        } else {
            home_paths.push(format!("{}/anaconda3/bin/conda", home));
            home_paths.push(format!("{}/miniconda3/bin/conda", home));
            home_paths.push(format!("{}/miniforge3/bin/conda", home));
            home_paths.push(format!("{}/mambaforge/bin/conda", home));
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
                    debug!("Found conda at: {}", path);
                    return Ok(path.to_string());
                }
            }
        }
    }

    // If still not found, display comprehensive installation options
    cli_ui::display_warning("Could not find conda executable in common locations.");
    info!("Please install one of the following conda distributions:");

    info!("1. Miniconda (RECOMMENDED): Lightweight installation");
    debug!("   • Minimal size (~60-100MB) and fast to install");
    debug!("   • Perfect for most scientific computing needs");
    debug!("   • Install from: https://docs.conda.io/en/latest/miniconda.html");

    debug!("2. Anaconda: Full installation with 250+ packages pre-installed");
    debug!("   • Large size (~3GB) and slower to install");
    debug!("   • Install from: https://www.anaconda.com/download");

    debug!("3. Miniforge or Mambaforge: Like Miniconda but with conda-forge channel");
    debug!("   • Install from: https://github.com/conda-forge/miniforge");

    info!("After installation, ensure conda is in your PATH and restart your terminal.");

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
        info!(
            "Using '{}' as the conda environment name instead.",
            conda_env_name
        );
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

/// Language type enumeration
#[derive(Debug, Clone, PartialEq)]
pub enum Language {
    Python,
    R,
    Matlab,
    Other,
}

impl From<&str> for Language {
    fn from(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "python" => Language::Python,
            "r" => Language::R,
            "matlab" => Language::Matlab,
            _ => Language::Other,
        }
    }
}

/// Create conda environment for specific language
pub fn create_language_conda_env(
    project_dir: &Path,
    project_name: &str, 
    language: &Language,
    python_version: Option<&str>,
    r_version: Option<&str>,
    with_cuda: bool,
    channels: Option<&[String]>,
) -> Result<bool> {
    // Generate environment file
    let env_file_content = generate_language_environment_yml(
        project_name, 
        language, 
        python_version,
        r_version,
        with_cuda,
        channels,
    );

    // Write environment file
    let env_file_path = project_dir.join("environment.yml");
    std::fs::write(&env_file_path, env_file_content)?;
    debug!("Created conda environment file at: {:?}", env_file_path);

    // Create environment
    cli_ui::display_progress("Creating conda environment...", "This may take a while...");
    let result = create_conda_environment(project_dir, "environment.yml")?;

    // If creation successful, configure additional package managers
    if result {
        let conda_env_name = sanitize_for_conda_env(project_name);
        
        match language {
            Language::Python => {
                // May need to install additional package managers for Python
                cli_ui::display_info("Setting up Python package managers...");
                setup_python_package_managers(&conda_env_name)?;
            },
            Language::R => {
                // Install renv for R
                cli_ui::display_info("Setting up R environment...");
                setup_r_environment(&conda_env_name)?;
            },
            _ => {
                // Specific settings for other languages
                cli_ui::display_info("Basic conda environment setup complete.");
            }
        }
        
        // Show environment activation prompt
        cli_ui::display_success(&format!("Environment '{}' created successfully!", conda_env_name));
        cli_ui::display_message(&format!("To activate: conda activate {}", conda_env_name));
    }
    
    Ok(result)
}

/// Setup package managers for Python
fn setup_python_package_managers(conda_env_name: &str) -> Result<()> {
    // 注意：这个函数只在通过conda_utils创建环境时被调用
    // 如果是通过python.rs流程创建的项目，应该尊重用户已经做出的选择
    // 因此我们需要跳过这些提示，避免重复询问

    // Skip asking again since user would have already chosen via the main setup flow
    // 将相关提示作为debug信息记录，而不是向用户展示
    debug!("Package managers will be installed later if needed.");
    debug!("If you want to install UV or Poetry manually, run:");
    debug!("  conda activate {} && pip install uv", conda_env_name);
    debug!("  conda activate {} && pip install poetry", conda_env_name);
    
    Ok(())
}

/// Setup R environment
fn setup_r_environment(conda_env_name: &str) -> Result<()> {
    // Ensure renv is installed
    cli_ui::display_progress("Setting up R environment...", "Installing renv...");
    
    let conda_path = find_conda_executable()?;
    let install_cmd = Command::new(&conda_path)
        .args(&["run", "-n", conda_env_name, "Rscript", "-e", 
                "if(!require('renv')) install.packages('renv', repos='https://cloud.r-project.org')"])
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .status();
    
    match install_cmd {
        Ok(status) if status.success() => {
            cli_ui::display_success("R environment setup completed successfully");
        }
        _ => {
            cli_ui::display_warning("Could not complete R environment setup");
            cli_ui::display_info("You may need to install renv manually: install.packages('renv')");
        }
    }
    
    Ok(())
}

/// Generate language-specific environment YAML
pub fn generate_language_environment_yml(
    project_name: &str, 
    language: &Language,
    python_version: Option<&str>,
    r_version: Option<&str>,
    with_cuda: bool,
    channels: Option<&[String]>,
) -> String {
    // Default channels for each language
    let default_channels = match language {
        Language::Python => vec!["conda-forge", "defaults"],
        Language::R => vec!["r", "conda-forge", "defaults"],
        _ => vec!["conda-forge", "defaults"],
    };
    
    // Use custom channels if provided, otherwise use defaults
    let channels: Vec<&str> = match channels {
        Some(channels) if !channels.is_empty() => {
            channels.iter().map(|s| s.as_str()).collect()
        },
        _ => default_channels.iter().copied().collect(),
    };
    
    // Base dependencies
    let mut dependencies = Vec::new();
    
    // Add language-specific dependencies
    match language {
        Language::Python => {
            // Python dependencies
            let python_ver = python_version.unwrap_or("3.12");
            dependencies.push(format!("python={}", python_ver));
            dependencies.push("pip".to_string());
            
            if with_cuda {
                dependencies.push("cudatoolkit=11.8".to_string());
                dependencies.push("cudnn=8.9".to_string());
                dependencies.push("numpy".to_string());
                dependencies.push("scipy".to_string());
                dependencies.push("matplotlib".to_string());
                dependencies.push("pandas".to_string());
            }
        },
        Language::R => {
            // R dependencies
            let r_ver = r_version.unwrap_or("4.3");
            dependencies.push(format!("r-base={}", r_ver));
            dependencies.push("r-renv".to_string());
            dependencies.push("r-essentials".to_string());
            dependencies.push("r-devtools".to_string());
            dependencies.push("r-testthat".to_string());
        },
        Language::Matlab => {
            // Matlab dependencies
            dependencies.push("python=3.11".to_string()); // Matlab typically integrates with Python
            dependencies.push("pip".to_string());
        },
        Language::Other => {
            // Default dependencies
            dependencies.push("python=3.11".to_string());
            dependencies.push("pip".to_string());
        }
    }
    
    // Generate environment file content
    let conda_env_name = sanitize_for_conda_env(project_name);
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

/// Generate conda environment file for development environment
pub fn generate_dev_environment_yml(
    project_name: &str, 
    language: &Language,
    python_version: Option<&str>,
    r_version: Option<&str>,
    with_cuda: bool,
    channels: Option<&[String]>,
) -> String {
    // First get the base environment
    let mut content = generate_language_environment_yml(
        project_name, 
        language,
        python_version,
        r_version,
        with_cuda,
        channels,
    );
    
    // Modify environment name to development version
    content = content.replace(&format!("name: {}", sanitize_for_conda_env(project_name)), 
                             &format!("name: {}-dev", sanitize_for_conda_env(project_name)));
    
    // Add development dependencies
    match language {
        Language::Python => {
            // Python development dependencies
            content.push_str("  - pytest\n");
            content.push_str("  - black\n");
            content.push_str("  - isort\n");
            content.push_str("  - flake8\n");
        },
        Language::R => {
            // R development dependencies
            content.push_str("  - r-testthat\n");
            content.push_str("  - r-roxygen2\n");
            content.push_str("  - r-rcpp\n");
        },
        _ => {
            // Generic development dependencies
            content.push_str("  - pytest\n");
        }
    }
    
    content
}

/// Verify language installation in conda environment
pub fn verify_language_installation(language: &Language) -> Result<bool> {
    match language {
        Language::Python => {
            // Verify Python installation
            let output = Command::new("python").arg("--version").output();
            if let Ok(output) = output {
                if output.status.success() {
                    let version = String::from_utf8_lossy(&output.stdout);
                    cli_ui::display_success(&format!("Python is installed: {}", version.trim()));
                    return Ok(true);
                }
            }
            cli_ui::display_warning("Could not verify Python installation.");
        },
        Language::R => {
            // Verify R installation
            let output = Command::new("Rscript").arg("--version").output();
            if let Ok(output) = output {
                let version_output = String::from_utf8_lossy(&output.stderr); // R prints version to stderr
                cli_ui::display_success(&format!(
                    "R is successfully installed: {}",
                    version_output.trim()
                ));
                return Ok(true);
            }
            cli_ui::display_warning("Could not verify R installation.");
        },
        _ => {
            cli_ui::display_warning("Verification not implemented for this language type.");
        }
    }
    
    Ok(false)
}
