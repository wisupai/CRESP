use std::path::Path;
use std::process::Command;

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
    
    // Find conda executable path
    let conda_path = find_conda_executable()?;
    
    // 确保环境文件存在
    let env_file_path = project_dir.join(environment_file);
    if !env_file_path.exists() {
        cli_ui::display_error(&format!("Environment file not found: {:?}", env_file_path));
        return Ok(false);
    }
    
    // 显示环境文件内容日志，帮助调试
    cli_ui::display_info(&format!("Using environment file: {:?}", env_file_path));
    if let Ok(content) = std::fs::read_to_string(&env_file_path) {
        cli_ui::display_info(&format!("Environment file content:\n{}", content));
    }
    
    // 强制刷新文件系统缓存，确保文件已写入磁盘
    #[cfg(unix)]
    {
        use std::process::Command;
        let _ = Command::new("sync").status();
    }
    
    // 使用Command::output捕获输出而不是只获取状态
    cli_ui::display_info(&format!("Running: {} env create -f {:?}", conda_path, env_file_path));
    
    // 使用绝对路径并显式转换
    let env_file_str = env_file_path.to_str().ok_or_else(|| {
        crate::error::Error::Validation("Invalid environment file path".into())
    })?;
    
    let conda_cmd = Command::new(&conda_path)
        .args(&["env", "create", "-f", env_file_str])
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
            let stdout = String::from_utf8_lossy(&output.stdout);
            let stderr = String::from_utf8_lossy(&output.stderr);
            
            cli_ui::display_info("Command output:");
            if !stdout.is_empty() {
                cli_ui::display_info(&format!("stdout: {}", stdout));
            }
            
            if !stderr.is_empty() {
                cli_ui::display_error("Error details:");
                for line in stderr.lines().take(10) {
                    // 显示更多行以便更好地诊断
                    cli_ui::display_warning(line);
                }
            }
            
            // 提供备选方案
            cli_ui::display_info("You can create the environment manually with this command:");
            cli_ui::display_info(&format!("cd {:?} && conda env create -f {}", 
                                          project_dir, environment_file));
            Ok(false)
        }
        Err(e) => {
            cli_ui::display_warning(&format!("Failed to execute conda command: {}", e));
            cli_ui::display_error(&format!("Error type: {:?}", e.kind()));
            
            // 打印更多调试信息
            cli_ui::display_info(&format!("Conda path: {}", conda_path));
            cli_ui::display_info(&format!("Working directory: {:?}", project_dir));
            cli_ui::display_info(&format!("Environment file: {}", environment_file));
            
            // 提供备选方案
            cli_ui::display_info("You can create the environment manually with this command:");
            cli_ui::display_info(&format!("cd {:?} && conda env create -f {}", 
                                          project_dir, environment_file));
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
    cli_ui::display_info("You can install conda from: https://docs.conda.io/en/latest/miniconda.html");
    
    // Return error with informative message
    Err(crate::error::Error::Command("Conda executable not found".into()))
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
