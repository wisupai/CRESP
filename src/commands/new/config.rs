use crate::error::Result;
use crate::utils::cli_ui;
use std::io::{self};
use std::process::Command;

#[derive(Debug, Clone)]
pub struct UserConfig {
    pub package_managers: Vec<PackageManager>,
    pub use_cuda: bool,
    pub cuda_version: Option<String>,
    pub cudnn_version: Option<String>,
    pub python_version: String,
    pub use_conda: bool,
    pub virtual_env_type: VirtualEnvType,
    pub pip_index_url: Option<String>,
    pub pip_trusted_hosts: Option<Vec<String>>,
    pub uv_installed: bool,
    pub poetry_installed: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub enum VirtualEnvType {
    Conda,
    None,
}

#[derive(Debug, Clone)]
pub enum CondaDistribution {
    Miniconda,
    Anaconda,
    CondaForge,
}

#[derive(Clone, Debug)]
pub enum PackageManager {
    Conda {
        channels: Vec<String>,
        environment_file: String,
        dev_environment_file: String,
    },
    Poetry {
        pyproject_file: String,
    },
    Pip {
        requirements_file: String,
        dev_requirements_file: String,
    },
    Uv {
        requirements_file: String,
        dev_requirements_file: String,
    },
}

impl Default for UserConfig {
    fn default() -> Self {
        UserConfig {
            package_managers: Vec::new(),
            use_cuda: false,
            cuda_version: None,
            cudnn_version: None,
            python_version: "3.12".to_string(),
            use_conda: false,
            virtual_env_type: VirtualEnvType::None,
            pip_index_url: None,
            pip_trusted_hosts: None,
            uv_installed: false,
            poetry_installed: false,
        }
    }
}

/// Check if system Python is available
pub fn check_system_python() -> Result<Option<String>> {
    let output = Command::new("python3").arg("--version").output();

    match output {
        Ok(output) if output.status.success() => {
            let version_str = if !output.stdout.is_empty() {
                String::from_utf8_lossy(&output.stdout)
            } else {
                String::from_utf8_lossy(&output.stderr)
            };

            let version = version_str.split_whitespace().nth(1).map(|s| s.to_string());
            Ok(version)
        }
        _ => Ok(None),
    }
}

/// Check if Conda is available
pub fn check_conda_available() -> Result<bool> {
    let output = Command::new("conda").arg("--version").output();
    Ok(output.is_ok() && output.unwrap().status.success())
}

/// Check CUDA availability
pub fn check_cuda_availability() -> Result<bool> {
    // Check if nvidia-smi is available
    if Command::new("nvidia-smi")
        .arg("--query-gpu=gpu_name")
        .output()
        .is_err()
    {
        return Ok(false);
    }

    // Check if CUDA toolkit is installed
    if Command::new("nvcc").arg("--version").output().is_err() {
        return Ok(false);
    }

    // Check if CUDA libraries are available
    if cfg!(target_os = "linux") && !std::path::Path::new("/usr/local/cuda").exists() {
        return Ok(false);
    }

    Ok(true)
}

/// Check if UV is available
pub fn check_uv_available() -> Result<bool> {
    let output = Command::new("uv").arg("--version").output();
    Ok(output.is_ok() && output.unwrap().status.success())
}

/// Install UV package manager based on operating system
pub fn install_uv() -> Result<bool> {
    cli_ui::display_info("Attempting to install UV package manager...");

    if cfg!(target_os = "windows") {
        let install_cmd = Command::new("powershell")
            .args([
                "-ExecutionPolicy",
                "ByPass",
                "-c",
                "irm https://astral.sh/uv/install.ps1 | iex",
            ])
            .status();

        if let Ok(status) = install_cmd {
            if status.success() {
                cli_ui::display_success("UV installation completed successfully.");
                cli_ui::display_info("You may need to restart your terminal to use UV.");
                return Ok(true);
            } else {
                cli_ui::display_error("UV installation failed with non-zero exit code.");
                return Ok(false);
            }
        } else {
            cli_ui::display_error("Failed to execute UV installation command.");
            return Ok(false);
        }
    } else {
        // For Unix systems (macOS/Linux)
        let install_cmd = Command::new("sh")
            .arg("-c")
            .arg("curl -LsSf https://astral.sh/uv/install.sh | sh")
            .status();

        if let Ok(status) = install_cmd {
            if status.success() {
                cli_ui::display_success("UV installation completed successfully.");

                // Source the environment file to make UV available in current session
                if cfg!(target_os = "macos") || cfg!(target_os = "linux") {
                    cli_ui::display_info("Making UV available in current session...");

                    // Try to source the environment file
                    let source_cmd = Command::new("sh")
                        .arg("-c")
                        .arg("source $HOME/.local/bin/env || true")
                        .status();

                    if source_cmd.is_err() || !source_cmd.unwrap().success() {
                        cli_ui::display_warning(
                            "Could not automatically make UV available in current session.",
                        );
                    }

                    // Check if UV is now in PATH after sourcing
                    let uv_check = Command::new("sh").arg("-c").arg("command -v uv").output();

                    if uv_check.is_err() || !uv_check.unwrap().status.success() {
                        cli_ui::display_warning(
                            "UV is installed but not available in current PATH.",
                        );
                        cli_ui::display_info("To use UV immediately, run one of these commands:");
                        cli_ui::display_info("  source $HOME/.local/bin/env");
                        cli_ui::display_info("  export PATH=\"$HOME/.local/bin:$PATH\"");
                        cli_ui::display_info("Or restart your terminal session.");
                    } else {
                        cli_ui::display_success(
                            "UV is now available in your current terminal session.",
                        );
                    }
                }

                return Ok(true);
            } else {
                cli_ui::display_error("UV installation script failed with non-zero exit code.");
                return Ok(false);
            }
        } else {
            cli_ui::display_error("Failed to execute UV installation script.");
            return Ok(false);
        }
    }
}

/// Check if Poetry is available
pub fn check_poetry_available() -> Result<bool> {
    let output = Command::new("poetry").arg("--version").output();
    Ok(output.is_ok() && output.unwrap().status.success())
}

/// Install Poetry package manager
pub fn install_poetry() -> Result<bool> {
    cli_ui::display_info("Attempting to install Poetry package manager...");

    let installer_cmd = if cfg!(target_os = "windows") {
        Command::new("powershell")
            .arg("-Command")
            .arg("(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | py -")
            .status()
    } else {
        Command::new("sh")
            .arg("-c")
            .arg("curl -sSL https://install.python-poetry.org | python3 -")
            .status()
    };

    if let Ok(status) = installer_cmd {
        return Ok(status.success());
    }

    Ok(false)
}

/// Get Python project configuration through interactive prompts
pub fn get_python_config() -> Result<UserConfig> {
    let mut config = UserConfig::default();

    // Check system Python availability and Conda availability
    let system_python = check_system_python()?;
    let conda_available = check_conda_available()?;

    // 1. Select Python version
    cli_ui::display_header("Python Configuration", "🐍");
    cli_ui::display_info("Select Python version:");
    let mut python_options = vec![
        "Python 3.12 (latest, recommended)",
        "Python 3.11 (stable)",
        "Python 3.10 (stable)",
        "Python 3.9 (stable)",
        "Custom version (e.g., 3.13, 3.14)",
    ];

    // Add system Python option if available
    let system_python_option;
    if let Some(ver) = &system_python {
        system_python_option = format!("Use system Python (version {})", ver);
        python_options.push(&system_python_option[..]);
    }

    let selection = cli_ui::prompt_select("Select Python version", &python_options)?;

    // Adjust selection index based on whether system Python is present
    let base_selection = if system_python.is_some() && selection == 5 {
        5
    } else {
        selection
    };

    // Set the selected Python version
    config.python_version = match base_selection {
        1 => "3.11".to_string(),
        2 => "3.10".to_string(),
        3 => "3.9".to_string(),
        4 => {
            // Custom version
            let version: String = cli_ui::prompt_input("Enter Python version (e.g., 3.13):", None)?;
            if version.matches('.').count() == 1
                && version.split('.').all(|n| n.parse::<u32>().is_ok())
            {
                version
            } else {
                "3.12".to_string() // Default value
            }
        }
        5 => system_python.clone().unwrap(),
        _ => "3.12".to_string(),
    };

    // 2. Enforce Conda as environment management method
    println!("\n🔧 Environment management:");
    cli_ui::display_info("Conda will be used as the environment management tool for this project.");
    
    if !conda_available {
        cli_ui::display_warning("Conda is not installed on your system.");
        cli_ui::display_info("You need to install Conda to continue with project creation.");
        
        let install_now = cli_ui::prompt_confirm("Would you like to install Conda now?", true)?;
        
        if !install_now {
            cli_ui::display_error("Conda is required for project creation. Exiting configuration.");
            return Err(std::io::Error::new(
                std::io::ErrorKind::Other,
                "Conda installation was cancelled by user",
            ).into());
        }
        
        // Setup Conda environment will handle installation
    }
    
    // Set Conda as the only option
    config.use_conda = true;
    config.virtual_env_type = VirtualEnvType::Conda;
    setup_conda_environment(&mut config, conda_available)?;

    // 3. Ask about package management (always conda + another package manager)
    println!("\n📦 Python package management:");
    cli_ui::display_info("Conda will be used as the base package manager plus one of the following:");
    
    let pkg_options = vec![
        "Conda only (use conda for all package management)",
        "Conda + Poetry (recommended for modern Python projects)",
        "Conda + uv (fastest package manager with optimized dependency resolution)",
    ];

    let selection = cli_ui::prompt_select("Select package management combination:", &pkg_options)?;
    
    // Always add Conda as the first package manager
    config.package_managers.push(PackageManager::Conda {
        channels: Vec::new(), // Will be populated in setup_conda_environment
        environment_file: "environment.yml".to_string(),
        dev_environment_file: "environment-dev.yml".to_string(),
    });

    // Add the selected additional package manager
    match selection {
        0 => {
            // Conda only option - do nothing as we already added Conda
            cli_ui::display_info("Using Conda for all package management.");
        },
        1 => {
            // Poetry option processing
            // Check if Poetry is installed
            let poetry_available = check_poetry_available()?;
            if !poetry_available {
                cli_ui::display_warning("Poetry package manager not found on your system.");

                // Offer to install Poetry automatically
                let install_now =
                    cli_ui::prompt_confirm("Would you like to install Poetry now?", true)?;

                if install_now {
                    let install_success = install_poetry()?;

                    if install_success {
                        cli_ui::display_success("Poetry was successfully installed!");
                        config.poetry_installed = true;
                    } else {
                        cli_ui::display_error("Failed to install Poetry automatically.");
                        cli_ui::display_warning(
                            "To install Poetry manually, run the following command:",
                        );
                        if cfg!(target_os = "windows") {
                            cli_ui::display_info("(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | py -");
                        } else {
                            cli_ui::display_info(
                                "curl -sSL https://install.python-poetry.org | python3 -",
                            );
                        }

                        let proceed_without_poetry = cli_ui::prompt_confirm(
                            "Continue without Poetry? (You'll need to install it later)",
                            true,
                        )?;
                        if !proceed_without_poetry {
                            return get_python_config(); // Restart the configuration process
                        }
                    }
                } else {
                    // User chose not to install now
                    let proceed_without_poetry = cli_ui::prompt_confirm(
                        "Continue without Poetry? (You'll need to install it later)",
                        true,
                    )?;
                    if !proceed_without_poetry {
                        return get_python_config(); // Restart the configuration process
                    }
                }
            } else {
                config.poetry_installed = true;
            }

            config.package_managers.push(PackageManager::Poetry {
                pyproject_file: "pyproject.toml".to_string(),
            });
        }
        2 => {
            // Check if UV is installed
            let uv_available = check_uv_available()?;

            // For Conda+UV combination, we'll install UV in the Conda environment later
            if config.use_conda {
                cli_ui::display_info(
                    "UV will be installed in your Conda environment during project setup.",
                );
                config.uv_installed = true; // Mark as installed so we can handle it during conda env creation
            } else if !uv_available {
                // For non-Conda environments, offer to install UV globally
                cli_ui::display_warning("UV package manager not found on your system.");

                // Offer to install UV automatically
                let install_now =
                    cli_ui::prompt_confirm("Would you like to install UV now?", true)?;

                if install_now {
                    let install_success = install_uv()?;

                    if install_success {
                        cli_ui::display_success("UV was successfully installed!");
                        config.uv_installed = true;

                        // 添加关于PATH的重要提示
                        cli_ui::display_warning("⚠️ Important: UV is installed but you may need to restart your terminal or update your PATH to use it.");
                        if cfg!(target_os = "windows") {
                            cli_ui::display_info("The UV installer should have added it to your PATH automatically.");
                        } else {
                            cli_ui::display_info(
                                "To use UV immediately in this terminal session, run:",
                            );
                            cli_ui::display_info("  source $HOME/.local/bin/env");
                            cli_ui::display_info(
                                "Or restart your terminal to use it in a new session.",
                            );
                        }
                    } else {
                        cli_ui::display_error("Failed to install UV automatically.");
                        cli_ui::display_warning(
                            "To install UV manually, run the following command:",
                        );
                        if cfg!(target_os = "windows") {
                            cli_ui::display_info("powershell -ExecutionPolicy ByPass -c \"irm https://astral.sh/uv/install.ps1 | iex\"");
                        } else {
                            cli_ui::display_info(
                                "curl -LsSf https://astral.sh/uv/install.sh | sh",
                            );
                        }

                        let proceed_without_uv = cli_ui::prompt_confirm(
                            "Continue without UV? (You'll need to install it later)",
                            true,
                        )?;
                        if !proceed_without_uv {
                            return get_python_config(); // Restart the configuration process
                        }
                    }
                } else {
                    // User chose not to install now
                    let proceed_without_uv = cli_ui::prompt_confirm(
                        "Continue without UV? (You'll need to install it later)",
                        true,
                    )?;
                    if !proceed_without_uv {
                        return get_python_config(); // Restart the configuration process
                    }
                }
            } else {
                // UV is already available
                config.uv_installed = true;
            }

            config.package_managers.push(PackageManager::Uv {
                requirements_file: "requirements.txt".to_string(),
                dev_requirements_file: "requirements-dev.txt".to_string(),
            });
        }
        _ => {}
    }

    // 4. Choose package source for pip/uv/poetry (if any of these managers are selected)
    if !config
        .package_managers
        .iter()
        .all(|pm| matches!(pm, PackageManager::Conda { .. }))
    {
        println!("\n📦 Package Index Configuration:");
        let index_options = &[
            "PyPI (default)",
            "Tsinghua Mirror (faster in China)",
            "Aliyun Mirror (faster in China)",
            "Custom index URL",
        ];

        let selection = cli_ui::prompt_select("Select package index:", index_options)?;

        match selection {
            1 => {
                config.pip_index_url = Some("https://pypi.tuna.tsinghua.edu.cn/simple".to_string());
                config.pip_trusted_hosts = Some(vec!["pypi.tuna.tsinghua.edu.cn".to_string()]);
            }
            2 => {
                config.pip_index_url = Some("https://mirrors.aliyun.com/pypi/simple".to_string());
                config.pip_trusted_hosts = Some(vec!["mirrors.aliyun.com".to_string()]);
            }
            3 => {
                let index_url: String = cli_ui::prompt_input("Enter custom index URL:", None)?;

                // Extract host from URL
                let host = index_url
                    .replace("http://", "")
                    .replace("https://", "")
                    .split('/')
                    .next()
                    .unwrap_or("")
                    .to_string();

                config.pip_index_url = Some(index_url);
                if !host.is_empty() {
                    config.pip_trusted_hosts = Some(vec![host]);
                }
            }
            _ => {
                config.pip_index_url = None;
                config.pip_trusted_hosts = None;
            }
        }
    }

    // 5. Check CUDA availability
    let cuda_available = check_cuda_availability()?;
    if cuda_available {
        let use_cuda = cli_ui::prompt_confirm("Enable CUDA support?", true)?;
        if use_cuda {
            config.use_cuda = true;
            config.cuda_version = Some("11.8".to_string());
            config.cudnn_version = Some("8.9".to_string());
        }
    }

    // 6. Display configuration summary
    println!("\n📋 Configuration Summary:");
    println!("Python Version: {}", config.python_version);
    println!(
        "Environment Management: {}",
        if config.use_conda {
            "Conda"
        } else {
            "System Python"
        }
    );
    println!(
        "Virtual Environment: {}",
        match config.virtual_env_type {
            VirtualEnvType::Conda => "Conda",
            VirtualEnvType::None => "None",
        }
    );
    println!("Package Managers:");
    for pm in &config.package_managers {
        match pm {
            PackageManager::Conda { channels, .. } => {
                println!("  - Conda (channels: {})", channels.join(", "));
            }
            PackageManager::Poetry { .. } => println!("  - Poetry"),
            PackageManager::Uv { .. } => println!("  - uv"),
            PackageManager::Pip { .. } => println!("  - pip"),
        }
    }
    if config.use_cuda {
        println!(
            "CUDA Support: Yes (CUDA {}, cuDNN {})",
            config.cuda_version.as_ref().unwrap(),
            config.cudnn_version.as_ref().unwrap()
        );
    }

    // 7. Ask if continuing with installation
    let proceed = cli_ui::prompt_confirm("\n🚀 Proceed with installation?", true)?;
    if !proceed {
        return Err(io::Error::new(io::ErrorKind::Other, "Installation cancelled by user").into());
    }

    Ok(config)
}

/// Setup Conda environment
fn setup_conda_environment(config: &mut UserConfig, conda_available: bool) -> Result<()> {
    if !conda_available {
        println!("\n🔧 Conda is not installed. Would you like to install it now?");
        let conda_options = &[
            "Install Miniconda (minimal installation)",
            "Install Anaconda (full installation)",
            "Install Conda-forge",
        ];

        let selection = cli_ui::prompt_select("Select Conda distribution:", conda_options)?;

        let distribution = match selection {
            1 => CondaDistribution::Anaconda,
            2 => CondaDistribution::CondaForge,
            _ => CondaDistribution::Miniconda,
        };

        install_conda(distribution)?;
    }

    // Ask for Conda channel selection
    println!("\n📦 Conda channel selection:");
    println!(
        "You can select multiple channels. Conda will search packages in the order specified."
    );
    let channel_options = &[
        "conda-forge (recommended general channel)",
        "defaults (Anaconda default channel)",
        "bioconda (for bioinformatics packages)",
        "pytorch (for PyTorch and related packages)",
        "nvidia (for CUDA and GPU acceleration)",
        "r (for R programming language packages)",
        "Add custom channel",
    ];

    println!("Select channels (enter numbers separated by commas, e.g., '1,3,5'):");
    for (i, option) in channel_options.iter().enumerate() {
        println!("{}. {}", i + 1, option);
    }

    let mut selected_channels: Vec<String> = Vec::new();
    loop {
        let input: String = cli_ui::prompt_input("> ", None)?;

        let trimmed = input.trim();
        if trimmed.to_lowercase() == "done" || trimmed.is_empty() {
            // If no channels selected, add conda-forge by default
            if selected_channels.is_empty() {
                selected_channels.push("conda-forge".to_string());
                println!("No channels selected, using conda-forge as default.");
            }
            break;
        }

        // Parse selected channel numbers
        for choice in trimmed.split(',') {
            let choice = choice.trim();
            if let Ok(num) = choice.parse::<usize>() {
                match num {
                    1 => {
                        if !selected_channels.contains(&"conda-forge".to_string()) {
                            selected_channels.push("conda-forge".to_string());
                            println!("Added conda-forge channel");
                        }
                    }
                    2 => {
                        if !selected_channels.contains(&"defaults".to_string()) {
                            selected_channels.push("defaults".to_string());
                            println!("Added defaults channel");
                        }
                    }
                    3 => {
                        if !selected_channels.contains(&"bioconda".to_string()) {
                            selected_channels.push("bioconda".to_string());
                            println!("Added bioconda channel");
                        }
                    }
                    4 => {
                        if !selected_channels.contains(&"pytorch".to_string()) {
                            selected_channels.push("pytorch".to_string());
                            println!("Added pytorch channel");
                        }
                    }
                    5 => {
                        if !selected_channels.contains(&"nvidia".to_string()) {
                            selected_channels.push("nvidia".to_string());
                            println!("Added nvidia channel");
                        }
                    }
                    6 => {
                        if !selected_channels.contains(&"r".to_string()) {
                            selected_channels.push("r".to_string());
                            println!("Added r channel");
                        }
                    }
                    7 => {
                        let custom_channel: String =
                            cli_ui::prompt_input("Enter custom channel name:", None)?;
                        if !custom_channel.is_empty()
                            && !selected_channels.contains(&custom_channel)
                        {
                            selected_channels.push(custom_channel.clone());
                            println!("Added custom channel: {}", custom_channel);
                        }
                    }
                    _ => println!("Invalid choice: {}", choice),
                }
            } else {
                println!("Invalid input: {}", choice);
            }
        }

        println!(
            "Currently selected channels: {}",
            selected_channels.join(", ")
        );
        println!("Enter more channel numbers or type 'done' to finish selection");
    }

    // 获取已添加的Conda包管理器并设置channels
    for pm in &mut config.package_managers {
        if let PackageManager::Conda { channels, .. } = pm {
            *channels = selected_channels;
            break;
        }
    }

    Ok(())
}

/// Install Conda
pub fn install_conda(distribution: CondaDistribution) -> Result<()> {
    let (install_script, base_url) = match distribution {
        CondaDistribution::Miniconda => {
            let script = if cfg!(target_os = "linux") {
                "Miniconda3-latest-Linux-x86_64.sh"
            } else if cfg!(target_os = "macos") {
                if cfg!(target_arch = "aarch64") {
                    "Miniconda3-latest-MacOSX-arm64.sh"
                } else {
                    "Miniconda3-latest-MacOSX-x86_64.sh"
                }
            } else {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    "Unsupported operating system",
                )
                .into());
            };
            (script, "https://repo.anaconda.com/miniconda/")
        }
        CondaDistribution::Anaconda => {
            let script = if cfg!(target_os = "linux") {
                "Anaconda3-latest-Linux-x86_64.sh"
            } else if cfg!(target_os = "macos") {
                if cfg!(target_arch = "aarch64") {
                    "Anaconda3-latest-MacOSX-arm64.sh"
                } else {
                    "Anaconda3-latest-MacOSX-x86_64.sh"
                }
            } else {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    "Unsupported operating system",
                )
                .into());
            };
            (script, "https://repo.anaconda.com/archive/")
        }
        CondaDistribution::CondaForge => {
            let script = if cfg!(target_os = "linux") {
                "Miniforge3-Linux-x86_64.sh"
            } else if cfg!(target_os = "macos") {
                if cfg!(target_arch = "aarch64") {
                    "Miniforge3-MacOSX-arm64.sh"
                } else {
                    "Miniforge3-MacOSX-x86_64.sh"
                }
            } else {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    "Unsupported operating system",
                )
                .into());
            };
            (
                script,
                "https://github.com/conda-forge/miniforge/releases/latest/download/",
            )
        }
    };

    // Download installation script
    Command::new("curl")
        .arg("-O")
        .arg(format!("{}{}", base_url, install_script))
        .status()?;

    // Run installation script
    Command::new("bash")
        .arg(install_script)
        .arg("-b")
        .arg("-p")
        .arg(format!("{}/.conda", std::env::var("HOME").unwrap()))
        .status()?;

    // Delete installation script
    std::fs::remove_file(install_script)?;

    // Initialize conda
    Command::new("conda").arg("init").status()?;

    // If conda-forge, set default channel
    if let CondaDistribution::CondaForge = distribution {
        Command::new("conda")
            .arg("config")
            .arg("--add")
            .arg("channels")
            .arg("conda-forge")
            .arg("--set")
            .status()?;
    }

    Ok(())
}

/// Check system R installation
pub fn check_system_r() -> Result<Option<String>> {
    let output = Command::new("R").arg("--version").output();

    match output {
        Ok(output) if output.status.success() => {
            let version_output = String::from_utf8_lossy(&output.stdout);
            // R typically outputs version info in the first line, e.g. "R version 4.2.1 (2022-06-23) -- "Bird Hippie""
            let version = version_output.lines().next().and_then(|line| {
                line.split("R version ")
                    .nth(1)
                    .and_then(|v| v.split(' ').next())
                    .map(|s| s.to_string())
            });
            Ok(version)
        }
        _ => Ok(None),
    }
}

/// Check if rig (R Installation Manager) is available on the system
pub fn check_rig_available() -> Result<bool> {
    let cmd = Command::new("rig").arg("--version").status();
    Ok(cmd.is_ok() && cmd.unwrap().success())
}

/// Check system MATLAB installation
pub fn check_matlab_available() -> Result<Option<String>> {
    // Try running matlab -batch version command
    let output = Command::new("matlab").arg("-batch").arg("version").output();

    match output {
        Ok(output) if output.status.success() => {
            let version_output = String::from_utf8_lossy(&output.stdout);
            // Parse the MATLAB version string
            // Version typically appears in format like "R2023a"
            // We can extract it by scanning for the pattern "R20xx"
            let version = version_output
                .lines()
                .find_map(|line| {
                    if line.contains("R20") {
                        line.split_whitespace()
                            .find(|word| word.starts_with("R20"))
                            .map(|s| s.to_string())
                    } else {
                        None
                    }
                })
                .or_else(|| {
                    // Alternative method: just get the last word
                    version_output
                        .split_whitespace()
                        .last()
                        .map(|s| s.to_string())
                });

            Ok(version)
        }
        _ => {
            // Try alternative command for detecting MATLAB
            // On some systems, matlab -n might provide version info
            let alt_output = Command::new("matlab").arg("-n").output();

            match alt_output {
                Ok(output) if output.status.success() => {
                    let version_output = String::from_utf8_lossy(&output.stdout);
                    // Try to extract the version
                    let version = version_output.lines().find_map(|line| {
                        if line.contains("R20") {
                            line.split_whitespace()
                                .find(|word| word.starts_with("R20"))
                                .map(|s| s.to_string())
                        } else {
                            None
                        }
                    });

                    Ok(version)
                }
                _ => Ok(None),
            }
        }
    }
}
