use super::super::config::{
    check_poetry_available, check_uv_available, install_poetry, install_uv, PackageManager,
    UserConfig, VirtualEnvType,
};
use super::super::utils::write_file;
use crate::error::Result;
use crate::utils::cli_ui;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Create a Python project with the specified configuration
pub fn create_python_project(project_dir: &PathBuf, config: &UserConfig) -> Result<()> {
    // 添加检查conda版本功能
    check_conda_version()?;
    
    // Check if required package managers are available
    let has_uv = config
        .package_managers
        .iter()
        .any(|pm| matches!(pm, PackageManager::Uv { .. }));

    let has_poetry = config
        .package_managers
        .iter()
        .any(|pm| matches!(pm, PackageManager::Poetry { .. }));

    // Only check and install package managers if they weren't already installed in the config step
    if has_uv && !config.uv_installed {
        let uv_available = check_uv_available()?;
        if !uv_available {
            cli_ui::display_warning("UV package manager is required but not found on your system.");

            // Offer to install UV automatically now
            let install_now = cli_ui::prompt_confirm("Would you like to install UV now?", true)?;

            if install_now {
                let installation_success = install_uv()?;

                if installation_success {
                    cli_ui::display_success("UV was successfully installed!");
                } else {
                    cli_ui::display_error("Failed to install UV automatically.");
                    cli_ui::display_warning(
                        "Some project commands may not work until UV is installed.",
                    );
                    cli_ui::display_info("To install UV later, run one of these commands:");
                    if cfg!(target_os = "windows") {
                        cli_ui::display_info("powershell -ExecutionPolicy ByPass -c \"irm https://astral.sh/uv/install.ps1 | iex\"");
                    } else {
                        cli_ui::display_info("curl -LsSf https://astral.sh/uv/install.sh | sh");
                    }
                }
            } else {
                // User chose not to install now
                cli_ui::display_info("You chose to continue without installing UV.");
                cli_ui::display_warning(
                    "Some project commands may not work until UV is installed.",
                );
                cli_ui::display_info("To install UV later, run one of these commands:");
                if cfg!(target_os = "windows") {
                    cli_ui::display_info("powershell -ExecutionPolicy ByPass -c \"irm https://astral.sh/uv/install.ps1 | iex\"");
                } else {
                    cli_ui::display_info("curl -LsSf https://astral.sh/uv/install.sh | sh");
                }
            }
        }
    }

    if has_poetry && !config.poetry_installed {
        let poetry_available = check_poetry_available()?;
        if !poetry_available {
            cli_ui::display_warning(
                "Poetry package manager is required but not found on your system.",
            );

            // Offer to install Poetry automatically now
            let install_now =
                cli_ui::prompt_confirm("Would you like to install Poetry now?", true)?;

            if install_now {
                let installation_success = install_poetry()?;

                if installation_success {
                    cli_ui::display_success("Poetry was successfully installed!");
                } else {
                    cli_ui::display_error("Failed to install Poetry automatically.");
                    cli_ui::display_warning(
                        "Some project commands may not work until Poetry is installed.",
                    );
                    cli_ui::display_info("To install Poetry later, run one of these commands:");
                    if cfg!(target_os = "windows") {
                        cli_ui::display_info("(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | py -");
                    } else {
                        cli_ui::display_info(
                            "curl -sSL https://install.python-poetry.org | python3 -",
                        );
                    }
                }
            } else {
                // User chose not to install now
                cli_ui::display_info("You chose to continue without installing Poetry.");
                cli_ui::display_warning(
                    "Some project commands may not work until Poetry is installed.",
                );
                cli_ui::display_info("To install Poetry later, run one of these commands:");
                if cfg!(target_os = "windows") {
                    cli_ui::display_info("(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | py -");
                } else {
                    cli_ui::display_info("curl -sSL https://install.python-poetry.org | python3 -");
                }
            }
        }
    }

    // Create project directory
    fs::create_dir_all(project_dir)?;
    std::env::set_current_dir(project_dir)?;

    cli_ui::display_info("Creating basic Python project structure...");
    // Create basic project structure
    let dirs = [
        "src",
        "tests",
        "docs",
        "examples",
        "scripts",
        "data",
        "notebooks",
    ];
    for dir in &dirs {
        fs::create_dir_all(dir)?;
    }

    cli_ui::display_info("Generating project files...");
    // Create README.md
    // Get package manager information
    let package_managers_desc = config
        .package_managers
        .iter()
        .map(|pm| match pm {
            PackageManager::Conda { .. } => "Conda",
            PackageManager::Poetry { .. } => "Poetry",
            PackageManager::Pip { .. } => "pip",
            PackageManager::Uv { .. } => "uv",
        })
        .collect::<Vec<_>>()
        .join(" + ");

    // Extract project name from project directory
    let project_name = project_dir
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("my-project");

    let readme_content = format!(
        "# {}\n\n\
        ## Project Overview\n\
        This is a Python {} project.\n\n\
        ## Requirements\n\
        - Python {}\n\
        - Environment Management: {}\n\
        - Package Manager: {}\n\n\
        ## Installation\n\
        ```bash\n\
        # Clone the repository\n\
        git clone <repository-url>\n\
        cd {}\n\n\
        # Install dependencies\n\
        {}\n\
        ```\n\n\
        ## Usage\n\
        To be added\n\n\
        ## Development\n\
        ```bash\n\
        # Install development dependencies\n\
        {}\n\
        ```\n\n\
        ## Testing\n\
        ```bash\n\
        # Run tests\n\
        {}\n\
        ```\n\n\
        ## Important Notes\n\
        - If using Conda: Remember to activate the environment before running commands with `conda activate {}`\n\
        - The environment name matches the project directory name by default\n{}\n\n\
        ## License\n\
        MIT\n",
        project_dir.display(),
        config.python_version,
        config.python_version,
        if config.use_conda {
            "Conda"
        } else {
            match config.virtual_env_type {
                VirtualEnvType::Venv => "venv",
                VirtualEnvType::Virtualenv => "virtualenv",
                _ => "System Python",
            }
        },
        package_managers_desc,
        project_dir.display(),
        get_install_command(config),
        get_dev_install_command(config),
        get_test_command(config),
        project_name,
        if config.use_conda {
            "\n- If using direnv: Run `direnv allow` in the project directory to enable automatic environment activation when entering the directory"
        } else {
            ""
        }
    );
    write_file(&Path::new("README.md"), &readme_content)?;

    // Create .gitignore
    let gitignore_content = "\
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# IDE
.idea/
.vscode/
*.swp
*.swo

# Jupyter Notebook
.ipynb_checkpoints

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Unit test / coverage reports
htmlcov/
.tox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
.hypothesis/

# Logs
*.log
logs/
";
    write_file(&Path::new(".gitignore"), gitignore_content)?;

    // Create main module
    let src_dir = Path::new("src").join(
        project_dir
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("my-project"),
    );
    fs::create_dir_all(&src_dir)?;

    // Create __init__.py
    let project_name = project_dir
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("my-project");

    write_file(
        &src_dir.join("__init__.py"),
        &format!(
            "\"\"\"{} package.\"\"\"\n\n\
            __version__ = \"0.1.0\"\n",
            project_name
        ),
    )?;

    // Create main.py
    write_file(
        &src_dir.join("main.py"),
        "\"\"\"\
Main module.

This module provides the main entry point for the application.
\"\"\"\n\n\
def main():\n\
    \"\"\"\
    Run the main function.
    
    Returns:
        None
    \"\"\"\n\
    print(\"Hello, World!\")\n\n\
if __name__ == \"__main__\":\n\
    main()\n",
    )?;

    // Create test files
    let tests_dir = Path::new("tests");
    write_file(
        &tests_dir.join("__init__.py"),
        "\"\"\"Test package.\"\"\"\n",
    )?;
    write_file(
        &tests_dir.join("test_main.py"),
        "\"\"\"\
Test main module.

This module contains tests for the main module.
\"\"\"\n\n\
def test_main():\n\
    \"\"\"\
    Test main function.
    
    Returns:
        None
    \"\"\"\n\
    assert True\n",
    )?;

    // Based on user selection, create environment configuration files
    let has_conda = config
        .package_managers
        .iter()
        .any(|pm| matches!(pm, PackageManager::Conda { .. }));

    for package_manager in &config.package_managers {
        match package_manager {
            PackageManager::Conda {
                channels,
                environment_file,
                dev_environment_file,
            } => {
                // Create environment.yml
                let mut env_content = format!(
                    "name: {}\n\
                    channels:\n",
                    project_name
                );
                for channel in channels {
                    env_content.push_str(&format!("  - {}\n", channel));
                }
                env_content.push_str("\ndependencies:\n");
                if config.use_cuda {
                    env_content.push_str(&format!(
                        "  - python={}\n\
                        - cudatoolkit={}\n\
                        - cudnn={}\n",
                        config.python_version,
                        config.cuda_version.as_ref().unwrap(),
                        config.cudnn_version.as_ref().unwrap()
                    ));
                } else {
                    env_content.push_str(&format!("  - python={}\n", config.python_version));
                }
                env_content.push_str("  - pip\n");

                // Add packages that need to be installed via conda
                if config.use_cuda {
                    env_content.push_str("  - numpy\n  - scipy\n  - matplotlib\n  - pandas\n");
                }

                // If Poetry/pip/uv, add pip section
                if config
                    .package_managers
                    .iter()
                    .any(|pm| !matches!(pm, PackageManager::Conda { .. }))
                {
                    env_content.push_str("  - pip:\n");
                    env_content.push_str("    - -e .\n"); // Install current project
                }

                write_file(&Path::new(environment_file), &env_content)?;

                // Create dev-environment.yml
                let mut dev_env_content = format!(
                    "name: {}-dev\n\
                    channels:\n",
                    project_name
                );
                for channel in channels {
                    dev_env_content.push_str(&format!("  - {}\n", channel));
                }
                dev_env_content.push_str("\ndependencies:\n");
                if config.use_cuda {
                    dev_env_content.push_str(&format!(
                        "  - python={}\n\
                        - cudatoolkit={}\n\
                        - cudnn={}\n",
                        config.python_version,
                        config.cuda_version.as_ref().unwrap(),
                        config.cudnn_version.as_ref().unwrap()
                    ));
                } else {
                    dev_env_content.push_str(&format!("  - python={}\n", config.python_version));
                }
                dev_env_content.push_str(
                    "  - pip\n\
                     - pytest\n\
                     - black\n\
                     - isort\n\
                     - flake8\n",
                );

                // If Poetry/pip/uv, add pip section
                if config
                    .package_managers
                    .iter()
                    .any(|pm| !matches!(pm, PackageManager::Conda { .. }))
                {
                    dev_env_content.push_str("  - pip:\n");
                    dev_env_content.push_str("    - -e .\n"); // Install current project
                }

                write_file(&Path::new(dev_environment_file), &dev_env_content)?;
            }
            PackageManager::Poetry { pyproject_file } => {
                // Create pyproject.toml
                let pyproject_content = format!(
                    "[tool.poetry]\n\
                    name = \"{}\"\n\
                    version = \"0.1.0\"\n\
                    description = \"A Python project created with CRESP\"\n\
                    authors = [\"Your Name <your.email@example.com>\"]\n\
                    readme = \"README.md\"\n\
                    packages = [{{include = \"{}\", from = \"src\"}}]\n\n\
                    [tool.poetry.dependencies]\n\
                    python = \"^{}\"\n\n\
                    [tool.poetry.group.dev.dependencies]\n\
                    pytest = \"^7.0.0\"\n\
                    black = \"^23.0.0\"\n\
                    isort = \"^5.0.0\"\n\
                    flake8 = \"^6.0.0\"\n\n\
                    [build-system]\n\
                    requires = [\"poetry-core\"]\n\
                    build-backend = \"poetry.core.masonry.api\"\n",
                    project_name, project_name, config.python_version
                );
                write_file(&Path::new(pyproject_file), &pyproject_content)?;

                // If using conda, create .envrc file for automatic environment switching
                if has_conda {
                    let envrc_content = format!(
                        "# Automatically activate conda environment\n\
                        if [ -e ${{HOME}}/.conda/etc/profile.d/conda.sh ]; then\n\
                        \tsource ${{HOME}}/.conda/etc/profile.d/conda.sh\n\
                        \tconda activate {}\n\
                        fi\n",
                        project_name
                    );
                    write_file(&Path::new(".envrc"), &envrc_content)?;

                    // We don't need to create poetry config file here
                    // Instead, we'll configure Poetry during installation via 'poetry config virtualenvs.create false'
                    // in the install commands
                } else {
                    // Only create Poetry config for non-Conda environments if needed
                    // For pure Poetry projects, we typically want Poetry to manage its own virtualenvs
                }
            }
            PackageManager::Pip {
                requirements_file,
                dev_requirements_file,
            }
            | PackageManager::Uv {
                requirements_file,
                dev_requirements_file,
            } => {
                // Create requirements.txt
                let mut req_content = format!(
                    "# Python {}\n\
                    # Core dependencies\n",
                    config.python_version
                );
                if config.use_cuda && !has_conda {
                    req_content.push_str(
                        "torch>=2.0.0\n\
                        torchvision>=0.15.0\n\
                        torchaudio>=2.0.0\n",
                    );
                }
                write_file(&Path::new(requirements_file), &req_content)?;

                // Create requirements-dev.txt
                let dev_req_content = "\
                    # Development dependencies\n\
                    pytest>=7.0.0\n\
                    black>=23.0.0\n\
                    isort>=5.0.0\n\
                    flake8>=6.0.0\n";
                write_file(&Path::new(dev_requirements_file), dev_req_content)?;

                // If using conda and pip/uv, create .envrc file
                if has_conda {
                    let envrc_content = format!(
                        "# Automatically activate conda environment\n\
                        if [ -e ${{HOME}}/.conda/etc/profile.d/conda.sh ]; then\n\
                        \tsource ${{HOME}}/.conda/etc/profile.d/conda.sh\n\
                        \tconda activate {}\n\
                        fi\n",
                        project_name
                    );
                    write_file(&Path::new(".envrc"), &envrc_content)?;
                }
            }
        }
    }

    // Create virtual environment
    match config.virtual_env_type {
        VirtualEnvType::Venv => {
            let python_cmd = if config.use_conda {
                format!("conda run python={}", config.python_version)
            } else {
                "python".to_string()
            };
            Command::new(&python_cmd)
                .arg("-m")
                .arg("venv")
                .arg(".venv")
                .status()?;
        }
        VirtualEnvType::Virtualenv => {
            Command::new("virtualenv").arg(".venv").status()?;
        }
        VirtualEnvType::Conda => {
            if config.use_conda {
                // Get project name for conda environment
                let project_name = project_dir
                    .file_name()
                    .and_then(|name| name.to_str())
                    .unwrap_or("my-project");

                // Create the Conda environment
                let status = Command::new("conda")
                    .arg("create")
                    .arg("-n")
                    .arg(&project_name)
                    .arg(format!("python={}", config.python_version))
                    .arg("-y")
                    .status()?;

                if status.success() {
                    println!(
                        "✅ Successfully created Conda environment: {}",
                        project_name
                    );
                    println!("ℹ️ Note: To use this environment, you'll need to activate it with 'conda activate {}'", project_name);

                    // Check if Poetry/UV is used alongside Conda
                    let has_poetry = config
                        .package_managers
                        .iter()
                        .any(|pm| matches!(pm, PackageManager::Poetry { .. }));

                    let has_uv = config
                        .package_managers
                        .iter()
                        .any(|pm| matches!(pm, PackageManager::Uv { .. }));

                    // Install Poetry in Conda environment if needed
                    if has_poetry {
                        println!("🔧 Installing Poetry in Conda environment...");

                        let installation_success;

                        // On Unix systems, we need to use a different approach for Conda activation
                        if cfg!(target_os = "windows") {
                            // Windows approach
                            let install_cmd = format!("activate {} && pip install poetry && poetry config virtualenvs.create false", project_name);
                            let status = Command::new("cmd").arg("/c").arg(install_cmd).status()?;

                            installation_success = status.success();
                        } else {
                            // Unix approach (macOS, Linux)
                            // For Unix, it's better to use conda run which doesn't require shell activation
                            let pip_status = Command::new("conda")
                                .arg("run")
                                .arg("-n")
                                .arg(&project_name)
                                .arg("pip")
                                .arg("install")
                                .arg("poetry")
                                .status()?;

                            if pip_status.success() {
                                let config_status = Command::new("conda")
                                    .arg("run")
                                    .arg("-n")
                                    .arg(&project_name)
                                    .arg("poetry")
                                    .arg("config")
                                    .arg("virtualenvs.create")
                                    .arg("false")
                                    .status()?;

                                installation_success = config_status.success();
                            } else {
                                installation_success = false;
                            }
                        }

                        if installation_success {
                            println!("✅ Installed Poetry in Conda environment: {}", project_name);
                            println!("📝 Poetry configured to use Conda environment (no separate virtualenv)");
                            println!("\n⚠️ Important: You need to manually activate the new environment to use it:");
                            println!("   conda activate {}", project_name);
                        } else {
                            println!("⚠️ Failed to install Poetry in Conda environment.");
                            println!("📝 You can install it manually later using the commands in README.md");
                        }
                    }

                    // Install UV in Conda environment if needed
                    if has_uv {
                        cli_ui::display_info("Installing UV in Conda environment...");

                        let installation_success;

                        // On Unix systems, we need to use a different approach for Conda activation
                        if cfg!(target_os = "windows") {
                            // Windows approach
                            let install_cmd =
                                format!("activate {} && pip install uv", project_name);
                            let status = Command::new("cmd").arg("/c").arg(install_cmd).status()?;

                            installation_success = status.success();
                        } else {
                            // Unix approach (macOS, Linux)
                            // For Unix, it's better to use conda run which doesn't require shell activation
                            let pip_status = Command::new("conda")
                                .arg("run")
                                .arg("-n")
                                .arg(&project_name)
                                .arg("pip")
                                .arg("install")
                                .arg("uv")
                                .status()?;

                            installation_success = pip_status.success();
                        }

                        if installation_success {
                            cli_ui::display_success(&format!(
                                "Installed UV in Conda environment: {}",
                                project_name
                            ));
                            cli_ui::display_info(
                                "UV is now available within your Conda environment.",
                            );
                            cli_ui::display_warning("Important: You need to manually activate the new environment to use UV:");
                            cli_ui::display_info(&format!("   conda activate {}", project_name));
                        } else {
                            cli_ui::display_error("Failed to install UV in Conda environment.");
                            cli_ui::display_info("You can install it manually later with:");
                            cli_ui::display_info(&format!(
                                "   conda activate {} && pip install uv",
                                project_name
                            ));
                        }
                    }
                }
            }
        }
        VirtualEnvType::None => {}
    }

    // Final check and message to user
    // For Conda environments, show an activation message instead of checking for UV
    if config.use_conda {
        let project_name = project_dir
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("my-project");

        cli_ui::display_info("\nTo use this project and its installed tools:");
        cli_ui::display_info(&format!("  conda activate {}", project_name));

        if has_uv {
            cli_ui::display_info("After activating the Conda environment, UV will be available for managing packages.");
            cli_ui::display_info(&format!("  Example: conda activate {} && uv pip install numpy pandas", project_name));
        }

        if has_poetry {
            cli_ui::display_info("After activating the Conda environment, Poetry will be available for managing packages.");
            cli_ui::display_info(&format!("  Example: conda activate {} && poetry add numpy pandas", project_name));
        }
        
        // 检查direnv并给出合适的提示
        check_direnv_and_prompt(project_dir)?;
    } else if has_uv && config.uv_installed {
        // Only check global UV installation for non-Conda environments
        let uv_check = Command::new("sh").arg("-c").arg("command -v uv").output();

        match uv_check {
            Ok(output) if output.status.success() => {
                cli_ui::display_success("UV is properly installed and available in your PATH.");

                // 获取UV路径，辅助用户了解安装位置
                if let Ok(path_output) = Command::new("sh").arg("-c").arg("which uv").output() {
                    if path_output.status.success() {
                        let path = String::from_utf8_lossy(&path_output.stdout)
                            .trim()
                            .to_string();
                        cli_ui::display_info(&format!("UV is installed at: {}", path));
                    }
                }
                
                // 添加UV使用示例
                cli_ui::display_info("Example usage of UV for package management:");
                cli_ui::display_info("  uv pip install numpy pandas    # Install packages");
                cli_ui::display_info("  uv pip freeze > requirements.txt    # Generate requirements file");
            }
            _ => {
                cli_ui::display_warning(
                    "UV was installed but doesn't seem to be available in your PATH.",
                );
                cli_ui::display_info("To use UV, you may need to:");
                cli_ui::display_info("1. Run 'source $HOME/.local/bin/env' in your terminal");
                cli_ui::display_info("2. Or restart your terminal session");
                cli_ui::display_info("3. Or add $HOME/.local/bin to your PATH with: export PATH=\"$HOME/.local/bin:$PATH\"");

                // 尝试自动使UV在当前会话中可用
                if cfg!(target_os = "macos") || cfg!(target_os = "linux") {
                    cli_ui::display_info("Attempting to make UV available in current session...");

                    let source_cmd = Command::new("sh")
                        .arg("-c")
                        .arg("source $HOME/.local/bin/env || true")
                        .status();

                    if source_cmd.is_ok() && source_cmd.unwrap().success() {
                        // 重新检查UV是否可用
                        let recheck = Command::new("sh").arg("-c").arg("command -v uv").status();

                        if recheck.is_ok() && recheck.unwrap().success() {
                            cli_ui::display_success("UV is now available in your current session!");
                        }
                    }
                }
            }
        }
    }

    Ok(())
}

/// 检查conda版本并显示更新提示
fn check_conda_version() -> Result<()> {
    let output = Command::new("conda").arg("--version").output();
    
    if let Ok(output) = output {
        if output.status.success() {
            let version_str = String::from_utf8_lossy(&output.stdout);
            if let Some(version) = version_str.split_whitespace().nth(1) {
                // 简单版本比较，只比较主版本号
                let parts: Vec<&str> = version.split('.').collect();
                if parts.len() >= 2 {
                    if let Ok(major) = parts[0].parse::<u32>() {
                        if major < 23 {
                            cli_ui::display_warning(&format!("You are using an older version of conda ({}). Consider updating it for better performance and compatibility.", version));
                            cli_ui::display_info("To update conda, run: conda update -n base -c defaults conda");
                        }
                    }
                }
            }
        }
    }
    
    Ok(())
}

/// 检查direnv并提示用户如何启用它
fn check_direnv_and_prompt(project_dir: &Path) -> Result<()> {
    if Path::new(".envrc").exists() {
        // 检查direnv是否安装
        let direnv_check = Command::new("sh").arg("-c").arg("command -v direnv").output();
        
        match direnv_check {
            Ok(output) if output.status.success() => {
                cli_ui::display_info("\nA .envrc file has been created for automatic environment activation.");
                cli_ui::display_warning("Important: You need to run the following command to enable it:");
                cli_ui::display_info(&format!("  cd {} && direnv allow", project_dir.display()));
                cli_ui::display_info("This will allow automatic activation of the conda environment when you enter the project directory.");
            },
            _ => {
                cli_ui::display_info("\nA .envrc file has been created, but direnv is not installed on your system.");
                cli_ui::display_info("direnv allows automatic environment switching when entering project directories.");
                cli_ui::display_info("To install direnv:");
                
                if cfg!(target_os = "macos") {
                    cli_ui::display_info("  brew install direnv");
                } else if cfg!(target_os = "linux") {
                    cli_ui::display_info("  sudo apt-get install direnv  # For Debian/Ubuntu");
                    cli_ui::display_info("  sudo yum install direnv      # For CentOS/RHEL");
                } else {
                    cli_ui::display_info("  Visit https://direnv.net/docs/installation.html");
                }
                
                cli_ui::display_info("\nAfter installation, add to your shell profile (~/.bashrc, ~/.zshrc, etc.):");
                cli_ui::display_info("  eval \"$(direnv hook bash)\"  # For bash");
                cli_ui::display_info("  eval \"$(direnv hook zsh)\"   # For zsh");
                
                cli_ui::display_info("\nThen run:");
                cli_ui::display_info(&format!("  cd {} && direnv allow", project_dir.display()));
            }
        }
    }
    
    Ok(())
}

/// Generate installation command based on user configuration
fn get_install_command(config: &UserConfig) -> String {
    let mut commands = Vec::new();

    // If there is a Conda environment, activate it first
    let has_conda = config
        .package_managers
        .iter()
        .any(|pm| matches!(pm, PackageManager::Conda { .. }));

    // Check if Poetry is used alongside Conda
    let has_poetry = config
        .package_managers
        .iter()
        .any(|pm| matches!(pm, PackageManager::Poetry { .. }));

    // Check if UV is used alongside Conda
    let has_uv = config
        .package_managers
        .iter()
        .any(|pm| matches!(pm, PackageManager::Uv { .. }));

    let conda_with_poetry = has_conda && has_poetry;
    let conda_with_uv = has_conda && has_uv;

    if has_conda {
        commands.push("# Create and activate Conda environment".to_string());
        for pm in &config.package_managers {
            if let PackageManager::Conda {
                environment_file, ..
            } = pm
            {
                commands.push(format!("conda env create -f {}", environment_file));
                commands.push("conda activate $(basename $PWD)".to_string());

                // If using Poetry with Conda, install Poetry in the Conda environment
                if conda_with_poetry {
                    commands.push("\n# Install Poetry in Conda environment".to_string());
                    commands.push("pip install poetry".to_string());
                    commands.push("poetry config virtualenvs.create false".to_string());
                }

                // If using UV with Conda, install UV in the Conda environment
                if conda_with_uv {
                    commands.push("\n# Install UV in Conda environment".to_string());
                    commands.push("pip install uv".to_string());
                }
            }
        }
    }

    // Add installation commands for other package managers
    for package_manager in &config.package_managers {
        match package_manager {
            PackageManager::Conda { .. } => {
                // Already handled
            }
            PackageManager::Poetry { .. } => {
                if !conda_with_poetry {
                    // Only run this section if we're not using Poetry with Conda
                    // (otherwise Poetry is already installed in the Conda environment)
                    commands.push("# Install Poetry dependencies".to_string());

                    // Check if poetry is installed
                    if cfg!(target_os = "windows") {
                        commands.push("# Check if poetry is installed".to_string());
                        commands.push("where poetry >nul 2>&1".to_string());
                        commands.push("if %ERRORLEVEL% neq 0 (".to_string());
                        commands
                            .push("    echo Poetry not found. Installing Poetry...".to_string());
                        commands.push("    powershell -Command \"(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | py -\"".to_string());
                        commands.push(")".to_string());
                    } else {
                        commands.push("# Check if poetry is installed".to_string());
                        commands.push("if ! command -v poetry &> /dev/null; then".to_string());
                        commands.push(
                            "    echo \"Poetry not found. Installing Poetry...\"".to_string(),
                        );
                        commands.push(
                            "    curl -sSL https://install.python-poetry.org | python3 -"
                                .to_string(),
                        );
                        commands.push("fi".to_string());
                    }
                }

                // Add the Poetry install command
                commands.push("# Install project dependencies using Poetry".to_string());
                if has_conda {
                    commands.push("poetry install --no-interaction".to_string());
                } else {
                    commands.push("poetry install".to_string());
                }
            }
            PackageManager::Pip {
                requirements_file, ..
            } => {
                commands.push("# Install pip dependencies".to_string());
                commands.push(format!("pip install -r {}", requirements_file));
            }
            PackageManager::Uv {
                requirements_file, ..
            } => {
                if !conda_with_uv {
                    // Only run this section if we're not using UV with Conda
                    // (otherwise UV is already installed in the Conda environment)
                    commands.push("# Install uv - the fastest Python package manager with optimized dependency resolution".to_string());

                    if cfg!(target_os = "windows") {
                        commands.push("# Check if uv is installed".to_string());
                        commands.push("where uv >nul 2>&1".to_string());
                        commands.push("if %ERRORLEVEL% neq 0 (".to_string());
                        commands.push("    echo UV not found. Installing UV...".to_string());
                        commands.push("    powershell -ExecutionPolicy ByPass -c \"irm https://astral.sh/uv/install.ps1 | iex\"".to_string());
                        commands.push("    echo To use UV in this terminal session, please restart the terminal or run a new command prompt".to_string());
                        commands.push(")".to_string());
                    } else {
                        commands.push("# Check if uv is installed".to_string());
                        commands.push("if ! command -v uv &> /dev/null; then".to_string());
                        commands.push("    echo \"UV not found. Installing UV...\"".to_string());
                        commands.push(
                            "    curl -LsSf https://astral.sh/uv/install.sh | sh".to_string(),
                        );
                        commands.push("    echo \"To use UV in this terminal session, run: source $HOME/.local/bin/env\"".to_string());
                        commands.push("    source $HOME/.local/bin/env || true".to_string());
                        commands.push("fi".to_string());
                    }
                } else {
                    commands.push(
                        "# Use UV from Conda environment (fast dependency resolution)".to_string(),
                    );
                }

                // 添加镜像配置
                if let Some(index_url) = &config.pip_index_url {
                    commands.push("\n# Configure UV to use specified mirror".to_string());
                    commands.push(format!("# Set mirror: {}", index_url));
                    
                    if cfg!(target_os = "windows") {
                        commands.push(format!("uv pip config set global.index-url {}", index_url));
                    } else {
                        commands.push(format!("uv pip config set global.index-url {}", index_url));
                    }
                }

                // 如果设置了pip镜像，使用这个镜像安装
                if let Some(index_url) = &config.pip_index_url {
                    commands.push(format!("uv pip install -r {} --index-url {}", requirements_file, index_url));
                } else {
                    commands.push(format!("uv pip install -r {}", requirements_file));
                }
            }
        }
    }

    commands.join("\n")
}

/// Generate development installation command based on user configuration
fn get_dev_install_command(config: &UserConfig) -> String {
    let mut commands = Vec::new();

    // If there is a Conda environment, activate it first
    let has_conda = config
        .package_managers
        .iter()
        .any(|pm| matches!(pm, PackageManager::Conda { .. }));

    // Check if Poetry is used alongside Conda
    let has_poetry = config
        .package_managers
        .iter()
        .any(|pm| matches!(pm, PackageManager::Poetry { .. }));

    // Check if UV is used alongside Conda
    let has_uv = config
        .package_managers
        .iter()
        .any(|pm| matches!(pm, PackageManager::Uv { .. }));

    let conda_with_poetry = has_conda && has_poetry;
    let conda_with_uv = has_conda && has_uv;

    if has_conda {
        commands.push("# Create and activate Conda development environment".to_string());
        for pm in &config.package_managers {
            if let PackageManager::Conda {
                dev_environment_file,
                ..
            } = pm
            {
                commands.push(format!("conda env create -f {}", dev_environment_file));
                commands.push("conda activate $(basename $PWD)-dev".to_string());

                // If using Poetry with Conda, install Poetry in the Conda environment
                if conda_with_poetry {
                    commands.push("\n# Install Poetry in Conda environment".to_string());
                    commands.push("pip install poetry".to_string());
                    commands.push("poetry config virtualenvs.create false".to_string());
                }

                // If using UV with Conda, install UV in the Conda environment
                if conda_with_uv {
                    commands.push("\n# Install UV in Conda environment".to_string());
                    commands.push("pip install uv".to_string());
                }
            }
        }
    }

    // Add installation commands for other package managers
    for package_manager in &config.package_managers {
        match package_manager {
            PackageManager::Conda { .. } => {
                // Already handled
            }
            PackageManager::Poetry { .. } => {
                if !conda_with_poetry {
                    // Only run this section if we're not using Poetry with Conda
                    // (otherwise Poetry is already installed in the Conda environment)
                    commands.push("# Install Poetry development dependencies".to_string());

                    // Check if poetry is installed
                    if cfg!(target_os = "windows") {
                        commands.push("# Check if poetry is installed".to_string());
                        commands.push("where poetry >nul 2>&1".to_string());
                        commands.push("if %ERRORLEVEL% neq 0 (".to_string());
                        commands
                            .push("    echo Poetry not found. Installing Poetry...".to_string());
                        commands.push("    powershell -Command \"(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | py -\"".to_string());
                        commands.push(")".to_string());
                    } else {
                        commands.push("# Check if poetry is installed".to_string());
                        commands.push("if ! command -v poetry &> /dev/null; then".to_string());
                        commands.push(
                            "    echo \"Poetry not found. Installing Poetry...\"".to_string(),
                        );
                        commands.push(
                            "    curl -sSL https://install.python-poetry.org | python3 -"
                                .to_string(),
                        );
                        commands.push("fi".to_string());
                    }
                }

                // Add the Poetry install command
                commands
                    .push("# Install project development dependencies using Poetry".to_string());
                if has_conda {
                    commands.push("poetry install --no-interaction --with dev".to_string());
                } else {
                    commands.push("poetry install --with dev".to_string());
                }
            }
            PackageManager::Pip {
                dev_requirements_file,
                ..
            } => {
                commands.push("# Install pip development dependencies".to_string());
                commands.push(format!("pip install -r {}", dev_requirements_file));
            }
            PackageManager::Uv {
                dev_requirements_file,
                ..
            } => {
                if !conda_with_uv {
                    // Only run this section if we're not using UV with Conda
                    // (otherwise UV is already installed in the Conda environment)
                    commands.push(
                        "# Install uv development dependencies with fast resolution".to_string(),
                    );

                    if cfg!(target_os = "windows") {
                        commands.push("# Check if uv is installed".to_string());
                        commands.push("where uv >nul 2>&1".to_string());
                        commands.push("if %ERRORLEVEL% neq 0 (".to_string());
                        commands.push("    echo UV not found. Installing UV...".to_string());
                        commands.push("    powershell -ExecutionPolicy ByPass -c \"irm https://astral.sh/uv/install.ps1 | iex\"".to_string());
                        commands.push("    echo To use UV in this terminal session, please restart the terminal or run a new command prompt".to_string());
                        commands.push(")".to_string());
                    } else {
                        commands.push("# Check if uv is installed".to_string());
                        commands.push("if ! command -v uv &> /dev/null; then".to_string());
                        commands.push("    echo \"UV not found. Installing UV...\"".to_string());
                        commands.push(
                            "    curl -LsSf https://astral.sh/uv/install.sh | sh".to_string(),
                        );
                        commands.push("    echo \"To use UV in this terminal session, run: source $HOME/.local/bin/env\"".to_string());
                        commands.push("    source $HOME/.local/bin/env || true".to_string());
                        commands.push("fi".to_string());
                    }
                } else {
                    commands.push(
                        "# Use UV from Conda development environment (fast dependency resolution)"
                            .to_string(),
                    );
                }

                // 最后部分需要修改，确保使用镜像配置
                // 如果设置了pip镜像，使用这个镜像安装
                if let Some(index_url) = &config.pip_index_url {
                    commands.push(format!("uv pip install -r {} --index-url {}", dev_requirements_file, index_url));
                } else {
                    commands.push(format!("uv pip install -r {}", dev_requirements_file));
                }
            }
        }
    }

    commands.join("\n")
}

/// Generate test command based on user configuration  
fn get_test_command(config: &UserConfig) -> String {
    let mut commands = Vec::new();

    for package_manager in &config.package_managers {
        match package_manager {
            PackageManager::Conda { .. } => {
                commands.push("pytest".to_string());
            }
            PackageManager::Poetry { .. } => {
                commands.push("poetry run pytest".to_string());
            }
            PackageManager::Pip { .. } | PackageManager::Uv { .. } => {
                commands.push("pytest".to_string());
            }
        }
    }

    commands.join(" && ")
}
