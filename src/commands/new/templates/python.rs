use super::super::config::{
    PackageManager, UserConfig, VirtualEnvType,
};
use super::super::templates::conda_utils;
use super::super::templates::conda_utils::Language;
use super::super::utils::write_file;
use crate::error::Result;
use crate::utils::cli_ui;
use crate::utils::validation::exports::sanitize_for_conda_env;
use log::{debug, info, trace};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Create a Python project with the specified configuration
pub fn create_python_project(project_dir: &PathBuf, config: &mut UserConfig) -> Result<()> {
    // Check conda version if available
    if config.use_conda {
        if let Some(version) = conda_utils::ensure_conda_available()? {
            conda_utils::check_conda_version(&version)?;
        } else if config.virtual_env_type == VirtualEnvType::Conda {
            return Err(crate::error::Error::Environment(
                "Conda is required but not found".to_string(),
            ));
        }
    }

    // Check if required package managers are intended to be used
    let has_uv = config
        .package_managers
        .iter()
        .any(|pm| matches!(pm, PackageManager::Uv { .. }));

    let has_poetry = config
        .package_managers
        .iter()
        .any(|pm| matches!(pm, PackageManager::Poetry { .. }));

    // Create project directory
    fs::create_dir_all(project_dir)?;
    std::env::set_current_dir(project_dir)?;

    cli_ui::display_progress("1/4", "Creating Python project structure...");
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
        trace!("Created directory: {}", dir);
    }

    cli_ui::display_progress("2/4", "Generating project files...");
    
    // Extract project name from project directory
    let project_name = project_dir
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("my-project");

    // If using conda, validate and sanitize project name
    let mut conda_env_name = if config.use_conda {
        let sanitized = sanitize_for_conda_env(project_name);
        if sanitized != project_name {
            cli_ui::display_warning(&format!(
                "Project name '{}' contains characters not allowed in conda environment names.",
                project_name
            ));
            info!(
                "Using '{}' as the conda environment name instead.",
                sanitized
            );
        }
        sanitized
    } else {
        project_name.to_string()
    };
    // Create README.md
    create_readme(project_dir, project_name, &conda_env_name, config)?;

    // Create .gitignore
    create_gitignore(project_dir)?;

    // Create module files
    create_module_files(project_dir, project_name)?;

    // Create test files 
    create_test_files(project_dir)?;

    // Setup package managers and environment files
    setup_package_managers(project_dir, project_name, config)?;

    // Create virtual environment
    if config.virtual_env_type == VirtualEnvType::Conda && config.use_conda {
        cli_ui::display_progress("3/4", "Creating conda environment...");
        
        // Get the channels from config
        let channels = config
            .package_managers
            .iter()
            .find_map(|pm| {
                if let PackageManager::Conda { channels, .. } = pm {
                    Some(channels)
                } else {
                    None
                }
            });
            
        // Use the generic conda environment setup function
        let (env_created, actual_env_name) = conda_utils::create_language_conda_env(
            project_dir,
            project_name,
            &Language::Python,
            Some(&config.python_version),
            None,
            config.use_cuda,
            channels.as_deref().map(|v| &**v),
        )?;
        
        if !env_created {
            cli_ui::display_warning("Failed to create conda environment. You may need to create it manually.");
        } else {
            // use the actual created environment name, not the original name
            conda_env_name = actual_env_name;
            
            // now install package managers in the newly created conda environment
            cli_ui::display_progress("4/4", "Installing package managers in conda environment...");
            
            // install Poetry (if needed)
            if has_poetry {
                if let Ok(installed) = install_poetry_in_conda_env(&conda_env_name) {
                    if installed {
                        config.poetry_installed = true;
                        cli_ui::display_success(&format!("Poetry installed in conda environment: {}", conda_env_name));
                    } else {
                        cli_ui::display_warning("Failed to install Poetry in conda environment. You can install it manually later.");
                    }
                }
            }
            
            // install UV (if needed)
            if has_uv {
                if let Ok(installed) = install_uv_in_conda_env(&conda_env_name) {
                    if installed {
                        config.uv_installed = true;
                        cli_ui::display_success(&format!("UV installed in conda environment: {}", conda_env_name));
                    } else {
                        cli_ui::display_warning("Failed to install UV in conda environment. You can install it manually later.");
                    }
                }
            }
        }
    }

    // Final check and message to user
    if config.use_conda {
        cli_ui::display_message("\nTo use this project and its installed tools:");
        cli_ui::display_message(&format!("  conda activate {}", conda_env_name));

        if has_uv {
            if config.uv_installed {
                cli_ui::display_message("UV package manager is installed in your conda environment.");
            } else {
                cli_ui::display_message("To install UV in your conda environment:");
                cli_ui::display_message(&format!("  conda activate {} && pip install uv", conda_env_name));
            }
            cli_ui::display_message(&format!(
                "Example: conda activate {} && uv pip install numpy pandas",
                conda_env_name
            ));
        }

        if has_poetry {
            if config.poetry_installed {
                cli_ui::display_message("Poetry package manager is installed in your conda environment.");
            } else {
                cli_ui::display_message("To install Poetry in your conda environment:");
                cli_ui::display_message(&format!("  conda activate {} && pip install poetry", conda_env_name));
                cli_ui::display_message(&format!("  conda activate {} && poetry config virtualenvs.create false", conda_env_name));
            }
            cli_ui::display_message(&format!(
                "Example: conda activate {} && poetry add numpy pandas",
                conda_env_name
            ));
        }
    }

    Ok(())
}

/// Create README.md file
fn create_readme(
    project_dir: &Path, 
    _project_name: &str, 
    conda_env_name: &str, 
    config: &UserConfig
) -> Result<()> {
    // Get package manager information
    let package_managers_desc = config
        .package_managers
        .iter()
        .map(|pm| match pm {
            PackageManager::Conda { .. } => "Conda",
            PackageManager::Poetry { .. } => "Poetry",
            PackageManager::Uv { .. } => "uv",
        })
        .collect::<Vec<_>>()
        .join(" + ");

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
        - The environment name matches the project directory name by default\n\n\
        ## License\n\
        MIT\n",
        project_dir.display(),
        config.python_version,
        config.python_version,
        if config.use_conda {
            "Conda"
        } else {
            "System Python"
        },
        package_managers_desc,
        project_dir.display(),
        get_install_command(config),
        get_dev_install_command(config),
        get_test_command(config),
        conda_env_name
    );
    
    write_file(&Path::new("README.md"), &readme_content)?;
    
    Ok(())
}

/// Create .gitignore file
fn create_gitignore(project_dir: &Path) -> Result<()> {
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
    write_file(&project_dir.join(".gitignore"), gitignore_content)?;
    
    Ok(())
}

/// Create module files
fn create_module_files(project_dir: &Path, project_name: &str) -> Result<()> {
    // Create main module
    let src_dir = project_dir.join("src").join(project_name);
    fs::create_dir_all(&src_dir)?;

    // Create __init__.py
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
    
    Ok(())
}

/// Create test files
fn create_test_files(project_dir: &Path) -> Result<()> {
    // Create test files
    let tests_dir = project_dir.join("tests");
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
    
    Ok(())
}

/// Setup package managers and environment files
fn setup_package_managers(
    project_dir: &Path, 
    project_name: &str, 
    config: &UserConfig
) -> Result<()> {
    let has_conda = config
        .package_managers
        .iter()
        .any(|pm| matches!(pm, PackageManager::Conda { .. }));

    // Generate environment files for different package managers
    for package_manager in &config.package_managers {
        match package_manager {
            PackageManager::Conda {
                channels,
                environment_file,
                dev_environment_file,
            } => {
                // Make sure we're only processing this once (in case there are somehow multiple Conda entries)
                // use the generic conda environment setup function
                // only create files but not immediately create environment, create it later
                let env_content = conda_utils::generate_language_environment_yml(
                    project_name,
                    &Language::Python,
                    Some(&config.python_version),
                    None,
                    config.use_cuda,
                    Some(channels),
                );

                let dev_env_content = conda_utils::generate_dev_environment_yml(
                    project_name,
                    &Language::Python,
                    Some(&config.python_version),
                    None,
                    config.use_cuda,
                    Some(channels),
                );

                // write environment files
                write_file(&project_dir.join(environment_file), &env_content)?;
                write_file(&project_dir.join(dev_environment_file), &dev_env_content)?;

                debug!("Created conda environment files");
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
                write_file(&project_dir.join(pyproject_file), &pyproject_content)?;
            }
            PackageManager::Uv {
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
                write_file(&project_dir.join(requirements_file), &req_content)?;

                // Create requirements-dev.txt
                let dev_req_content = "\
                    # Development dependencies\n\
                    pytest>=7.0.0\n\
                    black>=23.0.0\n\
                    isort>=5.0.0\n\
                    flake8>=6.0.0\n";
                write_file(&project_dir.join(dev_requirements_file), dev_req_content)?;
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

    if has_conda {
        commands.push("# Create and activate Conda environment".to_string());
        for pm in &config.package_managers {
            if let PackageManager::Conda {
                environment_file, ..
            } = pm
            {
                commands.push(format!("conda env create -f {}", environment_file));
                commands.push("conda activate $(basename $PWD)".to_string());

                // 在conda环境激活后安装包管理器
                if has_poetry {
                    commands.push("\n# Install Poetry in Conda environment".to_string());
                    commands.push("pip install poetry".to_string());
                    commands.push("poetry config virtualenvs.create false".to_string());
                    
                    // 添加Poetry安装依赖的命令
                    commands.push("\n# Install project dependencies using Poetry".to_string());
                    commands.push("poetry install --no-interaction".to_string());
                }

                if has_uv {
                    commands.push("\n# Install UV in Conda environment".to_string());
                    commands.push("pip install uv".to_string());
                    
                    // 如果有设置pip镜像，则添加镜像配置
                    if let Some(index_url) = &config.pip_index_url {
                        commands.push(format!("\n# Configure UV to use specified mirror: {}", index_url));
                        commands.push(format!("uv pip config set global.index-url {}", index_url));
                    }
                    
                    // 添加UV安装依赖的命令
                    for package_manager in &config.package_managers {
                        if let PackageManager::Uv { requirements_file, .. } = package_manager {
                            commands.push("\n# Install dependencies using UV".to_string());
                            if let Some(index_url) = &config.pip_index_url {
                                commands.push(format!("uv pip install -r {} --index-url {}", requirements_file, index_url));
                            } else {
                                commands.push(format!("uv pip install -r {}", requirements_file));
                            }
                        }
                    }
                }
                
                // 已经处理了所有conda+其他包管理器的情况，跳出循环
                break;
            }
        }
        
        // 已经在conda环境内安装了包管理器，不需要再单独处理
        return commands.join("\n");
    } else {
        // 非 conda 环境情况下，我们不提供任何命令，因为我们只支持 conda 环境
        commands.push("# This project requires Conda".to_string());
        commands.push("# Please install Conda first: https://docs.conda.io/en/latest/miniconda.html".to_string());
        return commands.join("\n");
    }
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

                // 在conda环境激活后安装包管理器
                if has_poetry {
                    commands.push("\n# Install Poetry in Conda environment".to_string());
                    commands.push("pip install poetry".to_string());
                    commands.push("poetry config virtualenvs.create false".to_string());
                    
                    // 添加诗歌安装的命令
                    commands.push("\n# Install project development dependencies using Poetry".to_string());
                    commands.push("poetry install --no-interaction --with dev".to_string());
                }

                if has_uv {
                    commands.push("\n# Install UV in Conda environment".to_string());
                    commands.push("pip install uv".to_string());
                    
                    // 如果有设置pip镜像，则添加镜像配置
                    if let Some(index_url) = &config.pip_index_url {
                        commands.push(format!("\n# Configure UV to use specified mirror: {}", index_url));
                        commands.push(format!("uv pip config set global.index-url {}", index_url));
                    }
                    
                    // 添加UV安装开发依赖的命令
                    for package_manager in &config.package_managers {
                        if let PackageManager::Uv { dev_requirements_file, .. } = package_manager {
                            commands.push("\n# Install development dependencies using UV".to_string());
                            if let Some(index_url) = &config.pip_index_url {
                                commands.push(format!("uv pip install -r {} --index-url {}", dev_requirements_file, index_url));
                            } else {
                                commands.push(format!("uv pip install -r {}", dev_requirements_file));
                            }
                        }
                    }
                }
                
                // 已经处理了所有conda+其他包管理器的情况，跳出循环
                break;
            }
        }
        
        // 已经在conda环境内安装了包管理器，不需要再单独处理
        return commands.join("\n");
    } else {
        // 非 conda 环境情况下，我们不提供任何命令，因为我们只支持 conda 环境
        commands.push("# This project requires Conda".to_string());
        commands.push("# Please install Conda first: https://docs.conda.io/en/latest/miniconda.html".to_string());
        return commands.join("\n");
    }
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
            PackageManager::Uv { .. } => {
                commands.push("pytest".to_string());
            }
        }
    }

    commands.join(" && ")
}

/// Install Poetry in Conda environment
fn install_poetry_in_conda_env(env_name: &str) -> Result<bool> {
    info!("Installing Poetry in Conda environment...");

    // Get conda executable path
    let conda_path = match conda_utils::find_conda_executable() {
        Ok(path) => path,
        Err(_) => {
            cli_ui::display_warning("Cannot find conda executable to install Poetry");
            return Ok(false);
        }
    };

    // Use conda run to install Poetry in the Conda environment
    let pip_status = Command::new(&conda_path)
        .arg("run")
        .arg("-n")
        .arg(env_name)
        .arg("pip")
        .arg("install")
        .arg("poetry")
        .status()?;

    if pip_status.success() {
        // Configure Poetry to use the Conda environment instead of creating a new virtualenv
        let config_status = Command::new(&conda_path)
            .arg("run")
            .arg("-n")
            .arg(env_name)
            .arg("poetry")
            .arg("config")
            .arg("virtualenvs.create")
            .arg("false")
            .status()?;

        if config_status.success() {
            cli_ui::display_success(&format!(
                "Installed Poetry in Conda environment: {}",
                env_name
            ));
            debug!("Poetry configured to use Conda environment (no separate virtualenv)");
            return Ok(true);
        }
    }

    cli_ui::display_warning("Failed to install Poetry in Conda environment.");
    info!("You can install it manually later using the commands in README.md");
    Ok(false)
}

/// Install UV in Conda environment
fn install_uv_in_conda_env(env_name: &str) -> Result<bool> {
    info!("Installing UV in Conda environment...");

    // Get conda executable path
    let conda_path = match conda_utils::find_conda_executable() {
        Ok(path) => path,
        Err(_) => {
            cli_ui::display_warning("Cannot find conda executable to install UV");
            return Ok(false);
        }
    };

    // Use conda run to install UV in the Conda environment
    let pip_status = Command::new(&conda_path)
        .arg("run")
        .arg("-n")
        .arg(env_name)
        .arg("pip")
        .arg("install")
        .arg("uv")
        .status()?;

    if pip_status.success() {
        cli_ui::display_success(&format!("Installed UV in Conda environment: {}", env_name));
        debug!("UV is now available within your Conda environment.");
        return Ok(true);
    }

    cli_ui::display_error("Failed to install UV in Conda environment.");
    info!("You can install it manually later with:");
    info!("   conda activate {} && pip install uv", env_name);
    Ok(false)
}
