use crate::error::Result;
use clap::Parser;
use log::{info, warn};
use std::fs;
use std::io::{self, Write};
use std::path::Path;
use std::path::PathBuf;
use std::process::Command;

#[derive(Parser, Debug)]
pub struct NewCommand {
    /// Project name
    #[arg(short, long)]
    name: Option<String>,

    /// Project description
    #[arg(short, long)]
    description: Option<String>,

    /// Primary programming language (python, r, matlab)
    #[arg(short, long)]
    language: Option<String>,

    /// Project template to use
    #[arg(short, long)]
    template: Option<String>,

    /// Output directory
    #[arg(short, long, default_value = ".")]
    output: PathBuf,
}

#[derive(Debug, Clone)]
struct UserConfig {
    package_managers: Vec<PackageManager>,
    use_cuda: bool,
    cuda_version: Option<String>,
    cudnn_version: Option<String>,
    python_version: String,
    use_conda: bool,
    virtual_env_type: VirtualEnvType,
    pip_index_url: Option<String>,
    pip_trusted_hosts: Option<Vec<String>>,
}

#[derive(Debug, Clone, PartialEq)]
enum VirtualEnvType {
    Venv,
    Virtualenv,
    Conda,
    None,
}

#[derive(Debug, Clone)]
enum CondaDistribution {
    Miniconda,
    Anaconda,
    CondaForge,
}

#[derive(Clone, Debug)]
enum PackageManager {
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

impl NewCommand {
    pub async fn execute(&self) -> Result<()> {
        info!("🚀 Creating new CRESP project...");

        // Interactive prompts
        let name = self.name.clone().unwrap_or_else(|| {
            print!("📝 Project name: ");
            io::stdout().flush().unwrap();
            let mut input = String::new();
            io::stdin().read_line(&mut input).unwrap();
            input.trim().to_string()
        });

        let description = self.description.clone().unwrap_or_else(|| {
            print!("📄 Project description: ");
            io::stdout().flush().unwrap();
            let mut input = String::new();
            io::stdin().read_line(&mut input).unwrap();
            input.trim().to_string()
        });

        let language = self.language.clone().unwrap_or_else(|| {
            println!("🔧 Select primary programming language:");
            println!("1. Python");
            println!("2. R");
            println!("3. MATLAB");
            print!("Choice (1-3): ");
            io::stdout().flush().unwrap();
            let mut input = String::new();
            io::stdin().read_line(&mut input).unwrap();
            match input.trim() {
                "1" => "python",
                "2" => "r",
                "3" => "matlab",
                _ => "python",
            }
            .to_string()
        });

        // Select project template
        let template = self.template.clone().unwrap_or_else(|| {
            println!("📁 Select project template:");
            println!("1. Basic (flat structure for simple experiments)");
            println!("2. Data Analysis (for data processing and analysis)");
            println!("3. Machine Learning (for ML/DL experiments)");
            println!("4. Scientific Computing (for numerical simulations)");
            println!("5. Custom (select your own structure)");
            print!("Choice (1-5) [1]: ");
            io::stdout().flush().unwrap();
            let mut input = String::new();
            io::stdin().read_line(&mut input).unwrap();
            input.trim().to_string()
        });

        // Get user configuration for Python projects
        let user_config = if language == "python" {
            self.get_python_config()?
        } else {
            UserConfig {
                package_managers: Vec::new(),
                use_cuda: false,
                cuda_version: None,
                cudnn_version: None,
                python_version: "3.9".to_string(),
                use_conda: false,
                virtual_env_type: VirtualEnvType::None,
                pip_index_url: None,
                pip_trusted_hosts: None,
            }
        };

        // Create project directory
        let project_dir = self.output.join(&name);
        if project_dir.exists() {
            warn!(
                "⚠️ Project directory already exists: {}",
                project_dir.display()
            );
            print!("Do you want to overwrite it? (y/N): ");
            io::stdout().flush().unwrap();
            let mut input = String::new();
            io::stdin().read_line(&mut input).unwrap();
            if input.trim().to_lowercase() != "y" {
                return Ok(());
            }
            std::fs::remove_dir_all(&project_dir)?;
        }

        // Create basic project structure
        std::fs::create_dir_all(&project_dir)?;

        // Get system information
        let system_info = self.get_system_info()?;

        // Create cresp.toml with detailed environment configuration
        let cresp_toml = format!(
            r#"# CRESP Protocol Configuration
# Documentation: https://cresp.resciencelab.ai

cresp_version = "1.0"

[experiment]
name = "{}"
description = "{}"
keywords = []
authors = [
    {{ name = "Your Name", email = "your.email@example.com", affiliation = "Your Institution", role = "Researcher" }}
]

###############################################################################
# Original Research Environment
###############################################################################

[experiment.environment]
description = "The original environment where the research was conducted"

[experiment.environment.hardware]
cpu = {{ model = "{}", architecture = "{}", cores = {}, threads = {}, frequency = "{}" }}
memory = {{ size = "{}", type = "{}" }}
gpu = {{ default_model = {{ model = "{}", memory = "{}", compute_capability = "{}" }}, driver_version = "{}" }}
storage = {{ type = "{}" }}
network = {{ type = "{}", bandwidth = "{}" }}

[experiment.environment.system]
os = {{ name = "{}", version = "{}", kernel = "{}", architecture = "{}", locale = "{}", timezone = "{}" }}
packages = [
    {}
]

[experiment.environment.system.limits]
max_open_files = {}
max_processes = {}
stack_size = "{}"
virtual_memory = "{}"

[experiment.environment.software]
{}

[experiment.environment.variables]
system = {{
    LANG = "{}",
    LC_ALL = "{}",
    TZ = "{}"
}}
{}.path = ["{}"]
{}
experiment = {{
    EXPERIMENT_DATA_DIR = "{}",
    EXPERIMENT_OUTPUT_DIR = "{}"
}}

[experiment.environment.dependencies]
type = "{}"
package_manager = {{ type = "{}", config_file = "{}", lock_file = "{}" }}
{}
{}
{}"#,
            name,
            description,
            system_info.cpu.model,
            system_info.cpu.architecture,
            system_info.cpu.cores,
            system_info.cpu.threads,
            system_info.cpu.frequency,
            system_info.memory.size,
            system_info.memory.memory_type,
            system_info.gpu.model,
            system_info.gpu.memory,
            system_info.gpu.compute_capability,
            system_info.gpu.driver_version,
            system_info.storage.storage_type,
            system_info.network.network_type,
            system_info.network.bandwidth,
            system_info.os.name,
            system_info.os.version,
            system_info.os.kernel,
            system_info.os.architecture,
            system_info.os.locale,
            system_info.os.timezone,
            system_info.packages.join(",\n    "),
            system_info.limits.max_open_files,
            system_info.limits.max_processes,
            system_info.limits.stack_size,
            system_info.limits.virtual_memory,
            if language == "python" {
                let mut software_config = String::new();
                // Python 配置
                software_config.push_str(&format!("python = {{\n    version = \"{}\", \n    interpreter = \"python{}\",\n    compile_flags = \"--enable-shared --enable-optimizations\",\n    pip_config = {{\n        index_url = \"{}\",\n        trusted_hosts = [\"{}\"]\n    }},\n    virtual_env = {{\n        type = \"{}\",\n        path = \".venv\",\n        activation_script = \".venv/bin/activate\"\n    }}\n}}\n",
                    user_config.python_version,
                    user_config.python_version,
                    user_config.pip_index_url.as_ref().unwrap_or(&"https://pypi.org/simple".to_string()),
                    user_config.pip_trusted_hosts.as_ref().and_then(|hosts| hosts.first()).unwrap_or(&"pypi.org".to_string()),
                    match user_config.virtual_env_type {
                        VirtualEnvType::Venv => "venv",
                        VirtualEnvType::Virtualenv => "virtualenv",
                        VirtualEnvType::Conda => "conda",
                        VirtualEnvType::None => "none"
                    }
                ));

                // Conda 配置 (如果使用)
                if user_config.virtual_env_type == VirtualEnvType::Conda {
                    // 修复临时值问题：使用let绑定创建持久值
                    let default_conda_version = "4.10.3".to_string();
                    let conda_version = system_info
                        .software
                        .get("conda")
                        .unwrap_or(&default_conda_version);

                    // 获取所有conda渠道
                    let channels = user_config
                        .package_managers
                        .iter()
                        .filter_map(|pm| {
                            if let PackageManager::Conda { channels, .. } = pm {
                                Some(channels.clone())
                            } else {
                                None
                            }
                        })
                        .next()
                        .unwrap_or_else(|| vec!["conda-forge".to_string()]);

                    // 格式化渠道字符串
                    let channels_str = channels
                        .iter()
                        .map(|ch| format!("\"{}\"", ch))
                        .collect::<Vec<_>>()
                        .join(", ");

                    software_config.push_str(&format!(
                        "conda = {{ version = \"{}\", channels = [{}] }}\n",
                        conda_version, channels_str
                    ));
                }

                // CUDA 配置 (如果使用)
                if user_config.use_cuda {
                    // 修复临时值问题：使用let绑定创建持久值
                    let default_cuda_version = "11.8".to_string();
                    let cuda_version = user_config
                        .cuda_version
                        .as_ref()
                        .unwrap_or(&default_cuda_version);

                    software_config.push_str(&format!(
                        "cuda = {{ version = \"{}\", toolkit = \"cuda_{}_linux\" }}\n",
                        cuda_version,
                        cuda_version.replace(".", "_")
                    ));

                    // cuDNN 配置
                    // 修复临时值问题：使用let绑定创建持久值
                    let default_cudnn_version = "8.9".to_string();
                    let cudnn_version = user_config
                        .cudnn_version
                        .as_ref()
                        .unwrap_or(&default_cudnn_version);

                    software_config.push_str(&format!(
                        "cudnn = {{ version = \"{}\", toolkit = \"cudnn-{}-linux-x64-v{}\" }}\n",
                        cudnn_version, cuda_version, cudnn_version
                    ));
                }

                // 移除容器平台配置
                software_config
            } else {
                format!(
                    "{} = {{ version = \"{}\" }}",
                    language,
                    system_info
                        .software
                        .get(&language)
                        .unwrap_or(&"latest".to_string())
                )
            },
            system_info.os.locale,
            system_info.os.locale,
            system_info.os.timezone,
            language,
            project_dir.display(),
            if user_config.use_cuda {
                format!("cuda = {{\n    version = \"{}\",\n    CUDA_HOME = \"{}\",\n    LD_LIBRARY_PATH = [\n        \"{}\",\n        \"{}\"\n    ]\n}}",
                    user_config.cuda_version.as_ref().unwrap_or(&"11.8".to_string()),
                    system_info.cuda.cuda_home,
                    system_info.cuda.ld_library_path.join("\",\n        \""),
                    system_info.cuda.cupti_path)
            } else {
                String::new()
            },
            project_dir.join("data").display(),
            project_dir.join("output").display(),
            language,
            match language.as_str() {
                "python" => "poetry",
                "r" => "renv",
                "matlab" => "none",
                _ => "none",
            },
            match language.as_str() {
                "python" => "pyproject.toml",
                "r" => "DESCRIPTION",
                "matlab" => "none",
                _ => "none",
            },
            match language.as_str() {
                "python" => "poetry.lock",
                "r" => "renv.lock",
                "matlab" => "none",
                _ => "none",
            },
            if user_config.use_cuda {
                "conda_fallback = { enabled = true, environment_file = \"environment.yml\", dev_environment_file = \"environment-dev.yml\" }".to_string()
            } else {
                String::new()
            },
            if !user_config.use_cuda {
                "pip_fallback = { enabled = true, requirements_file = \"requirements.txt\", dev_requirements_file = \"requirements-dev.txt\" }".to_string()
            } else {
                String::new()
            },
            if !user_config.use_cuda {
                "uv_fallback = { enabled = true, requirements_file = \"requirements.txt\", dev_requirements_file = \"requirements-dev.txt\" }".to_string()
            } else {
                String::new()
            }
        );

        std::fs::write(project_dir.join("cresp.toml"), cresp_toml)?;

        // Create project structure based on template
        match template.as_str() {
            "1" => self.create_basic_structure(&project_dir, &language)?,
            "2" => self.create_data_analysis_structure(&project_dir, &language)?,
            "3" => self.create_ml_structure(&project_dir, &language)?,
            "4" => self.create_scientific_structure(&project_dir, &language)?,
            "5" => self.create_custom_structure(&project_dir, &language)?,
            _ => self.create_basic_structure(&project_dir, &language)?,
        }

        // Create language-specific project files
        match language.as_str() {
            "python" => self.create_python_project(&project_dir, &user_config)?,
            "r" => self.create_r_project(&project_dir)?,
            "matlab" => self.create_matlab_project(&project_dir)?,
            _ => unreachable!(),
        }

        info!(
            "✨ Project created successfully at: {}",
            project_dir.display()
        );
        Ok(())
    }

    fn create_basic_structure(&self, project_dir: &Path, _language: &str) -> Result<()> {
        // Basic flat structure for simple experiments
        let dirs = vec![
            "data",      // Raw and processed data
            "output",    // Experiment outputs
            "notebooks", // Jupyter notebooks
            "scripts",   // Utility scripts
            "config",    // Configuration files
        ];

        for dir in dirs {
            std::fs::create_dir_all(project_dir.join(dir))?;
        }

        // Create README with basic structure explanation
        let readme = r#"# Project Structure

```
.
├── data/           # Raw and processed data
├── output/         # Experiment outputs
├── notebooks/      # Jupyter notebooks
├── scripts/        # Utility scripts
└── config/         # Configuration files
```

## Directory Structure

- `data/`: Store your raw data and processed data
- `output/`: Store experiment results and outputs
- `notebooks/`: Jupyter notebooks for interactive analysis
- `scripts/`: Utility scripts and helper functions
- `config/`: Configuration files and parameters

## Usage

1. Place your raw data in the `data/` directory
2. Use notebooks in `notebooks/` for interactive analysis
3. Save experiment outputs to `output/`
4. Store configuration in `config/`
"#
        .to_string();
        std::fs::write(project_dir.join("README.md"), readme)?;

        Ok(())
    }

    fn create_data_analysis_structure(&self, project_dir: &Path, _language: &str) -> Result<()> {
        // Structure for data analysis projects
        let dirs = vec![
            "data/raw",              // Raw data
            "data/processed",        // Processed data
            "data/external",         // External data sources
            "output/figures",        // Generated figures
            "output/tables",         // Generated tables
            "output/reports",        // Generated reports
            "notebooks/analysis",    // Analysis notebooks
            "notebooks/exploration", // Data exploration notebooks
            "scripts/data",          // Data processing scripts
            "scripts/analysis",      // Analysis scripts
            "scripts/visualization", // Visualization scripts
            "config",                // Configuration files
        ];

        for dir in dirs {
            std::fs::create_dir_all(project_dir.join(dir))?;
        }

        // Create README with data analysis structure explanation
        let readme = r#"# Data Analysis Project Structure

```
.
├── data/
│   ├── raw/          # Original, immutable data
│   ├── processed/    # Cleaned and processed data
│   └── external/     # External data sources
├── output/
│   ├── figures/      # Generated figures
│   ├── tables/       # Generated tables
│   └── reports/      # Generated reports
├── notebooks/
│   ├── analysis/     # Analysis notebooks
│   └── exploration/  # Data exploration notebooks
├── scripts/
│   ├── data/         # Data processing scripts
│   ├── analysis/     # Analysis scripts
│   └── visualization/# Visualization scripts
└── config/           # Configuration files
```

## Directory Structure

- `data/raw/`: Original, immutable data
- `data/processed/`: Cleaned and processed data
- `data/external/`: External data sources
- `output/figures/`: Generated figures and plots
- `output/tables/`: Generated tables and statistics
- `output/reports/`: Generated reports and documentation
- `notebooks/analysis/`: Analysis notebooks
- `notebooks/exploration/`: Data exploration notebooks
- `scripts/data/`: Data processing scripts
- `scripts/analysis/`: Analysis and statistical scripts
- `scripts/visualization/`: Visualization and plotting scripts
- `config/`: Configuration files and parameters

## Usage

1. Place raw data in `data/raw/`
2. Use `notebooks/exploration/` for initial data exploration
3. Use `notebooks/analysis/` for detailed analysis
4. Save processed data to `data/processed/`
5. Generate outputs in respective `output/` subdirectories
"#
        .to_string();
        std::fs::write(project_dir.join("README.md"), readme)?;

        Ok(())
    }

    fn create_ml_structure(&self, project_dir: &Path, _language: &str) -> Result<()> {
        // Structure for machine learning projects
        let dirs = vec![
            "data/raw",              // Raw data
            "data/processed",        // Processed data
            "data/external",         // External data sources
            "models",                // Trained models
            "output/predictions",    // Model predictions
            "output/figures",        // Generated figures
            "output/metrics",        // Model metrics
            "notebooks/exploration", // Data exploration
            "notebooks/experiments", // Experiment notebooks
            "scripts/data",          // Data processing scripts
            "scripts/models",        // Model scripts
            "scripts/training",      // Training scripts
            "scripts/evaluation",    // Evaluation scripts
            "config",                // Configuration files
        ];

        for dir in dirs {
            std::fs::create_dir_all(project_dir.join(dir))?;
        }

        // Create README with ML structure explanation
        let readme = r#"# Machine Learning Project Structure

```
.
├── data/
│   ├── raw/          # Original, immutable data
│   ├── processed/    # Cleaned and processed data
│   └── external/     # External data sources
├── models/           # Trained models
├── output/
│   ├── predictions/  # Model predictions
│   ├── figures/      # Generated figures
│   └── metrics/      # Model metrics
├── notebooks/
│   ├── exploration/  # Data exploration
│   └── experiments/  # Experiment notebooks
├── scripts/
│   ├── data/         # Data processing scripts
│   ├── models/       # Model scripts
│   ├── training/     # Training scripts
│   └── evaluation/   # Evaluation scripts
└── config/           # Configuration files
```

## Directory Structure

- `data/raw/`: Original, immutable data
- `data/processed/`: Cleaned and processed data
- `data/external/`: External data sources
- `models/`: Trained models and checkpoints
- `output/predictions/`: Model predictions
- `output/figures/`: Generated figures and plots
- `output/metrics/`: Model metrics and results
- `notebooks/exploration/`: Data exploration notebooks
- `notebooks/experiments/`: Experiment notebooks
- `scripts/data/`: Data processing scripts
- `scripts/models/`: Model architecture scripts
- `scripts/training/`: Training scripts
- `scripts/evaluation/`: Evaluation scripts
- `config/`: Configuration files and parameters

## Usage

1. Place raw data in `data/raw/`
2. Use `notebooks/exploration/` for data exploration
3. Use `notebooks/experiments/` for model experiments
4. Save trained models in `models/`
5. Generate predictions and metrics in respective `output/` subdirectories
"#
        .to_string();
        std::fs::write(project_dir.join("README.md"), readme)?;

        Ok(())
    }

    fn create_scientific_structure(&self, project_dir: &Path, _language: &str) -> Result<()> {
        // Structure for scientific computing projects
        let dirs = vec![
            "data/raw",                // Raw data
            "data/processed",          // Processed data
            "data/external",           // External data sources
            "output/results",          // Simulation results
            "output/figures",          // Generated figures
            "output/tables",           // Generated tables
            "notebooks/analysis",      // Analysis notebooks
            "notebooks/visualization", // Visualization notebooks
            "scripts/simulation",      // Simulation scripts
            "scripts/analysis",        // Analysis scripts
            "scripts/visualization",   // Visualization scripts
            "config",                  // Configuration files
        ];

        for dir in dirs {
            std::fs::create_dir_all(project_dir.join(dir))?;
        }

        // Create README with scientific computing structure explanation
        let readme = r#"# Scientific Computing Project Structure

```
.
├── data/
│   ├── raw/          # Original, immutable data
│   ├── processed/    # Cleaned and processed data
│   └── external/     # External data sources
├── output/
│   ├── results/      # Simulation results
│   ├── figures/      # Generated figures
│   └── tables/       # Generated tables
├── notebooks/
│   ├── analysis/     # Analysis notebooks
│   └── visualization/# Visualization notebooks
├── scripts/
│   ├── simulation/   # Simulation scripts
│   ├── analysis/     # Analysis scripts
│   └── visualization/# Visualization scripts
└── config/           # Configuration files
```

## Directory Structure

- `data/raw/`: Original, immutable data
- `data/processed/`: Cleaned and processed data
- `data/external/`: External data sources
- `output/results/`: Simulation results and outputs
- `output/figures/`: Generated figures and plots
- `output/tables/`: Generated tables and statistics
- `notebooks/analysis/`: Analysis notebooks
- `notebooks/visualization/`: Visualization notebooks
- `scripts/simulation/`: Simulation and computation scripts
- `scripts/analysis/`: Analysis and processing scripts
- `scripts/visualization/`: Visualization scripts
- `config/`: Configuration files and parameters

## Usage

1. Place raw data in `data/raw/`
2. Use `notebooks/analysis/` for data analysis
3. Use `notebooks/visualization/` for result visualization
4. Run simulations using scripts in `scripts/simulation/`
5. Save results and outputs in respective `output/` subdirectories
"#
        .to_string();
        std::fs::write(project_dir.join("README.md"), readme)?;

        Ok(())
    }

    fn create_custom_structure(&self, project_dir: &Path, _language: &str) -> Result<()> {
        println!("🔧 Custom project structure setup:");
        println!("1. Create basic directories");
        println!("2. Create detailed structure");
        print!("Choice (1-2) [1]: ");
        io::stdout().flush()?;
        let mut input = String::new();
        io::stdin().read_line(&mut input)?;

        match input.trim() {
            "2" => {
                println!("📁 Enter directory names (comma-separated):");
                print!("> ");
                io::stdout().flush()?;
                let mut dirs_input = String::new();
                io::stdin().read_line(&mut dirs_input)?;

                let dirs: Vec<&str> = dirs_input.trim().split(',').map(|s| s.trim()).collect();
                let dirs_clone = dirs.clone();

                // Create directories
                for dir in &dirs {
                    std::fs::create_dir_all(project_dir.join(dir))?;
                }

                // Create README with custom structure
                let mut readme = String::from("# Custom Project Structure\n\n");
                readme.push_str("```\n");
                readme.push_str(".\n");
                for dir in dirs {
                    readme.push_str(&format!("└── {}/\n", dir));
                }
                readme.push_str("```\n\n");

                readme.push_str("## Directory Structure\n\n");
                for dir in dirs_clone {
                    readme.push_str(&format!("- `{}/`: [Add description]\n", dir));
                }

                readme.push_str("\n## Usage\n\n");
                readme.push_str("1. [Add usage instructions]\n");
                readme.push_str("2. [Add usage instructions]\n");
                readme.push_str("3. [Add usage instructions]\n");

                std::fs::write(project_dir.join("README.md"), readme)?;
            }
            _ => {
                // Create basic directories
                let dirs = vec![
                    "data",      // Data directory
                    "output",    // Output directory
                    "notebooks", // Notebooks directory
                    "scripts",   // Scripts directory
                    "config",    // Configuration directory
                ];

                for dir in dirs {
                    std::fs::create_dir_all(project_dir.join(dir))?;
                }

                // Create README with basic structure
                let readme = r#"# Custom Project Structure

```
.
├── data/           # Data directory
├── output/         # Output directory
├── notebooks/      # Notebooks directory
├── scripts/        # Scripts directory
└── config/         # Configuration directory
```

## Directory Structure

- `data/`: Store your data
- `output/`: Store experiment outputs
- `notebooks/`: Store notebooks
- `scripts/`: Store scripts
- `config/`: Store configuration files

## Usage

1. [Add usage instructions]
2. [Add usage instructions]
3. [Add usage instructions]
"#;
                std::fs::write(project_dir.join("README.md"), readme)?;
            }
        }

        Ok(())
    }

    fn get_python_config(&self) -> Result<UserConfig> {
        let mut config = UserConfig {
            package_managers: Vec::new(),
            use_cuda: false,
            cuda_version: None,
            cudnn_version: None,
            python_version: "3.9".to_string(),
            use_conda: false,
            virtual_env_type: VirtualEnvType::None,
            pip_index_url: None,
            pip_trusted_hosts: None,
        };

        // 检查系统 Python 可用性和 Conda 可用性
        let system_python = self.check_system_python()?;
        let conda_available = self.check_conda_available()?;

        // 1. 选择 Python 版本
        println!("\n📦 Python version selection:");
        println!("1. Python 3.12 (latest, recommended)");
        println!("2. Python 3.11 (stable)");
        println!("3. Python 3.10 (stable)");
        println!("4. Python 3.9 (stable)");
        println!("5. Custom version (e.g., 3.13, 3.14)");
        if let Some(ver) = &system_python {
            println!("6. Use system Python (version {})", ver);
        }
        print!(
            "Choice (1-{}) [1]: ",
            if system_python.is_some() { "6" } else { "5" }
        );
        io::stdout().flush()?;
        let mut input = String::new();
        io::stdin().read_line(&mut input)?;

        // 设置选择的 Python 版本
        config.python_version = match input.trim() {
            "2" => "3.11".to_string(),
            "3" => "3.10".to_string(),
            "4" => "3.9".to_string(),
            "5" => {
                print!("Enter Python version (e.g., 3.13): ");
                io::stdout().flush()?;
                let mut custom_version = String::new();
                io::stdin().read_line(&mut custom_version)?;
                let version = custom_version.trim();
                if version.matches('.').count() == 1
                    && version.split('.').all(|n| n.parse::<u32>().is_ok())
                {
                    version.to_string()
                } else {
                    warn!("⚠️ Invalid version format, using default Python 3.12");
                    "3.12".to_string()
                }
            }
            "6" if system_python.is_some() => system_python.clone().unwrap(),
            _ => "3.12".to_string(),
        };

        // 2. 确定环境管理方式
        println!("\n🔧 Environment management:");

        // 检查所选Python版本是否可用于系统
        let system_has_selected_version = if let Some(sys_version) = &system_python {
            sys_version.starts_with(&config.python_version)
        } else {
            false
        };

        if system_has_selected_version {
            // 系统有所选版本，提供所有选项
            if conda_available {
                println!("1. Use Conda (recommended for scientific computing)");
                println!("2. Use virtual environment");
            } else {
                println!("1. Install and use Conda (recommended for scientific computing)");
                println!("2. Use virtual environment");
            }
            print!("Choice (1-2) [1]: ");
        } else {
            // 系统没有所选版本，只提供Conda选项
            println!(
                "1. Use Conda to install Python {} (recommended)",
                config.python_version
            );
            if !conda_available {
                println!("   Note: Conda will be installed first");
            }
            println!("2. Cancel and select a different Python version");
            print!("Choice (1-2) [1]: ");
        }

        let choice_range = if input.trim() == "6" && system_python.is_some() {
            "1-3"
        } else {
            "1-2"
        };
        print!("Choice ({}) [1]: ", choice_range);
        io::stdout().flush()?;
        input.clear();
        io::stdin().read_line(&mut input)?;

        if input.trim() == "3" && system_python.is_some() {
            // 直接使用系统Python，不使用虚拟环境
            config.use_conda = false;
            config.virtual_env_type = VirtualEnvType::None;
        } else {
            config.use_conda = input.trim() != "2";

            if config.use_conda {
                config.virtual_env_type = VirtualEnvType::Conda;

                // 如果 Conda 未安装，提示安装
                if !conda_available {
                    println!("\n🔧 Conda is not installed. Would you like to install it now?");
                    println!("1. Install Miniconda (minimal installation)");
                    println!("2. Install Anaconda (full installation)");
                    println!("3. Install Conda-forge");
                    print!("Choice (1-3) [1]: ");
                    io::stdout().flush()?;
                    input.clear();
                    io::stdin().read_line(&mut input)?;

                    let distribution = match input.trim() {
                        "2" => CondaDistribution::Anaconda,
                        "3" => CondaDistribution::CondaForge,
                        _ => CondaDistribution::Miniconda,
                    };
                    self.install_conda(distribution)?;
                }

                // 询问Conda渠道选择
                println!("\n📦 Conda channel selection:");
                println!("You can select multiple channels. Conda will search packages in the order specified.");
                println!("1. conda-forge (recommended general channel)");
                println!("2. defaults (Anaconda default channel)");
                println!("3. bioconda (for bioinformatics packages)");
                println!("4. pytorch (for PyTorch and related packages)");
                println!("5. nvidia (for CUDA and GPU acceleration)");
                println!("6. r (for R programming language packages)");
                println!("7. Add custom channel");
                println!("\nEnter channel numbers separated by commas (e.g., '1,3,5'), or 'done' when finished");

                let mut selected_channels: Vec<String> = Vec::new();
                loop {
                    print!("> ");
                    io::stdout().flush()?;
                    input.clear();
                    io::stdin().read_line(&mut input)?;

                    let trimmed = input.trim();
                    if trimmed.to_lowercase() == "done" || trimmed.is_empty() {
                        // 如果没有选择任何渠道，默认添加conda-forge
                        if selected_channels.is_empty() {
                            selected_channels.push("conda-forge".to_string());
                            println!("No channels selected, using conda-forge as default.");
                        }
                        break;
                    }

                    // 解析选择的渠道号
                    for choice in trimmed.split(',') {
                        let choice = choice.trim();
                        match choice {
                            "1" => {
                                if !selected_channels.contains(&"conda-forge".to_string()) {
                                    selected_channels.push("conda-forge".to_string());
                                    println!("Added conda-forge channel");
                                }
                            }
                            "2" => {
                                if !selected_channels.contains(&"defaults".to_string()) {
                                    selected_channels.push("defaults".to_string());
                                    println!("Added defaults channel");
                                }
                            }
                            "3" => {
                                if !selected_channels.contains(&"bioconda".to_string()) {
                                    selected_channels.push("bioconda".to_string());
                                    println!("Added bioconda channel");
                                }
                            }
                            "4" => {
                                if !selected_channels.contains(&"pytorch".to_string()) {
                                    selected_channels.push("pytorch".to_string());
                                    println!("Added pytorch channel");
                                }
                            }
                            "5" => {
                                if !selected_channels.contains(&"nvidia".to_string()) {
                                    selected_channels.push("nvidia".to_string());
                                    println!("Added nvidia channel");
                                }
                            }
                            "6" => {
                                if !selected_channels.contains(&"r".to_string()) {
                                    selected_channels.push("r".to_string());
                                    println!("Added r channel");
                                }
                            }
                            "7" => {
                                print!("Enter custom channel name: ");
                                io::stdout().flush()?;
                                let mut custom_input = String::new();
                                io::stdin().read_line(&mut custom_input)?;
                                let custom_channel = custom_input.trim().to_string();
                                if !custom_channel.is_empty()
                                    && !selected_channels.contains(&custom_channel)
                                {
                                    selected_channels.push(custom_channel.clone());
                                    println!("Added custom channel: {}", custom_channel);
                                }
                            }
                            _ => println!("Invalid choice: {}", choice),
                        }
                    }

                    println!(
                        "Currently selected channels: {}",
                        selected_channels.join(", ")
                    );
                    println!("Enter more channels or type 'done' to finish selection");
                }

                config.package_managers.push(PackageManager::Conda {
                    channels: selected_channels,
                    environment_file: "environment.yml".to_string(),
                    dev_environment_file: "dev-environment.yml".to_string(),
                });
            } else {
                // 选择虚拟环境类型
                println!("\n🔧 Virtual environment type:");
                println!("1. venv (recommended, built-in)");
                println!("2. virtualenv (third-party)");
                print!("Choice (1-2) [1]: ");
                io::stdout().flush()?;
                input.clear();
                io::stdin().read_line(&mut input)?;

                config.virtual_env_type = match input.trim() {
                    "2" => VirtualEnvType::Virtualenv,
                    _ => VirtualEnvType::Venv,
                };
            }
        }

        // 3. 询问包管理器选择 (不论是否使用conda)
        println!("\n📦 Python package management:");
        println!("1. Poetry (recommended for modern Python projects)");
        println!("2. uv (recommended for fast dependency resolution)");
        println!("3. pip (traditional, recommended for simple projects)");
        if config.use_conda {
            println!(
                "4. Only use conda for package management (no additional Python package manager)"
            );
        }
        print!(
            "Choice (1-{}) [1]: ",
            if config.use_conda { "4" } else { "3" }
        );
        io::stdout().flush()?;
        input.clear();
        io::stdin().read_line(&mut input)?;

        // 如果使用conda但选择了其他包管理器，保留conda且添加其他包管理器
        if !config.use_conda || input.trim() != "4" {
            match input.trim() {
                "2" => {
                    config.package_managers.push(PackageManager::Uv {
                        requirements_file: "requirements.txt".to_string(),
                        dev_requirements_file: "requirements-dev.txt".to_string(),
                    });
                }
                "3" => {
                    config.package_managers.push(PackageManager::Pip {
                        requirements_file: "requirements.txt".to_string(),
                        dev_requirements_file: "requirements-dev.txt".to_string(),
                    });
                }
                _ => {
                    config.package_managers.push(PackageManager::Poetry {
                        pyproject_file: "pyproject.toml".to_string(),
                    });
                }
            }
        }

        // 4. 选择 pip/uv/poetry 的包源 (如果选择了这些管理器)
        if !config
            .package_managers
            .iter()
            .all(|pm| matches!(pm, PackageManager::Conda { .. }))
        {
            println!("\n📦 Package Index Configuration:");
            println!("1. PyPI (default)");
            println!("2. Tsinghua Mirror (faster in China)");
            println!("3. Aliyun Mirror (faster in China)");
            println!("4. Custom index URL");
            print!("Choice (1-4) [1]: ");
            io::stdout().flush()?;
            input.clear();
            io::stdin().read_line(&mut input)?;

            match input.trim() {
                "2" => {
                    config.pip_index_url =
                        Some("https://pypi.tuna.tsinghua.edu.cn/simple".to_string());
                    config.pip_trusted_hosts = Some(vec!["pypi.tuna.tsinghua.edu.cn".to_string()]);
                }
                "3" => {
                    config.pip_index_url =
                        Some("https://mirrors.aliyun.com/pypi/simple".to_string());
                    config.pip_trusted_hosts = Some(vec!["mirrors.aliyun.com".to_string()]);
                }
                "4" => {
                    print!("Enter custom index URL: ");
                    io::stdout().flush()?;
                    input.clear();
                    io::stdin().read_line(&mut input)?;
                    let index_url = input.trim().to_string();

                    // 从URL中提取host部分
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

        // 5. 检查 CUDA 可用性
        config.use_cuda = self.check_cuda_availability()?;
        if config.use_cuda {
            config.cuda_version = Some("11.8".to_string());
            config.cudnn_version = Some("8.9".to_string());
        }

        // 6. 显示配置摘要
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
                VirtualEnvType::Venv => "venv",
                VirtualEnvType::Virtualenv => "virtualenv",
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

        // 7. 询问是否继续安装
        print!("\n🚀 Proceed with installation? (Y/n): ");
        io::stdout().flush()?;
        input.clear();
        io::stdin().read_line(&mut input)?;
        if input.trim().to_lowercase() == "n" {
            return Ok(config);
        }

        Ok(config)
    }

    fn check_system_python(&self) -> Result<Option<String>> {
        let output = Command::new("python3").arg("--version").output();

        match output {
            Ok(output) if output.status.success() => {
                let version = String::from_utf8_lossy(&output.stdout)
                    .split_whitespace()
                    .nth(1)
                    .map(|s| s.to_string());
                Ok(version)
            }
            _ => Ok(None),
        }
    }

    fn check_conda_available(&self) -> Result<bool> {
        let output = Command::new("conda").arg("--version").output();

        Ok(output.is_ok() && output.unwrap().status.success())
    }

    fn install_conda(&self, distribution: CondaDistribution) -> Result<()> {
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

        // 下载安装脚本
        Command::new("curl")
            .arg("-O")
            .arg(format!("{}{}", base_url, install_script))
            .status()?;

        // 运行安装脚本
        Command::new("bash")
            .arg(install_script)
            .arg("-b")
            .arg("-p")
            .arg(format!("{}/.conda", std::env::var("HOME").unwrap()))
            .status()?;

        // 删除安装脚本
        std::fs::remove_file(install_script)?;

        // 初始化 conda
        Command::new("conda").arg("init").status()?;

        // 如果是 conda-forge，设置默认 channel
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

    fn check_cuda_availability(&self) -> Result<bool> {
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

    fn create_python_project(&self, project_dir: &PathBuf, config: &UserConfig) -> Result<()> {
        // 创建项目目录
        fs::create_dir_all(project_dir)?;
        std::env::set_current_dir(project_dir)?;

        // 创建基本项目结构
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

        // 创建 README.md
        // 获取包管理器信息
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

        let readme_content = format!(
            "# {}\n\n\
            ## 项目简介\n\
            这是一个使用 Python {} 的项目。\n\n\
            ## 环境要求\n\
            - Python {}\n\
            - 环境管理: {}\n\
            - 包管理工具: {}\n\n\
            ## 安装\n\
            ```bash\n\
            # 克隆项目\n\
            git clone <repository-url>\n\
            cd {}\n\n\
            # 安装依赖\n\
            {}\n\
            ```\n\n\
            ## 使用\n\
            待补充\n\n\
            ## 开发\n\
            ```bash\n\
            # 安装开发依赖\n\
            {}\n\
            ```\n\n\
            ## 测试\n\
            ```bash\n\
            # 运行测试\n\
            {}\n\
            ```\n\n\
            ## 许可证\n\
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
                    _ => "系统Python",
                }
            },
            package_managers_desc,
            project_dir.display(),
            self.get_install_command(config),
            self.get_dev_install_command(config),
            self.get_test_command(config)
        );
        fs::write("README.md", readme_content)?;

        // 创建 .gitignore
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
        fs::write(".gitignore", gitignore_content)?;

        // 创建主模块
        let src_dir = Path::new("src").join(
            project_dir
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("my-project"),
        );
        fs::create_dir_all(&src_dir)?;
        fs::write(
            src_dir.join("__init__.py"),
            format!(
                "\"\"\"{} package.\"\"\"\n\n\
                __version__ = \"0.1.0\"\n",
                project_dir
                    .file_name()
                    .and_then(|name| name.to_str())
                    .unwrap_or("my-project")
            ),
        )?;
        fs::write(
            src_dir.join("main.py"),
            "\"\"\"Main module.\"\"\"\n\n\
            def main():\n\
                \"\"\"Run the main function.\"\"\"\n\
                print(\"Hello, World!\")\n\n\
            if __name__ == \"__main__\":\n\
                main()\n",
        )?;

        // 创建测试文件
        let tests_dir = Path::new("tests");
        fs::write(tests_dir.join("__init__.py"), "\"\"\"Test package.\"\"\"\n")?;
        fs::write(
            tests_dir.join("test_main.py"),
            "\"\"\"Test main module.\"\"\"\n\n\
            def test_main():\n\
                \"\"\"Test main function.\"\"\"\n\
                assert True\n",
        )?;

        // 根据用户选择创建环境配置文件
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
                    // 创建 environment.yml
                    let project_name = project_dir
                        .file_name()
                        .and_then(|name| name.to_str())
                        .unwrap_or("my-project");
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

                    // 添加需要通过conda安装的包
                    if config.use_cuda {
                        env_content.push_str("  - numpy\n  - scipy\n  - matplotlib\n  - pandas\n");
                    }

                    // 如果有Poetry/pip/uv，添加pip部分
                    if config
                        .package_managers
                        .iter()
                        .any(|pm| !matches!(pm, PackageManager::Conda { .. }))
                    {
                        env_content.push_str("  - pip:\n");
                        env_content.push_str("    - -e .\n"); // 安装当前项目
                    }

                    fs::write(environment_file, env_content)?;

                    // 创建 dev-environment.yml
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
                        dev_env_content
                            .push_str(&format!("  - python={}\n", config.python_version));
                    }
                    dev_env_content.push_str(
                        "  - pip\n\
                         - pytest\n\
                         - black\n\
                         - isort\n\
                         - flake8\n",
                    );

                    // 如果有Poetry/pip/uv，添加pip部分
                    if config
                        .package_managers
                        .iter()
                        .any(|pm| !matches!(pm, PackageManager::Conda { .. }))
                    {
                        dev_env_content.push_str("  - pip:\n");
                        dev_env_content.push_str("    - -e .\n"); // 安装当前项目
                    }

                    fs::write(dev_environment_file, dev_env_content)?;
                }
                PackageManager::Poetry { pyproject_file } => {
                    // 创建 pyproject.toml
                    let project_name = project_dir
                        .file_name()
                        .and_then(|name| name.to_str())
                        .unwrap_or("my-project");
                    let pyproject_content = format!(
                        "[tool.poetry]\n\
                        name = \"{}\"\n\
                        version = \"0.1.0\"\n\
                        description = \"{}\"\n\
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
                        project_name, project_name, project_name, config.python_version
                    );
                    fs::write(pyproject_file, pyproject_content)?;

                    // 如果同时使用conda，创建.envrc文件用于自动切换环境
                    if has_conda {
                        let envrc_content = format!(
                            "# 自动激活conda环境\n\
                            if [ -e ${{HOME}}/.conda/etc/profile.d/conda.sh ]; then\n\
                            \tsource ${{HOME}}/.conda/etc/profile.d/conda.sh\n\
                            \tconda activate {}\n\
                            fi\n\n\
                            # 设置Poetry使用已激活的Python环境\n\
                            export POETRY_VIRTUALENVS_CREATE=false\n",
                            project_name
                        );
                        fs::write(".envrc", envrc_content)?;

                        // 创建poetry配置文件，禁用虚拟环境创建
                        fs::create_dir_all(".poetry")?;
                        fs::write(".poetry/config.toml", "virtualenvs.create = false\n")?;
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
                    // 创建 requirements.txt
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
                    fs::write(requirements_file, req_content)?;

                    // 创建 requirements-dev.txt
                    let dev_req_content = "\
                        # Development dependencies\n\
                        pytest>=7.0.0\n\
                        black>=23.0.0\n\
                        isort>=5.0.0\n\
                        flake8>=6.0.0\n";
                    fs::write(dev_requirements_file, dev_req_content)?;

                    // 如果同时使用conda和pip/uv，创建.envrc文件
                    if has_conda {
                        let project_name = project_dir
                            .file_name()
                            .and_then(|name| name.to_str())
                            .unwrap_or("my-project");

                        let envrc_content = format!(
                            "# 自动激活conda环境\n\
                            if [ -e ${{HOME}}/.conda/etc/profile.d/conda.sh ]; then\n\
                            \tsource ${{HOME}}/.conda/etc/profile.d/conda.sh\n\
                            \tconda activate {}\n\
                            fi\n",
                            project_name
                        );
                        fs::write(".envrc", envrc_content)?;
                    }
                }
            }
        }

        // 创建虚拟环境
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
                    let project_name = project_dir
                        .file_name()
                        .and_then(|name| name.to_str())
                        .unwrap_or("my-project");
                    Command::new("conda")
                        .arg("create")
                        .arg("-n")
                        .arg(project_name)
                        .arg(format!("python={}", config.python_version))
                        .arg("-y")
                        .status()?;
                }
            }
            VirtualEnvType::None => {}
        }

        Ok(())
    }

    fn get_install_command(&self, config: &UserConfig) -> String {
        let mut commands = Vec::new();

        // 如果有Conda环境，先激活环境
        let has_conda = config
            .package_managers
            .iter()
            .any(|pm| matches!(pm, PackageManager::Conda { .. }));
        if has_conda {
            commands.push("# 创建并激活Conda环境".to_string());
            for pm in &config.package_managers {
                if let PackageManager::Conda {
                    environment_file, ..
                } = pm
                {
                    commands.push(format!("conda env create -f {}", environment_file));

                    // 修复：使用多步获取项目名称，避免临时值问题
                    let canonical_path = Path::new(".").canonicalize().unwrap();
                    let file_name = canonical_path.file_name().unwrap();
                    let project_name = file_name.to_str().unwrap();
                    commands.push(format!("conda activate {}", project_name));
                }
            }
        }

        // 添加其他包管理器的安装命令
        for package_manager in &config.package_managers {
            match package_manager {
                PackageManager::Conda { .. } => {
                    // 已经处理过了
                }
                PackageManager::Poetry { .. } => {
                    commands.push("# 安装Poetry依赖".to_string());
                    if has_conda {
                        commands.push("poetry install --no-interaction".to_string());
                    } else {
                        commands.push("poetry install".to_string());
                    }
                }
                PackageManager::Pip {
                    requirements_file, ..
                } => {
                    commands.push("# 安装pip依赖".to_string());
                    commands.push(format!("pip install -r {}", requirements_file));
                }
                PackageManager::Uv {
                    requirements_file, ..
                } => {
                    commands.push("# 安装uv依赖".to_string());
                    commands.push(format!("uv pip install -r {}", requirements_file));
                }
            }
        }

        commands.join("\n")
    }

    fn get_dev_install_command(&self, config: &UserConfig) -> String {
        let mut commands = Vec::new();

        // 如果有Conda环境，先激活环境
        let has_conda = config
            .package_managers
            .iter()
            .any(|pm| matches!(pm, PackageManager::Conda { .. }));
        if has_conda {
            commands.push("# 创建并激活Conda开发环境".to_string());
            for pm in &config.package_managers {
                if let PackageManager::Conda {
                    dev_environment_file,
                    ..
                } = pm
                {
                    commands.push(format!("conda env create -f {}", dev_environment_file));

                    // 修复：使用多步获取项目名称，避免临时值问题
                    let canonical_path = Path::new(".").canonicalize().unwrap();
                    let file_name = canonical_path.file_name().unwrap();
                    let project_name = file_name.to_str().unwrap();
                    commands.push(format!("conda activate {}-dev", project_name));
                }
            }
        }

        // 添加其他包管理器的开发依赖安装命令
        for package_manager in &config.package_managers {
            match package_manager {
                PackageManager::Conda { .. } => {
                    // 已经处理过了
                }
                PackageManager::Poetry { .. } => {
                    commands.push("# 安装Poetry开发依赖".to_string());
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
                    commands.push("# 安装pip开发依赖".to_string());
                    commands.push(format!("pip install -r {}", dev_requirements_file));
                }
                PackageManager::Uv {
                    dev_requirements_file,
                    ..
                } => {
                    commands.push("# 安装uv开发依赖".to_string());
                    commands.push(format!("uv pip install -r {}", dev_requirements_file));
                }
            }
        }

        commands.join("\n")
    }

    fn get_test_command(&self, config: &UserConfig) -> String {
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

    fn create_r_project(&self, project_dir: &Path) -> Result<()> {
        // 检查系统R可用性
        let system_r = self.check_system_r()?;
        let r_version = if system_r.is_some() {
            // 如果系统已安装R语言，询问用户选择
            println!("\n📦 R version selection:");
            println!("1. R 4.3.x (latest)");
            println!("2. R 4.2.x (stable)");
            println!("3. R 4.1.x");
            println!("4. R 4.0.x");
            println!("5. Custom version");
            println!("6. Use system R (version {})", system_r.as_ref().unwrap());
            print!("Choice (1-6) [6]: ");
            io::stdout().flush()?;
            let mut input = String::new();
            io::stdin().read_line(&mut input)?;
            
            match input.trim() {
                "1" => "4.3".to_string(),
                "2" => "4.2".to_string(),
                "3" => "4.1".to_string(),
                "4" => "4.0".to_string(),
                "5" => {
                    print!("Enter R version (e.g., 4.2.2): ");
                    io::stdout().flush()?;
                    let mut custom_version = String::new();
                    io::stdin().read_line(&mut custom_version)?;
                    custom_version.trim().to_string()
                },
                _ => system_r.as_ref().unwrap().clone(),
            }
        } else {
            // 如果系统未安装R，提供版本选择
            println!("\n⚠️ R is not installed on this system or not found in PATH");
            println!("📦 Select R version to use:");
            println!("1. R 4.3.x (latest)");
            println!("2. R 4.2.x (stable)");
            println!("3. R 4.1.x");
            println!("4. R 4.0.x");
            println!("5. Custom version");
            print!("Choice (1-5) [1]: ");
            io::stdout().flush()?;
            let mut input = String::new();
            io::stdin().read_line(&mut input)?;
            
            match input.trim() {
                "2" => "4.2".to_string(),
                "3" => "4.1".to_string(),
                "4" => "4.0".to_string(),
                "5" => {
                    print!("Enter R version (e.g., 4.2.2): ");
                    io::stdout().flush()?;
                    let mut custom_version = String::new();
                    io::stdin().read_line(&mut custom_version)?;
                    custom_version.trim().to_string()
                },
                _ => "4.3".to_string(),
            }
        };
        
        // 检查版本匹配情况
        let is_version_mismatch = match &system_r {
            Some(sys_ver) => !sys_ver.starts_with(&r_version),
            None => true,
        };
        
        // 如果版本不匹配，提供解决方案
        if is_version_mismatch {
            println!("\n⚠️ Selected R version ({}) differs from system version or R is not installed", r_version);
            println!("The following solutions are available:");
            println!("1. Install R {} and use renv for package management", r_version);
            
            if cfg!(target_os = "macos") {
                println!("2. Use Homebrew to manage R versions (macOS)");
                println!("3. Use rig (R Installation Manager) to manage multiple R versions");
                println!("4. Continue with current setup (may cause compatibility issues)");
                print!("Choice (1-4) [1]: ");
            } else {
                println!("2. Use rig (R Installation Manager) to manage multiple R versions");
                println!("3. Continue with current setup (may cause compatibility issues)");
                print!("Choice (1-3) [1]: ");
            }
            
            io::stdout().flush()?;
            let mut input = String::new();
            io::stdin().read_line(&mut input)?;
            
            if cfg!(target_os = "macos") {
                match input.trim() {
                    "2" => {
                        // Homebrew选项 (仅macOS)
                        let major_minor = r_version.split('.').take(2).collect::<Vec<&str>>().join(".");
                        println!("\n📋 To manage R versions with Homebrew:");
                        println!("# Install R {} with Homebrew", major_minor);
                        println!("brew install r@{}", major_minor);
                        println!("\n# Switch between versions");
                        println!("brew unlink r");
                        println!("brew link r@{} --force", major_minor);
                        println!("\n# Set R_HOME environment variable");
                        println!("echo 'export R_HOME=$(brew --prefix r@{})' >> ~/.zshrc # or ~/.bashrc", major_minor);
                        println!("source ~/.zshrc # or ~/.bashrc");
                    },
                    "3" => {
                        // 提供rig安装说明
                        println!("\n📋 To install rig (R Installation Manager):");
                        println!("brew install rig");
                        println!("\n📋 Then install R {}:", r_version);
                        println!("rig add {}", r_version);
                        println!("rig default {}", r_version);
                    },
                    "4" => {
                        println!("⚠️ Continuing with current setup. Be aware of potential compatibility issues.");
                    },
                    _ => {
                        // 默认：创建.Rprofile并使用renv
                        println!("\n📋 To install R {}:", r_version);
                        println!("Visit: https://cran.r-project.org/bin/macosx/");
                        
                        // 将创建增强的renv配置
                        self.create_enhanced_renv_setup(project_dir, &r_version)?;
                        println!("✅ Created enhanced renv setup for version management");
                    }
                }
            } else {
                match input.trim() {
                    "2" => {
                        // 提供rig安装说明
                        println!("\n📋 To install rig (R Installation Manager):");
                        if cfg!(target_os = "linux") {
                            println!("curl -Ls https://github.com/r-lib/rig/releases/download/latest/rig-linux-latest.tar.gz | sudo tar xz -C /usr/local");
                        } else {
                            println!("Visit: https://github.com/r-lib/rig");
                        }
                        println!("\n📋 Then install R {}:", r_version);
                        println!("rig add {}", r_version);
                        println!("rig default {}", r_version);
                    },
                    "3" => {
                        println!("⚠️ Continuing with current setup. Be aware of potential compatibility issues.");
                    },
                    _ => {
                        // 默认：创建.Rprofile并使用renv
                        println!("\n📋 To install R {}:", r_version);
                        if cfg!(target_os = "linux") {
                            println!("Visit: https://cran.r-project.org/bin/linux/");
                        } else {
                            println!("Visit: https://cran.r-project.org/bin/windows/base/");
                        }
                        
                        // 将创建增强的renv配置
                        self.create_enhanced_renv_setup(project_dir, &r_version)?;
                        println!("✅ Created enhanced renv setup for version management");
                    }
                }
            }
        }

        // Create DESCRIPTION
        let description = format!(r#"Package: myresearch
Title: My Research Project
Version: 0.1.0
Authors@R: 
    person("Your", "Name", email = "your.email@example.com", role = c("aut", "cre"))
Description: A research project using CRESP protocol.
License: MIT + file LICENSE
Encoding: UTF-8
Roxygen: list(markdown = TRUE)
RoxygenNote: 7.2.3
Imports: 
    renv,
    testthat
Suggests:
    knitr,
    rmarkdown
RdMacros: lifecycle
Config/testthat/edition: 3
"#);

        std::fs::write(project_dir.join("DESCRIPTION"), description)?;

        // Create R directory and main.R
        std::fs::create_dir_all(project_dir.join("R"))?;
        std::fs::write(
            project_dir.join("R/main.R"),
            r#"#' Main function
#' 
#' This is the main entry point for the research project.
#' 
#' @return NULL
#' @export
#'
#' @examples
#' main()
main <- function() {
    print("Hello, CRESP!")
    
    # Your analysis code goes here
}

if (interactive()) {
    main()
}
"#,
        )?;

        // Create tests directory and testthat.R
        std::fs::create_dir_all(project_dir.join("tests/testthat"))?;
        std::fs::write(
            project_dir.join("tests/testthat.R"),
            r#"library(testthat)
library(myresearch)

test_check("myresearch")
"#,
        )?;

        // Create a basic test file
        std::fs::write(
            project_dir.join("tests/testthat/test-main.R"),
            r#"test_that("main function works", {
  # Setup test environment
  
  # Call the function (without executing side effects)
  # result <- main()
  
  # Verify results
  expect_true(TRUE)
})
"#,
        )?;

        // Create README.md with version information
        std::fs::write(
            project_dir.join("README.md"),
            format!(r#"# My Research Project

This is a research project using CRESP protocol.

## R Version

This project uses R {} for development.

## Project Structure

```
.
├── R/              # R code files
├── data/           # Data directory
├── output/         # Output directory
├── tests/          # Tests directory
├── DESCRIPTION     # Package metadata
└── renv.lock       # Package dependency lock file
```

## Setup

1. Install R {} if not already installed.

2. Install renv:
```r
install.packages("renv")
```

3. Initialize renv:
```r
renv::init()
```

4. Install dependencies:
```r
renv::restore()
```

5. Run the project:
```r
source("R/main.R")
```

## Testing

Run tests with:
```r
testthat::test_package("myresearch")
# or
devtools::test()
```
"#, r_version, r_version),
        )?;

        // Create .Rbuildignore
        std::fs::write(
            project_dir.join(".Rbuildignore"),
            r#"^.*\.Rproj$
^\.Rproj\.user$
^data/
^output/
^renv$
^renv\.lock$
^\.renvignore$
^\.gitignore$
^cresp\.toml$
"#,
        )?;

        // Create .gitignore
        std::fs::write(
            project_dir.join(".gitignore"),
            r#"# R specific
.Rproj.user
.Rhistory
.RData
.Ruserdata
*.Rproj

# renv specific
renv/library/
renv/local/
renv/lock/
renv/python/
renv/staging/

# Output files
output/
*.html
*.pdf
*.png
*.jpg

# Large data files
data/**/*.csv
data/**/*.xlsx
data/**/*.rds
"#,
        )?;

        // Create renv.lock template
        let renv_lock = format!(r#"{{
  "R": {{
    "Version": "{}",
    "Repositories": [
      {{
        "Name": "CRAN",
        "URL": "https://cloud.r-project.org"
      }}
    ]
  }},
  "Packages": {{
    "renv": {{
      "Package": "renv",
      "Version": "1.0.2",
      "Source": "Repository",
      "Repository": "CRAN"
    }},
    "testthat": {{
      "Package": "testthat",
      "Version": "3.1.10",
      "Source": "Repository",
      "Repository": "CRAN"
    }}
  }}
}}"#, r_version);
        std::fs::write(project_dir.join("renv.lock"), renv_lock)?;

        Ok(())
    }

    // 辅助函数：创建增强的renv设置
    fn create_enhanced_renv_setup(&self, project_dir: &Path, r_version: &str) -> Result<()> {
        // 创建.Rprofile文件，设置renv
        let rprofile_content = format!(r#"# .Rprofile for CRESP project
# Automatically detect R version mismatch and configure renv

# Store information about target R version
target_r_version <- "{}"

# Check R version
current_r_version <- paste0(R.version$major, ".", strsplit(R.version$minor, "\\.")[[1]][1])

message("Current R version: ", R.version$version.string)
message("Target R version: ", target_r_version)

if (current_r_version != target_r_version) {{
  warning(
    "\\nR version mismatch. Project targets R ", target_r_version,
    " but you're running R ", current_r_version, ".\n",
    "This may cause compatibility issues.\n",
    "Consider switching to R ", target_r_version, " or updating your renv.lock file.\n"
  )
}}

# Setup renv
if (interactive()) {{
  suppressMessages(require(renv, quietly = TRUE))
  if (requireNamespace("renv", quietly = TRUE)) {{
    renv::activate()
  }} else {{
    message("Installing renv...")
    install.packages("renv")
    renv::activate()
  }}
}}
"#, r_version);
        std::fs::write(project_dir.join(".Rprofile"), rprofile_content)?;
        
        // 创建详细的renv安装指南
        let renv_setup_content = format!(r#"# Setting up R Environment

## 1. R Version
This project targets R version {}. If you have a different version installed, consider:

- Installing R {} from [CRAN](https://cran.r-project.org/)
- Using [rig](https://github.com/r-lib/rig) to manage multiple R versions
{}
## 2. Package Management with renv

This project uses `renv` for package management to ensure reproducibility.

### 2.1 Initial Setup

When opening the project for the first time, R will automatically activate renv.
If it doesn't, run:

```r
install.packages("renv")
renv::activate()
renv::restore()
```

### 2.2 Installing/Adding New Packages

When adding new packages to the project, use:

```r
# Install and record in lockfile
renv::install("packagename")

# After adding packages, update the lockfile
renv::snapshot()
```

### 2.3 Updating Packages

To update packages:

```r
# Update specific packages
renv::update("packagename")

# Update all packages
renv::update()
```

## 3. Sharing the Project

When sharing the project, the recipient should simply run:

```r
# Restore the exact environment
renv::restore()
```

## 4. Troubleshooting

If you encounter package compatibility issues:

1. Check R version matches target ({})
2. Try reinstalling problematic packages: `renv::install("packagename", force = TRUE)`
3. Update renv itself: `install.packages("renv")`
4. Consult the [renv documentation](https://rstudio.github.io/renv/)
"#, r_version, r_version, 
        if cfg!(target_os = "macos") {
            "- Using Homebrew to manage R versions on macOS: `brew install r@x.y`\n"
        } else {
            ""
        }, r_version);
        std::fs::create_dir_all(project_dir.join("docs"))?;
        std::fs::write(project_dir.join("docs/renv-setup.md"), renv_setup_content)?;
        
        Ok(())
    }

    // 检测系统R安装
    fn check_system_r(&self) -> Result<Option<String>> {
        let output = Command::new("R").arg("--version").output();

        match output {
            Ok(output) if output.status.success() => {
                let version_output = String::from_utf8_lossy(&output.stdout);
                // R通常会在第一行输出版本信息，例如 "R version 4.2.1 (2022-06-23) -- "Bird Hippie""
                let version = version_output
                    .lines()
                    .next()
                    .and_then(|line| {
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

    fn create_matlab_project(&self, project_dir: &Path) -> Result<()> {
        // Create project structure for MATLAB
        std::fs::create_dir_all(project_dir.join("src"))?;
        std::fs::create_dir_all(project_dir.join("test"))?;
        std::fs::create_dir_all(project_dir.join("data"))?;
        std::fs::create_dir_all(project_dir.join("results"))?;
        std::fs::create_dir_all(project_dir.join("docs"))?;

        // Create main.m in src
        std::fs::write(
            project_dir.join("src/main.m"),
            r#"function main()
% MAIN Main function of the project
%
% This function serves as the entry point for the research project.
%
% Example:
%   main()
%
% See also: processData, analyzeResults

    disp('Hello, CRESP!');
    
    % Your research code goes here
    
    % Example workflow:
    % 1. Load data
    % data = loadData('../data/sample.mat');
    
    % 2. Process data
    % processedData = processData(data);
    
    % 3. Analyze results
    % results = analyzeResults(processedData);
    
    % 4. Save results
    % saveResults(results, '../results');
end
"#,
        )?;

        // Create processData.m helper function
        std::fs::write(
            project_dir.join("src/processData.m"),
            r#"function processedData = processData(data)
% PROCESSDATA Process the raw data
%
% This function takes raw data and processes it for analysis.
%
% Args:
%   data: Raw data to process
%
% Returns:
%   processedData: Processed data ready for analysis
%
% Example:
%   data = loadData('../data/sample.mat');
%   processedData = processData(data);

    % Placeholder - replace with actual data processing
    processedData = data;
    disp('Processing data...');
end
"#,
        )?;

        // Create analyzeResults.m helper function
        std::fs::write(
            project_dir.join("src/analyzeResults.m"),
            r#"function results = analyzeResults(data)
% ANALYZERESULTS Analyze the processed data
%
% This function analyzes the processed data and returns results.
%
% Args:
%   data: Processed data to analyze
%
% Returns:
%   results: Analysis results
%
% Example:
%   processedData = processData(data);
%   results = analyzeResults(processedData);

    % Placeholder - replace with actual data analysis
    results = struct('data', data, 'timestamp', now);
    disp('Analyzing results...');
end
"#,
        )?;

        // Create a test script
        std::fs::write(
            project_dir.join("test/runTests.m"),
            r#"function results = runTests()
% RUNTESTS Run all tests for the project
%
% This function runs all tests and returns the results.
%
% Returns:
%   results: Test results
%
% Example:
%   results = runTests();

    disp('Running tests...');
    
    % Initialize test results
    results = struct('passed', 0, 'failed', 0, 'total', 0);
    
    % Run test_processData
    try
        test_processData();
        results.passed = results.passed + 1;
        disp('test_processData: PASSED');
    catch ME
        results.failed = results.failed + 1;
        disp(['test_processData: FAILED - ' ME.message]);
    end
    results.total = results.total + 1;
    
    % Run test_analyzeResults
    try
        test_analyzeResults();
        results.passed = results.passed + 1;
        disp('test_analyzeResults: PASSED');
    catch ME
        results.failed = results.failed + 1;
        disp(['test_analyzeResults: FAILED - ' ME.message]);
    end
    results.total = results.total + 1;
    
    % Display summary
    disp(['Test summary: ' num2str(results.passed) '/' num2str(results.total) ' tests passed']);
end

function test_processData()
    % Test the processData function
    testData = 1:10;
    result = processData(testData);
    assert(isequal(size(result), size(testData)), 'Output size should match input size');
end

function test_analyzeResults()
    % Test the analyzeResults function
    testData = 1:10;
    result = analyzeResults(testData);
    assert(isfield(result, 'data'), 'Result should have a data field');
    assert(isfield(result, 'timestamp'), 'Result should have a timestamp field');
end
"#,
        )?;

        // Create MATLAB project file
        std::fs::write(
            project_dir.join("project.prj"),
            r#"<?xml version="1.0" encoding="UTF-8"?>
<MATLABProject xmlns="http://www.mathworks.com/MATLABProjectFile" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" version="1.0"/>
"#,
        )?;

        // Create startup.m
        std::fs::write(
            project_dir.join("startup.m"),
            r#"% STARTUP Project startup script
%
% This script runs automatically when MATLAB starts in this directory
% and sets up the project environment.

% Add src directory to the MATLAB path
addpath(genpath('src'));
addpath(genpath('test'));

disp('Project environment initialized.');
disp('Type "help main" for usage information.');
"#,
        )?;

        // Create README.md
        std::fs::write(
            project_dir.join("README.md"),
            r#"# MATLAB Research Project

This is a MATLAB research project using CRESP protocol.

## Project Structure

```
.
├── src/        # Source code
├── test/       # Test scripts
├── data/       # Input data
├── results/    # Output results
├── docs/       # Documentation
├── project.prj # MATLAB project file
└── startup.m   # Project initialization script
```

## Setup

1. Start MATLAB in the project directory or run:
```matlab
cd /path/to/project
```

2. The startup.m script will automatically add the required paths. If it doesn't run automatically, execute:
```matlab
startup
```

## Usage

Run the main script:
```matlab
main
```

## Testing

Run the test suite:
```matlab
results = runTests()
```

## Adding Dependencies

For projects using MATLAB's package management, add dependencies to the project:

1. Open the project in MATLAB:
```matlab
openProject('/path/to/project/project.prj')
```

2. Use the project manager to add required MATLAB toolboxes or files.
"#,
        )?;

        // Create .gitignore
        std::fs::write(
            project_dir.join(".gitignore"),
            r#"# MATLAB specific
*.asv
*.mex*
*.mlx
*.mat
*.fig
slprj/
sccprj/
codegen/
*.slxc
.SimulinkProject/
*.autosave
*.slx.r*
*.mdl.r*

# Results directory
results/

# Avoid large data files
data/**/*.mat
data/**/*.csv
data/**/*.xlsx
"#,
        )?;

        Ok(())
    }

    fn get_system_info(&self) -> Result<SystemInfo> {
        let mut info = SystemInfo::default();

        // Get CPU info
        if cfg!(target_os = "linux") {
            let cpu_info = Command::new("lscpu").output()?.stdout;
            let cpu_info = String::from_utf8_lossy(&cpu_info);

            info.cpu.model = cpu_info
                .lines()
                .find(|line| line.starts_with("Model name:"))
                .map(|line| line.split(":").nth(1).unwrap().trim().to_string())
                .unwrap_or_else(|| "Unknown".to_string());

            info.cpu.architecture = cpu_info
                .lines()
                .find(|line| line.starts_with("Architecture:"))
                .map(|line| line.split(":").nth(1).unwrap().trim().to_string())
                .unwrap_or_else(|| "Unknown".to_string());

            info.cpu.cores = cpu_info
                .lines()
                .find(|line| line.starts_with("CPU(s):"))
                .and_then(|line| line.split(":").nth(1).unwrap().trim().parse().ok())
                .unwrap_or(1);

            info.cpu.threads = cpu_info
                .lines()
                .find(|line| line.starts_with("Thread(s) per core:"))
                .and_then(|line| line.split(":").nth(1).unwrap().trim().parse().ok())
                .unwrap_or(1)
                * info.cpu.cores;

            info.cpu.frequency = cpu_info
                .lines()
                .find(|line| line.starts_with("CPU MHz:"))
                .map(|line| format!("{}MHz", line.split(":").nth(1).unwrap().trim()))
                .unwrap_or_else(|| "Unknown".to_string());
        } else if cfg!(target_os = "macos") {
            let cpu_info = Command::new("sysctl").arg("machdep.cpu").output()?.stdout;
            let cpu_info = String::from_utf8_lossy(&cpu_info);

            info.cpu.model = cpu_info
                .lines()
                .find(|line| line.starts_with("machdep.cpu.brand_string:"))
                .map(|line| line.split(":").nth(1).unwrap().trim().to_string())
                .unwrap_or_else(|| "Unknown".to_string());

            info.cpu.architecture = "x86_64".to_string();

            info.cpu.cores = cpu_info
                .lines()
                .find(|line| line.starts_with("machdep.cpu.core_count:"))
                .and_then(|line| line.split(":").nth(1).unwrap().trim().parse().ok())
                .unwrap_or(1);

            info.cpu.threads = cpu_info
                .lines()
                .find(|line| line.starts_with("machdep.cpu.thread_count:"))
                .and_then(|line| line.split(":").nth(1).unwrap().trim().parse().ok())
                .unwrap_or(1);

            info.cpu.frequency = cpu_info
                .lines()
                .find(|line| line.starts_with("machdep.cpu.maxspeed:"))
                .map(|line| format!("{}MHz", line.split(":").nth(1).unwrap().trim()))
                .unwrap_or_else(|| "Unknown".to_string());
        }

        // Get memory info
        if cfg!(target_os = "linux") {
            let mem_info = Command::new("free").arg("-h").output()?.stdout;
            let mem_info = String::from_utf8_lossy(&mem_info);

            info.memory.size = mem_info
                .lines()
                .nth(1)
                .and_then(|line| line.split_whitespace().nth(1))
                .unwrap_or("Unknown")
                .to_string();

            info.memory.memory_type = "DDR4/DDR5".to_string(); // This is hard to detect
        } else if cfg!(target_os = "macos") {
            let _mem_info = Command::new("vm_stat").output()?.stdout;
            let _mem_info = String::from_utf8_lossy(&_mem_info);

            info.memory.size = Command::new("sysctl")
                .arg("hw.memsize")
                .output()?
                .stdout
                .iter()
                .map(|&b| b as char)
                .collect::<String>()
                .split(":")
                .nth(1)
                .unwrap_or("Unknown")
                .trim()
                .to_string();

            info.memory.memory_type = "DDR4/DDR5".to_string(); // This is hard to detect
        }

        // Get GPU info
        if cfg!(target_os = "linux") {
            if let Ok(nvidia_smi) = Command::new("nvidia-smi")
                .arg("--query-gpu=gpu_name,memory.total,compute_cap,driver_version")
                .arg("--format=csv,noheader")
                .output()
            {
                let gpu_info = String::from_utf8_lossy(&nvidia_smi.stdout);
                if let Some(line) = gpu_info.lines().next() {
                    let parts: Vec<&str> = line.split(", ").collect();
                    if parts.len() >= 4 {
                        info.gpu.model = parts[0].to_string();
                        info.gpu.memory = parts[1].to_string();
                        info.gpu.compute_capability = parts[2].to_string();
                        info.gpu.driver_version = parts[3].to_string();
                    }
                }
            }
        } else if cfg!(target_os = "macos") {
            if let Ok(system_profiler) = Command::new("system_profiler")
                .arg("SPDisplaysDataType")
                .output()
            {
                let gpu_info = String::from_utf8_lossy(&system_profiler.stdout);
                info.gpu.model = gpu_info
                    .lines()
                    .find(|line| line.contains("Chipset Model:"))
                    .map(|line| line.split(":").nth(1).unwrap().trim().to_string())
                    .unwrap_or_else(|| "Unknown".to_string());
                info.gpu.memory = "Integrated".to_string();
                info.gpu.compute_capability = "Unknown".to_string();
                info.gpu.driver_version = "Unknown".to_string();
            }
        }

        // Get storage info
        if cfg!(target_os = "linux") {
            let storage_info = Command::new("lsblk")
                .arg("-d")
                .arg("-o")
                .arg("NAME,SIZE,MODEL")
                .output()?
                .stdout;
            let storage_info = String::from_utf8_lossy(&storage_info);

            info.storage.storage_type = storage_info
                .lines()
                .nth(1)
                .and_then(|line| line.split_whitespace().nth(2))
                .unwrap_or("Unknown")
                .to_string();
        } else if cfg!(target_os = "macos") {
            let storage_info = Command::new("diskutil")
                .arg("info")
                .arg("/")
                .output()?
                .stdout;
            let storage_info = String::from_utf8_lossy(&storage_info);

            info.storage.storage_type = storage_info
                .lines()
                .find(|line| line.contains("Media Type:"))
                .map(|line| line.split(":").nth(1).unwrap().trim().to_string())
                .unwrap_or_else(|| "Unknown".to_string());
        }

        // Get network info
        if cfg!(target_os = "linux") {
            let network_info = Command::new("ethtool")
                .arg("eth0")
                .output()
                .ok()
                .map(|output| String::from_utf8_lossy(&output.stdout).to_string());

            info.network.network_type = network_info
                .as_ref()
                .and_then(|info| {
                    info.lines()
                        .find(|line| line.contains("Supported link modes"))
                })
                .map(|line| line.split(":").nth(1).unwrap().trim().to_string())
                .unwrap_or_else(|| "Unknown".to_string());

            info.network.bandwidth = network_info
                .as_ref()
                .and_then(|info| info.lines().find(|line| line.contains("Speed:")))
                .map(|line| line.split(":").nth(1).unwrap().trim().to_string())
                .unwrap_or_else(|| "Unknown".to_string());
        } else if cfg!(target_os = "macos") {
            let network_info = Command::new("networksetup")
                .arg("-getinfo")
                .arg("Wi-Fi")
                .output()
                .ok()
                .map(|output| String::from_utf8_lossy(&output.stdout).to_string());

            info.network.network_type = "Wi-Fi".to_string();
            info.network.bandwidth = network_info
                .as_ref()
                .and_then(|info| info.lines().find(|line| line.contains("Speed:")))
                .map(|line| line.split(":").nth(1).unwrap().trim().to_string())
                .unwrap_or_else(|| "Unknown".to_string());
        }

        // Get OS info
        if cfg!(target_os = "linux") {
            let os_info = Command::new("cat").arg("/etc/os-release").output()?.stdout;
            let os_info = String::from_utf8_lossy(&os_info);

            info.os.name = os_info
                .lines()
                .find(|line| line.starts_with("NAME="))
                .map(|line| {
                    line.split("=")
                        .nth(1)
                        .unwrap()
                        .trim_matches('"')
                        .to_string()
                })
                .unwrap_or_else(|| "Unknown".to_string());

            info.os.version = os_info
                .lines()
                .find(|line| line.starts_with("VERSION="))
                .map(|line| {
                    line.split("=")
                        .nth(1)
                        .unwrap()
                        .trim_matches('"')
                        .to_string()
                })
                .unwrap_or_else(|| "Unknown".to_string());

            info.os.kernel = Command::new("uname")
                .arg("-r")
                .output()?
                .stdout
                .iter()
                .map(|&b| b as char)
                .collect::<String>()
                .trim()
                .to_string();

            info.os.architecture = Command::new("uname")
                .arg("-m")
                .output()?
                .stdout
                .iter()
                .map(|&b| b as char)
                .collect::<String>()
                .trim()
                .to_string();
        } else if cfg!(target_os = "macos") {
            let os_info = Command::new("sw_vers").output()?.stdout;
            let os_info = String::from_utf8_lossy(&os_info);

            info.os.name = os_info
                .lines()
                .find(|line| line.starts_with("ProductName:"))
                .map(|line| line.split(":").nth(1).unwrap().trim().to_string())
                .unwrap_or_else(|| "Unknown".to_string());

            info.os.version = os_info
                .lines()
                .find(|line| line.starts_with("ProductVersion:"))
                .map(|line| line.split(":").nth(1).unwrap().trim().to_string())
                .unwrap_or_else(|| "Unknown".to_string());

            info.os.kernel = Command::new("uname")
                .arg("-r")
                .output()?
                .stdout
                .iter()
                .map(|&b| b as char)
                .collect::<String>()
                .trim()
                .to_string();

            info.os.architecture = Command::new("uname")
                .arg("-m")
                .output()?
                .stdout
                .iter()
                .map(|&b| b as char)
                .collect::<String>()
                .trim()
                .to_string();
        }

        // Get locale and timezone
        info.os.locale = Command::new("locale")
            .arg("LANG")
            .output()
            .ok()
            .and_then(|output| String::from_utf8(output.stdout).ok())
            .unwrap_or_else(|| "en_US.UTF-8".to_string())
            .trim()
            .to_string();

        info.os.timezone = Command::new("date")
            .arg("+%Z")
            .output()
            .ok()
            .and_then(|output| String::from_utf8(output.stdout).ok())
            .unwrap_or_else(|| "UTC".to_string())
            .trim()
            .to_string();

        // Get system limits
        if cfg!(target_os = "linux") {
            let limits = Command::new("ulimit").arg("-n").output()?.stdout;
            info.limits.max_open_files = String::from_utf8_lossy(&limits)
                .split_whitespace()
                .next()
                .unwrap_or("65535")
                .parse()
                .unwrap_or(65535);

            let limits = Command::new("ulimit").arg("-u").output()?.stdout;
            info.limits.max_processes = String::from_utf8_lossy(&limits)
                .split_whitespace()
                .next()
                .unwrap_or("32768")
                .parse()
                .unwrap_or(32768);

            let limits = Command::new("ulimit").arg("-s").output()?.stdout;
            info.limits.stack_size = format!(
                "{}K",
                String::from_utf8_lossy(&limits)
                    .split_whitespace()
                    .next()
                    .unwrap_or("8192")
                    .parse::<u64>()
                    .unwrap_or(8192)
            );

            info.limits.virtual_memory = "unlimited".to_string();
        }

        // Get installed packages
        if cfg!(target_os = "linux") {
            let packages = Command::new("dpkg").arg("-l").output()?.stdout;
            let packages = String::from_utf8_lossy(&packages);

            info.packages = packages
                .lines()
                .filter(|line| line.starts_with("ii"))
                .map(|line| {
                    let parts: Vec<&str> = line.split_whitespace().collect();
                    format!("{{ name = \"{}\", version = \"{}\" }}", parts[1], parts[2])
                })
                .collect();
        } else if cfg!(target_os = "macos") {
            let packages = Command::new("brew")
                .arg("list")
                .arg("--versions")
                .output()?
                .stdout;
            let packages = String::from_utf8_lossy(&packages);

            info.packages = packages
                .lines()
                .map(|line| {
                    let parts: Vec<&str> = line.split_whitespace().collect();
                    format!("{{ name = \"{}\", version = \"{}\" }}", parts[0], parts[1])
                })
                .collect();
        }

        // Get software versions
        if let Ok(python_version) = Command::new("python3").arg("--version").output() {
            info.software.insert(
                "python".to_string(),
                String::from_utf8_lossy(&python_version.stdout)
                    .split_whitespace()
                    .nth(1)
                    .unwrap_or("latest")
                    .to_string(),
            );
        }

        if let Ok(r_version) = Command::new("R").arg("--version").output() {
            info.software.insert(
                "r".to_string(),
                String::from_utf8_lossy(&r_version.stdout)
                    .lines()
                    .next()
                    .unwrap_or("")
                    .split_whitespace()
                    .nth(2)
                    .unwrap_or("latest")
                    .to_string(),
            );
        }

        if let Ok(matlab_version) = Command::new("matlab").arg("-batch").arg("version").output() {
            info.software.insert(
                "matlab".to_string(),
                String::from_utf8_lossy(&matlab_version.stdout)
                    .split_whitespace()
                    .last()
                    .unwrap_or("latest")
                    .to_string(),
            );
        }

        if let Ok(conda_version) = Command::new("conda").arg("--version").output() {
            info.software.insert(
                "conda".to_string(),
                String::from_utf8_lossy(&conda_version.stdout)
                    .split_whitespace()
                    .last()
                    .unwrap_or("4.10.3")
                    .to_string(),
            );
        }

        if let Ok(cuda_version) = Command::new("nvcc").arg("--version").output() {
            info.software.insert(
                "cuda".to_string(),
                String::from_utf8_lossy(&cuda_version.stdout)
                    .lines()
                    .nth(3)
                    .unwrap_or("")
                    .split_whitespace()
                    .nth(4)
                    .unwrap_or("11.3")
                    .to_string(),
            );
        }

        if let Ok(cudnn_version) = Command::new("cat")
            .arg("/usr/include/cudnn_version.h")
            .output()
        {
            let cudnn_info = String::from_utf8_lossy(&cudnn_version.stdout);
            info.software.insert(
                "cudnn".to_string(),
                cudnn_info
                    .lines()
                    .find(|line| line.contains("CUDNN_MAJOR"))
                    .and_then(|line| line.split_whitespace().nth(2))
                    .unwrap_or("8.2.0")
                    .to_string(),
            );
        }

        if let Ok(singularity_version) = Command::new("singularity").arg("--version").output() {
            info.software
                .insert("container".to_string(), "Singularity".to_string());
            info.software.insert(
                "container_version".to_string(),
                String::from_utf8_lossy(&singularity_version.stdout)
                    .split_whitespace()
                    .last()
                    .unwrap_or("3.8.0")
                    .to_string(),
            );
        }

        // Get CUDA paths
        if let Ok(cuda_path) = Command::new("which").arg("nvcc").output() {
            if let Some(cuda_home) = String::from_utf8_lossy(&cuda_path.stdout)
                .split("/bin")
                .next()
            {
                info.cuda.cuda_home = cuda_home.to_string();
                info.cuda
                    .ld_library_path
                    .push(format!("{}/lib64", cuda_home));
                info.cuda.cupti_path = format!("{}/extras/CUPTI/lib64", cuda_home);
            }
        }

        Ok(info)
    }
}

#[derive(Default)]
struct SystemInfo {
    cpu: CpuInfo,
    memory: MemoryInfo,
    gpu: GpuInfo,
    storage: StorageInfo,
    network: NetworkInfo,
    os: OsInfo,
    limits: SystemLimits,
    packages: Vec<String>,
    software: std::collections::HashMap<String, String>,
    cuda: CudaInfo,
}

#[derive(Default)]
struct CpuInfo {
    model: String,
    architecture: String,
    cores: u32,
    threads: u32,
    frequency: String,
}

#[derive(Default)]
struct MemoryInfo {
    size: String,
    memory_type: String,
}

#[derive(Default)]
struct GpuInfo {
    model: String,
    memory: String,
    compute_capability: String,
    driver_version: String,
}

#[derive(Default)]
struct StorageInfo {
    storage_type: String,
}

#[derive(Default)]
struct NetworkInfo {
    network_type: String,
    bandwidth: String,
}

#[derive(Default)]
struct OsInfo {
    name: String,
    version: String,
    kernel: String,
    architecture: String,
    locale: String,
    timezone: String,
}

#[derive(Default)]
struct SystemLimits {
    max_open_files: u64,
    max_processes: u64,
    stack_size: String,
    virtual_memory: String,
}

#[derive(Default)]
struct CudaInfo {
    cuda_home: String,
    ld_library_path: Vec<String>,
    cupti_path: String,
}
