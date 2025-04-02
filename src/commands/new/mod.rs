use crate::error::Result;
use clap::Parser;
use std::path::PathBuf;

mod config;
mod system_info;
mod templates;
mod utils;

use crate::utils::cli_ui;
use config::{UserConfig, get_python_config};
use system_info::collect_system_info;
use templates::{TemplateType, create_project_structure};
use utils::ensure_directory;

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

impl NewCommand {
    pub async fn execute(&self) -> Result<()> {
        cli_ui::display_header("Creating new CRESP project", "🚀");

        // Interactive prompts for basic project information
        let name = match &self.name {
            Some(name) => name.clone(),
            None => cli_ui::prompt_input("Project name", None::<String>)?
        };

        let description = match &self.description {
            Some(desc) => desc.clone(),
            None => cli_ui::prompt_input("Project description", None::<String>)?
        };

        // Select programming language
        let language_options = vec!["Python", "R", "MATLAB"];
        let language_idx = if let Some(lang) = &self.language {
            match lang.to_lowercase().as_str() {
                "python" => 0,
                "r" => 1,
                "matlab" => 2,
                _ => {
                    cli_ui::display_warning(&format!("Unknown language: {}. Defaulting to Python.", lang));
                    0
                }
            }
        } else {
            cli_ui::prompt_select("Select primary programming language", &language_options)?
        };
        
        let language = language_options[language_idx].to_lowercase();

        // Select project template
        let template_options = vec![
            "Basic (flat structure for simple experiments)",
            "Data Analysis (for data processing and analysis)",
            "Machine Learning (for ML/DL experiments)",
            "Scientific Computing (for numerical simulations)",
            "Custom (select your own structure)",
        ];
        
        let template_idx = if let Some(tmpl) = &self.template {
            match tmpl.as_str() {
                "1" | "basic" => 0,
                "2" | "data-analysis" => 1,
                "3" | "machine-learning" => 2,
                "4" | "scientific" => 3,
                "5" | "custom" => 4,
                _ => {
                    cli_ui::display_warning(&format!("Unknown template: {}. Defaulting to Basic.", tmpl));
                    0
                }
            }
        } else {
            cli_ui::prompt_select("Select project template", &template_options)?
        };
        
        let template = (template_idx + 1).to_string();

        // Get language-specific user configuration
        let user_config = if language == "python" {
            get_python_config()?
        } else {
            UserConfig::default()
        };

        // Create project directory
        let project_dir = self.output.join(&name);
        if !ensure_directory(&project_dir)? {
            return Ok(());
        }

        cli_ui::display_info("Collecting system information...");
        let system_info = collect_system_info()?;

        cli_ui::display_info("Creating CRESP configuration file...");
        create_cresp_toml(&project_dir, &name, &description, &language, &system_info, &user_config)?;

        cli_ui::display_info(&format!("Creating project structure using {} template...", template_options[template_idx]));
        let template_type = TemplateType::from(template.as_str());
        create_project_structure(&project_dir, template_type, &language)?;

        // Create language-specific project files
        cli_ui::display_info(&format!("Setting up {} environment...", language));
        match language.as_str() {
            "python" => templates::create_python_project(&project_dir, &user_config)?,
            "r" => templates::create_r_project(&project_dir)?,
            "matlab" => templates::create_matlab_project(&project_dir)?,
            _ => unreachable!(),
        }

        cli_ui::display_success(&format!("Project created successfully at: {}", project_dir.display()));
        
        // Output package manager installation reminders if needed
        if user_config.use_conda {
            // For Conda projects, remind user to activate the environment
            let project_name = project_dir
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("my-project");
                
            cli_ui::display_info(&format!("To use this project, activate the Conda environment:"));
            cli_ui::display_info(&format!("   conda activate {}", project_name));
        } else if user_config.uv_installed {
            // Only show this for non-Conda projects
            cli_ui::display_info("UV package manager was installed during project creation.");
            cli_ui::display_info("If UV commands are not working, try running:");
            cli_ui::display_info("  source $HOME/.local/bin/env");
            cli_ui::display_info("Or restart your terminal before running UV commands.");
        }
        
        if user_config.poetry_installed && !user_config.use_conda {
            // Only show this for non-Conda projects
            cli_ui::display_info("Poetry package manager was installed during project creation.");
            cli_ui::display_info("If Poetry commands are not working, check if it was properly added to your PATH.");
        }

        cli_ui::display_success("Command completed successfully");
        Ok(())
    }
}

/// Create CRESP configuration file
fn create_cresp_toml(
    project_dir: &PathBuf,
    name: &str,
    description: &str,
    language: &str,
    system_info: &system_info::SystemInfo,
    user_config: &UserConfig,
) -> Result<()> {
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
            // Python configuration
            software_config.push_str(&format!("python = {{\n    version = \"{}\", \n    interpreter = \"python{}\",\n    compile_flags = \"--enable-shared --enable-optimizations\",\n    pip_config = {{\n        index_url = \"{}\",\n        trusted_hosts = [\"{}\"]\n    }},\n    virtual_env = {{\n        type = \"{}\",\n        path = \".venv\",\n        activation_script = \".venv/bin/activate\"\n    }}\n}}\n",
                user_config.python_version,
                user_config.python_version,
                user_config.pip_index_url.as_ref().unwrap_or(&"https://pypi.org/simple".to_string()),
                user_config.pip_trusted_hosts.as_ref().and_then(|hosts| hosts.first()).unwrap_or(&"pypi.org".to_string()),
                match user_config.virtual_env_type {
                    config::VirtualEnvType::Venv => "venv",
                    config::VirtualEnvType::Virtualenv => "virtualenv",
                    config::VirtualEnvType::Conda => "conda",
                    config::VirtualEnvType::None => "none"
                }
            ));

            // Conda configuration (if used)
            if user_config.virtual_env_type == config::VirtualEnvType::Conda {
                // Safe defaults if not found
                let default_conda_version = "4.10.3".to_string();
                let conda_version = system_info
                    .software
                    .get("conda")
                    .unwrap_or(&default_conda_version);

                // Get all conda channels
                let channels = user_config
                    .package_managers
                    .iter()
                    .filter_map(|pm| {
                        if let config::PackageManager::Conda { channels, .. } = pm {
                            Some(channels.clone())
                        } else {
                            None
                        }
                    })
                    .next()
                    .unwrap_or_else(|| vec!["conda-forge".to_string()]);

                // Format channel string
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

            // CUDA configuration (if used)
            if user_config.use_cuda {
                // Safe defaults if not found
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

                // cuDNN configuration
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

            software_config
        } else {
            format!(
                "{} = {{ version = \"{}\" }}",
                language,
                system_info
                    .software
                    .get(language)
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
        match language {
            "python" => "poetry",
            "r" => "renv",
            "matlab" => "none",
            _ => "none",
        },
        match language {
            "python" => "pyproject.toml",
            "r" => "DESCRIPTION",
            "matlab" => "none",
            _ => "none",
        },
        match language {
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

    utils::write_file(&project_dir.join("cresp.toml"), &cresp_toml)?;
    Ok(())
} 