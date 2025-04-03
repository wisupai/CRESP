use crate::commands::new::config::UserConfig;
use crate::commands::new::system_info::SystemInfo;
use crate::error::Result;
use crate::utils::cli_ui;
use crate::utils::validation::exports::sanitize_for_conda_env;
use std::path::Path;
use super::super::utils::write_file;

/// Create CRESP configuration file
/// 
/// Args:
///     project_dir: Project directory path
///     name: Project name
///     description: Project description
///     language: Programming language (python, r, matlab)
///     system_info: System information
///     user_config: User configuration
///
/// Returns:
///     Result<()>: Operation result
pub fn create_cresp_toml(
    project_dir: &Path,
    name: &str,
    description: &str,
    language: &str,
    system_info: &SystemInfo,
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
        get_software_config(language, system_info, user_config, name),
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
        get_package_manager_type(language),
        get_config_file(language),
        get_lock_file(language),
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

    cli_ui::display_info("Writing CRESP configuration file...");
    write_file(&project_dir.join("cresp.toml"), &cresp_toml)?;
    cli_ui::display_success("CRESP configuration file created successfully.");
    Ok(())
}

/// Get software configuration part
fn get_software_config(language: &str, system_info: &SystemInfo, user_config: &UserConfig, name: &str) -> String {
    if language == "python" {
        let mut software_config = String::new();
        // Python configuration
        software_config.push_str(&format!("python = {{\n    version = \"{}\", \n    interpreter = \"python{}\",\n    compile_flags = \"--enable-shared --enable-optimizations\",\n    pip_config = {{\n        index_url = \"{}\",\n        trusted_hosts = [\"{}\"]\n    }},\n    virtual_env = {{\n        type = \"{}\",\n        {}\n    }}\n}}\n",
            user_config.python_version,
            user_config.python_version,
            user_config.pip_index_url.as_ref().unwrap_or(&"https://pypi.org/simple".to_string()),
            user_config.pip_trusted_hosts.as_ref().and_then(|hosts| hosts.first()).unwrap_or(&"pypi.org".to_string()),
            match user_config.virtual_env_type {
                crate::commands::new::config::VirtualEnvType::Conda => "conda",
                crate::commands::new::config::VirtualEnvType::None => "none",
            },
            match user_config.virtual_env_type {
                crate::commands::new::config::VirtualEnvType::Conda => {
                    let conda_env_name = sanitize_for_conda_env(name);
                    format!("name = \"{}\",\n        activation_command = \"conda activate {}\"", 
                        conda_env_name,
                        conda_env_name
                    )
                },
                crate::commands::new::config::VirtualEnvType::None => "path = \".venv\",\n        activation_script = \".venv/bin/activate\"".to_string(),
            }
        ));

        // Conda configuration (if used)
        if user_config.virtual_env_type == crate::commands::new::config::VirtualEnvType::Conda {
            // Safe default value
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
                    if let crate::commands::new::config::PackageManager::Conda { channels, .. } = pm {
                        if !channels.is_empty() {
                            Some(channels.clone())
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                })
                .next()
                .unwrap_or_else(|| vec!["conda-forge".to_string(), "defaults".to_string(), "bioconda".to_string()]);

            // Format channels string
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
            // Safe default value
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
    } else if language == "r" {
        // Create longer-lived variable
        let r_version_string = "latest".to_string();
        let r_version = match system_info.software.get("r") {
            Some(version) => version,
            None => &r_version_string
        };
        
        let conda_env_name = sanitize_for_conda_env(name);
        format!(
            r#"r = {{ 
    version = "{}", 
    interpreter = "R", 
    virtual_env = {{ 
        type = "conda",
        name = "{}",
        activation_command = "conda activate {}"
    }}
}}
conda = {{ version = "{}", channels = ["r", "conda-forge", "defaults"] }}"#,
            r_version,
            conda_env_name,
            conda_env_name,
            // Add the same fix for conda version
            match system_info.software.get("conda") {
                Some(version) => version,
                None => "4.10.3"
            }
        )
    } else if language == "matlab" {
        // Add the same fix for matlab version
        let matlab_version_string = "latest".to_string();
        let matlab_version = match system_info.software.get("matlab") {
            Some(version) => version,
            None => &matlab_version_string
        };
        
        format!("matlab = {{ version = \"{}\" }}", matlab_version)
    } else {
        // Add the same fix for other languages
        let lang_version_string = "latest".to_string();
        let lang_version = match system_info.software.get(language) {
            Some(version) => version,
            None => &lang_version_string
        };
        
        format!("{} = {{ version = \"{}\" }}", language, lang_version)
    }
}

/// Get package manager type
fn get_package_manager_type(language: &str) -> &'static str {
    match language {
        "python" => "poetry",
        "r" => "renv",
        "matlab" => "none",
        _ => "none",
    }
}

/// Get configuration file
fn get_config_file(language: &str) -> &'static str {
    match language {
        "python" => "pyproject.toml",
        "r" => "DESCRIPTION",
        "matlab" => "none",
        _ => "none",
    }
}

/// Get lock file
fn get_lock_file(language: &str) -> &'static str {
    match language {
        "python" => "poetry.lock",
        "r" => "renv.lock",
        "matlab" => "none",
        _ => "none",
    }
} 