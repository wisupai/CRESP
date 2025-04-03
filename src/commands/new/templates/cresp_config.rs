use crate::commands::new::config::UserConfig;
use crate::commands::new::system_info::SystemInfo;
use crate::error::Result;
use crate::utils::cli_ui;
use crate::utils::validation::exports::sanitize_for_conda_env;
use std::path::Path;
use super::super::utils::write_file;
use toml_edit::{DocumentMut, Item, value, array, Table, InlineTable, Value as TomlValue, Formatted};

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
    // Create new TOML document
    let mut doc = DocumentMut::new();
    
    // Add version with comment
    doc["cresp_version"] = value("1.0");
    if let Some(v) = doc.get_mut("cresp_version") {
        v.as_value_mut().unwrap().decor_mut().set_prefix("# CRESP Protocol Configuration\n# Documentation: https://cresp.resciencelab.ai\n\n");
    }
    
    // Experiment section
    let mut experiment = doc.as_table_mut().entry("experiment").or_insert(Item::Table(Default::default())).as_table_mut().unwrap();
    experiment["name"] = value(name);
    experiment["description"] = value(description);
    
    // Keywords
    let keywords = array();
    experiment["keywords"] = keywords;
    
    // Authors
    let mut author_array = array();
    let mut author = InlineTable::default();
    author.insert("name", TomlValue::String(Formatted::new("Your Name".to_string())));
    author.insert("email", TomlValue::String(Formatted::new("your.email@example.com".to_string())));
    author.insert("affiliation", TomlValue::String(Formatted::new("Your Institution".to_string())));
    author.insert("role", TomlValue::String(Formatted::new("Researcher".to_string())));
    
    // Create an empty array and use as_array_mut to get a mutable reference
    experiment["authors"] = array();
    if let Some(authors) = experiment["authors"].as_array_mut() {
        authors.push(TomlValue::InlineTable(author));
    }
    
    // Environment section
    let mut environment = experiment.entry("environment").or_insert(Item::Table(Default::default())).as_table_mut().unwrap();
    environment.decor_mut().set_prefix("\n\n###############################################################################\n# Original Research Environment\n###############################################################################\n\n");
    environment["description"] = value("The original environment where the research was conducted");
    
    // Hardware section
    let mut hardware = environment.entry("hardware").or_insert(Item::Table(Default::default())).as_table_mut().unwrap();
    
    // CPU
    hardware["cpu"] = Item::Table(Default::default());
    let mut cpu = hardware["cpu"].as_table_mut().unwrap();
    cpu["model"] = value(&system_info.cpu.model);
    cpu["architecture"] = value(&system_info.cpu.architecture);
    cpu["cores"] = value(system_info.cpu.cores.to_string());
    cpu["threads"] = value(system_info.cpu.threads.to_string());
    cpu["frequency"] = value(&system_info.cpu.frequency);
    
    // Memory
    hardware["memory"] = Item::Table(Default::default());
    let mut memory = hardware["memory"].as_table_mut().unwrap();
    memory["size"] = value(&system_info.memory.size);
    memory["type"] = value(&system_info.memory.memory_type);
    
    // GPU
    hardware["gpu"] = Item::Table(Default::default());
    let mut gpu = hardware["gpu"].as_table_mut().unwrap();
    gpu["default_model"] = Item::Table(Default::default());
    let mut default_model = gpu["default_model"].as_table_mut().unwrap();
    default_model["model"] = value(&system_info.gpu.model);
    default_model["memory"] = value(&system_info.gpu.memory);
    default_model["compute_capability"] = value(&system_info.gpu.compute_capability);
    gpu["driver_version"] = value(&system_info.gpu.driver_version);
    
    // Storage
    hardware["storage"] = Item::Table(Default::default());
    let mut storage = hardware["storage"].as_table_mut().unwrap();
    storage["type"] = value(&system_info.storage.storage_type);
    
    // Network
    hardware["network"] = Item::Table(Default::default());
    let mut network = hardware["network"].as_table_mut().unwrap();
    network["type"] = value(&system_info.network.network_type);
    network["bandwidth"] = value(&system_info.network.bandwidth);
    
    // System section
    let mut system = environment.entry("system").or_insert(Item::Table(Default::default())).as_table_mut().unwrap();
    
    // OS
    system["os"] = Item::Table(Default::default());
    let mut os = system["os"].as_table_mut().unwrap();
    os["name"] = value(&system_info.os.name);
    os["version"] = value(&system_info.os.version);
    os["kernel"] = value(&system_info.os.kernel);
    os["architecture"] = value(&system_info.os.architecture);
    os["locale"] = value(&system_info.os.locale);
    os["timezone"] = value(&system_info.os.timezone);
    
    // Packages
    system["packages"] = array();
    if let Some(packages) = system["packages"].as_array_mut() {
        for package in &system_info.packages {
            packages.push(TomlValue::String(Formatted::new(package.clone())));
        }
    }
    
    // Limits
    system["limits"] = Item::Table(Default::default());
    let mut limits = system["limits"].as_table_mut().unwrap();
    limits["max_open_files"] = value(system_info.limits.max_open_files.to_string());
    limits["max_processes"] = value(system_info.limits.max_processes.to_string());
    limits["stack_size"] = value(&system_info.limits.stack_size);
    limits["virtual_memory"] = value(&system_info.limits.virtual_memory);
    
    // Software section
    environment["software"] = Item::Table(Default::default());
    let mut software = environment["software"].as_table_mut().unwrap();
    add_software_config(software, language, system_info, user_config, name);
    
    // Variables section
    environment["variables"] = Item::Table(Default::default());
    let mut variables = environment["variables"].as_table_mut().unwrap();
    
    // System variables
    variables["system"] = Item::Table(Default::default());
    let mut system_vars = variables["system"].as_table_mut().unwrap();
    system_vars["LANG"] = value(&system_info.os.locale);
    system_vars["LC_ALL"] = value(&system_info.os.locale);
    system_vars["TZ"] = value(&system_info.os.timezone);
    
    // Language path
    variables[language] = Item::Table(Default::default());
    let mut lang_vars = variables[language].as_table_mut().unwrap();
    lang_vars["path"] = array();
    if let Some(lang_path) = lang_vars["path"].as_array_mut() {
        lang_path.push(TomlValue::String(Formatted::new(project_dir.display().to_string())));
    }
    
    // CUDA variables (if used)
    if user_config.use_cuda {
        variables["cuda"] = Item::Table(Default::default());
        let mut cuda_vars = variables["cuda"].as_table_mut().unwrap();
        cuda_vars["version"] = value(user_config.cuda_version.as_ref().unwrap_or(&"11.8".to_string()));
        cuda_vars["CUDA_HOME"] = value(&system_info.cuda.cuda_home);
        
        cuda_vars["LD_LIBRARY_PATH"] = array();
        if let Some(ld_library_path) = cuda_vars["LD_LIBRARY_PATH"].as_array_mut() {
            for path in &system_info.cuda.ld_library_path {
                ld_library_path.push(TomlValue::String(Formatted::new(path.clone())));
            }
            ld_library_path.push(TomlValue::String(Formatted::new(system_info.cuda.cupti_path.clone())));
        }
    }
    
    // Experiment variables
    variables["experiment"] = Item::Table(Default::default());
    let mut experiment_vars = variables["experiment"].as_table_mut().unwrap();
    experiment_vars["EXPERIMENT_DATA_DIR"] = value(project_dir.join("data").display().to_string());
    experiment_vars["EXPERIMENT_OUTPUT_DIR"] = value(project_dir.join("output").display().to_string());
    
    // Dependencies section
    environment["dependencies"] = Item::Table(Default::default());
    let mut dependencies = environment["dependencies"].as_table_mut().unwrap();
    dependencies["type"] = value(language);
    
    // Package manager
    dependencies["package_manager"] = Item::Table(Default::default());
    let mut package_manager = dependencies["package_manager"].as_table_mut().unwrap();
    package_manager["type"] = value(get_package_manager_type(language));
    package_manager["config_file"] = value(get_config_file(language));
    package_manager["lock_file"] = value(get_lock_file(language));
    
    // Fallbacks
    if user_config.use_cuda {
        dependencies["conda_fallback"] = Item::Table(Default::default());
        let mut conda_fallback = dependencies["conda_fallback"].as_table_mut().unwrap();
        conda_fallback["enabled"] = value(true);
        conda_fallback["environment_file"] = value("environment.yml");
        conda_fallback["dev_environment_file"] = value("environment-dev.yml");
    } else {
        dependencies["pip_fallback"] = Item::Table(Default::default());
        let mut pip_fallback = dependencies["pip_fallback"].as_table_mut().unwrap();
        pip_fallback["enabled"] = value(true);
        pip_fallback["requirements_file"] = value("requirements.txt");
        pip_fallback["dev_requirements_file"] = value("requirements-dev.txt");
        
        dependencies["uv_fallback"] = Item::Table(Default::default());
        let mut uv_fallback = dependencies["uv_fallback"].as_table_mut().unwrap();
        uv_fallback["enabled"] = value(true);
        uv_fallback["requirements_file"] = value("requirements.txt");
        uv_fallback["dev_requirements_file"] = value("requirements-dev.txt");
    }

    cli_ui::display_info("Writing CRESP configuration file...");
    write_file(&project_dir.join("cresp.toml"), &doc.to_string())?;
    cli_ui::display_success("CRESP configuration file created successfully.");
    Ok(())
}

/// Add software configuration based on language
fn add_software_config(
    software: &mut Table, 
    language: &str, 
    system_info: &SystemInfo, 
    user_config: &UserConfig, 
    name: &str
) {
    if language == "python" {
        // Python configuration
        software["python"] = Item::Table(Default::default());
        let mut python = software["python"].as_table_mut().unwrap();
        python["version"] = value(&user_config.python_version);
        python["interpreter"] = value(format!("python{}", user_config.python_version));
        python["compile_flags"] = value("--enable-shared --enable-optimizations");
        
        // Pip config
        python["pip_config"] = Item::Table(Default::default());
        let pip_config = python["pip_config"].as_table_mut().unwrap();
        pip_config["index_url"] = value(user_config.pip_index_url.as_ref().unwrap_or(&"https://pypi.org/simple".to_string()));
        
        pip_config["trusted_hosts"] = array();
        if let Some(trusted_hosts) = pip_config["trusted_hosts"].as_array_mut() {
            let host_string = user_config.pip_trusted_hosts.as_ref()
                .and_then(|hosts| hosts.first())
                .cloned()
                .unwrap_or_else(|| "pypi.org".to_string());
            trusted_hosts.push(TomlValue::String(Formatted::new(host_string)));
        }
        
        // Virtual env
        python["virtual_env"] = Item::Table(Default::default());
        let mut virtual_env = python["virtual_env"].as_table_mut().unwrap();
        match user_config.virtual_env_type {
            crate::commands::new::config::VirtualEnvType::Conda => {
                let conda_env_name = sanitize_for_conda_env(name);
                virtual_env["type"] = value("conda");
                virtual_env["name"] = value(&conda_env_name);
                virtual_env["activation_command"] = value(format!("conda activate {}", conda_env_name));
            },
            crate::commands::new::config::VirtualEnvType::None => {
                virtual_env["type"] = value("none");
                virtual_env["path"] = value(".venv");
                virtual_env["activation_script"] = value(".venv/bin/activate");
            }
        }
        
        // Conda configuration (if used)
        if user_config.virtual_env_type == crate::commands::new::config::VirtualEnvType::Conda {
            // Safe default value
            let default_conda_version = "4.10.3".to_string();
            let conda_version = system_info
                .software
                .get("conda")
                .unwrap_or(&default_conda_version);
            
            // Get conda environment name
            let conda_env_name = sanitize_for_conda_env(name);
            
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
            
            software["conda"] = Item::Table(Default::default());
            let mut conda = software["conda"].as_table_mut().unwrap();
            conda["version"] = value(conda_version);
            
            // Channels
            conda["channels"] = array();
            if let Some(channels_array) = conda["channels"].as_array_mut() {
                for channel in channels {
                    channels_array.push(TomlValue::String(Formatted::new(channel)));
                }
            }
            
            // Try to get installed packages
            match super::super::templates::conda_utils::get_conda_installed_packages(&conda_env_name) {
                Ok(packages) if !packages.is_empty() => {
                    conda["packages"] = array();
                    if let Some(packages_array) = conda["packages"].as_array_mut() {
                        for (pkg_name, version) in packages {
                            // Skip conda internal packages and some common base packages
                            if !["python", "conda", "pip"].contains(&pkg_name.as_str()) {
                                let mut package = InlineTable::default();
                                package.insert("name", TomlValue::String(Formatted::new(pkg_name)));
                                package.insert("version", TomlValue::String(Formatted::new(version)));
                                packages_array.push(TomlValue::InlineTable(package));
                            }
                        }
                    }
                },
                _ => {
                    // If no packages found, don't add any
                    cli_ui::display_info("No packages found in conda environment. The environment may not exist yet.");
                }
            };
        }
        
        // CUDA configuration (if used)
        if user_config.use_cuda {
            // Safe default value
            let default_cuda_version = "11.8".to_string();
            let cuda_version = user_config
                .cuda_version
                .as_ref()
                .unwrap_or(&default_cuda_version);
            
            software["cuda"] = Item::Table(Default::default());
            let mut cuda = software["cuda"].as_table_mut().unwrap();
            cuda["version"] = value(cuda_version);
            cuda["toolkit"] = value(format!("cuda_{}_linux", cuda_version.replace(".", "_")));
            
            // cuDNN configuration
            let default_cudnn_version = "8.9".to_string();
            let cudnn_version = user_config
                .cudnn_version
                .as_ref()
                .unwrap_or(&default_cudnn_version);
            
            software["cudnn"] = Item::Table(Default::default());
            let mut cudnn = software["cudnn"].as_table_mut().unwrap();
            cudnn["version"] = value(cudnn_version);
            cudnn["toolkit"] = value(format!("cudnn-{}-linux-x64-v{}", cuda_version, cudnn_version));
        }
    } else if language == "r" {
        // Create longer-lived variable
        let r_version_string = "latest".to_string();
        let r_version = match system_info.software.get("r") {
            Some(version) => version,
            None => &r_version_string
        };
        
        let conda_env_name = sanitize_for_conda_env(name);
        
        // R configuration
        software["r"] = Item::Table(Default::default());
        let mut r = software["r"].as_table_mut().unwrap();
        r["version"] = value(r_version);
        r["interpreter"] = value("R");
        
        // Virtual env
        r["virtual_env"] = Item::Table(Default::default());
        let mut virtual_env = r["virtual_env"].as_table_mut().unwrap();
        virtual_env["type"] = value("conda");
        virtual_env["name"] = value(&conda_env_name);
        virtual_env["activation_command"] = value(format!("conda activate {}", conda_env_name));
        
        // Conda configuration
        software["conda"] = Item::Table(Default::default());
        let mut conda = software["conda"].as_table_mut().unwrap();
        conda["version"] = value(match system_info.software.get("conda") {
            Some(version) => version,
            None => "4.10.3"
        });
        
        // Try to get installed packages
        match super::super::templates::conda_utils::get_conda_installed_packages(&conda_env_name) {
            Ok(packages) if !packages.is_empty() => {
                conda["packages"] = array();
                if let Some(packages_array) = conda["packages"].as_array_mut() {
                    for (pkg_name, version) in packages {
                        // Skip conda internal packages and R itself
                        if !["r-base", "conda", "r-essentials"].contains(&pkg_name.as_str()) {
                            let mut package = InlineTable::default();
                            package.insert("name", TomlValue::String(Formatted::new(pkg_name)));
                            package.insert("version", TomlValue::String(Formatted::new(version)));
                            packages_array.push(TomlValue::InlineTable(package));
                        }
                    }
                }
            },
            _ => {
                // If no packages found, don't add any
            }
        };
    } else if language == "matlab" {
        // Matlab configuration
        let matlab_version_string = "latest".to_string();
        let matlab_version = match system_info.software.get("matlab") {
            Some(version) => version,
            None => &matlab_version_string
        };
        
        software["matlab"] = Item::Table(Default::default());
        let mut matlab = software["matlab"].as_table_mut().unwrap();
        matlab["version"] = value(matlab_version);
    } else {
        // Generic language configuration
        let lang_version_string = "latest".to_string();
        let lang_version = match system_info.software.get(language) {
            Some(version) => version,
            None => &lang_version_string
        };
        
        software[language] = Item::Table(Default::default());
        let mut lang_config = software[language].as_table_mut().unwrap();
        lang_config["version"] = value(lang_version);
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