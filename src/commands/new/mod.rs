use crate::error::Result;
use clap::Parser;
use std::path::PathBuf;

mod config;
mod system_info;
mod templates;
mod utils;

use crate::utils::cli_ui;
use crate::utils::validation::{ProjectNameValidator, Validator};
use config::{get_python_config, UserConfig};
use system_info::collect_system_info;
use templates::{create_project_structure, TemplateType, create_cresp_toml};
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
            Some(name) => {
                // Also validate project name provided via command line
                let validator = ProjectNameValidator;
                let (is_valid, message) = validator.validate(name);
                if !is_valid {
                    cli_ui::display_error(&message);
                    cli_ui::display_error(
                        "Please provide a valid project name without spaces or special characters.",
                    );
                    return Err(crate::error::Error::Validation(
                        "Invalid project name provided via command line.".to_string(),
                    ));
                }
                name.clone()
            }
            None => {
                // 使用带验证的输入函数
                cli_ui::prompt_input_with_validation(
                    "Project name",
                    None::<String>,
                    ProjectNameValidator,
                    "Please enter a valid project name without spaces or special characters.",
                )?
            }
        };

        let description = match &self.description {
            Some(desc) => desc.clone(),
            None => cli_ui::prompt_input(
                "Project description (press Enter to skip)",
                Some(String::new()),
            )?,
        };

        // Select programming language
        let language_options = vec!["Python", "R", "MATLAB"];
        let language_idx = if let Some(lang) = &self.language {
            match lang.to_lowercase().as_str() {
                "python" => 0,
                "r" => 1,
                "matlab" => 2,
                _ => {
                    cli_ui::display_warning(&format!(
                        "Unknown language: {}. Defaulting to Python.",
                        lang
                    ));
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
                    cli_ui::display_warning(&format!(
                        "Unknown template: {}. Defaulting to Basic.",
                        tmpl
                    ));
                    0
                }
            }
        } else {
            cli_ui::prompt_select("Select project template", &template_options)?
        };

        let template = (template_idx + 1).to_string();

        // Get language-specific user configuration
        let mut user_config = if language == "python" {
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
        create_cresp_toml(
            &project_dir,
            &name,
            &description,
            &language,
            &system_info,
            &user_config,
        )?;

        cli_ui::display_info(&format!(
            "Creating project structure using {} template...",
            template_options[template_idx]
        ));
        let template_type = TemplateType::from(template.as_str());
        create_project_structure(&project_dir, template_type, &language)?;

        // Create language-specific project files
        cli_ui::display_info(&format!("Setting up {} environment...", language));
        match language.as_str() {
            "python" => templates::create_python_project(&project_dir, &mut user_config)?,
            "r" => templates::create_r_project(&project_dir)?,
            "matlab" => templates::create_matlab_project(&project_dir)?,
            _ => unreachable!(),
        }

        cli_ui::display_success(&format!(
            "Project created successfully at: {}",
            project_dir.display()
        ));

        // Output package manager installation reminders if needed
        if user_config.use_conda {
            // For Conda projects, remind user to activate the environment
            let project_name = project_dir
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("my-project");

            cli_ui::display_info(&format!(
                "To use this project, activate the Conda environment:"
            ));
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
            cli_ui::display_info(
                "If Poetry commands are not working, check if it was properly added to your PATH.",
            );
        }

        Ok(())
    }
}
