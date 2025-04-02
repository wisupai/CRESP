use clap::Parser;
use log::{info, warn};
use std::path::PathBuf;
use crate::error::Result;
use crate::utils::cli_ui;

#[derive(Parser, Debug)]
pub struct InitCommand {
    /// Project name
    #[arg(short, long)]
    name: Option<String>,

    /// Project description
    #[arg(short, long)]
    description: Option<String>,

    /// Primary programming language (python, r, matlab)
    #[arg(short, long)]
    language: Option<String>,

    /// Output directory
    #[arg(short, long, default_value = ".")]
    output: PathBuf,
}

impl InitCommand {
    pub async fn execute(&self) -> Result<()> {
        cli_ui::display_header("Initializing CRESP configuration", "📦");

        // Detect project type
        let language = match &self.language {
            Some(lang) => lang.clone(),
            None => {
                cli_ui::display_info("Detecting project type...");
                
                if self.output.join("requirements.txt").exists() || 
                   self.output.join("pyproject.toml").exists() || 
                   self.output.join("setup.py").exists() {
                    cli_ui::display_success("Detected Python project");
                    "python".to_string()
                } else if self.output.join("DESCRIPTION").exists() || 
                          self.output.join("NAMESPACE").exists() || 
                          self.output.join("renv.lock").exists() {
                    cli_ui::display_success("Detected R project");
                    "r".to_string()
                } else if self.output.join("matlab").exists() || 
                          self.output.join("*.m").exists() {
                    cli_ui::display_success("Detected MATLAB project");
                    "matlab".to_string()
                } else {
                    let options = vec!["Python", "R", "MATLAB"];
                    let selection = cli_ui::prompt_select("Select project type", &options)?;
                    match selection {
                        0 => "python",
                        1 => "r",
                        2 => "matlab",
                        _ => "python",
                    }.to_string()
                }
            }
        };

        // Interactive prompts
        let name = match &self.name {
            Some(name) => name.clone(),
            None => cli_ui::prompt_input("Project name", None::<String>)?
        };

        let description = match &self.description {
            Some(desc) => desc.clone(),
            None => cli_ui::prompt_input("Project description", None::<String>)?
        };

        // Check if cresp.toml already exists
        let cresp_toml_path = self.output.join("cresp.toml");
        if cresp_toml_path.exists() {
            cli_ui::display_warning(&format!("CRESP configuration already exists at: {}", cresp_toml_path.display()));
            let should_overwrite = cli_ui::prompt_confirm("Do you want to overwrite it?", false)?;
            if !should_overwrite {
                return Ok(());
            }
        }

        // Create cresp.toml
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

[experiment.environment]
description = "The original environment where the research was conducted"

[experiment.environment.software]
{}.version = "latest"

[experiment.data]
description = "Dataset configuration will be added here"

[execution]
description = "Execution configuration will be added here"
"#,
            name,
            description,
            language
        );

        std::fs::write(&cresp_toml_path, cresp_toml)?;

        // Setup language-specific package manager
        match language.as_str() {
            "python" => self.setup_python_project(&self.output)?,
            "r" => self.setup_r_project(&self.output)?,
            "matlab" => self.setup_matlab_project(&self.output)?,
            _ => unreachable!(),
        }

        cli_ui::display_success(&format!("CRESP configuration initialized successfully at: {}", cresp_toml_path.display()));
        Ok(())
    }

    fn setup_python_project(&self, project_dir: &PathBuf) -> Result<()> {
        // Check if poetry is installed
        if let Ok(_) = std::process::Command::new("poetry").arg("--version").output() {
            cli_ui::display_info("Poetry is already installed");
        } else {
            cli_ui::display_info("Installing Poetry...");
            std::process::Command::new("curl")
                .arg("-sSL")
                .arg("https://install.python-poetry.org")
                .arg("|")
                .arg("python3")
                .output()?;
        }

        // Initialize poetry project if pyproject.toml doesn't exist
        if !project_dir.join("pyproject.toml").exists() {
            cli_ui::display_info("Initializing Poetry project...");
            std::process::Command::new("poetry")
                .arg("init")
                .arg("--name")
                .arg("my-research")
                .arg("--description")
                .arg("")
                .arg("--author")
                .arg("Your Name <your.email@example.com>")
                .arg("--python")
                .arg("^3.9")
                .arg("--dependency")
                .arg("pytest")
                .arg("--dev-dependency")
                .arg("black")
                .arg("--dev-dependency")
                .arg("isort")
                .arg("--dev-dependency")
                .arg("flake8")
                .arg("--no-interaction")
                .current_dir(project_dir)
                .output()?;
        }

        Ok(())
    }

    fn setup_r_project(&self, project_dir: &PathBuf) -> Result<()> {
        // Check if renv is installed
        if let Ok(_) = std::process::Command::new("Rscript")
            .arg("-e")
            .arg("packageVersion('renv')")
            .output() {
            cli_ui::display_info("renv is already installed");
        } else {
            cli_ui::display_info("Installing renv...");
            std::process::Command::new("Rscript")
                .arg("-e")
                .arg("install.packages('renv')")
                .output()?;
        }

        // Initialize renv if renv.lock doesn't exist
        if !project_dir.join("renv.lock").exists() {
            cli_ui::display_info("Initializing renv project...");
            std::process::Command::new("Rscript")
                .arg("-e")
                .arg("renv::init()")
                .current_dir(project_dir)
                .output()?;
        }

        Ok(())
    }

    fn setup_matlab_project(&self, project_dir: &PathBuf) -> Result<()> {
        // Create MATLAB project structure
        let matlab_src_dir = project_dir.join("src");
        let matlab_test_dir = project_dir.join("tests");
        let matlab_data_dir = project_dir.join("data");
        
        if !matlab_src_dir.exists() {
            cli_ui::display_info("Creating MATLAB project structure...");
            std::fs::create_dir_all(matlab_src_dir)?;
            std::fs::create_dir_all(matlab_test_dir)?;
            std::fs::create_dir_all(matlab_data_dir)?;
            
            // Create a sample MATLAB startup file
            let startup_file = project_dir.join("startup.m");
            let startup_content = r#"% MATLAB Startup File
% Add all necessary paths for the project

% Add source directory to path
addpath(genpath('./src'));

% Add test directory to path
addpath(genpath('./tests'));

disp('Project environment initialized.');
"#;
            std::fs::write(startup_file, startup_content)?;
        }
        
        Ok(())
    }
} 