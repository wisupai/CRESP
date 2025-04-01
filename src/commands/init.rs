use clap::Parser;
use log::{info, warn};
use std::path::PathBuf;
use std::io::{self, Write};
use crate::error::Result;

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
        info!("📦 Initializing CRESP configuration in existing project...");

        // Detect project type
        let language = self.language.clone().unwrap_or_else(|| {
            println!("🔍 Detecting project type...");
            if self.output.join("requirements.txt").exists() || 
               self.output.join("pyproject.toml").exists() || 
               self.output.join("setup.py").exists() {
                println!("📊 Detected Python project");
                "python".to_string()
            } else if self.output.join("DESCRIPTION").exists() || 
                      self.output.join("NAMESPACE").exists() || 
                      self.output.join("renv.lock").exists() {
                println!("📊 Detected R project");
                "r".to_string()
            } else if self.output.join("matlab").exists() || 
                      self.output.join("*.m").exists() {
                println!("📊 Detected MATLAB project");
                "matlab".to_string()
            } else {
                println!("❓ Could not detect project type. Please select manually:");
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
                }.to_string()
            }
        });

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

        // Check if cresp.toml already exists
        let cresp_toml_path = self.output.join("cresp.toml");
        if cresp_toml_path.exists() {
            warn!("⚠️ CRESP configuration already exists at: {}", cresp_toml_path.display());
            print!("Do you want to overwrite it? (y/N): ");
            io::stdout().flush().unwrap();
            let mut input = String::new();
            io::stdin().read_line(&mut input).unwrap();
            if input.trim().to_lowercase() != "y" {
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

        info!("✨ CRESP configuration initialized successfully at: {}", cresp_toml_path.display());
        Ok(())
    }

    fn setup_python_project(&self, project_dir: &PathBuf) -> Result<()> {
        // Check if poetry is installed
        if let Ok(_) = std::process::Command::new("poetry").arg("--version").output() {
            info!("📦 Poetry is already installed");
        } else {
            info!("📦 Installing Poetry...");
            std::process::Command::new("curl")
                .arg("-sSL")
                .arg("https://install.python-poetry.org")
                .arg("|")
                .arg("python3")
                .output()?;
        }

        // Initialize poetry project if pyproject.toml doesn't exist
        if !project_dir.join("pyproject.toml").exists() {
            info!("📦 Initializing Poetry project...");
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
            info!("📦 renv is already installed");
        } else {
            info!("📦 Installing renv...");
            std::process::Command::new("Rscript")
                .arg("-e")
                .arg("install.packages('renv')")
                .output()?;
        }

        // Initialize renv if renv.lock doesn't exist
        if !project_dir.join("renv.lock").exists() {
            info!("📦 Initializing renv project...");
            std::process::Command::new("Rscript")
                .arg("-e")
                .arg("renv::init()")
                .current_dir(project_dir)
                .output()?;
        }

        Ok(())
    }

    fn setup_matlab_project(&self, project_dir: &PathBuf) -> Result<()> {
        // Create matlab directory if it doesn't exist
        std::fs::create_dir_all(project_dir.join("matlab"))?;
        Ok(())
    }
} 