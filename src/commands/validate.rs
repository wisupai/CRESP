use clap::Parser;
use log::info;
use std::path::PathBuf;
use toml::Value;

use crate::error::Result;

#[derive(Parser, Debug)]
pub struct ValidateCommand {
    /// Path to CRESP configuration file
    #[arg(short, long, default_value = "cresp.toml")]
    path: PathBuf,

    /// Enable strict validation mode
    #[arg(short, long)]
    strict: bool,

    /// Custom schema file path
    #[arg(short, long)]
    schema: Option<PathBuf>,
}

impl ValidateCommand {
    pub async fn execute(&self) -> Result<()> {
        info!("🔍 Validating CRESP configuration: {}", self.path.display());

        // Read and parse TOML file
        let contents = std::fs::read_to_string(&self.path)?;
        let config: Value = toml::from_str(&contents)?;

        // Validate basic structure
        self.validate_basic_structure(&config)?;

        // Validate experiment section
        self.validate_experiment_section(&config)?;

        // Validate environment section
        self.validate_environment_section(&config)?;

        // Validate data section
        self.validate_data_section(&config)?;

        // Validate execution section
        self.validate_execution_section(&config)?;

        info!("✅ Configuration validation successful");
        Ok(())
    }

    fn validate_basic_structure(&self, config: &Value) -> Result<()> {
        // Check required top-level fields
        let required_fields = ["cresp_version", "experiment"];
        for field in required_fields {
            if !config.get(field).is_some() {
                return Err(crate::error::Error::Validation(format!(
                    "Missing required field: {}",
                    field
                )));
            }
        }

        // Validate version
        if let Some(version) = config.get("cresp_version").and_then(|v| v.as_str()) {
            if !version.starts_with("1.") {
                return Err(crate::error::Error::Validation(format!(
                    "Unsupported CRESP version: {}",
                    version
                )));
            }
        }

        Ok(())
    }

    fn validate_experiment_section(&self, config: &Value) -> Result<()> {
        let experiment = config.get("experiment").ok_or_else(|| {
            crate::error::Error::Validation("Missing experiment section".to_string())
        })?;

        // Check required fields
        let required_fields = ["name", "description", "authors"];
        for field in required_fields {
            if !experiment.get(field).is_some() {
                return Err(crate::error::Error::Validation(format!(
                    "Missing required field in experiment section: {}",
                    field
                )));
            }
        }

        // Validate authors
        if let Some(authors) = experiment.get("authors").and_then(|v| v.as_array()) {
            for author in authors {
                if let Some(author) = author.as_table() {
                    let required_fields = ["name", "email"];
                    for field in required_fields {
                        if !author.contains_key(field) {
                            return Err(crate::error::Error::Validation(format!(
                                "Missing required field in author: {}",
                                field
                            )));
                        }
                    }
                }
            }
        }

        Ok(())
    }

    fn validate_environment_section(&self, config: &Value) -> Result<()> {
        let experiment = config.get("experiment").ok_or_else(|| {
            crate::error::Error::Validation("Missing experiment section".to_string())
        })?;

        let environment = experiment.get("environment").ok_or_else(|| {
            crate::error::Error::Validation("Missing environment section".to_string())
        })?;

        // Validate hardware section if present
        if let Some(hardware) = environment.get("hardware") {
            self.validate_hardware_section(hardware)?;
        }

        // Validate software section if present
        if let Some(software) = environment.get("software") {
            self.validate_software_section(software)?;
        }

        Ok(())
    }

    fn validate_hardware_section(&self, hardware: &Value) -> Result<()> {
        if let Some(cpu) = hardware.get("cpu") {
            let required_fields = ["model", "architecture", "cores"];
            for field in required_fields {
                if !cpu.get(field).is_some() {
                    return Err(crate::error::Error::Validation(format!(
                        "Missing required field in CPU configuration: {}",
                        field
                    )));
                }
            }
        }

        if let Some(memory) = hardware.get("memory") {
            if !memory.get("size").is_some() {
                return Err(crate::error::Error::Validation(
                    "Missing size field in memory configuration".to_string(),
                ));
            }
        }

        Ok(())
    }

    fn validate_software_section(&self, software: &Value) -> Result<()> {
        // Check for at least one language configuration
        let has_language = ["python", "r", "matlab"]
            .iter()
            .any(|lang| software.get(lang).is_some());
        if !has_language {
            return Err(crate::error::Error::Validation(
                "No language configuration found in software section".to_string(),
            ));
        }

        Ok(())
    }

    fn validate_data_section(&self, config: &Value) -> Result<()> {
        let experiment = config.get("experiment").ok_or_else(|| {
            crate::error::Error::Validation("Missing experiment section".to_string())
        })?;

        if let Some(data) = experiment.get("data") {
            if let Some(datasets) = data.get("datasets").and_then(|v| v.as_array()) {
                for dataset in datasets {
                    if let Some(dataset) = dataset.as_table() {
                        let required_fields = ["name", "source", "description"];
                        for field in required_fields {
                            if !dataset.contains_key(field) {
                                return Err(crate::error::Error::Validation(format!(
                                    "Missing required field in dataset: {}",
                                    field
                                )));
                            }
                        }
                    }
                }
            }
        }

        Ok(())
    }

    fn validate_execution_section(&self, config: &Value) -> Result<()> {
        let experiment = config.get("experiment").ok_or_else(|| {
            crate::error::Error::Validation("Missing experiment section".to_string())
        })?;

        if let Some(execution) = experiment.get("execution") {
            if let Some(steps) = execution.get("steps").and_then(|v| v.as_table()) {
                for (step_name, step) in steps {
                    if let Some(step) = step.as_str() {
                        if step.is_empty() {
                            return Err(crate::error::Error::Validation(format!(
                                "Empty command for step: {}",
                                step_name
                            )));
                        }
                    }
                }
            }
        }

        Ok(())
    }
}
