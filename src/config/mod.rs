use crate::error::Result;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Serialize, Deserialize)]
pub struct Config {
    pub version: String,
    pub project: ProjectConfig,
    pub environment: EnvironmentConfig,
    pub data: Option<DataConfig>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ProjectConfig {
    pub name: String,
    pub description: Option<String>,
    pub authors: Option<Vec<String>>,
    pub repository: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct EnvironmentConfig {
    pub os: Option<String>,
    pub dependencies: Option<Vec<DependencyConfig>>,
    pub variables: Option<std::collections::HashMap<String, String>>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct DependencyConfig {
    pub name: String,
    pub version: String,
    pub source: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct DataConfig {
    pub datasets: Option<Vec<DatasetConfig>>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct DatasetConfig {
    pub name: String,
    pub path: PathBuf,
    pub description: Option<String>,
    pub sha256: Option<String>,
}

impl Config {
    pub fn load(path: &PathBuf) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let config = toml::from_str(&content)?;
        Ok(config)
    }

    pub fn save(&self, path: &PathBuf) -> Result<()> {
        let content = toml::to_string_pretty(self)?;
        std::fs::write(path, content)?;
        Ok(())
    }
}
