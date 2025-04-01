use crate::config::Config;
use crate::error::Result;
use std::path::PathBuf;

pub struct Core {
    config: Config,
    config_path: PathBuf,
}

impl Core {
    pub fn new(config_path: PathBuf) -> Result<Self> {
        let config = Config::load(&config_path)?;
        Ok(Self {
            config,
            config_path,
        })
    }

    pub fn config(&self) -> &Config {
        &self.config
    }

    pub fn config_mut(&mut self) -> &mut Config {
        &mut self.config
    }

    pub fn save(&self) -> Result<()> {
        self.config.save(&self.config_path)
    }
} 