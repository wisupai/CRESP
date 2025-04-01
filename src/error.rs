use thiserror::Error;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Error, Debug)]
pub enum Error {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Configuration error: {0}")]
    Config(String),

    #[error("Validation error: {0}")]
    Validation(String),

    #[error("Environment error: {0}")]
    Environment(String),

    #[allow(dead_code)]
    #[error("Dependency error: {0}")]
    Dependency(String),

    #[error("Data error: {0}")]
    Data(String),

    #[allow(dead_code)]
    #[error("Container error: {0}")]
    Container(String),

    #[error("Network error: {0}")]
    Network(#[from] reqwest::Error),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("TOML deserialization error: {0}")]
    TomlDe(#[from] toml::de::Error),

    #[error("TOML serialization error: {0}")]
    TomlSer(#[from] toml::ser::Error),

    #[allow(dead_code)]
    #[error("Command execution error: {0}")]
    Command(String),

    #[allow(dead_code)]
    #[error("Unknown error: {0}")]
    Unknown(String),
}
