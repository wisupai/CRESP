use clap::Subcommand;
use crate::error::Result;

pub mod new;
pub mod validate;
pub mod verify;
pub mod export;
pub mod reproduce;
pub mod diff;
pub mod update;

#[derive(Subcommand, Debug)]
pub enum Command {
    New(new::NewCommand),
    Validate(validate::ValidateCommand),
    Verify(verify::VerifyCommand),
    Export(export::ExportCommand),
    Reproduce(reproduce::ReproduceCommand),
    Diff(diff::DiffCommand),
    Update(update::UpdateCommand),
}

impl Command {
    pub async fn execute(&self) -> Result<()> {
        match self {
            Command::New(cmd) => cmd.execute().await,
            Command::Validate(cmd) => cmd.execute().await,
            Command::Verify(cmd) => cmd.execute().await,
            Command::Export(cmd) => cmd.execute().await,
            Command::Reproduce(cmd) => cmd.execute().await,
            Command::Diff(cmd) => cmd.execute().await,
            Command::Update(cmd) => cmd.execute().await,
        }
    }
} 