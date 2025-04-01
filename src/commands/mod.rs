use crate::error::Result;
use clap::Subcommand;

pub mod diff;
pub mod export;
pub mod new;
pub mod reproduce;
pub mod update;
pub mod validate;
pub mod verify;

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
