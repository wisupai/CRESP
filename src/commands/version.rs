use crate::error::Result;
use crate::utils::cli_ui;
use clap::Args;

#[derive(Args, Debug)]
pub struct VersionCommand {}

impl VersionCommand {
    pub async fn execute(&self) -> Result<()> {
        let version = env!("CARGO_PKG_VERSION");
        let name = env!("CARGO_PKG_NAME");
        let authors = env!("CARGO_PKG_AUTHORS");
        let description = env!("CARGO_PKG_DESCRIPTION");

        cli_ui::display_header(&format!("{} v{}", name, version), "🔖");
        cli_ui::display_info(&format!("Description: {}", description));
        cli_ui::display_info(&format!("Authors: {}", authors));

        Ok(())
    }
}
