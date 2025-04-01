use clap::Parser;
use log::{info, LevelFilter};

mod commands;
mod config;
mod core;
mod error;
mod utils;

use error::Result;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[arg(short, long, action = clap::ArgAction::Count)]
    verbose: u8,

    #[arg(short, long)]
    quiet: bool,

    #[command(subcommand)]
    command: commands::Command,
}

#[tokio::main]
async fn main() -> Result<()> {
    // Parse command line arguments
    let cli = Cli::parse();

    // Setup logging
    let log_level = if cli.quiet {
        LevelFilter::Error
    } else {
        match cli.verbose {
            0 => LevelFilter::Info,
            1 => LevelFilter::Debug,
            _ => LevelFilter::Trace,
        }
    };
    env_logger::Builder::new()
        .filter_level(log_level)
        .format_timestamp(None)
        .init();

    info!("🚀 Starting CRESP CLI...");

    // Execute command
    cli.command.execute().await?;

    info!("✨ CRESP CLI completed successfully");
    Ok(())
}
