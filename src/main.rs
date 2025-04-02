use clap::Parser;
use log::LevelFilter;

mod commands;
mod config;
mod core;
mod error;
mod utils;

use console::Term;
use error::Result;
use utils::cli_ui;

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

    // Show welcome message
    if !cli.quiet {
        let term = Term::stdout();
        let _ = term.clear_screen();
        cli_ui::display_header(
            "CRESP - Computational Research Environment Standardization Protocol",
            "🧪",
        );
    }

    // Execute command
    match cli.command.execute().await {
        Ok(_) => {
            // 成功消息已由各子命令显示，这里不再重复
            Ok(())
        }
        Err(e) => {
            cli_ui::display_error(&format!("Error: {}", e));
            Err(e)
        }
    }
}
