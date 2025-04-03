use clap::Parser;
use env_logger::fmt::Color;
use log::LevelFilter;
use std::io::Write;

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

    // Get verbose level for log formatting
    let verbose_level = cli.verbose;

    // Configure custom log format without timestamp and module path
    env_logger::Builder::new()
        .filter_level(log_level)
        .format(move |buf, record| {
            // Only show module path and level for debug and above
            if verbose_level >= 1 {
                let mut level_style = buf.style();
                match record.level() {
                    log::Level::Error => level_style.set_color(Color::Red).set_bold(true),
                    log::Level::Warn => level_style.set_color(Color::Yellow).set_bold(true),
                    log::Level::Info => level_style.set_color(Color::Blue),
                    log::Level::Debug => level_style.set_color(Color::Cyan),
                    log::Level::Trace => level_style.set_color(Color::White),
                };

                writeln!(
                    buf,
                    "[{}] {}",
                    level_style.value(record.level()),
                    record.args()
                )
            } else {
                // Normal user mode: only show message content without module path and log level
                writeln!(buf, "{}", record.args())
            }
        })
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
            // Success message already shown by subcommands, no need to repeat here
            Ok(())
        }
        Err(e) => {
            cli_ui::display_error(&format!("Error: {}", e));
            Err(e)
        }
    }
}
