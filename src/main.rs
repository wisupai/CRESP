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

    // 获取 verbose 值用于日志格式化
    let verbose_level = cli.verbose;

    // 配置自定义日志格式，不显示时间戳和模块路径
    env_logger::Builder::new()
        .filter_level(log_level)
        .format(move |buf, record| {
            // 只有调试级别及以上才显示模块路径和级别
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
                // 正常用户模式：只显示消息内容，不显示模块路径和日志级别
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
            // 成功消息已由各子命令显示，这里不再重复
            Ok(())
        }
        Err(e) => {
            cli_ui::display_error(&format!("Error: {}", e));
            Err(e)
        }
    }
}
