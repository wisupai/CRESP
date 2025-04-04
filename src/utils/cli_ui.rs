use anyhow::Result;
use console::{style, Term};
use dialoguer::{theme::ColorfulTheme, Confirm, Input, Password, Select};
use std::fmt::{Debug, Display};
use std::str::FromStr;

/// Get a themed instance for dialoguer components
pub fn theme() -> ColorfulTheme {
    ColorfulTheme::default()
}

/// Display a styled header with emoji
pub fn display_header(title: &str, emoji: &str) {
    let term = Term::stdout();
    let _ = term.clear_line();
    println!("\n{} {}\n", style(emoji).bold(), style(title).bold().cyan());
}

/// Display a styled success message with emoji
pub fn display_success(message: &str) {
    println!("{} {}", style("✅").green().bold(), style(message).green());
}

/// Display a styled error message with emoji
pub fn display_error(message: &str) {
    eprintln!("{} {}", style("❌").red().bold(), style(message).red());
}

/// Display a styled warning message with emoji
pub fn display_warning(message: &str) {
    eprintln!(
        "{} {}",
        style("⚠️").yellow().bold(),
        style(message).yellow()
    );
}

/// Display a styled info message with emoji
pub fn display_info(message: &str) {
    println!("{} {}", style("ℹ️").blue().bold(), style(message).blue());
}

/// Prompt for text input with validation
pub fn prompt_input<T>(prompt: &str, default: Option<T>) -> Result<T>
where
    T: Display + FromStr + Clone,
    T::Err: Debug + Display,
{
    let theme = theme();
    let input = Input::with_theme(&theme);

    let input = input.with_prompt(prompt);

    let input = if let Some(default_value) = default {
        input.default(default_value)
    } else {
        input
    };

    Ok(input.interact()?)
}

/// Prompt for password input
pub fn prompt_password(prompt: &str, confirmation: bool) -> Result<String> {
    let theme = theme();
    let password = Password::with_theme(&theme).with_prompt(prompt);

    let password = if confirmation {
        password.with_confirmation("Confirm password", "Passwords don't match")
    } else {
        password
    };

    Ok(password.interact()?)
}

/// Prompt for confirmation
pub fn prompt_confirm(prompt: &str, default: bool) -> Result<bool> {
    let theme = theme();
    Ok(Confirm::with_theme(&theme)
        .with_prompt(prompt)
        .default(default)
        .interact()?)
}

/// Prompt for selection from a list
pub fn prompt_select<T>(prompt: &str, items: &[T]) -> Result<usize>
where
    T: AsRef<str> + Display,
{
    let theme = theme();
    Ok(Select::with_theme(&theme)
        .with_prompt(prompt)
        .items(items)
        .interact()?)
}
