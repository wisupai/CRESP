use anyhow::Result;
use console::{style, Term};
use dialoguer::{theme::ColorfulTheme, Confirm, Input, Password, Select};
use std::fmt::{Debug, Display};
use std::str::FromStr;

use crate::utils::validation::Validator;

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
    println!("{} {}", style("✓").green().bold(), style(message).green());
}

/// Display a styled error message with emoji
pub fn display_error(message: &str) {
    eprintln!("{} {}", style("x").red().bold(), style(message).red());
}

/// Display a styled warning message with emoji
pub fn display_warning(message: &str) {
    eprintln!("{} {}", style("!").yellow().bold(), style(message).yellow());
}

/// Display a styled info message with emoji
pub fn display_info(message: &str) {
    println!("{}", style(message).blue());
}

/// Display a styled debug message
pub fn display_debug(message: &str) {
    println!("{}", style(message).dim());
}

/// Display a simple message without styling
pub fn display_message(message: &str) {
    println!("{}", message);
}

/// Display a progress message
pub fn display_progress(step: &str, message: &str) {
    println!("{} {}", style(step).cyan().bold(), message);
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

/// Prompt for text input with custom validation
pub fn prompt_input_with_validation<T, V>(
    prompt: &str,
    default: Option<T>,
    validator: V,
    error_msg: &str,
) -> Result<T>
where
    T: Display + FromStr + Clone,
    T::Err: Debug + Display,
    V: Validator<T>,
{
    loop {
        let value = prompt_input(prompt, default.clone())?;
        let (is_valid, message) = validator.validate(&value);
        if is_valid {
            return Ok(value);
        }
        display_warning(&message);
        display_info(error_msg);
    }
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
