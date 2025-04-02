use std::io::{self, Write};
use std::path::Path;
use log::warn;
use crate::error::Result;

/// 从用户获取输入，带有可选的默认值
pub fn prompt_input(prompt: &str, default: Option<&str>) -> Result<String> {
    match default {
        Some(default_value) => print!("{} [{}]: ", prompt, default_value),
        None => print!("{}: ", prompt),
    }
    io::stdout().flush()?;
    
    let mut input = String::new();
    io::stdin().read_line(&mut input)?;
    let trimmed = input.trim().to_string();
    
    if trimmed.is_empty() && default.is_some() {
        Ok(default.unwrap().to_string())
    } else {
        Ok(trimmed)
    }
}

/// 从用户获取选择菜单中的选项
pub fn prompt_selection(prompt: &str, options: &[&str], default: Option<usize>) -> Result<usize> {
    println!("{}", prompt);
    
    for (i, option) in options.iter().enumerate() {
        println!("{}. {}", i + 1, option);
    }
    
    let default_str = default.map(|d| (d + 1).to_string());
    let input = prompt_input(
        &format!("Choice (1-{}) {}", 
            options.len(), 
            default_str.as_ref().map_or("".to_string(), |d| format!("[{}]", d))
        ),
        default_str.as_deref()
    )?;
    
    if let Ok(num) = input.parse::<usize>() {
        if num >= 1 && num <= options.len() {
            return Ok(num - 1);
        }
    }
    
    // 返回默认值或第一个选项
    Ok(default.unwrap_or(0))
}

/// 从用户获取确认 (y/n)
pub fn prompt_confirmation(prompt: &str, default: bool) -> Result<bool> {
    let default_str = if default { "Y/n" } else { "y/N" };
    let input = prompt_input(&format!("{} ({}): ", prompt, default_str), None)?;
    
    match input.to_lowercase().as_str() {
        "y" | "yes" => Ok(true),
        "n" | "no" => Ok(false),
        "" => Ok(default),
        _ => {
            warn!("Invalid input, using default: {}", default);
            Ok(default)
        }
    }
}

/// 创建目录及其父目录
pub fn create_directories(dirs: &[&str], base_dir: &Path) -> Result<()> {
    for dir in dirs {
        std::fs::create_dir_all(base_dir.join(dir))?;
    }
    Ok(())
}

/// 确保目录存在，如果已存在则提示覆盖
pub fn ensure_directory(dir_path: &Path) -> Result<bool> {
    if dir_path.exists() {
        warn!("⚠️ Directory already exists: {}", dir_path.display());
        let should_overwrite = prompt_confirmation("Do you want to overwrite it?", false)?;
        
        if should_overwrite {
            std::fs::remove_dir_all(dir_path)?;
            std::fs::create_dir_all(dir_path)?;
            return Ok(true);
        }
        Ok(false)
    } else {
        std::fs::create_dir_all(dir_path)?;
        Ok(true)
    }
}

/// 写入文件，如有必要创建父目录
pub fn write_file(path: &Path, content: &str) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, content)?;
    Ok(())
} 