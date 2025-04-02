use crate::error::Result;
use crate::utils::cli_ui;
use std::path::Path;

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
        cli_ui::display_warning(&format!("Directory already exists: {}", dir_path.display()));
        let should_overwrite = cli_ui::prompt_confirm("Do you want to overwrite it?", false)?;

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
