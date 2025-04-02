use crate::error::Result;
use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::Read;
use std::path::Path;

pub mod cli_ui;

#[allow(dead_code)]
pub fn calculate_sha256<P: AsRef<Path>>(path: P) -> Result<String> {
    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = [0; 8192];

    loop {
        let bytes_read = file.read(&mut buffer)?;
        if bytes_read == 0 {
            break;
        }
        hasher.update(&buffer[..bytes_read]);
    }

    Ok(format!("{:x}", hasher.finalize()))
}

#[allow(dead_code)]
pub fn ensure_directory<P: AsRef<Path>>(path: P) -> Result<()> {
    if let Some(parent) = path.as_ref().parent() {
        if !parent.exists() {
            std::fs::create_dir_all(parent)?;
        }
    }
    Ok(())
}

#[allow(dead_code)]
pub fn is_executable<P: AsRef<Path>>(path: P) -> bool {
    if let Ok(metadata) = std::fs::metadata(path) {
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            return metadata.permissions().mode() & 0o111 != 0;
        }
        #[cfg(windows)]
        {
            let path = path.as_ref();
            let extension = path.extension().and_then(|s| s.to_str()).unwrap_or("");
            return extension.eq_ignore_ascii_case("exe")
                || extension.eq_ignore_ascii_case("cmd")
                || extension.eq_ignore_ascii_case("bat");
        }
    }
    false
}
