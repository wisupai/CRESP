use super::super::utils::write_file;
use crate::error::Result;
use crate::utils::cli_ui;
use std::path::Path;
use std::process::Command;
use super::super::config::{check_system_r, check_rig_available};

/// Create R project with the specified configuration
pub fn create_r_project(project_dir: &Path) -> Result<()> {
    // Check system R availability
    let (system_r, r_info) = get_r_info()?;
    let rig_available = check_rig_available()?;
    
    // Setup R environment
    let r_version = setup_r_environment(system_r, r_info, rig_available)?;
    
    cli_ui::display_info("Creating R project structure...");
    // Create basic R project structure
    let dirs = &["R", "data", "output", "tests/testthat", "docs"];

    for dir in dirs {
        std::fs::create_dir_all(project_dir.join(dir))?;
    }

    cli_ui::display_info("Generating R project files...");
    
    // Create renv.lock file
    let renv_lock = r#"{
  "R": {
    "Version": "__R_VERSION__",
    "Repositories": [
      {
        "Name": "CRAN",
        "URL": "https://cloud.r-project.org"
      }
    ]
  },
  "Packages": {
    "renv": {
      "Package": "renv",
      "Version": "1.0.3",
      "Source": "Repository",
      "Repository": "CRAN"
    },
    "testthat": {
      "Package": "testthat",
      "Version": "3.2.1",
      "Source": "Repository",
      "Repository": "CRAN"
    }
  }
}"#;
    
    // Replace R version in renv.lock
    let renv_lock = renv_lock.replace("__R_VERSION__", &r_version);
    write_file(&project_dir.join("renv.lock"), &renv_lock)?;
    
    // Create DESCRIPTION file
    let description = r#"Package: myresearch
Title: My Research Project
Version: 0.1.0
Authors@R: 
    person("Your", "Name", email = "your.email@example.com", role = c("aut", "cre"))
Description: A research project using CRESP protocol.
License: MIT + file LICENSE
Encoding: UTF-8
Roxygen: list(markdown = TRUE)
RoxygenNote: 7.2.3
Imports: 
    renv,
    testthat
Suggests:
    knitr,
    rmarkdown
RdMacros: lifecycle
Config/testthat/edition: 3
"#;
    write_file(&project_dir.join("DESCRIPTION"), description)?;

    // Create main.R file
    let main_r = r#"#' Main function
#' 
#' This is the main entry point for the research project.
#' 
#' @return NULL
#' @export
#'
#' @examples
#' main()
main <- function() {
    print("Hello, CRESP!")
    
    # Your analysis code goes here
}

if (interactive()) {
    main()
}
"#;
    write_file(&project_dir.join("R/main.R"), main_r)?;

    // Create renv setup script
    let renv_setup = r#"# renv setup script
# This script initializes renv for your project

# Install renv if it's not already installed
if (!requireNamespace("renv", quietly = TRUE)) {
  install.packages("renv")
}

# Initialize renv for this project
renv::init()

# Install dependencies from renv.lock
renv::restore()

print("R environment setup complete!")
"#;
    write_file(&project_dir.join("setup.R"), renv_setup)?;

    // Create README.md
    let project_name = project_dir
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("myresearch");

    let readme = format!(
        r#"# {}: R Research Project

This is an R research project using CRESP protocol.

## Project Structure

```
.
├── R/              # R code files
├── data/           # Data directory
├── output/         # Output directory
├── tests/          # Tests directory
├── DESCRIPTION     # Package metadata
├── renv.lock       # Package dependency lock file
└── setup.R         # Environment setup script
```

## Setup

1. Install R (recommended version {}).

{}

2. Clone this repository and change to the project directory:
```bash
git clone <repository-url>
cd {}
```

3. Setup the environment by running the setup script:
```r
source("setup.R")
```

4. Run the project:
```r
source("R/main.R")
```

## Testing

Run tests with:
```r
testthat::test_package("{}")
```
"#,
        project_name, 
        r_version,
        get_r_installation_instructions(),
        project_name,
        project_name
    );
    write_file(&project_dir.join("README.md"), &readme)?;

    // Create .gitignore
    let gitignore = r#"# R specific
.Rproj.user
.Rhistory
.RData
.Ruserdata
*.Rproj

# renv specific
renv/library/
renv/local/
renv/lock/
renv/python/
renv/staging/

# Output files
output/
*.html
*.pdf
*.png
*.jpg

# Large data files
data/**/*.csv
data/**/*.xlsx
data/**/*.rds
"#;
    write_file(&project_dir.join(".gitignore"), gitignore)?;

    // Create basic test file
    let test_file = r#"test_that("main function works", {
  # Setup test environment
  
  # Call the function (without executing side effects)
  # result <- main()
  
  # Verify results
  expect_true(TRUE)
})
"#;
    write_file(&project_dir.join("tests/testthat/test-main.R"), test_file)?;

    // Create test runner
    let test_runner = r#"library(testthat)
library(myresearch)

test_check("myresearch")
"#;
    write_file(&project_dir.join("tests/testthat.R"), test_runner)?;

    Ok(())
}

/// Get detailed information about installed R
fn get_r_info() -> Result<(Option<String>, Option<RInfo>)> {
    // Check basic R version
    let system_r = check_system_r()?;
    
    if system_r.is_none() {
        return Ok((None, None));
    }
    
    // Try to get more detailed information
    let version = system_r.as_ref().unwrap().clone();
    
    // Get R executable path
    let r_path = get_r_path()?;
    
    // Determine installation method
    let install_method = if let Some(path) = &r_path {
        determine_install_method(path)
    } else {
        "Unknown".to_string()
    };
    
    // Get R arch if possible
    let r_arch = get_r_arch(&version, &r_path)?;
    
    Ok((system_r, Some(RInfo {
        version,
        path: r_path,
        install_method,
        arch: r_arch,
    })))
}

/// Struct to hold detailed R information
#[derive(Debug, Clone)]
struct RInfo {
    version: String,
    path: Option<String>,
    install_method: String,
    arch: String,
}

/// Get R executable path
fn get_r_path() -> Result<Option<String>> {
    let cmd = if cfg!(target_os = "windows") {
        Command::new("where").arg("R.exe").output()
    } else {
        Command::new("which").arg("R").output()
    };
    
    match cmd {
        Ok(output) if output.status.success() => {
            let path = String::from_utf8_lossy(&output.stdout)
                .trim()
                .to_string();
            if !path.is_empty() {
                Ok(Some(path))
            } else {
                Ok(None)
            }
        },
        _ => Ok(None),
    }
}

/// Determine how R was installed based on its path
fn determine_install_method(path: &str) -> String {
    if cfg!(target_os = "macos") {
        if path.contains("/usr/local/bin") {
            // Could be Homebrew
            if std::path::Path::new("/usr/local/Cellar/r").exists() || 
               std::path::Path::new("/opt/homebrew/Cellar/r").exists() {
                return "Homebrew".to_string();
            }
        }
        
        if path.contains("/Library/Frameworks/R.framework") {
            return "Official installer".to_string();
        }
        
        if path.contains(".rig/") || path.contains("/.r/") {
            return "rig (R Installation Manager)".to_string();
        }
    } else if cfg!(target_os = "windows") {
        if path.contains("\\Program Files\\R\\") {
            return "Official installer".to_string();
        }
    } else {
        // Linux
        if path.contains("/usr/bin") {
            if std::path::Path::new("/etc/debian_version").exists() {
                return "apt (Debian/Ubuntu package)".to_string();
            }
            if std::path::Path::new("/etc/fedora-release").exists() {
                return "dnf (Fedora package)".to_string();
            }
            if std::path::Path::new("/etc/arch-release").exists() {
                return "pacman (Arch Linux package)".to_string();
            }
            return "System package manager".to_string();
        }
        
        if path.contains("/.rig/") || path.contains("/.r/") {
            return "rig (R Installation Manager)".to_string();
        }
    }
    
    "Unknown source".to_string()
}

/// Get R architecture
fn get_r_arch(_version: &str, _path: &Option<String>) -> Result<String> {
    // Try to run R --version to get architecture info
    let output = Command::new("R").arg("--version").output();
    
    match output {
        Ok(output) if output.status.success() => {
            let version_str = String::from_utf8_lossy(&output.stdout);
            
            // Check for architecture information
            if version_str.contains("x86_64") {
                Ok("x86_64 (64-bit)".to_string())
            } else if version_str.contains("i386") || version_str.contains("i686") {
                Ok("i386/i686 (32-bit)".to_string())
            } else if version_str.contains("aarch64") || version_str.contains("arm64") {
                Ok("ARM64".to_string())
            } else {
                // Default to platform-specific architecture
                if cfg!(target_arch = "x86_64") {
                    Ok("x86_64 (64-bit)".to_string())
                } else if cfg!(target_arch = "aarch64") {
                    Ok("ARM64".to_string())
                } else {
                    Ok("Unknown".to_string())
                }
            }
        },
        _ => {
            // Fallback to platform-specific architecture
            if cfg!(target_arch = "x86_64") {
                Ok("x86_64 (64-bit)".to_string())
            } else if cfg!(target_arch = "aarch64") {
                Ok("ARM64".to_string())
            } else {
                Ok("Unknown".to_string())
            }
        }
    }
}

/// Setup R environment by checking existing installation and selecting version
fn setup_r_environment(system_r: Option<String>, r_info: Option<RInfo>, rig_available: bool) -> Result<String> {
    cli_ui::display_header("R Configuration", "📊");
    
    // Default R version
    let default_version = "4.3.2".to_string();
    
    // Flag to track if user rejected existing installation
    let mut rejected_existing = false;
    
    if let Some(ver) = &system_r {
        cli_ui::display_info(&format!("Detected installed R version: {}", ver));
        
        // Display additional R information if available
        if let Some(info) = &r_info {
            if let Some(path) = &info.path {
                cli_ui::display_info(&format!("R location: {}", path));
            }
            cli_ui::display_info(&format!("Installation method: {}", info.install_method));
            cli_ui::display_info(&format!("Architecture: {}", info.arch));
        }
        
        // Ask if user wants to use the detected version
        let use_detected = cli_ui::prompt_confirm(&format!("Use detected R version {}?", ver), true)?;
        
        if use_detected {
            return Ok(ver.clone());
        } else {
            // User explicitly rejected the existing installation
            rejected_existing = true;
        }
    } else {
        cli_ui::display_warning("No R installation detected on your system.");
    }
    
    // Present R version options
    let r_options = vec![
        "R 4.3 (latest stable)",
        "R 4.2 (stable)",
        "R 4.1 (stable)",
        "Custom version (specify)"
    ];
    
    let selection = cli_ui::prompt_select("Select R version", &r_options)?;
    
    let selected_version = match selection {
        0 => "4.3.2".to_string(),
        1 => "4.2.3".to_string(),
        2 => "4.1.3".to_string(),
        3 => {
            // Custom version
            cli_ui::prompt_input("Enter R version (e.g., 4.4.0):", Some(default_version))?
        },
        _ => default_version,
    };
    
    // Check if the selected version is installed
    if let Some(ver) = &system_r {
        if !rejected_existing && ver.starts_with(&selected_version.split('.').take(2).collect::<Vec<_>>().join(".")) {
            cli_ui::display_success(&format!("Found compatible R version: {}", ver));
            return Ok(selected_version);
        }
    }
    
    // R is not installed or the required version is not available
    cli_ui::display_info(&format!("R {} needs to be installed on your system.", selected_version));
    
    // Present installation method options based on platform
    if cfg!(target_os = "windows") {
        let install_options = vec![
            "Download and install manually (recommended)",
            "I'll install it later"
        ];
        
        let install_selection = cli_ui::prompt_select("How would you like to install R?", &install_options)?;
        
        match install_selection {
            0 => {
                cli_ui::display_info("Please download R from: https://cloud.r-project.org/bin/windows/base/");
                cli_ui::display_info("Run the installer and follow the instructions.");
                
                // Wait for user to confirm installation
                let installed = cli_ui::prompt_confirm("Have you installed R? Press 'y' when installation is complete.", true)?;
                
                if installed {
                    // Verify installation
                    verify_r_installation()?;
                }
            }
            _ => cli_ui::display_info("You can install R later using the instructions in the README."),
        }
    } else if cfg!(target_os = "macos") {
        // macOS-specific installation options
        let mut install_options = vec![
            "Using Homebrew (recommended if you have it installed)",
            "Download and install manually",
            "I'll install it later"
        ];
        
        if rig_available {
            install_options.insert(0, "Using rig (R Installation Manager)");
        }
        
        let install_selection = cli_ui::prompt_select("How would you like to install R?", &install_options)?;
        
        match install_selection {
            0 if rig_available => {
                cli_ui::display_info(&format!("Installing R {} using rig...", selected_version));
                
                let status = Command::new("rig")
                    .arg("add")
                    .arg(&selected_version)
                    .status();
                    
                if let Ok(status) = status {
                    if status.success() {
                        cli_ui::display_success(&format!("Successfully installed R {}", selected_version));
                        
                        // Verify and get installation info
                        verify_r_installation()?;
                    } else {
                        cli_ui::display_error(&format!("Failed to install R {}", selected_version));
                    }
                } else {
                    cli_ui::display_error("Failed to run rig command");
                }
            },
            i if (rig_available && i == 1) || (!rig_available && i == 0) => {
                // Check if Homebrew is installed
                let brew_check = Command::new("brew").arg("--version").output();
                
                if brew_check.is_err() || !brew_check.unwrap().status.success() {
                    cli_ui::display_error("Homebrew not found. Please install Homebrew first:");
                    cli_ui::display_info("/bin/bash -c \"$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\"");
                    cli_ui::display_info("After installing Homebrew, try again.");
                } else {
                    cli_ui::display_info("Installing R using Homebrew...");
                    cli_ui::display_info("Running: brew install --cask r");
                    
                    let status = Command::new("brew")
                        .arg("install")
                        .arg("--cask")
                        .arg("r")
                        .status();
                        
                    if let Ok(status) = status {
                        if status.success() {
                            cli_ui::display_success("Successfully installed R using Homebrew");
                            
                            // Verify and get installation info
                            verify_r_installation()?;
                        } else {
                            cli_ui::display_error("Failed to install R using Homebrew");
                        }
                    } else {
                        cli_ui::display_error("Failed to run Homebrew command");
                    }
                }
            },
            i if (rig_available && i == 2) || (!rig_available && i == 1) => {
                cli_ui::display_info("Please download R from: https://cloud.r-project.org/bin/macosx/");
                
                // Check platform architecture to recommend the right version
                if cfg!(target_arch = "aarch64") {
                    cli_ui::display_info("Since you're on Apple Silicon (M1/M2), download the arm64 version.");
                } else {
                    cli_ui::display_info("Since you're on Intel Mac, download the x86_64 version.");
                }
                
                cli_ui::display_info("Run the installer and follow the instructions.");
                
                // Wait for user to confirm installation
                let installed = cli_ui::prompt_confirm("Have you installed R? Press 'y' when installation is complete.", true)?;
                
                if installed {
                    // Verify installation
                    verify_r_installation()?;
                }
            },
            _ => cli_ui::display_info("You can install R later using the instructions in the README."),
        }
    } else {
        // Linux-specific installation options
        let mut install_options = vec![];
        
        // Add distribution-specific options based on detection
        if is_debian_based() {
            install_options.push("Using apt (Ubuntu/Debian)");
        }
        
        if is_fedora_based() {
            install_options.push("Using dnf (Fedora/RHEL/CentOS)");
        }
        
        if is_arch_based() {
            install_options.push("Using pacman (Arch Linux)");
        }
        
        if install_options.is_empty() {
            // General options for unknown Linux distributions
            install_options.push("Using system package manager");
        }
        
        if rig_available {
            install_options.insert(0, "Using rig (R Installation Manager)");
        }
        
        // Add general options
        install_options.push("Manual installation");
        install_options.push("I'll install it later");
        
        let install_selection = cli_ui::prompt_select("How would you like to install R?", &install_options)?;
        
        if install_selection < install_options.len() - 2 {
            // Selected a specific installation method
            let selected_option = install_options[install_selection];
            
            if selected_option.contains("rig") && rig_available {
                cli_ui::display_info(&format!("Installing R {} using rig...", selected_version));
                
                let status = Command::new("rig")
                    .arg("add")
                    .arg(&selected_version)
                    .status();
                    
                if let Ok(status) = status {
                    if status.success() {
                        cli_ui::display_success(&format!("Successfully installed R {}", selected_version));
                        
                        // Verify installation
                        verify_r_installation()?;
                    } else {
                        cli_ui::display_error(&format!("Failed to install R {}", selected_version));
                    }
                } else {
                    cli_ui::display_error("Failed to run rig command");
                }
            } else if selected_option.contains("apt") {
                cli_ui::display_info("Installing R using apt...");
                cli_ui::display_warning("This will require sudo privileges. You may be prompted for your password.");
                
                // Check for sudo access
                let sudo_check = Command::new("sudo").arg("-n").arg("true").status();
                let has_sudo = sudo_check.is_ok() && sudo_check.unwrap().success();
                
                if !has_sudo {
                    cli_ui::display_warning("You might not have sudo access or sudo requires a password.");
                }
                
                let install_commands = [
                    "sudo apt update",
                    "sudo apt install --no-install-recommends software-properties-common dirmngr",
                    "wget -qO- https://cloud.r-project.org/bin/linux/ubuntu/marutter_pubkey.asc | sudo tee -a /etc/apt/trusted.gpg.d/cran_ubuntu_key.asc",
                    "sudo add-apt-repository \"deb https://cloud.r-project.org/bin/linux/ubuntu $(lsb_release -cs)-cran40/\"",
                    "sudo apt update",
                    "sudo apt install -y r-base r-base-dev"
                ];
                
                cli_ui::display_info("Please execute these commands in your terminal:");
                for cmd in &install_commands {
                    cli_ui::display_info(cmd);
                }
                
                let proceed = cli_ui::prompt_confirm("Do you want to execute these commands now?", true)?;
                
                if proceed {
                    for cmd in &install_commands {
                        cli_ui::display_info(&format!("Running: {}", cmd));
                        
                        let status = Command::new("sh")
                            .arg("-c")
                            .arg(cmd)
                            .status();
                            
                        match status {
                            Ok(exit_status) => {
                                if !exit_status.success() {
                                    cli_ui::display_error(&format!("Command failed: {}", cmd));
                                    break;
                                }
                            },
                            Err(e) => {
                                cli_ui::display_error(&format!("Failed to execute command: {} ({})", cmd, e));
                                break;
                            }
                        }
                    }
                    
                    // Verify installation
                    verify_r_installation()?;
                }
            } else if selected_option.contains("dnf") {
                cli_ui::display_info("Installing R using dnf...");
                cli_ui::display_warning("This will require sudo privileges. You may be prompted for your password.");
                
                let cmd = "sudo dnf install -y R";
                cli_ui::display_info(&format!("Running: {}", cmd));
                
                let proceed = cli_ui::prompt_confirm("Do you want to execute this command now?", true)?;
                
                if proceed {
                    let status = Command::new("sh")
                        .arg("-c")
                        .arg(cmd)
                        .status();
                        
                    if let Ok(status) = status {
                        if status.success() {
                            cli_ui::display_success("Successfully installed R");
                            
                            // Verify installation
                            verify_r_installation()?;
                        } else {
                            cli_ui::display_error("Failed to install R using dnf");
                        }
                    } else {
                        cli_ui::display_error("Failed to execute dnf command");
                    }
                }
            } else if selected_option.contains("pacman") {
                cli_ui::display_info("Installing R using pacman...");
                cli_ui::display_warning("This will require sudo privileges. You may be prompted for your password.");
                
                let cmd = "sudo pacman -S --noconfirm r";
                cli_ui::display_info(&format!("Running: {}", cmd));
                
                let proceed = cli_ui::prompt_confirm("Do you want to execute this command now?", true)?;
                
                if proceed {
                    let status = Command::new("sh")
                        .arg("-c")
                        .arg(cmd)
                        .status();
                        
                    if let Ok(status) = status {
                        if status.success() {
                            cli_ui::display_success("Successfully installed R");
                            
                            // Verify installation
                            verify_r_installation()?;
                        } else {
                            cli_ui::display_error("Failed to install R using pacman");
                        }
                    } else {
                        cli_ui::display_error("Failed to execute pacman command");
                    }
                }
            } else if selected_option.contains("system package") {
                cli_ui::display_info("Please enter the appropriate command for your system to install R:");
                let cmd: String = cli_ui::prompt_input("Installation command:", Some("sudo apt install r-base".to_string()))?;
                
                if !cmd.trim().is_empty() {
                    let proceed = cli_ui::prompt_confirm(&format!("Run: {}?", cmd), true)?;
                    
                    if proceed {
                        cli_ui::display_info(&format!("Running: {}", cmd));
                        
                        let status = Command::new("sh")
                            .arg("-c")
                            .arg(&cmd)
                            .status();
                            
                        if let Ok(status) = status {
                            if status.success() {
                                cli_ui::display_success("Command executed successfully");
                                
                                // Verify installation
                                verify_r_installation()?;
                            } else {
                                cli_ui::display_error("Command returned non-zero exit status");
                            }
                        } else {
                            cli_ui::display_error("Failed to execute command");
                        }
                    }
                }
            }
        } else if install_selection == install_options.len() - 2 {
            // Manual installation selected
            cli_ui::display_info("Please follow the instructions for your distribution:");
            cli_ui::display_info("https://cloud.r-project.org/bin/linux/");
            
            // Wait for user to confirm installation
            let installed = cli_ui::prompt_confirm("Have you installed R? Press 'y' when installation is complete.", false)?;
            
            if installed {
                // Verify installation
                verify_r_installation()?;
            }
        } else {
            // Install later selected
            cli_ui::display_info("You can install R later using the instructions in the README.");
        }
    }
    
    Ok(selected_version)
}

/// Get platform-specific R installation instructions
fn get_r_installation_instructions() -> String {
    if cfg!(target_os = "windows") {
        r#"Windows Installation:
1. Download R from https://cloud.r-project.org/bin/windows/base/
2. Run the installer and follow the instructions
3. Optionally install RStudio from https://posit.co/download/rstudio-desktop/"#.to_string()
    } else if cfg!(target_os = "macos") {
        let arch_note = if cfg!(target_arch = "aarch64") {
            "Note: For Apple Silicon (M1/M2) Macs, make sure to download the arm64 version."
        } else {
            "Note: For Intel Macs, make sure to download the x86_64 version."
        };
        
        format!(
            r#"macOS Installation:
1. Using Homebrew (recommended):
   ```bash
   brew install --cask r
   ```

2. Alternative: Download from https://cloud.r-project.org/bin/macosx/
   {}

3. Consider installing R Installation Manager (rig) for managing multiple R versions:
   ```bash
   brew install rig
   ```"#,
            arch_note
        )
    } else {
        r#"Linux Installation:
1. Ubuntu/Debian:
   ```bash
   # Add CRAN repository
   sudo apt update
   sudo apt install --no-install-recommends software-properties-common dirmngr
   wget -qO- https://cloud.r-project.org/bin/linux/ubuntu/marutter_pubkey.asc | sudo tee -a /etc/apt/trusted.gpg.d/cran_ubuntu_key.asc
   sudo add-apt-repository "deb https://cloud.r-project.org/bin/linux/ubuntu $(lsb_release -cs)-cran40/"
   
   # Install R
   sudo apt update
   sudo apt install r-base r-base-dev
   ```

2. Fedora/RHEL/CentOS:
   ```bash
   sudo dnf install R
   ```

3. Arch Linux:
   ```bash
   sudo pacman -S r
   ```

4. Consider installing R Installation Manager (rig) for managing multiple R versions:
   https://github.com/r-lib/rig#installation"#.to_string()
    }
}

/// Verify that R is installed and get version information
fn verify_r_installation() -> Result<()> {
    cli_ui::display_info("Verifying R installation...");
    
    // Try to get R version
    let output = Command::new("R").arg("--version").output();
    
    match output {
        Ok(output) if output.status.success() => {
            let version_output = String::from_utf8_lossy(&output.stdout);
            let first_line = version_output.lines().next().unwrap_or("Unknown");
            
            cli_ui::display_success(&format!("R is successfully installed: {}", first_line));
            
            // Get installation path
            if let Ok(Some(path)) = get_r_path() {
                cli_ui::display_info(&format!("R is installed at: {}", path));
                
                // Try to detect installation method
                let install_method = determine_install_method(&path);
                cli_ui::display_info(&format!("Installation method: {}", install_method));
            }
            
            // Get architecture information
            if version_output.contains("x86_64") {
                cli_ui::display_info("Architecture: x86_64 (64-bit)");
            } else if version_output.contains("i386") || version_output.contains("i686") {
                cli_ui::display_info("Architecture: i386/i686 (32-bit)");
            } else if version_output.contains("aarch64") || version_output.contains("arm64") {
                cli_ui::display_info("Architecture: ARM64");
            }
            
            Ok(())
        },
        _ => {
            cli_ui::display_warning("Could not verify R installation. Make sure R is properly installed and in your PATH.");
            cli_ui::display_info("You may need to restart your terminal or add R to your PATH manually.");
            Ok(())
        }
    }
}

/// Check if system is Debian/Ubuntu based
fn is_debian_based() -> bool {
    std::path::Path::new("/etc/debian_version").exists() || 
    std::path::Path::new("/etc/apt").exists() ||
    Command::new("apt").arg("--version").status().is_ok()
}

/// Check if system is Fedora/RHEL/CentOS based
fn is_fedora_based() -> bool {
    std::path::Path::new("/etc/redhat-release").exists() || 
    std::path::Path::new("/etc/fedora-release").exists() ||
    Command::new("dnf").arg("--version").status().is_ok()
}

/// Check if system is Arch based
fn is_arch_based() -> bool {
    std::path::Path::new("/etc/arch-release").exists() ||
    Command::new("pacman").arg("--version").status().is_ok()
}
