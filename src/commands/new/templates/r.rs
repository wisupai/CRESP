use super::super::config::{check_rig_available, check_system_r};
use super::super::utils::write_file;
use crate::error::Result;
use crate::utils::cli_ui;
use std::path::Path;
use std::process::Command;

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

    Ok((
        system_r,
        Some(RInfo {
            _version: version,
            path: r_path,
            install_method,
            arch: r_arch,
        }),
    ))
}

/// Struct to hold detailed R information
#[derive(Debug, Clone)]
struct RInfo {
    _version: String,
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
            let path = String::from_utf8_lossy(&output.stdout).trim().to_string();
            if !path.is_empty() {
                Ok(Some(path))
            } else {
                Ok(None)
            }
        }
        _ => Ok(None),
    }
}

/// Determine how R was installed based on its path
fn determine_install_method(path: &str) -> String {
    if cfg!(target_os = "macos") {
        if path.contains("/usr/local/bin") {
            // Could be Homebrew
            if std::path::Path::new("/usr/local/Cellar/r").exists()
                || std::path::Path::new("/opt/homebrew/Cellar/r").exists()
            {
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
        }
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
fn setup_r_environment(system_r: Option<String>, r_info: Option<RInfo>, _: bool) -> Result<String> {
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
        let use_detected =
            cli_ui::prompt_confirm(&format!("Use detected R version {}?", ver), true)?;

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
        "Custom version (specify)",
    ];

    let selection = cli_ui::prompt_select("Select R version", &r_options)?;

    let selected_version = match selection {
        0 => "4.3.2".to_string(),
        1 => "4.2.3".to_string(),
        2 => "4.1.3".to_string(),
        3 => {
            // Custom version
            cli_ui::prompt_input("Enter R version (e.g., 4.4.0):", Some(default_version))?
        }
        _ => default_version,
    };

    // Check if the selected version is installed
    if let Some(ver) = &system_r {
        if !rejected_existing
            && ver.starts_with(
                &selected_version
                    .split('.')
                    .take(2)
                    .collect::<Vec<_>>()
                    .join("."),
            )
        {
            cli_ui::display_success(&format!("Found compatible R version: {}", ver));
            return Ok(selected_version);
        }
    }

    // R is not installed or the required version is not available
    cli_ui::display_info(&format!(
        "R {} needs to be installed on your system.",
        selected_version
    ));

    // Present installation method options based on platform
    if cfg!(target_os = "windows") {
        let mut install_options = vec![];

        // Check if rig is available
        let rig_available = check_rig_available()?;

        // Check if package managers are available
        let scoop_available = Command::new("scoop")
            .arg("--version")
            .status()
            .map_or(false, |status| status.success());
        let choco_available = Command::new("choco")
            .arg("--version")
            .status()
            .map_or(false, |status| status.success());
        let winget_available = Command::new("winget")
            .arg("--version")
            .status()
            .map_or(false, |status| status.success());

        // Add options with installation status
        if rig_available {
            install_options
                .push("Using rig (R Installation Manager) (already installed, recommended)");
        } else {
            install_options.push("Install and use rig (R Installation Manager) (recommended)");

            // Add specific installation methods if available
            if winget_available {
                install_options.push("Install rig using WinGet");
            }
            if choco_available {
                install_options.push("Install rig using Chocolatey");
            }
            if scoop_available {
                install_options.push("Install rig using Scoop");
            }
        }

        install_options.push("Download and install manually");
        install_options.push("I'll install it later");

        let install_selection =
            cli_ui::prompt_select("How would you like to install R?", &install_options)?;

        // Calculate the offset for the manual and "install later" options
        let manual_option_idx = if rig_available {
            1
        } else {
            1 + (winget_available as usize)
                + (choco_available as usize)
                + (scoop_available as usize)
        };

        if install_selection == 0 {
            // Using or installing rig
            if rig_available {
                // Rig is already installed
                cli_ui::display_info(&format!("Installing R {} using rig...", selected_version));

                let status = Command::new("rig")
                    .arg("add")
                    .arg(&selected_version)
                    .status();

                if let Ok(status) = status {
                    if status.success() {
                        cli_ui::display_success(&format!(
                            "Successfully installed R {}",
                            selected_version
                        ));

                        // Verify installation
                        verify_r_installation()?;
                    } else {
                        cli_ui::display_error(&format!("Failed to install R {}", selected_version));
                    }
                } else {
                    cli_ui::display_error("Failed to run rig command");
                }
            } else {
                // Need to install rig first using the installer
                cli_ui::display_info("To install rig on Windows:");
                cli_ui::display_info(
                    "1. Download the latest release from https://github.com/r-lib/rig/releases",
                );
                cli_ui::display_info("2. Run the installer and follow the instructions");
                cli_ui::display_info(
                    "3. You might need to restart your terminal after installation",
                );

                let installed = cli_ui::prompt_confirm(
                    "Have you installed rig? Press 'y' when installation is complete.",
                    true,
                )?;

                if installed {
                    // Now use rig to install R
                    cli_ui::display_info(&format!(
                        "Installing R {} using rig...",
                        selected_version
                    ));

                    let status = Command::new("rig")
                        .arg("add")
                        .arg(&selected_version)
                        .status();

                    if let Ok(status) = status {
                        if status.success() {
                            cli_ui::display_success(&format!(
                                "Successfully installed R {}",
                                selected_version
                            ));

                            // Verify installation
                            verify_r_installation()?;
                        } else {
                            cli_ui::display_error(&format!(
                                "Failed to install R {}",
                                selected_version
                            ));
                        }
                    } else {
                        cli_ui::display_error("Failed to run rig command. Make sure it's properly installed and restart your terminal if needed.");
                    }
                }
            }
        } else if !rig_available && install_selection > 0 && install_selection < manual_option_idx {
            // Handle rig installation via package managers
            let mut cmd = String::new();
            let mut mgr_name = String::new();

            // Determine which package manager to use
            let mut idx = 1;
            if winget_available {
                if install_selection == idx {
                    cmd = "winget install posit.rig".to_string();
                    mgr_name = "WinGet".to_string();
                }
                idx += 1;
            }

            if choco_available && cmd.is_empty() {
                if install_selection == idx {
                    cmd = "choco install rig".to_string();
                    mgr_name = "Chocolatey".to_string();
                }
                idx += 1;
            }

            if scoop_available && cmd.is_empty() {
                if install_selection == idx {
                    cmd = "scoop bucket add r-bucket https://github.com/cderv/r-bucket.git && scoop install rig".to_string();
                    mgr_name = "Scoop".to_string();
                }
            }

            if !cmd.is_empty() {
                cli_ui::display_info(&format!("Installing rig using {}...", mgr_name));
                cli_ui::display_info(&format!("Running: {}", cmd));

                let status = Command::new("cmd").args(&["/C", &cmd]).status();

                if let Ok(status) = status {
                    if status.success() {
                        cli_ui::display_success(&format!(
                            "Successfully installed rig using {}",
                            mgr_name
                        ));

                        // Now use rig to install R (might need to restart the terminal)
                        cli_ui::display_info("You might need to restart your terminal to use rig.");
                        let continue_with_rig = cli_ui::prompt_confirm(
                            "Do you want to try using rig now to install R?",
                            true,
                        )?;

                        if continue_with_rig {
                            cli_ui::display_info(&format!(
                                "Installing R {} using rig...",
                                selected_version
                            ));

                            let status = Command::new("rig")
                                .arg("add")
                                .arg(&selected_version)
                                .status();

                            if let Ok(status) = status {
                                if status.success() {
                                    cli_ui::display_success(&format!(
                                        "Successfully installed R {}",
                                        selected_version
                                    ));

                                    // Verify installation
                                    verify_r_installation()?;
                                } else {
                                    cli_ui::display_error(&format!("Failed to install R {}. You might need to restart your terminal first.", selected_version));
                                }
                            } else {
                                cli_ui::display_error("Failed to run rig command. You might need to restart your terminal first.");
                            }
                        }
                    } else {
                        cli_ui::display_error(&format!("Failed to install rig using {}", mgr_name));
                    }
                } else {
                    cli_ui::display_error(&format!("Failed to execute {} command", mgr_name));
                }
            }
        } else if install_selection == manual_option_idx {
            // Manual installation
            cli_ui::display_info(
                "Please download R from: https://cloud.r-project.org/bin/windows/base/",
            );
            cli_ui::display_info("Run the installer and follow the instructions.");

            // Wait for user to confirm installation
            let installed = cli_ui::prompt_confirm(
                "Have you installed R? Press 'y' when installation is complete.",
                true,
            )?;

            if installed {
                // Verify installation
                verify_r_installation()?;
            }
        } else {
            // Install later
            cli_ui::display_info("You can install R later using the instructions in the README.");
        }
    } else if cfg!(target_os = "macos") {
        // macOS-specific installation options
        let mut install_options = vec![];

        // Check if rig is available
        let rig_available = check_rig_available()?;

        // Check if Homebrew is installed
        let brew_available = Command::new("brew")
            .arg("--version")
            .status()
            .map_or(false, |status| status.success());

        // Add options with installation status
        if rig_available {
            install_options
                .push("Using rig (R Installation Manager) (already installed, recommended)");
        } else {
            install_options.push("Install and use rig (R Installation Manager) (recommended)");
        }

        if brew_available {
            install_options.push("Using Homebrew (already installed)");
        } else {
            install_options.push("Using Homebrew");
        }

        install_options.push("Download and install manually");
        install_options.push("I'll install it later");

        let install_selection =
            cli_ui::prompt_select("How would you like to install R?", &install_options)?;

        match install_selection {
            0 => {
                if rig_available {
                    // Rig is already installed
                    cli_ui::display_info(&format!(
                        "Installing R {} using rig...",
                        selected_version
                    ));

                    let status = Command::new("rig")
                        .arg("add")
                        .arg(&selected_version)
                        .status();

                    if let Ok(status) = status {
                        if status.success() {
                            cli_ui::display_success(&format!(
                                "Successfully installed R {}",
                                selected_version
                            ));

                            // Verify and get installation info
                            verify_r_installation()?;
                        } else {
                            cli_ui::display_error(&format!(
                                "Failed to install R {}",
                                selected_version
                            ));
                        }
                    } else {
                        cli_ui::display_error("Failed to run rig command");
                    }
                } else {
                    // Install rig first
                    cli_ui::display_info("Installing rig (R Installation Manager)...");

                    if brew_available {
                        // Install using Homebrew since it's available
                        cli_ui::display_info("Installing rig using Homebrew...");

                        // First add the tap if needed
                        let tap_status = Command::new("brew").arg("tap").arg("r-lib/rig").status();

                        if let Ok(status) = tap_status {
                            if status.success() {
                                // Install rig
                                let install_status = Command::new("brew")
                                    .arg("install")
                                    .arg("--cask")
                                    .arg("rig")
                                    .status();

                                if let Ok(status) = install_status {
                                    if status.success() {
                                        cli_ui::display_success("Successfully installed rig");

                                        // Now use rig to install R
                                        cli_ui::display_info(&format!(
                                            "Installing R {} using rig...",
                                            selected_version
                                        ));

                                        let status = Command::new("rig")
                                            .arg("add")
                                            .arg(&selected_version)
                                            .status();

                                        if let Ok(status) = status {
                                            if status.success() {
                                                cli_ui::display_success(&format!(
                                                    "Successfully installed R {}",
                                                    selected_version
                                                ));

                                                // Verify and get installation info
                                                verify_r_installation()?;
                                            } else {
                                                cli_ui::display_error(&format!(
                                                    "Failed to install R {}",
                                                    selected_version
                                                ));
                                            }
                                        }
                                    } else {
                                        cli_ui::display_error("Failed to install rig");
                                        cli_ui::display_info("Please install rig manually from https://github.com/r-lib/rig/releases");
                                    }
                                }
                            } else {
                                cli_ui::display_error("Failed to add r-lib/rig tap");
                            }
                        }
                    } else {
                        // No Homebrew, provide manual installation instructions
                        cli_ui::display_info("To install rig on macOS:");
                        cli_ui::display_info("1. Download the latest release from https://github.com/r-lib/rig/releases");
                        cli_ui::display_info(
                            "2. Install it the usual way by opening the downloaded file",
                        );

                        let installed = cli_ui::prompt_confirm(
                            "Have you installed rig? Press 'y' when installation is complete.",
                            true,
                        )?;

                        if installed {
                            // Now use rig to install R
                            cli_ui::display_info(&format!(
                                "Installing R {} using rig...",
                                selected_version
                            ));

                            let status = Command::new("rig")
                                .arg("add")
                                .arg(&selected_version)
                                .status();

                            if let Ok(status) = status {
                                if status.success() {
                                    cli_ui::display_success(&format!(
                                        "Successfully installed R {}",
                                        selected_version
                                    ));

                                    // Verify and get installation info
                                    verify_r_installation()?;
                                } else {
                                    cli_ui::display_error("Failed to run rig command. Make sure it's properly installed and in your PATH.");
                                }
                            } else {
                                cli_ui::display_error("Failed to run rig command. Make sure it's properly installed and in your PATH.");
                            }
                        }
                    }
                }
            }
            1 => {
                // Using Homebrew
                if !brew_available {
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
            }
            2 => {
                // Manual installation
                cli_ui::display_info(
                    "Please download R from: https://cloud.r-project.org/bin/macosx/",
                );

                // Check platform architecture to recommend the right version
                if cfg!(target_arch = "aarch64") {
                    cli_ui::display_info(
                        "Since you're on Apple Silicon (M1/M2), download the arm64 version.",
                    );
                } else {
                    cli_ui::display_info("Since you're on Intel Mac, download the x86_64 version.");
                }

                cli_ui::display_info("Run the installer and follow the instructions.");

                // Wait for user to confirm installation
                let installed = cli_ui::prompt_confirm(
                    "Have you installed R? Press 'y' when installation is complete.",
                    true,
                )?;

                if installed {
                    // Verify installation
                    verify_r_installation()?;
                }
            }
            _ => cli_ui::display_info(
                "You can install R later using the instructions in the README.",
            ),
        }
    } else {
        // Linux-specific installation options
        let mut install_options = vec![];

        // Check if rig is available
        let rig_available = check_rig_available()?;

        // Add distribution-specific options based on detection
        let is_debian = is_debian_based();
        let is_fedora = is_fedora_based();
        let is_arch = is_arch_based();
        let has_sudo = Command::new("sudo")
            .arg("-n")
            .arg("true")
            .status()
            .map_or(false, |status| status.success());

        // Add rig options
        if rig_available {
            install_options
                .push("Using rig (R Installation Manager) (already installed, recommended)");
        } else {
            install_options.push("Install and use rig (R Installation Manager) (recommended)");
        }

        // Add distribution-specific options
        if is_debian {
            install_options.push("Using apt (Ubuntu/Debian)");
        }

        if is_fedora {
            install_options.push("Using dnf (Fedora/RHEL/CentOS)");
        }

        if is_arch {
            install_options.push("Using pacman (Arch Linux)");
        }

        if install_options.len() <= 1 {
            // General options for unknown Linux distributions
            install_options.push("Using system package manager");
        }

        // Add general options
        install_options.push("Manual installation");
        install_options.push("I'll install it later");

        let install_selection =
            cli_ui::prompt_select("How would you like to install R?", &install_options)?;

        if install_selection == 0 {
            // Using or installing rig
            if rig_available {
                // Rig is already installed
                cli_ui::display_info(&format!("Installing R {} using rig...", selected_version));

                let status = Command::new("rig")
                    .arg("add")
                    .arg(&selected_version)
                    .status();

                if let Ok(status) = status {
                    if status.success() {
                        cli_ui::display_success(&format!(
                            "Successfully installed R {}",
                            selected_version
                        ));

                        // Verify installation
                        verify_r_installation()?;
                    } else {
                        cli_ui::display_error(&format!("Failed to install R {}", selected_version));
                    }
                } else {
                    cli_ui::display_error("Failed to run rig command");
                }
            } else {
                // Install rig first
                cli_ui::display_info("Installing rig (R Installation Manager)...");

                if is_debian {
                    // Install rig using apt for Debian/Ubuntu
                    if !has_sudo {
                        cli_ui::display_warning(
                            "You might not have sudo access or sudo requires a password.",
                        );
                        cli_ui::display_warning("The following commands require sudo privileges:");
                    }

                    let rig_install_commands = [
                        "curl -L https://rig.r-pkg.org/deb/rig.gpg -o /tmp/rig.gpg",
                        "sudo mv /tmp/rig.gpg /etc/apt/trusted.gpg.d/rig.gpg",
                        "echo \"deb http://rig.r-pkg.org/deb rig main\" | sudo tee /etc/apt/sources.list.d/rig.list",
                        "sudo apt update",
                        "sudo apt install r-rig"
                    ];

                    cli_ui::display_info(
                        "To install rig on Debian/Ubuntu, run the following commands:",
                    );
                    for cmd in &rig_install_commands {
                        cli_ui::display_info(cmd);
                    }

                    let proceed =
                        cli_ui::prompt_confirm("Do you want to execute these commands now?", true)?;

                    if proceed {
                        for cmd in &rig_install_commands {
                            cli_ui::display_info(&format!("Running: {}", cmd));

                            let status = Command::new("sh").arg("-c").arg(cmd).status();

                            match status {
                                Ok(exit_status) => {
                                    if !exit_status.success() {
                                        cli_ui::display_error(&format!("Command failed: {}", cmd));
                                        break;
                                    }
                                }
                                Err(e) => {
                                    cli_ui::display_error(&format!(
                                        "Failed to execute command: {} ({})",
                                        cmd, e
                                    ));
                                    break;
                                }
                            }
                        }

                        // Check if rig was installed
                        let rig_check = Command::new("rig").arg("--version").status();
                        if rig_check.map_or(false, |status| status.success()) {
                            cli_ui::display_success("Successfully installed rig");

                            // Now use rig to install R
                            cli_ui::display_info(&format!(
                                "Installing R {} using rig...",
                                selected_version
                            ));

                            let status = Command::new("rig")
                                .arg("add")
                                .arg(&selected_version)
                                .status();

                            if let Ok(status) = status {
                                if status.success() {
                                    cli_ui::display_success(&format!(
                                        "Successfully installed R {}",
                                        selected_version
                                    ));

                                    // Verify installation
                                    verify_r_installation()?;
                                } else {
                                    cli_ui::display_error(&format!(
                                        "Failed to install R {}",
                                        selected_version
                                    ));
                                }
                            }
                        } else {
                            cli_ui::display_error(
                                "Failed to install rig. Try manual installation.",
                            );
                        }
                    }
                } else if is_fedora || is_arch {
                    // RPM/Arch-based distros
                    let arch_cmd = if cfg!(target_arch = "aarch64") {
                        "aarch64"
                    } else {
                        "x86_64"
                    };

                    let rig_cmd = if is_fedora {
                        format!("sudo yum install -y https://github.com/r-lib/rig/releases/download/latest/r-rig-latest-1.{}.rpm", arch_cmd)
                    } else {
                        // Arch
                        format!("sudo zypper install -y --allow-unsigned-rpm https://github.com/r-lib/rig/releases/download/latest/r-rig-latest-1.{}.rpm", arch_cmd)
                    };

                    cli_ui::display_info(&format!("To install rig, run: {}", rig_cmd));

                    let proceed =
                        cli_ui::prompt_confirm("Do you want to execute this command now?", true)?;

                    if proceed {
                        cli_ui::display_info(&format!("Running: {}", rig_cmd));

                        let status = Command::new("sh").arg("-c").arg(&rig_cmd).status();

                        if let Ok(status) = status {
                            if status.success() {
                                cli_ui::display_success("Successfully installed rig");

                                // Now use rig to install R
                                cli_ui::display_info(&format!(
                                    "Installing R {} using rig...",
                                    selected_version
                                ));

                                let status = Command::new("rig")
                                    .arg("add")
                                    .arg(&selected_version)
                                    .status();

                                if let Ok(status) = status {
                                    if status.success() {
                                        cli_ui::display_success(&format!(
                                            "Successfully installed R {}",
                                            selected_version
                                        ));

                                        // Verify installation
                                        verify_r_installation()?;
                                    } else {
                                        cli_ui::display_error(&format!(
                                            "Failed to install R {}",
                                            selected_version
                                        ));
                                    }
                                }
                            } else {
                                cli_ui::display_error("Failed to install rig");
                            }
                        } else {
                            cli_ui::display_error("Failed to execute command");
                        }
                    }
                } else {
                    // Generic Linux - use tarball method
                    let arch_cmd = if cfg!(target_arch = "aarch64") {
                        "aarch64"
                    } else {
                        "x86_64"
                    };

                    let rig_cmd = format!("curl -Ls https://github.com/r-lib/rig/releases/download/latest/rig-linux-{}-latest.tar.gz | sudo tar xz -C /usr/local", arch_cmd);

                    cli_ui::display_info("To install rig on your Linux distribution:");
                    cli_ui::display_info(&format!("Run: {}", rig_cmd));

                    let proceed =
                        cli_ui::prompt_confirm("Do you want to execute this command now?", true)?;

                    if proceed {
                        cli_ui::display_info(&format!("Running: {}", rig_cmd));

                        let status = Command::new("sh").arg("-c").arg(&rig_cmd).status();

                        if let Ok(status) = status {
                            if status.success() {
                                cli_ui::display_success("Successfully installed rig");

                                // Now use rig to install R
                                cli_ui::display_info(&format!(
                                    "Installing R {} using rig...",
                                    selected_version
                                ));

                                let status = Command::new("rig")
                                    .arg("add")
                                    .arg(&selected_version)
                                    .status();

                                if let Ok(status) = status {
                                    if status.success() {
                                        cli_ui::display_success(&format!(
                                            "Successfully installed R {}",
                                            selected_version
                                        ));

                                        // Verify installation
                                        verify_r_installation()?;
                                    } else {
                                        cli_ui::display_error(&format!(
                                            "Failed to install R {}",
                                            selected_version
                                        ));
                                    }
                                } else {
                                    cli_ui::display_error(
                                        "Failed to run rig command. Make sure it's in your PATH.",
                                    );
                                }
                            } else {
                                cli_ui::display_error("Failed to install rig");
                            }
                        } else {
                            cli_ui::display_error("Failed to execute command");
                        }
                    }
                }
            }
        } else if (is_debian && install_selection == 1)
            || (!is_debian && is_fedora && install_selection == 1)
            || (!is_debian && !is_fedora && is_arch && install_selection == 1)
            || (!is_debian && !is_fedora && !is_arch && install_selection == 1)
        {
            // Using distribution package manager
            if is_debian {
                cli_ui::display_info("Installing R using apt...");
                cli_ui::display_warning(
                    "This will require sudo privileges. You may be prompted for your password.",
                );

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

                let proceed =
                    cli_ui::prompt_confirm("Do you want to execute these commands now?", true)?;

                if proceed {
                    for cmd in &install_commands {
                        cli_ui::display_info(&format!("Running: {}", cmd));

                        let status = Command::new("sh").arg("-c").arg(cmd).status();

                        match status {
                            Ok(exit_status) => {
                                if !exit_status.success() {
                                    cli_ui::display_error(&format!("Command failed: {}", cmd));
                                    break;
                                }
                            }
                            Err(e) => {
                                cli_ui::display_error(&format!(
                                    "Failed to execute command: {} ({})",
                                    cmd, e
                                ));
                                break;
                            }
                        }
                    }

                    // Verify installation
                    verify_r_installation()?;
                }
            } else if is_fedora {
                cli_ui::display_info("Installing R using dnf...");
                cli_ui::display_warning(
                    "This will require sudo privileges. You may be prompted for your password.",
                );

                let cmd = "sudo dnf install -y R";
                cli_ui::display_info(&format!("Running: {}", cmd));

                let proceed =
                    cli_ui::prompt_confirm("Do you want to execute this command now?", true)?;

                if proceed {
                    let status = Command::new("sh").arg("-c").arg(cmd).status();

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
            } else if is_arch {
                cli_ui::display_info("Installing R using pacman...");
                cli_ui::display_warning(
                    "This will require sudo privileges. You may be prompted for your password.",
                );

                let cmd = "sudo pacman -S --noconfirm r";
                cli_ui::display_info(&format!("Running: {}", cmd));

                let proceed =
                    cli_ui::prompt_confirm("Do you want to execute this command now?", true)?;

                if proceed {
                    let status = Command::new("sh").arg("-c").arg(cmd).status();

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
            } else {
                cli_ui::display_info(
                    "Please enter the appropriate command for your system to install R:",
                );
                let cmd: String = cli_ui::prompt_input(
                    "Installation command:",
                    Some("sudo apt install r-base".to_string()),
                )?;

                if !cmd.trim().is_empty() {
                    let proceed = cli_ui::prompt_confirm(&format!("Run: {}?", cmd), true)?;

                    if proceed {
                        cli_ui::display_info(&format!("Running: {}", cmd));

                        let status = Command::new("sh").arg("-c").arg(&cmd).status();

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
            let installed = cli_ui::prompt_confirm(
                "Have you installed R? Press 'y' when installation is complete.",
                false,
            )?;

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
1. Recommended: Install rig (R Installation Manager) first
   - Download from https://github.com/r-lib/rig/releases
   - Or use WinGet: winget install posit.rig 
   - Or use Chocolatey: choco install rig
   - Or use Scoop: scoop bucket add r-bucket https://github.com/cderv/r-bucket.git && scoop install rig
   
   Then install R with: rig add release

2. Alternative: Download R from https://cloud.r-project.org/bin/windows/base/
3. Optionally install RStudio from https://posit.co/download/rstudio-desktop/"#.to_string()
    } else if cfg!(target_os = "macos") {
        let arch_note = if cfg!(target_arch = "aarch64") {
            "Note: For Apple Silicon (M1/M2) Macs, make sure to download the arm64 version."
        } else {
            "Note: For Intel Macs, make sure to download the x86_64 version."
        };

        format!(
            r#"macOS Installation:
1. Recommended: Install rig (R Installation Manager) first
   ```bash
   # Using Homebrew
   brew tap r-lib/rig
   brew install --cask rig
   
   # Then install R
   rig add release
   ```

2. Using Homebrew directly:
   ```bash
   brew install --cask r
   ```

3. Alternative: Download from https://cloud.r-project.org/bin/macosx/
   {}
"#,
            arch_note
        )
    } else {
        r#"Linux Installation:
1. Recommended: Install rig (R Installation Manager) first

   Ubuntu/Debian:
   ```bash
   # Add rig repository
   curl -L https://rig.r-pkg.org/deb/rig.gpg -o /tmp/rig.gpg
   sudo mv /tmp/rig.gpg /etc/apt/trusted.gpg.d/rig.gpg
   echo "deb http://rig.r-pkg.org/deb rig main" | sudo tee /etc/apt/sources.list.d/rig.list
   sudo apt update
   sudo apt install r-rig
   
   # Then install R
   rig add release
   ```

   Fedora/RHEL/CentOS:
   ```bash
   # Install rig
   sudo yum install -y https://github.com/r-lib/rig/releases/download/latest/r-rig-latest-1.$(arch).rpm
   
   # Then install R
   rig add release
   ```

   Other Linux distributions:
   ```bash
   # Install rig
   curl -Ls https://github.com/r-lib/rig/releases/download/latest/rig-linux-$(arch)-latest.tar.gz | sudo tar xz -C /usr/local
   
   # Then install R
   rig add release
   ```

2. Alternative direct installation:

   Ubuntu/Debian:
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

   Fedora/RHEL/CentOS:
   ```bash
   sudo dnf install R
   ```

   Arch Linux:
   ```bash
   sudo pacman -S r
   ```"#.to_string()
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
        }
        _ => {
            cli_ui::display_warning("Could not verify R installation. Make sure R is properly installed and in your PATH.");
            cli_ui::display_info(
                "You may need to restart your terminal or add R to your PATH manually.",
            );
            Ok(())
        }
    }
}

/// Check if system is Debian/Ubuntu based
fn is_debian_based() -> bool {
    std::path::Path::new("/etc/debian_version").exists()
        || std::path::Path::new("/etc/apt").exists()
        || Command::new("apt").arg("--version").status().is_ok()
}

/// Check if system is Fedora/RHEL/CentOS based
fn is_fedora_based() -> bool {
    std::path::Path::new("/etc/redhat-release").exists()
        || std::path::Path::new("/etc/fedora-release").exists()
        || Command::new("dnf").arg("--version").status().is_ok()
}

/// Check if system is Arch based
fn is_arch_based() -> bool {
    std::path::Path::new("/etc/arch-release").exists()
        || Command::new("pacman").arg("--version").status().is_ok()
}
