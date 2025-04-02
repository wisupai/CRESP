use super::super::utils::write_file;
use crate::error::Result;
use crate::utils::cli_ui;
use std::path::Path;
use std::process::Command;
use super::super::config::{check_system_r, check_rig_available};

/// Create R project with the specified configuration
pub fn create_r_project(project_dir: &Path) -> Result<()> {
    // Check system R availability
    let system_r = check_system_r()?;
    let rig_available = check_rig_available()?;
    
    // Setup R environment
    let r_version = setup_r_environment(system_r, rig_available)?;
    
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

/// Setup R environment by checking existing installation and selecting version
fn setup_r_environment(system_r: Option<String>, rig_available: bool) -> Result<String> {
    cli_ui::display_header("R Configuration", "📊");
    
    // Default R version
    let default_version = "4.3.2".to_string();
    
    if let Some(ver) = &system_r {
        cli_ui::display_info(&format!("Detected installed R version: {}", ver));
        
        // Ask if user wants to use the detected version
        let use_detected = cli_ui::prompt_confirm(&format!("Use detected R version {}?", ver), true)?;
        
        if use_detected {
            return Ok(ver.clone());
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
        if ver.starts_with(&selected_version.split('.').take(2).collect::<Vec<_>>().join(".")) {
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
                    } else {
                        cli_ui::display_error(&format!("Failed to install R {}", selected_version));
                    }
                } else {
                    cli_ui::display_error("Failed to run rig command");
                }
            },
            i if (rig_available && i == 1) || (!rig_available && i == 0) => {
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
                    } else {
                        cli_ui::display_error("Failed to install R using Homebrew");
                    }
                } else {
                    cli_ui::display_error("Failed to run Homebrew command");
                }
            },
            i if (rig_available && i == 2) || (!rig_available && i == 1) => {
                cli_ui::display_info("Please download R from: https://cloud.r-project.org/bin/macosx/");
                cli_ui::display_info("Run the installer and follow the instructions.");
            },
            _ => cli_ui::display_info("You can install R later using the instructions in the README."),
        }
    } else {
        // Linux-specific installation options
        let mut install_options = vec![
            "Using apt (Ubuntu/Debian)",
            "Using dnf (Fedora/RHEL/CentOS)",
            "Using pacman (Arch Linux)",
            "Manual installation (other distributions)",
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
                    } else {
                        cli_ui::display_error(&format!("Failed to install R {}", selected_version));
                    }
                } else {
                    cli_ui::display_error("Failed to run rig command");
                }
            },
            i if (rig_available && i == 1) || (!rig_available && i == 0) => {
                cli_ui::display_info("Installing R using apt...");
                cli_ui::display_warning("This will require sudo privileges. You may be prompted for your password.");
                
                cli_ui::display_info("Please follow these commands to install R:");
                cli_ui::display_info("
sudo apt update
sudo apt install --no-install-recommends software-properties-common dirmngr
wget -qO- https://cloud.r-project.org/bin/linux/ubuntu/marutter_pubkey.asc | sudo tee -a /etc/apt/trusted.gpg.d/cran_ubuntu_key.asc
sudo add-apt-repository \"deb https://cloud.r-project.org/bin/linux/ubuntu $(lsb_release -cs)-cran40/\"
sudo apt update
sudo apt install r-base r-base-dev
                ");
            },
            i if (rig_available && i == 2) || (!rig_available && i == 1) => {
                cli_ui::display_info("Installing R using dnf...");
                cli_ui::display_warning("This will require sudo privileges. You may be prompted for your password.");
                cli_ui::display_info("Please run: sudo dnf install R");
            },
            i if (rig_available && i == 3) || (!rig_available && i == 2) => {
                cli_ui::display_info("Installing R using pacman...");
                cli_ui::display_warning("This will require sudo privileges. You may be prompted for your password.");
                cli_ui::display_info("Please run: sudo pacman -S r");
            },
            i if (rig_available && i == 4) || (!rig_available && i == 3) => {
                cli_ui::display_info("Please follow the R installation instructions for your distribution:");
                cli_ui::display_info("https://cloud.r-project.org/bin/linux/");
            },
            _ => cli_ui::display_info("You can install R later using the instructions in the README."),
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
        r#"macOS Installation:
1. Using Homebrew (recommended):
   ```bash
   brew install --cask r
   ```

2. Alternative: Download from https://cloud.r-project.org/bin/macosx/
3. Consider installing R Installation Manager (rig) for managing multiple R versions:
   ```bash
   brew install rig
   ```"#.to_string()
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
