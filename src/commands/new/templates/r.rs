use super::super::templates::conda_utils;
use super::super::utils::write_file;
use crate::error::Result;
use crate::utils::cli_ui;
use std::path::Path;
use std::process::Command;

/// Configuration for R project
#[derive(Debug)]
struct RProjectConfig {
    r_version: String,
    project_name: String,
    create_conda_env: bool,
}

/// Create R project with the specified configuration
pub fn create_r_project(project_dir: &Path) -> Result<()> {
    // PHASE 1: Check prerequisites and collect user configuration

    // Check conda availability (required for R projects)
    let conda_version = match conda_utils::ensure_conda_available()? {
        Some(version) => version,
        None => {
            return Err(crate::error::Error::Environment(
                "Conda is required but not found".to_string(),
            ));
        }
    };

    // Check conda version and show warning if outdated
    conda_utils::check_conda_version(&conda_version)?;

    // Collect project configuration from user
    let config = collect_r_project_config(project_dir)?;

    // PHASE 2: Create project structure and files based on configuration
    create_r_project_structure(project_dir, &config)?;

    // PHASE 3: Setup environment (if requested)
    if config.create_conda_env {
        if conda_utils::create_conda_environment(project_dir, "environment.yml")? {
            // Verify if R is correctly installed in the conda environment
            verify_r_installation()?;
        }
    }

    Ok(())
}

/// Collect all user settings for R project
fn collect_r_project_config(project_dir: &Path) -> Result<RProjectConfig> {
    cli_ui::display_info("Setting up r environment...");

    // Get R version from user
    let r_version = setup_r_environment()?;

    // Get project name for conda environment
    let project_name = project_dir
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("myresearch")
        .to_string();

    // Ask if conda environment should be created
    let create_conda_env = cli_ui::prompt_confirm("Create conda environment now?", true)?;

    Ok(RProjectConfig {
        r_version,
        project_name,
        create_conda_env,
    })
}

/// Create R project structure and files
fn create_r_project_structure(project_dir: &Path, config: &RProjectConfig) -> Result<()> {
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
    let renv_lock = renv_lock.replace("__R_VERSION__", &config.r_version);
    write_file(&project_dir.join("renv.lock"), &renv_lock)?;

    // Create conda environment.yml file
    // Using the new function to generate environment.yml content
    let channels = ["r", "conda-forge", "defaults"];
    let dependencies = [
        &format!("r-base={}", config.r_version),
        "r-renv",
        "r-essentials",
        "r-devtools",
        "r-testthat",
    ];

    let environment_yml =
        conda_utils::generate_base_environment_yml(&config.project_name, &channels, &dependencies);

    write_file(&project_dir.join("environment.yml"), &environment_yml)?;

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

# Execute main function if the script is run interactively
if (interactive()) {
    main()
}
"#;
    write_file(&project_dir.join("R/main.R"), main_r)?;

    // Create renv setup script (updated for conda compatibility)
    let renv_setup = r#"# renv setup script
# This script initializes renv for your project

# Check if running in a conda environment 
is_conda <- Sys.getenv("CONDA_PREFIX") != ""
if (!is_conda) {
  message("Warning: It's recommended to run this in a conda environment.")
  message("Please activate your conda environment with: conda activate project-name")
}

# Install renv if it's not already installed
if (!requireNamespace("renv", quietly = TRUE)) {
  message("Installing renv package...")
  if (is_conda) {
    # In conda environment, we can use install.packages() 
    # since r-renv should be part of the conda environment
    install.packages("renv", repos = "https://cloud.r-project.org")
  } else {
    # Fallback to CRAN installation
    install.packages("renv")
  }
}

# Initialize renv for this project
# Note: While conda manages the R version and core dependencies,
# renv manages the specific R package versions for reproducibility
renv::init()

# Install dependencies from renv.lock
renv::restore()

message("R environment setup complete! R version is managed by conda, package dependencies by renv.")
"#;
    write_file(&project_dir.join("setup.R"), renv_setup)?;

    // Create README.md with conda + renv instructions
    let readme = format!(
        r#"# {}: R Research Project

This is an R research project using CRESP protocol.

## Environment Management Strategy

This project uses a dual management approach:
- **Conda**: Manages the R language version and system-level dependencies
- **renv**: Manages R package dependencies within the R environment

This separation allows for both system-level reproducibility (hardware and core language) and package-level reproducibility (R packages and their versions).

## Project Structure

```
.
├── R/              # R code files
├── data/           # Data directory
├── output/         # Output directory
├── tests/          # Tests directory
├── DESCRIPTION     # Package metadata
├── renv.lock       # Package dependency lock file
├── environment.yml # Conda environment specification
└── setup.R         # Environment setup script
```

## Setup

1. Make sure you have conda installed.
   If not, install Miniconda from: https://docs.conda.io/en/latest/miniconda.html

2. Clone this repository and change to the project directory:
```bash
git clone <repository-url>
cd {}
```

3. Create and activate the conda environment:
```bash
conda env create -f environment.yml
conda activate {}
```

4. Setup the R environment using renv:
```r
Rscript -e "source('setup.R')"
```

5. Run the project:
```bash
Rscript -e "source('R/main.R')"
```

## Development

There are two ways to manage packages in this project:

1. **For regular R packages (preferred for most cases)**: Use R's built-in package management with renv:
```r
# Install a package
install.packages("package_name")

# After installing packages, always snapshot the environment
renv::snapshot()
```

2. **For packages with system dependencies**: Use conda (this installs both the R package and required system libraries):
```bash
# Prefer r channel for R packages
conda install -c r r-package_name

# Or from conda-forge
conda install -c conda-forge r-package_name

# Or specify multiple channels (recommended)
conda install -c r -c conda-forge r-package_name
```

After installing packages with conda, also update renv:
```r
# Make renv aware of packages installed via conda
renv::snapshot()
```

## Testing

Run tests with:
```r
testthat::test_package("{}")
```
"#,
        config.project_name, config.project_name, config.project_name, config.project_name
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

# conda specific
.conda/

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

/// Setup R environment - only use conda for R management
fn setup_r_environment() -> Result<String> {
    cli_ui::display_header("R Configuration", "📊");

    // Default R version
    let default_version = "4.3.2".to_string();

    cli_ui::display_info("Conda will be used to manage the R environment, providing better isolation and reproducibility.");

    // Present R version options
    let r_options = vec![
        "R 4.3 (latest stable, recommended)",
        "R 4.2 (stable)",
        "R 4.1 (stable)",
        "Custom version (specify)",
    ];

    let selection = cli_ui::prompt_select("Select R version to use with conda", &r_options)?;

    let selected_version = match selection {
        0 => "4.3".to_string(),
        1 => "4.2".to_string(),
        2 => "4.1".to_string(),
        3 => {
            // Custom version
            cli_ui::prompt_input("Enter R version (e.g., 4.4):", Some(default_version))?
        }
        _ => default_version,
    };

    cli_ui::display_success(&format!(
        "Selected R version: {} (will be managed by conda)",
        selected_version
    ));
    cli_ui::display_info(
        "A conda environment will be created with this R version when setting up the project.",
    );

    Ok(selected_version)
}

/// Verify that R is installed in the conda environment
fn verify_r_installation() -> Result<()> {
    cli_ui::display_info("Verifying R installation in conda environment...");

    // Try to get R version from conda environment
    let output = Command::new("Rscript").arg("--version").output();

    match output {
        Ok(output) => {
            let version_output = String::from_utf8_lossy(&output.stderr); // R prints version to stderr
            cli_ui::display_success(&format!(
                "R is successfully installed: {}",
                version_output.trim()
            ));
            Ok(())
        }
        _ => {
            cli_ui::display_warning("Could not verify R installation. Make sure R is properly installed in your conda environment.");
            Ok(())
        }
    }
}
