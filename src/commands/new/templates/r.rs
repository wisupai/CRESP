use std::path::Path;
use crate::error::Result;
use crate::utils::cli_ui;
use super::super::utils::write_file;

/// Create R project with the specified configuration
pub fn create_r_project(project_dir: &Path) -> Result<()> {
    cli_ui::display_info("Creating R project structure...");
    // Create basic R project structure 
    let dirs = &[
        "R",
        "data",
        "output",
        "tests/testthat",
        "docs",
    ];
    
    for dir in dirs {
        std::fs::create_dir_all(project_dir.join(dir))?;
    }
    
    cli_ui::display_info("Generating R project files...");
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
    
    // Create README.md
    let project_name = project_dir
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("myresearch");
        
    let readme = format!(r#"# {}: R Research Project

This is an R research project using CRESP protocol.

## Project Structure

```
.
├── R/              # R code files
├── data/           # Data directory
├── output/         # Output directory
├── tests/          # Tests directory
├── DESCRIPTION     # Package metadata
└── renv.lock       # Package dependency lock file
```

## Setup

1. Install R (recommended 4.x or newer).

2. Install renv:
```r
install.packages("renv")
```

3. Initialize renv:
```r
renv::init()
```

4. Install dependencies:
```r
renv::restore()
```

5. Run the project:
```r
source("R/main.R")
```

## Testing

Run tests with:
```r
testthat::test_package("{}")
```
"#, project_name, project_name);
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