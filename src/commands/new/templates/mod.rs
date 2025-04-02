use super::utils::{create_directories, write_file};
use crate::error::Result;
use crate::utils::cli_ui;
use std::path::Path;

mod matlab;
mod python;
mod r;

pub use matlab::create_matlab_project;
pub use python::create_python_project;
pub use r::create_r_project;

/// Template type for project structure
#[derive(Debug, Clone, Copy)]
pub enum TemplateType {
    Basic,
    DataAnalysis,
    MachineLearning,
    ScientificComputing,
    Custom,
}

impl From<&str> for TemplateType {
    fn from(s: &str) -> Self {
        match s {
            "1" => TemplateType::Basic,
            "2" => TemplateType::DataAnalysis,
            "3" => TemplateType::MachineLearning,
            "4" => TemplateType::ScientificComputing,
            "5" => TemplateType::Custom,
            _ => TemplateType::Basic,
        }
    }
}

/// Create project structure based on template type
pub fn create_project_structure(
    project_dir: &Path,
    template_type: TemplateType,
    language: &str,
) -> Result<()> {
    match template_type {
        TemplateType::Basic => {
            cli_ui::display_info("Creating basic project structure...");
            create_basic_structure(project_dir, language)
        }
        TemplateType::DataAnalysis => {
            cli_ui::display_info("Creating data analysis project structure...");
            create_data_analysis_structure(project_dir, language)
        }
        TemplateType::MachineLearning => {
            cli_ui::display_info("Creating machine learning project structure...");
            create_ml_structure(project_dir, language)
        }
        TemplateType::ScientificComputing => {
            cli_ui::display_info("Creating scientific computing project structure...");
            create_scientific_structure(project_dir, language)
        }
        TemplateType::Custom => {
            cli_ui::display_info("Creating custom project structure...");
            create_custom_structure(project_dir, language)
        }
    }
}

/// Create basic project structure (flat directories)
fn create_basic_structure(project_dir: &Path, _language: &str) -> Result<()> {
    // Basic flat structure for simple experiments
    let dirs = &[
        "data",      // Raw and processed data
        "output",    // Experiment outputs
        "notebooks", // Jupyter notebooks
        "scripts",   // Utility scripts
        "config",    // Configuration files
    ];

    create_directories(dirs, project_dir)?;

    // Create README with basic structure explanation
    let readme = r#"# Project Structure

```
.
├── data/           # Raw and processed data
├── output/         # Experiment outputs
├── notebooks/      # Jupyter notebooks
├── scripts/        # Utility scripts
└── config/         # Configuration files
```

## Directory Structure

- `data/`: Store your raw data and processed data
- `output/`: Store experiment results and outputs
- `notebooks/`: Jupyter notebooks for interactive analysis
- `scripts/`: Utility scripts and helper functions
- `config/`: Configuration files and parameters

## Usage

1. Place your raw data in the `data/` directory
2. Use notebooks in `notebooks/` for interactive analysis
3. Save experiment outputs to `output/`
4. Store configuration in `config/`
"#;
    write_file(&project_dir.join("README.md"), readme)?;

    Ok(())
}

/// Create data analysis project structure
fn create_data_analysis_structure(project_dir: &Path, _language: &str) -> Result<()> {
    // Structure for data analysis projects
    let dirs = &[
        "data/raw",              // Raw data
        "data/processed",        // Processed data
        "data/external",         // External data sources
        "output/figures",        // Generated figures
        "output/tables",         // Generated tables
        "output/reports",        // Generated reports
        "notebooks/analysis",    // Analysis notebooks
        "notebooks/exploration", // Data exploration notebooks
        "scripts/data",          // Data processing scripts
        "scripts/analysis",      // Analysis scripts
        "scripts/visualization", // Visualization scripts
        "config",                // Configuration files
    ];

    create_directories(dirs, project_dir)?;

    // Create README with data analysis structure explanation
    let readme = r#"# Data Analysis Project Structure

```
.
├── data/
│   ├── raw/          # Original, immutable data
│   ├── processed/    # Cleaned and processed data
│   └── external/     # External data sources
├── output/
│   ├── figures/      # Generated figures
│   ├── tables/       # Generated tables
│   └── reports/      # Generated reports
├── notebooks/
│   ├── analysis/     # Analysis notebooks
│   └── exploration/  # Data exploration notebooks
├── scripts/
│   ├── data/         # Data processing scripts
│   ├── analysis/     # Analysis scripts
│   └── visualization/# Visualization scripts
└── config/           # Configuration files
```

## Directory Structure

- `data/raw/`: Original, immutable data
- `data/processed/`: Cleaned and processed data
- `data/external/`: External data sources
- `output/figures/`: Generated figures and plots
- `output/tables/`: Generated tables and statistics
- `output/reports/`: Generated reports and documentation
- `notebooks/analysis/`: Analysis notebooks
- `notebooks/exploration/`: Data exploration notebooks
- `scripts/data/`: Data processing scripts
- `scripts/analysis/`: Analysis and statistical scripts
- `scripts/visualization/`: Visualization and plotting scripts
- `config/`: Configuration files and parameters

## Usage

1. Place raw data in `data/raw/`
2. Use `notebooks/exploration/` for initial data exploration
3. Use `notebooks/analysis/` for detailed analysis
4. Save processed data to `data/processed/`
5. Generate outputs in respective `output/` subdirectories
"#;
    write_file(&project_dir.join("README.md"), readme)?;

    Ok(())
}

/// Create machine learning project structure
fn create_ml_structure(project_dir: &Path, _language: &str) -> Result<()> {
    // Structure for machine learning projects
    let dirs = &[
        "data/raw",              // Raw data
        "data/processed",        // Processed data
        "data/external",         // External data sources
        "models",                // Trained models
        "output/predictions",    // Model predictions
        "output/figures",        // Generated figures
        "output/metrics",        // Model metrics
        "notebooks/exploration", // Data exploration
        "notebooks/experiments", // Experiment notebooks
        "scripts/data",          // Data processing scripts
        "scripts/models",        // Model scripts
        "scripts/training",      // Training scripts
        "scripts/evaluation",    // Evaluation scripts
        "config",                // Configuration files
    ];

    create_directories(dirs, project_dir)?;

    // Create README with ML structure explanation
    let readme = r#"# Machine Learning Project Structure

```
.
├── data/
│   ├── raw/          # Original, immutable data
│   ├── processed/    # Cleaned and processed data
│   └── external/     # External data sources
├── models/           # Trained models
├── output/
│   ├── predictions/  # Model predictions
│   ├── figures/      # Generated figures
│   └── metrics/      # Model metrics
├── notebooks/
│   ├── exploration/  # Data exploration
│   └── experiments/  # Experiment notebooks
├── scripts/
│   ├── data/         # Data processing scripts
│   ├── models/       # Model scripts
│   ├── training/     # Training scripts
│   └── evaluation/   # Evaluation scripts
└── config/           # Configuration files
```

## Directory Structure

- `data/raw/`: Original, immutable data
- `data/processed/`: Cleaned and processed data
- `data/external/`: External data sources
- `models/`: Trained models and checkpoints
- `output/predictions/`: Model predictions
- `output/figures/`: Generated figures and plots
- `output/metrics/`: Model metrics and results
- `notebooks/exploration/`: Data exploration notebooks
- `notebooks/experiments/`: Experiment notebooks
- `scripts/data/`: Data processing scripts
- `scripts/models/`: Model architecture scripts
- `scripts/training/`: Training scripts
- `scripts/evaluation/`: Evaluation scripts
- `config/`: Configuration files and parameters

## Usage

1. Place raw data in `data/raw/`
2. Use `notebooks/exploration/` for data exploration
3. Use `notebooks/experiments/` for model experiments
4. Save trained models in `models/`
5. Generate predictions and metrics in respective `output/` subdirectories
"#;
    write_file(&project_dir.join("README.md"), readme)?;

    Ok(())
}

/// Create scientific computing project structure
fn create_scientific_structure(project_dir: &Path, _language: &str) -> Result<()> {
    // Structure for scientific computing projects
    let dirs = &[
        "data/raw",                // Raw data
        "data/processed",          // Processed data
        "data/external",           // External data sources
        "output/results",          // Simulation results
        "output/figures",          // Generated figures
        "output/tables",           // Generated tables
        "notebooks/analysis",      // Analysis notebooks
        "notebooks/visualization", // Visualization notebooks
        "scripts/simulation",      // Simulation scripts
        "scripts/analysis",        // Analysis scripts
        "scripts/visualization",   // Visualization scripts
        "config",                  // Configuration files
    ];

    create_directories(dirs, project_dir)?;

    // Create README with scientific computing structure explanation
    let readme = r#"# Scientific Computing Project Structure

```
.
├── data/
│   ├── raw/          # Original, immutable data
│   ├── processed/    # Cleaned and processed data
│   └── external/     # External data sources
├── output/
│   ├── results/      # Simulation results
│   ├── figures/      # Generated figures
│   └── tables/       # Generated tables
├── notebooks/
│   ├── analysis/     # Analysis notebooks
│   └── visualization/# Visualization notebooks
├── scripts/
│   ├── simulation/   # Simulation scripts
│   ├── analysis/     # Analysis scripts
│   └── visualization/# Visualization scripts
└── config/           # Configuration files
```

## Directory Structure

- `data/raw/`: Original, immutable data
- `data/processed/`: Cleaned and processed data
- `data/external/`: External data sources
- `output/results/`: Simulation results and outputs
- `output/figures/`: Generated figures and plots
- `output/tables/`: Generated tables and statistics
- `notebooks/analysis/`: Analysis notebooks
- `notebooks/visualization/`: Visualization notebooks
- `scripts/simulation/`: Simulation and computation scripts
- `scripts/analysis/`: Analysis and processing scripts
- `scripts/visualization/`: Visualization scripts
- `config/`: Configuration files and parameters

## Usage

1. Place raw data in `data/raw/`
2. Use `notebooks/analysis/` for data analysis
3. Use `notebooks/visualization/` for result visualization
4. Run simulations using scripts in `scripts/simulation/`
5. Save results and outputs in respective `output/` subdirectories
"#;
    write_file(&project_dir.join("README.md"), readme)?;

    Ok(())
}

/// Create custom project structure
fn create_custom_structure(project_dir: &Path, _language: &str) -> Result<()> {
    cli_ui::display_info("Custom project structure setup:");
    let options = vec!["Create basic directories", "Create detailed structure"];

    let selection = cli_ui::prompt_select("Choose structure type", &options)?;

    if selection == 1 {
        // Create detailed structure
        cli_ui::display_info("Enter directory names (comma-separated):");
        let dirs_input: String = cli_ui::prompt_input("> ", None)?;

        let dirs: Vec<&str> = dirs_input.split(',').map(|s| s.trim()).collect();
        let dirs_clone = dirs.clone();

        // Create directories
        create_directories(&dirs, project_dir)?;

        // Create README with custom structure
        let mut readme = String::from("# Custom Project Structure\n\n");
        readme.push_str("```\n");
        readme.push_str(".\n");
        for dir in dirs {
            readme.push_str(&format!("└── {}/\n", dir));
        }
        readme.push_str("```\n\n");

        readme.push_str("## Directory Structure\n\n");
        for dir in dirs_clone {
            readme.push_str(&format!("- `{}/`: [Add description]\n", dir));
        }

        readme.push_str("\n## Usage\n\n");
        readme.push_str("1. [Add usage instructions]\n");
        readme.push_str("2. [Add usage instructions]\n");
        readme.push_str("3. [Add usage instructions]\n");

        write_file(&project_dir.join("README.md"), &readme)?;
    } else {
        // Create basic directories
        let dirs = &[
            "data",      // Data directory
            "output",    // Output directory
            "notebooks", // Notebooks directory
            "scripts",   // Scripts directory
            "config",    // Configuration directory
        ];

        create_directories(dirs, project_dir)?;

        // Create README with basic structure
        let readme = r#"# Custom Project Structure

```
.
├── data/           # Data directory
├── output/         # Output directory
├── notebooks/      # Notebooks directory
├── scripts/        # Scripts directory
└── config/         # Configuration directory
```

## Directory Structure

- `data/`: Store your data
- `output/`: Store experiment outputs
- `notebooks/`: Store notebooks
- `scripts/`: Store scripts
- `config/`: Store configuration files

## Usage

1. [Add usage instructions]
2. [Add usage instructions]
3. [Add usage instructions]
"#;
        write_file(&project_dir.join("README.md"), readme)?;
    }

    Ok(())
}
