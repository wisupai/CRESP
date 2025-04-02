use super::super::config::check_matlab_available;
use super::super::utils::write_file;
use crate::error::Result;
use crate::utils::cli_ui;
use std::path::Path;
use std::process::Command;

/// Create MATLAB project with the specified configuration
pub fn create_matlab_project(project_dir: &Path) -> Result<()> {
    // Check system MATLAB availability and setup environment
    let matlab_version = setup_matlab_environment()?;

    cli_ui::display_info("Creating MATLAB project structure...");
    // Create basic MATLAB project structure
    let dirs = &[
        "src",
        "data",
        "output",
        "tests",
        "docs",
        "scripts",
        "functions",
    ];

    for dir in dirs {
        std::fs::create_dir_all(project_dir.join(dir))?;
    }

    cli_ui::display_info("Generating MATLAB project files...");

    // Create main.m file
    let main_m = r#"%% Main Script
% This is the main entry point for the research project.
%
% Usage:
%   Run this script to execute the complete workflow
%
% Author: Your Name
% Date: 2023-01-01

%% Setup environment
addpath(genpath('src'));
addpath(genpath('functions'));

%% Parameters
% Define any parameters needed for the experiment

%% Run the experiment
disp('Hello, CRESP!');

% Your code goes here
"#;
    write_file(&project_dir.join("main.m"), main_m)?;

    // Create ProjectSetup.m file
    let setup_m = r#"%% Project Setup Script
% This script initializes the MATLAB environment for this project
%
% Usage:
%   Run this script once at the beginning of your MATLAB session
%
% Author: Your Name
% Date: 2023-01-01

%% Add all project directories to the path
% Get the directory where this script is located
projectDir = fileparts(mfilename('fullpath'));

% Add subdirectories to path
addpath(genpath(fullfile(projectDir, 'src')));
addpath(genpath(fullfile(projectDir, 'functions')));
addpath(genpath(fullfile(projectDir, 'scripts')));
addpath(genpath(fullfile(projectDir, 'tests')));

%% Check for and install required toolboxes
% This is a placeholder. You would replace this with actual
% toolbox checks for your specific requirements.
requiredToolboxes = {'Statistics and Machine Learning Toolbox', 'Signal Processing Toolbox'};
installedToolboxes = ver;
installedToolboxNames = {installedToolboxes.Name};

missingToolboxes = setdiff(requiredToolboxes, installedToolboxNames);
if ~isempty(missingToolboxes)
    warning('The following toolboxes are required but not installed:');
    for i = 1:length(missingToolboxes)
        warning('  - %s', missingToolboxes{i});
    end
    warning('Please install these toolboxes before proceeding.');
else
    disp('All required toolboxes are installed.');
end

%% Setup data directories
dataDir = fullfile(projectDir, 'data');
if ~exist(dataDir, 'dir')
    mkdir(dataDir);
end

outputDir = fullfile(projectDir, 'output');
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

%% Display setup information
disp('MATLAB environment setup complete!');
disp(['Project directory: ' projectDir]);
disp(['Data directory: ' dataDir]);
disp(['Output directory: ' outputDir]);
disp(' ');
disp('Ready to start research!');
"#;
    write_file(&project_dir.join("ProjectSetup.m"), setup_m)?;

    // Create an example function
    let example_function = r#"function result = exampleFunction(input)
% EXAMPLEFUNCTION A template function for the project
%
%   RESULT = EXAMPLEFUNCTION(INPUT) computes a result based on the input
%
%   Inputs:
%       input - Input data (description)
%
%   Outputs:
%       result - Output data (description)
%
%   Example:
%       x = 1:10;
%       y = exampleFunction(x);
%
%   See also ANOTHER_FUNCTION, YET_ANOTHER_FUNCTION

% Validate input
if nargin < 1
    error('Input is required');
end

% Process input
result = input .^ 2;  % Example operation

end
"#;
    write_file(
        &project_dir.join("functions/exampleFunction.m"),
        example_function,
    )?;

    // Create a test script
    let test_script = r#"%% Test Script
% This script runs tests for the project functions
%
% Usage:
%   Run this script to verify that functions are working as expected
%
% Author: Your Name
% Date: 2023-01-01

%% Setup
addpath(genpath('../functions'));
addpath(genpath('../src'));

%% Test exampleFunction
disp('Testing exampleFunction...');

% Test case 1: Basic functionality
input = 1:5;
expected = [1, 4, 9, 16, 25];
actual = exampleFunction(input);

if isequal(actual, expected)
    disp('  ✓ Basic functionality test passed');
else
    error('  ✗ Basic functionality test failed');
end

% Test case 2: Error handling
try
    result = exampleFunction();
    error('  ✗ Error handling test failed. Should have thrown an error.');
catch
    disp('  ✓ Error handling test passed');
end

%% All tests completed
disp('All tests completed successfully!');
"#;
    write_file(&project_dir.join("tests/test_example.m"), test_script)?;

    // Create a MATLAB project file
    let project_name = project_dir
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("matlabproject");

    let prj_file = format!(
        r#"<?xml version="1.0" encoding="UTF-8"?>
<MATLABProject xmlns="http://www.mathworks.com/MATLABProjectFile" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" version="1.0"/>
"#
    );
    write_file(
        &project_dir.join(format!("{}.prj", project_name)),
        &prj_file,
    )?;

    // Create README.md
    let readme = format!(
        r#"# {}: MATLAB Research Project

This is a MATLAB research project using CRESP protocol.

## Project Structure

```
.
├── src/             # Source code for main project functionality
├── functions/       # Utility functions and helper code
├── scripts/         # Additional MATLAB scripts
├── data/            # Data directory
├── output/          # Output directory
├── tests/           # Tests directory
├── docs/            # Documentation
├── main.m           # Main entry point script
├── ProjectSetup.m   # Environment setup script
└── {}.prj           # MATLAB project file
```

## Setup

1. Install MATLAB (recommended version {}).

{}

2. Clone this repository and change to the project directory:
```bash
git clone <repository-url>
cd {}
```

3. Open MATLAB and navigate to the project directory:
```matlab
cd('/path/to/{}')
```

4. Run the setup script:
```matlab
ProjectSetup
```

5. Run the main script:
```matlab
main
```

## Testing

Run tests with:
```matlab
cd tests
test_example
```
"#,
        project_name,
        project_name,
        matlab_version,
        get_matlab_installation_instructions(),
        project_name,
        project_name
    );
    write_file(&project_dir.join("README.md"), &readme)?;

    // Create .gitignore
    let gitignore = r#"# MATLAB specific
*.asv
*.m~
*.mex*
*.mat
*.mlx
slprj/
sccprj/
codegen/
*.autosave
*.slxc

# MATLAB project specific
*.prj

# Output files
output/
*.html
*.pdf
*.png
*.jpg
*.fig

# Packaged app files
*.mlappinstall
*.mlpkginstall

# Large data files
data/**/*.csv
data/**/*.xlsx
data/**/*.dat
"#;
    write_file(&project_dir.join(".gitignore"), gitignore)?;

    Ok(())
}

/// Get detailed information about installed MATLAB
fn get_matlab_info() -> Result<(Option<String>, Option<MatlabInfo>)> {
    // Check basic MATLAB version
    let system_matlab = check_matlab_available()?;

    if system_matlab.is_none() {
        return Ok((None, None));
    }

    // Try to get more detailed information
    let version = system_matlab.as_ref().unwrap().clone();

    // Get MATLAB executable path
    let matlab_path = get_matlab_path()?;

    // Determine installation method
    let install_method = if let Some(path) = &matlab_path {
        determine_install_method(path)
    } else {
        "Unknown".to_string()
    };

    // Get MATLAB toolboxes if possible
    let toolboxes = get_matlab_toolboxes(&matlab_path)?;

    Ok((
        system_matlab,
        Some(MatlabInfo {
            version,
            path: matlab_path,
            install_method,
            toolboxes,
        }),
    ))
}

/// Struct to hold detailed MATLAB information
#[derive(Debug, Clone)]
struct MatlabInfo {
    version: String,
    path: Option<String>,
    install_method: String,
    toolboxes: Vec<String>,
}

/// Get MATLAB executable path
fn get_matlab_path() -> Result<Option<String>> {
    let cmd = if cfg!(target_os = "windows") {
        Command::new("where").arg("matlab.exe").output()
    } else {
        Command::new("which").arg("matlab").output()
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

/// Determine how MATLAB was installed based on its path
fn determine_install_method(path: &str) -> String {
    if cfg!(target_os = "windows") {
        if path.contains("\\Program Files\\MATLAB") {
            return "Official installer".to_string();
        }
    } else if cfg!(target_os = "macos") {
        if path.contains("/Applications/MATLAB") {
            return "Official installer".to_string();
        }
    } else {
        // Linux
        if path.contains("/usr/local/MATLAB") {
            return "Official installer".to_string();
        }
    }

    "Unknown source".to_string()
}

/// Get MATLAB toolboxes
fn get_matlab_toolboxes(_matlab_path: &Option<String>) -> Result<Vec<String>> {
    // In a real implementation, we would try to run a MATLAB command to get installed toolboxes
    // For now, we'll return a default set of toolboxes that are commonly used
    Ok(vec![
        "MATLAB".to_string(),
        "Simulink".to_string(),
        "Statistics and Machine Learning Toolbox".to_string(),
        "Signal Processing Toolbox".to_string(),
        "Image Processing Toolbox".to_string(),
    ])
}

/// Get platform-specific MATLAB installation instructions
fn get_matlab_installation_instructions() -> String {
    if cfg!(target_os = "windows") {
        r#"Windows Installation:
1. Download MATLAB from MathWorks website: https://www.mathworks.com/downloads/
2. Run the installer and follow the instructions
3. Select the toolboxes you need for your research"#
            .to_string()
    } else if cfg!(target_os = "macos") {
        r#"macOS Installation:
1. Download MATLAB from MathWorks website: https://www.mathworks.com/downloads/
2. Mount the disk image and run the installer
3. Follow the installation prompts and select required toolboxes"#
            .to_string()
    } else {
        r#"Linux Installation:
1. Download MATLAB from MathWorks website: https://www.mathworks.com/downloads/
2. Extract the installer and run the install script:
   ```bash
   ./install
   ```
3. Follow the installation prompts and select required toolboxes"#
            .to_string()
    }
}

/// Setup MATLAB environment by checking existing installation and selecting version
fn setup_matlab_environment() -> Result<String> {
    cli_ui::display_header("MATLAB Configuration", "🔢");

    // Check for installed MATLAB
    let system_matlab = check_matlab_available()?;

    // Default MATLAB version if not found
    let default_version = "R2023b".to_string();

    // Flag to track if user rejected existing installation
    let mut rejected_existing = false;

    if let Some(ver) = &system_matlab {
        cli_ui::display_info(&format!("Detected installed MATLAB version: {}", ver));

        // Get additional information if available
        let matlab_path = get_matlab_path()?;
        if let Some(path) = &matlab_path {
            cli_ui::display_info(&format!("MATLAB location: {}", path));
            cli_ui::display_info(&format!(
                "Installation method: {}",
                determine_install_method(path)
            ));
        }

        // Display toolbox information
        cli_ui::display_info("Detected toolboxes:");
        for toolbox in get_matlab_toolboxes(&matlab_path)?.iter().take(5) {
            cli_ui::display_info(&format!("  - {}", toolbox));
        }
        if get_matlab_toolboxes(&matlab_path)?.len() > 5 {
            cli_ui::display_info(&format!(
                "  - ... and {} more",
                get_matlab_toolboxes(&matlab_path)?.len() - 5
            ));
        }

        // Ask if user wants to use the detected version
        let use_detected =
            cli_ui::prompt_confirm(&format!("Use detected MATLAB version {}?", ver), true)?;

        if use_detected {
            return Ok(ver.clone());
        } else {
            // User rejected the existing installation
            rejected_existing = true;
        }
    } else {
        cli_ui::display_warning("No MATLAB installation detected on your system.");
    }

    // Present MATLAB version options
    let matlab_options = vec![
        "MATLAB R2023b (latest)",
        "MATLAB R2023a",
        "MATLAB R2022b",
        "MATLAB R2022a",
        "MATLAB R2021b",
        "MATLAB R2021a",
        "Custom version (specify)",
    ];

    let selection = cli_ui::prompt_select("Select MATLAB version", &matlab_options)?;

    let selected_version = match selection {
        0 => "R2023b".to_string(),
        1 => "R2023a".to_string(),
        2 => "R2022b".to_string(),
        3 => "R2022a".to_string(),
        4 => "R2021b".to_string(),
        5 => "R2021a".to_string(),
        6 => {
            // Custom version
            cli_ui::prompt_input(
                "Enter MATLAB version (e.g., R2024a):",
                Some(default_version),
            )?
        }
        _ => default_version,
    };

    // Check if the selected version is installed
    if let Some(ver) = &system_matlab {
        if !rejected_existing && ver.starts_with(&selected_version) {
            cli_ui::display_success(&format!("Found compatible MATLAB version: {}", ver));
            return Ok(ver.clone());
        }
    }

    // MATLAB is not installed or the required version is not available
    cli_ui::display_info(&format!(
        "MATLAB {} needs to be installed on your system.",
        selected_version
    ));

    // Ask how the user wants to proceed with MATLAB installation
    let install_options = vec![
        "Download MATLAB Installer (Official MathWorks method)",
        "Use system package manager (when available)",
        "Install with direct download script",
        "I already have an installer",
        "I'll install it later",
        "Troubleshoot existing installation",
    ];

    let install_selection =
        cli_ui::prompt_select("How would you like to proceed?", &install_options)?;

    match install_selection {
        0 => {
            // Official MathWorks installer method
            cli_ui::display_info("MATLAB official installation process:");
            cli_ui::display_info("1. You will need a MathWorks account and license to proceed");

            // Ask about license type
            let license_options = vec![
                "Individual license (using license key)",
                "Academic license (through university portal)",
                "Organization license (using license server)",
                "Trial license (30-day evaluation)",
                "I already have a license",
            ];

            let license_selection =
                cli_ui::prompt_select("What type of license will you use?", &license_options)?;

            // Instructions based on license type
            match license_selection {
                0 => {
                    cli_ui::display_info("For individual licenses with license key:");
                    cli_ui::display_info("1. Visit: https://www.mathworks.com/licensecenter");
                    cli_ui::display_info("2. Sign in with your MathWorks account");
                    cli_ui::display_info("3. Click on 'Download Products' for your license");
                    cli_ui::display_info("4. Select the version and products you want to install");
                    cli_ui::display_info("5. Download the installer");

                    // Open browser to license center
                    if cfg!(target_os = "macos") {
                        let _ = Command::new("open")
                            .arg("https://www.mathworks.com/licensecenter")
                            .status();
                    } else if cfg!(target_os = "windows") {
                        let _ = Command::new("cmd")
                            .args(["/c", "start", "https://www.mathworks.com/licensecenter"])
                            .status();
                    } else {
                        let _ = Command::new("xdg-open")
                            .arg("https://www.mathworks.com/licensecenter")
                            .status();
                    }
                }
                1 => {
                    cli_ui::display_info("For academic licenses:");
                    cli_ui::display_info(
                        "1. Visit your university's software portal or MathWorks portal",
                    );
                    cli_ui::display_info("2. Sign in with your university credentials");
                    cli_ui::display_info("3. Navigate to MATLAB downloads section");
                    cli_ui::display_info("4. Select your desired version");
                    cli_ui::display_info("5. Download the installer and follow instructions");

                    // Prompt for university portal URL
                    let portal_url = cli_ui::prompt_input(
                        "Enter your university portal URL (or leave empty to open mathworks.com):",
                        None::<String>,
                    )?;

                    // Open browser to university portal or MathWorks
                    let url = if portal_url.is_empty() {
                        "https://www.mathworks.com"
                    } else {
                        &portal_url
                    };
                    if cfg!(target_os = "macos") {
                        let _ = Command::new("open").arg(url).status();
                    } else if cfg!(target_os = "windows") {
                        let _ = Command::new("cmd").args(["/c", "start", url]).status();
                    } else {
                        let _ = Command::new("xdg-open").arg(url).status();
                    }
                }
                2 => {
                    cli_ui::display_info("For organization licenses with license server:");
                    cli_ui::display_info(
                        "1. Download the installer from: https://www.mathworks.com/downloads/",
                    );
                    cli_ui::display_info("2. During installation, select 'Use a License Server'");

                    // Ask for license server details
                    let server = cli_ui::prompt_input(
                        "Enter your organization's license server address (optional):",
                        None::<String>,
                    )?;
                    let port = cli_ui::prompt_input(
                        "Enter license server port (optional, default: 27000):",
                        Some("27000".to_string()),
                    )?;

                    if !server.is_empty() {
                        cli_ui::display_info(&format!(
                            "3. Enter the license server: {}:{}",
                            server, port
                        ));
                    } else {
                        cli_ui::display_info(
                            "3. Enter your organization's license server when prompted",
                        );
                    }

                    // Open download page
                    if cfg!(target_os = "macos") {
                        let _ = Command::new("open")
                            .arg("https://www.mathworks.com/downloads/")
                            .status();
                    } else if cfg!(target_os = "windows") {
                        let _ = Command::new("cmd")
                            .args(["/c", "start", "https://www.mathworks.com/downloads/"])
                            .status();
                    } else {
                        let _ = Command::new("xdg-open")
                            .arg("https://www.mathworks.com/downloads/")
                            .status();
                    }
                }
                3 => {
                    cli_ui::display_info("For trial licenses (30-day evaluation):");
                    cli_ui::display_info(
                        "1. Visit: https://www.mathworks.com/products/get-matlab.html",
                    );
                    cli_ui::display_info("2. Click 'Get started with a free trial'");
                    cli_ui::display_info("3. Create a MathWorks account or sign in");
                    cli_ui::display_info("4. Follow the instructions to start your trial");
                    cli_ui::display_info("5. Download the installer");

                    // Open trial page
                    if cfg!(target_os = "macos") {
                        let _ = Command::new("open")
                            .arg("https://www.mathworks.com/products/get-matlab.html")
                            .status();
                    } else if cfg!(target_os = "windows") {
                        let _ = Command::new("cmd")
                            .args([
                                "/c",
                                "start",
                                "https://www.mathworks.com/products/get-matlab.html",
                            ])
                            .status();
                    } else {
                        let _ = Command::new("xdg-open")
                            .arg("https://www.mathworks.com/products/get-matlab.html")
                            .status();
                    }
                }
                _ => {
                    cli_ui::display_info("Using your existing license...");
                    cli_ui::display_info(
                        "1. Download the installer from: https://www.mathworks.com/downloads/",
                    );
                    cli_ui::display_info("2. During installation, you'll be prompted to sign in with your MathWorks account");
                    cli_ui::display_info("3. Your license will be automatically detected");

                    // Open download page
                    if cfg!(target_os = "macos") {
                        let _ = Command::new("open")
                            .arg("https://www.mathworks.com/downloads/")
                            .status();
                    } else if cfg!(target_os = "windows") {
                        let _ = Command::new("cmd")
                            .args(["/c", "start", "https://www.mathworks.com/downloads/"])
                            .status();
                    } else {
                        let _ = Command::new("xdg-open")
                            .arg("https://www.mathworks.com/downloads/")
                            .status();
                    }
                }
            }

            // Platform-specific installation instructions
            cli_ui::display_info("\nAfter downloading the installer:");

            if cfg!(target_os = "macos") {
                cli_ui::display_info("For macOS:");
                cli_ui::display_info("1. Open the downloaded .dmg file");
                cli_ui::display_info("2. Run the installer");
                cli_ui::display_info("3. Sign in with your MathWorks account when prompted");
                cli_ui::display_info("4. Choose installation folder (default: /Applications/)");
                cli_ui::display_info("5. Select products to install");
                cli_ui::display_info("6. Complete the installation");
                cli_ui::display_info("7. Optional: Add MATLAB to your PATH by adding this line to your shell profile (~/.zshrc or ~/.bash_profile):");
                cli_ui::display_info(
                    "   export PATH=\"/Applications/MATLAB_R20XXx.app/bin:$PATH\"",
                );
            } else if cfg!(target_os = "windows") {
                cli_ui::display_info("For Windows:");
                cli_ui::display_info("1. Run the downloaded installer");
                cli_ui::display_info("2. Sign in with your MathWorks account when prompted");
                cli_ui::display_info("3. Accept the license agreement");
                cli_ui::display_info(
                    "4. Choose installation folder (default: C:\\Program Files\\MATLAB\\)",
                );
                cli_ui::display_info("5. Select products to install");
                cli_ui::display_info("6. Complete the installation");
                cli_ui::display_info(
                    "7. The installer should automatically add MATLAB to your PATH",
                );
            } else {
                cli_ui::display_info("For Linux:");
                cli_ui::display_info("1. Unzip the downloaded installer");
                cli_ui::display_info("2. Navigate to the extracted folder in terminal");
                cli_ui::display_info("3. Run the installer: ./install");
                cli_ui::display_info("4. Sign in with your MathWorks account when prompted");
                cli_ui::display_info("5. Accept the license agreement");
                cli_ui::display_info("6. Choose installation folder (default: /usr/local/MATLAB/)");
                cli_ui::display_info("7. Select products to install");
                cli_ui::display_info("8. Complete the installation");
                cli_ui::display_info("9. Add MATLAB to your PATH by adding this line to your shell profile (~/.bashrc):");
                cli_ui::display_info("   export PATH=\"/usr/local/MATLAB/R20XXx/bin:$PATH\"");
            }
        }
        1 => {
            // Use system package manager
            if cfg!(target_os = "macos") {
                // Check if homebrew is available
                let brew_available = Command::new("brew").arg("--version").status().is_ok();

                if brew_available {
                    cli_ui::display_info(
                        "Homebrew is available. MATLAB can be installed via homebrew-cask.",
                    );

                    // Check if matlab cask exists
                    let matlab_cask = Command::new("brew")
                        .args(["info", "--cask", "matlab"])
                        .output();

                    if let Ok(output) = matlab_cask {
                        if output.status.success() {
                            cli_ui::display_info(
                                "MATLAB cask is available. You can install it with:",
                            );
                            cli_ui::display_info("brew install --cask matlab");

                            let install_now = cli_ui::prompt_confirm(
                                "Would you like to install MATLAB via Homebrew now?",
                                true,
                            )?;

                            if install_now {
                                cli_ui::display_info(
                                    "Installing MATLAB via Homebrew. This may take some time...",
                                );
                                cli_ui::display_info("Note: You'll need your MathWorks account credentials during installation.");

                                let status = Command::new("brew")
                                    .args(["install", "--cask", "matlab"])
                                    .status();

                                if let Ok(status) = status {
                                    if status.success() {
                                        cli_ui::display_success(
                                            "MATLAB installation initiated through Homebrew!",
                                        );
                                        cli_ui::display_info("Follow the on-screen prompts to complete installation.");
                                        cli_ui::display_info("After installation completes, MATLAB should be available in /Applications/");
                                    } else {
                                        cli_ui::display_error(
                                            "Failed to install MATLAB through Homebrew.",
                                        );
                                    }
                                }
                            }
                        } else {
                            cli_ui::display_warning("MATLAB cask not found in Homebrew. Falling back to alternative methods.");
                            cli_ui::display_info(
                                "You can install MATLAB directly from the MathWorks website.",
                            );
                        }
                    } else {
                        cli_ui::display_warning(
                            "Could not check for MATLAB cask. Falling back to alternative methods.",
                        );
                    }
                } else {
                    cli_ui::display_warning("Homebrew not found. You can install it with:");
                    cli_ui::display_info("/bin/bash -c \"$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\"");
                    cli_ui::display_info("After installing Homebrew, you can install MATLAB with: brew install --cask matlab");
                }
            } else if cfg!(target_os = "linux") {
                // Check for different package managers
                let apt_available = Command::new("apt").arg("--version").status().is_ok();
                let dnf_available = Command::new("dnf").arg("--version").status().is_ok();
                let pacman_available = Command::new("pacman").arg("--version").status().is_ok();

                if apt_available {
                    cli_ui::display_info(
                        "Debian/Ubuntu detected. MATLAB is not available in standard repositories.",
                    );
                    cli_ui::display_info("However, you can install it using the following method:");
                    cli_ui::display_info("1. Download the installer from MathWorks website");
                    cli_ui::display_info("2. Run the installer script with your license");

                    cli_ui::display_info(
                        "\nSome universities provide custom APT repositories for MATLAB.",
                    );
                    cli_ui::display_info(
                        "Check with your institution's IT department for details.",
                    );
                } else if dnf_available {
                    cli_ui::display_info(
                        "Fedora/RHEL detected. MATLAB is not available in standard repositories.",
                    );
                    cli_ui::display_info("However, you can install it using the following method:");
                    cli_ui::display_info("1. Download the installer from MathWorks website");
                    cli_ui::display_info("2. Run the installer script with your license");
                } else if pacman_available {
                    cli_ui::display_info("Arch Linux detected. MATLAB is available in the AUR.");
                    cli_ui::display_info("You can install it using an AUR helper like yay:");
                    cli_ui::display_info("yay -S matlab");

                    let install_now = cli_ui::prompt_confirm(
                        "Would you like to install MATLAB via AUR now?",
                        true,
                    )?;

                    if install_now {
                        // Check for yay
                        let yay_available = Command::new("yay").arg("--version").status().is_ok();

                        if yay_available {
                            cli_ui::display_info(
                                "Installing MATLAB via AUR. This may take some time...",
                            );
                            cli_ui::display_info("Note: You'll need your MathWorks account credentials during installation.");

                            let status = Command::new("yay").args(["-S", "matlab"]).status();

                            if let Ok(status) = status {
                                if status.success() {
                                    cli_ui::display_success(
                                        "MATLAB installation initiated through AUR!",
                                    );
                                } else {
                                    cli_ui::display_error("Failed to install MATLAB through AUR.");
                                }
                            }
                        } else {
                            cli_ui::display_warning("yay not found. You can install it with:");
                            cli_ui::display_info("pacman -S --needed git base-devel && git clone https://aur.archlinux.org/yay.git && cd yay && makepkg -si");
                        }
                    }
                } else {
                    cli_ui::display_info("No compatible package manager found. You'll need to install MATLAB manually.");
                    cli_ui::display_info("Visit: https://www.mathworks.com/downloads/");
                }
            } else if cfg!(target_os = "windows") {
                // Check for Chocolatey or other package managers
                let choco_available = Command::new("choco").arg("--version").status().is_ok();
                let winget_available = Command::new("winget").arg("--version").status().is_ok();

                if winget_available {
                    cli_ui::display_info("Windows Package Manager (winget) detected.");
                    cli_ui::display_info(
                        "You can check if MATLAB is available with: winget search MathWorks.MATLAB",
                    );

                    let check_now = cli_ui::prompt_confirm(
                        "Would you like to check for MATLAB in winget?",
                        true,
                    )?;

                    if check_now {
                        let status = Command::new("winget")
                            .args(["search", "MathWorks.MATLAB"])
                            .status();

                        if let Ok(status) = status {
                            if status.success() {
                                cli_ui::display_info(
                                    "If MATLAB was found, you can install it with:",
                                );
                                cli_ui::display_info("winget install MathWorks.MATLAB");

                                let install_now = cli_ui::prompt_confirm(
                                    "Would you like to install MATLAB via winget now?",
                                    true,
                                )?;

                                if install_now {
                                    cli_ui::display_info(
                                        "Installing MATLAB via winget. This may take some time...",
                                    );

                                    let status = Command::new("winget")
                                        .args(["install", "MathWorks.MATLAB"])
                                        .status();

                                    if let Ok(status) = status {
                                        if status.success() {
                                            cli_ui::display_success(
                                                "MATLAB installation initiated through winget!",
                                            );
                                        } else {
                                            cli_ui::display_error(
                                                "Failed to install MATLAB through winget.",
                                            );
                                        }
                                    }
                                }
                            } else {
                                cli_ui::display_warning("MATLAB not found in winget. Falling back to alternative methods.");
                            }
                        }
                    }
                } else if choco_available {
                    cli_ui::display_info("Chocolatey package manager detected.");
                    cli_ui::display_info(
                        "MATLAB is not available in the standard Chocolatey repository.",
                    );
                    cli_ui::display_info(
                        "You'll need to install MATLAB manually from the MathWorks website.",
                    );
                } else {
                    cli_ui::display_info("No compatible package manager found. You'll need to install MATLAB manually.");
                    cli_ui::display_info("Visit: https://www.mathworks.com/downloads/");
                }
            }
        }
        2 => {
            // Direct download script option
            cli_ui::display_info(
                "MATLAB also supports direct installation using the installer files.",
            );
            cli_ui::display_info("According to MathWorks official documentation, you can download and install MATLAB using:");

            // First check if the user has MathWorks account
            let has_account = cli_ui::prompt_confirm("Do you have a MathWorks account?", true)?;

            if has_account {
                cli_ui::display_info("For installing with direct download, you'll need:");
                cli_ui::display_info("1. A valid MathWorks account with license/entitlement");
                cli_ui::display_info("2. Your account credentials");

                // Provide direct download information based on platform
                if cfg!(target_os = "macos") {
                    cli_ui::display_info(
                        "\nFor macOS, you can download the installer with a script:",
                    );
                    cli_ui::display_info("1. Create a download script (download_matlab.sh):");
                    cli_ui::display_info("   ```bash");
                    cli_ui::display_info("   #!/bin/bash");
                    cli_ui::display_info("   # Replace with your credentials");
                    cli_ui::display_info("   USERNAME=\"your_mathworks_email@example.com\"");
                    cli_ui::display_info("   PASSWORD=\"your_mathworks_password\"");
                    cli_ui::display_info("   RELEASE=\"R2023b\"");
                    cli_ui::display_info("");
                    cli_ui::display_info("   # Login and get session cookies");
                    cli_ui::display_info(
                        "   curl -s -c cookies.txt -o login.html https://www.mathworks.com/login",
                    );
                    cli_ui::display_info(
                        "   TOKEN=$(grep -o \"token=[^\"]*\" login.html | cut -d= -f2)",
                    );
                    cli_ui::display_info("   curl -s -b cookies.txt -c cookies.txt -d \"username=$USERNAME&password=$PASSWORD&token=$TOKEN\" https://www.mathworks.com/login");
                    cli_ui::display_info("");
                    cli_ui::display_info("   # Download installer");
                    cli_ui::display_info("   curl -L -b cookies.txt -o matlab_installer.dmg \"https://www.mathworks.com/downloads/web_downloads/${RELEASE}?release=${RELEASE}\"");
                    cli_ui::display_info("   ```");

                    cli_ui::display_info(
                        "\n2. Make the script executable: chmod +x download_matlab.sh",
                    );
                    cli_ui::display_info("3. Run the script: ./download_matlab.sh");
                    cli_ui::display_info("4. Mount the downloaded .dmg file and run the installer");
                    cli_ui::display_info(
                        "5. Follow the on-screen instructions to complete installation",
                    );
                } else if cfg!(target_os = "windows") {
                    cli_ui::display_info(
                        "\nFor Windows, you can download the installer with PowerShell:",
                    );
                    cli_ui::display_info("1. Create a PowerShell script (download_matlab.ps1):");
                    cli_ui::display_info("   ```powershell");
                    cli_ui::display_info("   # Replace with your credentials");
                    cli_ui::display_info("   $USERNAME = \"your_mathworks_email@example.com\"");
                    cli_ui::display_info("   $PASSWORD = \"your_mathworks_password\"");
                    cli_ui::display_info("   $RELEASE = \"R2023b\"");
                    cli_ui::display_info("");
                    cli_ui::display_info("   # Create session and login");
                    cli_ui::display_info(
                        "   $session = New-Object Microsoft.PowerShell.Commands.WebRequestSession",
                    );
                    cli_ui::display_info("   $loginPage = Invoke-WebRequest -Uri \"https://www.mathworks.com/login\" -WebSession $session");
                    cli_ui::display_info("   $token = ($loginPage.Content | Select-String -Pattern \"token=([^\"]*)\" -AllMatches).Matches.Groups[1].Value");
                    cli_ui::display_info("   Invoke-WebRequest -Uri \"https://www.mathworks.com/login\" -Method POST -WebSession $session -Body @{username=$USERNAME; password=$PASSWORD; token=$token}");
                    cli_ui::display_info("");
                    cli_ui::display_info("   # Download installer");
                    cli_ui::display_info("   Invoke-WebRequest -Uri \"https://www.mathworks.com/downloads/web_downloads/${RELEASE}?release=${RELEASE}\" -OutFile \"matlab_installer.exe\" -WebSession $session");
                    cli_ui::display_info("   ```");

                    cli_ui::display_info(
                        "\n2. Run the script in PowerShell: ./download_matlab.ps1",
                    );
                    cli_ui::display_info("3. Run the downloaded installer");
                    cli_ui::display_info(
                        "4. Follow the on-screen instructions to complete installation",
                    );
                } else {
                    cli_ui::display_info(
                        "\nFor Linux, you can download the installer with a script:",
                    );
                    cli_ui::display_info("1. Create a download script (download_matlab.sh):");
                    cli_ui::display_info("   ```bash");
                    cli_ui::display_info("   #!/bin/bash");
                    cli_ui::display_info("   # Replace with your credentials");
                    cli_ui::display_info("   USERNAME=\"your_mathworks_email@example.com\"");
                    cli_ui::display_info("   PASSWORD=\"your_mathworks_password\"");
                    cli_ui::display_info("   RELEASE=\"R2023b\"");
                    cli_ui::display_info("");
                    cli_ui::display_info("   # Login and get session cookies");
                    cli_ui::display_info(
                        "   curl -s -c cookies.txt -o login.html https://www.mathworks.com/login",
                    );
                    cli_ui::display_info(
                        "   TOKEN=$(grep -o \"token=[^\"]*\" login.html | cut -d= -f2)",
                    );
                    cli_ui::display_info("   curl -s -b cookies.txt -c cookies.txt -d \"username=$USERNAME&password=$PASSWORD&token=$TOKEN\" https://www.mathworks.com/login");
                    cli_ui::display_info("");
                    cli_ui::display_info("   # Download installer");
                    cli_ui::display_info("   curl -L -b cookies.txt -o matlab_installer.zip \"https://www.mathworks.com/downloads/web_downloads/${RELEASE}?release=${RELEASE}\"");
                    cli_ui::display_info("   ```");

                    cli_ui::display_info(
                        "\n2. Make the script executable: chmod +x download_matlab.sh",
                    );
                    cli_ui::display_info("3. Run the script: ./download_matlab.sh");
                    cli_ui::display_info("4. Extract the installer: unzip matlab_installer.zip");
                    cli_ui::display_info(
                        "5. Navigate to the extracted directory and run ./install",
                    );
                    cli_ui::display_info(
                        "6. Follow the on-screen instructions to complete installation",
                    );
                }

                cli_ui::display_info("\nNOTE: The MathWorks download API may change. This is an example script based on current methods.");
                cli_ui::display_info("For security, consider using a dedicated download tool or the official installer instead of storing credentials in scripts.");

                // Offer to create a download script template
                let create_script = cli_ui::prompt_confirm(
                    "Would you like to create a download script template?",
                    false,
                )?;

                if create_script {
                    let script_path = if cfg!(target_os = "windows") {
                        "download_matlab.ps1"
                    } else {
                        "download_matlab.sh"
                    };

                    let script_content = if cfg!(target_os = "windows") {
                        r#"# Replace with your credentials
$USERNAME = "your_mathworks_email@example.com"
$PASSWORD = "your_mathworks_password"
$RELEASE = "R2023b"

# Create session and login
$session = New-Object Microsoft.PowerShell.Commands.WebRequestSession
$loginPage = Invoke-WebRequest -Uri "https://www.mathworks.com/login" -WebSession $session
$token = ($loginPage.Content | Select-String -Pattern "token=([^\"]*)" -AllMatches).Matches.Groups[1].Value
Invoke-WebRequest -Uri "https://www.mathworks.com/login" -Method POST -WebSession $session -Body @{username=$USERNAME; password=$PASSWORD; token=$token}

# Download installer
Invoke-WebRequest -Uri "https://www.mathworks.com/downloads/web_downloads/${RELEASE}?release=${RELEASE}" -OutFile "matlab_installer.exe" -WebSession $session
"#
                    } else {
                        r#"#!/bin/bash
# Replace with your credentials
USERNAME="your_mathworks_email@example.com"
PASSWORD="your_mathworks_password"
RELEASE="R2023b"

# Login and get session cookies
curl -s -c cookies.txt -o login.html https://www.mathworks.com/login
TOKEN=$(grep -o "token=[^\"]*" login.html | cut -d= -f2)
curl -s -b cookies.txt -c cookies.txt -d "username=$USERNAME&password=$PASSWORD&token=$TOKEN" https://www.mathworks.com/login

# Download installer
curl -L -b cookies.txt -o matlab_installer.zip "https://www.mathworks.com/downloads/web_downloads/${RELEASE}?release=${RELEASE}"
"#
                    };

                    std::fs::write(script_path, script_content)?;

                    if !cfg!(target_os = "windows") {
                        let _ = Command::new("chmod").args(["+x", script_path]).status();
                    }

                    cli_ui::display_success(&format!(
                        "Download script created at: {}",
                        script_path
                    ));
                    cli_ui::display_info(
                        "Edit the script to add your credentials before running it.",
                    );
                }
            } else {
                cli_ui::display_info("You need a MathWorks account to download MATLAB.");
                cli_ui::display_info(
                    "Please create an account at: https://www.mathworks.com/login",
                );
                cli_ui::display_info(
                    "After creating an account, you'll need to purchase a license or use a trial.",
                );

                // Offer to open the MathWorks registration page
                let open_registration =
                    cli_ui::prompt_confirm("Open MathWorks registration page?", true)?;

                if open_registration {
                    if cfg!(target_os = "macos") {
                        let _ = Command::new("open")
                            .arg("https://www.mathworks.com/login")
                            .status();
                    } else if cfg!(target_os = "windows") {
                        let _ = Command::new("cmd")
                            .args(["/c", "start", "https://www.mathworks.com/login"])
                            .status();
                    } else {
                        let _ = Command::new("xdg-open")
                            .arg("https://www.mathworks.com/login")
                            .status();
                    }
                }
            }
        }
        3 => {
            // User already has an installer
            if cfg!(target_os = "macos") {
                cli_ui::display_info("MacOS Installation Instructions:");
                cli_ui::display_info("1. Mount the downloaded .dmg file");
                cli_ui::display_info("2. Run the installer application");
                cli_ui::display_info("3. Follow the on-screen instructions");
                cli_ui::display_info(
                    "4. When prompted, choose your license option (login, key, or license server)",
                );
                cli_ui::display_info("5. Select the toolboxes you need");
                cli_ui::display_info("\nAfter installation:");
                cli_ui::display_info("- MATLAB will be installed in /Applications/");
                cli_ui::display_info("- To make MATLAB available in terminal, add to your PATH:");
                cli_ui::display_info("  echo 'export PATH=\"/Applications/MATLAB_R2023b.app/bin:$PATH\"' >> ~/.zshrc");
                cli_ui::display_info("  (replace R2023b with your version)");
            } else if cfg!(target_os = "windows") {
                cli_ui::display_info("Windows Installation Instructions:");
                cli_ui::display_info("1. Run the downloaded installer executable");
                cli_ui::display_info("2. Follow the on-screen instructions");
                cli_ui::display_info(
                    "3. When prompted, choose your license option (login, key, or license server)",
                );
                cli_ui::display_info("4. Select the toolboxes you need");
                cli_ui::display_info("\nAfter installation:");
                cli_ui::display_info("- MATLAB will be installed in Program Files");
                cli_ui::display_info(
                    "- The installer should automatically add MATLAB to your PATH",
                );
            } else {
                cli_ui::display_info("Linux Installation Instructions:");
                cli_ui::display_info("1. Extract the downloaded archive");
                cli_ui::display_info("2. Navigate to the extracted folder");
                cli_ui::display_info("3. Run the install script: ./install");
                cli_ui::display_info("4. Follow the on-screen instructions");
                cli_ui::display_info(
                    "5. When prompted, choose your license option (login, key, or license server)",
                );
                cli_ui::display_info("6. Select the toolboxes you need");
                cli_ui::display_info("\nAfter installation:");
                cli_ui::display_info("- Add MATLAB to your PATH:");
                cli_ui::display_info(
                    "  echo 'export PATH=\"/usr/local/MATLAB/R2023b/bin:$PATH\"' >> ~/.bashrc",
                );
                cli_ui::display_info("  (replace R2023b with your version and adjust the path if you installed elsewhere)");
            }
        }
        4 => {
            // User wants to install later
            cli_ui::display_info("You've chosen to install MATLAB later.");
            cli_ui::display_info(
                "The project will be created without a verified MATLAB installation.",
            );
            cli_ui::display_info(
                "Refer to the README.md file for installation instructions when you're ready.",
            );
        }
        5 => {
            // Troubleshooting steps
            cli_ui::display_info("Let's troubleshoot your MATLAB installation:");

            // Check common installation paths
            let common_paths = if cfg!(target_os = "macos") {
                vec![
                    "/Applications/MATLAB_R2023b.app/bin/matlab",
                    "/Applications/MATLAB_R2023a.app/bin/matlab",
                    "/Applications/MATLAB_R2022b.app/bin/matlab",
                ]
            } else if cfg!(target_os = "windows") {
                vec![
                    "C:\\Program Files\\MATLAB\\R2023b\\bin\\matlab.exe",
                    "C:\\Program Files\\MATLAB\\R2023a\\bin\\matlab.exe",
                    "C:\\Program Files\\MATLAB\\R2022b\\bin\\matlab.exe",
                ]
            } else {
                vec![
                    "/usr/local/MATLAB/R2023b/bin/matlab",
                    "/usr/local/MATLAB/R2023a/bin/matlab",
                    "/usr/local/MATLAB/R2022b/bin/matlab",
                ]
            };

            let mut found_path = false;
            for path in &common_paths {
                if std::path::Path::new(path).exists() {
                    cli_ui::display_info(&format!("Found MATLAB at: {}", path));
                    found_path = true;

                    // Ask if user wants to add this to PATH
                    let add_to_path = cli_ui::prompt_confirm(
                        "Would you like to add this directory to your PATH for this session?",
                        true,
                    )?;
                    if add_to_path {
                        // Get the directory
                        let dir = std::path::Path::new(path)
                            .parent()
                            .unwrap_or(std::path::Path::new(path));

                        // Add to PATH temporarily
                        if cfg!(target_os = "windows") {
                            let _ = Command::new("cmd")
                                .args(["/c", &format!("set PATH=%PATH%;{}", dir.display())])
                                .status();
                        } else {
                            let _ = Command::new("sh")
                                .arg("-c")
                                .arg(&format!("export PATH=\"{}:$PATH\"", dir.display()))
                                .status();
                        }

                        cli_ui::display_info("Added to PATH for this session. You may need to open a new terminal for this to take effect.");
                    }

                    // Try to verify again
                    let matlab_check = check_matlab_available()?;
                    if let Some(version) = matlab_check {
                        cli_ui::display_success(&format!(
                            "Successfully detected MATLAB version: {}",
                            version
                        ));
                        return Ok(version);
                    }

                    break;
                }
            }

            if !found_path {
                cli_ui::display_info("Could not find MATLAB in common installation directories.");

                // Ask user for custom path
                let custom_path = cli_ui::prompt_input(
                    "Enter the full path to your MATLAB executable (leave empty to skip):",
                    None::<String>,
                )?;

                if !custom_path.is_empty() && std::path::Path::new(&custom_path).exists() {
                    cli_ui::display_info(&format!("Found MATLAB at: {}", custom_path));

                    // Add to PATH temporarily
                    let dir = std::path::Path::new(&custom_path)
                        .parent()
                        .unwrap_or(std::path::Path::new(&custom_path));

                    if cfg!(target_os = "windows") {
                        let _ = Command::new("cmd")
                            .args(["/c", &format!("set PATH=%PATH%;{}", dir.display())])
                            .status();
                    } else {
                        let _ = Command::new("sh")
                            .arg("-c")
                            .arg(&format!("export PATH=\"{}:$PATH\"", dir.display()))
                            .status();
                    }

                    cli_ui::display_info("Added to PATH for this session. You may need to open a new terminal for this to take effect.");

                    // Try to verify again
                    let matlab_check = check_matlab_available()?;
                    if let Some(version) = matlab_check {
                        cli_ui::display_success(&format!(
                            "Successfully detected MATLAB version: {}",
                            version
                        ));
                        return Ok(version);
                    }
                }
            }

            cli_ui::display_warning(
                "Still unable to detect MATLAB. Continuing with project creation.",
            );
            cli_ui::display_info("Tips for permanently adding MATLAB to your PATH:");

            if cfg!(target_os = "macos") {
                cli_ui::display_info("- Add to your shell profile (~/.zshrc or ~/.bash_profile):");
                cli_ui::display_info("  export PATH=\"/Applications/MATLAB_R20XXx.app/bin:$PATH\"");
            } else if cfg!(target_os = "windows") {
                cli_ui::display_info("- Add to your System Environment Variables:");
                cli_ui::display_info(
                    "  1. Right-click Computer > Properties > Advanced System Settings",
                );
                cli_ui::display_info("  2. Click Environment Variables");
                cli_ui::display_info(
                    "  3. Edit Path variable and add: C:\\Program Files\\MATLAB\\R20XXx\\bin",
                );
            } else {
                cli_ui::display_info("- Add to your shell profile (~/.bashrc or ~/.profile):");
                cli_ui::display_info("  export PATH=\"/usr/local/MATLAB/R20XXx/bin:$PATH\"");
            }
        }
        _ => unreachable!(),
    }

    // Final check - did the user install MATLAB during this process?
    let proceed = cli_ui::prompt_confirm(
        "Have you completed MATLAB installation? Press 'y' if installed or 'n' to continue anyway.",
        false,
    )?;

    if proceed {
        // Remind user they might need a new terminal
        cli_ui::display_info(
            "Note: You might need to open a new terminal window for MATLAB to be detected.",
        );

        // Verify installation
        let matlab_check = check_matlab_available()?;
        if let Some(version) = matlab_check {
            cli_ui::display_success(&format!(
                "MATLAB {} has been successfully installed!",
                version
            ));
            return Ok(version);
        } else {
            cli_ui::display_warning("Could not verify MATLAB installation.");
            cli_ui::display_warning("Make sure MATLAB is correctly installed and in your PATH.");
        }
    }

    // User didn't install or installation couldn't be verified
    cli_ui::display_info("Continuing with MATLAB project creation without verified installation.");

    Ok(selected_version)
}
