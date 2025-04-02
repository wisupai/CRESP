use std::path::Path;
use crate::error::Result;
use crate::utils::cli_ui;
use super::super::utils::write_file;

/// Create MATLAB project with the specified configuration
pub fn create_matlab_project(project_dir: &Path) -> Result<()> {
    cli_ui::display_info("Creating MATLAB project structure...");
    // Create project structure for MATLAB
    let dirs = &[
        "src",
        "test",
        "data",
        "results", 
        "docs",
    ];
    
    for dir in dirs {
        std::fs::create_dir_all(project_dir.join(dir))?;
    }

    cli_ui::display_info("Generating MATLAB project files...");
    // Create main.m in src
    let main_m = r#"function main()
% MAIN Main function of the project
%
% This function serves as the entry point for the research project.
%
% Example:
%   main()
%
% See also: processData, analyzeResults

    disp('Hello, CRESP!');
    
    % Your research code goes here
    
    % Example workflow:
    % 1. Load data
    % data = loadData('../data/sample.mat');
    
    % 2. Process data
    % processedData = processData(data);
    
    % 3. Analyze results
    % results = analyzeResults(processedData);
    
    % 4. Save results
    % saveResults(results, '../results');
end
"#;
    write_file(&project_dir.join("src/main.m"), main_m)?;

    // Create processData.m helper function
    let process_data_m = r#"function processedData = processData(data)
% PROCESSDATA Process the raw data
%
% This function takes raw data and processes it for analysis.
%
% Args:
%   data: Raw data to process
%
% Returns:
%   processedData: Processed data ready for analysis
%
% Example:
%   data = loadData('../data/sample.mat');
%   processedData = processData(data);

    % Placeholder - replace with actual data processing
    processedData = data;
    disp('Processing data...');
end
"#;
    write_file(&project_dir.join("src/processData.m"), process_data_m)?;
    
    // Create analyzeResults.m helper function
    let analyze_results_m = r#"function results = analyzeResults(data)
% ANALYZERESULTS Analyze the processed data
%
% This function analyzes the processed data and returns results.
%
% Args:
%   data: Processed data to analyze
%
% Returns:
%   results: Analysis results
%
% Example:
%   processedData = processData(data);
%   results = analyzeResults(processedData);

    % Placeholder - replace with actual data analysis
    results = struct('data', data, 'timestamp', now);
    disp('Analyzing results...');
end
"#;
    write_file(&project_dir.join("src/analyzeResults.m"), analyze_results_m)?;

    // Create test runner
    let run_tests_m = r#"function results = runTests()
% RUNTESTS Run all tests for the project
%
% This function runs all tests and returns the results.
%
% Returns:
%   results: Test results
%
% Example:
%   results = runTests();

    disp('Running tests...');
    
    % Initialize test results
    results = struct('passed', 0, 'failed', 0, 'total', 0);
    
    % Run test_processData
    try
        test_processData();
        results.passed = results.passed + 1;
        disp('test_processData: PASSED');
    catch ME
        results.failed = results.failed + 1;
        disp(['test_processData: FAILED - ' ME.message]);
    end
    results.total = results.total + 1;
    
    % Run test_analyzeResults
    try
        test_analyzeResults();
        results.passed = results.passed + 1;
        disp('test_analyzeResults: PASSED');
    catch ME
        results.failed = results.failed + 1;
        disp(['test_analyzeResults: FAILED - ' ME.message]);
    end
    results.total = results.total + 1;
    
    % Display summary
    disp(['Test summary: ' num2str(results.passed) '/' num2str(results.total) ' tests passed']);
end

function test_processData()
    % Test the processData function
    testData = 1:10;
    result = processData(testData);
    assert(isequal(size(result), size(testData)), 'Output size should match input size');
end

function test_analyzeResults()
    % Test the analyzeResults function
    testData = 1:10;
    result = analyzeResults(testData);
    assert(isfield(result, 'data'), 'Result should have a data field');
    assert(isfield(result, 'timestamp'), 'Result should have a timestamp field');
end
"#;
    write_file(&project_dir.join("test/runTests.m"), run_tests_m)?;

    // Create MATLAB project file
    let project_prj = r#"<?xml version="1.0" encoding="UTF-8"?>
<MATLABProject xmlns="http://www.mathworks.com/MATLABProjectFile" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" version="1.0"/>
"#;
    write_file(&project_dir.join("project.prj"), project_prj)?;

    // Create startup.m
    let startup_m = r#"% STARTUP Project startup script
%
% This script runs automatically when MATLAB starts in this directory
% and sets up the project environment.

% Add src directory to the MATLAB path
addpath(genpath('src'));
addpath(genpath('test'));

disp('Project environment initialized.');
disp('Type "help main" for usage information.');
"#;
    write_file(&project_dir.join("startup.m"), startup_m)?;

    // Create README.md
    let readme = r#"# MATLAB Research Project

This is a MATLAB research project using CRESP protocol.

## Project Structure

```
.
├── src/        # Source code
├── test/       # Test scripts
├── data/       # Input data
├── results/    # Output results
├── docs/       # Documentation
├── project.prj # MATLAB project file
└── startup.m   # Project initialization script
```

## Setup

1. Start MATLAB in the project directory or run:
```matlab
cd /path/to/project
```

2. The startup.m script will automatically add the required paths. If it doesn't run automatically, execute:
```matlab
startup
```

## Usage

Run the main script:
```matlab
main
```

## Testing

Run the test suite:
```matlab
results = runTests()
```

## Adding Dependencies

For projects using MATLAB's package management, add dependencies to the project:

1. Open the project in MATLAB:
```matlab
openProject('/path/to/project/project.prj')
```

2. Use the project manager to add required MATLAB toolboxes or files.
"#;
    write_file(&project_dir.join("README.md"), readme)?;

    // Create .gitignore
    let gitignore = r#"# MATLAB specific
*.asv
*.mex*
*.mlx
*.mat
*.fig
slprj/
sccprj/
codegen/
*.slxc
.SimulinkProject/
*.autosave
*.slx.r*
*.mdl.r*

# Results directory
results/

# Avoid large data files
data/**/*.mat
data/**/*.csv
data/**/*.xlsx
"#;
    write_file(&project_dir.join(".gitignore"), gitignore)?;

    Ok(())
} 