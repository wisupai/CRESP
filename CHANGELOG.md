# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 未发布
与0.1.0-dev.1相同，将在开发稳定后从develop分支合并到main分支发布。

## [0.1.0-dev.1] - 2024-04-03

### Added
- Initial project setup
- Basic project structure
- Python project creation support
- R project creation support
- MATLAB project creation support
- Enhanced CLI user interface with Dialoguer and Console
- Interactive select menus, confirmations, and styled output
- MATLAB version detection and configuration functions
- Automated MATLAB installer download and execution with credential handling
- Platform-specific MATLAB installation assistance (DMG mounting, EXE execution, ZIP extraction)
- Added direnv detection and configuration guidance for Python projects
- Added conda version checking with update recommendations
- Added more comprehensive UV usage examples
- docs: Added CONTRIBUTING.md with comprehensive contribution guidelines including branch naming conventions and workflow

### Changed
- Improved R project creation with better rig package manager handling
- Refactored command line interface for a more user-friendly experience
- Enhanced MATLAB project initialization with better directory structure
- Optimized UV package manager integration with improved PATH handling and user guidance
- Enhanced installation process for UV to auto-source environment files
- Added detailed UV installation path information and performance hints
- Improved MATLAB installation process with comprehensive official MathWorks guidelines for different license types and platforms
- Enhanced project description prompt with skip option
- Improved code formatting and error messages in R template
- Automated MATLAB installation process with direct credential input and installer execution
- Enhanced MATLAB installation process with automatic script execution and cleanup
- Improved security handling for MATLAB installation credentials
- Enhanced Python project README with direnv usage instructions
- Improved mirror configuration for UV package manager
- Removed automatic generation of .envrc files for simplicity and compatibility
- Modified Python project creation to enforce Conda as the environment management tool with standardized package manager combinations
- Simplified Python package management options by removing pip option, limiting choices to Conda only, Conda+Poetry, or Conda+UV
- Standardized R project creation to exclusively use conda+renv for environment management, consistent with Python's approach
- Refined Python project creation by removing system Python option and ensuring all projects use conda-managed environments
- Removed "Conda only" package management option and made UV the recommended package manager to ensure compatibility with PyPI-only packages

### Fixed
- Fixed auto-activation of UV in current terminal session after installation
- Fixed UV installation sequence to provide clearer user instructions 
- Fixed formatting error in command generation for UV installation
- Fixed MATLAB installer download script to handle platform-specific installers
- Fixed compiler warnings by removing unused code and marking unused variables
- Fixed Python mirror configuration issue when using UV package manager
- Fixed missing direnv documentation and guidance after project creation

### Removed
- None
- Automatic .envrc file generation in Python projects for better compatibility 
- Support for pip as a standalone package manager, focusing on more performant alternatives
- Multiple R installation methods (rig, system package managers, etc.) in favor of a standardized conda approach
- System Python option from Python project creation, ensuring all environments use conda for better isolation and reproducibility 