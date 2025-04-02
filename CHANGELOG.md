# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

### Fixed
- Fixed auto-activation of UV in current terminal session after installation
- Fixed UV installation sequence to provide clearer user instructions 
- Fixed formatting error in command generation for UV installation
- Fixed MATLAB installer download script to handle platform-specific installers

### Removed
- None

### Security
- None 