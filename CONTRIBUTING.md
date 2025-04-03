# Contributing to CRESP

Thank you for your interest in contributing to the CRESP project! This guide will help you understand our development process and how to contribute effectively.

## Table of Contents

- [Contributing to CRESP](#contributing-to-cresp)
  - [Table of Contents](#table-of-contents)
  - [Code of Conduct](#code-of-conduct)
  - [Getting Started](#getting-started)
  - [Development Workflow](#development-workflow)
  - [Branch Naming Conventions](#branch-naming-conventions)
  - [Creating Branches](#creating-branches)
  - [Commit Guidelines](#commit-guidelines)
  - [Pull Request Process](#pull-request-process)
  - [Code Standards](#code-standards)

## Code of Conduct

We expect all contributors to follow our Code of Conduct. Please be respectful and constructive in all interactions.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```
   git clone https://github.com/YOUR-USERNAME/CRESP.git
   cd CRESP
   ```
3. Add the original repository as an upstream remote:
   ```
   git remote add upstream https://github.com/Wisup/CRESP.git
   ```
4. Install development dependencies:
   ```
   cargo build
   ```

## Development Workflow

We follow a branch-based development workflow:

1. Sync your fork with the upstream repository:
   ```
   git fetch upstream
   git checkout develop
   git merge upstream/develop
   ```
2. Create a new branch for your feature or fix from the `develop` branch (see [Branch Naming Conventions](#branch-naming-conventions))
3. Make your changes
4. Run tests and ensure your code builds without warnings:
   ```
   cargo build
   cargo test
   cargo fmt
   ```
5. Commit your changes following the [Commit Guidelines](#commit-guidelines)
6. Push your branch to your fork
7. Create a Pull Request to the `develop` branch

Note: The `main` branch is protected and can only be modified through approved pull requests from the `develop` branch. Direct commits to `main` are not allowed.

## Branch Naming Conventions

To maintain a clean and organized codebase, we follow these branch naming conventions:

- `main` - Main branch containing production-ready code. This branch is protected and can only be updated through pull requests.
- `develop` - Development branch containing the latest development code. All development work should start from this branch.
- `feature/[feature-name]` - New feature branches
- `bugfix/[issue-number]-[brief-description]` - Bug fix branches
- `hotfix/[issue-number]-[brief-description]` - Urgent fix branches
- `release/[version]` - Release branches
- `docs/[document-name]` - Documentation update branches
- `refactor/[component-name]` - Code refactoring branches
- `test/[test-name]` - Test-related branches

Examples:
```
feature/conda-integration
bugfix/42-fix-conda-version-detection
hotfix/57-critical-security-issue
release/v1.0.0
docs/installation-guide
refactor/package-manager
test/conda-installation
```

## Creating Branches

To create a new branch, first ensure you're on the latest `develop` branch:

```bash
git checkout develop
git pull upstream develop
```

Then create and switch to your new branch:

```bash
# For a new feature
git checkout -b feature/your-feature-name

# For a bug fix
git checkout -b bugfix/issue-number-brief-description

# For documentation update
git checkout -b docs/document-name

# For code refactoring
git checkout -b refactor/component-name

# For a hotfix (based on main branch)
git checkout main
git pull upstream main
git checkout -b hotfix/issue-number-brief-description
```

## Commit Guidelines

We use the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
<type>(<scope>): <description>
```

Types include:
- `feat`: A new feature
- `fix`: A bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring without functionality changes
- `test`: Adding or improving tests
- `chore`: Maintenance tasks

Example:
```
feat(python): add poetry support for package management
fix(conda): resolve version detection issue
docs(readme): update installation instructions
```

Keep commit messages in a single line (no line breaks).

## Pull Request Process

1. Update the README.md or documentation with details of changes if needed
2. Update the CHANGELOG.md with details of changes
3. The PR will be merged once you have the sign-off of at least one maintainer
4. Make sure all automated tests pass before requesting a review

## Code Standards

- Write clear, readable code with descriptive variable names
- Add comments for complex logic
- Use English for all code and comments
- Follow Rust's official style guidelines
- Use `rustfmt` for code formatting:
  ```bash
  cargo fmt
  ```
- Run `cargo fmt` before committing

Thank you for contributing to CRESP! 