# CRESP CLI Documentation

CRESP (Computational Research Environment Standardization Protocol) CLI is a command-line tool for managing and validating computational research environments.

## Commands

### `new`
Create a new CRESP project with interactive configuration.

```bash
cresp new [OPTIONS]
```

Options:
- `--name, -n <NAME>`: Project name
- `--description, -d <DESCRIPTION>`: Project description
- `--language, -l <LANGUAGE>`: Primary programming language (python, r, matlab)
- `--template, -t <TEMPLATE>`: Project template (basic, data-analysis, machine-learning, scientific)
- `--output, -o <PATH>`: Output directory (default: current directory)

### `init`
Initialize a new CRESP project in the current directory.

```bash
cresp init [OPTIONS]
```

Options:
- `--name <NAME>`: Project name
- `--description <DESCRIPTION>`: Project description
- `--language <LANGUAGE>`: Primary programming language (python, r, matlab)
- `--template <TEMPLATE>`: Project template to use

### `validate`
Validate a CRESP configuration file.

```bash
cresp validate [OPTIONS] <PATH>
```

Options:
- `--strict`: Enable strict validation mode
- `--schema <SCHEMA>`: Custom schema file path

### `verify`
Verify the current environment against a CRESP configuration.

```bash
cresp verify [OPTIONS] <PATH>
```

Options:
- `--hardware`: Verify hardware requirements
- `--software`: Verify software dependencies
- `--data`: Verify dataset availability and integrity
- `--all`: Verify all components (default)

### `export`
Export the current environment configuration to CRESP format.

```bash
cresp export [OPTIONS] <OUTPUT_PATH>
```

Options:
- `--format <FORMAT>`: Output format (toml, json)
- `--include-hardware`: Include hardware information
- `--include-software`: Include software information
- `--include-data`: Include dataset information

### `reproduce`
Reproduce a research environment from a CRESP configuration.

```bash
cresp reproduce [OPTIONS] <PATH>
```

Options:
- `--container`: Use container-based reproduction
- `--no-verify`: Skip environment verification
- `--force`: Force reproduction even if conflicts exist

### `diff`
Compare two CRESP configurations.

```bash
cresp diff [OPTIONS] <PATH1> <PATH2>
```

Options:
- `--format <FORMAT>`: Output format (text, json)
- `--ignore-whitespace`: Ignore whitespace differences

### `update`
Update a CRESP configuration to the latest version.

```bash
cresp update [OPTIONS] <PATH>
```

Options:
- `--dry-run`: Show what would be updated without making changes
- `--force`: Force update even if conflicts exist

### `completion`
Generate shell completion scripts.

```bash
cresp completion [OPTIONS] <SHELL>
```

Options:
- `--output <OUTPUT>`: Output file path

## Global Options

- `-v, --verbose`: Enable verbose output
- `-q, --quiet`: Suppress output
- `--color <WHEN>`: Color output (auto, always, never)
- `--version`: Show version information
- `-h, --help`: Show help information

## Examples

```bash
# Create a new Python project with interactive setup
cresp new --name my-research --language python

# Initialize a new Python project
cresp init --name my-research --language python

# Validate a CRESP configuration
cresp validate cresp.toml

# Verify current environment
cresp verify cresp.toml

# Export current environment
cresp export --format toml environment.toml

# Reproduce environment
cresp reproduce cresp.toml

# Compare configurations
cresp diff config1.toml config2.toml

# Update configuration
cresp update cresp.toml
``` 