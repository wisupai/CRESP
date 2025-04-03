# CRESP

CRESP (Computational Research Environment Standardization Protocol) is a command-line tool designed to standardize and simplify the creation, management, and reproducibility of computational research environments.

## Features

- Create standardized project environments for Python, R, and MATLAB
- Ensure reproducibility of computational experiments
- Manage project dependencies across different platforms
- Simplify environment setup for computational research

## Installation

### Using Installation Scripts

#### Linux and macOS

```bash
# Install the latest stable version
curl -sSL https://raw.githubusercontent.com/wisupai/CRESP/main/install.sh | bash

# Install a specific version
curl -sSL https://raw.githubusercontent.com/wisupai/CRESP/main/install.sh | bash -s v0.1.0
```

#### Windows (PowerShell)

```powershell
# Install the latest stable version
irm https://raw.githubusercontent.com/wisupai/CRESP/main/install.ps1 | iex

# Install a specific version
irm https://raw.githubusercontent.com/wisupai/CRESP/main/install.ps1 | iex -Args "v0.1.0"
```

### Manual Installation

1. Download the appropriate binary for your platform from the [Releases](https://github.com/wisupai/CRESP/releases) page
2. Extract the archive and move the binary to a location in your PATH

## Documentation

For detailed documentation and usage examples, please visit the [CRESP Documentation](https://github.com/wisupai/CRESP/wiki).

## Contributing

For information on how to contribute to this project, including our branch naming conventions and development workflow, please see [CONTRIBUTING.md](CONTRIBUTING.md).

## License

CRESP is licensed under [MIT License](LICENSE).

## About

CRESP is part of the Rescience Lab product from Wisup AI Ltd., aimed at facilitating computational research reproducibility and standardization.