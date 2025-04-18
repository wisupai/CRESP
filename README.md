# CRESP - Computational Research Environment Standardization Protocol

CRESP is a toolkit for standardizing and validating reproducibility of computational scientific experiments. This project aims to address reproducibility issues in computational science experiments and help researchers without programming expertise conduct computational research.

## Features

- Provides standardized configuration schemes for describing computational experiment environments and workflows
- Automatically records and validates outputs at each stage of the experiment
- Three validation levels:
  - `strict`: All output file hashes must match exactly
  - `standard`: Key results must match but allows differences in non-critical outputs; fixed random seeds but tolerates platform differences
  - `tolerant`: Results are acceptable within specified ranges
- Lowers technical barriers for researchers to implement reproducible experiments
- Integrates with existing tools (like Pixi)

## Installation

```bash
pip install cresp
```

## Usage

```bash
cresp --help
```

## Development Roadmap

### Phase 1: Core Modules

- Configuration Management (core/config.py)
  - YAML-based configuration schema
  - Environment validation
  - Path management

- Hash Calculation (core/hashing.py)
  - File hash computation
  - Directory hash computation
  - Hash comparison utilities

- Random Seed Management (core/seed.py)
  - Global seed control
  - Per-stage seed management
  - Framework-specific seed handling

### Phase 2: Functional Modules

- Stage Management (core/stage.py)
  - Stage definition and validation
  - Stage dependency resolution
  - Stage execution tracking

- Validation Logic (core/validation.py)
  - Multi-level validation strategies
  - Output comparison
  - Tolerance handling

### Phase 3: Public API

- Package Initialization (__init__.py)
- Public Function Exposure
- API Documentation

### Phase 4: CLI Interface

- Basic Command Structure
- Subcommand Implementation
  - init
  - validate
  - run
  - report

### Phase 5: Extended Features

- Report Generation
  - HTML reports
  - Markdown reports
  - Validation summaries

- Tool Integration
  - Pixi integration
  - Git integration
  - CI/CD support
