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