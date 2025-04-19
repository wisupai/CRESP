"""Core functionality for CRESP, including configuration, workflow management, and utilities."""

from .config import ConfigBatchUpdate, CrespConfig
from .constants import DEFAULT_CONFIG_NAME
from .exceptions import ReproductionError
from .hashing import calculate_artifact_hash, validate_artifact
from .models import (
    Artifact,
    ArtifactValidation,
    Author,
    Computing,
    CrespConfigModel,
    Environment,
    Metadata,
    ReproductionConfig,
    Stage,
    ValidationRule,
)
from .seed import get_reproducible_dataloader_kwargs, set_seed
from .utils import create_workflow_config, find_config
from .workflow import StageFunction, Workflow

__all__ = [
    # Config
    "CrespConfig",
    "ConfigBatchUpdate",
    # Workflow
    "Workflow",
    "StageFunction",
    # Exceptions
    "ReproductionError",
    # Models (Exporting key models for convenience)
    "Author",
    "Computing",
    "Environment",
    "ValidationRule",
    "ArtifactValidation",
    "Artifact",
    "Stage",
    "ReproductionConfig",
    "Metadata",
    "CrespConfigModel",
    # Utils
    "create_workflow_config",
    "find_config",
    # Hashing
    "calculate_artifact_hash",
    "validate_artifact",
    # Seed
    "set_seed",
    "get_reproducible_dataloader_kwargs",
    # Constants
    "DEFAULT_CONFIG_NAME",
]
