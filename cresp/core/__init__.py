"""Core functionality for CRESP, including configuration, workflow management, and utilities."""

from .config import CrespConfig, ConfigBatchUpdate
from .workflow import Workflow, StageFunction
from .exceptions import ReproductionError
from .models import (
    Author,
    Computing,
    Environment,
    ValidationRule,
    ArtifactValidation,
    Artifact,
    Stage,
    ReproductionConfig,
    Metadata,
    CrespConfigModel,
)
from .utils import create_workflow_config, find_config
from .hashing import calculate_artifact_hash, validate_artifact
from .seed import set_seed, get_reproducible_dataloader_kwargs
from .constants import DEFAULT_CONFIG_NAME

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
