"""Manages the definition, execution, and reproduction of computational workflows."""

from .stage import StageFunction, SKIPPED
from .workflow import Workflow

# Expose the main classes and objects at the package level for backwards compatibility
__all__ = ["Workflow", "StageFunction", "SKIPPED"]
