"""Utility functions for the cresp package."""

from typing import Any, Dict, List, Optional

# Import CrespConfig lazily or type hint with string to avoid circular import
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import CrespConfig


def create_workflow_config(
    title: str,
    authors: List[Dict[str, str]],
    description: Optional[str] = None,
    path: str = "cresp.yaml",  # Changed default from workflow.yaml to cresp.yaml
) -> "CrespConfig":
    """Create a new workflow configuration with better defaults."""
    # We need to import CrespConfig here if not using TYPE_CHECKING
    from .config import CrespConfig

    return CrespConfig.create(
        metadata={
            "title": title,
            "authors": authors,
            "description": description or f"{title} - Reproducible research workflow",
        },
        path=path,
    )


def find_config() -> Optional["CrespConfig"]:
    """Find and load configuration.

    Returns:
        CrespConfig: Configuration object, None if not found.
    """
    # We need to import CrespConfig here if not using TYPE_CHECKING
    from .config import CrespConfig

    try:
        return CrespConfig.load()
    except FileNotFoundError:
        return None
    except Exception as e:
        # Consider using logging instead of print for errors
        import logging

        logging.error(f"Error loading configuration: {e}")
        # print(f"Error loading configuration: {e}")
        return None
