"""Utility functions for the cresp package."""

import hashlib
import logging
from pathlib import Path

# Import CrespConfig lazily or type hint with string to avoid circular import
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .config import CrespConfig


def create_workflow_config(
    title: str,
    authors: list[dict[str, str]],
    description: str | None = None,
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
        logging.error(f"Error loading configuration: {e}")
        return None


def calculate_file_hash(file_path: str | Path, method: str = "sha256", chunk_size: int = 8192) -> str:
    """Calculate hash for a file"""
    hash_func = getattr(hashlib, method)()
    file_path = Path(file_path)
    with open(file_path, "rb") as f:
        while chunk := f.read(chunk_size):
            hash_func.update(chunk)
    return hash_func.hexdigest()


def calculate_dir_hash(dir_path: str | Path, method: str = "sha256") -> str:
    """Calculate hash for a directory by combining hashes of all files"""
    dir_path = Path(dir_path)
    if not dir_path.is_dir():
        raise ValueError(f"Not a directory: {dir_path}")
    files = sorted(p for p in dir_path.rglob("*") if p.is_file())
    hash_func = getattr(hashlib, method)()
    for file_path in files:
        rel_path = str(file_path.relative_to(dir_path))
        hash_func.update(rel_path.encode())
        file_hash = calculate_file_hash(file_path, method)
        hash_func.update(file_hash.encode())
    return hash_func.hexdigest()


def calculate_artifact_hash(path: str | Path, method: str = "sha256") -> str:
    """Calculate hash for an artifact (file or directory)"""
    path = Path(path)
    if path.is_file():
        return calculate_file_hash(path, method)
    elif path.is_dir():
        return calculate_dir_hash(path, method)
    else:
        raise ValueError(f"Path does not exist or is not a file/directory: {path}")
