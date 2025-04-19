"""Utility functions for workflow module."""

from pathlib import Path
from typing import Any, Optional, Tuple, Union


def resolve_output_path(path_str: str, is_shared: bool, active_output_dir: Path, shared_data_dir: Path) -> Path:
    """Resolve an output path based on whether it's shared or mode-specific.

    Args:
        path_str: The relative path string
        is_shared: Whether the path is in the shared directory
        active_output_dir: Path to the active output directory
        shared_data_dir: Path to the shared data directory

    Returns:
        The resolved absolute path
    """
    if is_shared:
        resolved_path = shared_data_dir / path_str
    else:
        resolved_path = active_output_dir / path_str

    # Ensure parent directory exists
    resolved_path.parent.mkdir(parents=True, exist_ok=True)

    return resolved_path


def get_output_scope_and_path(output_decl: Union[str, dict[str, Any]]) -> Tuple[Optional[str], bool]:
    """Extract path string and scope (shared/mode-specific) from output declaration.

    Args:
        output_decl: String path or dictionary with path and other settings

    Returns:
        Tuple of (path_str, is_shared)
    """
    path_str: Optional[str] = None
    is_shared = False  # Default to mode-specific

    if isinstance(output_decl, str):
        path_str = output_decl
        # Apply heuristic: path without separator is likely shared
        if "/" not in path_str and "\\" not in path_str:
            is_shared = True
    elif isinstance(output_decl, dict) and "path" in output_decl:
        path_str = output_decl["path"]
        # Check for shared flag with fallback to heuristic
        if "shared" in output_decl:
            is_shared = bool(output_decl.get("shared", False))
        elif path_str and "/" not in path_str and "\\" not in path_str:
            is_shared = True

    return path_str, is_shared
