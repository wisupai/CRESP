"""Standard validation implementation with tolerance options."""

from pathlib import Path

import numpy as np

from ..utils import calculate_artifact_hash


def compare_numeric_values(
    value1: float,
    value2: float,
    tolerance_absolute: float | None = None,
    tolerance_relative: float | None = None,
) -> bool:
    """Compare two numeric values with tolerances"""
    if tolerance_absolute is not None:
        if abs(value1 - value2) <= tolerance_absolute:
            return True

    if tolerance_relative is not None:
        # Avoid division by zero if both values are zero
        if value1 == 0 and value2 == 0:
            return True
        if max(abs(value1), abs(value2)) == 0:
            return False  # Should not happen if above check passes, but for safety
        relative_diff = abs(value1 - value2) / max(abs(value1), abs(value2))
        if relative_diff <= tolerance_relative:
            return True

    # Strict equality check if no tolerances are provided or met
    if tolerance_absolute is None and tolerance_relative is None:
        return value1 == value2

    return False


def compare_arrays(
    arr1: np.ndarray,
    arr2: np.ndarray,
    tolerance_absolute: float | None = None,
    tolerance_relative: float | None = None,
) -> bool:
    """Compare two numpy arrays with tolerances"""
    if arr1.shape != arr2.shape:
        return False

    # np.allclose handles both relative and absolute tolerance
    # Set default rtol and atol if None is provided
    rtol = tolerance_relative if tolerance_relative is not None else 1e-5
    atol = tolerance_absolute if tolerance_absolute is not None else 1e-8

    # If specific tolerances were provided, use them directly
    # If not, use numpy's defaults (or check for exact match if no tolerances given)
    if tolerance_absolute is not None or tolerance_relative is not None:
        return np.allclose(arr1, arr2, rtol=rtol, atol=atol)
    else:
        # Strict equality check if no tolerances are provided
        return np.array_equal(arr1, arr2)


def validate_standard(
    artifact_path: Path,
    reference_hash: str,
    tolerance_absolute: float | None = None,
    tolerance_relative: float | None = None,
) -> tuple[bool, str]:
    """Validate artifact using standard comparison (hash fallback with optional tolerance)."""
    success = False
    message = ""
    needs_hash_comparison = True  # Default to hash comparison

    if artifact_path.is_file():
        # Determine if specialized comparison should be attempted
        attempt_specialized = False
        if artifact_path.suffix in [".csv", ".json"] and (tolerance_absolute is not None or tolerance_relative is not None):
            attempt_specialized = True
            try:
                # Attempt specialized comparison for csv/json
                with open(artifact_path) as f:
                    pass  # Fallback to hash
            except Exception:
                # Failed specialized comparison, will fallback to hash
                pass  # Keep needs_hash_comparison = True

        elif artifact_path.suffix in [".npy", ".npz"] and (tolerance_absolute is not None or tolerance_relative is not None):
            attempt_specialized = True
            try:
                # Attempt specialized comparison for npy/npz
                # Specialized comparison logic for NumPy arrays would go here.
                # Current implementation falls back to hash comparison.
                pass  # Fallback to hash
            except Exception:
                # Failed specialized comparison, will fallback to hash
                pass  # Keep needs_hash_comparison = True

        # Proceed to hash comparison if specialized comparison wasn't applicable,
        # implemented, or failed.
        if needs_hash_comparison:
            current_hash = calculate_artifact_hash(artifact_path)
            if current_hash == reference_hash:
                success = True
                # Use hash match message only if no specialized success message exists
                if not message:
                    message = "Standard validation passed: exact hash match"
            else:
                success = False
                if attempt_specialized:  # Specialized was tried/skipped and hash also failed
                    message = "Standard validation failed: specialized comparison did not succeed and hash mismatch"
                else:  # Only hash comparison was performed and it failed
                    message = "Standard validation failed: hash mismatch"

    else:  # Artifact is a directory
        # Directories always use hash comparison in standard mode
        current_hash = calculate_artifact_hash(artifact_path)
        if current_hash == reference_hash:
            success = True
            message = "Standard validation passed: exact hash match for directory"
        else:
            success = False
            message = "Standard validation failed: directory hash mismatch"

    return success, message
