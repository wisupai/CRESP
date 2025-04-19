"""
CRESP hashing module

This module provides utilities for calculating and comparing hashes of files and directories.
"""

import hashlib
import json
from pathlib import Path

import numpy as np


def calculate_file_hash(file_path: str | Path, method: str = "sha256", chunk_size: int = 8192) -> str:
    """Calculate hash for a file

    Args:
        file_path: Path to the file
        method: Hash method (md5, sha1, sha256)
        chunk_size: Size of chunks to read

    Returns:
        Hash string
    """
    hash_func = getattr(hashlib, method)()

    with open(file_path, "rb") as f:
        while chunk := f.read(chunk_size):
            hash_func.update(chunk)

    return hash_func.hexdigest()


def calculate_dir_hash(dir_path: str | Path, method: str = "sha256") -> str:
    """Calculate hash for a directory by combining hashes of all files

    Args:
        dir_path: Path to the directory
        method: Hash method

    Returns:
        Hash string
    """
    dir_path = Path(dir_path)
    if not dir_path.is_dir():
        raise ValueError(f"Not a directory: {dir_path}")

    # Get all files recursively and sort them for deterministic order
    files = sorted(p for p in dir_path.rglob("*") if p.is_file())

    # Calculate hash for each file and combine them
    hash_func = getattr(hashlib, method)()

    for file_path in files:
        # Add relative path to hash for structure sensitivity
        rel_path = str(file_path.relative_to(dir_path))
        hash_func.update(rel_path.encode())

        # Add file content hash
        file_hash = calculate_file_hash(file_path, method)
        hash_func.update(file_hash.encode())

    return hash_func.hexdigest()


def calculate_artifact_hash(path: str | Path, method: str = "sha256") -> str:
    """Calculate hash for an artifact (file or directory)

    Args:
        path: Path to the artifact
        method: Hash method

    Returns:
        Hash string
    """
    path = Path(path)
    if path.is_file():
        return calculate_file_hash(path, method)
    elif path.is_dir():
        return calculate_dir_hash(path, method)
    else:
        raise ValueError(f"Path does not exist: {path}")


def compare_numeric_values(
    value1: float,
    value2: float,
    tolerance_absolute: float | None = None,
    tolerance_relative: float | None = None,
) -> bool:
    """Compare two numeric values with tolerances

    Args:
        value1: First value
        value2: Second value
        tolerance_absolute: Absolute tolerance
        tolerance_relative: Relative tolerance

    Returns:
        True if values match within tolerances
    """
    if tolerance_absolute is not None:
        if abs(value1 - value2) <= tolerance_absolute:
            return True

    if tolerance_relative is not None:
        relative_diff = abs(value1 - value2) / max(abs(value1), abs(value2))
        if relative_diff <= tolerance_relative:
            return True

    return False


def compare_arrays(
    arr1: np.ndarray,
    arr2: np.ndarray,
    tolerance_absolute: float | None = None,
    tolerance_relative: float | None = None,
) -> bool:
    """Compare two numpy arrays with tolerances

    Args:
        arr1: First array
        arr2: Second array
        tolerance_absolute: Absolute tolerance
        tolerance_relative: Relative tolerance

    Returns:
        True if arrays match within tolerances
    """
    if arr1.shape != arr2.shape:
        return False

    if tolerance_absolute is not None:
        if np.allclose(arr1, arr2, atol=tolerance_absolute):
            return True

    if tolerance_relative is not None:
        if np.allclose(arr1, arr2, rtol=tolerance_relative):
            return True

    return False


def validate_artifact(
    artifact_path: str | Path,
    reference_hash: str,
    validation_type: str = "strict",
    tolerance_absolute: float | None = None,
    tolerance_relative: float | None = None,
    similarity_threshold: float | None = None,
) -> tuple[bool, str]:
    """Validate an artifact against its reference hash

    Args:
        artifact_path: Path to the artifact
        reference_hash: Reference hash to compare against
        validation_type: Validation type (strict, standard, tolerant)
        tolerance_absolute: Absolute tolerance for numeric comparisons
        tolerance_relative: Relative tolerance for numeric comparisons
        similarity_threshold: Similarity threshold for content comparison

    Returns:
        (bool, str): (Success flag, Validation message)
    """
    artifact_path = Path(artifact_path)

    if not artifact_path.exists():
        return False, f"Artifact does not exist: {artifact_path}"

    try:
        if validation_type == "strict":
            # Strict mode: exact hash match
            current_hash = calculate_artifact_hash(artifact_path)
            if current_hash == reference_hash:
                return True, "Strict validation passed: exact hash match"
            return False, "Strict validation failed: hash mismatch"

        elif validation_type == "standard":
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
                            current_data = f.read()
                        current_json = json.loads(current_data)
                        # Assuming reference_hash is the reference data/value for comparison (problematic)
                        reference_json = json.loads(reference_hash)
                        if isinstance(current_json, int | float) and isinstance(reference_json, int | float):
                            if compare_numeric_values(current_json, reference_json, tolerance_absolute, tolerance_relative):
                                success = True
                                message = "Standard validation passed: numeric match within tolerances"
                                needs_hash_comparison = False  # Specialized comparison succeeded
                        # Add more sophisticated json comparison logic here if needed
                    except Exception:
                        # Failed specialized comparison, will fallback to hash
                        pass  # Keep needs_hash_comparison = True

                elif artifact_path.suffix in [".npy", ".npz"] and (tolerance_absolute is not None or tolerance_relative is not None):
                    attempt_specialized = True
                    try:
                        # Attempt specialized comparison for npy/npz
                        current_arr = np.load(artifact_path)
                        # Assuming reference_hash is path to reference array (problematic)
                        reference_arr = np.load(reference_hash)
                        if compare_arrays(current_arr, reference_arr, tolerance_absolute, tolerance_relative):
                            success = True
                            message = "Standard validation passed: array match within tolerances"
                            needs_hash_comparison = False  # Specialized comparison succeeded
                    except Exception:
                        # Failed specialized comparison, will fallback to hash
                        pass  # Keep needs_hash_comparison = True

                # If specialized comparison wasn't attempted (e.g. .txt, other types, or no tolerance set)
                # OR if it was attempted but failed, proceed to hash comparison if needed.
                if needs_hash_comparison:
                    current_hash = calculate_artifact_hash(artifact_path)
                    if current_hash == reference_hash:
                        success = True
                        # Use hash match message only if no specialized success message exists
                        if not message:
                            message = "Standard validation passed: exact hash match"
                    else:
                        success = False
                        if attempt_specialized:  # Specialized was tried but failed, and hash also failed
                            message = "Standard validation failed: specialized comparison failed and hash mismatch"
                        else:  # Only hash comparison was performed (e.g. for .txt) and it failed
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

        elif validation_type == "tolerant":
            # Tolerant mode: allow partial matches or similarity-based comparison
            if similarity_threshold is not None:
                # Implement similarity comparison based on file type
                # This is a placeholder for more sophisticated comparison methods
                current_hash = calculate_artifact_hash(artifact_path)
                similarity = sum(a == b for a, b in zip(current_hash, reference_hash, strict=False)) / len(reference_hash)

                if similarity >= similarity_threshold:
                    return (
                        True,
                        f"Tolerant validation passed: similarity {similarity:.2f} >= threshold {similarity_threshold}",
                    )

            return False, "Tolerant validation failed: below similarity threshold"

        else:
            return False, f"Unknown validation type: {validation_type}"

    except Exception as e:
        return False, f"Validation error: {str(e)}"
