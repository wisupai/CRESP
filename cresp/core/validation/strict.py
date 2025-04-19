"""Strict validation implementation."""

from pathlib import Path

from ..utils import calculate_artifact_hash


def validate_strict(artifact_path: Path, reference_hash: str) -> tuple[bool, str]:
    """Validate artifact using strict hash comparison.

    Args:
        artifact_path: Path to the artifact.
        reference_hash: The expected hash value.

    Returns:
        (bool, str): (Success flag, Validation message).
    """
    current_hash = calculate_artifact_hash(artifact_path)
    if current_hash == reference_hash:
        return True, "Strict validation passed: exact hash match"
    return False, "Strict validation failed: hash mismatch"
