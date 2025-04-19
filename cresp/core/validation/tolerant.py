"""Tolerant validation implementation using similarity."""

from pathlib import Path

from ..utils import calculate_artifact_hash


def validate_tolerant(artifact_path: Path, reference_hash: str, similarity_threshold: float | None) -> tuple[bool, str]:
    """Validate artifact using similarity comparison (currently placeholder)."""
    # Tolerant mode: allow partial matches or similarity-based comparison
    if similarity_threshold is not None:
        # Calculate similarity based on hash (basic placeholder).
        # Real implementation should use file-type specific logic.
        current_hash = calculate_artifact_hash(artifact_path)
        if not reference_hash:  # Avoid division by zero if reference hash is empty
            return False, "Tolerant validation failed: reference hash is empty"
        # Ensure comparison is done correctly even if lengths differ
        len_min = min(len(current_hash), len(reference_hash))
        matching_chars = sum(current_hash[i] == reference_hash[i] for i in range(len_min))
        # Similarity based on the length of the reference hash
        similarity = matching_chars / len(reference_hash)

        if similarity >= similarity_threshold:
            return (
                True,
                f"Tolerant validation passed: similarity {similarity:.2f} >= threshold {similarity_threshold}",
            )
        else:
            return False, f"Tolerant validation failed: similarity {similarity:.2f} < threshold {similarity_threshold}"

    # If similarity_threshold is None, tolerant mode effectively fails
    return False, "Tolerant validation failed: similarity threshold not provided"
