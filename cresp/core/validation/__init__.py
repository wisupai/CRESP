"""Validation module entry point."""

from pathlib import Path

# Import necessary functions
from ..utils import calculate_artifact_hash  # Updated import
from .standard import validate_standard
from .tolerant import validate_tolerant


def validate_artifact(
    artifact_path: str | Path,
    reference_hash: str,
    validation_type: str = "strict",
    tolerance_absolute: float | None = None,
    tolerance_relative: float | None = None,
    similarity_threshold: float | None = None,
) -> tuple[bool, str]:
    """Validate an artifact against its reference hash based on the specified mode.

    Performs an initial hash check. If hashes match, returns success immediately.
    Otherwise, proceeds with mode-specific validation.

    Args:
        artifact_path: Path to the artifact.
        reference_hash: Reference hash to compare against.
        validation_type: Validation type ("strict", "standard", "tolerant").
        tolerance_absolute: Absolute tolerance for numeric comparisons (standard mode).
        tolerance_relative: Relative tolerance for numeric comparisons (standard mode).
        similarity_threshold: Similarity threshold for content comparison (tolerant mode).

    Returns:
        (bool, str): (Success flag, Validation message).
    """
    artifact_path = Path(artifact_path)

    if not artifact_path.exists():
        return False, f"Artifact does not exist: {artifact_path}"

    # --- Initial Hash Check ---
    try:
        current_hash = calculate_artifact_hash(artifact_path)
        if current_hash == reference_hash:
            # If hashes match, the artifact is identical, regardless of mode
            return True, "Exact hash match"
        # If hash doesn't match, proceed to mode-specific validation ONLY IF mode is not strict
        # Strict mode requires exact hash match, which already failed here.
        elif validation_type == "strict":
            return False, "Strict validation failed: hash mismatch"

    except Exception as e:
        # If hash calculation fails, we cannot perform the initial check.
        # Proceed to mode-specific validation, but log a warning?
        # For now, let the mode-specific validation handle potential errors.
        # However, strict mode would fail here if hash couldn't be calculated.
        if validation_type == "strict":
            return False, f"Strict validation failed: could not calculate hash - {str(e)}"
        # For standard/tolerant, let them try their logic.
        pass

    # --- Mode-Specific Validation (only if initial hash check didn't pass/fail strictly) ---
    try:
        # Note: Strict case is handled above after the hash mismatch.
        if validation_type == "standard":
            # Pass relevant tolerances to the standard validator
            return validate_standard(
                artifact_path,
                reference_hash,  # Pass reference hash even if it didn't match initially
                tolerance_absolute=tolerance_absolute,
                tolerance_relative=tolerance_relative,
            )

        elif validation_type == "tolerant":
            # Pass similarity threshold to the tolerant validator
            return validate_tolerant(artifact_path, reference_hash, similarity_threshold)

        else:
            return False, f"Unknown validation type: {validation_type}"

    except Exception as e:
        # Catch potential errors during the specific validation mode logic
        return False, f"Validation error during '{validation_type}' mode specific check: {str(e)}"


# Expose the main validation function for easier import
__all__ = ["validate_artifact"]
