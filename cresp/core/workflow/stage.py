"""Stage function definition and handling for computational workflows."""

import functools
from collections.abc import Callable
from typing import Any, TypeVar, cast

# Type variables for function annotations
T = TypeVar("T")
R = TypeVar("R")

# Define sentinel object for skipped status
SKIPPED = object()


class StageFunction:
    """Represents a registered stage function in a workflow."""

    def __init__(
        self,
        func: Callable[..., R],
        stage_id: str,
        description: str | None = None,
        outputs: list[str | dict[str, Any]] | None = None,
        dependencies: list[str] | None = None,
        parameters: dict[str, Any] | None = None,
        reproduction_mode: str = "strict",
        tolerance_absolute: float | None = None,
        tolerance_relative: float | None = None,
        similarity_threshold: float | None = None,
        skip_if_unchanged: bool = False,
    ):
        """Initialize a stage function.

        Args:
            func: The function to execute for this stage.
            stage_id: Unique identifier for this stage.
            description: Optional description of what this stage does.
            outputs: List of output artifacts or paths. Each output can be:
                   - A string path
                   - A dict with at least a "path" key, and optionally:
                     - "shared": Boolean flag indicating if the output should be in shared directory (default: False)
                     - "description": Description of the output
                     - "reproduction": Settings for reproduction validation
            dependencies: List of stage IDs that this stage depends on.
            parameters: Additional parameters for stage execution.
            reproduction_mode: Reproduction mode (strict, standard, tolerant, ignore).
                               Defaults to "strict". If "ignore", validation is skipped for outputs of this stage unless overridden per-output.
            tolerance_absolute: Absolute tolerance for numeric comparisons.
            tolerance_relative: Relative tolerance for numeric comparisons.
            similarity_threshold: Similarity threshold for content comparison.
            skip_if_unchanged: Whether to skip this stage if outputs are unchanged.
        """
        self.func = func
        self.stage_id = stage_id
        self.description = description or func.__doc__ or f"Stage function {stage_id}"
        self.outputs = outputs or []
        self.dependencies = dependencies or []
        self.parameters = parameters or {}
        self.code_handler = f"{func.__module__}.{func.__qualname__}"
        # Validate reproduction_mode
        allowed_modes = {"strict", "standard", "tolerant", "ignore"}
        if reproduction_mode not in allowed_modes:
            raise ValueError(f"Invalid reproduction_mode '{reproduction_mode}' for stage '{stage_id}'. Allowed modes are: {allowed_modes}")
        self.reproduction_mode = reproduction_mode
        self.tolerance_absolute = tolerance_absolute
        self.tolerance_relative = tolerance_relative
        self.similarity_threshold = similarity_threshold
        self.skip_if_unchanged = skip_if_unchanged

        # Preserve function metadata
        functools.update_wrapper(self, func)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Call the wrapped function."""
        return self.func(*args, **kwargs)

    def to_stage_config(self) -> dict[str, Any]:
        """Convert to stage configuration dictionary suitable for CrespConfig."""
        artifacts = []
        for output in self.outputs:
            # Prepare the base artifact dictionary
            artifact_base: dict[str, Any] = {}
            if isinstance(output, str):
                artifact_base["path"] = output
            elif isinstance(output, dict) and "path" in output:
                artifact_base = output.copy()  # Start with the user-provided dict
            else:
                # Skip invalid output definitions
                # console.print(f"[yellow]Warning: Skipping invalid output definition in stage '{self.stage_id}': {output}[/yellow]")
                continue

            # Ensure reproduction settings are present, using stage defaults if needed
            repro_config: dict[str, Any] = artifact_base.get("reproduction", {})
            # --- Use stage's reproduction_mode as default ---
            repro_config.setdefault("mode", self.reproduction_mode)
            if self.tolerance_absolute is not None:
                repro_config.setdefault("tolerance_absolute", self.tolerance_absolute)
            if self.tolerance_relative is not None:
                repro_config.setdefault("tolerance_relative", self.tolerance_relative)
            if self.similarity_threshold is not None:
                repro_config.setdefault("similarity_threshold", self.similarity_threshold)

            # Only add reproduction section if it has non-default values or was explicitly defined
            # or if the mode is 'ignore' (to explicitly mark it)
            if repro_config or isinstance(output, dict) and "reproduction" in output or repro_config.get("mode") == "ignore":
                artifact_base["reproduction"] = repro_config

            # Ensure hash_method is present if hash exists
            if "hash" in artifact_base and "hash_method" not in artifact_base:
                artifact_base["hash_method"] = "sha256"  # Default hash method

            # Apply heuristic for shared detection if not explicitly set
            if "shared" not in artifact_base and isinstance(output, str):
                path_str = output
                if "/" not in path_str and "\\" not in path_str:
                    artifact_base["shared"] = True

            artifacts.append(artifact_base)

        return {
            "id": self.stage_id,
            "description": self.description,
            "dependencies": self.dependencies,
            "outputs": artifacts,
            "code_handler": self.code_handler,
            "parameters": self.parameters,
        }
