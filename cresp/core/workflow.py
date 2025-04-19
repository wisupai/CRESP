# cresp/core/workflow.py

"""Manages the definition, execution, and reproduction of computational workflows."""

import functools
import os
import re
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar, cast

# Rich imports for visualization
try:
    import rich.box
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import (
        BarColumn,
        Progress,
        SpinnerColumn,
        TextColumn,
        TimeElapsedColumn,
    )
    from rich.table import Table

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

# Import related cresp modules
from .config import CrespConfig
from .exceptions import ReproductionError
from .seed import get_reproducible_dataloader_kwargs, set_seed
from .utils import calculate_artifact_hash, create_workflow_config
from .validation import validate_artifact

# Define sentinel object for skipped status
SKIPPED = object()

# Create console instance for rich output
# We create it here so it's available for the Workflow class
if RICH_AVAILABLE:
    console = Console()
else:
    # Provide a dummy console if rich is not available
    class DummyConsole:
        def print(self, *args, **kwargs):
            print(*args)  # Fallback to standard print

    console = cast(Console, DummyConsole())  # Add type cast here

# Type variables for function annotations
T = TypeVar("T")
R = TypeVar("R")

# At the top of the file, add comment to disable specific mypy errors
# type: ignore[assignment]


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
            reproduction_mode: Reproduction mode (strict, standard, tolerant).
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
                console.print(f"[yellow]Warning: Skipping invalid output definition in stage '{self.stage_id}': {output}[/yellow]")
                continue

            # Ensure reproduction settings are present, using stage defaults if needed
            repro_config: dict[str, Any] = artifact_base.get("reproduction", {})
            repro_config.setdefault("mode", self.reproduction_mode)
            if self.tolerance_absolute is not None:
                repro_config.setdefault("tolerance_absolute", self.tolerance_absolute)
            if self.tolerance_relative is not None:
                repro_config.setdefault("tolerance_relative", self.tolerance_relative)
            if self.similarity_threshold is not None:
                repro_config.setdefault("similarity_threshold", self.similarity_threshold)

            # Only add reproduction section if it has non-default values or was explicitly defined
            if repro_config or isinstance(output, dict) and "reproduction" in output:
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


# Global variable to hold the rich progress instance during run
# This allows nested functions like _run_stage to interact with it
progress: Progress | None = None


class Workflow:
    """Workflow class for managing experiment stages and execution."""

    def __init__(
        self,
        title: str,
        authors: list[dict[str, str]],
        description: str | None = None,
        config_path: str = "cresp.yaml",
        seed: int | None = None,
        use_rich: bool = True,
        mode: str = "experiment",
        skip_unchanged: bool = False,
        reproduction_failure_mode: str = "stop",
        set_seed_at_init: bool = True,
        verbose_seed_setting: bool = True,
        experiment_output_dir: str = "experiment",
        reproduction_output_dir: str = "reproduction",
        shared_data_dir: str = "shared",
    ):
        """Initialize a new workflow.

        Args:
            title: Workflow title.
            authors: List of author information dictionaries.
            description: Optional workflow description.
            config_path: Path to save/load the configuration.
            seed: Optional random seed for reproducibility.
            use_rich: Enable rich visualizations (if available).
            mode: Workflow mode ("experiment" or "reproduction").
            skip_unchanged: If True, skip stage execution if outputs match stored hashes.
            reproduction_failure_mode: Behavior on reproduction failure ("stop" or "continue").
            set_seed_at_init: Whether to set random seeds at initialization.
            verbose_seed_setting: Whether to print seed setting information.
            experiment_output_dir: Directory for outputs in experiment mode.
            reproduction_output_dir: Directory for outputs in reproduction mode.
            shared_data_dir: Directory for shared data accessible in all modes.
        """
        self.title = title
        self.use_rich = use_rich and RICH_AVAILABLE
        self.mode = mode
        self.skip_unchanged = skip_unchanged
        self._stages: dict[str, StageFunction] = {}
        self._executed_stages: set[str] = set()
        self._validation_results: list[dict[str, Any]] = []
        self._stage_validation_status: dict[str, bool | object | None] = {}
        config_file_path = Path(config_path)
        self.reproduction_failure_mode = reproduction_failure_mode
        self._seed = seed
        self._verbose_seed_setting = verbose_seed_setting
        self._seed_initialized = False
        # Cache for storing results of _run_stage within a single workflow.run() call
        self._run_cache: dict[str, tuple[Any, list[tuple[str, str]], bool | object | None]] = {}

        # --- Determine active output directory based on mode ---
        self.experiment_output_dir = Path(experiment_output_dir)
        self.reproduction_output_dir = Path(reproduction_output_dir)
        if self.mode == "experiment":
            self.active_output_dir = self.experiment_output_dir
        elif self.mode == "reproduction":
            self.active_output_dir = self.reproduction_output_dir
        else:
            # Default or fallback if mode is somehow different
            self.active_output_dir = Path(".")  # Or raise an error?

        self.shared_data_dir = Path(shared_data_dir)

        try:
            self.config = CrespConfig.load(config_file_path)
            if self.use_rich:
                console.print(f"[dim]Loaded existing configuration from [bold]{self.config.path}[/bold][/dim]")
        except FileNotFoundError as e:
            if mode == "reproduction":
                raise FileNotFoundError(f"Configuration file '{config_file_path}' not found, required for reproduction mode.") from e
            # Use the helper function from utils
            self.config = create_workflow_config(title, authors, description, str(config_file_path))
            if self.use_rich:
                console.print(f"[dim]Creating new configuration at [bold]{self.config.path}[/bold][/dim]")
        except Exception as e:
            raise ValueError(f"Error loading or creating configuration '{config_file_path}': {e}") from e

        current_config_seed = self.config.data.get("reproduction", {}).get("random_seed")
        if seed is not None:
            if current_config_seed is not None and seed != current_config_seed and self.use_rich:
                console.print(f"[yellow]Warning: Overriding random seed from config ({current_config_seed}) with provided seed ({seed})[/yellow]")
            self.config.set_seed(seed)  # This method exists in CrespConfig
            if self.use_rich and self._verbose_seed_setting:
                console.print(f"[dim]Using random seed: {seed}[/dim]")
            self._seed = seed
        elif current_config_seed is not None:
            if self.use_rich and self._verbose_seed_setting:
                console.print(f"[dim]Using random seed from config: {current_config_seed}[/dim]")
            self._seed = current_config_seed

        if set_seed_at_init and self._seed is not None:
            self.set_random_seeds(verbose=self._verbose_seed_setting)
            self._seed_initialized = True

    def set_random_seeds(self, verbose: bool = False) -> None:
        """Set random seeds for all detected libraries."""
        if self._seed is None:
            if self.use_rich and verbose:
                console.print("[yellow]Warning: No random seed set. Skipping seed initialization.[/yellow]")
            return

        libraries = set_seed(self._seed, verbose=verbose)

        if self.use_rich and verbose:
            console.print(f"[dim]Random seeds set to {self._seed} for: {', '.join(libraries)}[/dim]")

    @property
    def seed(self) -> int | None:
        """Get the current random seed."""
        return self._seed

    def get_output_path(self, relative_path: str | Path) -> Path:
        """Construct the full output path based on the current mode.

        Args:
            relative_path: The relative path declared in the stage outputs.

        Returns:
            The full path including the mode-specific output directory.
        """
        full_path = self.active_output_dir / Path(relative_path)
        # Ensure the parent directory exists
        full_path.parent.mkdir(parents=True, exist_ok=True)
        return full_path

    def get_shared_data_path(self, relative_path: str | Path) -> Path:
        """Construct the full path for shared data.

        Args:
            relative_path: The relative path within the shared data directory.

        Returns:
            The full path including the shared data directory.
        """
        full_path = self.shared_data_dir / Path(relative_path)
        # Ensure the parent directory exists
        full_path.parent.mkdir(parents=True, exist_ok=True)
        return full_path

    def get_dataloader_kwargs(self) -> dict[str, Any]:
        """Get kwargs for PyTorch DataLoader to ensure reproducibility."""
        if self._seed is None:
            return {}

        return get_reproducible_dataloader_kwargs(self._seed)

    def stage(
        self,
        id: str | None = None,
        description: str | None = None,
        outputs: list[str | dict[str, Any]] | None = None,
        dependencies: list[str] | None = None,
        parameters: dict[str, Any] | None = None,
        reproduction_mode: str = "strict",
        tolerance_absolute: float | None = None,
        tolerance_relative: float | None = None,
        similarity_threshold: float | None = None,
        skip_if_unchanged: bool | None = None,
    ) -> Callable[[Callable[..., R]], StageFunction]:
        """Decorator for registering a stage function."""

        def decorator(func: Callable[..., R]) -> StageFunction:
            stage_id = id or func.__name__
            final_skip_setting = self.skip_unchanged if skip_if_unchanged is None else skip_if_unchanged

            stage_func = StageFunction(
                func=func,
                stage_id=stage_id,
                description=description,
                outputs=outputs,
                dependencies=dependencies,
                parameters=parameters,
                reproduction_mode=reproduction_mode,
                tolerance_absolute=tolerance_absolute,
                tolerance_relative=tolerance_relative,
                similarity_threshold=similarity_threshold,
                skip_if_unchanged=final_skip_setting,
            )

            self._register_stage(stage_func)
            return stage_func

        return decorator

    def _register_stage(self, stage_func: StageFunction) -> None:
        """Register a stage function and add/update it in the config."""
        if stage_func.stage_id in self._stages:
            raise ValueError(f"Stage ID '{stage_func.stage_id}' already registered for this workflow instance.")

        self._stages[stage_func.stage_id] = stage_func

        stage_config_data = stage_func.to_stage_config()
        existing_stage_data = self.config.get_stage(stage_func.stage_id)

        # Use batch update for potential modifications
        with self.config.batch_update():
            if existing_stage_data is None:
                # Add new stage if it doesn't exist in the config file
                try:
                    self.config.add_stage(stage_config_data, defer_save=True)  # defer_save is implicit in batch_update
                    # Log addition if needed (rich only)
                    # if self.use_rich:
                    #     console.print(f"[dim]Added new stage '{stage_func.stage_id}' to configuration.[/dim]")
                except ValueError as e:
                    if self.use_rich:
                        console.print(f"[yellow]Warning: Could not add stage '{stage_func.stage_id}' to config object: {e}[/yellow]")
            else:
                # Stage exists, potentially update non-hashed fields if they differ?
                # For now, we assume the code definition is the source of truth for
                # description, dependencies, code_handler, parameters. Outputs are more complex.
                # Let's update these simple fields. Hash/validation are handled during run.
                needs_update = False
                for key in ["description", "dependencies", "code_handler", "parameters"]:
                    if existing_stage_data.get(key) != stage_config_data.get(key):
                        existing_stage_data[key] = stage_config_data.get(key)
                        needs_update = True
                # Handle outputs: only update paths/descriptions, not hashes/validation from code
                # This is tricky. Let's just ensure all declared output paths exist in config.
                config_output_paths = {o.get("path") for o in existing_stage_data.get("outputs", []) if o.get("path")}
                code_outputs = stage_config_data.get("outputs", [])
                for code_out in code_outputs:
                    if code_out["path"] not in config_output_paths:
                        # Add minimal output entry if missing
                        existing_stage_data.setdefault("outputs", []).append(
                            {
                                "path": code_out["path"],
                                "description": code_out.get("description"),
                                # Hash/validation added during run
                            }
                        )
                        needs_update = True

                if needs_update:
                    # Mark config as modified if changes were made
                    self.config._modified = True
                    # Optional: Log update
                    # if self.use_rich:
                    #     console.print(f"[dim]Updated configuration for stage '{stage_func.stage_id}' based on code definition.[/dim]")
                    pass

    def _resolve_declared_output(self, output_decl: str | dict[str, Any]) -> tuple[Path, bool, str] | None:
        """Resolve a declared output into its full path, scope, and original string path.

        Args:
            output_decl: An item from the stage's 'outputs' list (str or dict).

        Returns:
            A tuple (resolved_path, is_shared, declared_path_str) or None if invalid.
        """
        path_str: str | None = None
        is_shared: bool = False  # Default scope is mode-specific

        if isinstance(output_decl, str):
            path_str = output_decl
            # String declarations are always considered mode-specific unless overridden later?
            # Let's stick to the rule: only dicts with shared=True are shared.
        elif isinstance(output_decl, dict):
            path_str = output_decl.get("path")
            # Explicit 'shared' flag takes precedence
            is_shared = bool(output_decl.get("shared", False))

        if not path_str:
            # console.print(f"[yellow]Warning: Invalid output declaration structure: {output_decl}[/yellow]")
            return None

        try:
            if is_shared:
                resolved_path = self.get_shared_data_path(path_str)
            else:
                resolved_path = self.get_output_path(path_str)
            return resolved_path, is_shared, path_str
        except Exception:
            # console.print(f"[yellow]Warning: Could not resolve path '{path_str}' (shared={is_shared}): {path_resolve_e}[/yellow]")
            return None

    def _check_outputs_unchanged(self, stage_id: str) -> bool:
        """Check if stage outputs exist and match stored hashes."""
        stage_config = self.config.get_stage(stage_id)
        registered_stage = self._stages.get(stage_id)

        if not registered_stage or not stage_config or "outputs" not in stage_config:
            # Cannot determine if unchanged if config or stage definition is missing
            return False

        declared_outputs = registered_stage.outputs  # From @workflow.stage(...)
        if not declared_outputs:
            # If no outputs are declared in the stage definition,
            # we cannot skip based on "unchanged outputs". Always run.
            return False  # Changed from True to False

        config_outputs = stage_config.get("outputs", [])
        if not config_outputs:
            return False  # Outputs declared in code, but none found in config (e.g., first run)

        # Create a lookup for config outputs by path
        expected_files_config: dict[str, dict[str, Any]] = {
            cfg_out["path"]: cfg_out
            for cfg_out in config_outputs
            if "path" in cfg_out and "hash" in cfg_out  # Only consider items with path and hash
        }

        if not expected_files_config:
            # We have outputs in config, but none have hashes yet
            return False

        all_match = True
        files_checked_count = 0

        # Iterate through outputs declared in the @workflow.stage decorator
        for output_decl in declared_outputs:
            # --- Determine reproduction settings from declaration and stage defaults ---
            default_repro_mode = registered_stage.reproduction_mode
            default_tol_abs = registered_stage.tolerance_absolute
            default_tol_rel = registered_stage.tolerance_relative
            default_sim_thresh = registered_stage.similarity_threshold

            output_specific_repro_config = {}
            if isinstance(output_decl, str):
                path_str = output_decl
                is_shared: bool = False  # Default to non-shared for strings
            elif isinstance(output_decl, dict) and "path" in output_decl:
                path_str = output_decl["path"]
                output_specific_repro_config = output_decl.get("reproduction", {})
                # Check shared flag EXPLICITLY
                is_shared = bool(output_decl.get("shared", False))
            else:
                continue  # Skip invalid declaration

            resolved_output = self._resolve_declared_output(output_decl)
            if resolved_output is None:
                # If we can't resolve the path, consider it changed
                # console.print(f"[yellow]Warning: Could not resolve declared output '{output_decl}' for stage '{stage_id}' during skip check.[/yellow]")
                return False

            resolved_path, is_shared, path_str = resolved_output

            # --- Find corresponding files in config and validate them ---
            try:
                if is_shared:
                    resolved_path = self.get_shared_data_path(path_str)
                else:
                    resolved_path = self.get_output_path(path_str)
            except Exception:
                # If we can't resolve the path, consider it changed
                return False

            files_to_validate_for_this_decl: dict[str, dict[str, Any]] = {}  # Key is config key (relative path str)

            # Case 1: Declaration is an exact file path present in config
            # --- REVISED LOGIC for config key matching ---
            # Config keys are now relative to base dirs. We need to check if the resolved path
            # corresponds to any config entry.

            # --- Determine the base directory for comparison ---
            base_dir = self.shared_data_dir if is_shared else self.active_output_dir

            # Calculate the relative path key this declaration would correspond to
            try:
                relative_key = resolved_path.relative_to(base_dir)
                relative_key_str = str(relative_key)
            except ValueError:
                # Path is not relative to the expected base, something is wrong
                # console.print(f"[yellow]Warning: Resolved path '{resolved_path}' not relative to base '{base_dir}' for stage '{stage_id}'. Cannot check if unchanged.[/yellow]")
                return False  # Cannot determine, assume changed

            # Case 1: Declared path is a file
            if resolved_path.exists() and resolved_path.is_file():
                if relative_key_str in expected_files_config:
                    config_entry = expected_files_config[relative_key_str]
                    # Check if scope matches (config should store 'shared' flag)
                    config_is_shared = bool(config_entry.get("shared", False))
                    if config_is_shared == is_shared:
                        files_to_validate_for_this_decl[relative_key_str] = config_entry
                    else:
                        # Mismatch in scope between declaration and config, assume changed
                        # console.print(f"[yellow]Warning: Scope mismatch for '{relative_key_str}' in stage '{stage_id}'. Declared shared={is_shared}, config shared={config_is_shared}.[/yellow]")
                        return False
                else:
                    # Declared file exists but not found in config (with hash)
                    return False

            # Case 2: Declared path is a directory
            elif resolved_path.exists() and resolved_path.is_dir():
                found_match_in_dir = False
                # Iterate through config entries and see if they fall under this directory
                for config_key, config_entry in expected_files_config.items():
                    config_is_shared = bool(config_entry.get("shared", False))
                    config_base_dir = self.shared_data_dir if config_is_shared else self.active_output_dir
                    try:
                        # Construct full path from config key
                        config_full_path = (config_base_dir / config_key).resolve()

                        # Check if config file path is within the declared directory path
                        if config_full_path.is_relative_to(resolved_path.resolve()):
                            # Check if file exists on disk
                            if not config_full_path.exists():
                                return False  # File listed in config (under dir) is missing

                            # Check scope consistency (config file must have same scope as declared dir)
                            if config_is_shared != is_shared:
                                # console.print(f"[yellow]Warning: Scope mismatch for file '{config_key}' within directory '{path_str}' for stage '{stage_id}'.[/yellow]")
                                return False  # Scope mismatch

                            files_to_validate_for_this_decl[config_key] = config_entry
                            found_match_in_dir = True
                    except ValueError:
                        # Config path not relative to declared directory path
                        continue
                    except Exception:
                        # Error resolving or checking path
                        # console.print(f"[yellow]Warning: Error checking config entry '{config_key}' against directory '{path_str}': {e}[/yellow]")
                        return False  # Error, assume changed

                if not found_match_in_dir:
                    # Directory declared, but no known/hashed files within found in config
                    return False
            else:
                # Declared path does not exist (as file or dir)
                return False

            # --- Validate the collected files for this declaration ---
            if not files_to_validate_for_this_decl:
                # This case might mean the declaration exists but has no hashed files associated yet.
                return False

            for file_path_str, file_cfg in files_to_validate_for_this_decl.items():
                files_checked_count += 1

                # Determine final validation parameters (File > Declaration > Stage)
                file_specific_repro_config = file_cfg.get("reproduction", {})

                final_repro_mode = file_specific_repro_config.get("mode", output_specific_repro_config.get("mode", default_repro_mode))
                final_tol_abs = file_specific_repro_config.get(
                    "tolerance_absolute",
                    output_specific_repro_config.get("tolerance_absolute", default_tol_abs),
                )
                final_tol_rel = file_specific_repro_config.get(
                    "tolerance_relative",
                    output_specific_repro_config.get("tolerance_relative", default_tol_rel),
                )
                final_sim_thresh = file_specific_repro_config.get(
                    "similarity_threshold",
                    output_specific_repro_config.get("similarity_threshold", default_sim_thresh),
                )

                # Resolve file path based on shared flag
                try:
                    file_is_shared = bool(file_cfg.get("shared", is_shared))
                    if file_is_shared:
                        actual_file_path = self.get_shared_data_path(file_path_str)
                    else:
                        actual_file_path = self.get_output_path(file_path_str)
                except Exception:
                    all_match = False
                    break

                # Perform validation using the hash from config
                success, _ = validate_artifact(
                    actual_file_path,
                    file_cfg["hash"],  # Hash must exist from earlier check
                    validation_type=final_repro_mode,
                    tolerance_absolute=final_tol_abs,
                    tolerance_relative=final_tol_rel,
                    similarity_threshold=final_sim_thresh,
                )

                if not success:
                    all_match = False
                    break  # Stop checking this declaration if one file fails

            if not all_match:
                break  # Stop checking other declarations if one failed

        # Return True only if all declarations were processed,
        # at least one file was checked, and all matched.
        return all_match and files_checked_count > 0

    def _run_stage(self, stage_id: str) -> tuple[Any, list[tuple[str, str]], bool | object | None]:
        """Run a specific stage, handling dependencies, execution, hashing/validation.

        Returns:
            Tuple containing:
                - stage execution result (or None if skipped)
                - list of (path, hash) tuples (empty if skipped or not experiment mode)
                - validation status (bool, None, or SKIPPED sentinel)
        """
        global progress  # Access the global progress bar

        # 1. Check cache for this run first
        if stage_id in self._run_cache:
            return self._run_cache[stage_id]

        stage_func = self._stages.get(stage_id)
        if not stage_func:
            raise ValueError(f"Stage function for ID '{stage_id}' not found in memory.")

        # 2. --- Dependency Execution & Status Check ---
        dependencies_were_run_or_failed = False
        for dep_id in stage_func.dependencies:
            # Recursively ensure dependency is processed. Status is retrieved from the call.
            try:
                _, _, dep_status = self._run_stage(dep_id)
                if dep_status is not SKIPPED:
                    # console.print(f"[dim]Dependency {dep_id} of {stage_id} was run/failed (Status: {dep_status}).[/dim]")
                    dependencies_were_run_or_failed = True
                # Handle potential upstream failure propagation if needed (though exceptions might handle it)
                if dep_status is False and self.reproduction_failure_mode == "stop":
                    # This dependency failed reproduction. Although _run_stage would raise,
                    # let's ensure this state is known.
                    dependencies_were_run_or_failed = True
            except Exception as dep_e:
                # If a dependency raised an error during its execution
                console.print(f"[red]Error running dependency '{dep_id}' for stage '{stage_id}': {dep_e}[/red]")
                # Re-raise to halt the current stage processing
                raise

        # 3. --- Skip Check for current stage ---
        should_skip = False
        if stage_func.skip_if_unchanged:
            if dependencies_were_run_or_failed:
                # console.print(f"[dim]Cannot skip {stage_id}: Dependencies were run/failed.[/dim]")
                pass
            else:
                # console.print(f"[dim]Checking outputs unchanged for {stage_id}...[/dim]")
                outputs_unchanged = self._check_outputs_unchanged(stage_id)
                if outputs_unchanged:
                    should_skip = True
                else:
                    # console.print(f"[dim]Cannot skip {stage_id}: Outputs changed.[/dim]")
                    pass
        else:
            # console.print(f"[dim]Cannot skip {stage_id}: skip_if_unchanged is False.[/dim]")
            pass

        if should_skip:
            self._executed_stages.add(stage_id)  # Keep track of processed stages
            status: bool | object | None = SKIPPED
            result_tuple: tuple[Any, list[tuple[str, str]], bool | object | None] = (
                None,
                [],
                status,
            )
            self._stage_validation_status[stage_id] = status  # Store final status
            self._run_cache[stage_id] = result_tuple  # Cache the result
            return result_tuple

        # 4. --- If not skipped, execute the stage ---
        calculated_hashes: list[tuple[str, str]] = []
        result = None
        stage_validation_passed: bool | object | None = None  # Can be True, False, None

        # --- Set random seeds ---
        if self._seed is not None:
            verbose_seed = self._verbose_seed_setting and not self._seed_initialized
            self.set_random_seeds(verbose=verbose_seed)
            self._seed_initialized = True  # Mark as initialized after first potential setting

        # --- Stage Execution ---
        if self.use_rich:
            try:
                result = stage_func()  # Call the __call__ method of StageFunction
            except Exception as e:
                console.print(f"[red]  ✗ Stage [bold]{stage_id}[/bold] execution failed: {str(e)}[/red]")
                raise  # Re-raise the exception to be caught by the main run loop
        else:
            # Non-rich execution
            try:
                result = stage_func()
            except Exception as e:
                print(f"  Error: Stage {stage_id} execution failed: {str(e)}")
                raise  # Re-raise

        self._executed_stages.add(stage_id)  # Mark as executed (run)

        # --- Output Handling (Hashing or Validation) ---
        if stage_func.outputs:
            if self.mode == "experiment":
                calculated_hashes = self._update_output_hashes(stage_id, stage_func.outputs)
                stage_validation_passed = None  # No validation in experiment mode
            elif self.mode == "reproduction":
                stage_validation_passed = self._validate_outputs(stage_id)
                # calculated_hashes remain empty in reproduction mode
            else:  # No outputs or other modes?
                stage_validation_passed = None  # Default to None if not reproduction
        else:  # No outputs declared for the stage
            stage_validation_passed = None  # Nothing to validate or hash

        # 5. --- Store results and return ---
        result_tuple = (result, calculated_hashes, stage_validation_passed)
        self._stage_validation_status[stage_id] = stage_validation_passed  # Store final status
        self._run_cache[stage_id] = result_tuple  # Cache the result
        return result_tuple

    def _update_output_hashes(self, stage_id: str, declared_outputs: list[str | dict[str, Any]]) -> list[tuple[str, str]]:
        """Calculate and store hashes for stage outputs in experiment mode."""
        hashes_calculated = []
        # Use batch update for efficiency when hashing multiple files/directories
        with self.config.batch_update():
            for output_decl in declared_outputs:
                # --- Determine path and hash method from declaration ---
                path_str: str | None = None
                hash_method = "sha256"  # Default hash method
                is_shared = False  # Default scope is mode-specific

                if isinstance(output_decl, str):
                    path_str = output_decl
                    # Apply heuristic: path without separator is likely shared
                    if "/" not in path_str and "\\" not in path_str:
                        is_shared = True
                elif isinstance(output_decl, dict):
                    path_str = output_decl.get("path")
                    # Allow overriding hash method per output declaration
                    hash_method = output_decl.get("hash_method", hash_method)
                    # Check for shared flag with fallback to heuristic
                    if "shared" in output_decl:
                        is_shared = bool(output_decl.get("shared", False))
                    elif path_str and "/" not in path_str and "\\" not in path_str:
                        is_shared = True

                if not path_str:
                    console.print(f"[yellow]Warning: Invalid output declaration in stage '{stage_id}': {output_decl}[/yellow]")
                    continue

                # --- Resolve path based on scope ---
                try:
                    if is_shared:
                        resolved_path = self.get_shared_data_path(path_str)
                    else:
                        resolved_path = self.get_output_path(path_str)
                except Exception as path_resolve_e:
                    console.print(f"[yellow]Warning: Could not resolve path '{path_str}' for stage '{stage_id}': {path_resolve_e}[/yellow]")
                    continue

                # --- Check existence using the RESOLVED path ---
                try:
                    if not resolved_path.exists():
                        if self.use_rich:
                            msg = f"Warning: Output path does not exist after running stage '{stage_id}': {resolved_path} (declared as '{path_str}')"
                            console.print(f"[yellow]{msg}[/yellow]")
                        continue

                    # --- Hash File or Directory Contents using RESOLVED path ---
                    if resolved_path.is_file():
                        hash_value = calculate_artifact_hash(resolved_path, method=hash_method)
                        # Update config using the RELATIVE path as the key
                        base_dir = self.shared_data_dir if is_shared else self.active_output_dir
                        try:
                            relative_key = str(resolved_path.relative_to(base_dir))
                        except ValueError:
                            relative_key = str(resolved_path)
                        config_entry = {
                            "path": relative_key,
                            "hash": hash_value,
                            "hash_method": hash_method,
                            "shared": is_shared,
                        }
                        if hasattr(self.config, "update_artifact"):
                            self.config.update_artifact(stage_id, relative_key, config_entry)
                        else:
                            self.config.update_hash(stage_id, relative_key, hash_value, hash_method)
                        hashes_calculated.append((relative_key, hash_value))
                    elif resolved_path.is_dir():
                        if self.use_rich:
                            console.print(f"[dim]Hashing contents of directory [bold]{resolved_path}[/bold] (declared as '{path_str}')...[/dim]")
                        files_hashed_count = 0
                        base_dir = self.shared_data_dir if is_shared else self.active_output_dir
                        for file_path in resolved_path.rglob("*"):
                            if file_path.is_file():
                                try:
                                    file_hash_value = calculate_artifact_hash(file_path, method=hash_method)
                                    try:
                                        relative_key_path_str = str(file_path.relative_to(base_dir))
                                    except ValueError:
                                        relative_key_path_str = str(file_path.resolve())
                                    config_entry = {
                                        "path": relative_key_path_str,
                                        "hash": file_hash_value,
                                        "hash_method": hash_method,
                                        "shared": is_shared,
                                    }
                                    if hasattr(self.config, "update_artifact"):
                                        self.config.update_artifact(stage_id, relative_key_path_str, config_entry)
                                    else:
                                        self.config.update_hash(
                                            stage_id,
                                            relative_key_path_str,
                                            file_hash_value,
                                            hash_method,
                                        )
                                    hashes_calculated.append((relative_key_path_str, file_hash_value))
                                    files_hashed_count += 1
                                except Exception as file_e:
                                    if self.use_rich:
                                        console.print(f"[yellow]  Warning: Failed to hash file {file_path}: {str(file_e)}[/yellow]")
                        if self.use_rich and files_hashed_count > 0:
                            console.print(f"[dim]Hashed {files_hashed_count} files in [bold]{resolved_path}[/bold].[/dim]")
                    else:
                        if self.use_rich:
                            console.print(f"[yellow]Warning: Resolved output path is not a file or directory: {resolved_path}[/yellow]")

                except Exception as e:
                    if self.use_rich:
                        msg = f"Warning: Failed to process output {resolved_path} (declared as '{path_str}') for stage '{stage_id}': {str(e)}"
                        console.print(f"[yellow]{msg}[/yellow]")

        return hashes_calculated

    def _validate_outputs(self, stage_id: str) -> bool:
        """Validate stage outputs against stored hashes in reproduction mode."""
        stage_config = self.config.get_stage(stage_id)
        registered_stage = self._stages.get(stage_id)  # Should always exist if we got here

        if not registered_stage:
            return True  # Should not happen, but safer

        declared_outputs = registered_stage.outputs

        if not stage_config or not stage_config.get("outputs"):
            # If outputs declared in code but none in config (maybe first run?), treat as warning/pass?
            # If no outputs declared in code, it's fine.
            if declared_outputs and self.use_rich:
                console.print(f"[yellow]Warning: No reference outputs/hashes found in config for stage '{stage_id}' to validate against.[/yellow]")
            return True  # Consider valid if nothing to compare against

        if not declared_outputs:
            return True  # Nothing declared in code, nothing to validate here.

        # Create a lookup for config outputs by path (must have hash)
        expected_files_config: dict[str, dict[str, Any]] = {
            cfg_out["path"]: cfg_out for cfg_out in stage_config.get("outputs", []) if cfg_out.get("path") and cfg_out.get("hash")
        }

        if not expected_files_config:
            if self.use_rich:
                console.print(f"[yellow]Warning: Outputs found in config for stage '{stage_id}', but none have recorded hashes.[/yellow]")
            # If outputs declared in code, this should probably be a failure? Or at least warning.
            # Let's treat it as pass for now, as we can't validate.
            return True

        stage_validation_passed = True  # Assume success until proven otherwise

        # Iterate through outputs declared in the @workflow.stage decorator
        for output_decl in declared_outputs:
            # --- Determine reproduction settings from declaration and stage defaults ---
            default_repro_mode = registered_stage.reproduction_mode
            default_tol_abs = registered_stage.tolerance_absolute
            default_tol_rel = registered_stage.tolerance_relative
            default_sim_thresh = registered_stage.similarity_threshold

            output_specific_repro_config = {}
            path_str: str | None = None
            is_shared = False  # Default to mode-specific scope

            if isinstance(output_decl, str):
                path_str = output_decl
                # Apply heuristic for backward compatibility
                if "/" not in path_str and "\\" not in path_str:
                    is_shared = True
            elif isinstance(output_decl, dict) and "path" in output_decl:
                path_str = output_decl["path"]
                output_specific_repro_config = output_decl.get("reproduction", {})
                # Check shared flag with fallback to heuristic
                if "shared" in output_decl:
                    is_shared = bool(output_decl.get("shared", False))
                # Apply heuristic as fallback
                elif path_str and "/" not in path_str and "\\" not in path_str:
                    is_shared = True
            else:
                continue  # Skip invalid declaration

            # --- Resolve path for validation ---
            try:
                if path_str is None:
                    # Skip if path is None
                    continue

                if is_shared:
                    resolved_path = self.get_shared_data_path(path_str)
                else:
                    resolved_path = self.get_output_path(path_str)
            except Exception as path_resolve_e:
                console.print(f"[yellow]Warning: Could not resolve path '{path_str}' for validation in stage '{stage_id}': {path_resolve_e}[/yellow]")
                continue

            # --- Find corresponding files in config based on ORIGINAL path_str ---
            # We still use path_str to lookup in expected_files_config which uses declared paths as keys
            files_to_validate_for_this_decl: dict[str, dict[str, Any]] = {}
            found_config_match = False

            # Case 1: Exact file path match in config
            if path_str in expected_files_config:
                # Override is_shared based on what's stored in config if available
                if "shared" in expected_files_config[path_str]:
                    is_shared = bool(expected_files_config[path_str].get("shared", False))
                files_to_validate_for_this_decl[path_str] = expected_files_config[path_str]
                found_config_match = True
                if self.use_rich:
                    console.print(f"[dim]Found exact config match for path [bold]{path_str}[/bold][/dim]")
            # Case 2: Directory path - find config entries within
            elif resolved_path.exists() and resolved_path.is_dir():
                # Check config paths starting with the declared path_str or within the resolved path
                config_paths_in_dir = []
                for expected_path, expected_cfg in expected_files_config.items():
                    # Normalize paths for comparison
                    # Skip None values to avoid errors in normpath
                    if expected_path is None or path_str is None:
                        continue

                    norm_expected = os.path.normpath(expected_path)
                    norm_declared = os.path.normpath(path_str)

                    # Is the config file directly within this directory?
                    is_in_dir = False

                    # Method 1: Check if expected path starts with declared path + separator
                    if norm_expected.startswith(norm_declared + os.sep):
                        is_in_dir = True
                    # Method 2: Physically check if the file is in the resolved directory
                    else:
                        # Try to resolve as shared or output path
                        try:
                            expected_is_shared = bool(expected_cfg.get("shared", False))
                            if expected_is_shared:
                                self.get_shared_data_path(expected_path)
                            else:
                                self.get_output_path(expected_path)
                            # Is this file physically within the declared directory?
                            try:
                                is_in_dir = True
                            except ValueError:
                                # Not within the directory
                                pass
                        except Exception:
                            # Couldn't resolve path
                            pass

                    if is_in_dir:
                        config_paths_in_dir.append(expected_path)
                        # Override is_shared based on what's stored in config if available
                        file_is_shared = is_shared
                        if "shared" in expected_cfg:
                            file_is_shared = bool(expected_cfg.get("shared", False))
                        # Store with is_shared flag explicitly known from config
                        expected_cfg_with_scope = expected_cfg.copy()
                        expected_cfg_with_scope["_is_shared"] = file_is_shared
                        # The key remains the declared path from config
                        files_to_validate_for_this_decl[expected_path] = expected_cfg_with_scope
                        found_config_match = True

                if config_paths_in_dir and self.use_rich:
                    console.print(f"[dim]Found {len(config_paths_in_dir)} files in config for directory [bold]{path_str}[/bold][/dim]")

            if not found_config_match:
                # Declared output has no corresponding hashed entry in config
                if self.use_rich:
                    console.print(
                        f"[yellow]Warning: No reference hash found in config matching output declaration '{path_str}' for stage '{stage_id}'. Cannot validate.[/yellow]"
                    )

                    # Debug - For directories, list all config paths to help diagnose matching issues
                    if resolved_path.exists() and resolved_path.is_dir():
                        console.print(f"[dim]Directory exists at [bold]{resolved_path}[/bold]. Config has the following entries:[/dim]")
                        for i, (config_path, _) in enumerate(expected_files_config.items()):
                            if i < 10:  # Limit to first 10 for readability
                                console.print(f"[dim]  - {config_path}[/dim]")
                            elif i == 10:
                                console.print(f"[dim]  - ... and {len(expected_files_config) - 10} more[/dim]")
                continue

            # --- Validate the collected files for this declaration ---
            if self.use_rich and files_to_validate_for_this_decl:
                # Log validation attempt using the originally declared path
                console.print(
                    f"[dim]Validating output declaration [bold]{path_str}[/bold] ({len(files_to_validate_for_this_decl)} file(s) in config)...[/dim]"
                )

            # Note: The loop below iterates through paths found in the config (expected_path)
            # We need to map these back to resolved paths on the filesystem for validation.
            for expected_path_key, file_cfg in files_to_validate_for_this_decl.items():
                # Double check hash exists (should from earlier filter)
                if "hash" not in file_cfg:
                    continue

                # --- Resolve the *expected* path from config to its actual filesystem location ---
                # Use _is_shared flag set above if available, otherwise fall back to global is_shared
                expected_is_shared = file_cfg.pop("_is_shared", is_shared)
                # Alternatively, check if shared is explicitly set in config
                if "shared" in file_cfg:
                    expected_is_shared = bool(file_cfg.get("shared", False))

                try:
                    if expected_is_shared:
                        actual_file_path_to_check = self.get_shared_data_path(expected_path_key)
                    else:
                        actual_file_path_to_check = self.get_output_path(expected_path_key)
                except Exception:
                    # If we can't even resolve the path from config, treat as validation failure
                    message = f"Could not resolve expected path from config: {expected_path_key}"
                    success = False
                    final_repro_mode = "N/A"
                    actual_file_path_display = expected_path_key  # Display the problematic key
                else:
                    actual_file_path_display = str(actual_file_path_to_check)  # For logging
                    # Check if the *actual* file exists on disk before attempting validation
                    if not actual_file_path_to_check.exists():
                        message = f"Output file not found: {actual_file_path_to_check}"
                        success = False
                        final_repro_mode = "N/A"  # Mode doesn't apply if file is missing
                    else:
                        # Determine final validation parameters (File > Declaration > Stage)
                        file_specific_repro_config = file_cfg.get("reproduction", {})
                        final_repro_mode = file_specific_repro_config.get("mode", output_specific_repro_config.get("mode", default_repro_mode))
                        final_tol_abs = file_specific_repro_config.get(
                            "tolerance_absolute",
                            output_specific_repro_config.get("tolerance_absolute", default_tol_abs),
                        )
                        final_tol_rel = file_specific_repro_config.get(
                            "tolerance_relative",
                            output_specific_repro_config.get("tolerance_relative", default_tol_rel),
                        )
                        final_sim_thresh = file_specific_repro_config.get(
                            "similarity_threshold",
                            output_specific_repro_config.get("similarity_threshold", default_sim_thresh),
                        )

                        # Perform validation using the ACTUAL file path
                        success, message = validate_artifact(
                            actual_file_path_to_check,
                            file_cfg["hash"],
                            validation_type=final_repro_mode,
                            tolerance_absolute=final_tol_abs,
                            tolerance_relative=final_tol_rel,
                            similarity_threshold=final_sim_thresh,
                        )

                # --- Record validation result ---
                self._validation_results.append(
                    {
                        "stage": stage_id,
                        "file": actual_file_path_display,  # Report the actual path checked
                        "status": "Passed" if success else "Failed",
                        "mode": final_repro_mode,
                        "message": message,
                    }
                )

                # --- Update overall stage status and print result ---
                if self.use_rich:
                    if success:
                        # Use the actual path in the success/fail message for clarity
                        console.print(f"[green]  ✓ {actual_file_path_display}: {message}[/green]")
                    else:
                        console.print(f"[red]  ✗ {actual_file_path_display}: {message}[/red]")
                        stage_validation_passed = False  # Mark stage as failed if any file fails

        # Check if any declared outputs were *not* found in the config for validation
        # This might indicate an incomplete config or new outputs not yet hashed.
        # We handled this with a warning above, so stage_validation_passed reflects only actual comparisons.

        return stage_validation_passed  # Return overall status for the stage

    def run(self, stage_id: str | None = None) -> dict[str, Any]:
        """Run workflow stages, either a single one or all in order."""
        global progress  # Use the global progress variable

        results = {}
        self._validation_results = []  # Clear previous validation results
        self._stage_validation_status = {}  # Clear previous stage statuses
        self._executed_stages = set()  # Clear executed stages for this run
        self._run_cache = {}  # Clear run cache for this run
        workflow_failed = False  # Track if any stage failed execution or reproduction

        run_subtitle = f"Mode: {self.mode.capitalize()}"
        if self.mode == "reproduction":
            run_subtitle += f" (Fail on: {self.reproduction_failure_mode})"
        if self._seed is not None:
            run_subtitle += f" | Seed: {self._seed}"

        # Add authors to subtitle if they exist
        authors_str = ", ".join([a.get("name", "Unknown") for a in self.config.data.get("authors", [])])
        if authors_str:
            run_subtitle += f" | Authors: {authors_str}"

        if self.use_rich:
            console.print(
                Panel(
                    f"[bold blue]{self.title}[/bold blue]",
                    title="🚀 CRESP Workflow Run",
                    subtitle=f"[dim]{run_subtitle}[/dim]",
                    expand=True,
                    border_style="bold green",  # Changed border style
                    padding=(1, 2),  # Add padding
                )
            )
            console.print()  # Add spacing

            # Determine execution plan
            execution_plan: list[str]
            if stage_id:
                # Need to include dependencies if running a single stage
                execution_plan = self._resolve_execution_order(target_stage=stage_id)
                console.print(f"[bold yellow]🎯 Running Target Stage [bold cyan]{stage_id}[/bold cyan] and Dependencies:[/bold yellow]")
            else:
                execution_plan = self._resolve_execution_order()
                console.print("[bold yellow]📋 Workflow Execution Plan:[/bold yellow]")

            # Display execution plan in a table
            plan_table = Table(
                title="Execution Order",
                box=rich.box.ROUNDED,  # Use rounded box
                show_header=True,
                header_style="bold magenta",
                padding=(0, 1),
                show_lines=True,  # Show lines between rows
                expand=True,  # Set expand to True
            )
            plan_table.add_column("Order", style="dim", justify="right", width=5)
            plan_table.add_column("Stage ID", style="cyan", no_wrap=True, min_width=15)
            plan_table.add_column("Description", style="blue")
            for i, stage_to_run in enumerate(execution_plan):
                desc = self._stages[stage_to_run].description or "[dim]No description[/dim]"
                plan_table.add_row(f"{i + 1}.", stage_to_run, desc)
            console.print(plan_table)
            console.print()  # Spacer

            # --- Run with Progress Bar ---
            try:
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                    TimeElapsedColumn(),
                    console=console,
                    transient=False,  # Keep progress visible after completion
                    refresh_per_second=4,  # Adjust refresh rate
                ) as prog:
                    progress = prog  # Assign to global variable

                    overall_task = progress.add_task(
                        f"[green]Workflow Progress ({len(execution_plan)} stages)",
                        total=len(execution_plan),
                    )

                    for idx, curr_stage_id in enumerate(execution_plan):
                        stage_task_description = f"[cyan]Stage {idx + 1}/{len(execution_plan)}: {curr_stage_id}"
                        stage_task = progress.add_task(stage_task_description, total=1, start=False)  # Don't start task immediately

                        # --- Run the stage ---
                        progress.start_task(stage_task)
                        calculated_hashes_for_stage: list[tuple[str, str]] = []
                        stage_validation_status: bool | object | None = None
                        stage_failed_execution = False
                        try:
                            # This call handles execution, dependency runs, hashing/validation
                            stage_result, calculated_hashes_for_stage, stage_validation_status = self._run_stage(curr_stage_id)
                            results[curr_stage_id] = stage_result

                        except ReproductionError:
                            # Specific error raised by _run_stage if repro failed and mode is 'stop'
                            stage_validation_status = False  # Mark as failed
                            workflow_failed = True
                            progress.update(
                                stage_task,
                                completed=1,
                                description=f"[red]✗ {curr_stage_id} failed reproduction (STOPPED)",
                            )
                            console.print(f"  [bold red]Workflow stopped due to reproduction failure in stage '{curr_stage_id}'.[/bold red]")
                            break  # Exit the loop
                        except Exception:
                            # General execution error within the stage function itself
                            stage_failed_execution = True
                            workflow_failed = True
                            progress.update(
                                stage_task,
                                completed=1,
                                description=f"[red]✗ {curr_stage_id} failed execution",
                            )
                            # Error details already printed in _run_stage
                            if self.reproduction_failure_mode == "stop":  # Stop on execution errors too?
                                console.print(f"  [bold red]Workflow stopped due to execution error in stage '{curr_stage_id}'.[/bold red]")
                                break  # Exit the loop
                            # Otherwise, continue if mode is 'continue'

                        # --- Update Progress Bar Based on Outcome ---
                        if stage_failed_execution:
                            # Already updated in except block
                            pass
                        elif stage_validation_status is SKIPPED:  # Check for sentinel explicitly
                            progress.update(
                                stage_task,
                                completed=1,
                                description=f"[yellow]✓ {curr_stage_id} skipped (unchanged)",
                            )
                        elif stage_validation_status is False:
                            progress.update(
                                stage_task,
                                completed=1,
                                description=f"[red]✗ {curr_stage_id} failed reproduction",
                            )
                            workflow_failed = True  # Mark workflow as failed (even if continuing)
                        elif stage_validation_status is True:
                            progress.update(
                                stage_task,
                                completed=1,
                                description=f"[green]✓ {curr_stage_id} passed reproduction",
                            )
                        else:  # Completed in experiment mode or no validation occurred
                            progress.update(
                                stage_task,
                                completed=1,
                                description=f"[green]✓ {curr_stage_id} completed",
                            )

                        # Advance overall progress regardless of individual stage outcome (unless stopped)
                        progress.update(overall_task, advance=1)

                        # --- Post-Stage Actions (if not skipped/failed) ---
                        if not stage_failed_execution and stage_validation_status is not None:
                            # Print hashes (only in experiment mode and if run successfully)
                            if self.mode == "experiment" and calculated_hashes_for_stage:
                                console.print(f"  [dim]Recorded Hashes for {curr_stage_id}:[/dim]")
                                for path_hash, hash_val in calculated_hashes_for_stage:
                                    console.print(f"    [cyan dim]{path_hash}:[/cyan dim] [dim]{hash_val[:8]}...[/dim]")

                            # Check for reproduction failure *after* logging, if continuing
                            if stage_validation_status is False and self.reproduction_failure_mode == "continue":
                                console.print(f"  [yellow]Continuing workflow after reproduction failure in stage '{curr_stage_id}'.[/yellow]")

            finally:
                # Ensure global progress is cleared after the run
                progress = None

            # --- Print Final Summary ---
            console.print()  # Spacer
            summary_table = Table(
                title="🏁 Workflow Execution Summary",
                show_header=True,
                header_style="bold magenta",
                box=rich.box.DOUBLE_EDGE,  # Changed box style
                padding=(0, 1),
                show_lines=True,
                expand=True,  # Set expand to True
            )
            summary_table.add_column("Stage", style="cyan", no_wrap=True)
            summary_table.add_column("Status", style="default", justify="center")
            summary_table.add_column("Outputs", style="blue", overflow="fold")
            summary_table.add_column("Dependencies", style="yellow")
            if self.mode == "reproduction":
                summary_table.add_column("Reproduction", style="default", justify="center")

            # Use the original execution plan for the summary order
            summary_stages = execution_plan

            # Track if any stage was actually skipped
            any_skipped = False

            for stage_id_summary in summary_stages:
                if stage_id_summary not in self._stages:
                    continue  # Should not happen normally

                stage_func = self._stages[stage_id_summary]
                final_status_str = "[grey50]Not run[/grey50]"
                repro_status_str = "[dim]N/A[/dim]"
                status_style = "grey50"  # Default style for not run

                # Determine status based on execution and validation results
                validation_status = self._stage_validation_status.get(stage_id_summary)
                stage_ran_or_skipped = stage_id_summary in self._executed_stages  # Use this set

                if stage_ran_or_skipped:
                    if validation_status is SKIPPED:
                        final_status_str = "Skipped"
                        repro_status_str = "⚪ Skipped"
                        status_style = "yellow"
                        any_skipped = True
                    elif validation_status is False:  # Failed reproduction
                        final_status_str = "Ran"  # The stage itself ran
                        repro_status_str = "❌ Failed"
                        status_style = "red"  # Mark Ran as red if repro failed
                    elif validation_status is True:  # Passed reproduction
                        final_status_str = "Ran"
                        repro_status_str = "✅ Passed"
                        status_style = "green"
                    else:  # Ran in experiment mode or no validation needed (validation_status is None)
                        final_status_str = "Ran"
                        repro_status_str = "[dim]N/A[/dim]"
                        status_style = "green"
                # else: Stage was not executed (e.g., workflow stopped early) - final_status_str remains "Not run"

                # Get output paths from the StageFunction definition for display
                outputs_str_parts = []
                if stage_func.outputs:
                    for out in stage_func.outputs:
                        path_str = None

                        is_shared = False  # Heuristic

                        if isinstance(out, str):
                            path_str = out
                            # Apply heuristic: path without separator is likely shared
                            if "/" not in path_str and "\\\\" not in path_str:
                                is_shared = True
                        elif isinstance(out, dict) and "path" in out:
                            path_str = out["path"]
                            # description = out.get("description")  # Removed unused variable
                            # Check shared flag with fallback to heuristic
                            if out.get("shared"):
                                is_shared = True
                            elif path_str and "/" not in path_str and "\\" not in path_str:
                                is_shared = True

                        if path_str:
                            # Construct the resolved path string for display
                            if is_shared:
                                # Use the relative path defined in the workflow init
                                resolved_display_path = f"{self.shared_data_dir.as_posix()}/{path_str} [shared]"
                            else:
                                # Use the relative path defined in the workflow init
                                resolved_display_path = f"{self.active_output_dir.as_posix()}/{path_str}"

                            display_str = f"{resolved_display_path}"
                            # Description part might be too verbose for the summary table, let's omit it here.
                            # if description:
                            #     display_str += f" ([dim]{description}[/dim])"
                            outputs_str_parts.append(display_str)

                outputs_display = ", ".join(outputs_str_parts) if outputs_str_parts else "[dim]None"
                # --- End construct resolved output paths ---

                # Get and format dependencies
                deps_list = stage_func.dependencies
                deps_str = ", ".join(deps_list) if deps_list else "[dim]None[/dim]"

                row_data = [
                    f"[{status_style}]{stage_id_summary}[/]",  # Apply style to stage ID
                    f"[{status_style}]{final_status_str}[/]",
                    outputs_display,
                    deps_str,
                ]
                if self.mode == "reproduction":
                    repro_style = "default"
                    if repro_status_str == "❌ Failed":
                        repro_style = "red"
                    elif repro_status_str == "✅ Passed":
                        repro_style = "green"
                    elif repro_status_str == "⚪ Skipped":
                        repro_style = "yellow"

                    row_data.append(f"[{repro_style}]{repro_status_str}[/]")

                summary_table.add_row(*row_data)

            console.print(summary_table)
            # Add a footnote if stages were skipped
            if any_skipped:
                console.print("[dim]⚪ Skipped stages indicate outputs were unchanged from the previous run.[/dim]")
            console.print()  # Add spacing before report/save messages

        else:
            # --- Non-Rich Execution Path ---
            print(f"Running Workflow: {self.title} (Mode: {self.mode})")
            execution_plan_simple: list[str]
            if stage_id:
                execution_plan_simple = self._resolve_execution_order(target_stage=stage_id)
                print(f"Execution Plan (Target: {stage_id}): {', '.join(execution_plan_simple)}")
            else:
                execution_plan_simple = self._resolve_execution_order()
                print(f"Execution Plan (All Stages): {', '.join(execution_plan_simple)}")

            for curr_stage_id in execution_plan_simple:
                print(f"---\nRunning Stage: {curr_stage_id}")
                try:
                    stage_result, _, stage_validation_status = self._run_stage(curr_stage_id)
                    results[curr_stage_id] = stage_result

                    status_msg = f"Stage {curr_stage_id}: "
                    if stage_validation_status is None and curr_stage_id in self._executed_stages:
                        status_msg += "Skipped (unchanged)"
                    elif stage_validation_status is False:
                        status_msg += "Completed (Reproduction FAILED)"
                        workflow_failed = True
                        if self.reproduction_failure_mode == "stop":
                            print("  ERROR: Reproduction failed. Stopping workflow.")
                            break
                    elif stage_validation_status is True:
                        status_msg += "Completed (Reproduction PASSED)"
                    else:
                        status_msg += "Completed"
                    print(status_msg)

                except Exception as e:
                    print(f"  ERROR: Stage {curr_stage_id} execution failed: {e}")
                    workflow_failed = True
                    if self.reproduction_failure_mode == "stop":
                        print("  ERROR: Stopping workflow due to execution error.")
                        break
            print("---")

        # --- Final Actions ---
        # Save config potentially modified by hashing or stage registration updates
        # Do this even if the workflow failed mid-way to capture any hashes generated before failure
        # --- Only save config in experiment mode ---
        if self.mode == "experiment":
            self.config.save()
            if self.use_rich:
                console.print(f"[dim]Configuration saved to [bold]{self.config.path}[/bold][/dim]")
        elif self.use_rich:
            console.print(f"[dim]⚙️ Skipping configuration save in [bold]{self.mode}[/bold] mode.[/dim]")

        if workflow_failed:
            if self.reproduction_failure_mode == "continue":
                if self.use_rich:
                    console.print("[bold yellow]Workflow completed with failures (check summary and report).[/bold yellow]")
                else:
                    print("WARNING: Workflow completed with failures.")
            # If failure mode was 'stop', the loop was already broken.
            # We might want to raise an exception here to signal failure programmatically.
            # raise WorkflowExecutionError("Workflow failed during execution.") # Consider adding a specific exception

        return results

    def _resolve_execution_order(self, target_stage: str | None = None) -> list[str]:
        """Resolve execution order using topological sort.
        If target_stage is provided, returns the order needed to run that stage.
        """
        order = []
        visited = set()  # Nodes permanently visited and added to order
        visiting = set()  # Nodes currently in the recursion stack (for cycle detection)

        all_stages = list(self._stages.keys())

        def visit(stage_id: str) -> None:
            if stage_id not in self._stages:
                # This check handles dependencies listed in config but not defined in code
                # OR typos in dependencies list
                raise ValueError(f"Dependency '{stage_id}' not found as a registered stage.")

            if stage_id in visited:
                return  # Already processed
            if stage_id in visiting:
                raise ValueError(f"Circular dependency detected involving stage: {stage_id}")

            visiting.add(stage_id)

            stage = self._stages[stage_id]
            if stage.dependencies:
                for dep_id in stage.dependencies:
                    visit(dep_id)

            visiting.remove(stage_id)
            visited.add(stage_id)
            order.append(stage_id)

        if target_stage:
            if target_stage not in self._stages:
                raise ValueError(f"Target stage '{target_stage}' is not registered.")
            visit(target_stage)  # Visit only the target and its dependencies
        else:
            # Visit all nodes if no target is specified
            for stage_id in all_stages:
                if stage_id not in visited:
                    visit(stage_id)

        return order

    def save_config(self, path: str | None = None) -> bool:
        """Save workflow configuration to file."""
        save_path = Path(path) if path else self.config.path
        if not save_path:
            console.print("[red]Error: Cannot save configuration, no path specified or loaded.[/red]")
            return False

        success = self.config.save(save_path)

        if success and self.use_rich:
            console.print(f"[green]Configuration explicitly saved to [bold]{save_path}[/bold][/green]")
        elif not success and self.use_rich:
            console.print(f"[red]Failed to explicitly save configuration to [bold]{save_path}[/bold][/red]")

        return success

    def visualize(self) -> None:
        """Visualize the workflow structure using Rich."""
        if not self.use_rich:
            print("Rich visualization requires the 'rich' package to be installed.")
            # Basic text visualization could be added here as a fallback
            return

        console.print(Panel(f"[bold]Workflow Structure: {self.title}[/bold]", expand=True, border_style="blue"))
        console.print()  # Add spacing

        try:
            execution_order = self._resolve_execution_order()

            table = Table(
                title="Stage Overview",  # Changed title
                show_header=True,
                header_style="bold magenta",
                box=rich.box.ROUNDED,  # Use rounded box
                show_lines=True,
                padding=(0, 1),
                expand=True,  # Set expand to True
            )
            table.add_column("Order", style="dim", justify="right", width=5)
            table.add_column("Stage ID", style="cyan", no_wrap=True, min_width=15)
            table.add_column("Description", style="blue")
            table.add_column("Dependencies", style="yellow", overflow="fold")  # Allow folding
            table.add_column("Declared Outputs", style="green", overflow="fold")  # Allow folding

            for idx, stage_id in enumerate(execution_order):
                if stage_id not in self._stages:
                    continue  # Skip if somehow not registered

                stage = self._stages[stage_id]
                deps = ", ".join(stage.dependencies) if stage.dependencies else "[dim]None[/dim]"

                # Format outputs nicely
                outputs_str_parts = []
                if stage.outputs:
                    for out in stage.outputs:
                        if isinstance(out, str):
                            outputs_str_parts.append(out)
                        elif isinstance(out, dict) and "path" in out:
                            out_str = out["path"]
                            if out.get("shared"):
                                out_str += " [shared]"
                            if out.get("description"):
                                out_str += f" ([dim]{out['description']}[/dim])"
                            outputs_str_parts.append(out_str)
                outputs_display = "\n".join(outputs_str_parts) if outputs_str_parts else "[dim]None[/dim]"

                table.add_row(str(idx + 1), stage_id, stage.description, deps, outputs_display)

            console.print(table)

        except ValueError as e:  # Catch circular dependencies or missing stages
            console.print(f"[red]Error resolving workflow order for visualization: {str(e)}[/red]")
        except Exception as e:
            console.print(f"[red]An unexpected error occurred during visualization: {str(e)}[/red]")
