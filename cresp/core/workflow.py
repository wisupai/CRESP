# cresp/core/workflow.py

"""Manages the definition, execution, and reproduction of computational workflows."""

import os
import time
import functools
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Callable, TypeVar, Set

# Rich imports for visualization
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
    from rich.table import Table
    from rich.style import Style
    from rich import print as rich_print
    import rich.box

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

# Import related cresp modules
from .config import CrespConfig
from .hashing import calculate_artifact_hash, validate_artifact
from .seed import set_seed, get_reproducible_dataloader_kwargs
from .exceptions import ReproductionError
from .utils import create_workflow_config  # Used in Workflow.__init__

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

    console = DummyConsole()

# Type variables for function annotations
T = TypeVar("T")
R = TypeVar("R")


class StageFunction:
    """Represents a registered stage function in a workflow."""

    def __init__(
        self,
        func: Callable[..., R],
        stage_id: str,
        description: Optional[str] = None,
        outputs: Optional[List[Union[str, Dict[str, Any]]]] = None,
        dependencies: Optional[List[str]] = None,
        parameters: Optional[Dict[str, Any]] = None,
        reproduction_mode: str = "strict",
        tolerance_absolute: Optional[float] = None,
        tolerance_relative: Optional[float] = None,
        similarity_threshold: Optional[float] = None,
        skip_if_unchanged: bool = False,
    ):
        """Initialize a stage function.

        Args:
            func: The function to execute for this stage.
            stage_id: Unique identifier for this stage.
            description: Optional description of what this stage does.
            outputs: List of output artifacts or paths.
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

    def __call__(self, *args, **kwargs) -> R:
        """Call the wrapped function."""
        return self.func(*args, **kwargs)

    def to_stage_config(self) -> Dict[str, Any]:
        """Convert to stage configuration dictionary suitable for CrespConfig."""
        artifacts = []
        for output in self.outputs:
            # Prepare the base artifact dictionary
            artifact_base = {}
            if isinstance(output, str):
                artifact_base["path"] = output
            elif isinstance(output, dict) and "path" in output:
                artifact_base = output.copy()  # Start with the user-provided dict
            else:
                # Skip invalid output definitions
                console.print(
                    f"[yellow]Warning: Skipping invalid output definition in stage '{self.stage_id}': {output}[/yellow]"
                )
                continue

            # Ensure reproduction settings are present, using stage defaults if needed
            repro_config = artifact_base.get("reproduction", {})
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
progress: Optional[Progress] = None


class Workflow:
    """Workflow class for managing experiment stages and execution."""

    def __init__(
        self,
        title: str,
        authors: List[Dict[str, str]],
        description: Optional[str] = None,
        config_path: str = "cresp.yaml",
        seed: Optional[int] = None,
        use_rich: bool = True,
        mode: str = "experiment",
        skip_unchanged: bool = False,
        reproduction_failure_mode: str = "stop",
        save_reproduction_report: bool = True,
        reproduction_report_path: str = "reproduction_report.md",
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
            save_reproduction_report: Whether to save a report in reproduction mode.
            reproduction_report_path: Path for the reproduction report.
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
        self._stages: Dict[str, StageFunction] = {}
        self._executed_stages: Set[str] = set()
        self._validation_results: List[Dict[str, Any]] = []
        self._stage_validation_status: Dict[str, Optional[bool]] = {}
        config_file_path = Path(config_path)
        self.reproduction_failure_mode = reproduction_failure_mode
        self.save_reproduction_report = save_reproduction_report
        self.reproduction_report_path = reproduction_report_path
        self._seed = seed
        self._verbose_seed_setting = verbose_seed_setting
        self._seed_initialized = False
        # Cache for storing results of _run_stage within a single workflow.run() call
        self._run_cache: Dict[
            str, Tuple[Any, List[Tuple[str, str]], Optional[Union[bool, object]]]
        ] = {}

        # --- Determine active output directory based on mode ---
        self.experiment_output_dir = Path(experiment_output_dir)
        self.reproduction_output_dir = Path(reproduction_output_dir)
        if self.mode == "experiment":
            self.active_output_dir = self.experiment_output_dir
        elif self.mode == "reproduction":
            self.active_output_dir = self.reproduction_output_dir
        else:
            # Default or fallback if mode is somehow different
            self.active_output_dir = Path(".") # Or raise an error?

        self.shared_data_dir = Path(shared_data_dir)

        try:
            self.config = CrespConfig.load(config_file_path)
            if self.use_rich:
                console.print(
                    f"[dim]Loaded existing configuration from [bold]{self.config.path}[/bold][/dim]"
                )
        except FileNotFoundError:
            if mode == "reproduction":
                raise FileNotFoundError(
                    f"Configuration file '{config_file_path}' not found, required for reproduction mode."
                )
            # Use the helper function from utils
            self.config = create_workflow_config(title, authors, description, str(config_file_path))
            if self.use_rich:
                console.print(
                    f"[dim]Creating new configuration at [bold]{self.config.path}[/bold][/dim]"
                )
        except Exception as e:
            raise ValueError(f"Error loading or creating configuration '{config_file_path}': {e}")

        current_config_seed = self.config.data.get("reproduction", {}).get("random_seed")
        if seed is not None:
            if current_config_seed is not None and seed != current_config_seed and self.use_rich:
                console.print(
                    f"[yellow]Warning: Overriding random seed from config ({current_config_seed}) with provided seed ({seed})[/yellow]"
                )
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
                console.print(
                    "[yellow]Warning: No random seed set. Skipping seed initialization.[/yellow]"
                )
            return

        libraries = set_seed(self._seed, verbose=verbose)

        if self.use_rich and verbose:
            console.print(
                f"[dim]Random seeds set to {self._seed} for: {', '.join(libraries)}[/dim]"
            )

    @property
    def seed(self) -> Optional[int]:
        """Get the current random seed."""
        return self._seed

    def get_output_path(self, relative_path: Union[str, Path]) -> Path:
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

    def get_shared_data_path(self, relative_path: Union[str, Path]) -> Path:
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

    def get_dataloader_kwargs(self) -> Dict[str, Any]:
        """Get kwargs for PyTorch DataLoader to ensure reproducibility."""
        if self._seed is None:
            return {}

        return get_reproducible_dataloader_kwargs(self._seed)

    def stage(
        self,
        id: Optional[str] = None,
        description: Optional[str] = None,
        outputs: Optional[List[Union[str, Dict[str, Any]]]] = None,
        dependencies: Optional[List[str]] = None,
        parameters: Optional[Dict[str, Any]] = None,
        reproduction_mode: str = "strict",
        tolerance_absolute: Optional[float] = None,
        tolerance_relative: Optional[float] = None,
        similarity_threshold: Optional[float] = None,
        skip_if_unchanged: Optional[bool] = None,
    ) -> Callable[[Callable[..., R]], StageFunction]:
        """Decorator for registering a stage function."""

        def decorator(func: Callable[..., R]) -> StageFunction:
            stage_id = id or func.__name__
            final_skip_setting = (
                self.skip_unchanged if skip_if_unchanged is None else skip_if_unchanged
            )

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
            raise ValueError(
                f"Stage ID '{stage_func.stage_id}' already registered for this workflow instance."
            )

        self._stages[stage_func.stage_id] = stage_func

        stage_config_data = stage_func.to_stage_config()
        existing_stage_data = self.config.get_stage(stage_func.stage_id)

        # Use batch update for potential modifications
        with self.config.batch_update():
            if existing_stage_data is None:
                # Add new stage if it doesn't exist in the config file
                try:
                    self.config.add_stage(
                        stage_config_data, defer_save=True
                    )  # defer_save is implicit in batch_update
                    # Log addition if needed (rich only)
                    # if self.use_rich:
                    #     console.print(f"[dim]Added new stage '{stage_func.stage_id}' to configuration.[/dim]")
                except ValueError as e:
                    if self.use_rich:
                        console.print(
                            f"[yellow]Warning: Could not add stage '{stage_func.stage_id}' to config object: {e}[/yellow]"
                        )
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
                config_output_paths = {
                    o.get("path") for o in existing_stage_data.get("outputs", []) if o.get("path")
                }
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
        expected_files_config: Dict[str, Dict[str, Any]] = {
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
            elif isinstance(output_decl, dict) and "path" in output_decl:
                path_str = output_decl["path"]
                output_specific_repro_config = output_decl.get("reproduction", {})
            else:
                continue  # Skip invalid declaration

            # --- Find corresponding files in config and validate them ---
            path = Path(path_str)
            files_to_validate_for_this_decl: Dict[str, Dict[str, Any]] = {}

            # Case 1: Declaration is an exact file path present in config
            if path_str in expected_files_config:
                # Check if file exists on disk
                if not Path(path_str).exists():
                    # console.print(f"[dim]  Debug: Declared file {path_str} not found on disk.[/dim]")
                    return False  # Exact file missing
                files_to_validate_for_this_decl[path_str] = expected_files_config[path_str]
            # Case 2: Declaration is a directory path
            elif path.is_dir():  # Check if the declared path is an existing directory
                found_match_in_dir = False
                # Find all config files that are within this directory
                for expected_path, expected_cfg in expected_files_config.items():
                    norm_expected = os.path.normpath(expected_path)
                    norm_declared = os.path.normpath(path_str)
                    # Check if expected_path starts with declared_path + separator
                    if norm_expected.startswith(norm_declared + os.sep):
                        # Check if file exists on disk
                        if not Path(expected_path).exists():
                            # console.print(f"[dim]  Debug: File {expected_path} (under {path_str}) not found on disk.[/dim]")
                            return False  # File within directory missing
                        files_to_validate_for_this_decl[expected_path] = expected_cfg
                        found_match_in_dir = True
                if not found_match_in_dir:
                    # console.print(f"[dim]  Debug: Declared dir {path_str} exists, but no hashed files found within it in config.[/dim]")
                    return False  # Directory declared, but no known files within found in config
            else:
                # Declared path is not in config directly and not an existing directory
                # console.print(f"[dim]  Debug: Declared path {path_str} not in config and not a directory.[/dim]")
                return False

            # --- Validate the collected files for this declaration ---
            if not files_to_validate_for_this_decl:
                # This case might mean the declaration exists but has no hashed files associated yet.
                # console.print(f"[dim]  Debug: No files found to validate for declaration {path_str}.[/dim]")
                return False

            for file_path_str, file_cfg in files_to_validate_for_this_decl.items():
                files_checked_count += 1

                # Determine final validation parameters (File > Declaration > Stage)
                file_specific_repro_config = file_cfg.get("reproduction", {})

                final_repro_mode = file_specific_repro_config.get(
                    "mode", output_specific_repro_config.get("mode", default_repro_mode)
                )
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

                # Perform validation using the hash from config
                success, _ = validate_artifact(
                    file_path_str,
                    file_cfg["hash"],  # Hash must exist from earlier check
                    validation_type=final_repro_mode,
                    tolerance_absolute=final_tol_abs,
                    tolerance_relative=final_tol_rel,
                    similarity_threshold=final_sim_thresh,
                )

                if not success:
                    # console.print(f"[dim]  Debug: Validation failed for {file_path_str}.[/dim]")
                    all_match = False
                    break  # Stop checking this declaration if one file fails

            if not all_match:
                break  # Stop checking other declarations if one failed

        # Return True only if all declarations were processed,
        # at least one file was checked, and all matched.
        # console.print(f"[dim]  Debug: _check_outputs_unchanged result for {stage_id}: {all_match and files_checked_count > 0}[/dim]")
        return all_match and files_checked_count > 0

    def _run_stage(
        self, stage_id: str
    ) -> Tuple[Any, List[Tuple[str, str]], Optional[Union[bool, object]]]:
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
                console.print(
                    f"[red]Error running dependency '{dep_id}' for stage '{stage_id}': {dep_e}[/red]"
                )
                # Re-raise to halt the current stage processing
                raise

        # 3. --- Skip Check for current stage ---
        should_skip = False
        skip_reason = ""
        if stage_func.skip_if_unchanged:
            if dependencies_were_run_or_failed:
                # console.print(f"[dim]Cannot skip {stage_id}: Dependencies were run/failed.[/dim]")
                skip_reason = "dependencies changed"
            else:
                # console.print(f"[dim]Checking outputs unchanged for {stage_id}...[/dim]")
                outputs_unchanged = self._check_outputs_unchanged(stage_id)
                if outputs_unchanged:
                    should_skip = True
                else:
                    # console.print(f"[dim]Cannot skip {stage_id}: Outputs changed.[/dim]")
                    skip_reason = "outputs changed"
        else:
            # console.print(f"[dim]Cannot skip {stage_id}: skip_if_unchanged is False.[/dim]")
            skip_reason = "skip_if_unchanged=False"

        if should_skip:
            if self.use_rich:
                console.print(
                    f"[green]✓ Skipping stage [bold]{stage_id}[/bold] (outputs unchanged, dependencies OK)[/green]"
                )
            # Mark as executed (skipped)
            self._executed_stages.add(stage_id)  # Keep track of processed stages
            status = SKIPPED
            result_tuple = (None, [], status)
            self._stage_validation_status[stage_id] = status  # Store final status
            self._run_cache[stage_id] = result_tuple  # Cache the result
            return result_tuple

        # 4. --- If not skipped, execute the stage ---
        # console.print(f"[dim]Executing stage {stage_id} (Reason not skipped: {skip_reason})[/dim]")
        calculated_hashes = []
        result = None
        stage_validation_passed: Optional[Union[bool, object]] = None  # Can be True, False, None

        # --- Set random seeds ---
        if self._seed is not None:
            verbose_seed = self._verbose_seed_setting and not self._seed_initialized
            self.set_random_seeds(verbose=verbose_seed)
            self._seed_initialized = True  # Mark as initialized after first potential setting

        # --- Stage Execution ---
        if self.use_rich:
            # Temporarily stop progress bar for cleaner stage execution logs
            progress_active = progress is not None
            if progress_active:
                try:
                    progress.stop()
                except Exception:
                    progress_active = False  # Handle cases where progress might be finished

            console.print(f"[bold blue]Executing {stage_id}...[/bold blue]")
            start_time = time.time()

            # Actually run the user's stage function
            try:
                result = stage_func()  # Call the __call__ method of StageFunction
            except Exception as e:
                console.print(
                    f"[red]  ✗ Stage [bold]{stage_id}[/bold] execution failed: {str(e)}[/red]"
                )
                # Resume progress if needed before re-raising
                if progress_active:
                    progress.start()
                raise  # Re-raise the exception to be caught by the main run loop

            end_time = time.time()
            execution_time = end_time - start_time
            console.print(f"[dim]Stage completed in {execution_time:.2f}s[/dim]")

            # Resume progress display
            if progress_active:
                try:
                    progress.start()
                except Exception:
                    pass  # Ignore errors if progress finished
        else:
            # Non-rich execution
            print(f"Executing {stage_id}...")
            # Need similar error handling for non-rich mode
            try:
                result = stage_func()
            except Exception as e:
                print(f"  Error: Stage {stage_id} execution failed: {str(e)}")
                raise  # Re-raise
            print(f"Stage {stage_id} completed.")

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

    def _update_output_hashes(
        self, stage_id: str, declared_outputs: List[Union[str, Dict[str, Any]]]
    ) -> List[Tuple[str, str]]:
        """Calculate and store hashes for stage outputs in experiment mode."""
        hashes_calculated = []
        # Use batch update for efficiency when hashing multiple files/directories
        with self.config.batch_update():
            for output_decl in declared_outputs:
                # --- Determine path and hash method from declaration ---
                path_str: Optional[str] = None
                hash_method = "sha256"  # Default hash method
                is_shared = False # Heuristic flag

                if isinstance(output_decl, str):
                    path_str = output_decl
                    # Apply heuristic: path without separator is likely shared
                    if '/' not in path_str and '\\\\' not in path_str:
                       is_shared = True
                elif isinstance(output_decl, dict):
                    path_str = output_decl.get("path")
                    # Allow overriding hash method per output declaration
                    hash_method = output_decl.get("hash_method", hash_method)
                    # Check for explicit scope or apply heuristic
                    if output_decl.get("scope") == "shared":
                        is_shared = True
                    elif output_decl.get("scope") == "output":
                        is_shared = False
                    elif path_str and '/' not in path_str and '\\\\' not in path_str:
                        is_shared = True


                if not path_str:
                    console.print(
                        f"[yellow]Warning: Invalid output declaration in stage '{stage_id}': {output_decl}[/yellow]"
                    )
                    continue

                # --- Resolve path based on heuristic/scope ---
                try:
                    if is_shared:
                        resolved_path = self.get_shared_data_path(path_str)
                    else:
                        resolved_path = self.get_output_path(path_str)
                except Exception as path_resolve_e:
                    console.print(
                        f"[yellow]Warning: Could not resolve path '{path_str}' for stage '{stage_id}': {path_resolve_e}[/yellow]"
                    )
                    continue

                # --- Check existence using the RESOLVED path ---
                try:
                    if not resolved_path.exists():
                        if self.use_rich:
                            console.print(
                                f"[yellow]Warning: Output path does not exist after running stage '{stage_id}': {resolved_path} (declared as '{path_str}')[/yellow]"
                            )
                        continue

                    # --- Hash File or Directory Contents using RESOLVED path ---
                    if resolved_path.is_file():
                        hash_value = calculate_artifact_hash(resolved_path, method=hash_method)
                        # Update config using the ORIGINAL declared path_str as the key
                        self.config.update_hash(stage_id, path_str, hash_value, hash_method)
                        hashes_calculated.append((path_str, hash_value))
                        # Minimal logging here, summary printed later
                    elif resolved_path.is_dir():
                        if self.use_rich:
                            console.print(
                                f"[dim]Hashing contents of directory [bold]{resolved_path}[/bold] (declared as '{path_str}')...[/dim]"
                            )
                        files_hashed_count = 0
                        for file_path in resolved_path.rglob("*"):
                            if file_path.is_file():
                                try:
                                    # Use the hash_method determined for the directory declaration
                                    file_hash_value = calculate_artifact_hash(
                                        file_path, method=hash_method
                                    )
                                    # Store path relative to workspace root for consistency?
                                    # OR store path relative to the *resolved_path* base?
                                    # Let's stick to storing the config key based on the original declaration structure.
                                    # If path_str was "data", the key in config for file "data/a.txt" should be "data/a.txt".
                                    # If path_str was "outputs", key for "outputs/b.txt" should be "outputs/b.txt".
                                    # calculate relative path from CWD for the config key
                                    try:
                                        relative_key_path_str = str(
                                            file_path.resolve().relative_to(Path.cwd().resolve())
                                        )
                                        # Ensure the key path uses the same structure as declaration
                                        # This part is tricky. Let's assume the user wants the key to be the full relative path from CWD for now.
                                    except ValueError:
                                        relative_key_path_str = str(file_path.resolve())


                                    # Update config using the calculated relative path as key
                                    self.config.update_hash(
                                        stage_id,
                                        relative_key_path_str,
                                        file_hash_value,
                                        hash_method,
                                    )
                                    # Record the originally declared path string and hash for the summary log? No, record the actual path hashed.
                                    hashes_calculated.append(
                                        (relative_key_path_str, file_hash_value)
                                    )
                                    files_hashed_count += 1
                                except Exception as file_e:
                                    if self.use_rich:
                                        console.print(
                                            f"[yellow]  Warning: Failed to hash file {file_path}: {str(file_e)}[/yellow]"
                                        )
                        if self.use_rich and files_hashed_count > 0:
                             console.print(
                                f"[dim]Hashed {files_hashed_count} files in [bold]{resolved_path}[/bold].[/dim]"
                            )
                    else:
                        # Handle other path types? Symlinks?
                        if self.use_rich:
                            console.print(
                                f"[yellow]Warning: Resolved output path is not a file or directory: {resolved_path}[/yellow]"
                            )

                except Exception as e:
                    if self.use_rich:
                        console.print(
                            f"[yellow]Warning: Failed to process output {resolved_path} (declared as '{path_str}') for stage '{stage_id}': {str(e)}[/yellow]"
                        )

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
                console.print(
                    f"[yellow]Warning: No reference outputs/hashes found in config for stage '{stage_id}' to validate against.[/yellow]"
                )
            return True  # Consider valid if nothing to compare against

        if not declared_outputs:
            return True  # Nothing declared in code, nothing to validate here.

        # Create a lookup for config outputs by path (must have hash)
        expected_files_config: Dict[str, Dict[str, Any]] = {
            cfg_out["path"]: cfg_out
            for cfg_out in stage_config.get("outputs", [])
            if cfg_out.get("path") and cfg_out.get("hash")
        }

        if not expected_files_config:
            if self.use_rich:
                console.print(
                    f"[yellow]Warning: Outputs found in config for stage '{stage_id}', but none have recorded hashes.[/yellow]"
                )
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
            path_str: Optional[str] = None
            is_shared = False # Heuristic

            if isinstance(output_decl, str):
                path_str = output_decl
                if '/' not in path_str and '\\\\' not in path_str:
                    is_shared = True
            elif isinstance(output_decl, dict) and "path" in output_decl:
                path_str = output_decl["path"]
                output_specific_repro_config = output_decl.get("reproduction", {})
                if output_decl.get("scope") == "shared": is_shared = True
                elif output_decl.get("scope") == "output": is_shared = False
                elif path_str and '/' not in path_str and '\\\\' not in path_str: is_shared = True
            else:
                continue  # Skip invalid declaration

            # --- Resolve path for validation ---
            try:
                if is_shared:
                    resolved_path = self.get_shared_data_path(path_str)
                else:
                    resolved_path = self.get_output_path(path_str)
            except Exception as path_resolve_e:
                console.print(
                    f"[yellow]Warning: Could not resolve path '{path_str}' for validation in stage '{stage_id}': {path_resolve_e}[/yellow]"
                )
                continue


            # --- Find corresponding files in config based on ORIGINAL path_str ---
            # We still use path_str to lookup in expected_files_config which uses declared paths as keys
            files_to_validate_for_this_decl: Dict[str, Dict[str, Any]] = {}
            found_config_match = False

            # Case 1: Exact file path match in config
            if path_str in expected_files_config:
                files_to_validate_for_this_decl[path_str] = expected_files_config[path_str]
                found_config_match = True
            # Case 2: Directory path - find config entries within
            else:
                # Check config paths starting with the declared path_str
                for expected_path, expected_cfg in expected_files_config.items():
                    norm_expected = os.path.normpath(expected_path)
                    norm_declared = os.path.normpath(path_str)
                    if norm_expected.startswith(norm_declared + os.sep):
                        # The key remains the declared path from config
                        files_to_validate_for_this_decl[expected_path] = expected_cfg
                        found_config_match = True

            if not found_config_match:
                # Declared output has no corresponding hashed entry in config
                if self.use_rich:
                    console.print(
                        f"[yellow]Warning: No reference hash found in config matching output declaration '{path_str}' for stage '{stage_id}'. Cannot validate.[/yellow]"
                    )
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
                # Apply the same heuristic to the key from the config dict
                expected_is_shared = False
                if '/' not in expected_path_key and '\\\\' not in expected_path_key:
                    expected_is_shared = True
                # We assume the scope defined in the @stage decorator applies to all files under a dir declaration?
                # Or should we check scope per file in config? Let's stick to the heuristic for now.

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
                    actual_file_path_display = expected_path_key # Display the problematic key
                else:
                    actual_file_path_display = str(actual_file_path_to_check) # For logging
                    # Check if the *actual* file exists on disk before attempting validation
                    if not actual_file_path_to_check.exists():
                        message = f"Output file not found: {actual_file_path_to_check}"
                        success = False
                        final_repro_mode = "N/A"  # Mode doesn't apply if file is missing
                    else:
                        # Determine final validation parameters (File > Declaration > Stage)
                        file_specific_repro_config = file_cfg.get("reproduction", {})
                        final_repro_mode = file_specific_repro_config.get(
                            "mode", output_specific_repro_config.get("mode", default_repro_mode)
                        )
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
                            output_specific_repro_config.get(
                                "similarity_threshold", default_sim_thresh
                            ),
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
                        "file": actual_file_path_display, # Report the actual path checked
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

    def run(self, stage_id: Optional[str] = None) -> Dict[str, Any]:
        """Run workflow stages, either a single one or all in order."""
        global progress  # Use the global progress variable

        results = {}
        self._validation_results = []  # Clear previous validation results
        self._stage_validation_status = {}  # Clear previous stage statuses
        self._executed_stages = set()  # Clear executed stages for this run
        self._run_cache = {}  # Clear run cache for this run
        workflow_failed = False  # Track if any stage failed execution or reproduction

        run_title = f"Workflow: {self.title}"
        run_subtitle = f"Mode: {self.mode.capitalize()}"
        if self.mode == "reproduction":
            run_subtitle += f" (Fail on: {self.reproduction_failure_mode})"
        if self._seed is not None:
            run_subtitle += f" | Seed: {self._seed}"

        if self.use_rich:
            console.print(
                Panel(
                    f"[bold blue]{self.title}[/bold blue]",
                    title="CRESP Workflow Run",
                    subtitle=run_subtitle,
                    expand=False,
                )
            )

            # Determine execution plan
            execution_plan: List[str]
            if stage_id:
                # Need to include dependencies if running a single stage
                execution_plan = self._resolve_execution_order(target_stage=stage_id)
                console.print(
                    f"[yellow]Running target stage [bold]{stage_id}[/bold] and its dependencies:[/yellow]"
                )
            else:
                execution_plan = self._resolve_execution_order()
                console.print("[yellow]Execution plan (all stages):[/yellow]")

            # Display execution plan
            plan_table = Table(show_header=False, box=None, padding=(0, 1))
            plan_table.add_column()  # Index
            plan_table.add_column()  # Stage ID
            plan_table.add_column(style="dim")  # Description
            for i, stage_to_run in enumerate(execution_plan):
                desc = self._stages[stage_to_run].description
                plan_table.add_row(
                    f" {i+1}.".ljust(4), f"[bold cyan]{stage_to_run}[/bold cyan]", f": {desc}"
                )
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
                        stage_task_description = (
                            f"[cyan]Stage {idx+1}/{len(execution_plan)}: {curr_stage_id}"
                        )
                        stage_task = progress.add_task(
                            stage_task_description, total=1, start=False
                        )  # Don't start task immediately

                        # --- Run the stage ---
                        progress.start_task(stage_task)
                        calculated_hashes_for_stage = []
                        stage_validation_status: Optional[Union[bool, object]] = None
                        stage_failed_execution = False
                        try:
                            # This call handles execution, dependency runs, hashing/validation
                            stage_result, calculated_hashes_for_stage, stage_validation_status = (
                                self._run_stage(curr_stage_id)
                            )
                            results[curr_stage_id] = stage_result

                        except ReproductionError as repro_err:
                            # Specific error raised by _run_stage if repro failed and mode is 'stop'
                            stage_validation_status = False  # Mark as failed
                            workflow_failed = True
                            progress.update(
                                stage_task,
                                completed=1,
                                description=f"[red]✗ {curr_stage_id} failed reproduction (STOPPED)",
                            )
                            console.print(
                                f"  [bold red]Workflow stopped due to reproduction failure in stage '{curr_stage_id}'.[/bold red]"
                            )
                            break  # Exit the loop
                        except Exception as exec_err:
                            # General execution error within the stage function itself
                            stage_failed_execution = True
                            workflow_failed = True
                            progress.update(
                                stage_task,
                                completed=1,
                                description=f"[red]✗ {curr_stage_id} failed execution",
                            )
                            # Error details already printed in _run_stage
                            if (
                                self.reproduction_failure_mode == "stop"
                            ):  # Stop on execution errors too?
                                console.print(
                                    f"  [bold red]Workflow stopped due to execution error in stage '{curr_stage_id}'.[/bold red]"
                                )
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
                                    console.print(
                                        f"    [cyan dim]{path_hash}:[/cyan dim] [dim]{hash_val[:8]}...[/dim]"
                                    )

                            # Check for reproduction failure *after* logging, if continuing
                            if (
                                stage_validation_status is False
                                and self.reproduction_failure_mode == "continue"
                            ):
                                console.print(
                                    f"  [yellow]Continuing workflow after reproduction failure in stage '{curr_stage_id}'.[/yellow]"
                                )

            finally:
                # Ensure global progress is cleared after the run
                progress = None

            # --- Print Final Summary ---
            console.print()  # Spacer
            summary_table = Table(
                title="Workflow Execution Summary",
                show_header=True,
                header_style="bold magenta",
                box=rich.box.ROUNDED,
            )
            summary_table.add_column("Stage", style="cyan", no_wrap=True)
            summary_table.add_column("Status", style="default", justify="center")
            summary_table.add_column("Outputs", style="blue", overflow="fold")
            summary_table.add_column("Dependencies", style="yellow")
            if self.mode == "reproduction":
                summary_table.add_column("Reproduction", style="default", justify="center")

            # Use the original execution plan for the summary order
            summary_stages = execution_plan

            for stage_id_summary in summary_stages:
                if stage_id_summary not in self._stages: continue # Should not happen normally

                stage_func = self._stages[stage_id_summary]
                final_status_str = "[grey50]Not run"
                repro_status_str = "N/A"

                # Determine status based on execution and validation results
                validation_status = self._stage_validation_status.get(stage_id_summary)
                stage_ran_or_skipped = stage_id_summary in self._executed_stages # Use this set

                if stage_ran_or_skipped:
                    if validation_status is SKIPPED:
                        final_status_str = "[yellow]Skipped"
                        repro_status_str = "⚪ Skipped"
                    elif validation_status is False: # Failed reproduction
                        final_status_str = "[green]Ran" # The stage itself ran
                        repro_status_str = "❌ Failed"
                    elif validation_status is True: # Passed reproduction
                        final_status_str = "[green]Ran"
                        repro_status_str = "✅ Passed"
                    else: # Ran in experiment mode or no validation needed (validation_status is None)
                        final_status_str = "[green]Ran"
                        repro_status_str = "N/A"
                # else: Stage was not executed (e.g., workflow stopped early) - final_status_str remains "Not run"

                # Get output paths from the StageFunction definition for display
                # output_paths = []
                # if stage_func.outputs:
                #     for out in stage_func.outputs:
                #         if isinstance(out, str):
                #             output_paths.append(out)
                #         elif isinstance(out, dict) and "path" in out:
                #             output_paths.append(out["path"])
                # outputs_str = ", ".join(output_paths) if output_paths else "[dim]None"

                # --- Construct resolved output paths for display ---
                outputs_str_parts = []
                if stage_func.outputs:
                    for out in stage_func.outputs:
                        path_str = None
                        description = None
                        is_shared = False # Heuristic

                        if isinstance(out, str):
                            path_str = out
                            # Apply heuristic: path without separator is likely shared
                            if '/' not in path_str and '\\\\' not in path_str:
                               is_shared = True
                        elif isinstance(out, dict) and "path" in out:
                            path_str = out["path"]
                            description = out.get("description")
                            # Check scope or apply heuristic
                            if out.get("scope") == "shared": is_shared = True
                            elif out.get("scope") == "output": is_shared = False
                            elif path_str and '/' not in path_str and '\\\\' not in path_str: is_shared = True

                        if path_str:
                            # Construct the resolved path string for display
                            if is_shared:
                                # Use the relative path defined in the workflow init
                                resolved_display_path = f"{self.shared_data_dir.as_posix()}/{path_str}"
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
                deps_str = ", ".join(deps_list) if deps_list else "[dim]None"

                row_data = [stage_id_summary, final_status_str, outputs_display, deps_str] # Use outputs_display
                if self.mode == "reproduction":
                    row_data.append(repro_status_str)

                summary_table.add_row(*row_data)

            console.print(summary_table)

            # --- Save Reproduction Report (if applicable) ---
            if (
                self.mode == "reproduction"
                and self.save_reproduction_report
                and self._validation_results
            ):
                self._save_reproduction_report()

        else:
            # --- Non-Rich Execution Path ---
            print(f"Running Workflow: {self.title} (Mode: {self.mode})")
            execution_plan: List[str]
            if stage_id:
                execution_plan = self._resolve_execution_order(target_stage=stage_id)
                print(f"Execution Plan (Target: {stage_id}): {', '.join(execution_plan)}")
            else:
                execution_plan = self._resolve_execution_order()
                print(f"Execution Plan (All Stages): {', '.join(execution_plan)}")

            for curr_stage_id in execution_plan:
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
                            print(f"  ERROR: Reproduction failed. Stopping workflow.")
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
                        print(f"  ERROR: Stopping workflow due to execution error.")
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
             console.print(f"[dim]Skipping configuration save in [bold]{self.mode}[/bold] mode.[/dim]")

        if workflow_failed:
            if self.reproduction_failure_mode == "continue":
                if self.use_rich:
                    console.print(
                        "[bold yellow]Workflow completed with failures (check summary and report).[/bold yellow]"
                    )
                else:
                    print("WARNING: Workflow completed with failures.")
            # If failure mode was 'stop', the loop was already broken.
            # We might want to raise an exception here to signal failure programmatically.
            # raise WorkflowExecutionError("Workflow failed during execution.") # Consider adding a specific exception

        return results

    def _resolve_execution_order(self, target_stage: Optional[str] = None) -> List[str]:
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

    def save_config(self, path: Optional[str] = None) -> bool:
        """Save workflow configuration to file."""
        save_path = Path(path) if path else self.config.path
        if not save_path:
            console.print(
                "[red]Error: Cannot save configuration, no path specified or loaded.[/red]"
            )
            return False

        success = self.config.save(save_path)

        if success and self.use_rich:
            console.print(
                f"[green]Configuration explicitly saved to [bold]{save_path}[/bold][/green]"
            )
        elif not success and self.use_rich:
            console.print(
                f"[red]Failed to explicitly save configuration to [bold]{save_path}[/bold][/red]"
            )

        return success

    def visualize(self) -> None:
        """Visualize the workflow structure using Rich."""
        if not self.use_rich:
            print("Rich visualization requires the 'rich' package to be installed.")
            # Basic text visualization could be added here as a fallback
            return

        console.print(Panel(f"[bold]Workflow Structure: {self.title}[/bold]", expand=False))

        try:
            execution_order = self._resolve_execution_order()

            table = Table(
                title="Stage Execution Order & Details",
                show_header=True,
                header_style="bold magenta",
                box=rich.box.ROUNDED,
                show_lines=True
            )
            table.add_column("Order", style="dim", justify="right")
            table.add_column("Stage ID", style="cyan", no_wrap=True)
            table.add_column("Description", style="blue")
            table.add_column("Dependencies", style="yellow")
            table.add_column("Declared Outputs", style="green", overflow="fold") # Added overflow

            for idx, stage_id in enumerate(execution_order):
                if stage_id not in self._stages:
                    continue  # Skip if somehow not registered

                stage = self._stages[stage_id]
                deps = ", ".join(stage.dependencies) if stage.dependencies else "[dim]None"

                # Format outputs nicely
                outputs_str_parts = []
                if stage.outputs:
                    for out in stage.outputs:
                        if isinstance(out, str):
                            outputs_str_parts.append(out)
                        elif isinstance(out, dict) and "path" in out:
                            out_str = out["path"]
                            if out.get("description"):
                                out_str += f" ([dim]{out['description']}[/dim])"
                            outputs_str_parts.append(out_str)
                outputs_display = "\n".join(outputs_str_parts) if outputs_str_parts else "[dim]None"

                table.add_row(str(idx + 1), stage_id, stage.description, deps, outputs_display)

            console.print(table)

        except ValueError as e:  # Catch circular dependencies or missing stages
            console.print(f"[red]Error resolving workflow order for visualization: {str(e)}[/red]")
        except Exception as e:
            console.print(f"[red]An unexpected error occurred during visualization: {str(e)}[/red]")

    def _save_reproduction_report(self):
        """Generate and save the reproduction report in Markdown format."""
        if not self._validation_results:
            if self.use_rich:
                console.print(
                    "[dim]No validation results recorded, skipping report generation.[/dim]"
                )
            return

        report_path = Path(self.reproduction_report_path)
        if self.use_rich:
            console.print(
                f"[dim]Generating reproduction report at [bold]{report_path}[/bold]...[/dim]"
            )

        # Group results by stage, maintaining order if possible (dicts are ordered in Python 3.7+)
        results_by_stage: Dict[str, List[Dict[str, Any]]] = {}
        processed_stages = set()  # Keep track of stages added to maintain run order
        for stage_id_run in self._resolve_execution_order():  # Get order stages were run/attempted
            results_for_stage = [r for r in self._validation_results if r["stage"] == stage_id_run]
            if results_for_stage:
                results_by_stage[stage_id_run] = results_for_stage
                processed_stages.add(stage_id_run)
        # Add any remaining stages that might have results but weren't in the resolved order (shouldn't happen)
        for result in self._validation_results:
            stage_id = result["stage"]
            if stage_id not in processed_stages:
                if stage_id not in results_by_stage:
                    results_by_stage[stage_id] = []
                results_by_stage[stage_id].append(result)

        # Build Markdown report
        report_lines = [
            f"# CRESP Reproduction Report",
            f"",
            f"- **Workflow:** {self.title}",
            f"- **Configuration:** `{self.config.path}`",
            f"- **Timestamp:** {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"- **Seed:** {self._seed if self._seed is not None else 'Not Set'}",
            f"- **Failure Mode:** {self.reproduction_failure_mode}",
        ]

        overall_passed = True
        passed_count = 0
        failed_count = 0
        details_exist = False  # Track if any messages exist

        # --- Stage Summary Table ---
        stage_summary_lines = [
            "",
            "## Stage Summary",
            "",
            "| Stage | Status | Files Passed | Files Failed |",
        ]
        stage_summary_lines.append("|-------|--------|--------------|--------------|")

        for stage_id, results in results_by_stage.items():
            stage_passed_count = sum(1 for r in results if r["status"] == "Passed")
            stage_failed_count = sum(1 for r in results if r["status"] == "Failed")
            passed_count += stage_passed_count
            failed_count += stage_failed_count

            stage_status = "✅ Passed" if stage_failed_count == 0 else "❌ Failed"
            if stage_failed_count > 0:
                overall_passed = False

            stage_summary_lines.append(
                f"| `{stage_id}` | {stage_status} | {stage_passed_count} | {stage_failed_count} |"
            )

        report_lines.extend(stage_summary_lines)

        # --- Overall Status ---
        report_lines.insert(
            7,
            f"- **Overall Status:** {'✅ Passed' if overall_passed else '❌ Failed'} ({passed_count} passed, {failed_count} failed)",
        )
        report_lines.append("")  # Add blank line after overall status

        # --- Detailed Results per Stage ---
        report_lines.append("## Detailed Results")
        report_lines.append("")

        for stage_id, results in results_by_stage.items():
            stage_status_icon = "✅" if all(r["status"] == "Passed" for r in results) else "❌"
            report_lines.append(f"### {stage_status_icon} Stage: `{stage_id}`")
            report_lines.append("")
            report_lines.append("| File | Status | Mode | Details |")
            report_lines.append("|------|--------|------|---------|")
            for r in results:
                status_symbol = "✅" if r["status"] == "Passed" else "❌"
                message = r["message"].replace("|", "\\|")  # Escape pipe characters in message
                if message and message != "Exact hash match":
                    details_exist = True
                report_lines.append(
                    f"| `{r['file']}` | {status_symbol} {r['status']} | `{r['mode']}` | {message} |"
                )
            report_lines.append("")  # Blank line after each stage table

        # Adjust header if no details were ever present
        if not details_exist:
            for i, line in enumerate(report_lines):
                if "| File | Status | Mode | Details |" in line:
                    report_lines[i] = "| File | Status | Mode |"
                    report_lines[i + 1] = "|------|--------|------|"  # Adjust separator
                elif re.match(r"\| `.*` \| .* \| `.*` \| .* \|", line):  # Match table rows
                    parts = line.split("|")
                    report_lines[i] = "|".join(parts[:4]) + "|"  # Keep only first 3 columns

        # --- Save Report ---
        try:
            report_path.parent.mkdir(parents=True, exist_ok=True)
            with open(report_path, "w", encoding="utf-8") as f:
                f.write("\\n".join(report_lines))
            if self.use_rich:
                console.print(f"[green]✓ Reproduction report saved successfully.[/green]")
        except Exception as e:
            if self.use_rich:
                console.print(
                    f"[red]✗ Failed to save reproduction report to {report_path}: {e}[/red]"
                )
