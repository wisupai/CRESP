"""Main Workflow class implementation."""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, TypeVar, cast, Callable, TypeAlias, Union, Protocol
from rich.console import Console
from rich.table import Table
from rich.progress import TaskID

# Import from other CRESP modules
from ..config import CrespConfig
from ..seed import get_reproducible_dataloader_kwargs, set_seed
from ..utils import create_workflow_config

# Import from workflow submodules
from .stage import StageFunction, SKIPPED
from .execution import resolve_execution_order, run_stage


# Define a dummy console for fallback
class DummyConsole:
    """A dummy console that implements ConsoleProtocol."""

    def print(self, *args: Any, **kwargs: Any) -> None:
        pass


# Import visualization components if available
try:
    from .visualization import console, RICH_AVAILABLE, create_workflow_panel, create_execution_plan_table, create_summary_table
    from .progress import create_workflow_progress, create_overall_task, create_stage_task, update_stage_progress, progress, Progress
except ImportError:
    RICH_AVAILABLE = False
    console = DummyConsole()
    progress = None

# Define type variables
T = TypeVar("T")
R = TypeVar("R")


class ConsoleProtocol(Protocol):
    def print(self, *args: Any, **kwargs: Any) -> None: ...


class Workflow:
    """Main workflow class for managing experiment stages and execution."""

    def __init__(
        self,
        config: Optional[Union[Dict[str, Any], CrespConfig]] = None,
        seed: Optional[int] = None,
        use_rich: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialize a new workflow.

        Args:
            config: Configuration for the workflow, can be a dict or CrespConfig object
            seed: Random seed for reproducibility
            use_rich: Whether to use rich for console output
            **kwargs: Additional keyword arguments passed to config
        """
        self.title = kwargs.get("title", "")
        self.authors = kwargs.get("authors", [])
        self.description = kwargs.get("description")
        self.config_path = kwargs.get("config_path", "cresp.yaml")
        self.mode = kwargs.get("mode", "experiment")
        self.skip_unchanged = kwargs.get("skip_unchanged", False)
        self.reproduction_failure_mode = kwargs.get("reproduction_failure_mode", "stop")
        self.set_seed_at_init = kwargs.get("set_seed_at_init", True)
        self.verbose_seed_setting = kwargs.get("verbose_seed_setting", True)
        self.experiment_output_dir = Path(kwargs.get("experiment_output_dir", "experiment"))
        self.reproduction_output_dir = Path(kwargs.get("reproduction_output_dir", "reproduction"))
        self.shared_data_dir = Path(kwargs.get("shared_data_dir", "shared"))

        # Initialize configuration
        if isinstance(config, CrespConfig):
            self.config = config
        elif config is not None or kwargs:
            # 使用get方法获取参数，不会从kwargs中移除它们
            title = kwargs.get("title")
            authors = kwargs.get("authors")
            description = kwargs.get("description")
            path = kwargs.get("config_path", "cresp.yaml")

            # Handle different cases based on what's provided
            if isinstance(config, dict):
                # If config is a dict, use it directly
                self.config = CrespConfig(config_data=config, path=Path(path))
            elif title and authors:
                # If we have title and authors, create a new config
                self.config = create_workflow_config(title=title, authors=authors, description=description, path=path)
            else:
                # Default fallback
                self.config = CrespConfig()
        else:
            # No config provided
            self.config = CrespConfig()

        # Set random seed if provided
        if seed is not None:
            set_seed(seed)
            self.config.seed = seed

        # Initialize console based on availability and preference
        self.console: ConsoleProtocol = console if use_rich and RICH_AVAILABLE else DummyConsole()

        # Initialize workflow state
        self.stages: Dict[str, StageFunction] = {}
        self.stage_dependencies: Dict[str, Set[str]] = {}
        self.stage_results: Dict[str, Any] = {}
        self.execution_order: List[str] = []
        self.current_stage: Optional[str] = None
        self.start_time = time.time()

        # --- Determine active output directory based on mode ---
        if self.mode == "experiment":
            self.active_output_dir = self.experiment_output_dir
        elif self.mode == "reproduction":
            self.active_output_dir = self.reproduction_output_dir
        else:
            # Default or fallback if mode is somehow different
            self.active_output_dir = Path(".")

        if self.set_seed_at_init and self.config.seed is not None:
            self.set_random_seeds(verbose=self.verbose_seed_setting)

    def set_random_seeds(self, verbose: bool = False) -> None:
        """Set random seeds for all detected libraries."""
        if self.config.seed is None:
            if self.console and verbose:
                self.console.print("[yellow]Warning: No random seed set. Skipping seed initialization.[/yellow]")
            return

        libraries = set_seed(self.config.seed, verbose=verbose)

        if self.console and verbose:
            self.console.print(f"[dim]Random seeds set to {self.config.seed} for: {', '.join(libraries)}[/dim]")

    @property
    def seed(self) -> int | None:
        """Get the random seed."""
        return self.config.seed

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
        if self.config.seed is None:
            return {}

        return get_reproducible_dataloader_kwargs(self.config.seed)

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
    ) -> Callable:
        """Decorator for registering a stage function."""

        def decorator(func):
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
        if stage_func.stage_id in self.stages:
            raise ValueError(f"Stage ID '{stage_func.stage_id}' already registered for this workflow instance.")

        self.stages[stage_func.stage_id] = stage_func

        stage_config_data = stage_func.to_stage_config()
        existing_stage_data = self.config.get_stage(stage_func.stage_id)

        # Use batch update for potential modifications
        with self.config.batch_update():
            if existing_stage_data is None:
                # Add new stage if it doesn't exist in the config file
                try:
                    self.config.add_stage(stage_config_data, defer_save=True)  # defer_save is implicit in batch_update
                except ValueError as e:
                    if self.console:
                        self.console.print(f"[yellow]Warning: Could not add stage '{stage_func.stage_id}' to config object: {e}[/yellow]")
            else:
                # Stage exists, potentially update non-hashed fields if they differ
                needs_update = False
                for key in ["description", "dependencies", "code_handler", "parameters"]:
                    if existing_stage_data.get(key) != stage_config_data.get(key):
                        existing_stage_data[key] = stage_config_data.get(key)
                        needs_update = True
                # Handle outputs: only update paths/descriptions, not hashes/validation from code
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

    def run(self, stage_id: str | None = None) -> dict[str, Any]:
        """Run workflow stages, either a single one or all in order."""
        global progress  # Use the global progress variable

        results = {}
        self._validation_results = []  # Clear previous validation results
        self._stage_validation_status = {}  # Clear previous stage statuses
        self._executed_stages = set()  # Clear executed stages for this run
        self._run_cache = {}  # Clear run cache for this run
        workflow_failed = False  # Track if any stage failed execution or reproduction

        # Determine execution plan
        execution_plan: list[str]
        if stage_id:
            # Need to include dependencies if running a single stage
            execution_plan = resolve_execution_order(self.stages, target_stage=stage_id)
        else:
            execution_plan = resolve_execution_order(self.stages)

        # --- Rich Visualization Path ---
        if self.console:
            # Create and display workflow panel
            panel = create_workflow_panel(self.title, self.mode, self.config.seed, self.config.data.get("authors", []))
            self.console.print(panel)
            self.console.print()  # Add spacing

            # Display execution plan information
            if stage_id:
                self.console.print(f"[bold yellow]🎯 Running Target Stage [bold cyan]{stage_id}[/bold cyan] and Dependencies:[/bold yellow]")
            else:
                self.console.print("[bold yellow]📋 Workflow Execution Plan:[/bold yellow]")

            # Display execution plan in a table
            plan_table = create_execution_plan_table(execution_plan, self.stages)
            self.console.print(plan_table)
            self.console.print()  # Spacer

            # --- Run with Progress Bar ---
            try:
                with create_workflow_progress() as prog:
                    progress = prog  # Assign to global variable

                    overall_task: TaskID = create_overall_task(prog, len(execution_plan))

                    for idx, curr_stage_id in enumerate(execution_plan):
                        stage_task: TaskID = create_stage_task(prog, curr_stage_id, idx, len(execution_plan))

                        # --- Run the stage ---
                        prog.start_task(stage_task)
                        calculated_hashes_for_stage: list[tuple[str, str]] = []
                        stage_validation_status: bool | object | None = None
                        stage_failed_execution = False
                        try:
                            # This call handles execution, dependency runs, hashing/validation
                            stage_result, calculated_hashes_for_stage, stage_validation_status = run_stage(
                                curr_stage_id,
                                self.stages,
                                self.config,
                                self.active_output_dir,
                                self.shared_data_dir,
                                self.mode,
                                self.reproduction_failure_mode,
                                self._executed_stages,
                                self._validation_results,
                                self._stage_validation_status,
                                self._run_cache,
                                self.console,
                            )
                            results[curr_stage_id] = stage_result

                        except ValueError as e:
                            # Error with stage definition or configuration
                            stage_failed_execution = True
                            workflow_failed = True
                            update_stage_progress(prog, stage_task, curr_stage_id, "failed")
                            self.console.print(f"[red]Error: {str(e)}[/red]")
                            if self.reproduction_failure_mode == "stop":
                                break
                        except Exception as e:
                            # General execution error within the stage function itself
                            stage_failed_execution = True
                            workflow_failed = True
                            update_stage_progress(prog, stage_task, curr_stage_id, "failed")
                            # Error details already printed in run_stage
                            if self.reproduction_failure_mode == "stop":
                                self.console.print(f"  [bold red]Workflow stopped due to error in stage '{curr_stage_id}'.[/bold red]")
                                break  # Exit the loop

                        # --- Update Progress Bar Based on Outcome ---
                        if stage_failed_execution:
                            # Already updated in except block
                            pass
                        elif stage_validation_status is SKIPPED:  # Check for sentinel explicitly
                            update_stage_progress(prog, stage_task, curr_stage_id, "skipped")
                        elif stage_validation_status is False:
                            update_stage_progress(prog, stage_task, curr_stage_id, "failed_reproduction")
                            workflow_failed = True  # Mark workflow as failed (even if continuing)
                        elif stage_validation_status is True:
                            update_stage_progress(prog, stage_task, curr_stage_id, "passed_reproduction")
                        else:  # Completed in experiment mode or no validation occurred
                            update_stage_progress(prog, stage_task, curr_stage_id, "completed")

                        # Advance overall progress regardless of individual stage outcome (unless stopped)
                        prog.update(overall_task, advance=1)

                        # --- Post-Stage Actions (if not skipped/failed) ---
                        if not stage_failed_execution and stage_validation_status is not SKIPPED:
                            # Print hashes (only in experiment mode and if run successfully)
                            if self.mode == "experiment" and calculated_hashes_for_stage:
                                self.console.print(f"  [dim]Recorded Hashes for {curr_stage_id}:[/dim]")
                                for path_hash, hash_val in calculated_hashes_for_stage:
                                    self.console.print(f"    [cyan dim]{path_hash}:[/cyan dim] [dim]{hash_val[:8]}...[/dim]")

                            # Check for reproduction failure *after* logging, if continuing
                            if stage_validation_status is False and self.reproduction_failure_mode == "continue":
                                self.console.print(f"  [yellow]Continuing workflow after reproduction failure in stage '{curr_stage_id}'.[/yellow]")

            finally:
                # Ensure global progress is cleared after the run
                progress = None

            # --- Print Final Summary ---
            self.console.print()  # Spacer

            # Create summary table
            summary_table, any_skipped = create_summary_table(
                execution_plan,
                self.stages,
                self._executed_stages,
                self._stage_validation_status,
                self.mode,
                self.active_output_dir,
                self.shared_data_dir,
            )

            self.console.print(summary_table)

            # Add a footnote if stages were skipped
            if any_skipped:
                self.console.print("[dim]⚪ Skipped stages indicate outputs were unchanged from the previous run.[/dim]")
            self.console.print()  # Add spacing before report/save messages

        else:
            # --- Non-Rich Execution Path ---
            print(f"Running Workflow: {self.title} (Mode: {self.mode})")
            print(f"Execution Plan: {', '.join(execution_plan)}")

            for curr_stage_id in execution_plan:
                print(f"---\nRunning Stage: {curr_stage_id}")
                try:
                    stage_result, calculated_hashes_for_stage, stage_validation_status = run_stage(
                        curr_stage_id,
                        self.stages,
                        self.config,
                        self.active_output_dir,
                        self.shared_data_dir,
                        self.mode,
                        self.reproduction_failure_mode,
                        self._executed_stages,
                        self._validation_results,
                        self._stage_validation_status,
                        self._run_cache,
                        self.console,
                    )
                    results[curr_stage_id] = stage_result

                    status_msg = f"Stage {curr_stage_id}: "
                    if stage_validation_status is SKIPPED:
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
        # Only save config in experiment mode
        if self.mode == "experiment":
            self.config.save()
            if self.console:
                self.console.print(f"[dim]Configuration saved to [bold]{self.config.path}[/bold][/dim]")
        elif self.console:
            self.console.print(f"[dim]⚙️ Skipping configuration save in [bold]{self.mode}[/bold] mode.[/dim]")

        if workflow_failed:
            if self.reproduction_failure_mode == "continue":
                if self.console:
                    self.console.print("[bold yellow]Workflow completed with failures (check summary).[/bold yellow]")
                else:
                    print("WARNING: Workflow completed with failures.")

        return results

    def save_config(self, path: str | None = None) -> bool:
        """Save workflow configuration to file."""
        save_path = Path(path) if path else self.config.path
        if not save_path:
            self.console.print("[red]Error: Cannot save configuration, no path specified or loaded.[/red]")
            return False

        success = self.config.save(save_path)

        if success and self.console:
            self.console.print(f"[green]Configuration explicitly saved to [bold]{save_path}[/bold][/green]")
        elif not success and self.console:
            self.console.print(f"[red]Failed to explicitly save configuration to [bold]{save_path}[/bold][/red]")

        return success

    def visualize(self) -> None:
        """Visualize the workflow structure using Rich."""
        if not self.console:
            print("Rich visualization requires the 'rich' package to be installed.")
            return

        try:
            execution_order = resolve_execution_order(self.stages)

            # Use the visualization module's function to create the table
            plan_table = create_execution_plan_table(execution_order, self.stages)

            self.console.print(f"[bold]Workflow Structure: {self.title}[/bold]")
            self.console.print(plan_table)

        except ValueError as e:  # Catch circular dependencies or missing stages
            self.console.print(f"[red]Error resolving workflow order for visualization: {str(e)}[/red]")
        except Exception as e:
            self.console.print(f"[red]An unexpected error occurred during visualization: {str(e)}[/red]")
