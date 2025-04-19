"""Execution utilities for workflow runs."""

from typing import Dict, List, Tuple, Any, Set, Optional
from rich.console import Console

from ..exceptions import ReproductionError
from .stage import SKIPPED
from .visualization import RICH_AVAILABLE, DummyConsole

# Try to import visualization and progress modules
try:
    from .progress import progress

    if RICH_AVAILABLE:
        console: Console = Console()
    else:
        console: Console = DummyConsole()  # type: ignore
except ImportError:
    RICH_AVAILABLE = False
    console: Console = DummyConsole()  # type: ignore
    progress = None


def run_stage(
    stage_id: str,
    stages: Dict[str, Any],
    config,
    active_output_dir,
    shared_data_dir,
    mode: str,
    reproduction_failure_mode: str,
    executed_stages: Set[str],
    validation_results: List[Dict[str, Any]],
    stage_validation_status: Dict[str, Any],
    run_cache: Dict[str, Tuple[Any, List[Tuple[str, str]], Any]],
    use_rich: bool = True,
) -> Tuple[Any, List[Tuple[str, str]], Any]:
    """Run a specific stage, handling dependencies, execution, hashing/validation.

    Args:
        stage_id: ID of the stage to run
        stages: Dictionary mapping stage IDs to StageFunction objects
        config: CrespConfig object
        active_output_dir: Path to the active output directory
        shared_data_dir: Path to the shared data directory
        mode: Workflow mode ("experiment" or "reproduction")
        reproduction_failure_mode: How to handle reproduction failures ("stop" or "continue")
        executed_stages: Set of already executed stage IDs
        validation_results: List to append validation results to
        stage_validation_status: Dictionary to store stage validation status
        run_cache: Cache for stage execution results
        use_rich: Whether to use rich output

    Returns:
        Tuple containing:
            - stage execution result (or None if skipped)
            - list of (path, hash) tuples (empty if skipped or not experiment mode)
            - validation status (bool, None, or SKIPPED sentinel)
    """
    # 1. Check cache for this run first
    if stage_id in run_cache:
        return run_cache[stage_id]

    # Import validation functions here to avoid circular imports
    from .validation import check_outputs_unchanged, update_output_hashes, validate_outputs

    stage_func = stages.get(stage_id)
    if not stage_func:
        raise ValueError(f"Stage function for ID '{stage_id}' not found in memory.")

    # 2. --- Dependency Execution & Status Check ---
    dependencies_were_run_or_failed = False
    for dep_id in stage_func.dependencies:
        # Recursively ensure dependency is processed. Status is retrieved from the call.
        try:
            result_tuple: Tuple[Any, List[Tuple[str, str]], Any] = run_stage(
                dep_id,
                stages,
                config,
                active_output_dir,
                shared_data_dir,
                mode,
                reproduction_failure_mode,
                executed_stages,
                validation_results,
                stage_validation_status,
                run_cache,
                use_rich,
            )
            if result_tuple[2] is not SKIPPED:
                dependencies_were_run_or_failed = True
            # Handle potential upstream failure propagation if needed
            if result_tuple[2] is False and reproduction_failure_mode == "stop":
                # This dependency failed reproduction
                dependencies_were_run_or_failed = True
        except Exception as dep_e:
            # If a dependency raised an error during its execution
            if use_rich and RICH_AVAILABLE:
                console.print(f"[red]Error running dependency '{dep_id}' for stage '{stage_id}': {dep_e}[/red]")
            else:
                print(f"Error running dependency '{dep_id}' for stage '{stage_id}': {dep_e}")
            # Re-raise to halt the current stage processing
            raise

    # 3. --- Skip Check for current stage ---
    should_skip = False
    if stage_func.skip_if_unchanged:
        if dependencies_were_run_or_failed:
            # Cannot skip if dependencies were run/failed
            pass
        else:
            outputs_unchanged = check_outputs_unchanged(stage_id, stage_func, config, active_output_dir, shared_data_dir)
            if outputs_unchanged:
                should_skip = True

    if should_skip:
        executed_stages.add(stage_id)  # Keep track of processed stages
        status = SKIPPED
        result_tuple = (None, [], status)
        stage_validation_status[stage_id] = status  # Store final status
        run_cache[stage_id] = result_tuple  # Cache the result
        return result_tuple

    # 4. --- If not skipped, execute the stage ---
    calculated_hashes: List[Tuple[str, str]] = []
    result = None
    stage_validation_passed = None  # Can be True, False, None

    # --- Stage Execution ---
    if use_rich and RICH_AVAILABLE:
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

    executed_stages.add(stage_id)  # Mark as executed (run)

    # --- Output Handling (Hashing or Validation) ---
    if stage_func.outputs:
        if mode == "experiment":
            calculated_hashes = update_output_hashes(stage_id, stage_func.outputs, config, active_output_dir, shared_data_dir, use_rich)
            stage_validation_passed = None  # No validation in experiment mode
        elif mode == "reproduction":
            stage_validation_passed = validate_outputs(stage_id, stage_func, config, active_output_dir, shared_data_dir, validation_results, use_rich)

            # Check for reproduction failure with stop mode
            if stage_validation_passed is False and reproduction_failure_mode == "stop":
                if use_rich and RICH_AVAILABLE:
                    console.print(f"[red]Stage '{stage_id}' failed reproduction validation. Stopping workflow.[/red]")
                else:
                    print(f"Stage '{stage_id}' failed reproduction validation. Stopping workflow.")
                raise ReproductionError(f"Stage '{stage_id}' failed reproduction validation")

            # calculated_hashes remain empty in reproduction mode
        else:  # No outputs or other modes?
            stage_validation_passed = None  # Default to None if not reproduction
    else:  # No outputs declared for the stage
        stage_validation_passed = None  # Nothing to validate or hash

    # 5. --- Store results and return ---
    result_tuple = (result, calculated_hashes, stage_validation_passed)
    stage_validation_status[stage_id] = stage_validation_passed  # Store final status
    run_cache[stage_id] = result_tuple  # Cache the result
    return result_tuple


def resolve_execution_order(stages: Dict[str, Any], target_stage: Optional[str] = None) -> List[str]:
    """Resolve execution order using topological sort.

    Args:
        stages: Dictionary mapping stage IDs to StageFunction objects
        target_stage: Optional target stage ID to run and its dependencies

    Returns:
        List of stage IDs in execution order
    """
    order = []
    visited = set()  # Nodes permanently visited and added to order
    visiting = set()  # Nodes currently in the recursion stack (for cycle detection)

    all_stages = list(stages.keys())

    def visit(stage_id: str) -> None:
        if stage_id not in stages:
            # This check handles dependencies listed in config but not defined in code
            # OR typos in dependencies list
            raise ValueError(f"Dependency '{stage_id}' not found as a registered stage.")

        if stage_id in visited:
            return  # Already processed
        if stage_id in visiting:
            raise ValueError(f"Circular dependency detected involving stage: {stage_id}")

        visiting.add(stage_id)

        stage = stages[stage_id]
        if stage.dependencies:
            for dep_id in stage.dependencies:
                visit(dep_id)

        visiting.remove(stage_id)
        visited.add(stage_id)
        order.append(stage_id)

    if target_stage:
        if target_stage not in stages:
            raise ValueError(f"Target stage '{target_stage}' is not registered.")
        visit(target_stage)  # Visit only the target and its dependencies
    else:
        # Visit all nodes if no target is specified
        for stage_id in all_stages:
            if stage_id not in visited:
                visit(stage_id)

    return order
