"""Visualization utilities for workflow execution."""

from typing import Any, Dict, List, Optional, Set, Tuple
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Rich imports for visualization
try:
    import rich.box

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


# Create console instance for rich output
class DummyConsole:
    def print(self, *args, **kwargs):
        print(*args)  # Fallback to standard print


if RICH_AVAILABLE:
    console: Console = Console()
else:
    console: Console = DummyConsole()  # type: ignore


def create_workflow_panel(title: str, mode: str, seed: int | None = None, authors: list[dict[str, str]] | None = None) -> Panel:
    """Create a rich panel with workflow title and information.

    Args:
        title: The workflow title
        mode: The workflow mode ("experiment" or "reproduction")
        seed: Optional random seed
        authors: Optional list of author information dictionaries

    Returns:
        A rich Panel object
    """
    if not RICH_AVAILABLE:
        raise ImportError("Rich library is required for visualization")

    run_subtitle = f"Mode: {mode.capitalize()}"
    if mode == "reproduction":
        run_subtitle += f" (Fail on: stop)"  # Default value, can be parameterized
    if seed is not None:
        run_subtitle += f" | Seed: {seed}"

    # Add authors to subtitle if they exist
    if authors:
        authors_str = ", ".join([a.get("name", "Unknown") for a in authors])
        if authors_str:
            run_subtitle += f" | Authors: {authors_str}"

    return Panel(
        f"[bold blue]{title}[/bold blue]",
        title="🚀 CRESP Workflow Run",
        subtitle=f"[dim]{run_subtitle}[/dim]",
        expand=True,
        border_style="bold green",
        padding=(1, 2),
    )


def create_execution_plan_table(execution_plan: list[str], stages: dict) -> Table:
    """Create a table showing the execution plan.

    Args:
        execution_plan: List of stage IDs in execution order
        stages: Dictionary mapping stage IDs to StageFunction objects

    Returns:
        A rich Table object
    """
    if not RICH_AVAILABLE:
        raise ImportError("Rich library is required for visualization")

    plan_table = Table(
        title="Execution Order",
        box=rich.box.ROUNDED,
        show_header=True,
        header_style="bold magenta",
        padding=(0, 1),
        show_lines=True,
        expand=True,
    )
    plan_table.add_column("Order", style="dim", justify="right", width=5)
    plan_table.add_column("Stage ID", style="cyan", no_wrap=True, min_width=15)
    plan_table.add_column("Description", style="blue")

    for i, stage_to_run in enumerate(execution_plan):
        desc = stages[stage_to_run].description or "[dim]No description[/dim]"
        plan_table.add_row(f"{i + 1}.", stage_to_run, desc)

    return plan_table


def create_summary_table(
    summary_stages: list[str],
    stages: dict,
    executed_stages: set,
    stage_validation_status: dict,
    mode: str,
    active_output_dir,
    shared_data_dir,
) -> Tuple[Table, bool]:
    """Create a summary table for workflow execution.

    Args:
        summary_stages: List of stage IDs to include in the summary
        stages: Dictionary mapping stage IDs to StageFunction objects
        executed_stages: Set of stage IDs that were executed
        stage_validation_status: Dictionary mapping stage IDs to validation status
        mode: Workflow mode ("experiment" or "reproduction")
        active_output_dir: Path to the active output directory
        shared_data_dir: Path to the shared data directory

    Returns:
        A tuple containing:
            - A rich Table object with the summary
            - A boolean indicating whether any stages were skipped
    """
    if not RICH_AVAILABLE:
        raise ImportError("Rich library is required for visualization")

    # Create the summary table
    summary_table = Table(
        title="🏁 Workflow Execution Summary",
        show_header=True,
        header_style="bold magenta",
        box=rich.box.DOUBLE_EDGE,
        padding=(0, 1),
        show_lines=True,
        expand=True,
    )
    summary_table.add_column("Stage", style="cyan", no_wrap=True)
    summary_table.add_column("Status", style="default", justify="center")
    summary_table.add_column("Outputs", style="blue", overflow="fold")
    summary_table.add_column("Dependencies", style="yellow")
    if mode == "reproduction":
        summary_table.add_column("Reproduction", style="default", justify="center")

    # Track if any stage was actually skipped
    any_skipped = False

    for stage_id_summary in summary_stages:
        if stage_id_summary not in stages:
            continue  # Should not happen normally

        stage_func = stages[stage_id_summary]
        final_status_str = "[grey50]Not run[/grey50]"
        repro_status_str = "[dim]N/A[/dim]"
        status_style = "grey50"  # Default style for not run

        # Determine status based on execution and validation results
        validation_status = stage_validation_status.get(stage_id_summary)
        stage_ran_or_skipped = stage_id_summary in executed_stages

        # Set status text and color based on execution and validation
        if stage_ran_or_skipped:
            if validation_status is object():  # SKIPPED sentinel
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
                    # Check shared flag with fallback to heuristic
                    if out.get("shared"):
                        is_shared = True
                    elif path_str and "/" not in path_str and "\\" not in path_str:
                        is_shared = True

                if path_str:
                    # Construct the resolved path string for display
                    if is_shared:
                        # Use the relative path defined in the workflow init
                        resolved_display_path = f"{shared_data_dir.as_posix()}/{path_str} [shared]"
                    else:
                        # Use the relative path defined in the workflow init
                        resolved_display_path = f"{active_output_dir.as_posix()}/{path_str}"

                    display_str = f"{resolved_display_path}"
                    outputs_str_parts.append(display_str)

        outputs_display = ", ".join(outputs_str_parts) if outputs_str_parts else "[dim]None"

        # Get and format dependencies
        deps_list = stage_func.dependencies
        deps_str = ", ".join(deps_list) if deps_list else "[dim]None[/dim]"

        row_data = [
            f"[{status_style}]{stage_id_summary}[/]",  # Apply style to stage ID
            f"[{status_style}]{final_status_str}[/]",
            outputs_display,
            deps_str,
        ]
        if mode == "reproduction":
            repro_style = "default"
            if repro_status_str == "❌ Failed":
                repro_style = "red"
            elif repro_status_str == "✅ Passed":
                repro_style = "green"
            elif repro_status_str == "⚪ Skipped":
                repro_style = "yellow"

            row_data.append(f"[{repro_style}]{repro_status_str}[/]")

        summary_table.add_row(*row_data)

    return summary_table, any_skipped
