"""Progress tracking utilities for workflow execution."""

from typing import Optional, TypeAlias, Union, Type, cast
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TaskID,
)
from rich.console import Console

# Rich imports for visualization
try:
    RICH_AVAILABLE = True
    ProgressType: Type[Progress] = Progress
except ImportError:
    RICH_AVAILABLE = False
    ProgressType = None  # type: ignore

# Global variable to hold the rich progress instance during run
progress: Optional[Progress] = None


def create_workflow_progress() -> Progress:
    """Create a rich Progress object for workflow execution.

    Returns:
        A rich Progress object
    """
    if not RICH_AVAILABLE:
        raise ImportError("Rich library is required for progress tracking")

    console = Console()

    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
        transient=False,  # Keep progress visible after completion
        refresh_per_second=4,  # Adjust refresh rate
    )


def create_overall_task(prog: Progress, total_stages: int) -> TaskID:
    """Create the overall workflow progress task.

    Args:
        prog: The Progress object
        total_stages: Total number of stages to execute

    Returns:
        Task ID for the overall progress
    """
    if not RICH_AVAILABLE:
        raise ImportError("Rich library is required for progress tracking")

    return prog.add_task(
        f"[green]Workflow Progress ({total_stages} stages)",
        total=total_stages,
    )


def create_stage_task(prog: Progress, stage_id: str, idx: int, total: int) -> TaskID:
    """Create a task for tracking a specific stage.

    Args:
        prog: The Progress object
        stage_id: The ID of the stage
        idx: The index of the stage in the execution plan
        total: The total number of stages

    Returns:
        Task ID for the stage progress
    """
    if not RICH_AVAILABLE:
        raise ImportError("Rich library is required for progress tracking")

    stage_task_description = f"[cyan]Stage {idx + 1}/{total}: {stage_id}"
    return prog.add_task(stage_task_description, total=1, start=False)


def update_stage_progress(prog: Progress, task_id: TaskID, stage_id: str, status: str) -> None:
    """Update the progress of a stage with the given status.

    Args:
        prog: The Progress object
        task_id: The task ID for the stage
        stage_id: The ID of the stage
        status: The status to display ('completed', 'failed', 'skipped', 'failed_reproduction', 'passed_reproduction')
    """
    if not RICH_AVAILABLE:
        raise ImportError("Rich library is required for progress tracking")

    # Define status text and color based on status
    status_text = ""
    if status == "completed":
        status_text = f"[green]✓ {stage_id} completed"
    elif status == "failed":
        status_text = f"[red]✗ {stage_id} failed execution"
    elif status == "skipped":
        status_text = f"[yellow]✓ {stage_id} skipped (unchanged)"
    elif status == "failed_reproduction":
        status_text = f"[red]✗ {stage_id} failed reproduction"
    elif status == "passed_reproduction":
        status_text = f"[green]✓ {stage_id} passed reproduction"
    elif status == "stopped":
        status_text = f"[red]✗ {stage_id} failed reproduction (STOPPED)"

    # Update the progress bar with the appropriate status
    prog.update(task_id, completed=1, description=status_text)
