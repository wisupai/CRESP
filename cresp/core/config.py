# cresp/core/config.py

"""
CRESP config module

This module handles the reading, validation, and updating of the cresp.yaml configuration file.
"""

import os
import shutil
import inspect
import functools
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Callable, TypeVar, Set

import yaml
from pydantic import BaseModel, Field, ValidationError, validator
from yaml.parser import ParserError

# Rich imports for visualization
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
    from rich.table import Table
    from rich.style import Style
    from rich import print as rich_print
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

# Create console instance for rich output
if RICH_AVAILABLE:
    console = Console()

class Author(BaseModel):
    """Author information model"""
    name: str
    affiliation: Optional[str] = None
    email: Optional[str] = None
    orcid: Optional[str] = None


class Computing(BaseModel):
    """Computing resource requirements model"""
    cpu: Optional[Dict[str, Any]] = None
    memory: Optional[Dict[str, Any]] = None
    gpu: Optional[Dict[str, Any]] = None
    estimated_runtime: Optional[str] = None
    estimated_storage: Optional[str] = None


class Environment(BaseModel):
    """Environment configuration model"""
    manager: str = "pixi"
    file: str = "pixi.toml"
    python_version: Optional[str] = None


class ValidationRule(BaseModel):
    """Validation rule model"""
    field: Optional[str] = None
    operator: str
    value: Any
    tolerance: Optional[float] = None
    reference: Optional[str] = None
    tolerance_absolute: Optional[float] = None
    tolerance_relative: Optional[float] = None
    similarity_threshold: Optional[float] = None
    method: Optional[str] = None


class ArtifactValidation(BaseModel):
    """Artifact validation model"""
    type: str = "strict"  # strict, weak
    rules: Optional[List[ValidationRule]] = None


class Artifact(BaseModel):
    """Artifact model"""
    path: str
    description: Optional[str] = None
    hash: Optional[str] = None
    hash_method: str = "file"  # file, content, selective
    validation: Optional[ArtifactValidation] = None


class Stage(BaseModel):
    """Experiment stage model"""
    id: str
    # execution_type: str = "pixi"  # Options: "pixi", "code"
    # pixi_task: Optional[str] = None
    description: Optional[str] = None
    dependencies: Optional[List[str]] = None
    outputs: Optional[List[Artifact]] = None
    code_handler: str # Now required as execution_type is always 'code' implicitly
    parameters: Optional[Dict[str, Any]] = None


class ReproductionConfig(BaseModel):
    """Reproduction configuration model"""
    reproducibility_mode: str = "standard"  # strict, standard, tolerant
    random_seed: Optional[int] = None
    comparison_methods: Optional[List[Dict[str, Any]]] = None


class Metadata(BaseModel):
    """Metadata model"""
    title: str
    authors: List[Author]
    description: Optional[str] = None
    keywords: Optional[List[str]] = None
    license: Optional[str] = None
    repository: Optional[str] = None
    created_date: Optional[str] = None


class CrespConfigModel(BaseModel):
    """Complete CRESP configuration model"""
    version: str = "1.0"
    metadata: Metadata
    environment: Environment
    computing: Optional[Computing] = None
    stages: List[Stage]
    reproduction: Optional[ReproductionConfig] = None


class CrespConfig:
    """CRESP configuration management class with context manager support"""
    
    DEFAULT_CONFIG_NAME = "cresp.yaml"
    
    def __init__(self, config_data: Dict[str, Any], path: Optional[Path] = None):
        """Initialize the configuration object
        
        Args:
            config_data: Configuration data dictionary
            path: Configuration file path
        """
        self._data = config_data
        self._path = path
        self._model = None
        self._modified = False
        self._validate_model()
    
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._modified and self._path:
            self.save()
    
    def _validate_model(self) -> None:
        """Validate that the configuration data matches the model specification"""
        try:
            self._model = CrespConfigModel(**self._data)
        except ValidationError as e:
            raise ValueError(f"Configuration validation failed: {e}")
    
    @classmethod
    def load(cls, path: Optional[Union[str, Path]] = None) -> 'CrespConfig':
        """Load configuration from file
        
        Args:
            path: Configuration file path, if None then search in the current directory and parent directories
            
        Returns:
            CrespConfig: Configuration object
            
        Exceptions:
            FileNotFoundError: If the configuration file is not found
            ValueError: If the configuration file format is invalid
        """
        config_path = cls._find_config_file(path)
        if not config_path:
            raise FileNotFoundError(f"Configuration file not found: {path or cls.DEFAULT_CONFIG_NAME}")
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f) or {}
                return cls(config_data, config_path)
        except ParserError as e:
            raise ValueError(f"YAML parsing error: {e}")
        except Exception as e:
            raise ValueError(f"Error loading configuration file: {e}")
    
    @classmethod
    def create(cls, metadata: Dict[str, Any], path: Optional[Union[str, Path]] = None) -> 'CrespConfig':
        """Create new configuration with context manager support"""
        config_data = {
            "version": "1.0",
            "metadata": metadata,
            "environment": {"manager": "pixi", "file": "pixi.toml"},
            "stages": []
        }
        config = cls(config_data)
        if path:
            config._path = Path(path)
        else:
            config._path = Path(cls.DEFAULT_CONFIG_NAME)
        config._modified = True
        return config
    
    @staticmethod
    def _find_config_file(path: Optional[Union[str, Path]] = None) -> Optional[Path]:
        """Find configuration file
        
        Args:
            path: Specified path, if None then search
            
        Returns:
            Path: Found configuration file path, None if not found
        """
        if path:
            p = Path(path)
            return p if p.exists() else None
        
        # Search in current directory and parent directories
        current_dir = Path.cwd()
        config_name = CrespConfig.DEFAULT_CONFIG_NAME
        
        while True:
            config_path = current_dir / config_name
            if config_path.exists():
                return config_path
            
            # Check if we've reached the root directory
            if current_dir.parent == current_dir:
                return None
                
            current_dir = current_dir.parent
    
    def save(self, path: Optional[Union[str, Path]] = None, encoding: str = 'utf-8') -> bool:
        """Save configuration to file
        
        Args:
            path: Save path, if None then use the path from loading
            encoding: File encoding
            
        Returns:
            bool: Success flag
            
        Exceptions:
            ValueError: If no path is specified and no loading path exists
        """
        if path:
            save_path = Path(path)
        elif self._path:
            save_path = self._path
        else:
            raise ValueError("No save path specified")
        
        # Create backup
        if save_path.exists():
            backup_path = save_path.with_suffix(".yaml.bak")
            shutil.copy2(save_path, backup_path)
        
        # Ensure directory exists
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save configuration
        try:
            with open(save_path, 'w', encoding=encoding) as f:
                yaml.dump(self._data, f, 
                         default_flow_style=False, 
                         sort_keys=False,
                         allow_unicode=True)
            self._path = save_path
            self._modified = False
            return True
        except Exception as e:
            raise IOError(f"Error saving configuration file: {e}")
    
    def get_stage(self, stage_id: str) -> Optional[Dict[str, Any]]:
        """Get configuration for a specific stage
        
        Args:
            stage_id: Stage identifier
            
        Returns:
            dict: Stage configuration, None if not exists
        """
        stages = self._data.get("stages", [])
        for stage in stages:
            if stage.get("id") == stage_id:
                return stage
        return None
    
    def add_stage(self, stage_data: Dict[str, Any], defer_save: bool = False) -> bool:
        """Add a new stage with option to defer saving"""
        if not stage_data.get("id"):
            raise ValueError("Stage must have an ID")
            
        stage_id = stage_data["id"]
        if self.get_stage(stage_id):
            raise ValueError(f"Stage ID already exists: {stage_id}")
        
        try:
            Stage(**stage_data)
        except ValidationError as e:
            raise ValueError(f"Invalid stage data: {e}")
        
        if "stages" not in self._data:
            self._data["stages"] = []
            
        self._data["stages"].append(stage_data)
        self._modified = True
        self._validate_model()
        
        if not defer_save and self._path:
            self.save()
        return True
    
    def update_hash(self, stage_id: str, artifact_path: str, hash_value: str, 
                    hash_method: str = "file") -> bool:
        """Update hash value for a specific artifact
        
        Args:
            stage_id: Stage identifier
            artifact_path: Artifact path
            hash_value: New hash value
            hash_method: Hash method
            
        Returns:
            bool: Success flag
            
        Exceptions:
            ValueError: If stage or artifact does not exist
        """
        stage = self.get_stage(stage_id)
        if not stage:
            raise ValueError(f"Stage not found: {stage_id}")
            
        if "outputs" not in stage:
            stage["outputs"] = []
            
        # Find matching artifact
        for output in stage["outputs"]:
            if output.get("path") == artifact_path:
                output["hash"] = hash_value
                output["hash_method"] = hash_method
                self._modified = True
                return True
                
        # If no matching artifact found, add a new one
        stage["outputs"].append({
            "path": artifact_path,
            "hash": hash_value,
            "hash_method": hash_method
        })
        self._modified = True
        return True
    
    def validate(self) -> Tuple[bool, str]:
        """Validate if the configuration is legal
        
        Returns:
            (bool, str): Validation result and error message
        """
        try:
            self._validate_model()
            return True, "Configuration is valid"
        except Exception as e:
            return False, str(e)
    
    @property
    def data(self) -> Dict[str, Any]:
        """Get configuration data"""
        return self._data
    
    @property
    def is_modified(self) -> bool:
        """Check if configuration has been modified"""
        return self._modified
    
    @property
    def path(self) -> Optional[Path]:
        """Get configuration file path"""
        return self._path
    
    def set_seed(self, seed: int) -> None:
        """Set random seed
        
        Args:
            seed: Random seed
        """
        if "reproduction" not in self._data:
            self._data["reproduction"] = {}
            
        self._data["reproduction"]["random_seed"] = seed
        self._modified = True

    def batch_update(self) -> 'ConfigBatchUpdate':
        """Return a context manager for batch updates"""
        return ConfigBatchUpdate(self)


class ConfigBatchUpdate:
    """Context manager for batch updates to config"""
    def __init__(self, config: CrespConfig):
        self.config = config
        
    def __enter__(self):
        return self.config
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.config._modified and self.config._path:
            self.config.save()


# Type variables for function annotations
T = TypeVar('T')
R = TypeVar('R')

class StageFunction:
    """Represents a registered stage function in a workflow"""
    
    def __init__(self, func: Callable, stage_id: str, description: Optional[str] = None, 
                 outputs: Optional[List[Union[str, Dict[str, Any]]]] = None,
                 dependencies: Optional[List[str]] = None,
                 parameters: Optional[Dict[str, Any]] = None):
        """Initialize a stage function
        
        Args:
            func: The function to execute for this stage
            stage_id: Unique identifier for this stage
            description: Optional description of what this stage does
            outputs: List of output artifacts or paths
            dependencies: List of stage IDs that this stage depends on
            parameters: Additional parameters for stage execution
        """
        self.func = func
        self.stage_id = stage_id
        self.description = description or func.__doc__ or f"Stage function {stage_id}"
        self.outputs = outputs or []
        self.dependencies = dependencies or []
        self.parameters = parameters or {}
        self.code_handler = f"{func.__module__}.{func.__qualname__}"
        
        # Preserve function metadata
        functools.update_wrapper(self, func)
    
    def __call__(self, *args, **kwargs):
        """Call the wrapped function"""
        return self.func(*args, **kwargs)
    
    def to_stage_config(self) -> Dict[str, Any]:
        """Convert to stage configuration dictionary"""
        # Convert string outputs to artifact dictionaries
        artifacts = []
        for output in self.outputs:
            if isinstance(output, str):
                artifacts.append({"path": output})
            else:
                artifacts.append(output)
                
        return {
            "id": self.stage_id,
            "description": self.description,
            "dependencies": self.dependencies,
            "outputs": artifacts,
            "code_handler": self.code_handler,
            "parameters": self.parameters
        }


class Workflow:
    """Workflow class for managing experiment stages and execution"""
    
    def __init__(self, title: str, authors: List[Dict[str, str]], 
                 description: Optional[str] = None, 
                 config_path: str = "cresp.yaml",
                 seed: Optional[int] = None,
                 use_rich: bool = True):
        """Initialize a new workflow
        
        Args:
            title: Workflow title
            authors: List of author information dictionaries
            description: Optional workflow description
            config_path: Path to save the configuration
            seed: Optional random seed for reproducibility
            use_rich: Enable rich visualizations (if available)
        """
        self.config = create_workflow_config(title, authors, description, config_path)
        self._stages: Dict[str, StageFunction] = {}
        self._executed_stages: Set[str] = set()
        self.title = title
        self.use_rich = use_rich and RICH_AVAILABLE
        
        # Set random seed if provided
        if seed is not None:
            self.config.set_seed(seed)
    
    def stage(self, id: Optional[str] = None, 
              description: Optional[str] = None,
              outputs: Optional[List[Union[str, Dict[str, Any]]]] = None,
              dependencies: Optional[List[str]] = None,
              parameters: Optional[Dict[str, Any]] = None) -> Callable[[Callable[..., R]], StageFunction]:
        """Decorator for registering a stage function
        
        Args:
            id: Stage identifier, defaults to function name if not provided
            description: Stage description
            outputs: List of output artifacts
            dependencies: List of stage dependencies
            parameters: Additional stage parameters
            
        Returns:
            Decorated function
        """
        def decorator(func: Callable[..., R]) -> StageFunction:
            # Use function name as ID if not provided
            stage_id = id or func.__name__
            
            # Create stage function
            stage_func = StageFunction(
                func=func,
                stage_id=stage_id,
                description=description,
                outputs=outputs,
                dependencies=dependencies,
                parameters=parameters
            )
            
            # Register stage
            self._register_stage(stage_func)
            
            return stage_func
        
        return decorator
    
    def _register_stage(self, stage_func: StageFunction) -> None:
        """Register a stage function
        
        Args:
            stage_func: Stage function to register
            
        Raises:
            ValueError: If stage ID already exists
        """
        if stage_func.stage_id in self._stages:
            raise ValueError(f"Stage ID already registered: {stage_func.stage_id}")
        
        self._stages[stage_func.stage_id] = stage_func
        
        # Add to configuration
        stage_config = stage_func.to_stage_config()
        self.config.add_stage(stage_config, defer_save=True)
    
    def run(self, stage_id: Optional[str] = None) -> Dict[str, Any]:
        """Run workflow stages
        
        Args:
            stage_id: Specific stage to run, or all stages if None
            
        Returns:
            Results from executing stages
            
        Raises:
            ValueError: If specified stage does not exist or if dependencies are not satisfied
        """
        results = {}
        
        if self.use_rich:
            # Print workflow header
            console.print(Panel(f"[bold blue]{self.title}[/bold blue]", 
                               title="CRESP Workflow", 
                               subtitle="Computational Research Environment Standardization Protocol"))
            
            # Create execution plan and display it
            if stage_id:
                execution_plan = [stage_id]
                console.print(f"[yellow]Running single stage: [bold]{stage_id}[/bold][/yellow]")
            else:
                execution_plan = self._resolve_execution_order()
                # Display execution plan
                console.print("[yellow]Execution plan:[/yellow]")
                for i, stage in enumerate(execution_plan):
                    desc = self._stages[stage].description
                    console.print(f"  {i+1}. [bold cyan]{stage}[/bold cyan]: {desc}")
            
            console.print()
            
            # Store the progress object in a global variable to access it in _run_stage
            global progress
            
            # Run stages with progress tracking
            with Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]{task.description}[/bold blue]"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
                console=console,
                transient=False,  # Don't hide progress when complete
                refresh_per_second=5  # Reduce refresh rate to allow other output
            ) as progress:
                overall_task = progress.add_task(f"[green]Running workflow ({len(execution_plan)} stages)", total=len(execution_plan))
                
                for idx, curr_stage_id in enumerate(execution_plan):
                    stage_desc = self._stages[curr_stage_id].description
                    stage_task = progress.add_task(f"[cyan]Stage {idx+1}/{len(execution_plan)}: {curr_stage_id}", total=1)
                    
                    # Run the stage and store result
                    try:
                        start_time = time.time()
                        results[curr_stage_id] = self._run_stage(curr_stage_id)
                        end_time = time.time()
                        
                        # Update progress
                        progress.update(stage_task, completed=1, description=f"[green]✓ {curr_stage_id} completed")
                        progress.update(overall_task, advance=1)
                        
                        # Print stage success
                        stage_time = end_time - start_time
                        console.print(f"  [green]✓ Stage [bold]{curr_stage_id}[/bold] completed in {stage_time:.2f}s[/green]")
                    except Exception as e:
                        progress.update(stage_task, completed=1, description=f"[red]✗ {curr_stage_id} failed")
                        console.print(f"  [red]✗ Stage [bold]{curr_stage_id}[/bold] failed: {str(e)}[/red]")
                        raise
            
            # Clean up global progress variable
            if 'progress' in globals():
                del globals()['progress']
            
            # Print summary table
            console.print()
            table = Table(title="Workflow Execution Summary")
            table.add_column("Stage", style="cyan")
            table.add_column("Status", style="green")
            table.add_column("Outputs", style="blue")
            
            for stage_id in execution_plan:
                stage = self._stages[stage_id]
                status = "[green]Completed" if stage_id in self._executed_stages else "[red]Not run"
                outputs = ", ".join([o["path"] if isinstance(o, dict) else o for o in stage.outputs]) if stage.outputs else "None"
                table.add_row(stage_id, status, outputs)
            
            console.print(table)
        
        else:
            # Non-rich execution path
            if stage_id:
                # Run specific stage
                if stage_id not in self._stages:
                    raise ValueError(f"Stage not found: {stage_id}")
                
                print(f"Running stage: {stage_id}")
                results[stage_id] = self._run_stage(stage_id)
                print(f"Stage {stage_id} completed")
            else:
                # Run all stages in dependency order
                execution_order = self._resolve_execution_order()
                print(f"Running workflow with {len(execution_order)} stages")
                
                for stage_id in execution_order:
                    print(f"Running stage: {stage_id}")
                    results[stage_id] = self._run_stage(stage_id)
                    print(f"Stage {stage_id} completed")
        
        # Save configuration with hashes
        self.config.save()
        
        return results
    
    def _run_stage(self, stage_id: str) -> Any:
        """Run a specific stage
        
        Args:
            stage_id: Stage identifier
            
        Returns:
            Stage execution result
            
        Raises:
            ValueError: If dependencies are not satisfied
        """
        # Check if already executed
        if stage_id in self._executed_stages:
            return None
        
        stage_func = self._stages[stage_id]
        
        # Check and run dependencies
        for dep_id in stage_func.dependencies:
            if dep_id not in self._executed_stages:
                self._run_stage(dep_id)
        
        # Execute stage function with rich visual feedback if enabled
        if self.use_rich:
            # We can't use console.status() here as it would create nested live displays
            # which is not supported by Rich. Instead, we'll just print status messages.
            console.print(f"[bold blue]Executing {stage_id}...[/bold blue]")
            
            # Temporarily disable the progress display to allow normal print statements to show
            if 'progress' in globals():
                # If we're inside a Progress context, need to pause it
                try:
                    progress.stop()
                except:
                    pass
            
            # Execute the function and allow it to print normally
            start_time = time.time()
            result = stage_func()
            end_time = time.time()
            execution_time = end_time - start_time
            
            # Resume progress display if it was active
            if 'progress' in globals():
                try:
                    progress.start()
                except:
                    pass
                
            console.print(f"[dim]Stage completed in {execution_time:.2f}s[/dim]")
        else:
            result = stage_func()
        
        # Mark as executed
        self._executed_stages.add(stage_id)
        
        # TODO: Calculate and update hash values for outputs
        
        return result
    
    def _resolve_execution_order(self) -> List[str]:
        """Resolve execution order based on dependencies
        
        Returns:
            List of stage IDs in execution order
        """
        # Simple topological sort
        order = []
        visited = set()
        temp_visited = set()
        
        def visit(stage_id: str) -> None:
            if stage_id in visited:
                return
            if stage_id in temp_visited:
                raise ValueError(f"Circular dependency detected involving stage: {stage_id}")
            
            temp_visited.add(stage_id)
            
            stage = self._stages[stage_id]
            for dep_id in stage.dependencies:
                if dep_id not in self._stages:
                    raise ValueError(f"Dependency '{dep_id}' of stage '{stage_id}' not found")
                visit(dep_id)
            
            temp_visited.remove(stage_id)
            visited.add(stage_id)
            order.append(stage_id)
        
        for stage_id in self._stages:
            if stage_id not in visited:
                visit(stage_id)
        
        return order
    
    def save_config(self, path: Optional[str] = None) -> bool:
        """Save workflow configuration to file
        
        Args:
            path: Optional path to save to
            
        Returns:
            Success flag
        """
        success = self.config.save(path)
        
        if success and self.use_rich:
            console.print(f"[green]Configuration saved to [bold]{self.config.path}[/bold][/green]")
        
        return success
    
    def visualize(self) -> None:
        """Visualize the workflow structure"""
        if not self.use_rich:
            print("Rich visualization not available. Install 'rich' package for enhanced visuals.")
            return
        
        # Create a table to visualize workflow structure
        table = Table(title=f"Workflow: {self.title}")
        table.add_column("Stage", style="cyan", no_wrap=True)
        table.add_column("Description", style="blue")
        table.add_column("Dependencies", style="yellow")
        table.add_column("Outputs", style="green")
        
        # Get execution order
        try:
            execution_order = self._resolve_execution_order()
            
            for idx, stage_id in enumerate(execution_order):
                stage = self._stages[stage_id]
                deps = ", ".join(stage.dependencies) if stage.dependencies else "None"
                outputs = ", ".join([o["path"] if isinstance(o, dict) else o for o in stage.outputs]) if stage.outputs else "None"
                
                table.add_row(
                    f"{idx+1}. {stage_id}", 
                    stage.description, 
                    deps, 
                    outputs
                )
            
            console.print(table)
            
        except Exception as e:
            console.print(f"[red]Error visualizing workflow: {str(e)}[/red]")


# Helper functions
def create_workflow_config(
    title: str,
    authors: List[Dict[str, str]],
    description: str = None,
    path: str = "workflow.yaml"
) -> CrespConfig:
    """Create a new workflow configuration with better defaults"""
    return CrespConfig.create(
        metadata={
            "title": title,
            "authors": authors,
            "description": description or f"{title} - Reproducible research workflow"
        },
        path=path
    )


def find_config() -> Optional[CrespConfig]:
    """Find and load configuration
    
    Returns:
        CrespConfig: Configuration object, None if not found
    """
    try:
        return CrespConfig.load()
    except FileNotFoundError:
        return None
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return None