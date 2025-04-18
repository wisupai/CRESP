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

# Import hashing module from the same package or adjacent file
from .hashing import calculate_artifact_hash, validate_artifact
# Import seed management module
from .seed import set_seed, get_reproducible_dataloader_kwargs

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
                 parameters: Optional[Dict[str, Any]] = None,
                 reproduction_mode: str = "strict",
                 tolerance_absolute: Optional[float] = None,
                 tolerance_relative: Optional[float] = None,
                 similarity_threshold: Optional[float] = None,
                 skip_if_unchanged: bool = False):
        """Initialize a stage function
        
        Args:
            func: The function to execute for this stage
            stage_id: Unique identifier for this stage
            description: Optional description of what this stage does
            outputs: List of output artifacts or paths
            dependencies: List of stage IDs that this stage depends on
            parameters: Additional parameters for stage execution
            reproduction_mode: Reproduction mode (strict, standard, tolerant)
            tolerance_absolute: Absolute tolerance for numeric comparisons
            tolerance_relative: Relative tolerance for numeric comparisons
            similarity_threshold: Similarity threshold for content comparison
            skip_if_unchanged: Whether to skip this stage if outputs are unchanged
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
    
    def __call__(self, *args, **kwargs):
        """Call the wrapped function"""
        return self.func(*args, **kwargs)
    
    def to_stage_config(self) -> Dict[str, Any]:
        """Convert to stage configuration dictionary"""
        # Convert string outputs to artifact dictionaries
        artifacts = []
        for output in self.outputs:
            if isinstance(output, str):
                artifacts.append({
                    "path": output,
                    "reproduction": {
                        "mode": self.reproduction_mode,
                        "tolerance_absolute": self.tolerance_absolute,
                        "tolerance_relative": self.tolerance_relative,
                        "similarity_threshold": self.similarity_threshold
                    }
                })
            else:
                # If output is already a dict, ensure it has reproduction config
                output_copy = output.copy()
                if "reproduction" not in output_copy:
                    output_copy["reproduction"] = {
                        "mode": self.reproduction_mode,
                        "tolerance_absolute": self.tolerance_absolute,
                        "tolerance_relative": self.tolerance_relative,
                        "similarity_threshold": self.similarity_threshold
                    }
                artifacts.append(output_copy)
        
        return {
            "id": self.stage_id,
            "description": self.description,
            "dependencies": self.dependencies,
            "outputs": artifacts,
            "code_handler": self.code_handler,
            "parameters": self.parameters
        }


# Define custom exception for reproduction failures
class ReproductionError(Exception):
    pass

class Workflow:
    """Workflow class for managing experiment stages and execution"""
    
    def __init__(self, title: str, authors: List[Dict[str, str]], 
                 description: Optional[str] = None, 
                 config_path: str = "cresp.yaml",
                 seed: Optional[int] = None,
                 use_rich: bool = True,
                 mode: str = "experiment",
                 skip_unchanged: bool = False,
                 reproduction_failure_mode: str = "stop", # New parameter: stop | continue
                 save_reproduction_report: bool = True,   # New parameter
                 reproduction_report_path: str = "reproduction_report.md", # New parameter
                 set_seed_at_init: bool = True, # New parameter to control when to set seeds
                 verbose_seed_setting: bool = True # New parameter to control seed setting output
                 ):
        """Initialize a new workflow
        
        Args:
            title: Workflow title
            authors: List of author information dictionaries
            description: Optional workflow description
            config_path: Path to save the configuration
            seed: Optional random seed for reproducibility
            use_rich: Enable rich visualizations (if available)
            mode: Workflow mode ("experiment" or "reproduction")
            skip_unchanged: If True, skip stage execution if outputs match stored hashes
            reproduction_failure_mode: Behavior on reproduction failure ("stop" or "continue")
            save_reproduction_report: Whether to save a report in reproduction mode
            reproduction_report_path: Path for the reproduction report
            set_seed_at_init: Whether to set random seeds at initialization
            verbose_seed_setting: Whether to print seed setting information
        """
        # --- Assign basic attributes first --- 
        self.title = title
        self.use_rich = use_rich and RICH_AVAILABLE
        self.mode = mode
        self.skip_unchanged = skip_unchanged
        self._stages: Dict[str, StageFunction] = {}
        self._executed_stages: Set[str] = set()
        self._validation_results: List[Dict[str, Any]] = [] # Initialize validation results store
        config_file_path = Path(config_path)
        self.reproduction_failure_mode = reproduction_failure_mode
        self.save_reproduction_report = save_reproduction_report
        self.reproduction_report_path = reproduction_report_path
        self._seed = seed  # Store the seed value
        self._verbose_seed_setting = verbose_seed_setting
        self._seed_initialized = False  # Track if we've already initialized seeds
        # --- End basic attributes assignment --- 

        # Ensure config is loaded *before* accessing its path or data
        try:
            self.config = CrespConfig.load(config_file_path)
            if self.use_rich:
                console.print(f"[dim]Loaded existing configuration from [bold]{self.config.path}[/bold][/dim]")
            # TODO: Maybe update metadata from config if needed?
        except FileNotFoundError:
            if mode == "reproduction":
                 raise FileNotFoundError(f"Configuration file '{config_file_path}' not found, required for reproduction mode.")
            # Create new config only if not found and in experiment mode
            self.config = create_workflow_config(title, authors, description, config_file_path)
            if self.use_rich:
                 console.print(f"[dim]Creating new configuration at [bold]{self.config.path}[/bold][/dim]")
        except Exception as e:
             raise ValueError(f"Error loading or creating configuration '{config_file_path}': {e}")
        
        # Set random seed if provided OR load from config
        current_config_seed = self.config.data.get("reproduction", {}).get("random_seed")
        if seed is not None:
             if current_config_seed is not None and seed != current_config_seed and self.use_rich:
                 console.print(f"[yellow]Warning: Overriding random seed from config ({current_config_seed}) with provided seed ({seed})[/yellow]")
             self.config.set_seed(seed)
             if self.use_rich and self._verbose_seed_setting:
                  console.print(f"[dim]Using random seed: {seed}[/dim]")
             self._seed = seed
        elif current_config_seed is not None:
             if self.use_rich and self._verbose_seed_setting:
                  console.print(f"[dim]Using random seed from config: {current_config_seed}[/dim]")
             self._seed = current_config_seed
        # If neither provided nor in config, no seed is set
        
        # Set random seeds at initialization if requested
        if set_seed_at_init and self._seed is not None:
            self.set_random_seeds(verbose=self._verbose_seed_setting)
            self._seed_initialized = True

    def set_random_seeds(self, verbose: bool = False) -> None:
        """Set random seeds for all detected libraries.
        
        This uses the seed.py module to automatically detect and set seeds
        for all available libraries that use random number generation.
        
        Args:
            verbose: Whether to print seed setting information
        """
        if self._seed is None:
            if self.use_rich and verbose:
                console.print("[yellow]Warning: No random seed set. Skipping seed initialization.[/yellow]")
            return
            
        # Use the seed module to set all seeds
        libraries = set_seed(self._seed, verbose=verbose)
        
        if self.use_rich and verbose:
            console.print(f"[dim]Random seeds set to {self._seed} for: {', '.join(libraries)}[/dim]")

    @property
    def seed(self) -> Optional[int]:
        """Get the current random seed."""
        return self._seed
        
    def get_dataloader_kwargs(self) -> Dict:
        """Get kwargs for PyTorch DataLoader to ensure reproducibility.
        
        Returns:
            Dict: Keyword arguments for DataLoader
        """
        if self._seed is None:
            return {}
            
        return get_reproducible_dataloader_kwargs(self._seed)

    def stage(self, id: Optional[str] = None, 
              description: Optional[str] = None,
              outputs: Optional[List[Union[str, Dict[str, Any]]]] = None,
              dependencies: Optional[List[str]] = None,
              parameters: Optional[Dict[str, Any]] = None,
              reproduction_mode: str = "strict",
              tolerance_absolute: Optional[float] = None,
              tolerance_relative: Optional[float] = None,
              similarity_threshold: Optional[float] = None,
              skip_if_unchanged: Optional[bool] = None) -> Callable[[Callable[..., R]], StageFunction]:
        """Decorator for registering a stage function
        
        Args:
            id: Stage identifier, defaults to function name if not provided
            description: Stage description
            outputs: List of output artifacts
            dependencies: List of stage dependencies
            parameters: Additional stage parameters
            reproduction_mode: Reproduction mode (strict, standard, tolerant)
            tolerance_absolute: Absolute tolerance for numeric comparisons
            tolerance_relative: Relative tolerance for numeric comparisons
            similarity_threshold: Similarity threshold for content comparison
            skip_if_unchanged: Override workflow default for skipping unchanged stages.
                               If None, uses the workflow's default skip_unchanged setting.
            
        Returns:
            Decorated function
        """
        def decorator(func: Callable[..., R]) -> StageFunction:
            # Use function name as ID if not provided
            stage_id = id or func.__name__
            
            # Determine the skip setting for this stage
            # Use stage-specific value if provided, otherwise use workflow default
            final_skip_setting = self.skip_unchanged if skip_if_unchanged is None else skip_if_unchanged
            
            # Create stage function
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
                skip_if_unchanged=final_skip_setting
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
            ValueError: If stage ID already registered *in memory* for this workflow instance
        """
        if stage_func.stage_id in self._stages:
            # This check prevents defining the same stage function twice in the Python script
            raise ValueError(f"Stage ID '{stage_func.stage_id}' already registered for this workflow instance.")
        
        self._stages[stage_func.stage_id] = stage_func
        
        # Check if stage already exists in the loaded config
        if self.config.get_stage(stage_func.stage_id) is None:
            # Only add to config object if it's not already there
            stage_config = stage_func.to_stage_config()
            try:
                self.config.add_stage(stage_config, defer_save=True)
                if self.use_rich:
                     # Optional: Log when a new stage is added to config
                     # console.print(f"[dim]Added new stage '{stage_func.stage_id}' to configuration.[/dim]")
                     pass
            except ValueError as e:
                 # This might happen if add_stage has internal checks beyond get_stage 
                 # (though currently it doesn't seem to). Log warning if it occurs.
                 if self.use_rich:
                      console.print(f"[yellow]Warning: Could not add stage '{stage_func.stage_id}' to config object: {e}[/yellow]")
        # else:
             # Stage already exists in config, no need to add again.
             # if self.use_rich:
                  # Optional: Log that we are using the existing config for this stage
                  # console.print(f"[dim]Stage '{stage_func.stage_id}' already exists in configuration.[/dim]")
             # pass
    
    def _check_outputs_unchanged(self, stage_id: str) -> bool:
        """Check if stage outputs exist and match stored hashes.
        
        Args:
            stage_id: The ID of the stage to check.
            
        Returns:
            True if all outputs exist and match hashes, False otherwise.
        """
        stage_config = self.config.get_stage(stage_id)
        registered_stage = self._stages.get(stage_id)

        if not registered_stage or not stage_config or "outputs" not in stage_config:
            return False 

        declared_outputs = registered_stage.outputs
        if not declared_outputs:
             return True 

        expected_files_config = {}
        for cfg_out in stage_config.get("outputs", []):
            if "path" in cfg_out and "hash" in cfg_out:
                expected_files_config[cfg_out["path"]] = cfg_out
            
        if not expected_files_config:
             return False

        all_match = True
        files_checked_count = 0

        for output_decl in declared_outputs:
            # --- Get default reproduction settings for this stage --- 
            default_repro_mode = registered_stage.reproduction_mode
            default_tol_abs = registered_stage.tolerance_absolute
            default_tol_rel = registered_stage.tolerance_relative
            default_sim_thresh = registered_stage.similarity_threshold
            # --- 
            
            output_specific_repro_config = {}
            if isinstance(output_decl, str):
                path_str = output_decl
            else:
                path_str = output_decl["path"]
                # Get output-specific overrides if they exist
                output_specific_repro_config = output_decl.get("reproduction", {})
            
            # Determine the reproduction settings to use for files matching this declaration
            current_repro_mode = output_specific_repro_config.get("mode", default_repro_mode)
            current_tol_abs = output_specific_repro_config.get("tolerance_absolute", default_tol_abs)
            current_tol_rel = output_specific_repro_config.get("tolerance_relative", default_tol_rel)
            current_sim_thresh = output_specific_repro_config.get("similarity_threshold", default_sim_thresh)
            
            path = Path(path_str)
            
            files_to_validate = {}
            if path_str in expected_files_config: 
                 files_to_validate[path_str] = expected_files_config[path_str]
                 if not Path(path_str).exists():
                     return False
            else: 
                 found_prefix_match = False
                 if not path.exists() or not path.is_dir(): 
                      return False
                 for expected_path, expected_cfg in expected_files_config.items():
                      norm_expected = os.path.normpath(expected_path)
                      norm_declared = os.path.normpath(path_str)
                      if norm_expected.startswith(norm_declared + os.sep):
                           files_to_validate[expected_path] = expected_cfg
                           found_prefix_match = True
                           if not Path(expected_path).exists():
                                return False
                 if not found_prefix_match:
                      return False
            
            if not files_to_validate:
                 return False

            for file_path_str, file_cfg in files_to_validate.items():
                files_checked_count += 1
                # Use file-specific config > output-specific config > stage default
                file_specific_repro_config = file_cfg.get("reproduction", {})
                final_repro_mode = file_specific_repro_config.get("mode", current_repro_mode)
                final_tol_abs = file_specific_repro_config.get("tolerance_absolute", current_tol_abs)
                final_tol_rel = file_specific_repro_config.get("tolerance_relative", current_tol_rel)
                final_sim_thresh = file_specific_repro_config.get("similarity_threshold", current_sim_thresh)
                
                success, _ = validate_artifact(
                    file_path_str, 
                    file_cfg["hash"],
                    validation_type=final_repro_mode,
                    tolerance_absolute=final_tol_abs,
                    tolerance_relative=final_tol_rel,
                    similarity_threshold=final_sim_thresh
                )
                
                if not success:
                    all_match = False
                    break 
            
            if not all_match:
                 break 

        return all_match and files_checked_count > 0

    def _run_stage(self, stage_id: str) -> Tuple[Any, List[Tuple[str, str]], Optional[bool]]:
        """Run a specific stage and return its result, calculated hashes, and validation status.
        Skips execution if the stage's skip_if_unchanged is True and outputs match stored hashes.
        
        Returns:
            Tuple containing:
                - stage execution result (or None if skipped)
                - list of (path, hash) tuples (empty if skipped or not experiment mode)
                - validation status (bool or None if skipped or not reproduction mode)
        """
        stage_validation_passed: Optional[bool] = None # Default to None
        stage_func = self._stages.get(stage_id)
        if not stage_func:
            raise ValueError(f"Stage function for ID '{stage_id}' not found in memory.")

        # --- Set random seeds before stage execution ---
        if self._seed is not None:
            # Only show verbose output if this is the first time we're setting seeds
            verbose = not self._seed_initialized
            self.set_random_seeds(verbose=verbose)
            self._seed_initialized = True
        # ---

        # --- Skip Check --- 
        if stage_func.skip_if_unchanged:
             if self._check_outputs_unchanged(stage_id):
                  if self.use_rich:
                       console.print(f"[green]✓ Skipping stage [bold]{stage_id}[/bold] (outputs unchanged)[/green]")
                  self._executed_stages.add(stage_id) 
                  return None, [], None # Return None result, empty hashes, None validation status
        # --- End Skip Check --- 

        if stage_id in self._executed_stages:
            return None, [], None
        
        calculated_hashes = []
        result = None
        
        # Check and run dependencies
        for dep_id in stage_func.dependencies:
            if dep_id not in self._executed_stages:
                 # We might need to handle failure propagation from dependencies later
                _, _, dep_validation_status = self._run_stage(dep_id)
                # If a dependency failed reproduction and mode is 'stop', execution would have already halted.
        
        # Execute stage function
        if self.use_rich:
            # Temporarily disable the progress display to allow normal print statements to show
            global progress
            progress_active = 'progress' in globals() and progress is not None
            if progress_active:
                try:
                    progress.stop()
                except Exception:
                     progress_active = False # Handle case where progress might be finished
            
            console.print(f"[bold blue]Executing {stage_id}...[/bold blue]")
            start_time = time.time()
            result = stage_func()
            end_time = time.time()
            execution_time = end_time - start_time
            
            # Resume progress display if it was active
            if progress_active:
                try:
                    progress.start()
                except Exception:
                    pass # Ignore errors if progress already finished
                
            console.print(f"[dim]Stage completed in {execution_time:.2f}s[/dim]")
        else:
            result = stage_func()
        
        self._executed_stages.add(stage_id)
        
        # Handle outputs based on mode
        if stage_func.outputs:
            if self.mode == "experiment":
                calculated_hashes = self._update_output_hashes(stage_id, stage_func.outputs)
            elif self.mode == "reproduction":
                # Validate outputs and get stage status
                stage_validation_passed = self._validate_outputs(stage_id)
        
        return result, calculated_hashes, stage_validation_passed

    def _update_output_hashes(self, stage_id: str, outputs: List[Union[str, Dict[str, Any]]]) -> List[Tuple[str, str]]:
        """Calculate and store hashes for stage outputs.
        If an output path is a directory, hashes for all files within it are calculated.
        
        Args:
            stage_id: Stage identifier
            outputs: List of output artifacts declared for the stage
        
        Returns:
            List of (path, hash) tuples for all files hashed.
        """
        hashes_calculated = []
        for output_decl in outputs:
            if isinstance(output_decl, str):
                path_str = output_decl
                hash_method = "sha256"
            else:
                path_str = output_decl["path"]
                hash_method = output_decl.get("hash_method", "sha256")
            
            path = Path(path_str)
            
            try:
                if not path.exists():
                     if self.use_rich:
                        console.print(f"[yellow]Warning: Output path does not exist: {path_str}[/yellow]")
                     continue

                if path.is_file():
                    # Handle single file
                    hash_value = calculate_artifact_hash(path, method=hash_method)
                    self.config.update_hash(stage_id, path_str, hash_value, hash_method)
                    hashes_calculated.append((path_str, hash_value))
                    if self.use_rich:
                        # This message is now redundant as we print summary later
                        # console.print(f"[dim]Hashed file [bold]{path_str}[/bold]: {hash_value[:8]}...[/dim]")
                        pass 
                elif path.is_dir():
                    # Handle directory - hash all files within
                    if self.use_rich:
                         console.print(f"[dim]Hashing contents of directory [bold]{path_str}[/bold]...[/dim]")
                    files_hashed_count = 0
                    for file_path in path.rglob('*'):
                        if file_path.is_file():
                            try:
                                file_hash_value = calculate_artifact_hash(file_path, method=hash_method)
                                # Store path relative to workspace root
                                relative_file_path_str = str(file_path.resolve().relative_to(Path.cwd().resolve()))
                                self.config.update_hash(stage_id, relative_file_path_str, file_hash_value, hash_method)
                                hashes_calculated.append((relative_file_path_str, file_hash_value))
                                files_hashed_count += 1
                            except Exception as file_e:
                                if self.use_rich:
                                    console.print(f"[yellow]  Warning: Failed to hash file {file_path}: {str(file_e)}[/yellow]")
                    if self.use_rich:
                         console.print(f"[dim]Hashed {files_hashed_count} files in [bold]{path_str}[/bold].[/dim]")
                 
            except Exception as e:
                if self.use_rich:
                    console.print(f"[yellow]Warning: Failed to process output {path_str}: {str(e)}[/yellow]")
        return hashes_calculated

    def _validate_outputs(self, stage_id: str) -> bool:
        """Validate stage outputs against stored hashes and record results.
        Returns True if all validated artifacts passed, False otherwise.
        """
        stage_config = self.config.get_stage(stage_id)
        registered_stage = self._stages[stage_id]
        declared_outputs = registered_stage.outputs

        if not stage_config or "outputs" not in stage_config:
            if self.use_rich:
                console.print(f"[yellow]Warning: No reference outputs/hashes found in config for stage {stage_id}[/yellow]")
            return True # No config to validate against, consider it passed?
        if not declared_outputs:
             return True # Nothing declared, passed

        expected_files_config = {}
        for cfg_out in stage_config.get("outputs", []):
            if "path" in cfg_out and "hash" in cfg_out:
                 expected_files_config[cfg_out["path"]] = cfg_out

        stage_validation_passed = True # Track overall stage status

        for output_decl in declared_outputs:
            # --- Get default reproduction settings for this stage --- 
            default_repro_mode = registered_stage.reproduction_mode
            default_tol_abs = registered_stage.tolerance_absolute
            default_tol_rel = registered_stage.tolerance_relative
            default_sim_thresh = registered_stage.similarity_threshold
            # ---
            
            output_specific_repro_config = {}
            if isinstance(output_decl, str):
                path_str = output_decl
            else:
                path_str = output_decl["path"]
                output_specific_repro_config = output_decl.get("reproduction", {})
            
            current_repro_mode = output_specific_repro_config.get("mode", default_repro_mode)
            current_tol_abs = output_specific_repro_config.get("tolerance_absolute", default_tol_abs)
            current_tol_rel = output_specific_repro_config.get("tolerance_relative", default_tol_rel)
            current_sim_thresh = output_specific_repro_config.get("similarity_threshold", default_sim_thresh)

            path = Path(path_str)
            
            files_to_validate = {}
            if path_str in expected_files_config: 
                 files_to_validate[path_str] = expected_files_config[path_str]
            else: 
                 for expected_path, expected_cfg in expected_files_config.items():
                      norm_expected = os.path.normpath(expected_path)
                      norm_declared = os.path.normpath(path_str)
                      if norm_expected.startswith(norm_declared + os.sep):
                           files_to_validate[expected_path] = expected_cfg
            
            if not files_to_validate:
                 if self.use_rich:
                      console.print(f"[yellow]Warning: No reference hashes found in config matching output declaration '{path_str}' for stage {stage_id}[/yellow]")
                 continue

            if self.use_rich:
                console.print(f"[dim]Validating output declaration [bold]{path_str}[/bold] ({len(files_to_validate)} files)...[/dim]")

            for file_path_str, file_cfg in files_to_validate.items():
                if "hash" not in file_cfg:
                    if self.use_rich:
                        console.print(f"[yellow]  Warning: No reference hash found for {file_path_str}[/yellow]")
                    continue
                
                file_specific_repro_config = file_cfg.get("reproduction", {})
                final_repro_mode = file_specific_repro_config.get("mode", current_repro_mode)
                final_tol_abs = file_specific_repro_config.get("tolerance_absolute", current_tol_abs)
                final_tol_rel = file_specific_repro_config.get("tolerance_relative", current_tol_rel)
                final_sim_thresh = file_specific_repro_config.get("similarity_threshold", current_sim_thresh)

                success, message = validate_artifact(
                    file_path_str, 
                    file_cfg["hash"],
                    validation_type=final_repro_mode,
                    tolerance_absolute=final_tol_abs,
                    tolerance_relative=final_tol_rel,
                    similarity_threshold=final_sim_thresh
                )
                
                # --- Record result --- 
                self._validation_results.append({
                    "stage": stage_id,
                    "file": file_path_str,
                    "status": "Passed" if success else "Failed",
                    "mode": final_repro_mode,
                    "message": message
                })
                # --- 
                
                if self.use_rich:
                    if success:
                        console.print(f"[green]  ✓ {file_path_str}: {message}[/green]")
                    else:
                        console.print(f"[red]  ✗ {file_path_str}: {message}[/red]")
                        stage_validation_passed = False # Mark stage as failed
        
        return stage_validation_passed # Return overall status for the stage

    def run(self, stage_id: Optional[str] = None) -> Dict[str, Any]:
        """Run workflow stages"""
        results = {}
        self._validation_results = [] # Clear previous results at the start of a run
        workflow_failed = False # Track overall workflow failure

        if self.use_rich:
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
                    stage_task = progress.add_task(f"[cyan]Stage {idx+1}/{len(execution_plan)}: {curr_stage_id}", total=1)
                    
                    calculated_hashes_for_stage = []
                    stage_validation_passed = None
                    try:
                        stage_result, calculated_hashes_for_stage, stage_validation_passed = self._run_stage(curr_stage_id)
                        results[curr_stage_id] = stage_result
                        
                        # Update progress based on whether it was skipped or ran
                        stage_skipped = stage_result is None and not calculated_hashes_for_stage and stage_validation_passed is None
                        if stage_skipped:
                            progress.update(stage_task, completed=1, description=f"[yellow]✓ {curr_stage_id} skipped (unchanged)")
                        elif stage_validation_passed is False:
                            progress.update(stage_task, completed=1, description=f"[red]✗ {curr_stage_id} failed reproduction")
                            workflow_failed = True # Mark workflow as failed
                        else:
                            progress.update(stage_task, completed=1, description=f"[green]✓ {curr_stage_id} completed")
                        
                        progress.update(overall_task, advance=1)

                        # Print hashes only if it ran (not skipped) and in experiment mode
                        if not stage_skipped and self.mode == "experiment" and calculated_hashes_for_stage:
                             console.print(f"  [dim]Recorded Hashes for {curr_stage_id}:[/dim]")
                             for path_hash, hash_val in calculated_hashes_for_stage:
                                  console.print(f"    [cyan]{path_hash}[/cyan]: {hash_val[:8]}...")
                        
                        # --- Check for reproduction failure --- 
                        if stage_validation_passed is False and self.reproduction_failure_mode == "stop":
                             console.print(f"[bold red]✗ Reproduction failed for stage [bold]{curr_stage_id}[/bold]. Stopping workflow as per configuration.[/bold red]")
                             raise ReproductionError(f"Reproduction failed for stage: {curr_stage_id}")
                             
                    except ReproductionError: # Propagate stop signal
                        raise 
                    except Exception as e:
                        progress.update(stage_task, completed=1, description=f"[red]✗ {curr_stage_id} failed execution")
                        console.print(f"  [red]✗ Stage [bold]{curr_stage_id}[/bold] execution failed: {str(e)}[/red]")
                        workflow_failed = True # Mark workflow as failed
                        if self.reproduction_failure_mode == "stop": # Also stop on execution errors if mode is stop? Assume yes for now.
                            raise # Re-raise the original error
                        # If mode is continue, we just log and proceed
            
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
            
            # --- Save Reproduction Report --- 
            if self.mode == "reproduction" and self.save_reproduction_report and self._validation_results:
                 self._save_reproduction_report()

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
        
        self.config.save()
        
        if workflow_failed and self.reproduction_failure_mode == "continue":
             console.print("[bold yellow]Workflow completed with reproduction failures.[/bold yellow]")

        return results
    
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

    def _save_reproduction_report(self):
        """Generate and save the reproduction report."""
        if not self._validation_results:
            if self.use_rich:
                console.print("[dim]No validation results to report.[/dim]")
            return

        report_path = Path(self.reproduction_report_path)
        if self.use_rich:
            console.print(f"[dim]Generating reproduction report at [bold]{report_path}[/bold]...[/dim]")

        # Group results by stage
        results_by_stage = {}
        for result in self._validation_results:
            stage = result["stage"]
            if stage not in results_by_stage:
                results_by_stage[stage] = []
            results_by_stage[stage].append(result)

        # Build Markdown report
        report_lines = [
            f"# CRESP Reproduction Report",
            f"Workflow: {self.title}",
            f"Config: {self.config.path}",
            f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            ""
        ]

        overall_passed = True
        for stage_id, results in results_by_stage.items():
            stage_passed = all(r["status"] == "Passed" for r in results)
            status_icon = "✅" if stage_passed else "❌"
            report_lines.append(f"## {status_icon} Stage: {stage_id}")
            report_lines.append("| File | Status | Mode | Details |")
            report_lines.append("|------|--------|------|---------|")
            for r in results:
                status_color = "green" if r["status"] == "Passed" else "red"
                report_lines.append(f"| `{r['file']}` | **{r['status']}** | `{r['mode']}` | {r['message']} |")
            report_lines.append("")
            if not stage_passed:
                overall_passed = False

        report_lines.insert(4, f"**Overall Status:** {'✅ Passed' if overall_passed else '❌ Failed'}")
        report_lines.insert(5, "")

        try:
            report_path.parent.mkdir(parents=True, exist_ok=True)
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("\n".join(report_lines))
            if self.use_rich:
                console.print(f"[green]✓ Reproduction report saved successfully.[/green]")
        except Exception as e:
            if self.use_rich:
                console.print(f"[red]✗ Failed to save reproduction report: {e}[/red]")


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