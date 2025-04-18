# cresp/core/config.py

"""
CRESP config module

This module handles the reading, validation, and updating of the cresp.yaml configuration file.
"""

import os
import shutil
# import inspect # No longer needed?
# import functools # No longer needed?
# import time # No longer needed?
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union # Keep basic types

import yaml
# from pydantic import BaseModel, Field, ValidationError, validator # Moved to models
from pydantic import ValidationError # Still needed for CrespConfig._validate_model
from yaml.parser import ParserError

# Rich imports removed

# Import specific models and constants needed
from .models import CrespConfigModel, Stage # Import base model and Stage for add_stage validation
from .constants import DEFAULT_CONFIG_NAME # Import constant for default name

# hashing and seed imports removed, assumed handled by Workflow


# All Pydantic models (Author, Computing, Environment, etc.) removed here
# ... existing code ...


class CrespConfig:
    """CRESP configuration management class with context manager support"""
    
    # Use imported constant or keep class attribute? Let's keep the attribute for clarity within the class.
    # DEFAULT_CONFIG_NAME = DEFAULT_CONFIG_NAME # Option 1: Use imported
    DEFAULT_CONFIG_NAME = "cresp.yaml" # Option 2: Keep as is
    
    def __init__(self, config_data: Dict[str, Any], path: Optional[Path] = None):
        """Initialize the configuration object
        
        Args:
            config_data: Configuration data dictionary
            path: Configuration file path
        """
        self._data = config_data
        self._path = path
        self._model: Optional[CrespConfigModel] = None # Type hint with imported model
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
            # Validate against the imported Pydantic model
            self._model = CrespConfigModel(**self._data)
        except ValidationError as e:
            # Keep ValidationError import
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
            # Use class attribute directly
            raise FileNotFoundError(f"Configuration file not found: {path or cls.DEFAULT_CONFIG_NAME}")
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f) or {}
                return cls(config_data, config_path)
        except ParserError as e: # Keep ParserError import
            raise ValueError(f"YAML parsing error: {e}")
        except Exception as e:
            raise ValueError(f"Error loading configuration file: {e}")
    
    @classmethod
    def create(cls, metadata: Dict[str, Any], path: Optional[Union[str, Path]] = None) -> 'CrespConfig':
        """Create new configuration with context manager support"""
        # Basic structure required by CrespConfigModel
        config_data = {
            "version": "1.0",
            "metadata": metadata,
            # Assume Environment default is handled by model, provide required fields
            "environment": {"manager": "pixi", "file": "pixi.toml"},
            "stages": []
        }
        # Validate against the model upon creation? Pydantic does this implicitly.
        # CrespConfigModel(**config_data) # Optional explicit validation
        
        config = cls(config_data) # __init__ calls _validate_model
        if path:
            config._path = Path(path)
        else:
            # Use class attribute directly
            config._path = Path(cls.DEFAULT_CONFIG_NAME)
        config._modified = True # Mark as modified since it's newly created
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
        # Use class attribute directly
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
            IOError: If saving fails
        """
        save_path: Path
        if path:
            save_path = Path(path)
        elif self._path:
            save_path = self._path
        else:
            raise ValueError("No save path specified for CrespConfig")
        
        # Create backup
        if save_path.exists():
            try:
                 backup_path = save_path.with_suffix(save_path.suffix + ".bak")
                 shutil.copy2(save_path, backup_path)
            except Exception as e:
                 # Log or print warning about backup failure?
                 print(f"Warning: Could not create backup for {save_path}: {e}")
        
        # Ensure directory exists
        try:
            save_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise IOError(f"Error creating directory for config file {save_path}: {e}")
        
        # Save configuration
        try:
            with open(save_path, 'w', encoding=encoding) as f:
                yaml.dump(self._data, f, 
                         default_flow_style=False, 
                         sort_keys=False,
                         allow_unicode=True,
                         width=88) # Optional: add width for better formatting
            self._path = save_path # Update internal path if saved successfully
            self._modified = False
            return True
        except Exception as e:
            raise IOError(f"Error saving configuration file to {save_path}: {e}")
    
    def get_stage(self, stage_id: str) -> Optional[Dict[str, Any]]:
        """Get configuration dictionary for a specific stage
        
        Args:
            stage_id: Stage identifier
            
        Returns:
            dict: Stage configuration dictionary, None if not found
        """
        # Ensure stages exist and is a list
        stages = self._data.get("stages")
        if not isinstance(stages, list):
             return None 
        # Find the stage by ID
        for stage in stages:
            # Ensure stage is a dict and has an 'id'
            if isinstance(stage, dict) and stage.get("id") == stage_id:
                return stage
        return None
    
    def add_stage(self, stage_data: Dict[str, Any], defer_save: bool = False) -> bool:
        """Add a new stage configuration dictionary.
        
        Args:
            stage_data: Dictionary representing the stage configuration.
            defer_save: If True, do not save the config immediately.
            
        Returns:
            bool: True if the stage was added successfully.
            
        Raises:
            ValueError: If stage ID is missing, already exists, or data is invalid.
        """
        stage_id = stage_data.get("id")
        if not stage_id:
            raise ValueError("Stage data must include an 'id'.")
            
        if self.get_stage(stage_id):
            raise ValueError(f"Stage ID '{stage_id}' already exists in the configuration.")
        
        # Validate the provided stage data against the Pydantic Stage model
        try:
            # Use the imported Stage model for validation
            Stage(**stage_data)
        except ValidationError as e:
            raise ValueError(f"Invalid stage data for ID '{stage_id}': {e}")
        
        # Ensure 'stages' list exists in the main data dictionary
        if "stages" not in self._data or not isinstance(self._data.get("stages"), list):
            self._data["stages"] = []
            
        # Add the validated stage data
        self._data["stages"].append(stage_data)
        self._modified = True
        
        # Re-validate the entire config model after adding the stage
        # This ensures overall consistency (e.g., dependencies) if models had cross-validators
        self._validate_model()
        
        if not defer_save and self._path:
            self.save() # Save immediately if not deferred and path exists
        return True
    
    def update_hash(self, stage_id: str, artifact_path: str, hash_value: str, 
                    hash_method: str = "sha256") -> bool: # Default to sha256
        """Update hash value for a specific artifact within a stage.
           Adds the artifact entry if it does not exist.
        
        Args:
            stage_id: Stage identifier.
            artifact_path: Path of the artifact relative to the project root.
            hash_value: New hash value.
            hash_method: Hash method used (e.g., 'sha256', 'file', 'content').
            
        Returns:
            bool: True if the hash was updated or added successfully.
            
        Raises:
            ValueError: If the specified stage does not exist.
        """
        stage = self.get_stage(stage_id)
        if not stage:
            raise ValueError(f"Cannot update hash: Stage '{stage_id}' not found.")
            
        # Ensure 'outputs' list exists within the stage dictionary
        if "outputs" not in stage or not isinstance(stage.get("outputs"), list):
            stage["outputs"] = []
            
        # Find matching artifact by path
        artifact_found = False
        for output in stage["outputs"]:
            # Ensure output is a dict and has a 'path'
            if isinstance(output, dict) and output.get("path") == artifact_path:
                output["hash"] = hash_value
                output["hash_method"] = hash_method
                # Remove validation section if hash changes? Or keep it? Let's keep it for now.
                self._modified = True
                artifact_found = True
                break # Assume only one artifact per path within a stage
                
        # If no matching artifact found, add a new one
        if not artifact_found:
            stage["outputs"].append({
                "path": artifact_path,
                "hash": hash_value,
                "hash_method": hash_method
                    # Add description? Maybe optionally passed to update_hash?
                    # "description": f"Generated artifact for stage {stage_id}"
            })
            self._modified = True
            
        # Re-validate the entire config model after update? Might be overkill.
        # self._validate_model() 
        
        # Note: This method does not automatically save. Saving is handled by run() or context manager.
        return True
    
    def validate(self) -> Tuple[bool, str]:
        """Validate if the current configuration data is valid according to the model.
        
        Returns:
            Tuple[bool, str]: (Validation success status, Message)
        """
        try:
            self._validate_model() # Use the internal validation method
            return True, "Configuration is valid."
        except ValueError as e: # Catch the specific error raised by _validate_model
            return False, str(e)
        except Exception as e: # Catch any other unexpected errors
            return False, f"An unexpected error occurred during validation: {str(e)}"
    
    @property
    def data(self) -> Dict[str, Any]:
        """Get the raw configuration data dictionary."""
        return self._data
    
    @property
    def is_modified(self) -> bool:
        """Check if configuration has been modified since loading or last save."""
        return self._modified
    
    @property
    def path(self) -> Optional[Path]:
        """Get the configuration file path, if available."""
        return self._path
    
    def set_seed(self, seed: Optional[int]) -> None: # Allow setting seed to None
        """Set the random seed in the configuration's reproduction section.
        
        Args:
            seed: Random seed value (int) or None to remove it.
        """
        # Ensure 'reproduction' dictionary exists
        if "reproduction" not in self._data or not isinstance(self._data.get("reproduction"), dict):
            # Only create if seed is not None, otherwise no need for the section
            if seed is None: 
                # If seed is None and section doesn't exist, do nothing
                if "reproduction" in self._data: 
                     # If section exists but seed is None, remove the key if present
                     if self._data["reproduction"].pop("random_seed", None) is not None:
                          self._modified = True
                return 
            else:
                 self._data["reproduction"] = {} # Create the section
            
        # Update or remove the random_seed
        current_seed = self._data["reproduction"].get("random_seed")
        if seed is None:
            if current_seed is not None:
                 del self._data["reproduction"]["random_seed"]
                 self._modified = True
        elif current_seed != seed:
            self._data["reproduction"]["random_seed"] = seed
            self._modified = True
        # No change if seed is the same or if setting None when it's already None/absent

    def batch_update(self) -> 'ConfigBatchUpdate':
        """Return a context manager for performing batch updates to the config.
           The configuration is saved only once upon exiting the context manager if modified.
        """
        return ConfigBatchUpdate(self)


class ConfigBatchUpdate:
    """Context manager for batch updates to CrespConfig.
       Ensures config is saved only once at the end if modifications occurred.
    """
    def __init__(self, config: CrespConfig):
        self._config = config
        # Store initial modification state? Not strictly necessary with current logic.
        
    def __enter__(self) -> CrespConfig:
        """Enter the context, return the config object."""
        return self._config
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context. Save the config if modified and no exception occurred."""
        # Only save if exiting cleanly and config was modified within the block
        if exc_type is None and self._config.is_modified and self._config.path:
            try:
                 self._config.save()
            except Exception as e:
                 # How to handle save errors within __exit__? Re-raise? Log?
                 # Re-raising might hide the original exception if one occurred in the block.
                 # Let's print a warning for now.
                 print(f"Warning: Error saving configuration on batch update exit: {e}")
                 # Potentially re-raise a specific BatchUpdateSaveError?
                 # raise BatchUpdateSaveError(f"Failed to save config during batch update: {e}") from e