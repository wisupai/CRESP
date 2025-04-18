#!/usr/bin/env python
# cresp/core/seed.py

"""
CRESP seed management module

This module provides utilities for setting random seeds consistently
across different libraries to ensure reproducibility.
"""

import os
import random
import importlib
import logging
from typing import Dict, List, Optional, Set, Union

# Create logger
logger = logging.getLogger("cresp.seed")

# Dictionary of known libraries that need seed setting and their setup functions
KNOWN_LIBRARIES = {
    "numpy": "set_numpy_seed",
    "torch": "set_torch_seed",
    "tensorflow": "set_tensorflow_seed",
    "jax": "set_jax_seed",
    "random": "set_python_random_seed",
    "python": "set_python_env_seed",
}

def is_library_available(library_name: str) -> bool:
    """Check if a library is available in the current environment.
    
    Args:
        library_name: Name of the library to check
        
    Returns:
        bool: True if the library is available, False otherwise
    """
    try:
        importlib.import_module(library_name)
        return True
    except ImportError:
        return False

def set_python_random_seed(seed: int) -> None:
    """Set Python's random module seed.
    
    Args:
        seed: The random seed to set
    """
    random.seed(seed)
    logger.debug(f"Set Python random seed to {seed}")

def set_python_env_seed(seed: int) -> None:
    """Set Python environment hash seed.
    
    Args:
        seed: The random seed to set
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    logger.debug(f"Set PYTHONHASHSEED to {seed}")

def set_numpy_seed(seed: int) -> None:
    """Set NumPy random seed.
    
    Args:
        seed: The random seed to set
    """
    try:
        import numpy as np
        np.random.seed(seed)
        logger.debug(f"Set NumPy seed to {seed}")
    except ImportError:
        logger.debug("NumPy not available, skipping seed setting")

def set_torch_seed(seed: int) -> None:
    """Set PyTorch random seeds, including CPU, CUDA, and MPS if available.
    
    Args:
        seed: The random seed to set
    """
    try:
        import torch
        torch.manual_seed(seed)
        logger.debug(f"Set PyTorch CPU seed to {seed}")

        # Set CUDA seeds if available
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)  # For multi-GPU

            # Make CUDA deterministic (important for reproducibility)
            # Note: Setting deterministic=True might impact performance
            # and some operations may not have deterministic implementations.
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            logger.debug(f"Set PyTorch CUDA seeds to {seed} and enabled CUDNN deterministic mode")
        
        # Set MPS seeds if available (Apple Silicon GPUs)
        if torch.backends.mps.is_available():
            torch.mps.manual_seed(seed)
            logger.debug(f"Set PyTorch MPS seed to {seed}")

    except ImportError:
        logger.debug("PyTorch not available, skipping seed setting")

def set_tensorflow_seed(seed: int) -> None:
    """Set TensorFlow random seeds.
    
    Args:
        seed: The random seed to set
    """
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
        logger.debug(f"Set TensorFlow seed to {seed}")
    except ImportError:
        logger.debug("TensorFlow not available, skipping seed setting")

def set_jax_seed(seed: int) -> None:
    """Set JAX random seeds.
    
    Args:
        seed: The random seed to set
    """
    try:
        import jax
        import jax.numpy as jnp
        jax.random.PRNGKey(seed)
        logger.debug(f"Set JAX seed to {seed}")
    except ImportError:
        logger.debug("JAX not available, skipping seed setting")

def detect_libraries() -> List[str]:
    """Detect which random number libraries are available in the environment.
    
    Returns:
        List[str]: List of available library names
    """
    available_libs = []
    
    # Check standard libraries
    for lib_name in KNOWN_LIBRARIES.keys():
        if lib_name in ("random", "python"):
            available_libs.append(lib_name)
            continue
            
        if is_library_available(lib_name):
            available_libs.append(lib_name)
    
    return available_libs

def set_seed(seed: int, libraries: Optional[Union[List[str], str]] = None, verbose: bool = False) -> None:
    """Set random seed for all detected libraries or specified libraries.
    
    This function automatically detects available libraries and sets
    their random seeds. It can also be limited to specific libraries.
    
    Args:
        seed: The random seed to set
        libraries: Optional list of library names or single library name
                  If None, automatically detect and set all available libraries
        verbose: Whether to print information about seed setting
    """
    if libraries is None:
        # Automatically detect libraries
        libs_to_set = detect_libraries()
    elif isinstance(libraries, str):
        libs_to_set = [libraries]
    else:
        libs_to_set = libraries
    
    # Always set Python's random and environment
    if "random" not in libs_to_set:
        libs_to_set.append("random")
    if "python" not in libs_to_set:
        libs_to_set.append("python")
    
    # Set seeds for all detected/specified libraries
    for lib_name in libs_to_set:
        if lib_name in KNOWN_LIBRARIES:
            # Get the function name for this library
            func_name = KNOWN_LIBRARIES[lib_name]
            
            # Get the function object
            func = globals().get(func_name)
            
            if func:
                try:
                    func(seed)
                except Exception as e:
                    logger.warning(f"Failed to set seed for {lib_name}: {e}")
        else:
            logger.warning(f"Unknown library: {lib_name}")
    
    if verbose:
        logger.info(f"🎲 Random seeds set to {seed} for: {', '.join(libs_to_set)}")
    return libs_to_set

def get_reproducible_dataloader_kwargs(seed: int) -> Dict:
    """Get kwargs for PyTorch DataLoader to ensure reproducibility.
    
    Args:
        seed: The random seed to set
        
    Returns:
        Dict: Keyword arguments for DataLoader
    """
    try:
        import torch
        generator = torch.Generator()
        generator.manual_seed(seed)
        
        return {
            "generator": generator,
            "num_workers": 0,  # Use single process for determinism
            "drop_last": False,  # Don't drop last batch for reproducibility
            "worker_init_fn": lambda worker_id: set_seed(seed + worker_id)
        }
    except ImportError:
        return {}
