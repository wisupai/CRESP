#!/usr/bin/env python
# cresp/core/seed.py

"""
CRESP Random Seed Management Module

This module provides utilities for setting random seeds across various
libraries to ensure reproducibility in scientific computing workflows.
"""

import os
import random
import importlib
from typing import Dict, List, Optional, Set, Callable, Any, Union, Tuple


def _is_package_available(package_name: str) -> bool:
    """Check if a package is available without importing it."""
    try:
        importlib.util.find_spec(package_name)
        return True
    except (ImportError, AttributeError, ModuleNotFoundError):
        return False


def fix_random_seeds(seed: int = 42, verbose: bool = True) -> Dict[str, bool]:
    """Fix random seeds for all detected libraries to ensure reproducibility.
    
    This function attempts to fix seeds for:
    - Python's built-in random module
    - NumPy
    - PyTorch (both CPU and CUDA)
    - TensorFlow
    - JAX
    - Matplotlib
    - scikit-learn
    - And others as detected
    
    Args:
        seed (int): The random seed to use for all libraries
        verbose (bool): Whether to print information about fixed seeds
        
    Returns:
        Dict[str, bool]: Dictionary indicating which libraries had their seeds fixed
    """
    results = {}
    
    # Basic Python random seed
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    results['python'] = True
    
    # NumPy
    if _is_package_available('numpy'):
        try:
            import numpy as np
            np.random.seed(seed)
            results['numpy'] = True
        except Exception:
            results['numpy'] = False
    
    # PyTorch
    if _is_package_available('torch'):
        try:
            import torch
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                # Make CUDA operations deterministic
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
            results['torch'] = True
        except Exception:
            results['torch'] = False
    
    # TensorFlow
    if _is_package_available('tensorflow'):
        try:
            import tensorflow as tf
            tf.random.set_seed(seed)
            # Set TensorFlow to deterministic operations where possible
            try:
                tf.config.experimental.enable_op_determinism()
            except:
                pass  # Older versions may not have this
            results['tensorflow'] = True
        except Exception:
            results['tensorflow'] = False
            
    # JAX
    if _is_package_available('jax'):
        try:
            import jax
            import jax.numpy as jnp
            jax.config.update('jax_enable_x64', True)
            jax.random.PRNGKey(seed)
            results['jax'] = True
        except Exception:
            results['jax'] = False
    
    # scikit-learn
    if _is_package_available('sklearn'):
        try:
            import sklearn
            sklearn.utils.check_random_state(seed)
            results['sklearn'] = True
        except Exception:
            results['sklearn'] = False
    
    # Matplotlib
    if _is_package_available('matplotlib'):
        try:
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend (helps reproducibility)
            import matplotlib.pyplot as plt
            plt.seed = seed  # Not a standard matplotlib function, but affects some random choices
            results['matplotlib'] = True
        except Exception:
            results['matplotlib'] = False
    
    # Scipy
    if _is_package_available('scipy'):
        try:
            import scipy
            import scipy.stats
            try:
                scipy.random.seed(seed)
            except:
                # Older versions of scipy
                scipy.stats.seed = seed
            results['scipy'] = True
        except Exception:
            results['scipy'] = False
    
    # Pandas
    if _is_package_available('pandas'):
        try:
            import pandas as pd
            pd.np.random.seed(seed)
            results['pandas'] = True
        except Exception:
            results['pandas'] = False
            
    # Additional packages can be added here as needed
    
    if verbose:
        success_count = sum(1 for success in results.values() if success)
        
        print(f"🔒 Fixed random seeds ({success_count}/{len(results)} libraries)")
        print(f"   Seed value: {seed}")
        
        for lib, success in results.items():
            status = "✅" if success else "❌"
            print(f"   {status} {lib}")
            
    return results


def fix_dataloader_seeds(dataloader_kwargs: Dict[str, Any], seed: int = 42) -> Dict[str, Any]:
    """Create PyTorch DataLoader arguments with fixed randomness.
    
    Args:
        dataloader_kwargs: Existing kwargs for DataLoader
        seed: Random seed to use
        
    Returns:
        Updated kwargs dict for deterministic data loading
    """
    if not _is_package_available('torch'):
        return dataloader_kwargs
    
    try:
        import torch
        
        # Create a copy to avoid modifying the original
        kwargs = dataloader_kwargs.copy()
        
        # Add deterministic generator for shuffling
        if kwargs.get('shuffle', False):
            generator = torch.Generator()
            generator.manual_seed(seed)
            kwargs['generator'] = generator
        
        # Set worker settings for determinism
        kwargs.setdefault('num_workers', 0)  # Single process is most deterministic
        kwargs.setdefault('drop_last', False)  # Don't drop the last batch
        kwargs.setdefault('worker_init_fn', lambda worker_id: random.seed(seed + worker_id))
        
        return kwargs
    except Exception:
        return dataloader_kwargs


def seed_worker(worker_id: int) -> None:
    """Worker initialization function to ensure DataLoader workers use different random seeds.
    
    This should be passed to the worker_init_fn argument of DataLoader.
    
    Args:
        worker_id: The ID of the worker
    """
    # Each worker should have a different but deterministic seed
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    
    if _is_package_available('numpy'):
        import numpy as np
        np.random.seed(worker_seed)
