"""Accelerator registration system for JAX implementations."""

# Authors: The JAX-xlearn developers
# SPDX-License-Identifier: BSD-3-Clause

import functools
import warnings
from typing import Any, Callable, Dict, Optional, Type, Union

from ._config import get_config


class AcceleratorRegistry:
    """Registry for JAX-accelerated estimator implementations."""
    
    def __init__(self):
        self._accelerators: Dict[Type, Type] = {}
        self._enabled = True
    
    def register(self, original_class: Type, accelerated_class: Type) -> None:
        """Register a JAX-accelerated implementation.
        
        Parameters
        ----------
        original_class : type
            The original jax-sklearn estimator class.
        accelerated_class : type
            The JAX-accelerated implementation class.
        """
        self._accelerators[original_class] = accelerated_class
    
    def get_accelerated(self, original_class: Type) -> Optional[Type]:
        """Get the JAX-accelerated implementation for a class.
        
        Parameters
        ----------
        original_class : type
            The original jax-sklearn estimator class.
            
        Returns
        -------
        accelerated_class : type or None
            The JAX-accelerated implementation, or None if not available.
        """
        if not self._enabled:
            return None
        return self._accelerators.get(original_class)
    
    def is_registered(self, original_class: Type) -> bool:
        """Check if a class has a registered JAX implementation."""
        return original_class in self._accelerators
    
    def enable(self) -> None:
        """Enable the accelerator registry."""
        self._enabled = True
    
    def disable(self) -> None:
        """Disable the accelerator registry."""
        self._enabled = False
    
    def list_accelerated(self) -> Dict[str, str]:
        """List all registered accelerated implementations.
        
        Returns
        -------
        accelerated : dict
            Mapping from original class names to accelerated class names.
        """
        return {
            original_class.__name__: accelerated_class.__name__
            for original_class, accelerated_class in self._accelerators.items()
        }


def accelerated_estimator(original_class: Type):
    """Decorator to register a JAX-accelerated estimator implementation.
    
    Parameters
    ----------
    original_class : type
        The original jax-sklearn estimator class to accelerate.
        
    Examples
    --------
    >>> from xlearn.linear_model import LinearRegression
    >>> @accelerated_estimator(LinearRegression)
    ... class LinearRegressionJAX:
    ...     pass  # JAX implementation
    """
    def decorator(accelerated_class: Type) -> Type:
        # Import registry here to avoid circular imports
        from . import _registry
        _registry.register(original_class, accelerated_class)
        return accelerated_class
    return decorator


def create_accelerated_estimator(original_class: Type, *args, **kwargs):
    """Create an accelerated estimator instance if available.
    
    Parameters
    ----------
    original_class : type
        The original jax-sklearn estimator class.
    *args, **kwargs
        Arguments to pass to the estimator constructor.
        
    Returns
    -------
    estimator
        Either a JAX-accelerated instance or the original instance.
    """
    from . import _registry
    
    config = get_config()
    
    # Check if JAX acceleration is enabled
    if not config["enable_jax"]:
        return original_class(*args, **kwargs)
    
    # Check if JAX is available
    try:
        import jax
    except ImportError:
        if config["fallback_on_error"]:
            return original_class(*args, **kwargs)
        else:
            raise ImportError(
                "JAX is not available but enable_jax=True. "
                "Install JAX: pip install jax jaxlib"
            )
    
    # Get accelerated implementation
    accelerated_class = _registry.get_accelerated(original_class)
    if accelerated_class is None:
        if config["fallback_on_error"]:
            return original_class(*args, **kwargs)
        else:
            warnings.warn(
                f"No JAX acceleration available for {original_class.__name__}. "
                "Using original implementation.",
                UserWarning
            )
            return original_class(*args, **kwargs)
    
    # Try to create accelerated instance
    try:
        return accelerated_class(*args, **kwargs)
    except Exception as e:
        if config["fallback_on_error"]:
            warnings.warn(
                f"Failed to create JAX-accelerated {original_class.__name__}: {e}. "
                "Falling back to original implementation.",
                UserWarning
            )
            return original_class(*args, **kwargs)
        else:
            raise


def jax_accelerate(func: Callable) -> Callable:
    """Decorator to apply JAX acceleration to a function.
    
    Parameters
    ----------
    func : callable
        Function to accelerate with JAX.
        
    Returns
    -------
    accelerated_func : callable
        JAX-accelerated version of the function.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        config = get_config()
        
        if not config["enable_jax"]:
            return func(*args, **kwargs)
        
        try:
            import jax
            import jax.numpy as jnp
            
            # Apply JIT compilation if enabled
            if config["jit_compilation"]:
                jit_func = jax.jit(func)
                return jit_func(*args, **kwargs)
            else:
                return func(*args, **kwargs)
                
        except Exception as e:
            if config["fallback_on_error"]:
                warnings.warn(
                    f"JAX acceleration failed for {func.__name__}: {e}. "
                    "Using original implementation.",
                    UserWarning
                )
                return func(*args, **kwargs)
            else:
                raise
    
    return wrapper
