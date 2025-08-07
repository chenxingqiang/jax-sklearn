"""Configuration management for JAX acceleration."""

# Authors: The JAX-xlearn developers  
# SPDX-License-Identifier: BSD-3-Clause

import threading
from contextlib import contextmanager
from typing import Any, Dict, Optional, Union

# Global configuration for JAX acceleration
_global_config = {
    "enable_jax": False,
    "jax_platform": "auto",  # auto, cpu, gpu, tpu
    "fallback_on_error": True,
    "memory_limit_gpu": None,  # Auto-detect
    "jit_compilation": True,
    "precision": "float32",  # float32, float64
    "debug_mode": False,
    "cache_compiled_functions": True,
}

_threadlocal = threading.local()

def _get_threadlocal_config() -> Dict[str, Any]:
    """Get a threadlocal **mutable** configuration.
    
    If the configuration does not exist, copy the default global configuration.
    """
    if not hasattr(_threadlocal, 'jax_config'):
        _threadlocal.jax_config = _global_config.copy()
    return _threadlocal.jax_config

def get_config() -> Dict[str, Any]:
    """Retrieve current values for JAX acceleration configuration.

    Returns
    -------
    config : dict
        Keys are parameter names that can be passed to :func:`set_config`.

    Examples
    --------
    >>> import xlearn._jax as jax_xlearn
    >>> config = jax_xlearn.get_config()
    >>> config['enable_jax']  # doctest: +SKIP
    True
    """
    # Return a copy so users can't modify the configuration directly
    return _get_threadlocal_config().copy()

def set_config(
    enable_jax: Optional[bool] = None,
    jax_platform: Optional[str] = None,
    fallback_on_error: Optional[bool] = None,
    memory_limit_gpu: Optional[int] = None,
    jit_compilation: Optional[bool] = None,
    precision: Optional[str] = None,
    debug_mode: Optional[bool] = None,
    cache_compiled_functions: Optional[bool] = None,
) -> None:
    """Set global JAX acceleration configuration.

    Parameters
    ----------
    enable_jax : bool, default=None
        Enable or disable JAX acceleration.
        
    jax_platform : str, default=None
        JAX platform to use ('auto', 'cpu', 'gpu', 'tpu').
        
    fallback_on_error : bool, default=None
        Whether to fallback to CPU implementation on JAX errors.
        
    memory_limit_gpu : int, default=None
        GPU memory limit in MB. None for auto-detection.
        
    jit_compilation : bool, default=None
        Enable JIT compilation of JAX functions.
        
    precision : str, default=None
        Numerical precision ('float32', 'float64').
        
    debug_mode : bool, default=None
        Enable debug mode with additional checks and logging.
        
    cache_compiled_functions : bool, default=None
        Cache compiled JAX functions for reuse.

    Examples
    --------
    >>> import xlearn._jax as jax_xlearn
    >>> jax_xlearn.set_config(enable_jax=True, jax_platform="gpu")
    """
    local_config = _get_threadlocal_config()
    
    if enable_jax is not None:
        local_config["enable_jax"] = enable_jax
    if jax_platform is not None:
        if jax_platform not in ("auto", "cpu", "gpu", "tpu"):
            raise ValueError(
                f"Invalid jax_platform: {jax_platform}. "
                "Must be one of: 'auto', 'cpu', 'gpu', 'tpu'"
            )
        local_config["jax_platform"] = jax_platform
    if fallback_on_error is not None:
        local_config["fallback_on_error"] = fallback_on_error
    if memory_limit_gpu is not None:
        if memory_limit_gpu <= 0:
            raise ValueError("memory_limit_gpu must be positive")
        local_config["memory_limit_gpu"] = memory_limit_gpu
    if jit_compilation is not None:
        local_config["jit_compilation"] = jit_compilation
    if precision is not None:
        if precision not in ("float32", "float64"):
            raise ValueError(
                f"Invalid precision: {precision}. "
                "Must be one of: 'float32', 'float64'"
            )
        local_config["precision"] = precision
    if debug_mode is not None:
        local_config["debug_mode"] = debug_mode
    if cache_compiled_functions is not None:
        local_config["cache_compiled_functions"] = cache_compiled_functions

@contextmanager
def config_context(**kwargs):
    """Context manager for JAX acceleration configuration.

    Parameters
    ----------
    **kwargs
        Configuration parameters to set temporarily.

    Examples
    --------
    >>> import xlearn._jax as jax_xlearn
    >>> with jax_xlearn.config_context(enable_jax=False):
    ...     # JAX acceleration disabled in this block
    ...     pass
    """
    old_config = get_config()
    set_config(**kwargs)
    
    try:
        yield
    finally:
        set_config(**old_config)

def _validate_config() -> None:
    """Validate current configuration."""
    config = get_config()
    
    # Check JAX availability if enabled
    if config["enable_jax"]:
        try:
            import jax
        except ImportError:
            raise ImportError(
                "JAX is not available but enable_jax=True. "
                "Install JAX: pip install jax jaxlib"
            )
    
    # Validate platform
    platform = config["jax_platform"]
    if platform not in ("auto", "cpu", "gpu", "tpu"):
        raise ValueError(f"Invalid jax_platform: {platform}")
    
    # Validate precision
    precision = config["precision"]
    if precision not in ("float32", "float64"):
        raise ValueError(f"Invalid precision: {precision}")
