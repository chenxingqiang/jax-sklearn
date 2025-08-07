"""JAX acceleration module for jax-sklearn.

This module provides JAX-accelerated implementations of jax-sklearn algorithms
while maintaining full API compatibility.

Usage:
    import xlearn
    # JAX acceleration is automatically enabled if JAX is available
    
    # Or explicitly configure:
    import xlearn._jax
    xlearn._jax.set_config(enable_jax=True, platform="gpu")
"""

# Authors: The JAX-xlearn developers
# SPDX-License-Identifier: BSD-3-Clause

import os
import warnings
from typing import Optional, Union

# Try to import JAX
try:
    import jax
    import jax.numpy as jnp
    _JAX_AVAILABLE = True
except ImportError:
    jax = None
    jnp = None
    _JAX_AVAILABLE = False

from ._config import get_config, set_config, config_context
from ._accelerator import AcceleratorRegistry

# Global accelerator registry
_registry = AcceleratorRegistry()

def is_jax_available() -> bool:
    """Check if JAX is available."""
    return _JAX_AVAILABLE

def get_jax_platform() -> Optional[str]:
    """Get the current JAX platform."""
    if not _JAX_AVAILABLE:
        return None
    try:
        return jax.default_backend()
    except Exception:
        return None

def enable_jax_acceleration(platform: Optional[str] = None) -> bool:
    """Enable JAX acceleration.
    
    Parameters
    ----------
    platform : str, optional
        JAX platform to use ('cpu', 'gpu', 'tpu'). If None, uses default.
        
    Returns
    -------
    bool
        True if JAX acceleration was successfully enabled.
    """
    if not _JAX_AVAILABLE:
        warnings.warn(
            "JAX is not available. Install JAX to enable acceleration: "
            "pip install jax jaxlib",
            UserWarning
        )
        return False
    
    try:
        if platform:
            # Configure JAX platform if specified
            os.environ['JAX_PLATFORM_NAME'] = platform.lower()
        
        # Update configuration
        set_config(enable_jax=True, jax_platform=platform or "auto")
        return True
        
    except Exception as e:
        warnings.warn(f"Failed to enable JAX acceleration: {e}", UserWarning)
        return False

def disable_jax_acceleration():
    """Disable JAX acceleration."""
    set_config(enable_jax=False)

# Auto-enable JAX if available and not explicitly disabled
if (_JAX_AVAILABLE and 
    os.environ.get('XLEARN_DISABLE_JAX', '').lower() not in ('1', 'true', 'yes')):
    enable_jax_acceleration()

__all__ = [
    'is_jax_available',
    'get_jax_platform', 
    'enable_jax_acceleration',
    'disable_jax_acceleration',
    'get_config',
    'set_config',
    'config_context',
]
