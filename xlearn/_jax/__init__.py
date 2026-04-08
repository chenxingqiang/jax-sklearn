"""JAX acceleration module for jax-sklearn.

This module provides JAX-accelerated implementations of jax-sklearn algorithms
while maintaining full API compatibility.

Usage:
    import xlearn
    # JAX acceleration is automatically enabled if JAX is available
    
    # Or explicitly configure:
    import xlearn._jax
    xlearn._jax.set_config(enable_jax=True, platform="gpu")

Features:
    - Transparent acceleration of linear models, preprocessing, clustering, and decomposition
    - Automatic fallback to NumPy on errors
    - Device management for CPU, GPU, and TPU
    - Performance monitoring and profiling
    - Gradient computation support
    - Batched processing for large datasets
"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import os
import warnings
from typing import Optional, Union, Dict, Any

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
    """Check if JAX is available.
    
    Returns
    -------
    bool
        True if JAX is installed and importable.
        
    Examples
    --------
    >>> from xlearn._jax import is_jax_available
    >>> if is_jax_available():
    ...     print("JAX acceleration is available!")
    """
    return _JAX_AVAILABLE


def get_jax_platform() -> Optional[str]:
    """Get the current JAX platform/backend.
    
    Returns
    -------
    str or None
        The current JAX backend ('cpu', 'gpu', 'tpu'), or None if JAX unavailable.
        
    Examples
    --------
    >>> from xlearn._jax import get_jax_platform
    >>> platform = get_jax_platform()
    >>> print(f"Running on: {platform}")
    Running on: gpu
    """
    if not _JAX_AVAILABLE:
        return None
    try:
        return jax.default_backend()
    except Exception:
        return None


def get_device_info() -> Dict[str, Any]:
    """Get detailed information about available JAX devices.
    
    Returns
    -------
    dict
        Dictionary containing device information:
        - 'jax_available': bool
        - 'platform': str or None (current default backend)
        - 'devices': dict mapping device type to list of devices
        - 'device_count': dict mapping device type to count
        - 'default_device': device info
        - 'gpu_info': GPU-specific information (if available)
        
    Examples
    --------
    >>> from xlearn._jax import get_device_info
    >>> info = get_device_info()
    >>> print(info['platform'])
    >>> print(f"GPU count: {info['device_count'].get('gpu', 0)}")
    
    Notes
    -----
    JAX supports multiple backends:
    - CPU: Always available
    - GPU: NVIDIA (CUDA) or AMD (ROCm)
    - TPU: Google Cloud TPU
    - Metal: Apple Silicon (M1/M2/M3)
    """
    if not _JAX_AVAILABLE:
        return {
            'jax_available': False,
            'platform': None,
            'devices': {},
            'device_count': {},
            'default_device': None,
            'gpu_info': {}
        }
    
    try:
        from ._universal_jax import get_backend_info
        info = get_backend_info()
        info['jax_available'] = True
        return info
    except Exception as e:
        return {
            'jax_available': True,
            'platform': None,
            'devices': {},
            'device_count': {},
            'default_device': None,
            'gpu_info': {},
            'error': str(e)
        }


def check_gpu_available() -> bool:
    """Check if GPU acceleration is available.
    
    Returns
    -------
    bool
        True if GPU devices are available.
        
    Examples
    --------
    >>> from xlearn._jax import check_gpu_available
    >>> if check_gpu_available():
    ...     print("GPU acceleration available!")
    """
    if not _JAX_AVAILABLE:
        return False
    try:
        from ._universal_jax import check_device_available
        return check_device_available('gpu')
    except Exception:
        return False


def check_tpu_available() -> bool:
    """Check if TPU acceleration is available.
    
    Returns
    -------
    bool
        True if TPU devices are available.
        
    Examples
    --------
    >>> from xlearn._jax import check_tpu_available
    >>> if check_tpu_available():
    ...     print("TPU acceleration available!")
    """
    if not _JAX_AVAILABLE:
        return False
    try:
        from ._universal_jax import check_device_available
        return check_device_available('tpu')
    except Exception:
        return False


def check_metal_available() -> bool:
    """Check if Apple Metal (MPS) acceleration is available.
    
    Metal provides GPU acceleration on Apple Silicon (M1/M2/M3/M4) Macs.
    Requires the jax-metal plugin to be installed.
    
    Returns
    -------
    bool
        True if Metal devices are available.
        
    Examples
    --------
    >>> from xlearn._jax import check_metal_available
    >>> if check_metal_available():
    ...     print("Apple Metal GPU acceleration available!")
    
    Notes
    -----
    To enable Metal support on Apple Silicon:
        pip install jax-metal
    """
    if not _JAX_AVAILABLE:
        return False
    try:
        from ._universal_jax import check_device_available
        return check_device_available('metal')
    except Exception:
        return False


# Alias for PyTorch users familiar with MPS
check_mps_available = check_metal_available


def get_metal_status() -> Dict[str, Any]:
    """Get detailed Apple Metal (MPS) status information.
    
    This function provides comprehensive information about Metal GPU
    acceleration availability, including compatibility issues.
    
    Returns
    -------
    dict
        Dictionary containing:
        - 'installed': bool - jax-metal package installed
        - 'available': bool - Metal devices detected
        - 'working': bool - Metal operations functional
        - 'error': str or None - Error message if not working
        - 'suggestion': str - Suggested action
        
    Examples
    --------
    >>> from xlearn._jax import get_metal_status
    >>> status = get_metal_status()
    >>> if status['working']:
    ...     print("Metal GPU acceleration is working!")
    >>> else:
    ...     print(status['suggestion'])
        
    Notes
    -----
    jax-metal is experimental and may have compatibility issues with
    certain JAX versions. If Metal is installed but not working,
    you can:
    1. Use CPU: Set JAX_PLATFORMS=cpu environment variable
    2. Wait for updated jax-metal release
    3. Try compatible JAX version: pip install jax==0.4.30 jaxlib==0.4.30
    """
    if not _JAX_AVAILABLE:
        return {
            'installed': False,
            'available': False,
            'working': False,
            'error': 'JAX not available',
            'suggestion': 'Install JAX first: pip install jax jaxlib'
        }
    
    try:
        from ._universal_jax import _get_metal_status
        return _get_metal_status()
    except Exception as e:
        return {
            'installed': False,
            'available': False,
            'working': False,
            'error': str(e),
            'suggestion': 'Error checking Metal status'
        }


def print_device_info():
    """Print detailed device information to console.
    
    Useful for debugging and verifying hardware setup.
    
    Examples
    --------
    >>> from xlearn._jax import print_device_info
    >>> print_device_info()
    JAX Device Information
    ======================
    Default backend: gpu
    Available backends: cpu, gpu
    
    Devices:
      cpu:0 - TFRT_CPU_0
      gpu:0 - NVIDIA A100-SXM4-40GB
    """
    if not _JAX_AVAILABLE:
        print("JAX is not available")
        return
    
    try:
        from ._universal_jax import print_device_info as _print_info
        _print_info()
    except Exception as e:
        print(f"Error getting device info: {e}")


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
        
    Examples
    --------
    >>> from xlearn._jax import enable_jax_acceleration
    >>> success = enable_jax_acceleration('gpu')
    >>> if success:
    ...     print("JAX GPU acceleration enabled!")
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
    """Disable JAX acceleration.
    
    After calling this, all operations will use NumPy implementations.
    
    Examples
    --------
    >>> from xlearn._jax import disable_jax_acceleration
    >>> disable_jax_acceleration()
    >>> # All subsequent operations will use NumPy
    """
    set_config(enable_jax=False)


def warmup(X_shape=(1000, 100), algorithms=None):
    """Warmup JAX JIT compilation to avoid overhead in benchmarks.
    
    This function compiles JAX functions with dummy data, ensuring that
    subsequent calls don't include JIT compilation time.
    
    Parameters
    ----------
    X_shape : tuple, default=(1000, 100)
        Shape of dummy data for warmup.
    algorithms : list, optional
        List of algorithms to warmup. If None, warms up common ones:
        ['linear', 'ridge', 'pca', 'kmeans', 'standard_scaler']
        
    Examples
    --------
    >>> from xlearn._jax import warmup
    >>> warmup((1000, 100), ['linear', 'pca'])
    >>> # Now benchmarks won't include JIT compilation time
    """
    if not _JAX_AVAILABLE:
        return
    
    try:
        from ._universal_jax import warmup_jax
        warmup_jax(X_shape, algorithms)
    except Exception:
        pass  # Silently ignore warmup failures


def get_performance_monitor():
    """Get the global performance monitor for profiling.
    
    Returns
    -------
    PerformanceMonitor
        The global performance monitor instance.
        
    Examples
    --------
    >>> from xlearn._jax import get_performance_monitor
    >>> monitor = get_performance_monitor()
    >>> with monitor.track("my_operation"):
    ...     # do something
    ...     pass
    >>> print(monitor.summary())
    """
    if not _JAX_AVAILABLE:
        return None
    
    try:
        from ._universal_jax import get_performance_monitor as _get_monitor
        return _get_monitor()
    except Exception:
        return None


# Auto-enable JAX if available and not explicitly disabled
if (_JAX_AVAILABLE and 
    os.environ.get('XLEARN_DISABLE_JAX', '').lower() not in ('1', 'true', 'yes')):
    enable_jax_acceleration()
    
    # Auto-configure for current hardware (with warnings suppressed by default)
    if os.environ.get('XLEARN_VERBOSE', '').lower() in ('1', 'true', 'yes'):
        try:
            from ._universal_jax import auto_configure
            config = auto_configure()
            if config.get('warnings'):
                for warning in config['warnings']:
                    warnings.warn(warning, UserWarning)
        except Exception:
            pass


def detect_hardware():
    """Detect available hardware and provide installation suggestions.
    
    Returns
    -------
    info : dict
        Dictionary containing hardware info and recommendations.
        
    Examples
    --------
    >>> from xlearn._jax import detect_hardware
    >>> info = detect_hardware()
    >>> for rec in info['recommendations']:
    ...     print(f"{rec['priority']}: {rec['action']}")
    ...     if rec['command']:
    ...         print(f"  Run: {rec['command']}")
    """
    if not _JAX_AVAILABLE:
        return {
            'system': None,
            'hardware': {},
            'jax_status': {'error': 'JAX not available'},
            'recommendations': [{
                'priority': 'high',
                'action': 'Install JAX',
                'command': 'pip install jax jaxlib',
                'note': 'Required for JAX acceleration'
            }]
        }
    
    try:
        from ._universal_jax import detect_hardware as _detect
        return _detect()
    except Exception as e:
        return {'error': str(e)}


def get_installation_command(use_uv: bool = True) -> str:
    """Get the recommended installation command for current hardware.
    
    Parameters
    ----------
    use_uv : bool, default=True
        If True, returns uv command (10-100x faster). If False, returns pip command.
    
    Returns
    -------
    command : str
        Installation command optimized for current hardware.
        
    Examples
    --------
    >>> from xlearn._jax import get_installation_command
    >>> print(get_installation_command())
    uv pip install jax-sklearn[jax-metal]  # On Apple Silicon (with uv)
    
    >>> print(get_installation_command(use_uv=False))
    pip install jax-sklearn[jax-metal]  # On Apple Silicon (with pip)
    """
    if not _JAX_AVAILABLE:
        pkg_cmd = 'uv pip install' if use_uv else 'pip install'
        return f'{pkg_cmd} jax-sklearn[jax]'
    
    try:
        from ._universal_jax import get_installation_command as _get_cmd
        return _get_cmd(use_uv=use_uv)
    except Exception:
        pkg_cmd = 'uv pip install' if use_uv else 'pip install'
        return f'{pkg_cmd} jax-sklearn[jax]'


def get_uv_commands() -> dict:
    """Get all uv installation commands for different platforms.
    
    Returns
    -------
    commands : dict
        Dictionary mapping platform to uv installation command.
        
    Examples
    --------
    >>> from xlearn._jax import get_uv_commands
    >>> commands = get_uv_commands()
    >>> print(commands['apple_silicon'])
    uv pip install jax-sklearn[jax-metal]
    """
    if not _JAX_AVAILABLE:
        return {
            'cpu': 'uv pip install jax-sklearn[jax-cpu]',
            'nvidia_cuda12': 'uv pip install jax-sklearn[jax-gpu]',
            'apple_silicon': 'uv pip install jax-sklearn[jax-metal]',
        }
    
    try:
        from ._universal_jax import get_uv_commands as _get_cmds
        return _get_cmds()
    except Exception:
        return {
            'cpu': 'uv pip install jax-sklearn[jax-cpu]',
            'nvidia_cuda12': 'uv pip install jax-sklearn[jax-gpu]',
            'apple_silicon': 'uv pip install jax-sklearn[jax-metal]',
        }


__all__ = [
    # Availability checks
    'is_jax_available',
    'get_jax_platform',
    'get_device_info',
    'check_gpu_available',
    'check_tpu_available',
    'check_metal_available',
    'check_mps_available',  # Alias for Metal
    'get_metal_status',     # Detailed Metal status
    'print_device_info',
    
    # Hardware detection and installation
    'detect_hardware',
    'get_installation_command',
    'get_uv_commands',
    
    # Enable/disable
    'enable_jax_acceleration',
    'disable_jax_acceleration',
    
    # Configuration
    'get_config',
    'set_config',
    'config_context',
    
    # Performance
    'warmup',
    'get_performance_monitor',
]
