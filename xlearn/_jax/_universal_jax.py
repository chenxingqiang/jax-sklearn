"""Universal JAX implementations for common algorithm patterns."""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy import linalg
from typing import Any, Dict, Optional, Tuple, Union, Callable
from functools import partial

from ._config import get_config
from ._data_conversion import to_jax, to_numpy


# =============================================================================
# Device Management
# =============================================================================

# Supported device/platform types in JAX
SUPPORTED_PLATFORMS = [
    'cpu',      # CPU (always available)
    'gpu',      # NVIDIA GPU (CUDA) or AMD GPU (ROCm)
    'cuda',     # Alias for NVIDIA GPU
    'rocm',     # AMD GPU
    'tpu',      # Google TPU
    'metal',    # Apple Metal (M1/M2/M3/M4 chips) - requires jax-metal plugin
    'mps',      # Alias for Metal Performance Shaders
    'iree',     # IREE compiler backend
]


def _is_jax_metal_installed() -> bool:
    """Check if jax-metal package is installed."""
    try:
        import importlib.util
        # Check for jax_metal module
        if importlib.util.find_spec('jax_metal') is not None:
            return True
        # Check for jax-metal as a JAX plugin
        if importlib.util.find_spec('jax_plugins.metal_plugin') is not None:
            return True
        # Check via pip metadata
        import importlib.metadata
        try:
            importlib.metadata.version('jax-metal')
            return True
        except importlib.metadata.PackageNotFoundError:
            pass
    except Exception:
        pass
    return False


def _check_metal_available() -> bool:
    """Check if JAX Metal plugin is available and working."""
    try:
        # Check if jax-metal is installed
        if not _is_jax_metal_installed():
            return False

        # Check if Metal devices are available
        try:
            metal_devices = jax.devices('METAL')
            if not metal_devices:
                return False
        except (RuntimeError, ValueError):
            return False

        # Test if Metal actually works (compatibility check)
        try:
            # Simple operation test
            test_arr = jnp.array([1.0, 2.0, 3.0])
            _ = test_arr + 1.0
            return True
        except Exception as e:
            # Metal plugin installed but not working (version incompatibility)
            error_msg = str(e).lower()
            if 'unimplemented' in error_msg or 'not supported' in error_msg:
                return False
            return False

    except Exception:
        pass
    return False


def _get_metal_status() -> dict:
    """Get detailed Metal status information.

    Returns
    -------
    status : dict
        Dictionary with:
        - 'installed': bool - jax-metal package installed
        - 'available': bool - Metal devices detected
        - 'working': bool - Metal operations functional
        - 'error': str or None - Error message if not working
        - 'suggestion': str - Suggested action
    """
    status = {
        'installed': False,
        'available': False,
        'working': False,
        'error': None,
        'suggestion': ''
    }

    status['installed'] = _is_jax_metal_installed()

    if not status['installed']:
        status['suggestion'] = 'Install jax-metal: pip install jax-metal'
        return status

    try:
        metal_devices = jax.devices('METAL')
        status['available'] = len(metal_devices) > 0
    except Exception as e:
        status['error'] = str(e)
        status['suggestion'] = 'Metal devices not detected'
        return status

    if not status['available']:
        status['suggestion'] = 'No Metal devices found'
        return status

    # Test functionality
    try:
        test_arr = jnp.array([1.0, 2.0, 3.0])
        _ = test_arr + 1.0
        status['working'] = True
        status['suggestion'] = 'Metal GPU acceleration is working!'
    except Exception as e:
        status['error'] = str(e)
        error_lower = str(e).lower()
        if 'unimplemented' in error_lower or 'not supported' in error_lower:
            status['suggestion'] = (
                'jax-metal version incompatible with current JAX version. '
                'Options:\n'
                '  1. Use CPU: JAX_PLATFORMS=cpu python your_script.py\n'
                '  2. Wait for updated jax-metal release\n'
                '  3. Downgrade JAX: pip install jax==0.4.30 jaxlib==0.4.30'
            )
        else:
            status['suggestion'] = f'Metal error: {e}'

    return status


def get_available_devices() -> Dict[str, list]:
    """Get all available JAX devices.

    Returns
    -------
    devices : dict
        Dictionary mapping device type to list of devices.
        Possible keys: 'cpu', 'gpu', 'tpu', 'metal', etc.

    Examples
    --------
    >>> devices = get_available_devices()
    >>> print(devices)
    {'cpu': [CpuDevice(id=0)], 'gpu': [GpuDevice(id=0), GpuDevice(id=1)]}

    Notes
    -----
    JAX supports multiple backends:
    - CPU: Always available
    - GPU: NVIDIA (CUDA) or AMD (ROCm)
    - TPU: Google Cloud TPU
    - Metal/MPS: Apple Silicon (M1/M2/M3/M4) - requires `pip install jax-metal`

    For Apple Silicon GPU acceleration, install jax-metal:
        pip install jax-metal
    """
    result = {}

    # Standard platforms
    for platform in ['cpu', 'gpu', 'tpu']:
        try:
            devices = jax.devices(platform)
            if devices:
                result[platform] = devices
        except (RuntimeError, ValueError):
            pass

    # Check for Metal (Apple Silicon)
    try:
        # Try to get Metal devices
        metal_devices = jax.devices('METAL')
        if metal_devices:
            result['metal'] = metal_devices
    except (RuntimeError, ValueError):
        pass

    # Alternative: check default backend for Metal
    try:
        default = jax.default_backend()
        if default.lower() == 'metal' and 'metal' not in result:
            result['metal'] = jax.devices()
    except Exception:
        pass

    return result


def get_all_devices() -> list:
    """Get a flat list of all available JAX devices.

    Returns
    -------
    devices : list
        List of all available devices.
    """
    return jax.devices()


def get_default_device():
    """Get the default JAX device.

    Returns
    -------
    device : jax.Device
        The default device used by JAX.

    Examples
    --------
    >>> device = get_default_device()
    >>> print(device.platform)  # 'cpu', 'gpu', 'tpu', etc.
    """
    return jax.devices()[0]


def get_backend_info() -> Dict[str, Any]:
    """Get detailed information about the JAX backend.

    Returns
    -------
    info : dict
        Dictionary containing:
        - 'default_backend': str - Current default backend
        - 'available_backends': list - All available backends
        - 'devices': dict - Devices per platform
        - 'device_count': dict - Device count per platform
        - 'total_devices': int - Total device count
        - 'gpu_info': dict - GPU-specific information (if available)
        - 'memory_info': dict - Memory information per device

    Examples
    --------
    >>> info = get_backend_info()
    >>> print(f"Backend: {info['default_backend']}")
    >>> print(f"GPU count: {info['device_count'].get('gpu', 0)}")
    """
    info = {
        'default_backend': jax.default_backend(),
        'available_backends': [],
        'devices': {},
        'device_count': {},
        'total_devices': 0,
        'gpu_info': {},
        'memory_info': {},
    }

    # Get devices per platform
    devices_by_platform = get_available_devices()
    info['devices'] = {k: [str(d) for d in v] for k, v in devices_by_platform.items()}
    info['device_count'] = {k: len(v) for k, v in devices_by_platform.items()}
    info['total_devices'] = sum(info['device_count'].values())

    # Get available backends
    for platform in ['cpu', 'gpu', 'tpu']:
        try:
            if jax.devices(platform):
                info['available_backends'].append(platform)
        except (RuntimeError, ValueError):
            pass

    # Get GPU-specific info if available
    if 'gpu' in devices_by_platform:
        try:
            gpu_devices = devices_by_platform['gpu']
            info['gpu_info'] = {
                'count': len(gpu_devices),
                'devices': [
                    {
                        'id': i,
                        'name': str(d),
                        'platform': d.platform,
                    }
                    for i, d in enumerate(gpu_devices)
                ]
            }
        except Exception:
            pass

    # Get memory info (requires device memory stats)
    try:
        for platform, devices in devices_by_platform.items():
            for i, device in enumerate(devices):
                try:
                    # Try to get memory stats if available
                    memory_stats = device.memory_stats()
                    if memory_stats:
                        key = f"{platform}:{i}"
                        info['memory_info'][key] = memory_stats
                except (AttributeError, Exception):
                    pass
    except Exception:
        pass

    return info


def select_device(device_type: str = 'auto', device_id: int = 0):
    """Select a specific device for JAX operations.

    Parameters
    ----------
    device_type : str, default='auto'
        Device type. Options:
        - 'auto': Use JAX default (typically best available)
        - 'cpu': CPU device
        - 'gpu' or 'cuda': NVIDIA/AMD GPU
        - 'tpu': Google TPU
        - 'metal' or 'mps': Apple Metal (M1/M2/M3/M4)
    device_id : int, default=0
        Device ID when multiple devices of same type exist.

    Returns
    -------
    device : jax.Device
        The selected device.

    Raises
    ------
    RuntimeError
        If the requested device type is not available.
    ValueError
        If the device_id is out of range.

    Examples
    --------
    >>> # Use default device
    >>> device = select_device('auto')

    >>> # Use first GPU (NVIDIA/AMD)
    >>> device = select_device('gpu', 0)

    >>> # Use Apple Metal GPU
    >>> device = select_device('metal', 0)

    >>> # Use MPS (alias for Metal)
    >>> device = select_device('mps', 0)
    """
    if device_type == 'auto':
        return get_default_device()

    # Handle aliases
    device_type = device_type.lower()
    if device_type == 'cuda':
        device_type = 'gpu'
    elif device_type == 'mps':
        device_type = 'metal'

    devices = get_available_devices()
    available = devices.get(device_type, [])

    if not available:
        available_types = [k for k, v in devices.items() if v]

        # Provide helpful message for Metal
        if device_type == 'metal':
            raise RuntimeError(
                f"No Metal devices available. "
                f"For Apple Silicon GPU, install jax-metal: pip install jax-metal\n"
                f"Available device types: {available_types}"
            )

        raise RuntimeError(
            f"No {device_type} devices available. "
            f"Available device types: {available_types}"
        )

    if device_id >= len(available):
        raise ValueError(
            f"Device {device_type}:{device_id} not found. "
            f"Available IDs for {device_type}: 0-{len(available)-1}"
        )

    return available[device_id]


def put_on_device(data: jnp.ndarray, device=None):
    """Put data on a specific device.

    Parameters
    ----------
    data : jnp.ndarray
        JAX array to move.
    device : jax.Device, optional
        Target device. If None, uses default.

    Returns
    -------
    data : jnp.ndarray
        Data on the target device.

    Examples
    --------
    >>> X_gpu = put_on_device(X, select_device('gpu', 0))
    """
    if device is None:
        return data
    return jax.device_put(data, device)


def get_device_memory(device=None) -> Optional[Dict[str, Any]]:
    """Get memory information for a device.

    Parameters
    ----------
    device : jax.Device, optional
        Device to query. If None, uses default.

    Returns
    -------
    memory_info : dict or None
        Dictionary with memory statistics, or None if not available.
        May include: 'bytes_in_use', 'bytes_limit', 'peak_bytes_in_use', etc.
    """
    if device is None:
        device = get_default_device()

    try:
        return device.memory_stats()
    except (AttributeError, Exception):
        return None


def check_device_available(device_type: str) -> bool:
    """Check if a specific device type is available.

    Parameters
    ----------
    device_type : str
        Device type to check:
        - 'cpu': CPU
        - 'gpu' or 'cuda': NVIDIA/AMD GPU
        - 'tpu': Google TPU
        - 'metal' or 'mps': Apple Metal

    Returns
    -------
    available : bool
        True if the device type is available.

    Examples
    --------
    >>> if check_device_available('gpu'):
    ...     print("GPU acceleration available!")

    >>> if check_device_available('metal'):
    ...     print("Apple Metal acceleration available!")
    """
    device_type = device_type.lower()
    if device_type == 'cuda':
        device_type = 'gpu'
    elif device_type == 'mps':
        device_type = 'metal'

    devices = get_available_devices()
    return len(devices.get(device_type, [])) > 0


def get_best_device() -> Any:
    """Get the best available device for computation.

    Priority: TPU > GPU > Metal > CPU

    Returns
    -------
    device : jax.Device
        The best available device.

    Examples
    --------
    >>> device = get_best_device()
    >>> print(f"Using: {device.platform}")

    Notes
    -----
    Priority order:
    1. TPU (if available)
    2. GPU (NVIDIA/AMD CUDA)
    3. Metal (Apple Silicon)
    4. CPU (fallback)
    """
    devices = get_available_devices()

    # Priority order: TPU > GPU > Metal > CPU
    for platform in ['tpu', 'gpu', 'metal', 'cpu']:
        if platform in devices and devices[platform]:
            return devices[platform][0]

    # Fallback to JAX default
    return get_default_device()


def distribute_across_devices(data: jnp.ndarray, devices: list = None) -> list:
    """Distribute data across multiple devices.

    Parameters
    ----------
    data : jnp.ndarray
        Data to distribute (will be split along first axis).
    devices : list, optional
        List of devices. If None, uses all available GPUs or CPUs.

    Returns
    -------
    shards : list
        List of data shards, one per device.

    Examples
    --------
    >>> # Distribute across all GPUs
    >>> shards = distribute_across_devices(X_large)
    """
    if devices is None:
        all_devices = get_available_devices()
        devices = all_devices.get('gpu', all_devices.get('cpu', [get_default_device()]))

    if len(devices) == 0:
        devices = [get_default_device()]

    n_devices = len(devices)
    n_samples = data.shape[0]

    # Split data
    chunk_size = (n_samples + n_devices - 1) // n_devices
    shards = []

    for i, device in enumerate(devices):
        start = i * chunk_size
        end = min(start + chunk_size, n_samples)
        if start < n_samples:
            shard = jax.device_put(data[start:end], device)
            shards.append(shard)

    return shards


def print_device_info():
    """Print detailed device information to console.

    Useful for debugging and verifying hardware setup.

    Examples
    --------
    >>> print_device_info()
    JAX Device Information
    ======================
    Default backend: gpu
    Available backends: cpu, gpu

    Devices:
      cpu:0 - TFRT_CPU_0
      gpu:0 - NVIDIA A100-SXM4-40GB
      gpu:1 - NVIDIA A100-SXM4-40GB

    For Apple Silicon (Metal):
      metal:0 - Apple M2 Pro
    """
    info = get_backend_info()

    print("JAX Device Information")
    print("=" * 50)
    print(f"Default backend: {info['default_backend']}")
    print(f"Available backends: {', '.join(info['available_backends'])}")
    print(f"Total devices: {info['total_devices']}")
    print()

    print("Devices:")
    for platform, devices in info['devices'].items():
        for i, device in enumerate(devices):
            # Special formatting for Metal
            if platform == 'metal':
                print(f"  {platform}:{i} - {device} (Apple Silicon GPU)")
            else:
                print(f"  {platform}:{i} - {device}")

    # Show platform-specific info
    if info['gpu_info']:
        print()
        print("GPU Information:")
        for gpu in info['gpu_info'].get('devices', []):
            print(f"  GPU {gpu['id']}: {gpu['name']}")

    if info['memory_info']:
        print()
        print("Memory Information:")
        for device_key, mem_stats in info['memory_info'].items():
            if isinstance(mem_stats, dict):
                used = mem_stats.get('bytes_in_use', 0) / (1024**3)
                limit = mem_stats.get('bytes_limit', 0) / (1024**3)
                print(f"  {device_key}: {used:.2f} GB / {limit:.2f} GB")

    # Show installation hints and Metal status
    print()
    import platform as plat
    if plat.system() == 'Darwin' and plat.machine() == 'arm64':
        print("Apple Silicon Status:")
        metal_status = _get_metal_status()
        print(f"  jax-metal installed: {metal_status['installed']}")
        print(f"  Metal devices available: {metal_status['available']}")
        print(f"  Metal working: {metal_status['working']}")
        if metal_status['error']:
            print(f"  Error: {metal_status['error'][:100]}...")
        print()
        print(f"  {metal_status['suggestion']}")
    elif 'gpu' not in info['devices']:
        print("Installation hints:")
        print("  For NVIDIA GPU support:")
        print("    pip install --upgrade jax[cuda12]")
        print("  For AMD GPU support:")
        print("    pip install --upgrade jax[rocm]")
    else:
        print("GPU acceleration is available!")


# =============================================================================
# Hardware Detection and Installation Suggestions
# =============================================================================

def detect_hardware() -> Dict[str, Any]:
    """Detect available hardware and provide installation suggestions.

    Returns
    -------
    info : dict
        Dictionary containing:
        - 'system': str - Operating system
        - 'machine': str - CPU architecture
        - 'hardware': dict - Detected hardware capabilities
        - 'jax_status': dict - Current JAX configuration
        - 'recommendations': list - Installation recommendations

    Examples
    --------
    >>> info = detect_hardware()
    >>> print(info['recommendations'])
    """
    import platform as plat
    import os

    info = {
        'system': plat.system(),
        'machine': plat.machine(),
        'python_version': plat.python_version(),
        'hardware': {
            'is_apple_silicon': False,
            'is_nvidia_available': False,
            'is_amd_available': False,
            'is_tpu_available': False,
            'cpu_count': os.cpu_count() or 1,
        },
        'jax_status': {
            'version': None,
            'backend': None,
            'devices': {},
        },
        'recommendations': [],
    }

    # Detect Apple Silicon
    if plat.system() == 'Darwin' and plat.machine() == 'arm64':
        info['hardware']['is_apple_silicon'] = True

    # Get JAX status
    try:
        info['jax_status']['version'] = jax.__version__
        info['jax_status']['backend'] = jax.default_backend()
        devices = get_available_devices()
        info['jax_status']['devices'] = {k: len(v) for k, v in devices.items()}

        # Check for GPU/TPU
        if 'gpu' in devices:
            info['hardware']['is_nvidia_available'] = True
        if 'tpu' in devices:
            info['hardware']['is_tpu_available'] = True
    except Exception as e:
        info['jax_status']['error'] = str(e)

    # Generate recommendations
    recs = info['recommendations']

    # Apple Silicon recommendations
    if info['hardware']['is_apple_silicon']:
        metal_status = _get_metal_status()
        if not metal_status['installed']:
            recs.append({
                'priority': 'high',
                'action': 'Install jax-metal for Apple Silicon GPU',
                'command': 'pip install jax-sklearn[jax-metal]',
                'note': 'Provides 2-3x speedup for matrix operations'
            })
        elif not metal_status['working']:
            recs.append({
                'priority': 'medium',
                'action': 'Fix jax-metal compatibility',
                'command': 'pip install jax==0.4.35 jaxlib==0.4.35 jax-metal',
                'note': 'Current jax-metal has version compatibility issues'
            })

    # NVIDIA GPU recommendations
    elif info['hardware']['is_nvidia_available']:
        recs.append({
            'priority': 'info',
            'action': 'NVIDIA GPU detected and working',
            'command': None,
            'note': 'No action needed'
        })

    # No GPU detected
    elif not info['hardware']['is_apple_silicon']:
        # Check if this might be a system with NVIDIA GPU but no CUDA
        recs.append({
            'priority': 'medium',
            'action': 'Consider GPU acceleration',
            'command': 'pip install jax-sklearn[jax-gpu]',
            'note': 'For NVIDIA GPU with CUDA 12'
        })

    # CPU-only fallback note
    if info['jax_status'].get('backend', '').lower() == 'cpu':
        recs.append({
            'priority': 'info',
            'action': 'Running on CPU',
            'command': None,
            'note': 'JAX will still provide optimizations via XLA compilation'
        })

    return info


def get_installation_command(use_uv: bool = True) -> str:
    """Get the recommended installation command for current hardware.

    Parameters
    ----------
    use_uv : bool, default=True
        If True, returns uv command (faster). If False, returns pip command.

    Returns
    -------
    command : str
        Installation command optimized for current hardware.

    Examples
    --------
    >>> cmd = get_installation_command()
    >>> print(cmd)
    uv pip install jax-sklearn[jax-metal]  # On Apple Silicon

    >>> cmd = get_installation_command(use_uv=False)
    >>> print(cmd)
    pip install jax-sklearn[jax-metal]
    """
    import platform as plat
    import shutil

    # Determine package manager
    if use_uv:
        # Check if uv is available
        if shutil.which('uv'):
            pkg_cmd = 'uv pip install'
        else:
            pkg_cmd = 'pip install'  # Fallback if uv not installed
    else:
        pkg_cmd = 'pip install'

    # Determine the right extras based on hardware
    # Apple Silicon
    if plat.system() == 'Darwin' and plat.machine() == 'arm64':
        return f'{pkg_cmd} jax-sklearn[jax-metal]'

    # Check for NVIDIA GPU
    try:
        devices = get_available_devices()
        if 'gpu' in devices:
            return f'{pkg_cmd} jax-sklearn[jax-gpu]'
        if 'tpu' in devices:
            return f'{pkg_cmd} jax-sklearn[jax-tpu]'
    except Exception:
        pass

    # Default CPU installation
    return f'{pkg_cmd} jax-sklearn[jax-cpu]'


def get_uv_commands() -> Dict[str, str]:
    """Get all uv installation commands for different platforms.

    Returns
    -------
    commands : dict
        Dictionary mapping platform to uv installation command.

    Examples
    --------
    >>> commands = get_uv_commands()
    >>> print(commands['apple_silicon'])
    uv pip install jax-sklearn[jax-metal]
    """
    return {
        'cpu': 'uv pip install jax-sklearn[jax-cpu]',
        'nvidia_cuda12': 'uv pip install jax-sklearn[jax-gpu]',
        'nvidia_cuda11': 'uv pip install jax-sklearn[jax-cuda11]',
        'amd_rocm': 'uv pip install jax-sklearn[jax-gpu]',
        'google_tpu': 'uv pip install jax-sklearn[jax-tpu]',
        'apple_silicon': 'uv pip install jax-sklearn[jax-metal]',
        'dev': 'uv pip install -e ".[tests,benchmark]"',
    }


def auto_configure() -> Dict[str, Any]:
    """Auto-configure JAX for optimal performance on current hardware.

    This function detects available hardware and configures JAX appropriately.

    Returns
    -------
    config : dict
        Applied configuration settings.

    Examples
    --------
    >>> config = auto_configure()
    >>> print(f"Using backend: {config['backend']}")

    Notes
    -----
    This function:
    1. Detects available hardware (GPU, TPU, Metal, CPU)
    2. Selects the best available backend
    3. Configures memory and precision settings
    4. Handles experimental backends (Metal) with fallback
    """
    import platform as plat
    import os

    config = {
        'backend': 'cpu',
        'device_count': 1,
        'precision': 'float32',
        'memory_fraction': None,
        'warnings': [],
    }

    try:
        devices = get_available_devices()
        backend = jax.default_backend().lower()
        config['backend'] = backend

        # Count devices
        total_devices = sum(len(v) for v in devices.values())
        config['device_count'] = total_devices

        # Handle Metal backend
        if backend == 'metal':
            metal_status = _get_metal_status()
            if not metal_status['working']:
                config['warnings'].append(
                    f"Metal backend detected but not fully functional. "
                    f"Some operations will fall back to CPU. "
                    f"Consider: pip install jax==0.4.35 jaxlib==0.4.35"
                )

        # Set precision based on backend
        if backend == 'metal':
            # Metal doesn't support float64
            config['precision'] = 'float32'
        elif backend in ('gpu', 'tpu'):
            # Mixed precision is often beneficial on GPU/TPU
            config['precision'] = 'float32'

        # Memory configuration for GPU
        if backend == 'gpu':
            # Check if XLA memory fraction is set
            mem_frac = os.environ.get('XLA_PYTHON_CLIENT_MEM_FRACTION')
            if mem_frac:
                config['memory_fraction'] = float(mem_frac)

    except Exception as e:
        config['warnings'].append(f"Auto-configuration error: {e}")

    return config


# =============================================================================
# Performance Monitoring
# =============================================================================

class PerformanceMonitor:
    """Monitor and profile JAX operations.

    Examples
    --------
    >>> monitor = PerformanceMonitor()
    >>> with monitor.track("fit"):
    ...     model.fit(X, y)
    >>> print(monitor.get_stats())
    """

    def __init__(self):
        self._timings = {}
        self._memory = {}
        self._enabled = True

    def enable(self):
        """Enable performance monitoring."""
        self._enabled = True

    def disable(self):
        """Disable performance monitoring."""
        self._enabled = False

    def track(self, operation_name: str):
        """Context manager to track operation timing.

        Parameters
        ----------
        operation_name : str
            Name of the operation to track.
        """
        return _TimingContext(self, operation_name)

    def record_timing(self, operation_name: str, duration: float):
        """Record a timing measurement.

        Parameters
        ----------
        operation_name : str
            Name of the operation.
        duration : float
            Duration in seconds.
        """
        if not self._enabled:
            return
        if operation_name not in self._timings:
            self._timings[operation_name] = []
        self._timings[operation_name].append(duration)

    def get_stats(self) -> Dict[str, Dict[str, float]]:
        """Get performance statistics.

        Returns
        -------
        stats : dict
            Dictionary of operation statistics.
        """
        stats = {}
        for op_name, timings in self._timings.items():
            if timings:
                stats[op_name] = {
                    'count': len(timings),
                    'total': sum(timings),
                    'mean': sum(timings) / len(timings),
                    'min': min(timings),
                    'max': max(timings),
                }
        return stats

    def reset(self):
        """Reset all recorded statistics."""
        self._timings = {}
        self._memory = {}

    def summary(self) -> str:
        """Get a human-readable summary of performance.

        Returns
        -------
        summary : str
            Formatted performance summary.
        """
        stats = self.get_stats()
        if not stats:
            return "No performance data recorded."

        lines = ["Performance Summary:", "-" * 50]
        for op_name, op_stats in stats.items():
            lines.append(f"{op_name}:")
            lines.append(f"  Count: {op_stats['count']}")
            lines.append(f"  Total: {op_stats['total']:.4f}s")
            lines.append(f"  Mean:  {op_stats['mean']:.4f}s")
            lines.append(f"  Min:   {op_stats['min']:.4f}s")
            lines.append(f"  Max:   {op_stats['max']:.4f}s")
        return "\n".join(lines)


class _TimingContext:
    """Context manager for timing operations."""

    def __init__(self, monitor: PerformanceMonitor, operation_name: str):
        self.monitor = monitor
        self.operation_name = operation_name
        self.start_time = None

    def __enter__(self):
        import time
        # Ensure JAX operations are complete before starting timer
        jax.block_until_ready(jnp.array(0.0))
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        import time
        # Ensure JAX operations are complete before stopping timer
        jax.block_until_ready(jnp.array(0.0))
        duration = time.perf_counter() - self.start_time
        self.monitor.record_timing(self.operation_name, duration)
        return False


# Global performance monitor
_global_monitor = PerformanceMonitor()


def get_performance_monitor() -> PerformanceMonitor:
    """Get the global performance monitor."""
    return _global_monitor


# =============================================================================
# Batched Processing
# =============================================================================

def process_in_batches(
    func: Callable,
    X: jnp.ndarray,
    batch_size: int = 10000,
    **kwargs
) -> jnp.ndarray:
    """Process data in batches to manage memory.

    Parameters
    ----------
    func : callable
        Function to apply to each batch.
    X : jnp.ndarray
        Input data.
    batch_size : int, default=10000
        Size of each batch.
    **kwargs
        Additional arguments to pass to func.

    Returns
    -------
    result : jnp.ndarray
        Concatenated results from all batches.
    """
    n_samples = X.shape[0]
    results = []

    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        batch = X[start_idx:end_idx]
        batch_result = func(batch, **kwargs)
        results.append(batch_result)

    return jnp.concatenate(results, axis=0)


def estimate_memory_usage(X: jnp.ndarray, algorithm: str = 'linear') -> Dict[str, float]:
    """Estimate memory usage for JAX operations.

    Parameters
    ----------
    X : jnp.ndarray
        Input data.
    algorithm : str, default='linear'
        Algorithm type.

    Returns
    -------
    memory : dict
        Estimated memory usage in MB.
    """
    n_samples, n_features = X.shape
    dtype_size = X.dtype.itemsize
    base_size = n_samples * n_features * dtype_size / (1024 ** 2)  # MB

    # Algorithm-specific multipliers
    multipliers = {
        'linear': 3.0,  # X, XtX, Xty
        'ridge': 3.0,
        'pca': 4.0,  # X, U, S, Vt
        'kmeans': 2.5,  # X, distances, centers
        'standard_scaler': 2.0,  # X, transformed
    }

    multiplier = multipliers.get(algorithm, 3.0)

    return {
        'input_data_mb': base_size,
        'estimated_peak_mb': base_size * multiplier,
        'algorithm': algorithm,
    }


# =============================================================================
# Gradient Computation
# =============================================================================

def compute_gradient(
    loss_fn: Callable,
    params: Dict[str, jnp.ndarray],
    X: jnp.ndarray,
    y: jnp.ndarray,
    **kwargs
) -> Dict[str, jnp.ndarray]:
    """Compute gradients using JAX autodiff.

    Parameters
    ----------
    loss_fn : callable
        Loss function taking (params, X, y, **kwargs) -> scalar.
    params : dict
        Dictionary of model parameters.
    X : jnp.ndarray
        Input features.
    y : jnp.ndarray
        Target values.
    **kwargs
        Additional arguments for loss_fn.

    Returns
    -------
    gradients : dict
        Dictionary of gradients for each parameter.

    Examples
    --------
    >>> def mse_loss(params, X, y):
    ...     pred = X @ params['coef'] + params['intercept']
    ...     return jnp.mean((pred - y) ** 2)
    >>> grads = compute_gradient(mse_loss, {'coef': w, 'intercept': b}, X, y)
    """
    grad_fn = jax.grad(loss_fn)
    return grad_fn(params, X, y, **kwargs)


def compute_hessian(
    loss_fn: Callable,
    params: jnp.ndarray,
    X: jnp.ndarray,
    y: jnp.ndarray,
) -> jnp.ndarray:
    """Compute Hessian matrix using JAX autodiff.

    Parameters
    ----------
    loss_fn : callable
        Loss function taking (params, X, y) -> scalar.
    params : jnp.ndarray
        Flattened model parameters.
    X : jnp.ndarray
        Input features.
    y : jnp.ndarray
        Target values.

    Returns
    -------
    hessian : jnp.ndarray
        Hessian matrix.
    """
    return jax.hessian(loss_fn)(params, X, y)


def value_and_grad(
    loss_fn: Callable,
    params: Dict[str, jnp.ndarray],
    X: jnp.ndarray,
    y: jnp.ndarray,
    **kwargs
) -> Tuple[float, Dict[str, jnp.ndarray]]:
    """Compute loss value and gradients efficiently.

    Parameters
    ----------
    loss_fn : callable
        Loss function.
    params : dict
        Model parameters.
    X : jnp.ndarray
        Input features.
    y : jnp.ndarray
        Target values.

    Returns
    -------
    loss : float
        Loss value.
    gradients : dict
        Gradients for each parameter.
    """
    val_grad_fn = jax.value_and_grad(loss_fn)
    return val_grad_fn(params, X, y, **kwargs)


# =============================================================================
# Loss Functions
# =============================================================================

@jax.jit
def mse_loss(params: Dict[str, jnp.ndarray], X: jnp.ndarray, y: jnp.ndarray) -> float:
    """Mean Squared Error loss."""
    pred = X @ params['coef'] + params['intercept']
    return jnp.mean((pred - y) ** 2)


@jax.jit
def ridge_loss(
    params: Dict[str, jnp.ndarray],
    X: jnp.ndarray,
    y: jnp.ndarray,
    alpha: float = 1.0
) -> float:
    """Ridge regression loss (MSE + L2 regularization)."""
    pred = X @ params['coef'] + params['intercept']
    mse = jnp.mean((pred - y) ** 2)
    l2_reg = alpha * jnp.sum(params['coef'] ** 2)
    return mse + l2_reg


@jax.jit
def lasso_loss(
    params: Dict[str, jnp.ndarray],
    X: jnp.ndarray,
    y: jnp.ndarray,
    alpha: float = 1.0
) -> float:
    """Lasso regression loss (MSE + L1 regularization)."""
    pred = X @ params['coef'] + params['intercept']
    mse = jnp.mean((pred - y) ** 2)
    l1_reg = alpha * jnp.sum(jnp.abs(params['coef']))
    return mse + l1_reg


@jax.jit
def elastic_net_loss(
    params: Dict[str, jnp.ndarray],
    X: jnp.ndarray,
    y: jnp.ndarray,
    alpha: float = 1.0,
    l1_ratio: float = 0.5
) -> float:
    """Elastic Net loss (MSE + L1 + L2 regularization)."""
    pred = X @ params['coef'] + params['intercept']
    mse = jnp.mean((pred - y) ** 2)
    l1_reg = alpha * l1_ratio * jnp.sum(jnp.abs(params['coef']))
    l2_reg = alpha * (1 - l1_ratio) * 0.5 * jnp.sum(params['coef'] ** 2)
    return mse + l1_reg + l2_reg


@jax.jit
def log_loss(
    params: Dict[str, jnp.ndarray],
    X: jnp.ndarray,
    y: jnp.ndarray,
    C: float = 1.0
) -> float:
    """Logistic regression loss (binary cross-entropy + L2 regularization)."""
    logits = X @ params['coef'] + params['intercept']
    # Numerically stable log-sigmoid
    log_probs = jax.nn.log_sigmoid(logits)
    log_1_minus_probs = jax.nn.log_sigmoid(-logits)
    bce = -jnp.mean(y * log_probs + (1 - y) * log_1_minus_probs)
    l2_reg = 0.5 / C * jnp.sum(params['coef'] ** 2)
    return bce + l2_reg


# =============================================================================
# Universal JAX Mixin
# =============================================================================

class UniversalJAXMixin:
    """Mixin class that provides universal JAX acceleration for common operations."""

    def __init__(self):
        self._jax_compiled_functions = {}
        self._performance_cache = {}
        self._device = None
        self._monitor = None

    def set_device(self, device_type: str = 'auto', device_id: int = 0):
        """Set the device for JAX operations.

        Parameters
        ----------
        device_type : str, default='auto'
            Device type: 'auto', 'cpu', 'gpu', or 'tpu'.
        device_id : int, default=0
            Device ID when multiple devices exist.
        """
        self._device = select_device(device_type, device_id)
        return self

    def enable_profiling(self):
        """Enable performance profiling for this estimator."""
        self._monitor = PerformanceMonitor()
        return self

    def get_profiling_stats(self) -> Dict[str, Dict[str, float]]:
        """Get profiling statistics."""
        if self._monitor is None:
            return {}
        return self._monitor.get_stats()

    def _should_use_jax(self, X: np.ndarray, algorithm_name: str = None) -> bool:
        """Determine if JAX should be used based on configuration.

        By default, JAX is always used when enabled. The threshold-based
        heuristic can be enabled via config if needed for specific use cases.
        """
        config = get_config()
        if not config.get("enable_jax", True):
            return False

        # If auto_threshold is disabled (default), always use JAX
        if not config.get("jax_auto_threshold", False):
            return True

        # Otherwise, use heuristic based on data size
        n_samples, n_features = X.shape

        # Cache key for performance decision
        cache_key = (n_samples, n_features, algorithm_name or self.__class__.__name__)
        if cache_key in self._performance_cache:
            return self._performance_cache[cache_key]

        # Heuristic decision based on data size and algorithm type
        decision = self._performance_heuristic(n_samples, n_features, algorithm_name)
        self._performance_cache[cache_key] = decision

        return decision

    def _performance_heuristic(self, n_samples: int, n_features: int, algorithm_name: str = None) -> bool:
        """Heuristic to decide whether to use JAX based on problem characteristics."""
        complexity = n_samples * n_features

        # Algorithm-specific thresholds based on our testing
        thresholds = {
            # Linear models - benefit from JAX on large problems
            'LinearRegression': {'min_complexity': 1e8, 'min_samples': 10000},
            'linear': {'min_complexity': 1e8, 'min_samples': 10000},  # alias for jax_fit
            'Ridge': {'min_complexity': 1e8, 'min_samples': 10000},
            'ridge': {'min_complexity': 1e8, 'min_samples': 10000},  # alias for jax_fit
            'Lasso': {'min_complexity': 5e7, 'min_samples': 5000},  # Iterative, benefits earlier
            'lasso': {'min_complexity': 5e7, 'min_samples': 5000},
            'ElasticNet': {'min_complexity': 5e7, 'min_samples': 5000},
            'elastic_net': {'min_complexity': 5e7, 'min_samples': 5000},
            'LogisticRegression': {'min_complexity': 5e7, 'min_samples': 5000},
            'logistic': {'min_complexity': 5e7, 'min_samples': 5000},

            # Preprocessing - always benefits from vectorization
            'StandardScaler': {'min_complexity': 1e6, 'min_samples': 1000},
            'standard_scaler': {'min_complexity': 1e6, 'min_samples': 1000},
            'MinMaxScaler': {'min_complexity': 1e6, 'min_samples': 1000},
            'Normalizer': {'min_complexity': 1e6, 'min_samples': 1000},

            # Clustering - benefit from vectorization
            'KMeans': {'min_complexity': 1e6, 'min_samples': 5000},
            'kmeans': {'min_complexity': 1e6, 'min_samples': 5000},  # alias
            'DBSCAN': {'min_complexity': 1e6, 'min_samples': 1000},
            'MiniBatchKMeans': {'min_complexity': 1e5, 'min_samples': 1000},

            # Decomposition - matrix operations benefit greatly
            'PCA': {'min_complexity': 1e7, 'min_samples': 5000},
            'pca': {'min_complexity': 1e7, 'min_samples': 5000},  # alias
            'TruncatedSVD': {'min_complexity': 1e7, 'min_samples': 5000},
            'NMF': {'min_complexity': 5e6, 'min_samples': 2000},

            # Tree-based - limited JAX benefit but some operations can be accelerated
            'RandomForestClassifier': {'min_complexity': 1e5, 'min_samples': 1000},
            'RandomForestRegressor': {'min_complexity': 1e5, 'min_samples': 1000},

            # Default for unknown algorithms
            'default': {'min_complexity': 1e7, 'min_samples': 10000}
        }

        # Get threshold for this algorithm
        algo_name = algorithm_name or self.__class__.__name__
        threshold = thresholds.get(algo_name, thresholds['default'])

        return (complexity >= threshold['min_complexity'] and
                n_samples >= threshold['min_samples'])

    @staticmethod
    @jax.jit
    def _jax_solve_linear_system(A: jnp.ndarray, b: jnp.ndarray, regularization: float = 1e-10) -> jnp.ndarray:
        """JAX-compiled function to solve linear system Ax = b."""
        # Add regularization for numerical stability
        if A.ndim == 2 and A.shape[0] == A.shape[1]:
            A_reg = A + regularization * jnp.eye(A.shape[0])
        else:
            # For overdetermined systems (A^T A x = A^T b)
            AtA = A.T @ A
            A_reg = AtA + regularization * jnp.eye(AtA.shape[0])
            b = A.T @ b

        return linalg.solve(A_reg, b)

    @staticmethod
    def _jax_linear_regression_fit(X: jnp.ndarray, y: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """JAX linear regression fitting.

        Uses pseudoinverse for Metal compatibility (linalg.solve not supported on Metal).
        """
        n_samples, n_features = X.shape

        # Center the data
        X_mean = jnp.mean(X, axis=0)
        y_mean = jnp.mean(y)
        X_centered = X - X_mean
        y_centered = y - y_mean

        # Use pseudoinverse instead of solve for Metal compatibility
        # coef = (X^T X)^-1 X^T y = X^+ y
        # This is more robust and works on Metal
        try:
            # Try to use solve first (faster on CUDA/TPU)
            XtX = X_centered.T @ X_centered
            Xty = X_centered.T @ y_centered
            regularization = 1e-10 * jnp.trace(XtX) / n_features
            coef = linalg.solve(XtX + regularization * jnp.eye(n_features), Xty)
        except Exception:
            # Fallback to pseudoinverse for Metal
            coef = jnp.linalg.lstsq(X_centered, y_centered, rcond=None)[0]

        # Calculate intercept
        intercept = y_mean - X_mean @ coef

        return coef, intercept

    @staticmethod
    def _jax_linear_regression_fit_gd(
        X: jnp.ndarray,
        y: jnp.ndarray,
        max_iter: int = 1000,
        lr: float = 0.1,
        tol: float = 1e-6
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """JAX linear regression using gradient descent (Metal-compatible).

        This version uses only matrix multiplication and basic arithmetic,
        which are fully supported on Apple Metal.

        Parameters
        ----------
        X : jnp.ndarray
            Feature matrix of shape (n_samples, n_features)
        y : jnp.ndarray
            Target vector of shape (n_samples,)
        max_iter : int
            Maximum number of iterations
        lr : float
            Learning rate (will be automatically scaled)
        tol : float
            Convergence tolerance
        """
        n_samples, n_features = X.shape

        # Center the data
        X_mean = jnp.mean(X, axis=0)
        y_mean = jnp.mean(y)
        X_centered = X - X_mean
        y_centered = y - y_mean

        # Normalize X for better convergence
        X_std = jnp.std(X_centered, axis=0)
        X_std = jnp.where(X_std == 0, 1.0, X_std)
        X_norm = X_centered / X_std

        # Initialize coefficients
        coef = jnp.zeros(n_features)

        # Precompute XtX and Xty for efficiency
        XtX = X_norm.T @ X_norm / n_samples
        Xty = X_norm.T @ y_centered / n_samples

        # Adaptive learning rate based on data scale
        # Use a simple estimate of the largest eigenvalue
        max_eigval_approx = jnp.max(jnp.sum(jnp.abs(XtX), axis=1))
        lr_adapted = lr / (max_eigval_approx + 1e-8)

        # Gradient descent
        for i in range(max_iter):
            # Gradient: 2 * (XtX @ coef - Xty)
            grad = 2 * (XtX @ coef - Xty)

            # Update
            coef_new = coef - lr_adapted * grad

            # Check convergence
            diff = jnp.max(jnp.abs(coef_new - coef))
            coef = coef_new

            if diff < tol:
                break

        # Rescale coefficients back
        coef = coef / X_std

        # Calculate intercept
        intercept = y_mean - X_mean @ coef

        return coef, intercept

    @staticmethod
    def _jax_linear_regression_fit_cg(
        X: jnp.ndarray,
        y: jnp.ndarray,
        max_iter: int = 100,
        tol: float = 1e-8
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """JAX linear regression using Conjugate Gradient (Metal-compatible).

        CG is more efficient than gradient descent for linear systems.
        Uses only matrix-vector products which work on Metal.
        """
        n_samples, n_features = X.shape

        # Center the data
        X_mean = jnp.mean(X, axis=0)
        y_mean = jnp.mean(y)
        X_centered = X - X_mean
        y_centered = y - y_mean

        # We want to solve: (X^T X + λI) coef = X^T y
        # Using CG with A = X^T X + λI, b = X^T y
        regularization = 1e-8 * n_samples
        Xty = X_centered.T @ y_centered

        # Function to compute A @ x = (X^T X + λI) @ x
        def matvec(v):
            return X_centered.T @ (X_centered @ v) + regularization * v

        # CG iteration
        coef = jnp.zeros(n_features)
        r = Xty - matvec(coef)  # Initial residual
        p = r.copy()
        rsold = r @ r

        for i in range(min(max_iter, n_features)):
            Ap = matvec(p)
            alpha = rsold / (p @ Ap + 1e-10)
            coef = coef + alpha * p
            r = r - alpha * Ap
            rsnew = r @ r

            if jnp.sqrt(rsnew) < tol:
                break

            p = r + (rsnew / (rsold + 1e-10)) * p
            rsold = rsnew

        # Calculate intercept
        intercept = y_mean - X_mean @ coef

        return coef, intercept

    @staticmethod
    @jax.jit
    def _jax_linear_predict(X: jnp.ndarray, coef: jnp.ndarray, intercept: jnp.ndarray) -> jnp.ndarray:
        """JAX-compiled linear prediction."""
        return X @ coef + intercept

    @staticmethod
    def _jax_ridge_regression_fit(X: jnp.ndarray, y: jnp.ndarray, alpha: float) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """JAX Ridge regression fitting (uses linalg.solve, not Metal-compatible)."""
        n_samples, n_features = X.shape

        # Center the data
        X_mean = jnp.mean(X, axis=0)
        y_mean = jnp.mean(y)
        X_centered = X - X_mean
        y_centered = y - y_mean

        # Solve regularized normal equations: (X^T X + alpha*I) coef = X^T y
        XtX = X_centered.T @ X_centered
        Xty = X_centered.T @ y_centered

        coef = linalg.solve(XtX + alpha * jnp.eye(n_features), Xty)
        intercept = y_mean - X_mean @ coef

        return coef, intercept

    @staticmethod
    def _jax_ridge_regression_fit_cg(
        X: jnp.ndarray,
        y: jnp.ndarray,
        alpha: float,
        max_iter: int = 100,
        tol: float = 1e-8
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """JAX Ridge regression using Conjugate Gradient (Metal-compatible).

        Solves (X^T X + alpha*I) coef = X^T y using CG.
        """
        n_samples, n_features = X.shape

        # Center the data
        X_mean = jnp.mean(X, axis=0)
        y_mean = jnp.mean(y)
        X_centered = X - X_mean
        y_centered = y - y_mean

        # b = X^T y
        Xty = X_centered.T @ y_centered

        # A @ x = (X^T X + alpha*I) @ x
        def matvec(v):
            return X_centered.T @ (X_centered @ v) + alpha * v

        # CG iteration
        coef = jnp.zeros(n_features)
        r = Xty - matvec(coef)
        p = r.copy()
        rsold = r @ r

        for i in range(min(max_iter, n_features)):
            Ap = matvec(p)
            alpha_cg = rsold / (p @ Ap + 1e-10)
            coef = coef + alpha_cg * p
            r = r - alpha_cg * Ap
            rsnew = r @ r

            if jnp.sqrt(rsnew) < tol:
                break

            p = r + (rsnew / (rsold + 1e-10)) * p
            rsold = rsnew

        intercept = y_mean - X_mean @ coef

        return coef, intercept

    @staticmethod
    @jax.jit
    def _jax_elastic_net_step(
        coef: jnp.ndarray,
        X: jnp.ndarray,
        y: jnp.ndarray,
        alpha: float,
        l1_ratio: float,
        learning_rate: float = 0.01
    ) -> jnp.ndarray:
        """JAX-compiled ElasticNet gradient step using coordinate descent."""
        n_samples = X.shape[0]
        residual = y - X @ coef

        # Gradient of MSE
        grad_mse = -2.0 / n_samples * X.T @ residual

        # L2 gradient
        grad_l2 = alpha * (1 - l1_ratio) * coef

        # L1 subgradient (soft thresholding)
        l1_penalty = alpha * l1_ratio

        # Update step
        coef_new = coef - learning_rate * (grad_mse + grad_l2)

        # Soft thresholding for L1
        coef_new = jnp.sign(coef_new) * jnp.maximum(jnp.abs(coef_new) - learning_rate * l1_penalty, 0)

        return coef_new

    @staticmethod
    @jax.jit
    def _jax_logistic_fit_step(
        params: Dict[str, jnp.ndarray],
        X: jnp.ndarray,
        y: jnp.ndarray,
        C: float,
        learning_rate: float = 0.1
    ) -> Dict[str, jnp.ndarray]:
        """JAX-compiled logistic regression gradient step."""
        coef = params['coef']
        intercept = params['intercept']

        n_samples = X.shape[0]

        # Predictions
        logits = X @ coef + intercept
        probs = jax.nn.sigmoid(logits)

        # Gradients
        error = probs - y
        grad_coef = (1.0 / n_samples) * X.T @ error + (1.0 / C) * coef
        grad_intercept = jnp.mean(error)

        # Update
        new_coef = coef - learning_rate * grad_coef
        new_intercept = intercept - learning_rate * grad_intercept

        return {'coef': new_coef, 'intercept': new_intercept}

    @staticmethod
    @jax.jit
    def _jax_logistic_predict_proba(
        X: jnp.ndarray,
        coef: jnp.ndarray,
        intercept: jnp.ndarray
    ) -> jnp.ndarray:
        """JAX-compiled logistic regression probability prediction."""
        logits = X @ coef + intercept
        probs = jax.nn.sigmoid(logits)
        return jnp.stack([1 - probs, probs], axis=1)

    @staticmethod
    def _jax_pca_fit(X: jnp.ndarray, n_components: int) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """JAX-compiled PCA fitting.

        Note: Not using @jax.jit here because n_components needs to be static.
        The SVD itself is already optimized by XLA.
        """
        n_samples, n_features = X.shape

        # Center the data
        X_mean = jnp.mean(X, axis=0)
        X_centered = X - X_mean

        # Compute SVD
        U, s, Vt = jnp.linalg.svd(X_centered, full_matrices=False)

        # Select top components (n_components is known at call time)
        components = Vt[:n_components]
        explained_variance = (s[:n_components] ** 2) / (n_samples - 1)

        return components, explained_variance, X_mean

    @staticmethod
    def _jax_pca_fit_power(
        X: jnp.ndarray,
        n_components: int,
        max_iter: int = 100,
        tol: float = 1e-6
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """JAX PCA fitting using power iteration (Metal-compatible).

        Uses only matrix multiplication, works on all backends including Metal.
        """
        n_samples, n_features = X.shape

        # Center the data
        X_mean = jnp.mean(X, axis=0)
        X_centered = X - X_mean

        # Compute covariance matrix (or gram matrix for efficiency)
        # For n_samples >> n_features, use covariance
        # For n_features >> n_samples, use gram matrix
        if n_samples >= n_features:
            # Covariance approach
            C = X_centered.T @ X_centered / (n_samples - 1)
            use_cov = True
        else:
            # Gram matrix approach
            C = X_centered @ X_centered.T / (n_samples - 1)
            use_cov = False

        components_list = []
        eigenvalues = []

        for k in range(n_components):
            # Initialize random vector
            key = jax.random.PRNGKey(k)
            v = jax.random.normal(key, (C.shape[0],))
            v = v / jnp.linalg.norm(v)

            # Power iteration
            for _ in range(max_iter):
                v_new = C @ v
                v_new_norm = jnp.linalg.norm(v_new)
                v_new = v_new / (v_new_norm + 1e-10)

                # Check convergence
                if jnp.abs(jnp.abs(v @ v_new) - 1.0) < tol:
                    break
                v = v_new

            # Compute eigenvalue
            eigenvalue = v @ C @ v
            eigenvalues.append(eigenvalue)

            if use_cov:
                components_list.append(v)
            else:
                # Convert from gram eigenvector to covariance eigenvector
                u = X_centered.T @ v
                u = u / (jnp.linalg.norm(u) + 1e-10)
                components_list.append(u)

            # Deflate matrix
            if use_cov:
                C = C - eigenvalue * jnp.outer(v, v)
            else:
                C = C - eigenvalue * jnp.outer(v, v)

        components = jnp.stack(components_list)
        explained_variance = jnp.array(eigenvalues)

        return components, explained_variance, X_mean

    @staticmethod
    @jax.jit
    def _jax_pca_transform(X: jnp.ndarray, mean: jnp.ndarray, components: jnp.ndarray) -> jnp.ndarray:
        """JAX-compiled PCA transform."""
        X_centered = X - mean
        return X_centered @ components.T

    @staticmethod
    @jax.jit
    def _jax_pca_inverse_transform(
        X_transformed: jnp.ndarray,
        mean: jnp.ndarray,
        components: jnp.ndarray
    ) -> jnp.ndarray:
        """JAX-compiled PCA inverse transform."""
        return X_transformed @ components + mean

    @staticmethod
    @jax.jit
    def _jax_standard_scaler_fit(X: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """JAX-compiled StandardScaler fitting."""
        mean = jnp.mean(X, axis=0)
        std = jnp.std(X, axis=0)
        # Prevent division by zero
        std = jnp.where(std == 0, 1.0, std)
        return mean, std

    @staticmethod
    @jax.jit
    def _jax_standard_scaler_transform(
        X: jnp.ndarray,
        mean: jnp.ndarray,
        scale: jnp.ndarray
    ) -> jnp.ndarray:
        """JAX-compiled StandardScaler transform."""
        return (X - mean) / scale

    @staticmethod
    @jax.jit
    def _jax_standard_scaler_inverse_transform(
        X: jnp.ndarray,
        mean: jnp.ndarray,
        scale: jnp.ndarray
    ) -> jnp.ndarray:
        """JAX-compiled StandardScaler inverse transform."""
        return X * scale + mean

    @staticmethod
    @jax.jit
    def _jax_minmax_scaler_fit(X: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """JAX-compiled MinMaxScaler fitting."""
        data_min = jnp.min(X, axis=0)
        data_max = jnp.max(X, axis=0)
        data_range = data_max - data_min
        # Prevent division by zero
        data_range = jnp.where(data_range == 0, 1.0, data_range)
        return data_min, data_range

    @staticmethod
    @jax.jit
    def _jax_minmax_scaler_transform(
        X: jnp.ndarray,
        data_min: jnp.ndarray,
        data_range: jnp.ndarray,
        feature_range: Tuple[float, float] = (0.0, 1.0)
    ) -> jnp.ndarray:
        """JAX-compiled MinMaxScaler transform."""
        X_std = (X - data_min) / data_range
        return X_std * (feature_range[1] - feature_range[0]) + feature_range[0]

    @staticmethod
    @jax.jit
    def _jax_normalizer_transform(X: jnp.ndarray, norm: str = 'l2') -> jnp.ndarray:
        """JAX-compiled Normalizer transform."""
        if norm == 'l2':
            norms = jnp.linalg.norm(X, axis=1, keepdims=True)
        elif norm == 'l1':
            norms = jnp.sum(jnp.abs(X), axis=1, keepdims=True)
        else:  # max
            norms = jnp.max(jnp.abs(X), axis=1, keepdims=True)

        # Prevent division by zero
        norms = jnp.where(norms == 0, 1.0, norms)
        return X / norms

    @staticmethod
    def _jax_kmeans_step(X: jnp.ndarray, centers: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, float]:
        """JAX-compiled K-means iteration step.

        Note: Not using @jax.jit on outer function because of dynamic loop.
        Inner computations are still optimized.
        """
        # Compute squared distances to all centers efficiently
        # ||x - c||^2 = ||x||^2 - 2*x.c + ||c||^2
        X_sq = jnp.sum(X ** 2, axis=1, keepdims=True)
        centers_sq = jnp.sum(centers ** 2, axis=1)
        cross = X @ centers.T
        distances = X_sq - 2 * cross + centers_sq

        # Assign points to closest centers
        labels = jnp.argmin(distances, axis=1)

        # Compute inertia
        min_distances = jnp.min(distances, axis=1)
        inertia = jnp.sum(min_distances)

        # Update centers - vectorized approach
        n_clusters = centers.shape[0]
        n_features = centers.shape[1]

        # One-hot encode labels
        one_hot = jnp.eye(n_clusters)[labels]  # (n_samples, n_clusters)

        # Compute cluster sums and counts
        cluster_sums = one_hot.T @ X  # (n_clusters, n_features)
        cluster_counts = jnp.sum(one_hot, axis=0)  # (n_clusters,)

        # Compute new centers, keep old center if cluster is empty
        cluster_counts_safe = jnp.where(cluster_counts > 0, cluster_counts, 1)
        new_centers = cluster_sums / cluster_counts_safe[:, None]

        # Where cluster is empty, keep old center
        empty_mask = (cluster_counts == 0)[:, None]
        new_centers = jnp.where(empty_mask, centers, new_centers)

        return new_centers, labels, float(inertia)

    @staticmethod
    @jax.jit
    def _jax_euclidean_distances(X: jnp.ndarray, Y: jnp.ndarray = None) -> jnp.ndarray:
        """JAX-compiled pairwise Euclidean distances."""
        if Y is None:
            Y = X

        X_sq = jnp.sum(X ** 2, axis=1, keepdims=True)
        Y_sq = jnp.sum(Y ** 2, axis=1)
        cross = X @ Y.T
        distances = jnp.sqrt(jnp.maximum(X_sq - 2 * cross + Y_sq, 0))
        return distances

    @staticmethod
    @jax.jit
    def _jax_cosine_similarity(X: jnp.ndarray, Y: jnp.ndarray = None) -> jnp.ndarray:
        """JAX-compiled pairwise cosine similarity."""
        if Y is None:
            Y = X

        X_norm = X / jnp.linalg.norm(X, axis=1, keepdims=True)
        Y_norm = Y / jnp.linalg.norm(Y, axis=1, keepdims=True)
        return X_norm @ Y_norm.T

    def _apply_jax_linear_regression(self, X: np.ndarray, y: np.ndarray) -> Dict[str, np.ndarray]:
        """Apply JAX-accelerated linear regression."""
        X_jax = to_jax(X)
        y_jax = to_jax(y)

        if self._device:
            X_jax = put_on_device(X_jax, self._device)
            y_jax = put_on_device(y_jax, self._device)

        # Check if we're on Metal backend (needs special handling)
        backend = jax.default_backend().lower()

        if backend == 'metal':
            # Use Metal-compatible implementation (Conjugate Gradient)
            # CG only uses matrix-vector products which work on Metal
            coef_jax, intercept_jax = self._jax_linear_regression_fit_cg(X_jax, y_jax)
        else:
            # Use standard implementation with linalg.solve
            try:
                coef_jax, intercept_jax = self._jax_linear_regression_fit(X_jax, y_jax)
            except Exception:
                # Fallback to CG version
                coef_jax, intercept_jax = self._jax_linear_regression_fit_cg(X_jax, y_jax)

        return {
            'coef_': to_numpy(coef_jax),
            'intercept_': to_numpy(intercept_jax)
        }

    def _apply_jax_linear_predict(
        self,
        X: np.ndarray,
        coef: np.ndarray,
        intercept: np.ndarray
    ) -> np.ndarray:
        """Apply JAX-accelerated linear prediction."""
        X_jax = to_jax(X)
        coef_jax = to_jax(coef)
        intercept_jax = to_jax(intercept)

        if self._device:
            X_jax = put_on_device(X_jax, self._device)
            coef_jax = put_on_device(coef_jax, self._device)

        pred_jax = self._jax_linear_predict(X_jax, coef_jax, intercept_jax)
        return to_numpy(pred_jax)

    def _apply_jax_ridge_regression(self, X: np.ndarray, y: np.ndarray, alpha: float = 1.0) -> Dict[str, np.ndarray]:
        """Apply JAX-accelerated Ridge regression."""
        X_jax = to_jax(X)
        y_jax = to_jax(y)

        if self._device:
            X_jax = put_on_device(X_jax, self._device)
            y_jax = put_on_device(y_jax, self._device)

        # Check if we're on Metal backend (needs special handling)
        backend = jax.default_backend().lower()

        if backend == 'metal':
            # Use Metal-compatible CG implementation
            coef_jax, intercept_jax = self._jax_ridge_regression_fit_cg(X_jax, y_jax, alpha)
        else:
            # Use standard implementation
            try:
                coef_jax, intercept_jax = self._jax_ridge_regression_fit(X_jax, y_jax, alpha)
            except Exception:
                # Fallback to CG
                coef_jax, intercept_jax = self._jax_ridge_regression_fit_cg(X_jax, y_jax, alpha)

        return {
            'coef_': to_numpy(coef_jax),
            'intercept_': to_numpy(intercept_jax)
        }

    def _apply_jax_elastic_net(
        self,
        X: np.ndarray,
        y: np.ndarray,
        alpha: float = 1.0,
        l1_ratio: float = 0.5,
        max_iter: int = 1000,
        tol: float = 1e-4
    ) -> Dict[str, np.ndarray]:
        """Apply JAX-accelerated ElasticNet regression."""
        X_jax = to_jax(X)
        y_jax = to_jax(y)

        n_samples, n_features = X.shape

        # Initialize coefficients
        coef = jnp.zeros(n_features)

        # Center y
        y_mean = jnp.mean(y_jax)
        y_centered = y_jax - y_mean

        # Center X
        X_mean = jnp.mean(X_jax, axis=0)
        X_centered = X_jax - X_mean

        # Coordinate descent
        for i in range(max_iter):
            coef_old = coef
            coef = self._jax_elastic_net_step(
                coef, X_centered, y_centered, alpha, l1_ratio
            )

            # Check convergence
            if jnp.max(jnp.abs(coef - coef_old)) < tol:
                break

        intercept = y_mean - X_mean @ coef

        return {
            'coef_': to_numpy(coef),
            'intercept_': to_numpy(intercept),
            'n_iter_': i + 1
        }

    def _apply_jax_logistic_regression(
        self,
        X: np.ndarray,
        y: np.ndarray,
        C: float = 1.0,
        max_iter: int = 100,
        tol: float = 1e-4
    ) -> Dict[str, np.ndarray]:
        """Apply JAX-accelerated logistic regression."""
        X_jax = to_jax(X)
        y_jax = to_jax(y)

        n_samples, n_features = X.shape

        # Initialize parameters
        params = {
            'coef': jnp.zeros(n_features),
            'intercept': jnp.array(0.0)
        }

        # Gradient descent
        for i in range(max_iter):
            params_old = params.copy()
            params = self._jax_logistic_fit_step(params, X_jax, y_jax, C)

            # Check convergence
            coef_diff = jnp.max(jnp.abs(params['coef'] - params_old['coef']))
            if coef_diff < tol:
                break

        return {
            'coef_': to_numpy(params['coef']).reshape(1, -1),
            'intercept_': to_numpy(params['intercept']).reshape(1,),
            'n_iter_': [i + 1]
        }

    def _apply_jax_pca(self, X: np.ndarray, n_components: int) -> Dict[str, np.ndarray]:
        """Apply JAX-accelerated PCA."""
        X_jax = to_jax(X)

        if self._device:
            X_jax = put_on_device(X_jax, self._device)

        # Check if we're on Metal backend (needs special handling)
        backend = jax.default_backend().lower()

        if backend == 'metal':
            # Use Metal-compatible power iteration implementation
            components_jax, explained_variance_jax, mean_jax = self._jax_pca_fit_power(X_jax, n_components)
        else:
            # Use standard SVD implementation
            try:
                components_jax, explained_variance_jax, mean_jax = self._jax_pca_fit(X_jax, n_components)
            except Exception:
                # Fallback to power iteration
                components_jax, explained_variance_jax, mean_jax = self._jax_pca_fit_power(X_jax, n_components)

        return {
            'components_': to_numpy(components_jax),
            'explained_variance_': to_numpy(explained_variance_jax),
            'mean_': to_numpy(mean_jax)
        }

    def _apply_jax_pca_transform(
        self,
        X: np.ndarray,
        mean: np.ndarray,
        components: np.ndarray
    ) -> np.ndarray:
        """Apply JAX-accelerated PCA transform."""
        X_jax = to_jax(X)
        mean_jax = to_jax(mean)
        components_jax = to_jax(components)

        transformed = self._jax_pca_transform(X_jax, mean_jax, components_jax)
        return to_numpy(transformed)

    def _apply_jax_standard_scaler_fit(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Apply JAX-accelerated StandardScaler fitting."""
        X_jax = to_jax(X)

        mean_jax, scale_jax = self._jax_standard_scaler_fit(X_jax)

        return {
            'mean_': to_numpy(mean_jax),
            'scale_': to_numpy(scale_jax),
            'var_': to_numpy(scale_jax ** 2),
            'n_features_in_': X.shape[1],
            'n_samples_seen_': X.shape[0]
        }

    def _apply_jax_standard_scaler_transform(
        self,
        X: np.ndarray,
        mean: np.ndarray,
        scale: np.ndarray
    ) -> np.ndarray:
        """Apply JAX-accelerated StandardScaler transform."""
        X_jax = to_jax(X)
        mean_jax = to_jax(mean)
        scale_jax = to_jax(scale)

        transformed = self._jax_standard_scaler_transform(X_jax, mean_jax, scale_jax)
        return to_numpy(transformed)

    def _apply_jax_kmeans_iteration(self, X: np.ndarray, centers: np.ndarray) -> Dict[str, np.ndarray]:
        """Apply JAX-accelerated K-means iteration."""
        X_jax = to_jax(X)
        centers_jax = to_jax(centers)

        if self._device:
            X_jax = put_on_device(X_jax, self._device)
            centers_jax = put_on_device(centers_jax, self._device)

        new_centers_jax, labels_jax, inertia = self._jax_kmeans_step(X_jax, centers_jax)

        return {
            'cluster_centers_': to_numpy(new_centers_jax),
            'labels_': to_numpy(labels_jax),
            'inertia_': float(inertia)
        }


class JAXLinearModelMixin(UniversalJAXMixin):
    """Mixin for JAX-accelerated linear models."""

    def jax_fit(self, X: np.ndarray, y: np.ndarray, algorithm: str = 'linear') -> 'JAXLinearModelMixin':
        """JAX-accelerated fitting for linear models."""
        if not self._should_use_jax(X, algorithm):
            # Fallback to original implementation
            return self._original_fit(X, y)

        try:
            if self._monitor:
                with self._monitor.track(f"fit_{algorithm}"):
                    results = self._do_jax_fit(X, y, algorithm)
            else:
                results = self._do_jax_fit(X, y, algorithm)

            if results is None:
                return self._original_fit(X, y)

            # Set attributes
            for attr_name, attr_value in results.items():
                setattr(self, attr_name, attr_value)

            self._fitted_with_jax = True
            return self

        except Exception as e:
            config = get_config()
            if config.get("fallback_on_error", True):
                import warnings
                warnings.warn(f"JAX fitting failed: {e}. Using original implementation.")
                return self._original_fit(X, y)
            else:
                raise

    def _do_jax_fit(self, X: np.ndarray, y: np.ndarray, algorithm: str) -> Optional[Dict]:
        """Perform the actual JAX fitting."""
        if algorithm == 'linear':
            return self._apply_jax_linear_regression(X, y)
        elif algorithm == 'ridge':
            alpha = getattr(self, 'alpha', 1.0)
            return self._apply_jax_ridge_regression(X, y, alpha)
        elif algorithm == 'elastic_net':
            alpha = getattr(self, 'alpha', 1.0)
            l1_ratio = getattr(self, 'l1_ratio', 0.5)
            max_iter = getattr(self, 'max_iter', 1000)
            tol = getattr(self, 'tol', 1e-4)
            return self._apply_jax_elastic_net(X, y, alpha, l1_ratio, max_iter, tol)
        elif algorithm == 'logistic':
            C = getattr(self, 'C', 1.0)
            max_iter = getattr(self, 'max_iter', 100)
            tol = getattr(self, 'tol', 1e-4)
            return self._apply_jax_logistic_regression(X, y, C, max_iter, tol)
        else:
            return None

    def jax_predict(self, X: np.ndarray) -> np.ndarray:
        """JAX-accelerated prediction for linear models."""
        if not getattr(self, '_fitted_with_jax', False):
            return self._original_predict(X)

        try:
            coef = getattr(self, 'coef_', None)
            intercept = getattr(self, 'intercept_', None)

            if coef is None:
                raise ValueError("Model not fitted")

            # Handle 2D coef (for multiclass)
            if coef.ndim == 2:
                coef = coef.squeeze()

            if self._monitor:
                with self._monitor.track("predict"):
                    result = self._apply_jax_linear_predict(X, coef, intercept)
            else:
                result = self._apply_jax_linear_predict(X, coef, intercept)

            return result

        except Exception as e:
            config = get_config()
            if config.get("fallback_on_error", True):
                import warnings
                warnings.warn(f"JAX prediction failed: {e}. Using original implementation.")
                return self._original_predict(X)
            else:
                raise


class JAXPreprocessingMixin(UniversalJAXMixin):
    """Mixin for JAX-accelerated preprocessing transformers."""

    def jax_fit(self, X: np.ndarray, y: np.ndarray = None) -> 'JAXPreprocessingMixin':
        """JAX-accelerated fitting for preprocessing transformers."""
        algorithm = self.__class__.__name__.lower()
        if not self._should_use_jax(X, algorithm):
            return self._original_fit(X, y)

        try:
            if 'standardscaler' in algorithm:
                results = self._apply_jax_standard_scaler_fit(X)
            elif 'minmaxscaler' in algorithm:
                X_jax = to_jax(X)
                data_min, data_range = self._jax_minmax_scaler_fit(X_jax)
                results = {
                    'data_min_': to_numpy(data_min),
                    'data_range_': to_numpy(data_range),
                    'data_max_': to_numpy(data_min + data_range),
                    'n_features_in_': X.shape[1],
                    'n_samples_seen_': X.shape[0]
                }
            else:
                return self._original_fit(X, y)

            for attr_name, attr_value in results.items():
                setattr(self, attr_name, attr_value)

            self._fitted_with_jax = True
            return self

        except Exception as e:
            config = get_config()
            if config.get("fallback_on_error", True):
                import warnings
                warnings.warn(f"JAX preprocessing fit failed: {e}. Using original implementation.")
                return self._original_fit(X, y)
            else:
                raise

    def jax_transform(self, X: np.ndarray) -> np.ndarray:
        """JAX-accelerated transform for preprocessing."""
        if not getattr(self, '_fitted_with_jax', False):
            return self._original_transform(X)

        try:
            algorithm = self.__class__.__name__.lower()

            if 'standardscaler' in algorithm:
                mean = getattr(self, 'mean_', None)
                scale = getattr(self, 'scale_', None)
                if mean is None or scale is None:
                    raise ValueError("Model not fitted")
                return self._apply_jax_standard_scaler_transform(X, mean, scale)

            elif 'minmaxscaler' in algorithm:
                X_jax = to_jax(X)
                data_min = to_jax(self.data_min_)
                data_range = to_jax(self.data_range_)
                feature_range = getattr(self, 'feature_range', (0, 1))
                transformed = self._jax_minmax_scaler_transform(
                    X_jax, data_min, data_range, feature_range
                )
                return to_numpy(transformed)

            elif 'normalizer' in algorithm:
                X_jax = to_jax(X)
                norm = getattr(self, 'norm', 'l2')
                transformed = self._jax_normalizer_transform(X_jax, norm)
                return to_numpy(transformed)

            else:
                return self._original_transform(X)

        except Exception as e:
            config = get_config()
            if config.get("fallback_on_error", True):
                import warnings
                warnings.warn(f"JAX transform failed: {e}. Using original implementation.")
                return self._original_transform(X)
            else:
                raise

    def jax_fit_transform(self, X: np.ndarray, y: np.ndarray = None) -> np.ndarray:
        """JAX-accelerated fit_transform."""
        self.jax_fit(X, y)
        return self.jax_transform(X)

    def jax_inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """JAX-accelerated inverse transform."""
        if not getattr(self, '_fitted_with_jax', False):
            return self._original_inverse_transform(X)

        try:
            algorithm = self.__class__.__name__.lower()

            if 'standardscaler' in algorithm:
                X_jax = to_jax(X)
                mean_jax = to_jax(self.mean_)
                scale_jax = to_jax(self.scale_)
                result = self._jax_standard_scaler_inverse_transform(X_jax, mean_jax, scale_jax)
                return to_numpy(result)

            else:
                return self._original_inverse_transform(X)

        except Exception as e:
            config = get_config()
            if config.get("fallback_on_error", True):
                import warnings
                warnings.warn(f"JAX inverse_transform failed: {e}. Using original.")
                return self._original_inverse_transform(X)
            else:
                raise


class JAXClusterMixin(UniversalJAXMixin):
    """Mixin for JAX-accelerated clustering algorithms."""

    def jax_fit(self, X: np.ndarray, y: np.ndarray = None) -> 'JAXClusterMixin':
        """JAX-accelerated fitting for clustering algorithms."""
        if not self._should_use_jax(X, 'KMeans'):
            return self._original_fit(X)

        try:
            # Initialize centers (this is algorithm-specific)
            n_clusters = getattr(self, 'n_clusters', 8)
            centers = self._initialize_centers(X, n_clusters)

            # Iterative K-means with JAX acceleration
            max_iter = getattr(self, 'max_iter', 300)
            tol = getattr(self, 'tol', 1e-4)

            best_inertia = float('inf')
            best_centers = centers
            best_labels = None

            for i in range(max_iter):
                results = self._apply_jax_kmeans_iteration(X, centers)
                new_centers = results['cluster_centers_']
                inertia = results.get('inertia_', float('inf'))

                # Track best result
                if inertia < best_inertia:
                    best_inertia = inertia
                    best_centers = new_centers
                    best_labels = results['labels_']

                # Check convergence
                if np.allclose(centers, new_centers, atol=tol):
                    break

                centers = new_centers

            # Set final results
            self.cluster_centers_ = best_centers
            self.labels_ = best_labels if best_labels is not None else results['labels_']
            self.inertia_ = best_inertia
            self.n_iter_ = i + 1
            self._fitted_with_jax = True

            return self

        except Exception as e:
            config = get_config()
            if config.get("fallback_on_error", True):
                import warnings
                warnings.warn(f"JAX clustering failed: {e}. Using original implementation.")
                return self._original_fit(X)
            else:
                raise

    def jax_predict(self, X: np.ndarray) -> np.ndarray:
        """JAX-accelerated cluster prediction."""
        if not getattr(self, '_fitted_with_jax', False):
            return self._original_predict(X)

        try:
            X_jax = to_jax(X)
            centers_jax = to_jax(self.cluster_centers_)

            # Compute distances and assign labels
            distances = self._jax_euclidean_distances(X_jax, centers_jax)
            labels = jnp.argmin(distances, axis=1)

            return to_numpy(labels)

        except Exception as e:
            config = get_config()
            if config.get("fallback_on_error", True):
                import warnings
                warnings.warn(f"JAX cluster prediction failed: {e}. Using original.")
                return self._original_predict(X)
            else:
                raise

    def jax_transform(self, X: np.ndarray) -> np.ndarray:
        """JAX-accelerated transform to cluster-distance space."""
        if not getattr(self, '_fitted_with_jax', False):
            return self._original_transform(X)

        try:
            X_jax = to_jax(X)
            centers_jax = to_jax(self.cluster_centers_)

            distances = self._jax_euclidean_distances(X_jax, centers_jax)
            return to_numpy(distances)

        except Exception as e:
            config = get_config()
            if config.get("fallback_on_error", True):
                import warnings
                warnings.warn(f"JAX cluster transform failed: {e}. Using original.")
                return self._original_transform(X)
            else:
                raise

    def _initialize_centers(self, X: np.ndarray, n_clusters: int) -> np.ndarray:
        """Initialize cluster centers using k-means++."""
        n_samples, n_features = X.shape
        rng = np.random.RandomState(getattr(self, 'random_state', None))

        # Initialize first center randomly
        centers = np.zeros((n_clusters, n_features))
        center_idx = rng.randint(n_samples)
        centers[0] = X[center_idx]

        # K-means++ initialization
        for k in range(1, n_clusters):
            # Compute distances to nearest center
            distances = np.zeros(n_samples)
            for i in range(n_samples):
                min_dist = float('inf')
                for j in range(k):
                    dist = np.sum((X[i] - centers[j]) ** 2)
                    min_dist = min(min_dist, dist)
                distances[i] = min_dist

            # Choose next center with probability proportional to distance^2
            probs = distances / distances.sum()
            center_idx = rng.choice(n_samples, p=probs)
            centers[k] = X[center_idx]

        return centers


class JAXDecompositionMixin(UniversalJAXMixin):
    """Mixin for JAX-accelerated decomposition algorithms."""

    def jax_fit(self, X: np.ndarray, y: np.ndarray = None) -> 'JAXDecompositionMixin':
        """JAX-accelerated fitting for decomposition algorithms."""
        if not self._should_use_jax(X, 'PCA'):
            return self._original_fit(X)

        try:
            n_components = getattr(self, 'n_components', min(X.shape))
            if n_components is None:
                n_components = min(X.shape)

            if self._monitor:
                with self._monitor.track("fit_pca"):
                    results = self._apply_jax_pca(X, n_components)
            else:
                results = self._apply_jax_pca(X, n_components)

            # Set attributes
            for attr_name, attr_value in results.items():
                setattr(self, attr_name, attr_value)

            # Calculate explained variance ratio
            total_var = np.sum(results['explained_variance_'])
            if total_var > 0:
                self.explained_variance_ratio_ = results['explained_variance_'] / total_var
            else:
                self.explained_variance_ratio_ = np.zeros_like(results['explained_variance_'])

            self.n_components_ = n_components
            self._fitted_with_jax = True

            return self

        except Exception as e:
            config = get_config()
            if config.get("fallback_on_error", True):
                import warnings
                warnings.warn(f"JAX decomposition failed: {e}. Using original implementation.")
                return self._original_fit(X)
            else:
                raise

    def jax_transform(self, X: np.ndarray) -> np.ndarray:
        """JAX-accelerated PCA transform."""
        if not getattr(self, '_fitted_with_jax', False):
            return self._original_transform(X)

        try:
            mean = getattr(self, 'mean_', None)
            components = getattr(self, 'components_', None)

            if mean is None or components is None:
                raise ValueError("Model not fitted")

            if self._monitor:
                with self._monitor.track("transform_pca"):
                    result = self._apply_jax_pca_transform(X, mean, components)
            else:
                result = self._apply_jax_pca_transform(X, mean, components)

            return result

        except Exception as e:
            config = get_config()
            if config.get("fallback_on_error", True):
                import warnings
                warnings.warn(f"JAX transform failed: {e}. Using original implementation.")
                return self._original_transform(X)
            else:
                raise

    def jax_fit_transform(self, X: np.ndarray, y: np.ndarray = None) -> np.ndarray:
        """JAX-accelerated fit and transform."""
        self.jax_fit(X)
        return self.jax_transform(X)

    def jax_inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """JAX-accelerated inverse transform."""
        if not getattr(self, '_fitted_with_jax', False):
            return self._original_inverse_transform(X)

        try:
            X_jax = to_jax(X)
            mean_jax = to_jax(self.mean_)
            components_jax = to_jax(self.components_)

            result = self._jax_pca_inverse_transform(X_jax, mean_jax, components_jax)
            return to_numpy(result)

        except Exception as e:
            config = get_config()
            if config.get("fallback_on_error", True):
                import warnings
                warnings.warn(f"JAX inverse_transform failed: {e}. Using original.")
                return self._original_inverse_transform(X)
            else:
                raise


class JAXMetricsMixin(UniversalJAXMixin):
    """Mixin for JAX-accelerated metric computations."""

    @staticmethod
    @jax.jit
    def _jax_accuracy_score(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> float:
        """JAX-compiled accuracy score."""
        return jnp.mean(y_true == y_pred)

    @staticmethod
    @jax.jit
    def _jax_mean_squared_error(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> float:
        """JAX-compiled mean squared error."""
        return jnp.mean((y_true - y_pred) ** 2)

    @staticmethod
    @jax.jit
    def _jax_mean_absolute_error(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> float:
        """JAX-compiled mean absolute error."""
        return jnp.mean(jnp.abs(y_true - y_pred))

    @staticmethod
    @jax.jit
    def _jax_r2_score(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> float:
        """JAX-compiled R² score."""
        ss_res = jnp.sum((y_true - y_pred) ** 2)
        ss_tot = jnp.sum((y_true - jnp.mean(y_true)) ** 2)
        return 1 - ss_res / ss_tot


def create_jax_accelerated_class(original_class: type, mixin_class: type) -> type:
    """Create a JAX-accelerated version of a class using a mixin.

    Parameters
    ----------
    original_class : type
        The original xlearn class
    mixin_class : type
        The JAX mixin class to use

    Returns
    -------
    accelerated_class : type
        JAX-accelerated class
    """
    class JAXAcceleratedClass(mixin_class, original_class):
        def __init__(self, *args, **kwargs):
            original_class.__init__(self, *args, **kwargs)
            mixin_class.__init__(self)

            # Store original methods
            self._original_fit = lambda X, y=None: original_class.fit(self, X, y) if y is not None else original_class.fit(self, X)
            self._original_predict = lambda X: original_class.predict(self, X) if hasattr(original_class, 'predict') else None
            self._original_transform = lambda X: original_class.transform(self, X) if hasattr(original_class, 'transform') else None
            self._original_inverse_transform = lambda X: original_class.inverse_transform(self, X) if hasattr(original_class, 'inverse_transform') else None
            self._fitted_with_jax = False

        def fit(self, X, y=None, **kwargs):
            """Override fit to use JAX acceleration when beneficial."""
            return self.jax_fit(X, y, **kwargs) if y is not None else self.jax_fit(X, **kwargs)

        def predict(self, X):
            """Override predict to use JAX acceleration."""
            if hasattr(self, 'jax_predict'):
                return self.jax_predict(X)
            return self._original_predict(X)

        def transform(self, X):
            """Override transform to use JAX acceleration."""
            if hasattr(self, 'jax_transform'):
                return self.jax_transform(X)
            return self._original_transform(X)

        def fit_transform(self, X, y=None, **kwargs):
            """Override fit_transform to use JAX acceleration."""
            if hasattr(self, 'jax_fit_transform'):
                return self.jax_fit_transform(X, y)
            self.fit(X, y, **kwargs)
            return self.transform(X)

        def inverse_transform(self, X):
            """Override inverse_transform to use JAX acceleration."""
            if hasattr(self, 'jax_inverse_transform'):
                return self.jax_inverse_transform(X)
            return self._original_inverse_transform(X)

    # Copy metadata
    JAXAcceleratedClass.__name__ = f"JAX{original_class.__name__}"
    JAXAcceleratedClass.__qualname__ = f"JAX{original_class.__qualname__}"
    JAXAcceleratedClass.__module__ = original_class.__module__
    JAXAcceleratedClass.__doc__ = original_class.__doc__

    return JAXAcceleratedClass


# =============================================================================
# Convenience Functions
# =============================================================================

def warmup_jax(X_shape: Tuple[int, int] = (1000, 100), algorithms: list = None):
    """Warmup JAX JIT compilation with dummy data.

    This helps avoid JIT compilation overhead during actual usage.

    Parameters
    ----------
    X_shape : tuple, default=(1000, 100)
        Shape of dummy data for warmup.
    algorithms : list, optional
        List of algorithms to warmup. If None, warms up common ones.

    Examples
    --------
    >>> warmup_jax((1000, 100), ['linear', 'ridge', 'pca'])
    """
    if algorithms is None:
        algorithms = ['linear', 'ridge', 'pca', 'kmeans', 'standard_scaler']

    n_samples, n_features = X_shape
    X_dummy = np.random.randn(n_samples, n_features).astype(np.float32)
    y_dummy = np.random.randn(n_samples).astype(np.float32)

    mixin = UniversalJAXMixin()
    mixin.__init__()

    for algo in algorithms:
        try:
            if algo in ['linear', 'ridge']:
                mixin._apply_jax_linear_regression(X_dummy, y_dummy)
            elif algo == 'pca':
                mixin._apply_jax_pca(X_dummy, min(10, n_features))
            elif algo == 'kmeans':
                centers = X_dummy[:8]
                mixin._apply_jax_kmeans_iteration(X_dummy, centers)
            elif algo == 'standard_scaler':
                mixin._apply_jax_standard_scaler_fit(X_dummy)
        except Exception:
            pass  # Ignore warmup failures

    # Force compilation to complete
    jax.block_until_ready(jnp.array(0.0))
