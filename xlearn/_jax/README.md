# JAX Acceleration Module

This module provides transparent JAX acceleration for JAX-sklearn.

## 🔧 Installation

### Using uv (Recommended - 10-100x faster than pip)

```bash
# Install uv first (if not installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# CPU only (all platforms)
uv pip install jax-sklearn[jax-cpu]

# NVIDIA GPU with CUDA 12
uv pip install jax-sklearn[jax-gpu]

# NVIDIA GPU with CUDA 11
uv pip install jax-sklearn[jax-cuda11]

# Google TPU
uv pip install jax-sklearn[jax-tpu]

# Apple Silicon (M1/M2/M3/M4)
uv pip install jax-sklearn[jax-metal]

# Development installation
uv pip install -e ".[tests,benchmark]"
```

### Using pip

```bash
# CPU only (all platforms)
pip install jax-sklearn[jax-cpu]

# NVIDIA GPU with CUDA 12
pip install jax-sklearn[jax-gpu]

# NVIDIA GPU with CUDA 11
pip install jax-sklearn[jax-cuda11]

# Google TPU
pip install jax-sklearn[jax-tpu]

# Apple Silicon (M1/M2/M3/M4)
pip install jax-sklearn[jax-metal]
```

### Auto-detect hardware and get recommended command

```python
from xlearn._jax import get_installation_command, detect_hardware

# Get recommended installation for your hardware
print(get_installation_command())

# Get detailed hardware info and recommendations
info = detect_hardware()
for rec in info['recommendations']:
    print(f"{rec['action']}: {rec['command']}")
```

### Apple Silicon (Metal) Notes

The `jax-metal` plugin is **experimental**. Known limitations:
- Some linear algebra operations not supported (`linalg.solve`, `linalg.inv`, `eigh`)
- Only `float32` precision (no `float64`)
- Requires compatible JAX version

If you encounter issues:
```bash
# Install compatible versions
pip install jax==0.4.35 jaxlib==0.4.35 jax-metal

# Or force CPU fallback
JAX_PLATFORMS=cpu python your_script.py
```

## 📁 File Structure

```
xlearn/_jax/
├── __init__.py              # Module entry point, JAX availability check
├── _config.py               # Configuration management system
├── _data_conversion.py      # NumPy ↔ JAX data conversion utilities
├── _accelerator.py          # Accelerator registration and management system
├── _proxy.py                # Intelligent proxy system
├── _universal_jax.py        # Universal JAX acceleration implementation
├── tests/                   # Unit tests
│   ├── __init__.py
│   └── test_jax_module.py
└── README.md                # This documentation
```

## 🚀 Core Architecture

### 1. Intelligent Proxy Pattern (`_proxy.py`)
- **EstimatorProxy**: Transparently switches between JAX and original implementations
- **create_intelligent_proxy**: Automatically creates JAX acceleration proxy for any algorithm
- **Automatic Fallback**: Automatically uses original implementation when JAX fails

### 2. Universal JAX Implementation (`_universal_jax.py`)

#### Mixins for Different Algorithm Types
- **UniversalJAXMixin**: Base JAX acceleration mixin class
- **JAXLinearModelMixin**: JAX acceleration for linear models (LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression)
- **JAXPreprocessingMixin**: JAX acceleration for preprocessing (StandardScaler, MinMaxScaler, Normalizer)
- **JAXClusterMixin**: JAX acceleration for clustering algorithms (KMeans, etc.)
- **JAXDecompositionMixin**: JAX acceleration for dimensionality reduction (PCA, etc.)
- **JAXMetricsMixin**: JAX acceleration for metric computations

#### Accelerated Operations
- `fit()` - Model training
- `predict()` - Predictions for regression/classification
- `transform()` - Data transformation
- `fit_transform()` - Combined fit and transform
- `inverse_transform()` - Reverse transformations

### 3. Configuration System (`_config.py`)

```python
import xlearn._jax as jax_config

# Check JAX status
config = jax_config.get_config()
print(config)
# {'enable_jax': True, 'jax_platform': 'auto', 'fallback_on_error': True, ...}

# Configure JAX settings
jax_config.set_config(
    enable_jax=True,
    jax_platform="gpu",
    precision="float32",
    debug_mode=False
)

# Temporary configuration
with jax_config.config_context(enable_jax=False):
    # Force NumPy implementation
    pass
```

### 4. Device Management (`_universal_jax.py`)

JAX supports multiple accelerator backends:
- **CPU**: Always available
- **GPU**: NVIDIA (CUDA) or AMD (ROCm)
- **TPU**: Google Cloud TPU
- **Metal/MPS**: Apple Silicon (M1/M2/M3/M4)

```python
from xlearn._jax._universal_jax import (
    get_available_devices,
    get_default_device,
    select_device,
    put_on_device,
    get_best_device,
    check_device_available,
    print_device_info
)

# List all devices
devices = get_available_devices()
print(devices)
# {'cpu': [CpuDevice(id=0)], 'gpu': [GpuDevice(id=0)], 'metal': [MetalDevice(id=0)]}

# Check device availability
check_device_available('gpu')    # NVIDIA/AMD GPU
check_device_available('metal')  # Apple Metal
check_device_available('mps')    # Alias for Metal
check_device_available('tpu')    # Google TPU

# Select specific device
device = select_device('gpu', 0)     # First NVIDIA/AMD GPU
device = select_device('metal', 0)   # Apple Metal GPU
device = select_device('mps', 0)     # Same as 'metal'
device = select_device('auto')       # Best available

# Get best device (priority: TPU > GPU > Metal > CPU)
device = get_best_device()

# Put data on device
X_gpu = put_on_device(X_jax, device)

# Print device info
print_device_info()
```

#### Apple Silicon (M1/M2/M3/M4) Setup

To enable Metal GPU acceleration on Apple Silicon Macs:

```bash
pip install jax-metal
```

Then verify:
```python
from xlearn._jax import check_metal_available, print_device_info
print(check_metal_available())  # Should be True
print_device_info()  # Shows Metal device
```

### 5. Performance Monitoring (`_universal_jax.py`)

```python
from xlearn._jax._universal_jax import PerformanceMonitor, get_performance_monitor

# Create monitor
monitor = PerformanceMonitor()

# Track operations
with monitor.track("fit"):
    model.fit(X, y)

with monitor.track("predict"):
    y_pred = model.predict(X_test)

# Get statistics
stats = monitor.get_stats()
print(stats)
# {'fit': {'count': 1, 'total': 0.5, 'mean': 0.5, 'min': 0.5, 'max': 0.5},
#  'predict': {'count': 1, 'total': 0.01, ...}}

# Human-readable summary
print(monitor.summary())
```

### 6. Gradient Computation (`_universal_jax.py`)

```python
from xlearn._jax._universal_jax import (
    compute_gradient,
    compute_hessian,
    value_and_grad,
    mse_loss,
    ridge_loss,
    lasso_loss,
    elastic_net_loss,
    log_loss
)

# Define parameters
params = {
    'coef': jnp.zeros(n_features),
    'intercept': jnp.array(0.0)
}

# Compute gradients
gradients = compute_gradient(mse_loss, params, X, y)

# Compute loss and gradients together (more efficient)
loss_value, gradients = value_and_grad(ridge_loss, params, X, y, alpha=1.0)

# Compute Hessian for second-order optimization
hessian = compute_hessian(loss_fn, flat_params, X, y)
```

### 7. Batched Processing (`_universal_jax.py`)

```python
from xlearn._jax._universal_jax import (
    process_in_batches,
    estimate_memory_usage
)

# Process large data in batches
result = process_in_batches(transform_fn, X_large, batch_size=10000)

# Estimate memory usage before running
memory = estimate_memory_usage(X, algorithm='pca')
print(memory)
# {'input_data_mb': 38.15, 'estimated_peak_mb': 152.6, 'algorithm': 'pca'}
```

### 8. Data Conversion (`_data_conversion.py`)
- **to_jax()**: NumPy → JAX array conversion
- **to_numpy()**: JAX → NumPy array conversion
- **is_jax_array()**: Check if array is JAX
- **auto_convert_arrays**: Decorator for automatic data conversion

### 9. Registration System (`_accelerator.py`)
- **AcceleratorRegistry**: Manages JAX implementation registration
- **@accelerated_estimator**: Decorator for registering JAX implementations
- **create_accelerated_estimator**: Creates accelerated instances

## ⚡ How It Works

1. **Automatic Detection**: Checks JAX availability at system startup
2. **Dynamic Proxy**: Creates intelligent proxy for each algorithm class
3. **Performance Decision**: Intelligently selects implementation based on data scale
4. **Transparent Switching**: Seamless JAX/NumPy switching without user awareness
5. **Error Fallback**: Automatically uses original implementation when JAX fails

## 🎯 Supported Algorithms

### Linear Models
| Algorithm | fit | predict | gradient |
|-----------|-----|---------|----------|
| LinearRegression | ✅ | ✅ | ✅ |
| Ridge | ✅ | ✅ | ✅ |
| Lasso | ✅ | ✅ | ✅ |
| ElasticNet | ✅ | ✅ | ✅ |
| LogisticRegression | ✅ | ✅ | ✅ |

### Preprocessing
| Algorithm | fit | transform | inverse_transform |
|-----------|-----|-----------|-------------------|
| StandardScaler | ✅ | ✅ | ✅ |
| MinMaxScaler | ✅ | ✅ | ⬜ |
| Normalizer | ⬜ | ✅ | ⬜ |

### Clustering
| Algorithm | fit | predict | transform |
|-----------|-----|---------|-----------|
| KMeans | ✅ | ✅ | ✅ |

### Decomposition
| Algorithm | fit | transform | inverse_transform |
|-----------|-----|-----------|-------------------|
| PCA | ✅ | ✅ | ✅ |

## 🔧 Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `enable_jax` | bool | True | Enable/disable JAX acceleration |
| `jax_platform` | str | "auto" | Platform: 'auto', 'cpu', 'gpu', 'tpu' |
| `fallback_on_error` | bool | True | Fall back to NumPy on errors |
| `jit_compilation` | bool | True | Enable JIT compilation |
| `precision` | str | "float32" | Numerical precision |
| `debug_mode` | bool | False | Enable debug logging |
| `cache_compiled_functions` | bool | True | Cache JIT-compiled functions |
| `jax_auto_threshold` | bool | False | Use size-based heuristics |

## 📊 Usage Examples

### Basic Usage
```python
import xlearn as sklearn  # JAX automatically enabled

# Normal usage, JAX automatically accelerates in the background
model = sklearn.linear_model.LinearRegression()
model.fit(X, y)  # Automatically uses JAX for large data
predictions = model.predict(X_test)

# Check if JAX was used
print(f"Using JAX: {getattr(model, 'is_using_jax', False)}")
```

### Explicit JAX Control
```python
from xlearn._jax import set_config, config_context

# Globally enable JAX with GPU
set_config(enable_jax=True, jax_platform='gpu')

# Temporarily disable JAX for a section
with config_context(enable_jax=False):
    model.fit(X, y)  # Uses NumPy
```

### Performance Profiling
```python
from xlearn._jax._universal_jax import (
    UniversalJAXMixin, 
    PerformanceMonitor,
    warmup_jax
)

# Warmup JAX to avoid JIT overhead in benchmarks
warmup_jax((1000, 100), ['linear', 'pca', 'kmeans'])

# Profile operations
monitor = PerformanceMonitor()

with monitor.track("linear_fit"):
    mixin._apply_jax_linear_regression(X, y)

print(monitor.summary())
```

### Custom Device Selection
```python
from xlearn._jax._universal_jax import select_device, put_on_device

# Use specific GPU
device = select_device('gpu', 1)  # Second GPU

# Or use the mixin's device setting
mixin = UniversalJAXMixin()
mixin.set_device('gpu', 0)
```

## 🧪 Running Tests

```bash
# Run all JAX module tests
pytest xlearn/_jax/tests/ -v

# Run with coverage
pytest xlearn/_jax/tests/ --cov=xlearn/_jax

# Run specific test class
pytest xlearn/_jax/tests/test_jax_module.py::TestLinearModels -v
```

## 🔧 Extending with New Algorithms

Adding JAX support for new algorithms:

```python
# 1. Add specialized mixin in _universal_jax.py
class JAXNewAlgorithmMixin(UniversalJAXMixin):
    
    @staticmethod
    @jax.jit
    def _jax_new_algorithm_fit(X, y, param):
        """JAX-compiled fitting."""
        # JAX implementation
        return result
    
    def jax_fit(self, X, y=None):
        """JAX-accelerated fitting."""
        if not self._should_use_jax(X, 'NewAlgorithm'):
            return self._original_fit(X, y)
        
        result = self._jax_new_algorithm_fit(X, y, self.param)
        # Set attributes
        return self
    
    def jax_predict(self, X):
        """JAX-accelerated prediction."""
        # Implementation
        pass

# 2. Add algorithm detection in _proxy.py
def create_universal_jax_class(original_class):
    if 'new_algorithm' in module_name:
        mixin_class = JAXNewAlgorithmMixin
    # ...
```

## 🎉 Features

- ✅ **100% API Compatible**: Fully compatible with scikit-learn interface
- ✅ **Transparent Acceleration**: Users need no code modifications
- ✅ **Intelligent Fallback**: Automatically uses original implementation on errors
- ✅ **Performance Optimization**: Intelligent decisions based on data scale
- ✅ **Device Management**: Support for CPU, GPU, and TPU
- ✅ **Performance Monitoring**: Built-in profiling tools
- ✅ **Gradient Support**: Access to JAX's automatic differentiation
- ✅ **Batched Processing**: Memory-efficient processing of large datasets
- ✅ **Easy to Extend**: Modular design facilitates adding new algorithms

This architecture ensures that JAX-sklearn provides both performance improvements and complete compatibility and stability.
