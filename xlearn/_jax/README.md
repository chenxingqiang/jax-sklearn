# JAX Acceleration Module

This module provides transparent JAX acceleration for JAX-sklearn.

## 📁 File Structure

```
xlearn/_jax/
├── __init__.py              # Module entry point, JAX availability check
├── _config.py              # Configuration management system
├── _data_conversion.py     # NumPy ↔ JAX data conversion utilities
├── _accelerator.py         # Accelerator registration and management system
├── _proxy.py              # Intelligent proxy system
├── _universal_jax.py      # Universal JAX acceleration implementation
└── README.md              # This documentation
```

## 🚀 Core Architecture

### 1. Intelligent Proxy Pattern (`_proxy.py`)
- **EstimatorProxy**: Transparently switches between JAX and original implementations
- **create_intelligent_proxy**: Automatically creates JAX acceleration proxy for any algorithm
- **Automatic Fallback**: Automatically uses original implementation when JAX fails

### 2. Universal JAX Implementation (`_universal_jax.py`)
- **UniversalJAXMixin**: Base JAX acceleration mixin class
- **JAXLinearModelMixin**: JAX acceleration for linear models
- **JAXClusterMixin**: JAX acceleration for clustering algorithms  
- **JAXDecompositionMixin**: JAX acceleration for dimensionality reduction algorithms
- **Performance Heuristics**: Intelligently decides when to use JAX

### 3. Configuration System (`_config.py`)
```python
import xlearn._jax as jax_config

# Check JAX status
jax_config.get_config()

# Configure JAX settings
jax_config.set_config(enable_jax=True, jax_platform="gpu")

# Temporary configuration
with jax_config.config_context(enable_jax=False):
    # Force NumPy implementation
    pass
```

### 4. Data Conversion (`_data_conversion.py`)
- **to_jax()**: NumPy → JAX array conversion
- **to_numpy()**: JAX → NumPy array conversion
- **auto_convert_arrays**: Decorator for automatic data conversion

### 5. Registration System (`_accelerator.py`)
- **AcceleratorRegistry**: Manages JAX implementation registration
- **@accelerated_estimator**: Decorator for registering JAX implementations
- **create_accelerated_estimator**: Creates accelerated instances

## ⚡ How It Works

1. **Automatic Detection**: Checks JAX availability at system startup
2. **Dynamic Proxy**: Creates intelligent proxy for each algorithm class
3. **Performance Decision**: Intelligently selects implementation based on data scale
4. **Transparent Switching**: Seamless JAX/NumPy switching without user awareness
5. **Error Fallback**: Automatically uses original implementation when JAX fails

## 🎯 Performance Optimization

### Heuristic Rules
```python
# Algorithm-specific thresholds
thresholds = {
    'LinearRegression': {'min_complexity': 1e8, 'min_samples': 10000},
    'KMeans': {'min_complexity': 1e6, 'min_samples': 5000},
    'PCA': {'min_complexity': 1e7, 'min_samples': 5000},
    # ...
}
```

### JIT Compilation Optimization
- Static function compilation: `@jax.jit` decorates core computations
- Function caching: Avoids repeated compilation overhead
- Numerical stability: Adds regularization to prevent numerical issues

## 🔧 Extending New Algorithms

Adding JAX support for new algorithms:

```python
# 1. Add specialized mixin in _universal_jax.py
class JAXNewAlgorithmMixin(UniversalJAXMixin):
    def jax_fit(self, X, y=None):
        # JAX implementation
        pass

# 2. Add algorithm detection in _proxy.py
def create_universal_jax_class(original_class):
    if 'new_algorithm' in module_name:
        mixin_class = JAXNewAlgorithmMixin
    # ...
```

## 📊 Usage Examples

```python
import xlearn as sklearn  # JAX automatically enabled

# Normal usage, JAX automatically accelerates in the background
model = sklearn.linear_model.LinearRegression()
model.fit(X, y)  # Automatically uses JAX for large data
predictions = model.predict(X_test)

# Check if JAX was used
print(f"Using JAX: {getattr(model, 'is_using_jax', False)}")
```

## 🎉 Features

- ✅ **100% API Compatible**: Fully compatible with scikit-learn interface
- ✅ **Transparent Acceleration**: Users need no code modifications
- ✅ **Intelligent Fallback**: Automatically uses original implementation on errors
- ✅ **Performance Optimization**: Intelligent decisions based on data scale
- ✅ **Easy to Extend**: Modular design facilitates adding new algorithms

This architecture ensures that JAX-sklearn provides both performance improvements and complete compatibility and stability.
