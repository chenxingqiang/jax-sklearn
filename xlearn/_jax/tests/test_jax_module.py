"""Tests for JAX acceleration module."""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

# Check if JAX is available
try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

pytestmark = pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")


class TestConfig:
    """Test configuration management."""
    
    def test_get_config(self):
        """Test get_config returns expected keys."""
        from xlearn._jax import get_config
        
        config = get_config()
        assert isinstance(config, dict)
        assert "enable_jax" in config
        assert "jax_platform" in config
        assert "fallback_on_error" in config
        assert "precision" in config
    
    def test_set_config(self):
        """Test set_config modifies configuration."""
        from xlearn._jax import get_config, set_config
        
        original = get_config()
        
        try:
            set_config(enable_jax=False)
            assert get_config()["enable_jax"] == False
            
            set_config(precision="float64")
            assert get_config()["precision"] == "float64"
        finally:
            # Restore original config
            set_config(**original)
    
    def test_config_context(self):
        """Test config_context as context manager."""
        from xlearn._jax import get_config, config_context
        
        original_enable = get_config()["enable_jax"]
        
        with config_context(enable_jax=False):
            assert get_config()["enable_jax"] == False
        
        # Should be restored after context
        assert get_config()["enable_jax"] == original_enable
    
    def test_invalid_platform(self):
        """Test that invalid platform raises error."""
        from xlearn._jax import set_config
        
        with pytest.raises(ValueError, match="Invalid jax_platform"):
            set_config(jax_platform="invalid")
    
    def test_invalid_precision(self):
        """Test that invalid precision raises error."""
        from xlearn._jax import set_config
        
        with pytest.raises(ValueError, match="Invalid precision"):
            set_config(precision="float16")


class TestDataConversion:
    """Test data conversion utilities."""
    
    def test_to_jax(self):
        """Test conversion to JAX array."""
        from xlearn._jax._data_conversion import to_jax, is_jax_array
        
        X_np = np.array([[1.0, 2.0], [3.0, 4.0]])
        X_jax = to_jax(X_np)
        
        assert is_jax_array(X_jax)
        assert_allclose(np.asarray(X_jax), X_np)
    
    def test_to_numpy(self):
        """Test conversion to NumPy array."""
        from xlearn._jax._data_conversion import to_jax, to_numpy, is_numpy_array
        
        X_np = np.array([[1.0, 2.0], [3.0, 4.0]])
        X_jax = to_jax(X_np)
        X_back = to_numpy(X_jax)
        
        assert is_numpy_array(X_back)
        assert_allclose(X_back, X_np)
    
    def test_ensure_2d(self):
        """Test ensure_2d function."""
        from xlearn._jax._data_conversion import ensure_2d, to_jax
        
        X_1d = np.array([1.0, 2.0, 3.0])
        X_2d = ensure_2d(X_1d)
        
        assert X_2d.ndim == 2
        assert X_2d.shape == (1, 3)
        
        # Test with JAX array
        X_jax = to_jax(X_1d)
        X_jax_2d = ensure_2d(X_jax)
        assert X_jax_2d.ndim == 2
    
    def test_get_array_module(self):
        """Test get_array_module function."""
        from xlearn._jax._data_conversion import get_array_module, to_jax
        
        X_np = np.array([1.0, 2.0])
        X_jax = to_jax(X_np)
        
        assert get_array_module(X_np) is np
        # JAX module check
        jax_module = get_array_module(X_jax)
        assert hasattr(jax_module, 'array')


class TestDeviceManagement:
    """Test device management utilities."""
    
    def test_get_available_devices(self):
        """Test getting available devices."""
        from xlearn._jax._universal_jax import get_available_devices
        
        devices = get_available_devices()
        assert isinstance(devices, dict)
        assert 'cpu' in devices
        # At least CPU should be available
        assert len(devices['cpu']) > 0
    
    def test_get_default_device(self):
        """Test getting default device."""
        from xlearn._jax._universal_jax import get_default_device
        
        device = get_default_device()
        assert device is not None
    
    def test_select_device_cpu(self):
        """Test selecting CPU device."""
        from xlearn._jax._universal_jax import select_device
        
        device = select_device('cpu', 0)
        assert 'cpu' in str(device).lower()
    
    def test_select_device_invalid(self):
        """Test selecting invalid device raises error."""
        from xlearn._jax._universal_jax import select_device
        
        # This should raise since 'invalid' is not a valid device type
        with pytest.raises(RuntimeError):
            select_device('invalid', 0)


class TestPerformanceMonitor:
    """Test performance monitoring."""
    
    def test_basic_timing(self):
        """Test basic timing functionality."""
        from xlearn._jax._universal_jax import PerformanceMonitor
        import time
        
        monitor = PerformanceMonitor()
        
        with monitor.track("test_op"):
            time.sleep(0.01)
        
        stats = monitor.get_stats()
        assert "test_op" in stats
        assert stats["test_op"]["count"] == 1
        assert stats["test_op"]["total"] > 0
    
    def test_multiple_tracks(self):
        """Test tracking multiple operations."""
        from xlearn._jax._universal_jax import PerformanceMonitor
        
        monitor = PerformanceMonitor()
        
        for _ in range(3):
            with monitor.track("op1"):
                pass
        
        for _ in range(2):
            with monitor.track("op2"):
                pass
        
        stats = monitor.get_stats()
        assert stats["op1"]["count"] == 3
        assert stats["op2"]["count"] == 2
    
    def test_reset(self):
        """Test reset functionality."""
        from xlearn._jax._universal_jax import PerformanceMonitor
        
        monitor = PerformanceMonitor()
        
        with monitor.track("test"):
            pass
        
        assert len(monitor.get_stats()) > 0
        
        monitor.reset()
        assert len(monitor.get_stats()) == 0
    
    def test_disable(self):
        """Test disable functionality."""
        from xlearn._jax._universal_jax import PerformanceMonitor
        
        monitor = PerformanceMonitor()
        monitor.disable()
        
        with monitor.track("test"):
            pass
        
        # Should not record when disabled
        assert len(monitor.get_stats()) == 0


class TestLinearModels:
    """Test JAX-accelerated linear models."""
    
    def test_linear_regression_fit(self):
        """Test linear regression fitting."""
        from xlearn._jax._universal_jax import UniversalJAXMixin
        
        mixin = UniversalJAXMixin()
        mixin._device = None
        mixin._monitor = None
        
        # Generate test data
        np.random.seed(42)
        n_samples, n_features = 100, 5
        X = np.random.randn(n_samples, n_features).astype(np.float32)
        coef_true = np.random.randn(n_features).astype(np.float32)
        y = X @ coef_true + 0.1 * np.random.randn(n_samples).astype(np.float32)
        
        result = mixin._apply_jax_linear_regression(X, y)
        
        assert 'coef_' in result
        assert 'intercept_' in result
        assert result['coef_'].shape == (n_features,)
        
        # Check prediction accuracy
        y_pred = X @ result['coef_'] + result['intercept_']
        r2 = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - y.mean()) ** 2)
        assert r2 > 0.9, f"R² should be > 0.9, got {r2}"
    
    def test_ridge_regression_fit(self):
        """Test Ridge regression fitting."""
        from xlearn._jax._universal_jax import UniversalJAXMixin
        
        mixin = UniversalJAXMixin()
        mixin._device = None
        mixin._monitor = None
        
        np.random.seed(42)
        X = np.random.randn(100, 5).astype(np.float32)
        y = np.random.randn(100).astype(np.float32)
        
        result = mixin._apply_jax_ridge_regression(X, y, alpha=1.0)
        
        assert 'coef_' in result
        assert 'intercept_' in result
    
    def test_linear_predict(self):
        """Test linear prediction."""
        from xlearn._jax._universal_jax import UniversalJAXMixin
        
        mixin = UniversalJAXMixin()
        mixin._device = None
        
        X = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        coef = np.array([0.5, 0.5], dtype=np.float32)
        intercept = np.array(1.0, dtype=np.float32)
        
        y_pred = mixin._apply_jax_linear_predict(X, coef, intercept)
        
        expected = X @ coef + intercept
        assert_allclose(y_pred, expected, rtol=1e-5)


class TestPreprocessing:
    """Test JAX-accelerated preprocessing."""
    
    def test_standard_scaler_fit(self):
        """Test StandardScaler fitting."""
        from xlearn._jax._universal_jax import UniversalJAXMixin
        
        mixin = UniversalJAXMixin()
        
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
        
        result = mixin._apply_jax_standard_scaler_fit(X)
        
        assert 'mean_' in result
        assert 'scale_' in result
        assert_allclose(result['mean_'], np.mean(X, axis=0), rtol=1e-5)
        assert_allclose(result['scale_'], np.std(X, axis=0), rtol=1e-5)
    
    def test_standard_scaler_transform(self):
        """Test StandardScaler transform."""
        from xlearn._jax._universal_jax import UniversalJAXMixin
        
        mixin = UniversalJAXMixin()
        
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
        mean = np.mean(X, axis=0)
        scale = np.std(X, axis=0)
        
        result = mixin._apply_jax_standard_scaler_transform(X, mean, scale)
        
        expected = (X - mean) / scale
        assert_allclose(result, expected, rtol=1e-5)


class TestClustering:
    """Test JAX-accelerated clustering."""
    
    def test_kmeans_step(self):
        """Test K-means iteration step."""
        from xlearn._jax._universal_jax import UniversalJAXMixin
        
        mixin = UniversalJAXMixin()
        mixin._device = None
        
        np.random.seed(42)
        X = np.random.randn(100, 5).astype(np.float32)
        centers = X[:3].copy()  # Use first 3 points as initial centers
        
        result = mixin._apply_jax_kmeans_iteration(X, centers)
        
        assert 'cluster_centers_' in result
        assert 'labels_' in result
        assert result['cluster_centers_'].shape == (3, 5)
        assert result['labels_'].shape == (100,)
        assert set(result['labels_']).issubset({0, 1, 2})


class TestDecomposition:
    """Test JAX-accelerated decomposition."""
    
    def test_pca_fit(self):
        """Test PCA fitting."""
        from xlearn._jax._universal_jax import UniversalJAXMixin
        
        mixin = UniversalJAXMixin()
        mixin._device = None
        
        np.random.seed(42)
        X = np.random.randn(100, 10).astype(np.float32)
        n_components = 3
        
        result = mixin._apply_jax_pca(X, n_components)
        
        assert 'components_' in result
        assert 'explained_variance_' in result
        assert 'mean_' in result
        assert result['components_'].shape == (n_components, 10)
        assert result['explained_variance_'].shape == (n_components,)
    
    def test_pca_transform(self):
        """Test PCA transform."""
        from xlearn._jax._universal_jax import UniversalJAXMixin
        
        mixin = UniversalJAXMixin()
        mixin._device = None
        
        np.random.seed(42)
        X = np.random.randn(100, 10).astype(np.float32)
        
        # Fit PCA
        fit_result = mixin._apply_jax_pca(X, 3)
        
        # Transform
        X_transformed = mixin._apply_jax_pca_transform(
            X, fit_result['mean_'], fit_result['components_']
        )
        
        assert X_transformed.shape == (100, 3)


class TestGradients:
    """Test gradient computation utilities."""
    
    def test_compute_gradient(self):
        """Test gradient computation."""
        from xlearn._jax._universal_jax import compute_gradient, mse_loss
        
        np.random.seed(42)
        X = np.random.randn(50, 5).astype(np.float32)
        y = np.random.randn(50).astype(np.float32)
        
        params = {
            'coef': jnp.zeros(5),
            'intercept': jnp.array(0.0)
        }
        
        grads = compute_gradient(mse_loss, params, jnp.array(X), jnp.array(y))
        
        assert 'coef' in grads
        assert 'intercept' in grads
        assert grads['coef'].shape == (5,)
    
    def test_value_and_grad(self):
        """Test value and gradient computation."""
        from xlearn._jax._universal_jax import value_and_grad, mse_loss
        
        np.random.seed(42)
        X = np.random.randn(50, 5).astype(np.float32)
        y = np.random.randn(50).astype(np.float32)
        
        params = {
            'coef': jnp.zeros(5),
            'intercept': jnp.array(0.0)
        }
        
        loss, grads = value_and_grad(mse_loss, params, jnp.array(X), jnp.array(y))
        
        assert isinstance(float(loss), float)
        assert 'coef' in grads


class TestLossFunctions:
    """Test JAX loss functions."""
    
    def test_mse_loss(self):
        """Test MSE loss."""
        from xlearn._jax._universal_jax import mse_loss
        
        X = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        y = jnp.array([1.0, 2.0])
        params = {'coef': jnp.array([0.5, 0.0]), 'intercept': jnp.array(0.5)}
        
        loss = mse_loss(params, X, y)
        
        # Manual calculation
        pred = X @ params['coef'] + params['intercept']
        expected = float(jnp.mean((pred - y) ** 2))
        
        assert_allclose(float(loss), expected, rtol=1e-5)
    
    def test_ridge_loss(self):
        """Test Ridge loss."""
        from xlearn._jax._universal_jax import ridge_loss
        
        X = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        y = jnp.array([1.0, 2.0])
        params = {'coef': jnp.array([0.5, 0.5]), 'intercept': jnp.array(0.0)}
        alpha = 1.0
        
        loss = ridge_loss(params, X, y, alpha)
        
        # Should include L2 penalty
        pred = X @ params['coef'] + params['intercept']
        expected_mse = float(jnp.mean((pred - y) ** 2))
        expected_l2 = alpha * float(jnp.sum(params['coef'] ** 2))
        
        assert float(loss) > expected_mse  # Should be greater due to penalty


class TestBatchedProcessing:
    """Test batched processing utilities."""
    
    def test_process_in_batches(self):
        """Test batch processing."""
        from xlearn._jax._universal_jax import process_in_batches
        
        X = jnp.arange(100).reshape(100, 1).astype(jnp.float32)
        
        def square(x):
            return x ** 2
        
        result = process_in_batches(square, X, batch_size=30)
        
        expected = X ** 2
        assert_allclose(np.asarray(result), np.asarray(expected))
    
    def test_estimate_memory_usage(self):
        """Test memory estimation."""
        from xlearn._jax._universal_jax import estimate_memory_usage
        
        X = jnp.zeros((1000, 100), dtype=jnp.float32)
        
        memory = estimate_memory_usage(X, 'linear')
        
        assert 'input_data_mb' in memory
        assert 'estimated_peak_mb' in memory
        assert memory['input_data_mb'] > 0
        assert memory['estimated_peak_mb'] >= memory['input_data_mb']


class TestWarmup:
    """Test JAX warmup functionality."""
    
    def test_warmup_jax(self):
        """Test warmup function runs without error."""
        from xlearn._jax._universal_jax import warmup_jax
        
        # Should not raise any errors
        warmup_jax((100, 10), ['linear', 'pca'])


class TestProxy:
    """Test proxy system."""
    
    def test_create_proxy_class(self):
        """Test creating proxy class."""
        from xlearn._jax._proxy import create_proxy_class
        
        class DummyEstimator:
            def __init__(self, param=1):
                self.param = param
            
            def fit(self, X, y):
                return self
            
            def predict(self, X):
                return X
        
        ProxyClass = create_proxy_class(DummyEstimator)
        
        assert ProxyClass.__name__ == "DummyEstimator"
        
        instance = ProxyClass(param=5)
        assert instance._init_kwargs['param'] == 5
    
    def test_estimator_proxy_getattr(self):
        """Test EstimatorProxy attribute delegation."""
        from xlearn._jax._proxy import EstimatorProxy
        
        class DummyEstimator:
            def __init__(self):
                self.value = 42
            
            def fit(self, X, y):
                return self
        
        proxy = EstimatorProxy(DummyEstimator)
        
        # Should delegate to underlying implementation
        assert proxy.value == 42


class TestIntegration:
    """Integration tests for JAX acceleration."""
    
    def test_end_to_end_linear_regression(self):
        """Test end-to-end linear regression workflow."""
        from xlearn._jax._universal_jax import UniversalJAXMixin
        from xlearn._jax import get_config, set_config
        
        # Ensure JAX is enabled
        original_config = get_config()
        set_config(enable_jax=True)
        
        try:
            mixin = UniversalJAXMixin()
            mixin._device = None
            mixin._monitor = None
            
            # Generate data
            np.random.seed(123)
            X_train = np.random.randn(200, 10).astype(np.float32)
            y_train = X_train @ np.random.randn(10).astype(np.float32)
            X_test = np.random.randn(50, 10).astype(np.float32)
            
            # Fit
            result = mixin._apply_jax_linear_regression(X_train, y_train)
            
            # Predict
            y_pred = mixin._apply_jax_linear_predict(
                X_test, result['coef_'], result['intercept_']
            )
            
            assert y_pred.shape == (50,)
            
        finally:
            set_config(**original_config)
    
    def test_config_affects_behavior(self):
        """Test that config settings affect behavior."""
        from xlearn._jax._universal_jax import UniversalJAXMixin
        from xlearn._jax import set_config, config_context
        
        mixin = UniversalJAXMixin()
        mixin._performance_cache = {}
        
        X = np.random.randn(100, 10).astype(np.float32)
        
        # With auto_threshold disabled, should always use JAX
        with config_context(enable_jax=True, jax_auto_threshold=False):
            assert mixin._should_use_jax(X, 'linear') == True
        
        # With JAX disabled, should not use JAX
        with config_context(enable_jax=False):
            mixin._performance_cache = {}  # Clear cache
            assert mixin._should_use_jax(X, 'linear') == False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
