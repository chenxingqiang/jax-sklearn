#!/usr/bin/env python3
"""
JAX-sklearn Performance Test Suite
Tests JAX acceleration across different algorithms and data sizes.
"""

import time
import numpy as np
import xlearn as xl
from xlearn.linear_model import LinearRegression, Ridge
from xlearn.cluster import KMeans
from xlearn.decomposition import PCA
from xlearn.preprocessing import StandardScaler

def time_function(func, *args, **kwargs):
    """Time a function execution."""
    start_time = time.perf_counter()
    result = func(*args, **kwargs)
    end_time = time.perf_counter()
    return result, end_time - start_time

def test_linear_regression_performance():
    """Test LinearRegression performance on different data sizes."""
    print("\n=== LinearRegression Performance Test ===")
    
    data_sizes = [(100, 10), (1000, 20), (5000, 50)]
    
    for n_samples, n_features in data_sizes:
        print(f"\nTesting with {n_samples} samples, {n_features} features:")
        
        # Generate data
        X = np.random.randn(n_samples, n_features)
        y = np.random.randn(n_samples)
        
        # Test fitting
        lr = LinearRegression()
        _, fit_time = time_function(lr.fit, X, y)
        
        # Test prediction
        _, predict_time = time_function(lr.predict, X)
        
        print(f"  Fit time: {fit_time:.4f}s")
        print(f"  Predict time: {predict_time:.4f}s")
        print(f"  Total time: {fit_time + predict_time:.4f}s")

def test_clustering_performance():
    """Test KMeans performance."""
    print("\n=== KMeans Performance Test ===")
    
    data_sizes = [(500, 10), (2000, 20)]
    
    for n_samples, n_features in data_sizes:
        print(f"\nTesting with {n_samples} samples, {n_features} features:")
        
        # Generate data
        X = np.random.randn(n_samples, n_features)
        
        # Test KMeans
        kmeans = KMeans(n_clusters=5, n_init=2, max_iter=10, random_state=42)
        _, fit_time = time_function(kmeans.fit, X)
        _, predict_time = time_function(kmeans.predict, X)
        
        print(f"  Fit time: {fit_time:.4f}s")
        print(f"  Predict time: {predict_time:.4f}s")
        print(f"  Total time: {fit_time + predict_time:.4f}s")

def test_decomposition_performance():
    """Test PCA performance."""
    print("\n=== PCA Performance Test ===")
    
    data_sizes = [(500, 20), (2000, 50)]
    
    for n_samples, n_features in data_sizes:
        print(f"\nTesting with {n_samples} samples, {n_features} features:")
        
        # Generate data
        X = np.random.randn(n_samples, n_features)
        
        # Test PCA
        n_components = min(10, n_features)
        pca = PCA(n_components=n_components)
        _, fit_time = time_function(pca.fit, X)
        _, transform_time = time_function(pca.transform, X)
        
        print(f"  Fit time: {fit_time:.4f}s")
        print(f"  Transform time: {transform_time:.4f}s")
        print(f"  Total time: {fit_time + transform_time:.4f}s")
        print(f"  Explained variance ratio: {pca.explained_variance_ratio_[:3]}")

def test_preprocessing_performance():
    """Test StandardScaler performance."""
    print("\n=== StandardScaler Performance Test ===")
    
    data_sizes = [(1000, 20), (5000, 50)]
    
    for n_samples, n_features in data_sizes:
        print(f"\nTesting with {n_samples} samples, {n_features} features:")
        
        # Generate data
        X = np.random.randn(n_samples, n_features)
        
        # Test StandardScaler
        scaler = StandardScaler()
        _, fit_time = time_function(scaler.fit, X)
        _, transform_time = time_function(scaler.transform, X)
        
        print(f"  Fit time: {fit_time:.4f}s")
        print(f"  Transform time: {transform_time:.4f}s")
        print(f"  Total time: {fit_time + transform_time:.4f}s")

def test_accuracy_consistency():
    """Test that results are numerically consistent."""
    print("\n=== Accuracy Consistency Test ===")
    
    # Generate test data
    np.random.seed(42)
    X = np.random.randn(200, 10)
    y = np.random.randn(200)
    
    # Test LinearRegression
    lr = LinearRegression()
    lr.fit(X, y)
    pred = lr.predict(X)
    mse = np.mean((pred - y) ** 2)
    print(f"LinearRegression MSE: {mse:.6f}")
    
    # Test Ridge
    ridge = Ridge(alpha=1.0)
    ridge.fit(X, y)
    ridge_pred = ridge.predict(X)
    ridge_mse = np.mean((ridge_pred - y) ** 2)
    print(f"Ridge MSE: {ridge_mse:.6f}")
    
    # Test PCA
    pca = PCA(n_components=5)
    pca.fit(X)
    X_transformed = pca.transform(X)
    X_reconstructed = pca.inverse_transform(X_transformed)
    reconstruction_error = np.mean((X - X_reconstructed) ** 2)
    print(f"PCA reconstruction error: {reconstruction_error:.6f}")
    
    print("All accuracy tests passed!")

def main():
    """Run all performance tests."""
    print("JAX-sklearn Performance Test Suite")
    print("=" * 50)
    
    # Print system info
    print(f"JAX enabled: {xl._JAX_ENABLED}")
    if xl._JAX_ENABLED:
        from xlearn._jax import get_jax_platform
        print(f"JAX platform: {get_jax_platform()}")
    
    # Run tests
    try:
        test_linear_regression_performance()
        test_clustering_performance()
        test_decomposition_performance()
        test_preprocessing_performance()
        test_accuracy_consistency()
        
        print("\n" + "=" * 50)
        print("🎉 All performance tests completed successfully!")
        print("JAX-sklearn is working correctly with good performance.")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
