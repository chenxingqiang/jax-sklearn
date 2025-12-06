"""Universal JAX implementations for common algorithm patterns."""

# Authors: The JAX-xlearn developers
# SPDX-License-Identifier: BSD-3-Clause

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy import linalg
from typing import Any, Dict, Optional, Tuple, Union

from ._config import get_config
from ._data_conversion import to_jax, to_numpy


class UniversalJAXMixin:
    """Mixin class that provides universal JAX acceleration for common operations."""
    
    def __init__(self):
        self._jax_compiled_functions = {}
        self._performance_cache = {}
    
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
            'LogisticRegression': {'min_complexity': 5e7, 'min_samples': 5000},
            
            # Clustering - benefit from vectorization
            'KMeans': {'min_complexity': 1e6, 'min_samples': 5000},
            'kmeans': {'min_complexity': 1e6, 'min_samples': 5000},  # alias
            'DBSCAN': {'min_complexity': 1e6, 'min_samples': 1000},
            
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
    @jax.jit
    def _jax_linear_regression_fit(X: jnp.ndarray, y: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """JAX-compiled linear regression fitting."""
        n_samples, n_features = X.shape
        
        # Center the data
        X_mean = jnp.mean(X, axis=0)
        y_mean = jnp.mean(y)
        X_centered = X - X_mean
        y_centered = y - y_mean
        
        # Solve normal equations: (X^T X) coef = X^T y
        XtX = X_centered.T @ X_centered
        Xty = X_centered.T @ y_centered
        
        # Add small regularization for numerical stability
        regularization = 1e-10 * jnp.trace(XtX) / n_features
        coef = linalg.solve(XtX + regularization * jnp.eye(n_features), Xty)
        
        # Calculate intercept
        intercept = y_mean - X_mean @ coef
        
        return coef, intercept
    
    @staticmethod
    @jax.jit
    def _jax_ridge_regression_fit(X: jnp.ndarray, y: jnp.ndarray, alpha: float) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """JAX-compiled Ridge regression fitting."""
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
    @jax.jit
    def _jax_pca_fit(X: jnp.ndarray, n_components: int) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """JAX-compiled PCA fitting."""
        n_samples, n_features = X.shape
        
        # Center the data
        X_mean = jnp.mean(X, axis=0)
        X_centered = X - X_mean
        
        # Compute SVD
        U, s, Vt = jnp.linalg.svd(X_centered, full_matrices=False)
        
        # Select top components
        components = Vt[:n_components]
        explained_variance = (s[:n_components] ** 2) / (n_samples - 1)
        
        return components, explained_variance, X_mean
    
    @staticmethod
    @jax.jit
    def _jax_kmeans_step(X: jnp.ndarray, centers: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """JAX-compiled K-means iteration step."""
        # Compute distances to all centers
        distances = jnp.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
        
        # Assign points to closest centers
        labels = jnp.argmin(distances, axis=1)
        
        # Update centers
        new_centers = jnp.array([
            jnp.mean(X[labels == k], axis=0) 
            for k in range(centers.shape[0])
        ])
        
        return new_centers, labels
    
    def _apply_jax_linear_regression(self, X: np.ndarray, y: np.ndarray) -> Dict[str, np.ndarray]:
        """Apply JAX-accelerated linear regression."""
        X_jax = to_jax(X)
        y_jax = to_jax(y)
        
        coef_jax, intercept_jax = self._jax_linear_regression_fit(X_jax, y_jax)
        
        return {
            'coef_': to_numpy(coef_jax),
            'intercept_': to_numpy(intercept_jax)
        }
    
    def _apply_jax_ridge_regression(self, X: np.ndarray, y: np.ndarray, alpha: float = 1.0) -> Dict[str, np.ndarray]:
        """Apply JAX-accelerated Ridge regression."""
        X_jax = to_jax(X)
        y_jax = to_jax(y)
        
        coef_jax, intercept_jax = self._jax_ridge_regression_fit(X_jax, y_jax, alpha)
        
        return {
            'coef_': to_numpy(coef_jax),
            'intercept_': to_numpy(intercept_jax)
        }
    
    def _apply_jax_pca(self, X: np.ndarray, n_components: int) -> Dict[str, np.ndarray]:
        """Apply JAX-accelerated PCA."""
        X_jax = to_jax(X)
        
        components_jax, explained_variance_jax, mean_jax = self._jax_pca_fit(X_jax, n_components)
        
        return {
            'components_': to_numpy(components_jax),
            'explained_variance_': to_numpy(explained_variance_jax),
            'mean_': to_numpy(mean_jax)
        }
    
    def _apply_jax_kmeans_iteration(self, X: np.ndarray, centers: np.ndarray) -> Dict[str, np.ndarray]:
        """Apply JAX-accelerated K-means iteration."""
        X_jax = to_jax(X)
        centers_jax = to_jax(centers)
        
        new_centers_jax, labels_jax = self._jax_kmeans_step(X_jax, centers_jax)
        
        return {
            'cluster_centers_': to_numpy(new_centers_jax),
            'labels_': to_numpy(labels_jax)
        }


class JAXLinearModelMixin(UniversalJAXMixin):
    """Mixin for JAX-accelerated linear models."""
    
    def jax_fit(self, X: np.ndarray, y: np.ndarray, algorithm: str = 'linear') -> 'JAXLinearModelMixin':
        """JAX-accelerated fitting for linear models."""
        if not self._should_use_jax(X, algorithm):
            # Fallback to original implementation
            return self._original_fit(X, y)
        
        try:
            if algorithm == 'linear':
                results = self._apply_jax_linear_regression(X, y)
            elif algorithm == 'ridge':
                alpha = getattr(self, 'alpha', 1.0)
                results = self._apply_jax_ridge_regression(X, y, alpha)
            else:
                # Fallback for unsupported algorithms
                return self._original_fit(X, y)
            
            # Set attributes
            for attr_name, attr_value in results.items():
                setattr(self, attr_name, attr_value)
            
            return self
            
        except Exception as e:
            config = get_config()
            if config.get("fallback_on_error", True):
                import warnings
                warnings.warn(f"JAX fitting failed: {e}. Using original implementation.")
                return self._original_fit(X, y)
            else:
                raise


class JAXClusterMixin(UniversalJAXMixin):
    """Mixin for JAX-accelerated clustering algorithms."""
    
    def jax_fit(self, X: np.ndarray) -> 'JAXClusterMixin':
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
            
            for i in range(max_iter):
                results = self._apply_jax_kmeans_iteration(X, centers)
                new_centers = results['cluster_centers_']
                
                # Check convergence
                if np.allclose(centers, new_centers, atol=tol):
                    break
                    
                centers = new_centers
            
            # Set final results
            self.cluster_centers_ = centers
            self.labels_ = results['labels_']
            self.n_iter_ = i + 1
            
            return self
            
        except Exception as e:
            config = get_config()
            if config.get("fallback_on_error", True):
                import warnings
                warnings.warn(f"JAX clustering failed: {e}. Using original implementation.")
                return self._original_fit(X)
            else:
                raise
    
    def _initialize_centers(self, X: np.ndarray, n_clusters: int) -> np.ndarray:
        """Initialize cluster centers."""
        # Simple random initialization - can be improved
        n_samples, n_features = X.shape
        rng = np.random.RandomState(getattr(self, 'random_state', None))
        indices = rng.choice(n_samples, n_clusters, replace=False)
        return X[indices].copy()


class JAXDecompositionMixin(UniversalJAXMixin):
    """Mixin for JAX-accelerated decomposition algorithms."""
    
    def jax_fit(self, X: np.ndarray) -> 'JAXDecompositionMixin':
        """JAX-accelerated fitting for decomposition algorithms."""
        if not self._should_use_jax(X, 'PCA'):
            return self._original_fit(X)
        
        try:
            n_components = getattr(self, 'n_components', min(X.shape))
            results = self._apply_jax_pca(X, n_components)
            
            # Set attributes
            for attr_name, attr_value in results.items():
                setattr(self, attr_name, attr_value)
            
            # Calculate explained variance ratio
            total_var = np.sum(results['explained_variance_'])
            self.explained_variance_ratio_ = results['explained_variance_'] / total_var
            
            return self
            
        except Exception as e:
            config = get_config()
            if config.get("fallback_on_error", True):
                import warnings
                warnings.warn(f"JAX decomposition failed: {e}. Using original implementation.")
                return self._original_fit(X)
            else:
                raise


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
            
            # Store original fit method
            self._original_fit = original_class.fit
        
        def fit(self, X, y=None, **kwargs):
            """Override fit to use JAX acceleration when beneficial."""
            return self.jax_fit(X, y, **kwargs) if y is not None else self.jax_fit(X, **kwargs)
    
    # Copy metadata
    JAXAcceleratedClass.__name__ = f"JAX{original_class.__name__}"
    JAXAcceleratedClass.__qualname__ = f"JAX{original_class.__qualname__}"
    JAXAcceleratedClass.__module__ = original_class.__module__
    
    return JAXAcceleratedClass
