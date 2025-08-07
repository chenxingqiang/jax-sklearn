"""JAX-accelerated Linear Regression implementation."""

# Authors: The JAX-xlearn developers
# SPDX-License-Identifier: BSD-3-Clause

import warnings
import numpy as np
from typing import Optional, Union

from ._config import get_config
from ._data_conversion import to_jax, to_numpy, convert_input_data
from ._accelerator import accelerated_estimator

# Import original LinearRegression for fallback
from ..linear_model import LinearRegression as _OriginalLinearRegression

# JAX imports (conditional)
try:
    import jax
    import jax.numpy as jnp
    from jax import lax
    _JAX_AVAILABLE = True
except ImportError:
    jax = None
    jnp = None
    lax = None
    _JAX_AVAILABLE = False


@jax.jit if _JAX_AVAILABLE else lambda x: x
def _solve_normal_equations(X, y, alpha=0.0):
    """Solve normal equations using JAX.
    
    Solves: (X.T @ X + alpha * I) @ coef = X.T @ y
    """
    XtX = jnp.dot(X.T, X)
    if alpha > 0:
        # Add regularization
        XtX += alpha * jnp.eye(XtX.shape[0])
    
    Xty = jnp.dot(X.T, y)
    
    # Use JAX's solve for numerical stability
    return lax.linalg.solve(XtX, Xty)


@jax.jit if _JAX_AVAILABLE else lambda x: x  
def _predict_jax(X, coef, intercept):
    """JAX prediction function."""
    prediction = jnp.dot(X, coef)
    if intercept is not None:
        prediction += intercept
    return prediction


@accelerated_estimator(_OriginalLinearRegression)
class LinearRegressionJAX:
    """JAX-accelerated Linear Regression.
    
    This is a drop-in replacement for xlearn.linear_model.LinearRegression
    that uses JAX for acceleration while maintaining full API compatibility.
    
    Parameters
    ----------
    fit_intercept : bool, default=True
        Whether to calculate the intercept for this model.
        
    copy_X : bool, default=True
        If True, X will be copied; else, it may be overwritten.
        
    n_jobs : int, default=None
        Number of jobs to use for computation. Ignored in JAX implementation.
        
    positive : bool, default=False
        When set to True, forces coefficients to be positive. 
        Not yet implemented in JAX version.
    """
    
    def __init__(
        self,
        *,
        fit_intercept=True,
        copy_X=True,
        n_jobs=None,
        positive=False,
    ):
        self.fit_intercept = fit_intercept
        self.copy_X = copy_X
        self.n_jobs = n_jobs
        self.positive = positive
        
        # JAX-specific attributes
        self._is_fitted = False
        self.coef_ = None
        self.intercept_ = None
        self.n_features_in_ = None
        self.feature_names_in_ = None
        
        # Check if JAX is available
        if not _JAX_AVAILABLE:
            raise ImportError(
                "JAX is not available. Install JAX to use JAX-accelerated LinearRegression: "
                "pip install jax jaxlib"
            )
        
        # Warn about unsupported features
        if positive:
            warnings.warn(
                "The 'positive' parameter is not yet supported in the JAX implementation. "
                "It will be ignored.",
                UserWarning
            )
    
    def fit(self, X, y, sample_weight=None):
        """Fit linear model.
        
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training data.
            
        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target values.
            
        sample_weight : array-like of shape (n_samples,), default=None
            Individual weights for each sample. Not yet supported in JAX version.
            
        Returns
        -------
        self : object
            Fitted estimator.
        """
        # Check for unsupported features
        if sample_weight is not None:
            warnings.warn(
                "sample_weight is not yet supported in the JAX implementation. "
                "It will be ignored.",
                UserWarning
            )
        
        # Convert and validate input data
        config = get_config()
        X, y = convert_input_data(X, y, to_jax=True, dtype=config["precision"])
        
        # Store input information
        self.n_features_in_ = X.shape[1]
        if hasattr(X, 'columns'):
            self.feature_names_in_ = np.array(X.columns, dtype=object)
        
        # Handle 1D y
        y = jnp.atleast_1d(y)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        
        # Center data if fitting intercept
        if self.fit_intercept:
            X_mean = jnp.mean(X, axis=0)
            y_mean = jnp.mean(y, axis=0)
            X_centered = X - X_mean
            y_centered = y - y_mean
        else:
            X_centered = X
            y_centered = y
            X_mean = None
            y_mean = None
        
        # Solve normal equations
        try:
            self.coef_ = _solve_normal_equations(X_centered, y_centered)
            
            # Calculate intercept
            if self.fit_intercept:
                self.intercept_ = y_mean - jnp.dot(X_mean, self.coef_)
            else:
                self.intercept_ = jnp.zeros(y.shape[1])
            
            # Convert back to NumPy for compatibility
            self.coef_ = to_numpy(self.coef_)
            self.intercept_ = to_numpy(self.intercept_)
            
            # Handle single target case
            if self.coef_.shape[1] == 1:
                self.coef_ = self.coef_.ravel()
                self.intercept_ = self.intercept_.item()
            
            self._is_fitted = True
            return self
            
        except Exception as e:
            config = get_config()
            if config["fallback_on_error"]:
                warnings.warn(
                    f"JAX fitting failed: {e}. Falling back to original implementation.",
                    UserWarning
                )
                # Create fallback estimator
                fallback = _OriginalLinearRegression(
                    fit_intercept=self.fit_intercept,
                    copy_X=self.copy_X,
                    n_jobs=self.n_jobs,
                    positive=self.positive,
                )
                fallback.fit(to_numpy(X), to_numpy(y.ravel() if y.shape[1] == 1 else y), sample_weight)
                
                # Copy attributes
                self.coef_ = fallback.coef_
                self.intercept_ = fallback.intercept_
                self.n_features_in_ = fallback.n_features_in_
                if hasattr(fallback, 'feature_names_in_'):
                    self.feature_names_in_ = fallback.feature_names_in_
                
                self._is_fitted = True
                return self
            else:
                raise
    
    def predict(self, X):
        """Predict using the linear model.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.
            
        Returns
        -------
        C : array of shape (n_samples,) or (n_samples, n_targets)
            Returns predicted values.
        """
        if not self._is_fitted:
            raise ValueError("This LinearRegressionJAX instance is not fitted yet.")
        
        # Convert input to JAX
        config = get_config()
        X = to_jax(X, dtype=config["precision"])
        
        # Convert coefficients to JAX
        coef_jax = to_jax(self.coef_)
        if coef_jax.ndim == 1:
            coef_jax = coef_jax.reshape(-1, 1)
        
        intercept_jax = to_jax(self.intercept_) if self.intercept_ is not None else None
        
        # Make prediction
        try:
            prediction = _predict_jax(X, coef_jax, intercept_jax)
            
            # Convert back to NumPy
            prediction = to_numpy(prediction)
            
            # Handle single target case
            if prediction.shape[1] == 1:
                prediction = prediction.ravel()
            
            return prediction
            
        except Exception as e:
            config = get_config()
            if config["fallback_on_error"]:
                warnings.warn(
                    f"JAX prediction failed: {e}. Using NumPy fallback.",
                    UserWarning
                )
                # NumPy fallback
                X_np = to_numpy(X)
                prediction = np.dot(X_np, self.coef_)
                if self.intercept_ is not None:
                    prediction += self.intercept_
                return prediction
            else:
                raise
    
    def score(self, X, y, sample_weight=None):
        """Return the coefficient of determination R^2 of the prediction.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
            
        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            True values for X.
            
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. Not yet supported in JAX version.
            
        Returns
        -------
        score : float
            R^2 score.
        """
        if sample_weight is not None:
            warnings.warn(
                "sample_weight is not yet supported in JAX R^2 calculation. "
                "It will be ignored.",
                UserWarning
            )
        
        # Make prediction
        y_pred = self.predict(X)
        
        # Convert to JAX for computation
        y_true = to_jax(y)
        y_pred = to_jax(y_pred)
        
        # Calculate R^2 using JAX
        ss_res = jnp.sum((y_true - y_pred) ** 2)
        ss_tot = jnp.sum((y_true - jnp.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        
        return float(to_numpy(r2))
    
    def _more_tags(self):
        """Get tags for this estimator."""
        return {
            'requires_positive_X': False,
            'requires_positive_y': False,
            'requires_fit': True,
            'preserves_dtype': [np.float64, np.float32],
            'allow_nan': False,
            'stateless': False,
            'binary_only': False,
            '_xfail_checks': {},
            'poor_score': False,
            'multiclass_only': False,
            'multioutput_only': False,
            'multilabel': False,
            'multioutput': True,
            'X_types': ['2darray'],
            'y_types': ['1dlabels', '2dlabels'],
            'jax_accelerated': True,  # Custom tag
        }
