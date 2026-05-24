"""JAX-accelerated SVM predict/decision_function."""

import warnings
import jax
import jax.numpy as jnp
import numpy as np

from ._config import get_config
from ._data_conversion import to_jax, to_numpy


# =============================================================================
# JAX-compiled kernel functions for SVM
# =============================================================================

@jax.jit
def _jax_svm_linear_kernel(X, SV):
    """Linear kernel: X @ SV.T"""
    return X @ SV.T


@jax.jit
def _jax_svm_rbf_kernel(X, SV, gamma):
    """RBF kernel: exp(-gamma * ||X - SV||^2)"""
    X_norm = jnp.sum(X**2, axis=1, keepdims=True)
    SV_norm = jnp.sum(SV**2, axis=1, keepdims=True)
    dist2 = X_norm + SV_norm.T - 2.0 * (X @ SV.T)
    dist2 = jnp.maximum(dist2, 0.0)
    return jnp.exp(-gamma * dist2)


@jax.jit
def _jax_svm_poly_kernel(X, SV, gamma, coef0, degree):
    """Polynomial kernel: (gamma * X @ SV.T + coef0)^degree"""
    return (gamma * (X @ SV.T) + coef0) ** degree


@jax.jit
def _jax_svm_sigmoid_kernel(X, SV, gamma, coef0):
    """Sigmoid kernel: tanh(gamma * X @ SV.T + coef0)"""
    return jnp.tanh(gamma * (X @ SV.T) + coef0)


@jax.jit
def _jax_svm_decision_function(K, dual_coef, intercept):
    """Decision function: K @ dual_coef - intercept"""
    return K @ dual_coef - intercept


@jax.jit
def _jax_svm_predict(K, dual_coef, intercept):
    """Predict: sign of decision function"""
    return jnp.sign(K @ dual_coef - intercept)


@jax.jit
def _jax_svm_rbf_decision(X, SV, dual_coef, intercept, gamma):
    """Combined RBF decision function."""
    K = _jax_svm_rbf_kernel(X, SV, gamma)
    return K @ dual_coef - intercept


# =============================================================================
# JAXSVMMixin
# =============================================================================

class JAXSVMMixin:
    """Mixin for JAX-accelerated SVM predict and decision_function."""

    def __init__(self):
        self._fitted_with_jax = False
        self._svm_kernel_type = None
        self._svm_jax_params = {}

    def _extract_svm_params(self):
        """Extract SVM parameters for JAX from fitted model."""
        if not hasattr(self, 'support_vectors_') or self.support_vectors_ is None:
            return False

        # Get kernel type from the estimator
        kernel = getattr(self, 'kernel', 'rbf')

        self._svm_kernel_type = kernel
        self._svm_jax_params = {
            'dual_coef': to_jax(self.dual_coef_),
            'support_vectors': to_jax(self.support_vectors_),
            'intercept': to_jax(self.intercept_),
            'gamma': float(getattr(self, '_gamma', 1.0)),
            'coef0': float(getattr(self, 'coef0', 0.0)),
            'degree': int(getattr(self, 'degree', 3)),
            'n_support': np.asarray(getattr(self, 'n_support_', [])),
        }

        # For multi-class, build the multi-class decision info
        if hasattr(self, 'n_support_') and len(self.n_support_) > 2:
            # Multi-class: compute pairwise intercept offsets
            pass

        return True

    def jax_fit(self, X, y=None, **kwargs):
        """Fit using original implementation (SMO solver stays in C++)."""
        # SVM fit should always use LIBSVM (no JAX acceleration for training)
        config = get_config()
        result = self._original_fit(X, y, **kwargs) if y is not None else self._original_fit(X, **kwargs)

        # After fitting, extract params for JAX predict
        try:
            self._extract_svm_params()
            self._fitted_with_jax = True
        except Exception:
            self._fitted_with_jax = False

        return result

    def jax_predict(self, X):
        """JAX-accelerated predict."""
        config = get_config()
        if not self._fitted_with_jax:
            return self._original_predict(X) if hasattr(self, '_original_predict') else None

        try:
            from sklearn.utils.validation import validate_data
            X = validate_data(self, X, ensure_2d=True, dtype='numeric', reset=False)
            X_jax = to_jax(X)

            dec = self._jax_decision_function_impl(X_jax)
            pred = jnp.sign(dec)

            # Convert to original class labels
            pred_np = to_numpy(pred).astype(np.int32)

            # Map binary {-1, 1} to original classes if needed
            if hasattr(self, 'classes_') and len(self.classes_) == 2:
                pred_np = np.where(pred_np == -1, self.classes_[0], self.classes_[1])
            elif hasattr(self, 'classes_'):
                # Multi-class: argmax of decision values
                if dec.ndim > 1 and dec.shape[1] > 1:
                    from sklearn.utils.multiclass import _ovr_decision_function
                    dec_np = to_numpy(dec)
                    pred_np = self.classes_[dec_np.argmax(axis=1)]

            return pred_np

        except Exception as e:
            if config.get("fallback_on_error", True):
                warnings.warn(f"JAX SVM predict failed: {e}. Using original.", UserWarning)
                return self._original_predict(X)
            raise

    def jax_decision_function(self, X):
        """JAX-accelerated decision_function."""
        config = get_config()
        if not self._fitted_with_jax:
            original_dec = getattr(self, '_original_decision_function', None)
            if original_dec:
                return original_dec(X)
            return self._original_predict(X)

        try:
            from sklearn.utils.validation import validate_data
            X = validate_data(self, X, ensure_2d=True, dtype='numeric', reset=False)
            X_jax = to_jax(X)
            dec = self._jax_decision_function_impl(X_jax)
            return to_numpy(dec)
        except Exception as e:
            if config.get("fallback_on_error", True):
                warnings.warn(f"JAX SVM decision_function failed: {e}. Using original.", UserWarning)
                original_dec = getattr(self, '_original_decision_function', None)
                if original_dec:
                    return original_dec(X)
                return self._original_predict(X)
            raise

    def _jax_decision_function_impl(self, X_jax):
        """Compute decision function with JAX based on kernel type."""
        kernel = self._svm_kernel_type
        params = self._svm_jax_params
        SV = params['support_vectors']
        dual_coef = params['dual_coef']
        intercept = params['intercept']
        gamma = params['gamma']

        if kernel == 'linear':
            K = _jax_svm_linear_kernel(X_jax, SV)
        elif kernel == 'rbf':
            K = _jax_svm_rbf_kernel(X_jax, SV, gamma)
        elif kernel == 'poly':
            K = _jax_svm_poly_kernel(X_jax, SV, gamma, params['coef0'], params['degree'])
        elif kernel == 'sigmoid':
            K = _jax_svm_sigmoid_kernel(X_jax, SV, gamma, params['coef0'])
        else:
            K = _jax_svm_rbf_kernel(X_jax, SV, gamma)

        # Multi-class OVA or binary
        if dual_coef.ndim == 2 and dual_coef.shape[1] > 1:
            # OVA decision values for each class
            decisions = []
            for i in range(dual_coef.shape[1]):
                d = K @ dual_coef[:, i:i+1] - intercept[i] if intercept.ndim > 0 else K @ dual_coef[:, i:i+1]
                decisions.append(d)
            return jnp.concatenate(decisions, axis=1)
        else:
            return K @ dual_coef - intercept
