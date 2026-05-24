"""JAX-accelerated Gaussian Process implementation."""

import warnings
from numbers import Integral, Real

import jax
import jax.numpy as jnp
import numpy as np
import scipy.optimize
from jax.scipy import linalg
from operator import itemgetter

from ._config import get_config
from ._data_conversion import to_jax, to_numpy


# =============================================================================
# JAX-compiled kernel functions
# =============================================================================

@jax.jit
def _jax_rbf_kernel(X, Y, length_scale):
    """JAX-compiled RBF kernel: exp(-0.5 * ||x-y||^2 / l^2)"""
    X_norm = jnp.sum(X**2, axis=1, keepdims=True)
    Y_norm = jnp.sum(Y**2, axis=1, keepdims=True)
    dist2 = X_norm + Y_norm.T - 2.0 * (X @ Y.T)
    dist2 = jnp.maximum(dist2, 0.0)
    return jnp.exp(-0.5 * dist2 / (length_scale**2))


@jax.jit
def _jax_matern_kernel(X, Y, length_scale, nu=1.5):
    """JAX-compiled Matern kernel with nu in {0.5, 1.5, 2.5}."""
    X_norm = jnp.sum(X**2, axis=1, keepdims=True)
    Y_norm = jnp.sum(Y**2, axis=1, keepdims=True)
    dist2 = X_norm + Y_norm.T - 2.0 * (X @ Y.T)
    dist2 = jnp.maximum(dist2, 0.0)
    d = jnp.sqrt(dist2) / length_scale

    sqrt3 = jnp.sqrt(3.0)
    sqrt5 = jnp.sqrt(5.0)
    cond_05 = d == d  # always true, but structure the branch
    cond_15 = True
    # Use select-style for JIT compatibility
    return jnp.where(nu == 0.5, jnp.exp(-d),
                     jnp.where(nu == 1.5,
                               (1.0 + sqrt3 * d) * jnp.exp(-sqrt3 * d),
                               (1.0 + sqrt5 * d + (5.0 / 3.0) * dist2 / (length_scale**2)) * jnp.exp(-sqrt5 * d)))


@jax.jit
def _jax_white_kernel_diag(X, noise_level):
    return jnp.full(X.shape[0], noise_level**2)


# =============================================================================
# Kernel structure detection and evaluation
# =============================================================================

def _detect_kernel_structure(kernel):
    """Convert a sklearn kernel into a JAX-compatible tuple representation."""
    name = kernel.__class__.__name__
    if name == 'RBF':
        return ('RBF', float(kernel.length_scale))
    elif name == 'Matern':
        return ('Matern', float(kernel.length_scale), float(kernel.nu))
    elif name == 'ConstantKernel':
        return ('ConstantKernel', float(kernel.constant_value))
    elif name == 'WhiteKernel':
        return ('WhiteKernel', float(kernel.noise_level))
    elif name == 'DotProduct':
        return ('DotProduct', float(kernel.sigma_0) if hasattr(kernel, 'sigma_0') else 0.0)
    elif name == 'Sum':
        return ('Sum', _detect_kernel_structure(kernel.k1),
                _detect_kernel_structure(kernel.k2))
    elif name == 'Product':
        return ('Product', _detect_kernel_structure(kernel.k1),
                _detect_kernel_structure(kernel.k2))
    elif name == 'Exponentiation':
        return ('Exponentiation', _detect_kernel_structure(kernel.kernel),
                float(kernel.exponent))
    else:
        return ('RBF', 1.0)


def _eval_kernel_structure(struct, X, Y, diag=False):
    """Evaluate a kernel structure on (X, Y) using JAX.

    Parameters
    ----------
    struct : tuple
        Kernel structure from _detect_kernel_structure.
    X, Y : jnp.ndarray
        Input arrays.
    diag : bool, default=False
        If True, only compute the diagonal (for kernel.diag(X)).

    Returns
    -------
    K : jnp.ndarray
        Kernel matrix or diagonal.
    """
    kernel_type = struct[0]

    if kernel_type == 'Sum':
        return (_eval_kernel_structure(struct[1], X, Y, diag) +
                _eval_kernel_structure(struct[2], X, Y, diag))
    elif kernel_type == 'Product':
        return (_eval_kernel_structure(struct[1], X, Y, diag) *
                _eval_kernel_structure(struct[2], X, Y, diag))
    elif kernel_type == 'Exponentiation':
        base = _eval_kernel_structure(struct[1], X, Y, diag)
        return base ** struct[2]

    if diag:
        if kernel_type in ('RBF', 'Matern', 'Sum', 'Product'):
            return jnp.ones(X.shape[0])
        elif kernel_type == 'ConstantKernel':
            return jnp.full(X.shape[0], struct[1]**2)
        elif kernel_type == 'WhiteKernel':
            return _jax_white_kernel_diag(X, struct[1])
        elif kernel_type == 'DotProduct':
            return jnp.sum(X**2, axis=1) + struct[1]**2
        else:
            return jnp.ones(X.shape[0])
    else:
        if kernel_type == 'RBF':
            return _jax_rbf_kernel(X, Y, struct[1])
        elif kernel_type == 'Matern':
            return _jax_matern_kernel(X, Y, struct[1], struct[2] if len(struct) > 2 else 1.5)
        elif kernel_type == 'ConstantKernel':
            cv = struct[1]
            return jnp.full((X.shape[0], Y.shape[0]), cv**2)
        elif kernel_type == 'WhiteKernel':
            nl = struct[1]
            K = jnp.zeros((X.shape[0], Y.shape[0]))
            if X.shape[0] == Y.shape[0]:
                K = K.at[jnp.arange(X.shape[0]), jnp.arange(X.shape[0])].set(nl**2)
            return K
        elif kernel_type == 'DotProduct':
            return X @ Y.T + struct[1]**2
        else:
            return X @ Y.T


# =============================================================================
# JAX-compiled GPR core operations
# =============================================================================

@jax.jit
def _jax_gpr_fit_core(K, y, alpha_val):
    """JIT-compiled GPR core: add noise, cholesky, solve for alpha.

    Returns (L, alpha, log_likelihood).
    """
    n = K.shape[0]
    K_reg = K + alpha_val * jnp.eye(n)
    L = jax.scipy.linalg.cholesky(K_reg, lower=True)
    alpha = jax.scipy.linalg.cho_solve((L, True), y)

    # log-marginal likelihood: -0.5*y^T*alpha - sum(log(diag(L))) - n/2*log(2*pi)
    y_t = y.squeeze() if y.ndim > 1 and y.shape[1] == 1 else y
    log_likelihood = -0.5 * jnp.dot(y_t.T, alpha).squeeze() if y_t.ndim == 1 else -0.5 * jnp.einsum('ik,ik->k', y_t, alpha).sum()
    log_likelihood -= jnp.sum(jnp.log(jnp.diag(L)))
    log_likelihood -= n / 2.0 * jnp.log(2.0 * jnp.pi)

    return L, alpha, log_likelihood


@jax.jit
def _jax_gpr_predict_core(K_trans, alpha, L, K_diag):
    """JIT-compiled GPR predict: mean and std."""
    y_mean = K_trans @ alpha
    V = jax.scipy.linalg.solve_triangular(L, K_trans.T, lower=True)
    var = K_diag - jnp.sum(V**2, axis=0)
    var = jnp.maximum(var, 0.0)
    return y_mean, jnp.sqrt(var)


# =============================================================================
# JAXGaussianProcessMixin
# =============================================================================

class JAXGaussianProcessMixin:
    """Mixin for JAX-accelerated Gaussian Process computations."""

    def __init__(self):
        self._kernel_struct = None
        self._X_train_jax = None
        self._fitted_with_jax = False

    def jax_fit(self, X, y=None, **kwargs):
        """JAX-accelerated fit for GaussianProcessRegressor."""
        config = get_config()
        algorithm = 'GaussianProcessRegressor'
        if not self._should_use_jax(X, algorithm):
            return self._original_fit(X, y, **kwargs) if y is not None else self._original_fit(X, **kwargs)

        # Full JAX-accelerated fit
        if y is None:
            return self._original_fit(X, **kwargs)

        # Step 1: Validation and normalization via original class methods
        try:
            return self._jax_fit_impl(X, y, **kwargs)
        except Exception as e:
            if config.get("fallback_on_error", True):
                warnings.warn(f"JAX GP fit failed: {e}. Using original.", UserWarning)
                return self._original_fit(X, y, **kwargs)
            raise

    def _jax_fit_impl(self, X, y, **kwargs):
        """JAX GP fit implementation."""
        # Setup kernel
        from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
        from sklearn.base import clone
        from sklearn.utils import check_random_state
        from sklearn.utils._param_validation import validate_params
        from sklearn.utils.validation import validate_data

        if self.kernel is None:
            self.kernel_ = C() * RBF()
        else:
            self.kernel_ = clone(self.kernel)

        self._rng = check_random_state(self.random_state)

        # Validate data
        X, y = validate_data(self, X, y, multi_output=True, y_numeric=True,
                             ensure_2d=True, dtype='numeric')

        # Normalize y
        if self.normalize_y:
            self._y_train_mean = np.mean(y, axis=0)
            self._y_train_std = np.std(y, axis=0)
            self._y_train_std = np.where(self._y_train_std == 0, 1.0, self._y_train_std)
            y = (y - self._y_train_mean) / self._y_train_std
        else:
            shape = (y.shape[1],) if y.ndim == 2 else 1
            self._y_train_mean = np.zeros(shape=shape)
            self._y_train_std = np.ones(shape=shape)

        # Handle alpha
        alpha_val = float(self.alpha) if np.ndim(self.alpha) == 0 else float(self.alpha[0])

        self.X_train_ = np.copy(X) if self.copy_X_train else X
        self.y_train_ = np.copy(y) if self.copy_X_train else y

        # Detect kernel structure for JAX
        self._kernel_struct = _detect_kernel_structure(self.kernel_)

        # Convert to JAX
        X_jax = to_jax(X)
        y_jax = to_jax(y)

        # Compute kernel matrix with JAX
        K = _eval_kernel_structure(self._kernel_struct, X_jax, X_jax)

        # JIT-compiled core: Cholesky + solve + log-likelihood
        L, alpha, log_likelihood = _jax_gpr_fit_core(K, y_jax, alpha_val)

        # Convert back
        self.L_ = to_numpy(L)
        self.alpha_ = to_numpy(alpha)
        self.log_marginal_likelihood_value_ = to_numpy(log_likelihood)
        self._X_train_jax = X_jax
        self._fitted_with_jax = True

        # Optimize kernel hyperparameters (if needed)
        if self.optimizer is not None and self.kernel_.n_dims > 0:
            def obj_func(theta, eval_gradient=True):
                if eval_gradient:
                    lml, grad = self.log_marginal_likelihood(theta, eval_gradient=True, clone_kernel=False)
                    return -lml, -grad
                else:
                    return -self.log_marginal_likelihood(theta, clone_kernel=False)

            optima = [(self._constrained_optimization(obj_func, self.kernel_.theta, self.kernel_.bounds))]

            if self.n_restarts_optimizer > 0:
                import numpy as np
                if not np.isfinite(self.kernel_.bounds).all():
                    raise ValueError("Multiple optimizer restarts requires all bounds finite.")
                bounds = self.kernel_.bounds
                for iteration in range(self.n_restarts_optimizer):
                    theta_initial = self._rng.uniform(bounds[:, 0], bounds[:, 1])
                    optima.append(self._constrained_optimization(obj_func, theta_initial, bounds))

            lml_values = list(map(itemgetter(1), optima))
            self.kernel_.theta = optima[np.argmin(lml_values)][0]
            self.kernel_._check_bounds_params()
            self.log_marginal_likelihood_value_ = -np.min(lml_values)

            # Recompute L and alpha with optimized hyperparameters
            self._kernel_struct = _detect_kernel_structure(self.kernel_)
            K = _eval_kernel_structure(self._kernel_struct, X_jax, X_jax)
            L, alpha, log_likelihood = _jax_gpr_fit_core(K, y_jax, alpha_val)
            self.L_ = to_numpy(L)
            self.alpha_ = to_numpy(alpha)
            self.log_marginal_likelihood_value_ = to_numpy(log_likelihood)

        return self

    def jax_predict(self, X, return_std=False, return_cov=False):
        """JAX-accelerated predict for GaussianProcessRegressor."""
        config = get_config()
        if not self._fitted_with_jax:
            return (self._original_predict(X, return_std=return_std, return_cov=return_cov)
                    if hasattr(self, '_original_predict') else None)

        try:
            from sklearn.utils.validation import validate_data
            X = validate_data(self, X, ensure_2d=True, dtype='numeric', reset=False)

            X_jax = to_jax(X)
            K_trans = _eval_kernel_structure(self._kernel_struct, X_jax, self._X_train_jax)
            L_jax = to_jax(self.L_)
            alpha_jax = to_jax(self.alpha_)

            # Predict mean
            y_mean = K_trans @ alpha_jax
            y_mean = to_numpy(self._y_train_std * y_mean + self._y_train_mean)
            if y_mean.ndim > 1 and y_mean.shape[1] == 1:
                y_mean = np.squeeze(y_mean, axis=1)

            if return_cov or return_std:
                K_diag = _eval_kernel_structure(self._kernel_struct, X_jax, X_jax, diag=True)
                y_mean_jax, y_std_jax = _jax_gpr_predict_core(K_trans, alpha_jax, L_jax, K_diag)
                y_std_np = to_numpy(y_std_jax)

                if return_cov:
                    K_xx = _eval_kernel_structure(self._kernel_struct, X_jax, X_jax)
                    V = jax.scipy.linalg.solve_triangular(L_jax, K_trans.T, lower=True)
                    y_cov = to_numpy(K_xx - V.T @ V)
                    y_cov = np.outer(y_cov, self._y_train_std**2).reshape(*y_cov.shape, -1)
                    if y_cov.shape[2] == 1:
                        y_cov = np.squeeze(y_cov, axis=2)
                    return y_mean, y_cov

                if return_std:
                    y_var = np.outer(y_std_np**2, self._y_train_std**2).reshape(*y_std_np.shape, -1)
                    if y_var.shape[1] == 1:
                        y_var = np.squeeze(y_var, axis=1)
                    return y_mean, np.sqrt(y_var)

            return y_mean

        except Exception as e:
            if config.get("fallback_on_error", True):
                warnings.warn(f"JAX GP predict failed: {e}. Using original.", UserWarning)
                return self._original_predict(X, return_std=return_std, return_cov=return_cov)
            raise

    def jax_log_marginal_likelihood(self, theta=None, eval_gradient=False, clone_kernel=True):
        """JAX-accelerated log-marginal likelihood computation."""
        if theta is None:
            if eval_gradient:
                raise ValueError("Gradient can only be evaluated for theta!=None")
            return getattr(self, 'log_marginal_likelihood_value_', -np.inf)

        # The kernel's n_dims check and bounds are in the original class
        # This is called during optimization where scipy needs the gradient
        # For full JAX gradient support, we'd need JAX-compatible gradient kernels
        # For now, delegate to the original implementation
        if hasattr(self, '_original_log_marginal_likelihood'):
            return self._original_log_marginal_likelihood(theta, eval_gradient, clone_kernel)
        return -np.inf

    def _constrained_optimization(self, obj_func, initial_theta, bounds):
        """Optimize using scipy (external loop, each iteration calls JIT-compiled code)."""
        if self.optimizer == "fmin_l_bfgs_b":
            opt_res = scipy.optimize.minimize(
                obj_func, initial_theta, method="L-BFGS-B", jac=True, bounds=bounds,
            )
            from sklearn.utils.optimize import _check_optimize_result
            _check_optimize_result("lbfgs", opt_res)
            return opt_res.x, opt_res.fun
        elif callable(self.optimizer):
            return self.optimizer(obj_func, initial_theta, bounds=bounds)
        raise ValueError(f"Unknown optimizer {self.optimizer}.")
