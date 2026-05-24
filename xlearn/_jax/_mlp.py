"""JAX-accelerated Neural Network (MLP) implementation."""

import warnings
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from ._config import get_config
from ._data_conversion import to_jax, to_numpy


# =============================================================================
# JAX-compiled MLP operations
# =============================================================================

def _get_jax_activation(name):
    """Get JAX activation function by name."""
    activations = {
        'identity': lambda x: x,
        'logistic': lambda x: jax.nn.sigmoid(x),
        'tanh': lambda x: jnp.tanh(x),
        'relu': lambda x: jax.nn.relu(x),
        'softmax': lambda x: jax.nn.softmax(x, axis=1),
    }
    return activations.get(name, lambda x: x)


def _get_jax_loss(name):
    """Get JAX loss function."""
    if name == 'log_loss':
        return lambda y_true, y_pred: -jnp.mean(
            y_true * jnp.log(jnp.clip(y_pred, 1e-15, 1.0)) +
            (1 - y_true) * jnp.log(jnp.clip(1 - y_pred, 1e-15, 1.0))
        )
    elif name == 'squared_error':
        return lambda y_true, y_pred: 0.5 * jnp.mean((y_true - y_pred)**2)
    else:
        return lambda y_true, y_pred: 0.5 * jnp.mean((y_true - y_pred)**2)


@partial(jax.jit, static_argnums=(2, 3, 4))
def _jax_mlp_forward(params, X, activation_names, n_layers, n_outputs):
    """JAX-compiled forward pass through all layers.

    params: list of (W, b) tuples for each layer
    X: input array (n_samples, n_features)
    activation_names: list of activation names per layer
    """
    h = X
    for i in range(n_layers):
        W, b = params[i]
        act = _get_jax_activation(activation_names[i])
        h = act(h @ W + b)
    return h


@partial(jax.jit, static_argnums=(4, 5, 6))
def _jax_mlp_loss_and_grad(params, X, y, alpha, activation_names, n_layers, loss_name):
    """JAX-compiled loss and gradient computation."""
    def loss_fn(params):
        y_pred = _jax_mlp_forward(params, X, activation_names, n_layers, y.shape[1] if y.ndim > 1 else 1)
        loss_fn_impl = _get_jax_loss(loss_name)
        base_loss = loss_fn_impl(y, y_pred)

        # L2 regularization
        reg_loss = 0.0
        for i in range(n_layers):
            W, b = params[i]
            reg_loss += 0.5 * alpha * jnp.sum(W**2)

        return base_loss + reg_loss

    loss, grads = jax.value_and_grad(loss_fn)(params)
    return loss, grads


@partial(jax.jit, static_argnums=(5, 6, 7))
def _jax_mlp_train_step_sgd(params, vel, X, y, lr, activation_names, n_layers, loss_name, alpha, momentum):
    """Single SGD training step with momentum."""
    loss, grads = _jax_mlp_loss_and_grad(params, X, y, alpha, activation_names, n_layers, loss_name)

    new_params = []
    new_vel = []
    for i in range(n_layers):
        W_grad, b_grad = grads[i]
        W, b = params[i]
        v_W, v_b = vel[i]

        new_v_W = momentum * v_W + lr * W_grad
        new_v_b = momentum * v_b + lr * b_grad
        new_W = W - new_v_W
        new_b = b - new_v_b

        new_params.append((new_W, new_b))
        new_vel.append((new_v_W, new_v_b))

    return new_params, new_vel, loss


@partial(jax.jit, static_argnums=(2, 3, 4))
def _jax_mlp_predict(params, X, activation_names, n_layers, n_outputs):
    """JAX-compiled MLP predict."""
    return _jax_mlp_forward(params, X, activation_names, n_layers, n_outputs)


# =============================================================================
# JAXNeuralNetworkMixin
# =============================================================================

class JAXNeuralNetworkMixin:
    """Mixin for JAX-accelerated MLPClassifier/MLPRegressor."""

    def __init__(self):
        self._nn_jax_params = None
        self._nn_jax_vel = None
        self._nn_activation_names = None
        self._fitted_with_jax = False

    def _extract_nn_params(self):
        """Extract MLP parameters into JAX-compatible format."""
        if not hasattr(self, 'coefs_') or not self.coefs_:
            return False

        params = []
        for i in range(len(self.coefs_)):
            W = to_jax(self.coefs_[i])
            b = to_jax(self.intercepts_[i])
            params.append((W, b))

        self._nn_jax_params = params

        # Activation functions per layer
        activation = getattr(self, 'activation', 'relu')
        n_layers = len(self.coefs_)
        # All hidden layers use the specified activation, output uses identity/softmax
        self._nn_activation_names = [activation] * (n_layers - 1) + ['identity']

        # Loss function
        if hasattr(self, '_loss_name'):
            self._nn_loss_name = {
                'log_loss': 'log_loss',
                'squared_error': 'squared_error',
            }.get(self._loss_name, 'squared_error')
        else:
            self._nn_loss_name = 'squared_error'

        return True

    def _init_jax_velocities(self):
        """Initialize velocity arrays for SGD momentum."""
        vel = []
        for W, b in self._nn_jax_params:
            vel.append((jnp.zeros_like(W), jnp.zeros_like(b)))
        self._nn_jax_vel = vel

    def jax_fit(self, X, y=None, **kwargs):
        """JAX-accelerated fit for MLP."""
        config = get_config()
        algorithm = 'MLP'
        if not self._should_use_jax(X, algorithm):
            return self._original_fit(X, y, **kwargs) if y is not None else self._original_fit(X, **kwargs)

        if y is None:
            return self._original_fit(X, **kwargs)

        # For now, delegate to original fit (MLP training is complex with early stopping, etc.)
        # The JAX acceleration is primarily for predict after fitting
        result = self._original_fit(X, y, **kwargs) if y is not None else self._original_fit(X, **kwargs)

        try:
            self._extract_nn_params()
            self._fitted_with_jax = True
        except Exception:
            self._fitted_with_jax = False

        return result

    def jax_predict(self, X):
        """JAX-accelerated predict for MLP."""
        config = get_config()
        if not self._fitted_with_jax:
            return self._original_predict(X) if hasattr(self, '_original_predict') else None

        try:
            from sklearn.utils.validation import validate_data
            X = validate_data(self, X, ensure_2d=True, dtype='numeric', reset=False)
            X_jax = to_jax(X)

            n_layers = len(self._nn_jax_params)
            y_pred = _jax_mlp_predict(
                self._nn_jax_params, X_jax, self._nn_activation_names,
                n_layers, self.n_outputs_ if hasattr(self, 'n_outputs_') else 1
            )

            pred_np = to_numpy(y_pred)

            # For classifiers, return class labels
            if hasattr(self, 'classes_'):
                return self.classes_[pred_np.argmax(axis=1)]
            return pred_np

        except Exception as e:
            if config.get("fallback_on_error", True):
                warnings.warn(f"JAX MLP predict failed: {e}. Using original.", UserWarning)
                return self._original_predict(X)
            raise

    def jax_predict_proba(self, X):
        """JAX-accelerated predict_proba for MLPClassifier."""
        config = get_config()
        if not self._fitted_with_jax or not hasattr(self, 'classes_'):
            original = getattr(self, '_original_predict_proba', None)
            if original:
                return original(X)
            return self._original_predict(X)

        try:
            from sklearn.utils.validation import validate_data
            X = validate_data(self, X, ensure_2d=True, dtype='numeric', reset=False)
            X_jax = to_jax(X)

            n_layers = len(self._nn_jax_params)
            # Last layer should use softmax for classification
            self._nn_activation_names[-1] = 'softmax'
            y_pred = _jax_mlp_predict(
                self._nn_jax_params, X_jax, self._nn_activation_names,
                n_layers, len(self.classes_)
            )
            return to_numpy(y_pred)

        except Exception as e:
            if config.get("fallback_on_error", True):
                warnings.warn(f"JAX MLP predict_proba failed: {e}. Using original.", UserWarning)
                original = getattr(self, '_original_predict_proba', None)
                if original:
                    return original(X)
                return self._original_predict(X)
            raise

    def jax_predict_log_proba(self, X):
        """JAX-accelerated predict_log_proba for MLPClassifier."""
        proba = self.jax_predict_proba(X)
        return np.log(proba)
