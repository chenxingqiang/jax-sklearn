"""JAX-accelerated DecisionTree/RandomForest predict."""

import warnings
import jax
import jax.numpy as jnp
import numpy as np

from ._config import get_config
from ._data_conversion import to_jax, to_numpy


# =============================================================================
# JAX-compiled tree walk
# =============================================================================

@jax.jit
def _jax_tree_predict_single(X, children_left, children_right, feature, threshold,
                               value, missing_go_to_left, n_classes):
    """JAX-compiled single tree predict using while_loop.

    All samples walk the tree simultaneously until reaching leaves.
    """
    n_samples = X.shape[0]
    node_ids = jnp.zeros(n_samples, dtype=jnp.int32)
    n_nodes = children_left.shape[0]

    def cond_fun(state):
        node_ids = state
        # Check if any sample is not at a leaf
        not_leaf = (children_left[node_ids] != -1)
        return jnp.any(not_leaf)

    def body_fun(state):
        node_ids = state
        # Get current node info
        feat = feature[node_ids]
        thresh = threshold[node_ids]
        is_leaf = (children_left[node_ids] == -1)

        # Gather feature values
        feat_clipped = jnp.clip(feat, 0, X.shape[1] - 1)
        x_val = X[jnp.arange(n_samples), feat_clipped]

        # Decide direction
        go_left = jnp.where(
            jnp.isnan(x_val),
            missing_go_to_left[node_ids],
            x_val <= thresh
        )

        # Update node ids (only for non-leaf nodes)
        new_ids = jnp.where(
            is_leaf,
            node_ids,
            jnp.where(go_left, children_left[node_ids], children_right[node_ids])
        )
        return new_ids

    node_ids = jax.lax.while_loop(cond_fun, body_fun, node_ids)

    # Gather leaf values (n_samples, n_outputs, n_classes) -> squeeze to (n_samples,)
    if value.ndim == 3:
        leaf_values = value[node_ids]  # (n_samples, n_outputs, n_classes)
        if n_classes <= 1:
            # Regression: return single value
            return leaf_values[:, 0, 0]
        else:
            # Classification: return class probabilities
            return leaf_values[:, 0, :]
    else:
        return value[node_ids]


@jax.jit
def _jax_tree_predict_proba_single(X, children_left, children_right, feature,
                                    threshold, value, missing_go_to_left, n_classes):
    """JAX-compiled tree predict_proba."""
    leaf_values = _jax_tree_predict_single(
        X, children_left, children_right, feature, threshold,
        value, missing_go_to_left, n_classes
    )
    if n_classes > 1:
        # Normalize to probabilities
        probs = leaf_values / jnp.maximum(jnp.sum(leaf_values, axis=1, keepdims=True), 1e-15)
        return probs
    return leaf_values


# =============================================================================
# JAXTreePredictorMixin
# =============================================================================

class JAXTreePredictorMixin:
    """Mixin for JAX-accelerated DecisionTree/RandomForest predict."""

    def __init__(self):
        self._tree_jax_data = {}
        self._fitted_with_jax = False

    def _extract_tree_data(self, tree):
        """Extract tree data into JAX arrays."""
        data = {
            'children_left': to_jax(np.asarray(tree.children_left, dtype=np.int32)),
            'children_right': to_jax(np.asarray(tree.children_right, dtype=np.int32)),
            'feature': to_jax(np.asarray(tree.feature, dtype=np.int32)),
            'threshold': to_jax(np.asarray(tree.threshold, dtype=np.float32)),
            'value': to_jax(np.asarray(tree.value, dtype=np.float32)),
            'missing_go_to_left': to_jax(np.asarray(tree.missing_go_to_left, dtype=np.bool_)),
            'n_classes': tree.n_classes[0] if hasattr(tree, 'n_classes') else 1,
            'n_outputs': tree.n_outputs if hasattr(tree, 'n_outputs') else 1,
        }
        return data

    def _extract_forest_data(self):
        """Extract all tree data from a forest."""
        if not hasattr(self, 'estimators_') or not self.estimators_:
            return False

        trees_data = []
        for est in self.estimators_:
            if hasattr(est, 'tree_'):
                trees_data.append(self._extract_tree_data(est.tree_))
        self._forest_tree_data = trees_data
        return len(trees_data) > 0

    def _extract_single_tree_data(self):
        """Extract data from a single DecisionTree."""
        if hasattr(self, 'tree_'):
            self._tree_jax_data = self._extract_tree_data(self.tree_)
            return True
        return False

    def jax_fit(self, X, y=None, **kwargs):
        """Fit using original implementation. Extract tree data for JAX predict."""
        config = get_config()
        result = self._original_fit(X, y, **kwargs) if y is not None else self._original_fit(X, **kwargs)

        try:
            if hasattr(self, 'estimators_'):
                self._extract_forest_data()
            elif hasattr(self, 'tree_'):
                self._extract_single_tree_data()
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

            if hasattr(self, 'estimators_') and self._forest_tree_data:
                # Forest predict: average across trees
                pred_sum = None
                n_trees = len(self._forest_tree_data)

                for td in self._forest_tree_data:
                    pred = _jax_tree_predict_single(
                        X_jax, td['children_left'], td['children_right'],
                        td['feature'], td['threshold'], td['value'],
                        td['missing_go_to_left'], td['n_classes']
                    )
                    if pred_sum is None:
                        pred_sum = pred
                    else:
                        pred_sum = pred_sum + pred

                pred_mean = pred_sum / n_trees
                return to_numpy(pred_mean)

            elif self._tree_jax_data:
                td = self._tree_jax_data
                pred = _jax_tree_predict_single(
                    X_jax, td['children_left'], td['children_right'],
                    td['feature'], td['threshold'], td['value'],
                    td['missing_go_to_left'], td['n_classes']
                )
                return to_numpy(pred)

            return self._original_predict(X)

        except Exception as e:
            if config.get("fallback_on_error", True):
                warnings.warn(f"JAX tree predict failed: {e}. Using original.", UserWarning)
                return self._original_predict(X)
            raise

    def jax_predict_proba(self, X):
        """JAX-accelerated predict_proba."""
        config = get_config()
        if not self._fitted_with_jax:
            original = getattr(self, '_original_predict_proba', None)
            if original:
                return original(X)
            return self._original_predict(X)

        try:
            from sklearn.utils.validation import validate_data
            X = validate_data(self, X, ensure_2d=True, dtype='numeric', reset=False)
            X_jax = to_jax(X)

            if hasattr(self, 'estimators_') and self._forest_tree_data:
                # Forest predict_proba: average probabilities
                proba_sum = None
                n_trees = len(self._forest_tree_data)

                for td in self._forest_tree_data:
                    proba = _jax_tree_predict_proba_single(
                        X_jax, td['children_left'], td['children_right'],
                        td['feature'], td['threshold'], td['value'],
                        td['missing_go_to_left'], td['n_classes']
                    )
                    if proba_sum is None:
                        proba_sum = proba
                    else:
                        proba_sum = proba_sum + proba

                proba_mean = proba_sum / n_trees
                return to_numpy(proba_mean)

            elif self._tree_jax_data:
                td = self._tree_jax_data
                proba = _jax_tree_predict_proba_single(
                    X_jax, td['children_left'], td['children_right'],
                    td['feature'], td['threshold'], td['value'],
                    td['missing_go_to_left'], td['n_classes']
                )
                return to_numpy(proba)

            original = getattr(self, '_original_predict_proba', None)
            if original:
                return original(X)
            return self._original_predict(X)

        except Exception as e:
            if config.get("fallback_on_error", True):
                warnings.warn(f"JAX tree predict_proba failed: {e}. Using original.", UserWarning)
                original = getattr(self, '_original_predict_proba', None)
                if original:
                    return original(X)
                return self._original_predict(X)
            raise
