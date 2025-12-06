"""Proxy system for transparent JAX acceleration."""

# Authors: The JAX-xlearn developers
# SPDX-License-Identifier: BSD-3-Clause

import functools
import warnings
from typing import Any, Type, Optional

from ._config import get_config
from ._accelerator import create_accelerated_estimator
from ._data_conversion import to_numpy, is_jax_array
from ._universal_jax import (
    JAXLinearModelMixin,
    JAXClusterMixin,
    JAXDecompositionMixin,
    create_jax_accelerated_class
)


class EstimatorProxy:
    """Proxy class that transparently switches between JAX and original implementations."""

    def __init__(self, original_class: Type, *args, **kwargs):
        """Initialize the proxy estimator.

        Parameters
        ----------
        original_class : type
            The original jax-sklearn estimator class.
        *args, **kwargs
            Arguments to pass to the estimator constructor.
        """
        self._original_class = original_class
        self._init_args = args
        self._init_kwargs = kwargs
        self._impl = None
        self._using_jax = False

        # Create the actual implementation
        self._create_implementation()

    def _create_implementation(self):
        """Create the underlying implementation (JAX or original)."""
        config = get_config()

        if config["enable_jax"]:
            try:
                # Try to create JAX-accelerated version
                self._impl = create_accelerated_estimator(
                    self._original_class,
                    *self._init_args,
                    **self._init_kwargs
                )

                # Check if we got a JAX implementation
                from ._accelerator import AcceleratorRegistry
                from . import _registry
                accelerated_class = _registry.get_accelerated(self._original_class)
                self._using_jax = (accelerated_class is not None and
                                 isinstance(self._impl, accelerated_class))

            except Exception as e:
                if config["fallback_on_error"]:
                    warnings.warn(
                        f"Failed to create JAX implementation: {e}. "
                        "Using original implementation.",
                        UserWarning
                    )
                    self._impl = self._original_class(*self._init_args, **self._init_kwargs)
                    self._using_jax = False
                else:
                    raise
        else:
            # JAX disabled, use original implementation
            self._impl = self._original_class(*self._init_args, **self._init_kwargs)
            self._using_jax = False

    def _convert_output(self, result):
        """Convert output from JAX to NumPy if needed."""
        if is_jax_array(result):
            return to_numpy(result)
        elif isinstance(result, (list, tuple)):
            return type(result)(
                to_numpy(item) if is_jax_array(item) else item
                for item in result
            )
        return result

    def __getattr__(self, name):
        """Delegate attribute access to the underlying implementation."""
        if name.startswith('_'):
            # Private attributes should be handled by this class
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

        attr = getattr(self._impl, name)

        # If it's a method, wrap it to handle data conversion
        if callable(attr):
            # Capture attr in a closure variable to avoid UnboundLocalError
            captured_attr = attr
            
            @functools.wraps(captured_attr)
            def wrapper(*args, **kwargs):
                try:
                    result = captured_attr(*args, **kwargs)

                    # Convert JAX outputs to NumPy for compatibility
                    if self._using_jax:
                        result = self._convert_output(result)

                    # Return self for chaining methods like fit()
                    if result is self._impl:
                        return self

                    return result

                except Exception as e:
                    config = get_config()
                    if config["fallback_on_error"] and self._using_jax:
                        warnings.warn(
                            f"JAX method {name} failed: {e}. "
                            "Recreating with original implementation.",
                            UserWarning
                        )
                        # Recreate with original implementation
                        self._impl = self._original_class(*self._init_args, **self._init_kwargs)
                        self._using_jax = False

                        # Retry the method call with new implementation
                        retry_attr = getattr(self._impl, name)
                        result = retry_attr(*args, **kwargs)
                        return self if result is self._impl else result
                    else:
                        raise

            return wrapper
        else:
            return attr

    def __setattr__(self, name, value):
        """Handle attribute setting."""
        if name.startswith('_') or name in ('_original_class', '_init_args', '_init_kwargs', '_impl', '_using_jax'):
            # Private attributes of the proxy
            super().__setattr__(name, value)
        else:
            # Delegate to the underlying implementation
            if hasattr(self, '_impl') and self._impl is not None:
                setattr(self._impl, name, value)
            else:
                super().__setattr__(name, value)

    def __repr__(self):
        """String representation of the proxy."""
        if self._impl is not None:
            impl_repr = repr(self._impl)
            if self._using_jax:
                return f"JAX-accelerated {impl_repr}"
            else:
                return impl_repr
        else:
            return f"EstimatorProxy({self._original_class.__name__})"

    def __str__(self):
        """String representation of the proxy."""
        return self.__repr__()

    @property
    def is_using_jax(self) -> bool:
        """Check if the proxy is using JAX acceleration."""
        return self._using_jax

    @property
    def implementation_class(self) -> Type:
        """Get the class of the underlying implementation."""
        return type(self._impl) if self._impl else None


def create_proxy_class(original_class: Type) -> Type:
    """Create a proxy class for an estimator.

    Parameters
    ----------
    original_class : type
        The original xlearn estimator class.

    Returns
    -------
    proxy_class : type
        A proxy class that transparently handles JAX acceleration.
    """
    class ProxyClass(EstimatorProxy):
        def __init__(self, *args, **kwargs):
            super().__init__(original_class, *args, **kwargs)

    # Copy metadata from original class
    ProxyClass.__name__ = original_class.__name__
    ProxyClass.__qualname__ = original_class.__qualname__
    ProxyClass.__module__ = original_class.__module__
    ProxyClass.__doc__ = original_class.__doc__

    return ProxyClass


def create_universal_jax_class(original_class: Type) -> Type:
    """Create a JAX-accelerated version of any xlearn class.

    Parameters
    ----------
    original_class : type
        The original xlearn estimator class.

    Returns
    -------
    jax_class : type
        JAX-accelerated version of the class.
    """
    # Determine the appropriate mixin based on class name/module
    class_name = original_class.__name__
    module_name = original_class.__module__

    # Select appropriate mixin
    if 'linear_model' in module_name or any(keyword in class_name.lower() for keyword in
                                           ['linear', 'regression', 'ridge', 'lasso', 'elastic', 'logistic']):
        mixin_class = JAXLinearModelMixin
    elif 'cluster' in module_name or any(keyword in class_name.lower() for keyword in
                                        ['kmeans', 'cluster', 'dbscan', 'agglomerative']):
        mixin_class = JAXClusterMixin
    elif 'decomposition' in module_name or any(keyword in class_name.lower() for keyword in
                                              ['pca', 'svd', 'nmf', 'ica', 'decomposition']):
        mixin_class = JAXDecompositionMixin
    else:
        # For other algorithms, use the base mixin with minimal acceleration
        mixin_class = JAXLinearModelMixin  # Fallback to basic mixin

    return create_jax_accelerated_class(original_class, mixin_class)


def create_intelligent_proxy(original_class: Type) -> Type:
    """Create an intelligent proxy that automatically creates JAX acceleration.

    Parameters
    ----------
    original_class : type
        The original xlearn estimator class.

    Returns
    -------
    proxy_class : type
        Intelligent proxy class that handles JAX acceleration.
    """
    # First, try to create a JAX-accelerated version
    try:
        jax_class = create_universal_jax_class(original_class)

        # Register it with the accelerator system
        from ._accelerator import AcceleratorRegistry
        from . import _registry
        _registry.register(original_class, jax_class)

    except Exception as e:
        warnings.warn(f"Failed to create JAX acceleration for {original_class.__name__}: {e}")

    # Create and return proxy class
    return create_proxy_class(original_class)


def monkey_patch_estimator(original_class: Type) -> None:
    """Monkey patch an estimator class to use JAX acceleration.

    Parameters
    ----------
    original_class : type
        The original jax-sklearn estimator class to patch.
    """
    # Store the original class
    original_class._original_xlearn_class = original_class

    # Create proxy class
    proxy_class = create_proxy_class(original_class)

    # Replace the class in its module
    module = original_class.__module__
    if module in globals():
        globals()[module].__dict__[original_class.__name__] = proxy_class

    return proxy_class
