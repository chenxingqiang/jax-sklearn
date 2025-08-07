"""Data conversion utilities for JAX acceleration."""

# Authors: The JAX-xlearn developers
# SPDX-License-Identifier: BSD-3-Clause

import functools
import numpy as np
from typing import Any, Union, Tuple, Optional

from ._config import get_config

# Type hints
ArrayLike = Union[np.ndarray, Any]  # Any for JAX arrays when available

def _get_jax_modules():
    """Get JAX modules if available."""
    try:
        import jax
        import jax.numpy as jnp
        return jax, jnp
    except ImportError:
        return None, None

def is_jax_array(data: Any) -> bool:
    """Check if data is a JAX array."""
    jax, jnp = _get_jax_modules()
    if jax is None:
        return False
    return isinstance(data, jnp.ndarray)

def is_numpy_array(data: Any) -> bool:
    """Check if data is a NumPy array."""
    return isinstance(data, np.ndarray)

def to_jax(data: ArrayLike, dtype: Optional[str] = None) -> Any:
    """Convert data to JAX array.

    Parameters
    ----------
    data : array-like
        Input data to convert.
    dtype : str, optional
        Target dtype. If None, uses current configuration.

    Returns
    -------
    jax_array
        JAX array representation of the data.
    """
    jax, jnp = _get_jax_modules()
    if jax is None:
        raise ImportError("JAX is not available")

    config = get_config()
    if dtype is None:
        dtype = config["precision"]

    # Handle different input types
    if is_jax_array(data):
        # Already a JAX array, just ensure correct dtype
        if data.dtype != dtype:
            return data.astype(dtype)
        return data
    elif is_numpy_array(data):
        # NumPy array
        return jnp.asarray(data, dtype=dtype)
    elif hasattr(data, '__array__'):
        # Array-like (pandas, etc.)
        np_array = np.asarray(data)
        return jnp.asarray(np_array, dtype=dtype)
    else:
        # Try to convert to numpy first
        try:
            np_array = np.asarray(data)
            return jnp.asarray(np_array, dtype=dtype)
        except Exception as e:
            raise ValueError(f"Cannot convert data to JAX array: {e}")

def to_numpy(data: ArrayLike) -> np.ndarray:
    """Convert data to NumPy array.

    Parameters
    ----------
    data : array-like
        Input data to convert.

    Returns
    -------
    numpy_array : np.ndarray
        NumPy array representation of the data.
    """
    if is_jax_array(data):
        # JAX array - convert to NumPy
        return np.asarray(data)
    elif is_numpy_array(data):
        # Already NumPy
        return data
    else:
        # Try to convert to NumPy
        return np.asarray(data)

def ensure_2d(data: ArrayLike) -> ArrayLike:
    """Ensure data is 2D array.

    Parameters
    ----------
    data : array-like
        Input data.

    Returns
    -------
    data_2d : array-like
        2D version of the data.
    """
    if is_jax_array(data):
        jax, jnp = _get_jax_modules()
        if data.ndim == 1:
            return jnp.expand_dims(data, axis=0)
        return data
    else:
        data = np.asarray(data)
        if data.ndim == 1:
            return data.reshape(1, -1)
        return data

def get_array_module(data: ArrayLike):
    """Get the appropriate array module for the data.

    Parameters
    ----------
    data : array-like
        Input data.

    Returns
    -------
    module
        Either numpy or jax.numpy depending on the data type.
    """
    if is_jax_array(data):
        _, jnp = _get_jax_modules()
        return jnp
    else:
        return np

def convert_input_data(*arrays, **kwargs) -> Tuple[ArrayLike, ...]:
    """Convert input data arrays to appropriate format.

    Parameters
    ----------
    *arrays : array-like
        Input arrays to convert.
    to_jax : bool, default=True
        Whether to convert to JAX arrays.
    dtype : str, optional
        Target dtype.

    Returns
    -------
    converted_arrays : tuple
        Tuple of converted arrays.
    """
    to_jax_flag = kwargs.get('to_jax', True)
    dtype = kwargs.get('dtype', None)

    config = get_config()
    if not config["enable_jax"] or not to_jax_flag:
        # Convert to NumPy
        return tuple(to_numpy(arr) for arr in arrays)

    # Convert to JAX
    try:
        return tuple(to_jax(arr, dtype=dtype) for arr in arrays)
    except Exception as e:
        if config["fallback_on_error"]:
            return tuple(to_numpy(arr) for arr in arrays)
        else:
            raise

def auto_convert_arrays(to_jax_arrays: bool = True):
    """Decorator for automatic array conversion.

    Parameters
    ----------
    to_jax_arrays : bool, default=True
        Whether to convert arrays to JAX format.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            config = get_config()

            # Skip conversion if JAX is disabled
            if not config["enable_jax"] or not to_jax_arrays:
                return func(*args, **kwargs)

            try:
                # Convert array arguments
                converted_args = []
                for arg in args:
                    if isinstance(arg, (np.ndarray, list, tuple)) and not isinstance(arg, str):
                        try:
                            converted_args.append(to_jax(arg))
                        except Exception:
                            converted_args.append(arg)
                    else:
                        converted_args.append(arg)

                # Convert array keyword arguments
                converted_kwargs = {}
                for key, value in kwargs.items():
                    if isinstance(value, (np.ndarray, list, tuple)) and not isinstance(value, str):
                        try:
                            converted_kwargs[key] = to_jax(value)
                        except Exception:
                            converted_kwargs[key] = value
                    else:
                        converted_kwargs[key] = value

                # Call function with converted arrays
                result = func(*converted_args, **converted_kwargs)

                # Convert result back to NumPy for compatibility
                if is_jax_array(result):
                    return to_numpy(result)
                elif isinstance(result, (list, tuple)):
                    return type(result)(
                        to_numpy(item) if is_jax_array(item) else item
                        for item in result
                    )

                return result

            except Exception as e:
                if config["fallback_on_error"]:
                    # Fall back to original function with original arguments
                    return func(*args, **kwargs)
                else:
                    raise

        return wrapper
    return decorator

class DataConverter:
    """Utility class for data conversion operations."""

    @staticmethod
    def validate_input(X, y=None, ensure_2d=True, dtype=None):
        """Validate and convert input data.

        Parameters
        ----------
        X : array-like
            Input features.
        y : array-like, optional
            Target values.
        ensure_2d : bool, default=True
            Whether to ensure X is 2D.
        dtype : str, optional
            Target dtype.

        Returns
        -------
        X_converted : array-like
            Converted features.
        y_converted : array-like or None
            Converted targets (if provided).
        """
        config = get_config()

        if dtype is None:
            dtype = config["precision"]

        # Convert X
        if config["enable_jax"]:
            try:
                X = to_jax(X, dtype=dtype)
                if ensure_2d and X.ndim == 1:
                    jax, jnp = _get_jax_modules()
                    X = jnp.expand_dims(X, axis=0)
            except Exception as e:
                if config["fallback_on_error"]:
                    X = to_numpy(X).astype(dtype)
                    if ensure_2d and X.ndim == 1:
                        X = X.reshape(1, -1)
                else:
                    raise
        else:
            X = to_numpy(X).astype(dtype)
            if ensure_2d and X.ndim == 1:
                X = X.reshape(1, -1)

        # Convert y if provided
        if y is not None:
            if config["enable_jax"]:
                try:
                    y = to_jax(y, dtype=dtype)
                except Exception as e:
                    if config["fallback_on_error"]:
                        y = to_numpy(y).astype(dtype)
                    else:
                        raise
            else:
                y = to_numpy(y).astype(dtype)

        return X, y
