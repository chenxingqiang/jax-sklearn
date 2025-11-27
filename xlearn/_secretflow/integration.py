# Author: Chen Xingqiang
# SPDX-License-Identifier: BSD-3-Clause

"""
Integration utilities for connecting jax-sklearn with SecretFlow

Provides automatic adapter creation and mode detection for seamless integration.
"""

from enum import Enum
from typing import Type, Optional


class SecretFlowIntegrationMode(Enum):
    """Integration modes for jax-sklearn with SecretFlow"""
    
    SS = "simple_sealed"          # Simple Sealed - full SPU execution
    FL_PYU = "fl_pyu_accel"       # FL with PYU-side acceleration
    FL_HYBRID = "fl_hybrid"       # FL with hybrid computation
    NATIVE = "native_sf"          # Use native SecretFlow implementation
    AUTO = "auto"                 # Automatic mode detection


def integrate_with_secretflow(
    sklearn_class: Type,
    mode: SecretFlowIntegrationMode = SecretFlowIntegrationMode.AUTO,
    **kwargs
):
    """
    Integrate a scikit-learn/jax-sklearn algorithm with SecretFlow
    
    This function automatically creates an appropriate SecretFlow adapter
    based on the algorithm type and requested integration mode.
    
    Parameters
    ----------
    sklearn_class : Type
        The scikit-learn or jax-sklearn algorithm class to integrate
    mode : SecretFlowIntegrationMode, default=AUTO
        Integration mode to use. If AUTO, automatically detects best mode.
    **kwargs : dict
        Additional arguments passed to the adapter constructor
    
    Returns
    -------
    adapter_class : Type
        SecretFlow-compatible adapter class
    
    Examples
    --------
    >>> from xlearn.cluster import KMeans
    >>> from xlearn._secretflow import integrate_with_secretflow
    >>> 
    >>> # Create SS (Simple Sealed) adapter
    >>> SSKMeans = integrate_with_secretflow(KMeans, mode="ss")
    >>> 
    >>> # Use in SecretFlow
    >>> model = SSKMeans(spu)
    >>> model.fit(vertical_data, n_clusters=10)
    """
    if mode == SecretFlowIntegrationMode.AUTO:
        mode = _detect_best_mode(sklearn_class)
    
    if mode == SecretFlowIntegrationMode.SS:
        return _create_ss_adapter(sklearn_class, **kwargs)
    elif mode == SecretFlowIntegrationMode.FL_PYU:
        return _create_fl_pyu_adapter(sklearn_class, **kwargs)
    elif mode == SecretFlowIntegrationMode.FL_HYBRID:
        return _create_fl_hybrid_adapter(sklearn_class, **kwargs)
    else:
        # Return original class for native mode
        return sklearn_class


def _detect_best_mode(sklearn_class: Type) -> SecretFlowIntegrationMode:
    """
    Automatically detect the best integration mode for an algorithm
    
    Rules:
    - Clustering algorithms: SS mode (data can be aggregated)
    - Classification/Regression: Check if supports partial_fit
        - If yes: FL_PYU mode (incremental learning)
        - If no: SS mode (batch learning)
    """
    class_name = sklearn_class.__name__.lower()
    
    # Clustering algorithms typically use SS mode
    if any(keyword in class_name for keyword in ['cluster', 'kmeans', 'dbscan']):
        return SecretFlowIntegrationMode.SS
    
    # Check if supports incremental learning
    if hasattr(sklearn_class, 'partial_fit'):
        return SecretFlowIntegrationMode.FL_PYU
    
    # Default to SS mode
    return SecretFlowIntegrationMode.SS


def _create_ss_adapter(sklearn_class: Type, **kwargs):
    """Create a Simple Sealed adapter for the algorithm"""
    from .ss_adapter import _ModelBase
    
    class SSAdapter(_ModelBase):
        """Auto-generated SS adapter"""
        
        def __init__(self, spu):
            super().__init__(spu)
            self._sklearn_class = sklearn_class
            self._kwargs = kwargs
        
        def fit(self, x, y=None, **fit_kwargs):
            """Fit in SPU environment"""
            # Convert to SPU dataset
            spu_x = self._to_spu_dataset(x)
            spu_y = self._to_spu(y)[0] if y is not None else None
            
            def _spu_fit(x, y=None):
                model = self._sklearn_class(**self._kwargs)
                if y is not None:
                    model.fit(x, y, **fit_kwargs)
                else:
                    model.fit(x, **fit_kwargs)
                return model
            
            if spu_y is not None:
                self.model = self.spu(_spu_fit)(spu_x, spu_y)
            else:
                self.model = self.spu(_spu_fit)(spu_x)
            
            return self
    
    SSAdapter.__name__ = f"SS{sklearn_class.__name__}"
    SSAdapter.__doc__ = f"Simple Sealed adapter for {sklearn_class.__name__}"
    
    return SSAdapter


def _create_fl_pyu_adapter(sklearn_class: Type, **kwargs):
    """Create a Federated Learning PYU adapter"""
    # TODO: Implement FL PYU adapter
    raise NotImplementedError("FL PYU adapter not yet implemented")


def _create_fl_hybrid_adapter(sklearn_class: Type, **kwargs):
    """Create a hybrid FL adapter"""
    # TODO: Implement FL hybrid adapter
    raise NotImplementedError("FL hybrid adapter not yet implemented")

