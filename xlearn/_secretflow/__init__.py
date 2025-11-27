# Author: Chen Xingqiang
# SPDX-License-Identifier: BSD-3-Clause

"""
SecretFlow Integration Layer for JAX-sklearn

This module provides seamless integration between jax-sklearn and SecretFlow,
enabling privacy-preserving machine learning with JAX acceleration.

Integration Modes:
- SS (Simple Sealed): Algorithms running on SPU with full privacy protection
- FL (Federated Learning): Local computation acceleration with data remaining in PYUs
- HYBRID: Automatic mode selection based on algorithm and data characteristics
"""

from .ss_adapter import SSKMeans, SSGaussianNB, SSKNNClassifier
from .integration import integrate_with_secretflow, SecretFlowIntegrationMode

__all__ = [
    # SS Algorithms
    'SSKMeans',
    'SSGaussianNB',
    'SSKNNClassifier',
    # Integration utilities
    'integrate_with_secretflow',
    'SecretFlowIntegrationMode',
]

