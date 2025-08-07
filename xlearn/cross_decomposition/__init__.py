"""Algorithms for cross decomposition."""

# Authors: The jax-sklearn developers
# SPDX-License-Identifier: BSD-3-Clause

from ._pls import CCA, PLSSVD, PLSCanonical, PLSRegression

__all__ = ["CCA", "PLSSVD", "PLSCanonical", "PLSRegression"]
