"""Support vector machine algorithms."""

# See http://jax-sklearn.sourceforge.net/modules/svm.html for complete
# documentation.

# Authors: The jax-sklearn developers
# SPDX-License-Identifier: BSD-3-Clause

from ._bounds import l1_min_c
from ._classes import SVC, SVR, LinearSVC, LinearSVR, NuSVC, NuSVR, OneClassSVM

__all__ = [
    "SVC",
    "SVR",
    "LinearSVC",
    "LinearSVR",
    "NuSVC",
    "NuSVR",
    "OneClassSVM",
    "l1_min_c",
]
