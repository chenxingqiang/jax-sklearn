"""
The :mod:`xlearn._loss` module includes loss function classes suitable for
fitting classification and regression tasks.
"""

# Authors: The jax-sklearn developers
# SPDX-License-Identifier: BSD-3-Clause

from .loss import (
    AbsoluteError,
    HalfBinomialLoss,
    HalfGammaLoss,
    HalfMultinomialLoss,
    HalfPoissonLoss,
    HalfSquaredError,
    HalfTweedieLoss,
    HalfTweedieLossIdentity,
    HuberLoss,
    PinballLoss,
)

__all__ = [
    "AbsoluteError",
    "HalfBinomialLoss",
    "HalfGammaLoss",
    "HalfMultinomialLoss",
    "HalfPoissonLoss",
    "HalfSquaredError",
    "HalfTweedieLoss",
    "HalfTweedieLossIdentity",
    "HuberLoss",
    "PinballLoss",
]
