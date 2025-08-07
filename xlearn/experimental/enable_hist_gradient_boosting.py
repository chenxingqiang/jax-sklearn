"""This is now a no-op and can be safely removed from your code.

It used to enable the use of
:class:`~xlearn.ensemble.HistGradientBoostingClassifier` and
:class:`~xlearn.ensemble.HistGradientBoostingRegressor` when they were still
:term:`experimental`, but these estimators are now stable and can be imported
normally from `xlearn.ensemble`.
"""

# Authors: The jax-sklearn developers
# SPDX-License-Identifier: BSD-3-Clause

# Don't remove this file, we don't want to break users code just because the
# feature isn't experimental anymore.

import warnings

warnings.warn(
    "Since version 1.0, "
    "it is not needed to import enable_hist_gradient_boosting anymore. "
    "HistGradientBoostingClassifier and HistGradientBoostingRegressor are now "
    "stable and can be normally imported from xlearn.ensemble."
)
