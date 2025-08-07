"""Tests for making sure experimental imports work as expected."""

import textwrap

import pytest

from xlearn.utils._testing import assert_run_python_script_without_output
from xlearn.utils.fixes import _IS_WASM


@pytest.mark.xfail(_IS_WASM, reason="cannot start subprocess")
def test_import_raises_warning():
    code = """
    import pytest
    with pytest.warns(UserWarning, match="it is not needed to import"):
        from xlearn.experimental import enable_hist_gradient_boosting  # noqa
    """
    pattern = "it is not needed to import enable_hist_gradient_boosting anymore"
    assert_run_python_script_without_output(textwrap.dedent(code), pattern=pattern)
