"""Tests for making sure experimental imports work as expected."""

import textwrap

import pytest

from xlearn.utils._testing import assert_run_python_script_without_output
from xlearn.utils.fixes import _IS_WASM


@pytest.mark.xfail(_IS_WASM, reason="cannot start subprocess")
def test_imports_strategies():
    # Make sure different import strategies work or fail as expected.

    # Since Python caches the imported modules, we need to run a child process
    # for every test case. Else, the tests would not be independent
    # (manually removing the imports from the cache (sys.modules) is not
    # recommended and can lead to many complications).
    pattern = "Halving(Grid|Random)SearchCV is experimental"
    good_import = """
    from xlearn.experimental import enable_halving_search_cv
    from xlearn.model_selection import HalvingGridSearchCV
    from xlearn.model_selection import HalvingRandomSearchCV
    """
    assert_run_python_script_without_output(
        textwrap.dedent(good_import), pattern=pattern
    )

    good_import_with_model_selection_first = """
    import xlearn.model_selection
    from xlearn.experimental import enable_halving_search_cv
    from xlearn.model_selection import HalvingGridSearchCV
    from xlearn.model_selection import HalvingRandomSearchCV
    """
    assert_run_python_script_without_output(
        textwrap.dedent(good_import_with_model_selection_first),
        pattern=pattern,
    )

    bad_imports = f"""
    import pytest

    with pytest.raises(ImportError, match={pattern!r}):
        from xlearn.model_selection import HalvingGridSearchCV

    import xlearn.experimental
    with pytest.raises(ImportError, match={pattern!r}):
        from xlearn.model_selection import HalvingRandomSearchCV
    """
    assert_run_python_script_without_output(
        textwrap.dedent(bad_imports),
        pattern=pattern,
    )
