#!/bin/bash

set -e
set -x

PYTHON_VERSION=$1
PROJECT_DIR=$2

python $PROJECT_DIR/build_tools/wheels/check_license.py

FREE_THREADED_BUILD="$(python -c"import sysconfig; print(bool(sysconfig.get_config_var('Py_GIL_DISABLED')))")"

if [[ $FREE_THREADED_BUILD == "False" ]]; then
    # Run the tests for the jax-sklearn wheel in a minimal Windows environment
    # without any developer runtime libraries installed to ensure that it does not
    # implicitly rely on the presence of the DLLs of such runtime libraries.
    docker container run \
        --rm jax-sklearn/minimal-windows \
        powershell -Command "python -c 'import xlearn; xlearn.show_versions()'"

    docker container run \
        -e XLEARN_SKIP_NETWORK_TESTS=1 \
        --rm jax-sklearn/minimal-windows \
        powershell -Command "pytest --pyargs xlearn"
else
    # This is too cumbersome to use a Docker image in the free-threaded case
    export PYTHON_GIL=0
    python -c "import xlearn; xlearn.show_versions()"
    pytest --pyargs xlearn
fi
