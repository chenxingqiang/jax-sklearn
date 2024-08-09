#!/bin/bash

set -e
set -x

PYTHON_VERSION=$1

docker container run \
    --rm jax-ml/minimal-windows \
    powershell -Command "python -c 'import xlearn; xlearn.show_versions()'"

docker container run \
    -e XLEARN_SKIP_NETWORK_TESTS=1 \
    --rm jax-ml/minimal-windows \
    powershell -Command "pytest --pyargs xlearn"
