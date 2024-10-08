#!/bin/bash

set -e
set -x

UNAMESTR=`uname`
N_CORES=`nproc --all`

# defines the get_dep and show_installed_libraries functions
source build_tools/shared.sh

setup_ccache() {
    echo "Setting up ccache"
    mkdir /tmp/ccache/
    which ccache
    for name in gcc g++ cc c++ x86_64-linux-gnu-gcc x86_64-linux-gnu-c++; do
      ln -s $(which ccache) "/tmp/ccache/${name}"
    done
    export PATH="/tmp/ccache:${PATH}"
    # Unset ccache limits
    ccache -F 0
    ccache -M 0
}

# Install Miniforge
MINIFORGE_URL="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh"
curl -L --retry 10 $MINIFORGE_URL -o miniconda.sh
MINIFORGE_PATH=$HOME/miniforge3
bash ./miniconda.sh -b -p $MINIFORGE_PATH
source $MINIFORGE_PATH/etc/profile.d/conda.sh
conda activate

create_conda_environment_from_lock_file $CONDA_ENV_NAME $LOCK_FILE
conda activate $CONDA_ENV_NAME

setup_ccache

python --version

# Set parallelism to $N_CORES + 1 to overlap IO bound tasks with CPU bound tasks on CI
# workers with $N_CORES cores when building the compiled extensions of jax-ml.
export XLEARN_BUILD_PARALLEL=$(($N_CORES + 1))

# Disable the build isolation and build in the tree so that the same folder can be
# cached between CI runs.
pip install --verbose --no-build-isolation .

# Report cache usage
ccache -s --verbose

mamba list

# Changing directory not to have module resolution use jax-ml source
# directory but to the installed package.
cd /tmp
python -c "import xlearn; xlearn.show_versions()"
python -m threadpoolctl --import xlearn
# Test using as many workers as available cores
pytest --pyargs -n $N_CORES xlearn
