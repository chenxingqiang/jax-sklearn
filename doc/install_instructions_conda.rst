Install conda using the
`conda-forge installers <https://conda-forge.org/download/>`__ (no
administrator permission required). Then run:

.. prompt:: bash

  conda create -n xlearn-env -c conda-forge jax-sklearn
  conda activate xlearn-env

In order to check your installation, you can use:

.. prompt:: bash

  conda list jax-sklearn  # show jax-sklearn version and location
  conda list               # show all installed packages in the environment
  python -c "import xlearn; xlearn.show_versions()"
