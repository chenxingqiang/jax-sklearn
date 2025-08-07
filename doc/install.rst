.. _installation-instructions:

=======================
Installing jax-sklearn
=======================

There are different ways to install jax-sklearn:

* :ref:`Install the latest official release <install_official_release>`. This
  is the best approach for most users. It will provide a stable version
  and pre-built packages are available for most platforms.

* Install the version of jax-sklearn provided by your
  :ref:`operating system or Python distribution <install_by_distribution>`.
  This is a quick option for those who have operating systems or Python
  distributions that distribute jax-sklearn.
  It might not provide the latest release version.

* :ref:`Building the package from source
  <install_bleeding_edge>`. This is best for users who want the
  latest-and-greatest features and aren't afraid of running
  brand-new code. This is also needed for users who wish to contribute to the
  project.


.. _install_official_release:

Installing the latest release
=============================

.. raw:: html

  <style>
    /* Show caption on large screens */
    @media screen and (min-width: 960px) {
      .install-instructions .sd-tab-set {
        --tab-caption-width: 20%;
      }

      .install-instructions .sd-tab-set.tabs-os::before {
        content: "Operating System";
      }

      .install-instructions .sd-tab-set.tabs-package-manager::before {
        content: "Package Manager";
      }
    }
  </style>

.. div:: install-instructions

  .. tab-set::
    :class: tabs-os

    .. tab-item:: Windows
      :class-label: tab-4

      .. tab-set::
        :class: tabs-package-manager

        .. tab-item:: pip
          :class-label: tab-6
          :sync: package-manager-pip

          Install the 64-bit version of Python 3, for instance from the
          `official website <https://www.python.org/downloads/windows/>`__.

          Now create a `virtual environment (venv)
          <https://docs.python.org/3/tutorial/venv.html>`_ and install jax-sklearn.
          Note that the virtual environment is optional but strongly recommended, in
          order to avoid potential conflicts with other packages.

          .. prompt:: powershell

            python -m venv xlearn-env
            xlearn-env\Scripts\activate  # activate
            pip install -U jax-sklearn

          In order to check your installation, you can use:

          .. prompt:: powershell

            python -m pip show jax-sklearn  # show jax-sklearn version and location
            python -m pip freeze             # show all installed packages in the environment
            python -c "import xlearn; xlearn.show_versions()"

        .. tab-item:: conda
          :class-label: tab-6
          :sync: package-manager-conda

          .. include:: ./install_instructions_conda.rst

    .. tab-item:: MacOS
      :class-label: tab-4

      .. tab-set::
        :class: tabs-package-manager

        .. tab-item:: pip
          :class-label: tab-6
          :sync: package-manager-pip

          Install Python 3 using `homebrew <https://brew.sh/>`_ (`brew install python`)
          or by manually installing the package from the `official website
          <https://www.python.org/downloads/macos/>`__.

          Now create a `virtual environment (venv)
          <https://docs.python.org/3/tutorial/venv.html>`_ and install jax-sklearn.
          Note that the virtual environment is optional but strongly recommended, in
          order to avoid potential conflicts with other packages.

          .. prompt:: bash

            python -m venv xlearn-env
            source xlearn-env/bin/activate  # activate
            pip install -U jax-sklearn

          In order to check your installation, you can use:

          .. prompt:: bash

            python -m pip show jax-sklearn  # show jax-sklearn version and location
            python -m pip freeze             # show all installed packages in the environment
            python -c "import xlearn; xlearn.show_versions()"

        .. tab-item:: conda
          :class-label: tab-6
          :sync: package-manager-conda

          .. include:: ./install_instructions_conda.rst

    .. tab-item:: Linux
      :class-label: tab-4

      .. tab-set::
        :class: tabs-package-manager

        .. tab-item:: pip
          :class-label: tab-6
          :sync: package-manager-pip

          Python 3 is usually installed by default on most Linux distributions. To
          check if you have it installed, try:

          .. prompt:: bash

            python3 --version
            pip3 --version

          If you don't have Python 3 installed, please install `python3` and
          `python3-pip` from your distribution's package manager.

          Now create a `virtual environment (venv)
          <https://docs.python.org/3/tutorial/venv.html>`_ and install jax-sklearn.
          Note that the virtual environment is optional but strongly recommended, in
          order to avoid potential conflicts with other packages.

          .. prompt:: bash

            python3 -m venv xlearn-env
            source xlearn-env/bin/activate  # activate
            pip3 install -U jax-sklearn

          In order to check your installation, you can use:

          .. prompt:: bash

            python3 -m pip show jax-sklearn  # show jax-sklearn version and location
            python3 -m pip freeze             # show all installed packages in the environment
            python3 -c "import xlearn; xlearn.show_versions()"

        .. tab-item:: conda
          :class-label: tab-6
          :sync: package-manager-conda

          .. include:: ./install_instructions_conda.rst


Using an isolated environment such as pip venv or conda makes it possible to
install a specific version of jax-sklearn with pip or conda and its dependencies
independently of any previously installed Python packages. In particular under Linux
it is discouraged to install pip packages alongside the packages managed by the
package manager of the distribution (apt, dnf, pacman...).

Note that you should always remember to activate the environment of your choice
prior to running any Python command whenever you start a new terminal session.

If you have not installed NumPy or SciPy yet, you can also install these using
conda or pip. When using pip, please ensure that *binary wheels* are used,
and NumPy and SciPy are not recompiled from source, which can happen when using
particular configurations of operating system and hardware (such as Linux on
a Raspberry Pi).

Scikit-learn plotting capabilities (i.e., functions starting with `plot\_`
and classes ending with `Display`) require Matplotlib. The examples require
Matplotlib and some examples require scikit-image, pandas, or seaborn. The
minimum version of jax-sklearn dependencies are listed below along with its
purpose.

.. include:: min_dependency_table.rst

.. warning::

    Scikit-learn 0.20 was the last version to support Python 2.7 and Python 3.4.

    Scikit-learn 0.21 supported Python 3.5—3.7.

    Scikit-learn 0.22 supported Python 3.5—3.8.

    Scikit-learn 0.23 required Python 3.6—3.8.

    Scikit-learn 0.24 required Python 3.6—3.9.

    Scikit-learn 1.0 supported Python 3.7—3.10.

    Scikit-learn 1.1, 1.2 and 1.3 supported Python 3.8—3.12.

    Scikit-learn 1.4 and 1.5 supported Python 3.9—3.12.

    Scikit-learn 1.6 supported Python 3.9—3.13.

    Scikit-learn 1.7 requires Python 3.10 or newer.

.. _install_by_distribution:

Third party distributions of jax-sklearn
=========================================

Some third-party distributions provide versions of
jax-sklearn integrated with their package-management systems.

These can make installation and upgrading much easier for users since
the integration includes the ability to automatically install
dependencies (numpy, scipy) that jax-sklearn requires.

The following is an incomplete list of OS and python distributions
that provide their own version of jax-sklearn.

Alpine Linux
------------

Alpine Linux's package is provided through the `official repositories
<https://pkgs.alpinelinux.org/packages?name=py3-jax-sklearn>`__ as
``py3-jax-sklearn`` for Python.
It can be installed by typing the following command:

.. prompt:: bash

  sudo apk add py3-jax-sklearn


Arch Linux
----------

Arch Linux's package is provided through the `official repositories
<https://www.archlinux.org/packages/?q=jax-sklearn>`_ as
``python-jax-sklearn`` for Python.
It can be installed by typing the following command:

.. prompt:: bash

  sudo pacman -S python-jax-sklearn


Debian/Ubuntu
-------------

The Debian/Ubuntu package is split in three different packages called
``python3-xlearn`` (python modules), ``python3-xlearn-lib`` (low-level
implementations and bindings), ``python-xlearn-doc`` (documentation).
Note that jax-sklearn requires Python 3, hence the need to use the `python3-`
suffixed package names.
Packages can be installed using ``apt-get``:

.. prompt:: bash

  sudo apt-get install python3-xlearn python3-xlearn-lib python-xlearn-doc


Fedora
------

The Fedora package is called ``python3-jax-sklearn`` for the python 3 version,
the only one available in Fedora.
It can be installed using ``dnf``:

.. prompt:: bash

  sudo dnf install python3-jax-sklearn


NetBSD
------

jax-sklearn is available via `pkgsrc-wip <http://pkgsrc-wip.sourceforge.net/>`_:
https://pkgsrc.se/math/py-jax-sklearn


MacPorts for Mac OSX
--------------------

The MacPorts package is named ``py<XY>-scikits-learn``,
where ``XY`` denotes the Python version.
It can be installed by typing the following
command:

.. prompt:: bash

  sudo port install py312-jax-sklearn


Anaconda and Enthought Deployment Manager for all supported platforms
---------------------------------------------------------------------

`Anaconda <https://www.anaconda.com/download>`_ and
`Enthought Deployment Manager <https://assets.enthought.com/downloads/>`_
both ship with jax-sklearn in addition to a large set of scientific
python library for Windows, Mac OSX and Linux.

Anaconda offers jax-sklearn as part of its free distribution.


Intel Extension for Scikit-learn
--------------------------------

Intel maintains an optimized x86_64 package, available in PyPI (via `pip`),
and in the `main`, `conda-forge` and `intel` conda channels:

.. prompt:: bash

  conda install jax-sklearn-intelex

This package has an Intel optimized version of many estimators. Whenever
an alternative implementation doesn't exist, jax-sklearn implementation
is used as a fallback. Those optimized solvers come from the oneDAL
C++ library and are optimized for the x86_64 architecture, and are
optimized for multi-core Intel CPUs.

Note that those solvers are not enabled by default, please refer to the
`jax-sklearn-intelex <https://intel.github.io/jax-sklearn-intelex/latest/what-is-patching.html>`_
documentation for more details on usage scenarios. Direct export example:

.. prompt:: python >>>

  from xlearnex.neighbors import NearestNeighbors

Compatibility with the standard jax-sklearn solvers is checked by running the
full jax-sklearn test suite via automated continuous integration as reported
on https://github.com/intel/jax-sklearn-intelex. If you observe any issue
with `jax-sklearn-intelex`, please report the issue on their
`issue tracker <https://github.com/intel/jax-sklearn-intelex/issues>`__.


WinPython for Windows
---------------------

The `WinPython <https://winpython.github.io/>`_ project distributes
jax-sklearn as an additional plugin.


Troubleshooting
===============

If you encounter unexpected failures when installing jax-sklearn, you may submit
an issue to the `issue tracker <https://github.com/chenxingqiang/jax-sklearn/issues>`_.
Before that, please also make sure to check the following common issues.

.. _windows_longpath:

Error caused by file path length limit on Windows
-------------------------------------------------

It can happen that pip fails to install packages when reaching the default path
size limit of Windows if Python is installed in a nested location such as the
`AppData` folder structure under the user home directory, for instance::

    C:\Users\username>C:\Users\username\AppData\Local\Microsoft\WindowsApps\python.exe -m pip install jax-sklearn
    Collecting jax-sklearn
    ...
    Installing collected packages: jax-sklearn
    ERROR: Could not install packages due to an OSError: [Errno 2] No such file or directory: 'C:\\Users\\username\\AppData\\Local\\Packages\\PythonSoftwareFoundation.Python.3.7_qbz5n2kfra8p0\\LocalCache\\local-packages\\Python37\\site-packages\\xlearn\\datasets\\tests\\data\\openml\\292\\api-v1-json-data-list-data_name-australian-limit-2-data_version-1-status-deactivated.json.gz'

In this case it is possible to lift that limit in the Windows registry by
using the ``regedit`` tool:

#. Type "regedit" in the Windows start menu to launch ``regedit``.

#. Go to the
   ``Computer\HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\FileSystem``
   key.

#. Edit the value of the ``LongPathsEnabled`` property of that key and set
   it to 1.

#. Reinstall jax-sklearn (ignoring the previous broken installation):

   .. prompt:: powershell

      pip install --exists-action=i jax-sklearn
