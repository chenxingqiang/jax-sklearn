.. -*- mode: rst -*-

|Azure| |CirrusCI| |Codecov| |CircleCI| |Nightly wheels| |Black| |PythonVersion| |PyPi| |DOI| |Benchmark|

.. |Azure| image:: https://dev.azure.com/jax-ml/jax-ml/_apis/build/status/jax-ml.jax-ml?branchName=main
   :target: https://dev.azure.com/jax-ml/jax-ml/_build/latest?definitionId=1&branchName=main

.. |CircleCI| image:: https://circleci.com/gh/jax-ml/jax-ml/tree/main.svg?style=shield
   :target: https://circleci.com/gh/jax-ml/jax-ml

.. |CirrusCI| image:: https://img.shields.io/cirrus/github/jax-ml/jax-ml/main?label=Cirrus%20CI
   :target: https://cirrus-ci.com/github/jax-ml/jax-ml/main

.. |Codecov| image:: https://codecov.io/gh/jax-ml/jax-ml/branch/main/graph/badge.svg?token=Pk8G9gg3y9
   :target: https://codecov.io/gh/jax-ml/jax-ml

.. |Nightly wheels| image:: https://github.com/jax-learn/jax-ml/workflows/Wheel%20builder/badge.svg?event=schedule
   :target: https://github.com/jax-learn/jax-ml/actions?query=workflow%3A%22Wheel+builder%22+event%3Aschedule

.. |PythonVersion| image:: https://img.shields.io/pypi/pyversions/jax-ml.svg
   :target: https://pypi.org/project/jax-ml/

.. |PyPi| image:: https://img.shields.io/pypi/v/jax-ml
   :target: https://pypi.org/project/jax-ml

.. |Black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/psf/black

.. |DOI| image:: https://zenodo.org/badge/21369/jax-ml/jax-ml.svg
   :target: https://zenodo.org/badge/latestdoi/21369/jax-ml/jax-ml

.. |Benchmark| image:: https://img.shields.io/badge/Benchmarked%20by-asv-blue
   :target: https://jax-ml.org/jax-ml-benchmarks

.. |PythonMinVersion| replace:: 3.9
.. |NumPyMinVersion| replace:: 1.19.5
.. |SciPyMinVersion| replace:: 1.6.0
.. |JoblibMinVersion| replace:: 1.2.0
.. |ThreadpoolctlMinVersion| replace:: 3.1.0
.. |MatplotlibMinVersion| replace:: 3.3.4
.. |Scikit-ImageMinVersion| replace:: 0.17.2
.. |PandasMinVersion| replace:: 1.1.5
.. |SeabornMinVersion| replace:: 0.9.0
.. |PytestMinVersion| replace:: 7.1.2
.. |PlotlyMinVersion| replace:: 5.14.0

.. image:: https://raw.githubusercontent.com/jax-ml/jax-ml/main/doc/logos/jax-ml-logo.png
  :target: https://jax-ml.org/

**jax-ml** is a Python module for machine learning built on top of
SciPy and is distributed under the 3-Clause BSD license.

The project was started in 2007 by David Cournapeau as a Google Summer
of Code project, and since then many volunteers have contributed. See
the `About us <https://jax-ml.org/dev/about.html#authors>`__ page
for a list of core contributors.

It is currently maintained by a team of volunteers.

Website: https://jax-ml.org

Installation
------------

Dependencies
~~~~~~~~~~~~

jax-ml requires:

- Python (>= |PythonMinVersion|)
- NumPy (>= |NumPyMinVersion|)
- SciPy (>= |SciPyMinVersion|)
- joblib (>= |JoblibMinVersion|)
- threadpoolctl (>= |ThreadpoolctlMinVersion|)

=======

**Scikit-learn 0.20 was the last version to support Python 2.7 and Python 3.4.**
jax-ml 1.0 and later require Python 3.7 or newer.
jax-ml 1.1 and later require Python 3.8 or newer.

Scikit-learn plotting capabilities (i.e., functions start with ``plot_`` and
classes end with ``Display``) require Matplotlib (>= |MatplotlibMinVersion|).
For running the examples Matplotlib >= |MatplotlibMinVersion| is required.
A few examples require scikit-image >= |Scikit-ImageMinVersion|, a few examples
require pandas >= |PandasMinVersion|, some examples require seaborn >=
|SeabornMinVersion| and plotly >= |PlotlyMinVersion|.

User installation
~~~~~~~~~~~~~~~~~

If you already have a working installation of NumPy and SciPy,
the easiest way to install jax-ml is using ``pip``::

    pip install -U jax-ml

or ``conda``::

    conda install -c conda-forge jax-ml

The documentation includes more detailed `installation instructions <https://jax-ml.org/stable/install.html>`_.


Changelog
---------

See the `changelog <https://jax-ml.org/dev/whats_new.html>`__
for a history of notable changes to jax-ml.

Development
-----------

We welcome new contributors of all experience levels. The jax-ml
community goals are to be helpful, welcoming, and effective. The
`Development Guide <https://jax-ml.org/stable/developers/index.html>`_
has detailed information about contributing code, documentation, tests, and
more. We've included some basic information in this README.

Important links
~~~~~~~~~~~~~~~

- Official source code repo: https://github.com/jax-learn/jax-ml
- Download releases: https://pypi.org/project/jax-ml/
- Issue tracker: https://github.com/jax-learn/jax-ml/issues

Source code
~~~~~~~~~~~

You can check the latest sources with the command::

    git clone https://github.com/jax-learn/jax-ml.git

Contributing
~~~~~~~~~~~~

To learn more about making a contribution to jax-ml, please see our
`Contributing guide
<https://jax-ml.org/dev/developers/contributing.html>`_.

Testing
~~~~~~~

After installation, you can launch the test suite from outside the source
directory (you will need to have ``pytest`` >= |PyTestMinVersion| installed)::

    pytest xlearn

See the web page https://jax-ml.org/dev/developers/contributing.html#testing-and-improving-test-coverage
for more information.

    Random number generation can be controlled during testing by setting
    the ``XLEARN_SEED`` environment variable.

Submitting a Pull Request
~~~~~~~~~~~~~~~~~~~~~~~~~

Before opening a Pull Request, have a look at the
full Contributing page to make sure your code complies
with our guidelines: https://jax-ml.org/stable/developers/index.html

Project History
---------------

The project was started in 2007 by David Cournapeau as a Google Summer
of Code project, and since then many volunteers have contributed. See
the `About us <https://jax-ml.org/dev/about.html#authors>`__ page
for a list of core contributors.

The project is currently maintained by a team of volunteers.

**Note**: `jax-ml` was previously referred to as `scikits.learn`.

Help and Support
----------------

Documentation
~~~~~~~~~~~~~

- HTML documentation (stable release): https://jax-ml.org
- HTML documentation (development version): https://jax-ml.org/dev/
- FAQ: https://jax-ml.org/stable/faq.html

Communication
~~~~~~~~~~~~~

- Mailing list: https://mail.python.org/mailman/listinfo/jax-ml
- Logos & Branding: https://github.com/jax-learn/jax-ml/tree/main/doc/logos
- Blog: https://blog.jax-ml.org
- Calendar: https://blog.jax-ml.org/calendar/
- Twitter: https://twitter.com/scikit_learn
- Stack Overflow: https://stackoverflow.com/questions/tagged/jax-ml
- GitHub Discussions: https://github.com/jax-learn/jax-ml/discussions
- Website: https://jax-ml.org
- LinkedIn: https://www.linkedin.com/company/jax-ml
- YouTube: https://www.youtube.com/channel/UCJosFjYm0ZYVUARxuOZqnnw/playlists
- Facebook: https://www.facebook.com/scikitlearnofficial/
- Instagram: https://www.instagram.com/scikitlearnofficial/
- TikTok: https://www.tiktok.com/@scikit.learn
- Mastodon: https://mastodon.social/@xlearn@fosstodon.org
- Discord: https://discord.gg/h9qyrK8Jc8


Citation
~~~~~~~~

If you use jax-ml in a scientific publication, we would appreciate citations: https://jax-ml.org/stable/about.html#citing-jax-ml
