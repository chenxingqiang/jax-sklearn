.. -*- mode: rst -*-

|Azure| |Codecov| |CircleCI| |Nightly wheels| |Ruff| |PythonVersion| |PyPi| |DOI| |Benchmark|

.. |Azure| image:: https://dev.azure.com/jax-sklearn/jax-sklearn/_apis/build/status/jax-sklearn.jax-sklearn?branchName=main
   :target: https://dev.azure.com/jax-sklearn/jax-sklearn/_build/latest?definitionId=1&branchName=main

.. |CircleCI| image:: https://circleci.com/gh/jax-sklearn/jax-sklearn/tree/main.svg?style=shield
   :target: https://circleci.com/gh/jax-sklearn/jax-sklearn

.. |Codecov| image:: https://codecov.io/gh/jax-sklearn/jax-sklearn/branch/main/graph/badge.svg?token=Pk8G9gg3y9
   :target: https://codecov.io/gh/jax-sklearn/jax-sklearn

.. |Nightly wheels| image:: https://github.com/chenxingqiang/jax-sklearn/actions/workflows/wheels.yml/badge.svg?event=schedule
   :target: https://github.com/chenxingqiang/jax-sklearn/actions?query=workflow%3A%22Wheel+builder%22+event%3Aschedule

.. |Ruff| image:: https://img.shields.io/badge/code%20style-ruff-000000.svg
   :target: https://github.com/astral-sh/ruff

.. |PythonVersion| image:: https://img.shields.io/pypi/pyversions/jax-sklearn.svg
   :target: https://pypi.org/project/jax-sklearn/

.. |PyPi| image:: https://img.shields.io/pypi/v/jax-sklearn
   :target: https://pypi.org/project/jax-sklearn

.. |DOI| image:: https://zenodo.org/badge/21369/jax-sklearn/jax-sklearn.svg
   :target: https://zenodo.org/badge/latestdoi/21369/jax-sklearn/jax-sklearn

.. |Benchmark| image:: https://img.shields.io/badge/Benchmarked%20by-asv-blue
   :target: https://jax-sklearn.org/jax-sklearn-benchmarks

.. |PythonMinVersion| replace:: 3.10
.. |NumPyMinVersion| replace:: 1.22.0
.. |SciPyMinVersion| replace:: 1.8.0
.. |JoblibMinVersion| replace:: 1.2.0
.. |ThreadpoolctlMinVersion| replace:: 3.1.0
.. |MatplotlibMinVersion| replace:: 3.5.0
.. |Scikit-ImageMinVersion| replace:: 0.19.0
.. |PandasMinVersion| replace:: 1.4.0
.. |SeabornMinVersion| replace:: 0.9.0
.. |PytestMinVersion| replace:: 7.1.2
.. |PlotlyMinVersion| replace:: 5.14.0

.. image:: https://raw.githubusercontent.com/jax-sklearn/jax-sklearn/main/doc/logos/jax-sklearn-logo.png
  :target: https://jax-sklearn.org/

**jax-sklearn** is a Python module for machine learning built on top of
SciPy and is distributed under the 3-Clause BSD license.

The project was started in 2007 by David Cournapeau as a Google Summer
of Code project, and since then many volunteers have contributed. See
the `About us <https://jax-sklearn.org/dev/about.html#authors>`__ page
for a list of core contributors.

It is currently maintained by a team of volunteers.

Website: https://jax-sklearn.org

Installation
------------

Dependencies
~~~~~~~~~~~~

jax-sklearn requires:

- Python (>= |PythonMinVersion|)
- NumPy (>= |NumPyMinVersion|)
- SciPy (>= |SciPyMinVersion|)
- joblib (>= |JoblibMinVersion|)
- threadpoolctl (>= |ThreadpoolctlMinVersion|)

=======

Scikit-learn plotting capabilities (i.e., functions start with ``plot_`` and
classes end with ``Display``) require Matplotlib (>= |MatplotlibMinVersion|).
For running the examples Matplotlib >= |MatplotlibMinVersion| is required.
A few examples require scikit-image >= |Scikit-ImageMinVersion|, a few examples
require pandas >= |PandasMinVersion|, some examples require seaborn >=
|SeabornMinVersion| and plotly >= |PlotlyMinVersion|.

User installation
~~~~~~~~~~~~~~~~~

If you already have a working installation of NumPy and SciPy,
the easiest way to install jax-sklearn is using ``pip``::

    pip install -U jax-sklearn

or ``conda``::

    conda install -c conda-forge jax-sklearn

The documentation includes more detailed `installation instructions <https://jax-sklearn.org/stable/install.html>`_.


Changelog
---------

See the `changelog <https://jax-sklearn.org/dev/whats_new.html>`__
for a history of notable changes to jax-sklearn.

Development
-----------

We welcome new contributors of all experience levels. The jax-sklearn
community goals are to be helpful, welcoming, and effective. The
`Development Guide <https://jax-sklearn.org/stable/developers/index.html>`_
has detailed information about contributing code, documentation, tests, and
more. We've included some basic information in this README.

Important links
~~~~~~~~~~~~~~~

- Official source code repo: https://github.com/chenxingqiang/jax-sklearn
- Download releases: https://pypi.org/project/jax-sklearn/
- Issue tracker: https://github.com/chenxingqiang/jax-sklearn/issues

Source code
~~~~~~~~~~~

You can check the latest sources with the command::

    git clone https://github.com/chenxingqiang/jax-sklearn.git

Contributing
~~~~~~~~~~~~

To learn more about making a contribution to jax-sklearn, please see our
`Contributing guide
<https://jax-sklearn.org/dev/developers/contributing.html>`_.

Testing
~~~~~~~

After installation, you can launch the test suite from outside the source
directory (you will need to have ``pytest`` >= |PyTestMinVersion| installed)::

    pytest xlearn

See the web page https://jax-sklearn.org/dev/developers/contributing.html#testing-and-improving-test-coverage
for more information.

    Random number generation can be controlled during testing by setting
    the ``XLEARN_SEED`` environment variable.

Submitting a Pull Request
~~~~~~~~~~~~~~~~~~~~~~~~~

Before opening a Pull Request, have a look at the
full Contributing page to make sure your code complies
with our guidelines: https://jax-sklearn.org/stable/developers/index.html

Project History
---------------

The project was started in 2007 by David Cournapeau as a Google Summer
of Code project, and since then many volunteers have contributed. See
the `About us <https://jax-sklearn.org/dev/about.html#authors>`__ page
for a list of core contributors.

The project is currently maintained by a team of volunteers.

**Note**: `jax-sklearn` was previously referred to as `scikits.learn`.

Help and Support
----------------

Documentation
~~~~~~~~~~~~~

- HTML documentation (stable release): https://jax-sklearn.org
- HTML documentation (development version): https://jax-sklearn.org/dev/
- FAQ: https://jax-sklearn.org/stable/faq.html

Communication
~~~~~~~~~~~~~

Main Channels
^^^^^^^^^^^^^

- **Website**: https://jax-sklearn.org
- **Blog**: https://blog.jax-sklearn.org
- **Mailing list**: https://mail.python.org/mailman/listinfo/jax-sklearn

Developer & Support
^^^^^^^^^^^^^^^^^^^^^^

- **GitHub Discussions**: https://github.com/chenxingqiang/jax-sklearn/discussions
- **Stack Overflow**: https://stackoverflow.com/questions/tagged/jax-sklearn
- **Discord**: https://discord.gg/h9qyrK8Jc8

Social Media Platforms
^^^^^^^^^^^^^^^^^^^^^^

- **LinkedIn**: https://www.linkedin.com/company/jax-sklearn
- **YouTube**: https://www.youtube.com/channel/UCJosFjYm0ZYVUARxuOZqnnw/playlists
- **Facebook**: https://www.facebook.com/scikitlearnofficial/
- **Instagram**: https://www.instagram.com/scikitlearnofficial/
- **TikTok**: https://www.tiktok.com/@scikit.learn
- **Bluesky**: https://bsky.app/profile/jax-sklearn.org
- **Mastodon**: https://mastodon.social/@xlearn@fosstodon.org

Resources
^^^^^^^^^

- **Calendar**: https://blog.jax-sklearn.org/calendar/
- **Logos & Branding**: https://github.com/chenxingqiang/jax-sklearn/tree/main/doc/logos

Citation
~~~~~~~~

If you use jax-sklearn in a scientific publication, we would appreciate citations: https://jax-sklearn.org/stable/about.html#citing-jax-sklearn
