.. _real_world_datasets:

Real world datasets
===================

.. currentmodule:: xlearn.datasets

jax-sklearn provides tools to load larger datasets, downloading them if
necessary.

They can be loaded using the following functions:

.. autosummary::

   fetch_olivetti_faces
   fetch_20newsgroups
   fetch_20newsgroups_vectorized
   fetch_lfw_people
   fetch_lfw_pairs
   fetch_covtype
   fetch_rcv1
   fetch_kddcup99
   fetch_california_housing
   fetch_species_distributions

.. include:: ../../xlearn/datasets/descr/olivetti_faces.rst

.. include:: ../../xlearn/datasets/descr/twenty_newsgroups.rst

.. include:: ../../xlearn/datasets/descr/lfw.rst

.. include:: ../../xlearn/datasets/descr/covtype.rst

.. include:: ../../xlearn/datasets/descr/rcv1.rst

.. include:: ../../xlearn/datasets/descr/kddcup99.rst

.. include:: ../../xlearn/datasets/descr/california_housing.rst

.. include:: ../../xlearn/datasets/descr/species_distributions.rst
