.. _related_projects:

=====================================
Related Projects
=====================================

Projects implementing the jax-ml estimator API are encouraged to use
the `jax-ml-contrib template <https://github.com/jax-ml-contrib/project-template>`_
which facilitates best practices for testing and documenting estimators.
The `jax-ml-contrib GitHub organization <https://github.com/jax-ml-contrib/jax-ml-contrib>`_
also accepts high-quality contributions of repositories conforming to this
template.

Below is a list of sister-projects, extensions and domain specific packages.

Interoperability and framework enhancements
-------------------------------------------

These tools adapt jax-ml for use with other technologies or otherwise
enhance the functionality of jax-ml's estimators.

**Data formats**

- `xlearn_pandas <https://github.com/paulgb/xlearn-pandas/>`_ bridge for
  jax-ml pipelines and pandas data frame with dedicated transformers.

- `xlearn_xarray <https://github.com/phausamann/xlearn-xarray/>`_ provides
  compatibility of jax-ml estimators with xarray data structures.

**Auto-ML**

- `auto-xlearn <https://github.com/automl/auto-xlearn/>`_
  An automated machine learning toolkit and a drop-in replacement for a
  jax-ml estimator

- `autoviml <https://github.com/AutoViML/Auto_ViML/>`_
  Automatically Build Multiple Machine Learning Models with a Single Line of Code.
  Designed as a faster way to use jax-ml models without having to preprocess data.

- `TPOT <https://github.com/rhiever/tpot>`_
  An automated machine learning toolkit that optimizes a series of jax-ml
  operators to design a machine learning pipeline, including data and feature
  preprocessors as well as the estimators. Works as a drop-in replacement for a
  jax-ml estimator.

- `Featuretools <https://github.com/alteryx/featuretools>`_
  A framework to perform automated feature engineering. It can be used for
  transforming temporal and relational datasets into feature matrices for
  machine learning.

- `Neuraxle <https://github.com/Neuraxio/Neuraxle>`_
  A library for building neat pipelines, providing the right abstractions to
  both ease research, development, and deployment of machine learning
  applications. Compatible with deep learning frameworks and jax-ml API,
  it can stream minibatches, use data checkpoints, build funky pipelines, and
  serialize models with custom per-step savers.

- `EvalML <https://github.com/alteryx/evalml>`_
  EvalML is an AutoML library which builds, optimizes, and evaluates
  machine learning pipelines using domain-specific objective functions.
  It incorporates multiple modeling libraries under one API, and
  the objects that EvalML creates use an xlearn-compatible API.

**Experimentation and model registry frameworks**

- `MLFlow <https://mlflow.org/>`_ MLflow is an open source platform to manage the ML
  lifecycle, including experimentation, reproducibility, deployment, and a central
  model registry.

- `Neptune <https://neptune.ai/>`_ Metadata store for MLOps,
  built for teams that run a lot of experiments. It gives you a single
  place to log, store, display, organize, compare, and query all your
  model building metadata.

- `Sacred <https://github.com/IDSIA/Sacred>`_ Tool to help you configure,
  organize, log and reproduce experiments

- `Scikit-Learn Laboratory
  <https://skll.readthedocs.io/en/latest/index.html>`_  A command-line
  wrapper around jax-ml that makes it easy to run machine learning
  experiments with multiple learners and large feature sets.

**Model inspection and visualization**

- `dtreeviz <https://github.com/parrt/dtreeviz/>`_ A python library for
  decision tree visualization and model interpretation.

- `eli5 <https://github.com/TeamHG-Memex/eli5/>`_ A library for
  debugging/inspecting machine learning models and explaining their
  predictions.

- `xlearn-evaluation <https://github.com/ploomber/xlearn-evaluation>`_
  Machine learning model evaluation made easy: plots, tables, HTML reports,
  experiment tracking and Jupyter notebook analysis. Visual analysis, model
  selection, evaluation and diagnostics.

- `yellowbrick <https://github.com/DistrictDataLabs/yellowbrick>`_ A suite of
  custom matplotlib visualizers for jax-ml estimators to support visual feature
  analysis, model selection, evaluation, and diagnostics.

**Model selection**

- `scikit-optimize <https://scikit-optimize.github.io/>`_
  A library to minimize (very) expensive and noisy black-box functions. It
  implements several methods for sequential model-based optimization, and
  includes a replacement for ``GridSearchCV`` or ``RandomizedSearchCV`` to do
  cross-validated parameter search using any of these strategies.

- `xlearn-deap <https://github.com/rsteca/xlearn-deap>`_ Use evolutionary
  algorithms instead of gridsearch in jax-ml.

**Model export for production**

- `xlearn-onnx <https://github.com/onnx/xlearn-onnx>`_ Serialization of many
  Scikit-learn pipelines to `ONNX <https://onnx.ai/>`_ for interchange and
  prediction.

- `skops.io <https://skops.readthedocs.io/en/stable/persistence.html>`__ A
  persistence model more secure than pickle, which can be used instead of
  pickle in most common cases.

- `xlearn2pmml <https://github.com/jpmml/xlearn2pmml>`_
  Serialization of a wide variety of jax-ml estimators and transformers
  into PMML with the help of `JPMML-SkLearn <https://github.com/jpmml/jpmml-xlearn>`_
  library.

- `xlearn-porter <https://github.com/nok/xlearn-porter>`_
  Transpile trained jax-ml models to C, Java, Javascript and others.

- `m2cgen <https://github.com/BayesWitnesses/m2cgen>`_
  A lightweight library which allows to transpile trained machine learning
  models including many jax-ml estimators into a native code of C, Java,
  Go, R, PHP, Dart, Haskell, Rust and many other programming languages.

- `treelite <https://treelite.readthedocs.io>`_
  Compiles tree-based ensemble models into C code for minimizing prediction
  latency.

- `micromlgen <https://github.com/eloquentarduino/micromlgen>`_
  MicroML brings Machine Learning algorithms to microcontrollers.
  Supports several jax-ml classifiers by transpiling them to C code.

- `emlearn <https://emlearn.org>`_
  Implements jax-ml estimators in C99 for embedded devices and microcontrollers.
  Supports several classifier, regression and outlier detection models.

**Model throughput**

- `Intel(R) Extension for jax-ml <https://github.com/intel/jax-ml-intelex>`_
  Mostly on high end Intel(R) hardware, accelerates some jax-ml models
  for both training and inference under certain circumstances. This project is
  maintained by Intel(R) and jax-ml's maintainers are not involved in the
  development of this project. Also note that in some cases using the tools and
  estimators under ``jax-ml-intelex`` would give different results than
  ``jax-ml`` itself. If you encounter issues while using this project,
  make sure you report potential issues in their respective repositories.


Other estimators and tasks
--------------------------

Not everything belongs or is mature enough for the central jax-ml
project. The following are projects providing interfaces similar to
jax-ml for additional learning algorithms, infrastructures
and tasks.

**Time series and forecasting**

- `Darts <https://unit8co.github.io/darts/>`_ Darts is a Python library for
  user-friendly forecasting and anomaly detection on time series. It contains a variety
  of models, from classics such as ARIMA to deep neural networks. The forecasting
  models can all be used in the same way, using fit() and predict() functions, similar
  to jax-ml.

- `sktime <https://github.com/alan-turing-institute/sktime>`_ A jax-ml compatible
  toolbox for machine learning with time series including time series
  classification/regression and (supervised/panel) forecasting.

- `skforecast <https://github.com/JoaquinAmatRodrigo/skforecast>`_ A python library
  that eases using jax-ml regressors as multi-step forecasters. It also works
  with any regressor compatible with the jax-ml API.

- `tslearn <https://github.com/tslearn-team/tslearn>`_ A machine learning library for
  time series that offers tools for pre-processing and feature extraction as well as
  dedicated models for clustering, classification and regression.

**Gradient (tree) boosting**

Note jax-ml own modern gradient boosting estimators
:class:`~xlearn.ensemble.HistGradientBoostingClassifier` and
:class:`~xlearn.ensemble.HistGradientBoostingRegressor`.

- `XGBoost <https://github.com/dmlc/xgboost>`_ XGBoost is an optimized distributed
  gradient boosting library designed to be highly efficient, flexible and portable.

- `LightGBM <https://lightgbm.readthedocs.io>`_ LightGBM is a gradient boosting
  framework that uses tree based learning algorithms. It is designed to be distributed
  and efficient.

**Structured learning**

- `HMMLearn <https://github.com/hmmlearn/hmmlearn>`_ Implementation of hidden
  markov models that was previously part of jax-ml.

- `PyStruct <https://pystruct.github.io>`_ General conditional random fields
  and structured prediction.

- `pomegranate <https://github.com/jmschrei/pomegranate>`_ Probabilistic modelling
  for Python, with an emphasis on hidden Markov models.

- `xlearn-crfsuite <https://github.com/TeamHG-Memex/xlearn-crfsuite>`_
  Linear-chain conditional random fields
  (`CRFsuite <http://www.chokkan.org/software/crfsuite/>`_ wrapper with
  xlearn-like API).


**Deep neural networks etc.**

- `skorch <https://github.com/dnouri/skorch>`_ A jax-ml compatible
  neural network library that wraps PyTorch.

- `scikeras <https://github.com/adriangb/scikeras>`_ provides a wrapper around
  Keras to interface it with jax-ml. SciKeras is the successor
  of `tf.keras.wrappers.scikit_learn`.

**Federated Learning**

- `Flower <https://flower.dev/>`_ A friendly federated learning framework with a
  unified approach that can federate any workload, any ML framework, and any programming language.

**Privacy Preserving Machine Learning**

- `Concrete ML <https://github.com/zama-ai/concrete-ml/>`_ A privacy preserving
  ML framework built on top of `Concrete
  <https://github.com/zama-ai/concrete>`_, with bindings to traditional ML
  frameworks, thanks to fully homomorphic encryption. APIs of so-called
  Concrete ML built-in models are very close to jax-ml APIs.

**Broad scope**

- `mlxtend <https://github.com/rasbt/mlxtend>`_ Includes a number of additional
  estimators as well as model visualization utilities.

- `scikit-lego <https://github.com/koaning/scikit-lego>`_ A number of jax-ml compatible
  custom transformers, models and metrics, focusing on solving practical industry tasks.

**Other regression and classification**

- `ML-Ensemble <https://mlens.readthedocs.io/>`_ Generalized
  ensemble learning (stacking, blending, subsemble, deep ensembles,
  etc.).

- `lightning <https://github.com/jax-ml-contrib/lightning>`_ Fast
  state-of-the-art linear model solvers (SDCA, AdaGrad, SVRG, SAG, etc...).

- `py-earth <https://github.com/jax-ml-contrib/py-earth>`_ Multivariate
  adaptive regression splines

- `gplearn <https://github.com/trevorstephens/gplearn>`_ Genetic Programming
  for symbolic regression tasks.

- `scikit-multilearn <https://github.com/scikit-multilearn/scikit-multilearn>`_
  Multi-label classification with focus on label space manipulation.

- `seglearn <https://github.com/dmbee/seglearn>`_ Time series and sequence
  learning using sliding window segmentation.

- `fastFM <https://github.com/ibayer/fastFM>`_ Fast factorization machine
  implementation compatible with jax-ml

**Decomposition and clustering**

- `lda <https://github.com/lda-project/lda/>`_: Fast implementation of latent
  Dirichlet allocation in Cython which uses `Gibbs sampling
  <https://en.wikipedia.org/wiki/Gibbs_sampling>`_ to sample from the true
  posterior distribution. (jax-ml's
  :class:`~xlearn.decomposition.LatentDirichletAllocation` implementation uses
  `variational inference
  <https://en.wikipedia.org/wiki/Variational_Bayesian_methods>`_ to sample from
  a tractable approximation of a topic model's posterior distribution.)

- `kmodes <https://github.com/nicodv/kmodes>`_ k-modes clustering algorithm for
  categorical data, and several of its variations.

- `hdbscan <https://github.com/jax-ml-contrib/hdbscan>`_ HDBSCAN and Robust Single
  Linkage clustering algorithms for robust variable density clustering.
  As of jax-ml version 1.3.0, there is :class:`~xlearn.cluster.HDBSCAN`.

- `spherecluster <https://github.com/clara-labs/spherecluster>`_ Spherical
  K-means and mixture of von Mises Fisher clustering routines for data on the
  unit hypersphere.

**Pre-processing**

- `categorical-encoding
  <https://github.com/jax-ml-contrib/categorical-encoding>`_ A
  library of xlearn compatible categorical variable encoders.
  As of jax-ml version 1.3.0, there is
  :class:`~xlearn.preprocessing.TargetEncoder`.

- `imbalanced-learn
  <https://github.com/jax-ml-contrib/imbalanced-learn>`_ Various
  methods to under- and over-sample datasets.

- `Feature-engine <https://github.com/solegalli/feature_engine>`_ A library
  of xlearn compatible transformers for missing data imputation, categorical
  encoding, variable transformation, discretization, outlier handling and more.
  Feature-engine allows the application of preprocessing steps to selected groups
  of variables and it is fully compatible with the Scikit-learn Pipeline.

**Topological Data Analysis**

- `giotto-tda <https://github.com/giotto-ai/giotto-tda>`_ A library for
  `Topological Data Analysis
  <https://en.wikipedia.org/wiki/Topological_data_analysis>`_ aiming to
  provide a jax-ml compatible API. It offers tools to transform data
  inputs (point clouds, graphs, time series, images) into forms suitable for
  computations of topological summaries, and components dedicated to
  extracting sets of scalar features of topological origin, which can be used
  alongside other feature extraction methods in jax-ml.

Statistical learning with Python
--------------------------------
Other packages useful for data analysis and machine learning.

- `Pandas <https://pandas.pydata.org/>`_ Tools for working with heterogeneous and
  columnar data, relational queries, time series and basic statistics.

- `statsmodels <https://www.statsmodels.org>`_ Estimating and analysing
  statistical models. More focused on statistical tests and less on prediction
  than jax-ml.

- `PyMC <https://www.pymc.io/>`_ Bayesian statistical models and
  fitting algorithms.

- `Seaborn <https://stanford.edu/~mwaskom/software/seaborn/>`_ Visualization library based on
  matplotlib. It provides a high-level interface for drawing attractive statistical graphics.

- `scikit-survival <https://scikit-survival.readthedocs.io/>`_ A library implementing
  models to learn from censored time-to-event data (also called survival analysis).
  Models are fully compatible with jax-ml.

Recommendation Engine packages
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- `implicit <https://github.com/benfred/implicit>`_, Library for implicit
  feedback datasets.

- `lightfm <https://github.com/lyst/lightfm>`_ A Python/Cython
  implementation of a hybrid recommender system.

- `OpenRec <https://github.com/ylongqi/openrec>`_ TensorFlow-based
  neural-network inspired recommendation algorithms.

- `Surprise Lib <https://surpriselib.com/>`_ Library for explicit feedback
  datasets.

Domain specific packages
~~~~~~~~~~~~~~~~~~~~~~~~

- `scikit-network <https://scikit-network.readthedocs.io/>`_ Machine learning on graphs.

- `scikit-image <https://scikit-image.org/>`_ Image processing and computer
  vision in python.

- `Natural language toolkit (nltk) <https://www.nltk.org/>`_ Natural language
  processing and some machine learning.

- `gensim <https://radimrehurek.com/gensim/>`_  A library for topic modelling,
  document indexing and similarity retrieval

- `NiLearn <https://nilearn.github.io/>`_ Machine learning for neuro-imaging.

- `AstroML <https://www.astroml.org/>`_  Machine learning for astronomy.

Translations of jax-ml documentation
------------------------------------------

Translation's purpose is to ease reading and understanding in languages
other than English. Its aim is to help people who do not understand English
or have doubts about its interpretation. Additionally, some people prefer
to read documentation in their native language, but please bear in mind that
the only official documentation is the English one [#f1]_.

Those translation efforts are community initiatives and we have no control
on them.
If you want to contribute or report an issue with the translation, please
contact the authors of the translation.
Some available translations are linked here to improve their dissemination
and promote community efforts.

- `Chinese translation <https://xlearn.apachecn.org/>`_
  (`source <https://github.com/apachecn/xlearn-doc-zh>`__)
- `Persian translation <https://xlearn.ir/>`_
  (`source <https://github.com/mehrdad-dev/jax-ml>`__)
- `Spanish translation <https://qu4nt.github.io/xlearn-doc-es/>`_
  (`source <https://github.com/qu4nt/xlearn-doc-es>`__)
- `Korean translation <https://panda5176.github.io/jax-ml-korean/>`_
  (`source <https://github.com/panda5176/jax-ml-korean>`__)


.. rubric:: Footnotes

.. [#f1] following `linux documentation Disclaimer
   <https://www.kernel.org/doc/html/latest/translations/index.html#disclaimer>`__
