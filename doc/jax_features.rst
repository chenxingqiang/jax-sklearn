⚡ JAX Features
==============

JAX-sklearn brings the power of JAX acceleration to traditional machine learning through intelligent hardware utilization and seamless API compatibility.

🚀 Key JAX Advantages
--------------------

**Just-In-Time (JIT) Compilation**
   JAX compiles your Python functions to optimized XLA code, providing significant speedups especially for iterative algorithms.

**Automatic Differentiation**
   Built-in support for forward and reverse-mode automatic differentiation, enabling efficient gradient-based optimization.

**Vectorization (VMAP)**
   Automatic vectorization of functions across batch dimensions, perfect for processing multiple samples efficiently.

**Hardware Acceleration**
   Native support for GPUs and TPUs with automatic memory management and device placement.

⚡ JAX-Accelerated Algorithms
----------------------------

Linear Models
~~~~~~~~~~~~~

* **LinearRegression**: 5.53x faster training on large datasets
* **Ridge Regression**: Optimized regularized linear regression
* **Lasso**: L1-regularized regression with coordinate descent
* **ElasticNet**: Combined L1/L2 regularization

**Performance Example:**

.. code-block:: python

   import numpy as np
   import xlearn as sklearn
   
   # Large dataset: 100K samples × 1K features
   X = np.random.randn(100000, 1000)
   y = X @ np.random.randn(1000) + 0.1 * np.random.randn(100000)
   
   # JAX-accelerated training
   model = sklearn.linear_model.LinearRegression()
   model.fit(X, y)  # 0.060s vs 0.331s (scikit-learn)

Clustering
~~~~~~~~~~

* **K-Means**: GPU-accelerated clustering for large datasets
* **Mini-Batch K-Means**: Memory-efficient clustering

**Clustering Example:**

.. code-block:: python

   # Large clustering problem: 50K samples × 100 features
   X = np.random.randn(50000, 100)
   
   kmeans = sklearn.cluster.KMeans(n_clusters=10)
   labels = kmeans.fit_predict(X)  # Automatically uses GPU if beneficial

Decomposition
~~~~~~~~~~~~~

* **PCA**: Principal Component Analysis with SVD acceleration
* **TruncatedSVD**: Efficient dimensionality reduction

Preprocessing
~~~~~~~~~~~~~

* **StandardScaler**: Fast feature standardization
* **MinMaxScaler**: Efficient min-max normalization

🧠 Intelligent Acceleration
---------------------------

JAX-sklearn automatically decides when to use JAX based on:

**Data Size Heuristics**

.. code-block:: python

   # Algorithm-specific thresholds
   thresholds = {
       'LinearRegression': {
           'min_complexity': 1e8,      # 100K × 1K dataset
           'min_samples': 10000
       },
       'KMeans': {
           'min_complexity': 1e6,      # 10K × 100 dataset
           'min_samples': 5000
       },
       'PCA': {
           'min_complexity': 1e7,      # 32K × 300 dataset
           'min_samples': 5000
       }
   }

**Performance Decision Matrix**

+-------------------+------------------+------------------+
| Problem Size      | Recommended      | Expected Speedup |
+===================+==================+==================+
| < 1K samples      | CPU              | 1.0x - 1.5x      |
+-------------------+------------------+------------------+
| 1K - 10K          | CPU/GPU (auto)   | 1.5x - 3.0x      |
+-------------------+------------------+------------------+
| 10K - 100K        | GPU (preferred)  | 3.0x - 6.0x      |
+-------------------+------------------+------------------+
| 100K - 1M         | GPU/TPU (auto)   | 5.0x - 10x       |
+-------------------+------------------+------------------+
| > 1M samples      | TPU (preferred)  | 8.0x - 15x       |
+-------------------+------------------+------------------+

🔧 JAX Configuration
--------------------

**Basic Configuration**

.. code-block:: python

   import xlearn._jax as jax_config
   
   # Check JAX status
   print(f"JAX available: {jax_config.is_jax_available()}")
   print(f"JAX platform: {jax_config.get_jax_platform()}")
   
   # Configure JAX settings
   jax_config.set_config(
       enable_jax=True,
       jax_platform="gpu",
       memory_limit_gpu=8192,  # 8GB limit
       jit_compilation=True,
       precision="float32"
   )

**Context Managers**

.. code-block:: python

   # Temporary JAX settings
   with jax_config.config_context(jax_platform="gpu"):
       model = sklearn.linear_model.LinearRegression()
       model.fit(X, y)  # Uses GPU
   
   with jax_config.config_context(enable_jax=False):
       model = sklearn.linear_model.LinearRegression()
       model.fit(X, y)  # Uses NumPy

**Debug Mode**

.. code-block:: python

   # Enable debug mode to see acceleration decisions
   jax_config.set_config(debug_mode=True)
   
   model = sklearn.linear_model.LinearRegression()
   model.fit(X, y)  # Prints hardware selection reasoning

🔄 Fallback System
------------------

JAX-sklearn includes a robust fallback system:

**Automatic Fallback Scenarios**

1. **JAX Not Available**: Falls back to original scikit-learn
2. **Hardware Issues**: GPU/TPU unavailable → CPU
3. **Memory Issues**: Out of GPU memory → CPU
4. **Compilation Errors**: JIT compilation fails → NumPy
5. **Numerical Issues**: JAX numerical instability → Original implementation

**Fallback Configuration**

.. code-block:: python

   jax_config.set_config(
       enable_jax=True,
       fallback_on_error=True,  # Enable automatic fallback
       debug_mode=True          # Show fallback reasons
   )

🔬 Advanced JAX Features
-----------------------

**Memory Management**

.. code-block:: python

   import os
   
   # Configure JAX memory allocation
   os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
   os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.8'  # Use 80% of GPU memory

**Multi-GPU Support**

.. code-block:: python

   # Use specific GPU
   os.environ['CUDA_VISIBLE_DEVICES'] = '0'
   
   # Use multiple GPUs
   os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

**Precision Control**

.. code-block:: python

   # Use float64 for higher precision
   jax_config.set_config(precision="float64")
   
   # Use float32 for better performance (default)
   jax_config.set_config(precision="float32")

📊 Performance Benchmarks
-------------------------

**Linear Regression (100K × 1K dataset)**

.. code-block:: text

   Hardware        Training Time    Speedup    Memory Usage
   ─────────────────────────────────────────────────────────
   TPU             0.035s          9.46x      0.25 GB
   GPU             0.060s          5.53x      0.37 GB
   CPU (JAX)       0.180s          1.84x      0.37 GB
   NumPy           0.331s          1.00x      0.37 GB

**K-Means Clustering (50K × 100 dataset)**

.. code-block:: text

   Hardware        Training Time    Speedup    Convergence
   ─────────────────────────────────────────────────────────
   GPU             0.045s          6.22x      15 iterations
   CPU (JAX)       0.120s          2.33x      15 iterations
   NumPy           0.280s          1.00x      15 iterations

**Batch Processing (50 problems)**

.. code-block:: text

   Method          Total Time      Speedup    Problems/sec
   ─────────────────────────────────────────────────────────
   JAX-TPU         0.055s         9.82x      909
   JAX-GPU         0.097s         5.57x      515
   JAX-CPU         0.220s         2.45x      227
   Sequential      0.540s         1.00x      93

🎯 Best Practices
-----------------

**When to Use JAX**

* Large datasets (>10K samples)
* High-dimensional problems (>100 features)
* Iterative algorithms (clustering, optimization)
* Batch processing multiple problems
* GPU/TPU hardware available

**When to Stick with NumPy**

* Small datasets (<1K samples)
* Simple one-time computations
* Memory-constrained environments
* Debugging and development

**Optimization Tips**

1. **Batch Your Data**: JAX performs best on larger batches
2. **Use Consistent Shapes**: Avoid recompilation by keeping array shapes consistent
3. **Enable JIT**: Keep JIT compilation enabled for best performance
4. **Monitor Memory**: Use memory limits to prevent OOM errors
5. **Profile Your Code**: Use JAX profiling tools to identify bottlenecks

🔗 Related Documentation
------------------------

* :doc:`hardware_guide` - Detailed hardware setup and optimization
* :doc:`install` - Installation instructions for different hardware
* :doc:`getting_started` - Quick start guide with examples
* :doc:`faq` - Frequently asked questions about JAX acceleration
