# JAX-sklearn: JAX-Accelerated Machine Learning

**JAX-sklearn** is a drop-in replacement for scikit-learn that provides **automatic JAX acceleration** for machine learning algorithms while maintaining **100% API compatibility**.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![JAX](https://img.shields.io/badge/JAX-0.4.20+-orange.svg)](https://github.com/google/jax)
[![License](https://img.shields.io/badge/license-BSD--3--Clause-green.svg)](COPYING)
[![Version](https://img.shields.io/badge/version-0.1.10-brightgreen.svg)](https://pypi.org/project/jax-sklearn/)
[![CI](https://img.shields.io/badge/CI-Azure%20Pipelines-blue.svg)](https://dev.azure.com/chenxingqiang/jax-sklearn)
[![Tests](https://img.shields.io/badge/tests-34752%20passed-success.svg)](#-test-results)

---

## Key Features

- **Drop-in Replacement**: Use `import xlearn as sklearn` — no code changes needed
- **JAX Acceleration**: 4-20x speedup on CPU, 100x+ on GPU/TPU
- **Multi-Hardware**: CPU, NVIDIA GPU (CUDA), Apple Silicon (Metal), Google TPU
- **Auto Fallback**: Graceful degradation when JAX is unavailable
- **Prerequisite for [Secret-Learn](https://github.com/chenxingqiang/secret-learn)**: Privacy-preserving federated ML with SecretFlow

---

## Quick Install

```bash
# Apple Silicon (recommended)
uv pip install jax-sklearn[jax-metal]
# NVIDIA GPU
uv pip install jax-sklearn[jax-gpu]
# CPU only
uv pip install jax-sklearn[jax-cpu]
```

Or with pip:

```bash
pip install jax-sklearn
```

> **Build prerequisites**: When installing from source (no wheel for your platform), you need C/C++ tooling and Python headers. See [Troubleshooting](#troubleshooting) below.

---

## Quick Start

```python
import xlearn as sklearn
from xlearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X, y)
predictions = model.predict(X_test)
# JAX acceleration applied automatically when beneficial
```

JAX acceleration can be configured via `xlearn._jax`:

```python
import xlearn._jax as jax_config

# Default: always enable JAX (best for GPU/TPU)
jax_config.set_config(enable_jax=True)

# Threshold mode: only use JAX for large datasets (CPU users)
jax_config.set_config(enable_jax=True, jax_auto_threshold=True)

# Disable JAX (pure sklearn)
jax_config.set_config(enable_jax=False)
```

---

## Performance

### LinearRegression on Apple Silicon M2 (CPU)

| Data Size | XLearn | sklearn | Speedup |
|-----------|--------|---------|---------|
| 10K × 100 | 0.0097s | 0.0113s | 1.16x |
| 10K × 1K | 0.0384s | 0.1590s | **4.14x** |
| 10K × 10K | 2.82s | 55.96s | **19.86x** |
| 50K × 2K | 0.54s | 1.96s | **3.60x** |
| 100K × 1K | 0.40s | 1.23s | **3.04x** |

> JAX has ~0.2s JIT compilation overhead on first run. Crossover point is ~10K × 100 on CPU.

### Hardware Scaling

| Hardware | Small Data | Medium Data | Large Data |
|----------|------------|-------------|------------|
| CPU | ~1x | 0.2-0.5x | 4-20x |
| Metal (M1-M4) | ~1x | 1.5-2x | 2-3x |
| CUDA GPU | 1-2x | 5-10x | 50-100x |
| TPU | 2-5x | 10-20x | 100x+ |

---

## Supported Algorithms

### JAX-Accelerated
- **Linear Models**: LinearRegression, Ridge, Lasso, ElasticNet
- **Clustering**: KMeans
- **Decomposition**: PCA, TruncatedSVD
- **Preprocessing**: StandardScaler, MinMaxScaler

All other scikit-learn algorithms (RandomForest, SVM, Neural Networks, etc.) are fully available via automatic fallback to the original implementation.

---

## Requirements

- **Python**: 3.10+
- **JAX**: 0.4.20+
- **NumPy**: 1.22.0+, **SciPy**: 1.8.0+

Hardware-specific: NVIDIA GPU (CUDA 11.1+), Apple Silicon (macOS 12+), or Google TPU.

---

## Troubleshooting

### Build Issues
```bash
# Install Python headers (Linux)
sudo apt-get install python3-dev   # Debian/Ubuntu
sudo dnf install python3-devel     # RHEL/Fedora

# macOS
xcode-select --install

# Disable build isolation if needed
pip install --no-build-isolation jax-sklearn
```

### JAX Not Found / GPU Not Detected
```python
import jax
print("Devices:", jax.devices())
print("Backend:", jax.default_backend())
```

---

## Contributing

```bash
git clone https://github.com/chenxingqiang/jax-sklearn.git
cd jax-sklearn
pip install -e ".[tests]"
pytest xlearn/tests/ -v
```

---

## License & Related Projects

JAX-sklearn is [BSD 3-Clause](COPYING) licensed.

- **[Secret-Learn](https://github.com/chenxingqiang/secret-learn)**: Privacy-preserving federated ML with SecretFlow (348 algorithm implementations)
