"""
JAX-sklearn GPU (T4) Benchmark for Google Colab

Instructions:
1. Open Google Colab: https://colab.research.google.com/
2. Go to Runtime -> Change runtime type -> GPU (T4)
3. Copy and paste this script into cells and run

Or run directly:
!wget -q https://raw.githubusercontent.com/chenxingqiang/jax-sklearn/main/benchmarks/colab_gpu_benchmark.py
%run colab_gpu_benchmark.py
"""

# ============================================================
# Cell 1: Install dependencies
# ============================================================
print("=" * 70)
print("Step 1: Installing dependencies...")
print("=" * 70)

import subprocess
import sys

# Install JAX with GPU support
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "jax[cuda12]"])
# Install latest jax-sklearn from GitHub (includes all bug fixes)
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "git+https://github.com/chenxingqiang/jax-sklearn.git"])

print("✅ Dependencies installed!")

# ============================================================
# Cell 2: Verify GPU setup
# ============================================================
print("\n" + "=" * 70)
print("Step 2: Verifying GPU setup...")
print("=" * 70)

import jax
import jax.numpy as jnp

print(f"JAX version: {jax.__version__}")
print(f"JAX backend: {jax.default_backend()}")
print(f"JAX devices: {jax.devices()}")

# Check GPU info
if jax.default_backend() == 'gpu':
    print("✅ GPU is available!")
    # Get GPU info
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'], 
                              capture_output=True, text=True)
        print(f"GPU Info: {result.stdout.strip()}")
    except:
        pass
else:
    print("⚠️ GPU not detected. Make sure you selected GPU runtime.")
    print("   Go to: Runtime -> Change runtime type -> GPU")

# ============================================================
# Cell 3: Import libraries
# ============================================================
print("\n" + "=" * 70)
print("Step 3: Importing libraries...")
print("=" * 70)

import numpy as np
import time
import warnings
warnings.filterwarnings('ignore')

import xlearn
from xlearn import linear_model as xlearn_lm
from xlearn.cluster import KMeans
from xlearn.decomposition import PCA
from xlearn.preprocessing import StandardScaler
import sklearn.linear_model
from sklearn.cluster import KMeans as SkKMeans
from sklearn.decomposition import PCA as SkPCA
from sklearn.preprocessing import StandardScaler as SkStandardScaler

print(f"XLearn version: {xlearn.__version__}")
print(f"JAX enabled: {xlearn._JAX_ENABLED}")
print("✅ Libraries imported!")

# ============================================================
# Cell 4: Benchmark function
# ============================================================
def run_benchmark(name, xlearn_model, sklearn_model, X, y=None, warmup=True):
    """Run a single benchmark comparing XLearn and sklearn."""
    print(f"\n{'='*60}")
    print(f"Benchmark: {name}")
    print(f"Data shape: {X.shape}, Complexity: {X.shape[0] * X.shape[1]:.2e}")
    print("-" * 60)
    
    # Warmup run for JIT compilation
    if warmup:
        X_small = X[:min(1000, len(X))]
        y_small = y[:min(1000, len(y))] if y is not None else None
        if y_small is not None:
            xlearn_model.fit(X_small, y_small)
        else:
            xlearn_model.fit(X_small)
    
    # XLearn benchmark
    start = time.perf_counter()
    if y is not None:
        xlearn_model.fit(X, y)
    else:
        xlearn_model.fit(X)
    xlearn_time = time.perf_counter() - start
    
    # Check if JAX was used
    is_jax = getattr(xlearn_model, 'is_using_jax', 'N/A')
    print(f"XLearn: {xlearn_time:.4f}s (JAX: {is_jax})")
    
    # sklearn benchmark
    start = time.perf_counter()
    if y is not None:
        sklearn_model.fit(X, y)
    else:
        sklearn_model.fit(X)
    sklearn_time = time.perf_counter() - start
    print(f"sklearn: {sklearn_time:.4f}s")
    
    # Calculate speedup
    speedup = sklearn_time / xlearn_time if xlearn_time > 0 else 0
    emoji = "🚀" if speedup > 5 else "✅" if speedup > 1 else "⚠️"
    print(f"Speedup: {speedup:.2f}x {emoji}")
    
    return {
        'name': name,
        'xlearn_time': xlearn_time,
        'sklearn_time': sklearn_time,
        'speedup': speedup,
        'is_jax': is_jax
    }

# ============================================================
# Cell 5: Run benchmarks
# ============================================================
print("\n" + "=" * 70)
print("Step 4: Running GPU Benchmarks")
print("=" * 70)

results = []

# ============================================================
# Benchmark 1: Linear Regression - Various sizes
# ============================================================
print("\n" + "=" * 70)
print("📊 LINEAR REGRESSION BENCHMARKS")
print("=" * 70)

lr_configs = [
    (10000, 10000, "10K × 10K (1e8) - Square matrix"),
    (50000, 2000, "50K × 2K (1e8) - Tall matrix"),
    (100000, 1000, "100K × 1K (1e8) - Very tall matrix"),
    (20000, 5000, "20K × 5K (1e8) - Balanced"),
]

for n_samples, n_features, desc in lr_configs:
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    y = X @ np.random.randn(n_features).astype(np.float32) + 0.1 * np.random.randn(n_samples).astype(np.float32)
    
    result = run_benchmark(
        f"LinearRegression {desc}",
        xlearn_lm.LinearRegression(),
        sklearn.linear_model.LinearRegression(),
        X, y,
        warmup=False
    )
    results.append(result)
    
    # Numerical accuracy check
    pred_xl = xlearn_lm.LinearRegression().fit(X, y).predict(X[:100])
    pred_sk = sklearn.linear_model.LinearRegression().fit(X, y).predict(X[:100])
    mse_diff = np.mean((pred_xl - pred_sk) ** 2)
    print(f"MSE difference: {mse_diff:.2e}")
    
    del X, y

# ============================================================
# Benchmark 2: Ridge Regression
# ============================================================
print("\n" + "=" * 70)
print("📊 RIDGE REGRESSION BENCHMARKS")
print("=" * 70)

from xlearn.linear_model import Ridge
from sklearn.linear_model import Ridge as SkRidge

X = np.random.randn(50000, 2000).astype(np.float32)
y = X @ np.random.randn(2000).astype(np.float32) + 0.1 * np.random.randn(50000).astype(np.float32)

result = run_benchmark(
    "Ridge 50K × 2K (alpha=1.0)",
    Ridge(alpha=1.0),
    SkRidge(alpha=1.0),
    X, y,
    warmup=False
)
results.append(result)
del X, y

# ============================================================
# Benchmark 3: KMeans
# ============================================================
print("\n" + "=" * 70)
print("📊 KMEANS BENCHMARKS")
print("=" * 70)

km_configs = [
    (50000, 100, 10, "50K × 100, k=10"),
    (100000, 50, 20, "100K × 50, k=20"),
]

for n_samples, n_features, n_clusters, desc in km_configs:
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    
    result = run_benchmark(
        f"KMeans {desc}",
        KMeans(n_clusters=n_clusters, n_init=10, random_state=42),
        SkKMeans(n_clusters=n_clusters, n_init=10, random_state=42),
        X,
        warmup=False
    )
    results.append(result)
    del X

# ============================================================
# Benchmark 4: PCA
# ============================================================
print("\n" + "=" * 70)
print("📊 PCA BENCHMARKS")
print("=" * 70)

pca_configs = [
    (50000, 500, 50, "50K × 500 → 50"),
    (100000, 200, 20, "100K × 200 → 20"),
]

for n_samples, n_features, n_components, desc in pca_configs:
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    
    result = run_benchmark(
        f"PCA {desc}",
        PCA(n_components=n_components),
        SkPCA(n_components=n_components),
        X,
        warmup=False
    )
    results.append(result)
    del X

# ============================================================
# Benchmark 5: StandardScaler
# ============================================================
print("\n" + "=" * 70)
print("📊 STANDARDSCALER BENCHMARKS")
print("=" * 70)

X = np.random.randn(100000, 1000).astype(np.float32)

result = run_benchmark(
    "StandardScaler 100K × 1K",
    StandardScaler(),
    SkStandardScaler(),
    X,
    warmup=False
)
results.append(result)
del X

# ============================================================
# Cell 6: Summary
# ============================================================
print("\n" + "=" * 70)
print("📋 GPU BENCHMARK SUMMARY")
print("=" * 70)
print(f"\nHardware: {jax.default_backend().upper()}")
print(f"JAX Version: {jax.__version__}")
print(f"Devices: {jax.devices()}")

# Try to get GPU name
try:
    import subprocess
    result_gpu = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], 
                          capture_output=True, text=True)
    print(f"GPU: {result_gpu.stdout.strip()}")
except:
    pass

print("\n" + "-" * 80)
print(f"{'Benchmark':<40} {'XLearn':>10} {'sklearn':>10} {'Speedup':>10}")
print("-" * 80)

for r in results:
    emoji = "🚀" if r['speedup'] > 5 else "✅" if r['speedup'] > 1 else "⚠️"
    print(f"{r['name']:<40} {r['xlearn_time']:>9.3f}s {r['sklearn_time']:>9.3f}s {r['speedup']:>8.2f}x {emoji}")

print("-" * 80)

# Calculate statistics
speedups = [r['speedup'] for r in results]
avg_speedup = np.mean(speedups)
max_speedup = np.max(speedups)
min_speedup = np.min(speedups)

print(f"{'Average Speedup':<40} {'':<10} {'':<10} {avg_speedup:>8.2f}x")
print(f"{'Max Speedup':<40} {'':<10} {'':<10} {max_speedup:>8.2f}x")
print(f"{'Min Speedup':<40} {'':<10} {'':<10} {min_speedup:>8.2f}x")

print("\n" + "=" * 70)
print("✅ GPU Benchmark Complete!")
print("=" * 70)

# ============================================================
# Cell 7: Export results
# ============================================================
print("\n📝 Markdown table for README:\n")
print("| Benchmark | XLearn | sklearn | Speedup |")
print("|-----------|--------|---------|---------|")
for r in results:
    emoji = "🚀" if r['speedup'] > 5 else "" if r['speedup'] > 1 else "⚠️"
    print(f"| {r['name']} | {r['xlearn_time']:.3f}s | {r['sklearn_time']:.3f}s | **{r['speedup']:.2f}x** {emoji} |")

# Save to CSV
try:
    import pandas as pd
    df = pd.DataFrame(results)
    df['hardware'] = jax.default_backend()
    df['jax_version'] = jax.__version__
    df.to_csv('gpu_benchmark_results.csv', index=False)
    print("\n📁 Results saved to gpu_benchmark_results.csv")
except ImportError:
    print("\n(pandas not available, skipping CSV export)")

