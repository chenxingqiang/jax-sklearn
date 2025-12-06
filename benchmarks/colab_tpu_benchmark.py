"""
JAX-sklearn TPU Benchmark for Google Colab

Run this script in Google Colab with TPU runtime:
1. Go to Runtime -> Change runtime type -> TPU
2. Run the cells below

Or copy this entire script into a Colab notebook.
"""

# ============================================================
# Cell 1: Install dependencies
# ============================================================
print("=" * 70)
print("Step 1: Installing dependencies...")
print("=" * 70)


print("✅ Dependencies installed!")

# ============================================================
# Cell 2: Verify TPU setup
# ============================================================
print("\n" + "=" * 70)
print("Step 2: Verifying TPU setup...")
print("=" * 70)

import jax
print(f"JAX version: {jax.__version__}")
print(f"JAX backend: {jax.default_backend()}")
print(f"JAX devices: {jax.devices()}")

# Check if TPU is available
if jax.default_backend() == 'tpu':
    print("✅ TPU is available!")
else:
    print("⚠️ TPU not detected. Make sure you selected TPU runtime.")
    print("   Go to: Runtime -> Change runtime type -> TPU")

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
import sklearn.linear_model
from sklearn.cluster import KMeans as SkKMeans
from sklearn.decomposition import PCA as SkPCA

print(f"XLearn version: {xlearn.__version__}")
print(f"JAX enabled: {xlearn._JAX_ENABLED}")
print("✅ Libraries imported!")

# ============================================================
# Cell 4: Benchmark function
# ============================================================
def run_benchmark(name, xlearn_model, sklearn_model, X, y=None, fit_method='fit'):
    """Run a single benchmark comparing XLearn and sklearn."""
    print(f"\n{'='*60}")
    print(f"Benchmark: {name}")
    print(f"Data shape: {X.shape}, Complexity: {X.shape[0] * X.shape[1]:.2e}")
    print("-" * 60)
    
    # XLearn benchmark
    if fit_method == 'fit':
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
    print(f"Speedup: {speedup:.2f}x")
    
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
print("Step 4: Running TPU Benchmarks")
print("=" * 70)

results = []

# Benchmark 1: Linear Regression - At threshold (1e8)
print("\n📊 Linear Regression Benchmarks")
print("-" * 60)

configs = [
    (10000, 10000, "10K × 10K (1e8)"),
    (50000, 2000, "50K × 2K (1e8)"),
    (100000, 1000, "100K × 1K (1e8)"),
    (100000, 2000, "100K × 2K (2e8)"),
]

for n_samples, n_features, desc in configs:
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    y = X @ np.random.randn(n_features).astype(np.float32) + 0.1 * np.random.randn(n_samples).astype(np.float32)
    
    result = run_benchmark(
        f"LinearRegression {desc}",
        xlearn_lm.LinearRegression(),
        sklearn.linear_model.LinearRegression(),
        X, y
    )
    results.append(result)
    
    # Clean up memory
    del X, y

# Benchmark 2: KMeans
print("\n📊 KMeans Benchmarks")
print("-" * 60)

X_km = np.random.randn(50000, 100).astype(np.float32)
result = run_benchmark(
    "KMeans 50K × 100 (k=10)",
    KMeans(n_clusters=10, n_init=10, random_state=42),
    SkKMeans(n_clusters=10, n_init=10, random_state=42),
    X_km
)
results.append(result)
del X_km

# Benchmark 3: PCA
print("\n📊 PCA Benchmarks")
print("-" * 60)

X_pca = np.random.randn(50000, 500).astype(np.float32)
result = run_benchmark(
    "PCA 50K × 500 → 50",
    PCA(n_components=50),
    SkPCA(n_components=50),
    X_pca
)
results.append(result)
del X_pca

# ============================================================
# Cell 6: Summary
# ============================================================
print("\n" + "=" * 70)
print("📋 BENCHMARK SUMMARY")
print("=" * 70)
print(f"\nHardware: {jax.default_backend().upper()}")
print(f"JAX Version: {jax.__version__}")
print(f"Devices: {jax.devices()}")

print("\n" + "-" * 70)
print(f"{'Benchmark':<35} {'XLearn':>10} {'sklearn':>10} {'Speedup':>10}")
print("-" * 70)

for r in results:
    print(f"{r['name']:<35} {r['xlearn_time']:>9.3f}s {r['sklearn_time']:>9.3f}s {r['speedup']:>9.2f}x")

print("-" * 70)

# Calculate average speedup
avg_speedup = np.mean([r['speedup'] for r in results])
print(f"{'Average Speedup':<35} {'':<10} {'':<10} {avg_speedup:>9.2f}x")

print("\n" + "=" * 70)
print("✅ TPU Benchmark Complete!")
print("=" * 70)

# ============================================================
# Cell 7: Export results
# ============================================================
# Save results to CSV for later analysis
import pandas as pd

df = pd.DataFrame(results)
df['hardware'] = jax.default_backend()
df['jax_version'] = jax.__version__
df.to_csv('tpu_benchmark_results.csv', index=False)
print("\n📁 Results saved to tpu_benchmark_results.csv")

# Print markdown table for README
print("\n📝 Markdown table for README:")
print("\n| Benchmark | XLearn | sklearn | Speedup |")
print("|-----------|--------|---------|---------|")
for r in results:
    print(f"| {r['name']} | {r['xlearn_time']:.3f}s | {r['sklearn_time']:.3f}s | **{r['speedup']:.2f}x** |")

