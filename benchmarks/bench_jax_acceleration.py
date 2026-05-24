"""Benchmark JAX acceleration — large-scale tests."""
import time, warnings, sys
import numpy as np
from xlearn.datasets import make_regression, make_classification
from xlearn.model_selection import train_test_split
warnings.filterwarnings("ignore")
import xlearn
for mod in ['gaussian_process', 'svm', 'neural_network', 'tree']:
    getattr(xlearn, mod)

def warmup():
    import jax.numpy as jnp; import jax
    jax.jit(lambda x: x.T @ x)(jnp.ones((100,10))).block_until_ready()

def bench(label, fn_jax, fn_orig, n_repeat=5):
    best_jax = min((time.perf_counter(), fn_jax())[0] for _ in range(3))
    # Actually measure properly
    # warmup
    fn_jax(); fn_orig()
    t0 = time.perf_counter()
    for _ in range(n_repeat): fn_jax()
    t_jax = (time.perf_counter() - t0) / n_repeat
    t0 = time.perf_counter()
    for _ in range(n_repeat): fn_orig()
    t_orig = (time.perf_counter() - t0) / n_repeat
    speedup = t_orig / t_jax if t_jax > 0 else float('inf')
    tag = "🚀" if speedup > 1.15 else ("✅" if speedup > 1.0 else "⚠️")
    print(f"  {label:<35s} JAX={t_jax:.4f}s  Orig={t_orig:.4f}s  {tag} {speedup:.2f}x")

print("="*60)
print("JAX Acceleration Benchmarks (Large Data)")
print("="*60)
print("Warming up JAX...", end=" ", flush=True); warmup(); print("Done.\n")

# 1. GPR fit (larger)
print("GaussianProcessRegressor fit")
from xlearn.gaussian_process.kernels import RBF, ConstantKernel as C
GPR = xlearn.gaussian_process.GaussianProcessRegressor
for n in [2000]:
    X, y = make_regression(n_samples=n, n_features=20, noise=0.1, random_state=42)
    def run(enable):
        with xlearn._jax.config_context(enable_jax=enable, fallback_on_error=True):
            gp = GPR(kernel=C() * RBF(), optimizer=None, random_state=42)
            gp.fit(X, y)
            return gp
    bench(f"n={n}", lambda: run(True), lambda: run(False))

# 2. GPR predict
print("\nGPR predict (return_std=True)")
X, y = make_regression(n_samples=2000, n_features=20, noise=0.1, random_state=42)
X_tr, X_te = X[:1500], X[1500:]
def mk_gp(enable):
    with xlearn._jax.config_context(enable_jax=enable, fallback_on_error=True):
        gp = GPR(kernel=C() * RBF(), optimizer=None, random_state=42)
        gp.fit(X_tr, y[:1500])
        return gp
gp_on = mk_gp(True); gp_off = mk_gp(False)
bench("predict 500 samples", lambda: gp_on.predict(X_te, return_std=True), lambda: gp_off.predict(X_te, return_std=True))

# 3. SVC predict (large)
print("\nSVC predict")
SVC = xlearn.svm.SVC
X, y = make_classification(n_samples=10000, n_features=100, n_informative=50, random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=5000, random_state=42)
for kernel in ['rbf', 'linear']:
    def mk_svc(enable):
        with xlearn._jax.config_context(enable_jax=enable, fallback_on_error=True):
            svc = SVC(kernel=kernel, gamma='scale', random_state=42)
            svc.fit(X_tr, y_tr)
            return svc
    svc_on = mk_svc(True); svc_off = mk_svc(False)
    bench(f"{kernel} predict 5000", lambda: svc_on.predict(X_te), lambda: svc_off.predict(X_te), n_repeat=10)

# 4. MLP predict
print("\nMLPClassifier predict")
MLP = xlearn.neural_network.MLPClassifier
X, y = make_classification(n_samples=10000, n_features=100, n_classes=2, n_informative=50, random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=5000, random_state=42)
def mk_mlp(enable):
    with xlearn._jax.config_context(enable_jax=enable, fallback_on_error=True):
        mlp = MLP(hidden_layer_sizes=(100, 50), max_iter=5, random_state=42)
        mlp.fit(X_tr, y_tr)
        return mlp
mlp_on = mk_mlp(True); mlp_off = mk_mlp(False)
bench("predict 5000", lambda: mlp_on.predict(X_te), lambda: mlp_off.predict(X_te), n_repeat=50)

# 5. JAX kernel microbenchmark (raw speed comparison)
print("\n── Raw JAX kernel microbenchmarks ──")
import jax.numpy as jnp
import jax
n, d = 5000, 100
X = np.random.randn(n, d).astype(np.float32)
SV = np.random.randn(2000, d).astype(np.float32)

# RBF kernel
@jax.jit
def rbf_jax(X, SV, gamma=0.01):
    X2 = jnp.sum(X**2, axis=1, keepdims=True)
    SV2 = jnp.sum(SV**2, axis=1, keepdims=True)
    return jnp.exp(-gamma * jnp.maximum(X2 + SV2.T - 2.0 * (X @ SV.T), 0.0))

def rbf_np(X, SV, gamma=0.01):
    from scipy.spatial.distance import cdist
    return np.exp(-gamma * cdist(X, SV, 'sqeuclidean'))

# Warmup
rbf_jax(jnp.array(X), jnp.array(SV)).block_until_ready()
rbf_np(X[:100], SV[:100])

t0 = time.perf_counter(); _ = rbf_jax(jnp.array(X), jnp.array(SV)); t1 = time.perf_counter()
_ = rbf_jax(jnp.array(X), jnp.array(SV)); t2 = time.perf_counter()
t_jax = time.perf_counter() - t1
t0 = time.perf_counter(); rbf_np(X, SV); t_np = time.perf_counter() - t0
print(f"  RBF kernel 5K×100         JAX={t_jax:.4f}s  NumPy={t_np:.4f}s  {'🚀' if t_np/t_jax>1 else '⚠️'} {t_np/t_jax:.2f}x")

# Matrix multiply microbenchmark
A = np.random.randn(5000, 5000).astype(np.float32)
B = np.random.randn(5000, 5000).astype(np.float32)
@jax.jit
def mm_jax(A, B): return A @ B
mm_jax(jnp.array(A), jnp.array(B)).block_until_ready()
t0 = time.perf_counter(); mm_jax(jnp.array(A), jnp.array(B)); t_jax = time.perf_counter() - t0
t0 = time.perf_counter(); A @ B; t_np = time.perf_counter() - t0
print(f"  MatMul 5000×5000          JAX={t_jax:.4f}s  NumPy={t_np:.4f}s  {'🚀' if t_np/t_jax>1 else '⚠️'} {t_np/t_jax:.2f}x")

print("\nDone!")
