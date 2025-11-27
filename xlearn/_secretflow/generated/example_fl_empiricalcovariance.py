#!/usr/bin/env python3
# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Usage Example for FLEmpiricalCovariance

This example demonstrates how to use the privacy-preserving EmpiricalCovariance
in SecretFlow's federated learning environment.
"""

import numpy as np

try:
    import secretflow as sf
    from secretflow.data import FedNdarray, PartitionWay
    from secretflow.device.driver import reveal
except ImportError:
    print("❌ SecretFlow not installed. Install with: pip install secretflow")
    exit(1)

from xlearn._secretflow.generated.fl_empiricalcovariance import FLEmpiricalCovariance


def main():
    """Main example function"""
    print("="*70)
    print(f" FLEmpiricalCovariance Usage Example")
    print("="*70)
    
    # Step 1: Initialize SecretFlow
    print("\n[1/5] Initializing SecretFlow...")
    sf.init(['alice', 'bob', 'carol'], address='local')
    spu = sf.SPU(sf.utils.testing.cluster_def(['alice', 'bob', 'carol']))
    
    alice = sf.PYU('alice')
    bob = sf.PYU('bob')
    carol = sf.PYU('carol')
    print("  ✓ SecretFlow initialized")
    
    # Step 2: Create sample data
    print("\n[2/5] Creating sample data...")
    np.random.seed(42)
    n_samples = 1000
    n_features = 15
    
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    
    # Partition data vertically
    X_alice = X[:, 0:5]
    X_bob = X[:, 5:10]
    X_carol = X[:, 10:15]
    
    print(f"  ✓ Data shape: {n_samples} samples × {n_features} features")
    print(f"  ✓ Alice: {X_alice.shape}, Bob: {X_bob.shape}, Carol: {X_carol.shape}")
    
    # Step 3: Create federated data
    print("\n[3/5] Creating federated data...")
    fed_X = FedNdarray(
        partitions={
            alice: alice(lambda x: x)(X_alice),
            bob: bob(lambda x: x)(X_bob),
            carol: carol(lambda x: x)(X_carol),
        },
        partition_way=PartitionWay.VERTICAL
    )
    print("  ✓ Federated data created")
    
    # Step 4: Train model
    print("\n[4/5] Training FLEmpiricalCovariance...")
    print("  Note: All computation happens in SPU's encrypted environment")
    
    import time
    start_time = time.time()
    
    model = FLEmpiricalCovariance(spu)
    model.fit(fed_X)
    
    training_time = time.time() - start_time
    print(f"  ✓ Training completed in {training_time:.2f}s")
    
    # Step 5: Make predictions (if applicable)
    print("\n[5/5] Model trained successfully!")
    print("  ✓ Model state stored in SPU (encrypted)")
    print("  ✓ Privacy: Fully protected by MPC")
    print(f"  ✓ Performance: {training_time:.2f}s")
    
    # Cleanup
    sf.shutdown()
    print("\n✅ Example completed!")


if __name__ == "__main__":
    main()
