#!/usr/bin/env python3
"""
Phase 2 Algorithm Migration
扩展更多算法到 SecretFlow
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from xlearn._secretflow.algorithm_migrator_standalone import StandaloneAlgorithmMigrator

# Import algorithms to migrate
from sklearn.preprocessing import (
    RobustScaler, MaxAbsScaler, 
    QuantileTransformer, PowerTransformer,
    Normalizer, Binarizer
)
from sklearn.linear_model import (
    SGDClassifier, SGDRegressor,
    PassiveAggressiveClassifier, PassiveAggressiveRegressor,
    HuberRegressor, RANSACRegressor,
    RidgeClassifier
)
from sklearn.cluster import (
    AgglomerativeClustering, SpectralClustering,
    MeanShift, AffinityPropagation, Birch
)
from sklearn.ensemble import (
    IsolationForest, 
)
from sklearn.neighbors import (
    RadiusNeighborsClassifier, RadiusNeighborsRegressor,
    NearestCentroid
)
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
)


def main():
    """Phase 2 批量迁移"""
    
    print("="*70)
    print(" Phase 2: 扩展更多算法到 SecretFlow")
    print("="*70)
    
    migrator = StandaloneAlgorithmMigrator()
    
    # 定义要迁移的算法
    algorithms = [
        # ========== 预处理算法 (6个) ==========
        ("Preprocessing", [
            (RobustScaler, "ss"),
            (MaxAbsScaler, "ss"),
            (QuantileTransformer, "ss"),
            (PowerTransformer, "ss"),
            (Normalizer, "ss"),
            (Binarizer, "ss"),
        ]),
        
        # ========== 回归算法 (4个) ==========
        ("Regression", [
            (SGDRegressor, "fl"),  # FL 模式，增量学习
            (HuberRegressor, "ss"),
            (RANSACRegressor, "ss"),
            (PassiveAggressiveRegressor, "fl"),
        ]),
        
        # ========== 分类算法 (5个) ==========
        ("Classification", [
            (SGDClassifier, "fl"),  # FL 模式，增量学习
            (PassiveAggressiveClassifier, "fl"),
            (RidgeClassifier, "ss"),
            (LinearDiscriminantAnalysis, "ss"),
            (QuadraticDiscriminantAnalysis, "ss"),
        ]),
        
        # ========== 聚类算法 (5个) ==========
        ("Clustering", [
            (AgglomerativeClustering, "ss"),
            (SpectralClustering, "ss"),
            (MeanShift, "ss"),
            (AffinityPropagation, "ss"),
            (Birch, "ss"),
        ]),
        
        # ========== 异常检测 (1个) ==========
        ("Anomaly Detection", [
            (IsolationForest, "ss"),
        ]),
        
        # ========== 最近邻 (3个) ==========
        ("Neighbors", [
            (RadiusNeighborsClassifier, "ss"),
            (RadiusNeighborsRegressor, "ss"),
            (NearestCentroid, "ss"),
        ]),
    ]
    
    total_count = sum(len(algs) for _, algs in algorithms)
    current = 0
    success_count = 0
    
    print(f"\n📊 总计: {total_count} 个算法\n")
    
    # 按类别迁移
    for category, algs in algorithms:
        print(f"\n{'='*70}")
        print(f" {category} ({len(algs)} 个算法)")
        print(f"{'='*70}\n")
        
        for sklearn_class, mode in algs:
            current += 1
            try:
                print(f"[{current}/{total_count}] ", end="")
                migrator.migrate_algorithm(sklearn_class, mode, use_xlearn=True)
                success_count += 1
                print()
            except Exception as e:
                print(f"❌ 迁移失败: {e}\n")
                continue
    
    # 总结
    print("\n" + "="*70)
    print(f" Phase 2 完成!")
    print("="*70)
    print(f"✅ 成功: {success_count}/{total_count} 个算法")
    print(f"📁 输出目录: xlearn/_secretflow/generated/")
    print()
    
    # 显示新增算法总数
    print("📊 算法库扩展统计:")
    print(f"  Phase 1: 12 个算法")
    print(f"  Phase 2: {success_count} 个算法")
    print(f"  总计: {12 + success_count} 个算法")
    print()


if __name__ == "__main__":
    main()

