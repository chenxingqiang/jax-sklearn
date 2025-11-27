#!/bin/bash
# Phase 2 Algorithm Migration Script

echo "======================================================================"
echo " Phase 2: 扩展更多算法到 SecretFlow"
echo "======================================================================"
echo ""

cd "$(dirname "$0")/.."

total=0
success=0

# 预处理算法 (6个)
echo "======================================================================"
echo " Preprocessing Algorithms (6 个)"
echo "======================================================================"
echo ""

algorithms=(
    "sklearn.preprocessing.RobustScaler"
    "sklearn.preprocessing.MaxAbsScaler"
    "sklearn.preprocessing.QuantileTransformer"
    "sklearn.preprocessing.PowerTransformer"
    "sklearn.preprocessing.Normalizer"
    "sklearn.preprocessing.Binarizer"
)

for algo in "${algorithms[@]}"; do
    ((total++))
    echo "[$total] Migrating $algo..."
    if python xlearn/_secretflow/algorithm_migrator_standalone.py --algorithm "$algo" --mode ss; then
        ((success++))
    fi
    echo ""
done

# 回归算法 (4个)
echo "======================================================================"
echo " Regression Algorithms (4 个)"
echo "======================================================================"
echo ""

algorithms=(
    "sklearn.linear_model.SGDRegressor"
    "sklearn.linear_model.HuberRegressor"
    "sklearn.linear_model.RANSACRegressor"
    "sklearn.linear_model.PassiveAggressiveRegressor"
)

modes=("fl" "ss" "ss" "fl")

for i in "${!algorithms[@]}"; do
    ((total++))
    algo="${algorithms[$i]}"
    mode="${modes[$i]}"
    echo "[$total] Migrating $algo (mode=$mode)..."
    if python xlearn/_secretflow/algorithm_migrator_standalone.py --algorithm "$algo" --mode "$mode"; then
        ((success++))
    fi
    echo ""
done

# 分类算法 (5个)
echo "======================================================================"
echo " Classification Algorithms (5 个)"
echo "======================================================================"
echo ""

algorithms=(
    "sklearn.linear_model.SGDClassifier"
    "sklearn.linear_model.PassiveAggressiveClassifier"
    "sklearn.linear_model.RidgeClassifier"
    "sklearn.discriminant_analysis.LinearDiscriminantAnalysis"
    "sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis"
)

modes=("fl" "fl" "ss" "ss" "ss")

for i in "${!algorithms[@]}"; do
    ((total++))
    algo="${algorithms[$i]}"
    mode="${modes[$i]}"
    echo "[$total] Migrating $algo (mode=$mode)..."
    if python xlearn/_secretflow/algorithm_migrator_standalone.py --algorithm "$algo" --mode "$mode"; then
        ((success++))
    fi
    echo ""
done

# 聚类算法 (5个)
echo "======================================================================"
echo " Clustering Algorithms (5 个)"
echo "======================================================================"
echo ""

algorithms=(
    "sklearn.cluster.AgglomerativeClustering"
    "sklearn.cluster.SpectralClustering"
    "sklearn.cluster.MeanShift"
    "sklearn.cluster.AffinityPropagation"
    "sklearn.cluster.Birch"
)

for algo in "${algorithms[@]}"; do
    ((total++))
    echo "[$total] Migrating $algo..."
    if python xlearn/_secretflow/algorithm_migrator_standalone.py --algorithm "$algo" --mode ss; then
        ((success++))
    fi
    echo ""
done

# 异常检测 (1个)
echo "======================================================================"
echo " Anomaly Detection (1 个)"
echo "======================================================================"
echo ""

((total++))
echo "[$total] Migrating sklearn.ensemble.IsolationForest..."
if python xlearn/_secretflow/algorithm_migrator_standalone.py --algorithm "sklearn.ensemble.IsolationForest" --mode ss; then
    ((success++))
fi
echo ""

# 最近邻 (3个)
echo "======================================================================"
echo " Neighbors Algorithms (3 个)"
echo "======================================================================"
echo ""

algorithms=(
    "sklearn.neighbors.RadiusNeighborsClassifier"
    "sklearn.neighbors.RadiusNeighborsRegressor"
    "sklearn.neighbors.NearestCentroid"
)

for algo in "${algorithms[@]}"; do
    ((total++))
    echo "[$total] Migrating $algo..."
    if python xlearn/_secretflow/algorithm_migrator_standalone.py --algorithm "$algo" --mode ss; then
        ((success++))
    fi
    echo ""
done

# 总结
echo ""
echo "======================================================================"
echo " Phase 2 完成!"
echo "======================================================================"
echo "✅ 成功: $success/$total 个算法"
echo "📁 输出目录: xlearn/_secretflow/generated/"
echo ""
echo "📊 算法库扩展统计:"
echo "  Phase 1: 12 个算法"
echo "  Phase 2: $success 个算法"
echo "  总计: $((12 + success)) 个算法"
echo ""

