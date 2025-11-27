#!/bin/bash
# Phase 3 Algorithm Migration Script
# 高级算法扩展

echo "======================================================================"
echo " Phase 3: 高级算法扩展到 SecretFlow"
echo "======================================================================"
echo ""

cd "$(dirname "$0")/.."

total=0
success=0

# 集成学习 - 树模型 (6个)
echo "======================================================================"
echo " Ensemble Learning - Tree Models (6 个)"
echo "======================================================================"
echo ""

algorithms=(
    "sklearn.tree.DecisionTreeClassifier"
    "sklearn.tree.DecisionTreeRegressor"
    "sklearn.ensemble.RandomForestClassifier"
    "sklearn.ensemble.RandomForestRegressor"
    "sklearn.ensemble.ExtraTreesClassifier"
    "sklearn.ensemble.ExtraTreesRegressor"
)

for algo in "${algorithms[@]}"; do
    ((total++))
    echo "[$total] Migrating $algo..."
    if python xlearn/_secretflow/algorithm_migrator_standalone.py --algorithm "$algo" --mode ss; then
        ((success++))
    fi
    echo ""
done

# 集成学习 - Boosting (4个)
echo "======================================================================"
echo " Ensemble Learning - Boosting (4 个)"
echo "======================================================================"
echo ""

algorithms=(
    "sklearn.ensemble.AdaBoostClassifier"
    "sklearn.ensemble.AdaBoostRegressor"
    "sklearn.ensemble.GradientBoostingClassifier"
    "sklearn.ensemble.GradientBoostingRegressor"
)

for algo in "${algorithms[@]}"; do
    ((total++))
    echo "[$total] Migrating $algo..."
    if python xlearn/_secretflow/algorithm_migrator_standalone.py --algorithm "$algo" --mode ss; then
        ((success++))
    fi
    echo ""
done

# 集成学习 - Bagging & Voting (4个)
echo "======================================================================"
echo " Ensemble Learning - Bagging & Voting (4 个)"
echo "======================================================================"
echo ""

algorithms=(
    "sklearn.ensemble.BaggingClassifier"
    "sklearn.ensemble.BaggingRegressor"
    "sklearn.ensemble.VotingClassifier"
    "sklearn.ensemble.VotingRegressor"
)

for algo in "${algorithms[@]}"; do
    ((total++))
    echo "[$total] Migrating $algo..."
    if python xlearn/_secretflow/algorithm_migrator_standalone.py --algorithm "$algo" --mode ss; then
        ((success++))
    fi
    echo ""
done

# 核方法 (3个) - 暂时跳过 SVC/SVR，因为它们可能需要特殊处理
echo "======================================================================"
echo " Kernel Methods (3 个)"
echo "======================================================================"
echo ""

algorithms=(
    "sklearn.decomposition.KernelPCA"
    "sklearn.kernel_ridge.KernelRidge"
    "sklearn.gaussian_process.GaussianProcessRegressor"
)

for algo in "${algorithms[@]}"; do
    ((total++))
    echo "[$total] Migrating $algo..."
    if python xlearn/_secretflow/algorithm_migrator_standalone.py --algorithm "$algo" --mode ss; then
        ((success++))
    fi
    echo ""
done

# 神经网络 (2个)
echo "======================================================================"
echo " Neural Networks (2 个)"
echo "======================================================================"
echo ""

algorithms=(
    "sklearn.neural_network.MLPClassifier"
    "sklearn.neural_network.MLPRegressor"
)

for algo in "${algorithms[@]}"; do
    ((total++))
    echo "[$total] Migrating $algo..."
    if python xlearn/_secretflow/algorithm_migrator_standalone.py --algorithm "$algo" --mode ss; then
        ((success++))
    fi
    echo ""
done

# 高级预处理 (5个)
echo "======================================================================"
echo " Advanced Preprocessing (5 个)"
echo "======================================================================"
echo ""

algorithms=(
    "sklearn.preprocessing.PolynomialFeatures"
    "sklearn.preprocessing.SplineTransformer"
    "sklearn.preprocessing.KBinsDiscretizer"
    "sklearn.preprocessing.LabelEncoder"
    "sklearn.preprocessing.OrdinalEncoder"
)

for algo in "${algorithms[@]}"; do
    ((total++))
    echo "[$total] Migrating $algo..."
    if python xlearn/_secretflow/algorithm_migrator_standalone.py --algorithm "$algo" --mode ss; then
        ((success++))
    fi
    echo ""
done

# 其他分类器 (3个)
echo "======================================================================"
echo " Other Classifiers (3 个)"
echo "======================================================================"
echo ""

algorithms=(
    "sklearn.linear_model.Perceptron"
    "sklearn.dummy.DummyClassifier"
    "sklearn.dummy.DummyRegressor"
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
echo " Phase 3 完成!"
echo "======================================================================"
echo "✅ 成功: $success/$total 个算法"
echo "📁 输出目录: xlearn/_secretflow/generated/"
echo ""
echo "📊 累计统计:"
echo "  Phase 1: 12 个算法"
echo "  Phase 2: 20 个算法"
echo "  Phase 3: $success 个算法"
echo "  总计: $((12 + 20 + success)) 个算法"
echo ""
echo "🎉 SecretFlow 算法生态扩展完成!"
echo ""

