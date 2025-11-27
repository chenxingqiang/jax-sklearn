#!/bin/bash
# Generate FL mode for all incremental learning algorithms

cd "$(dirname "$0")/.."

echo "Generating FL mode for all incremental learning algorithms..."
echo ""

total=0
success=0

# Linear models with partial_fit
algorithms=(
    "sklearn.linear_model.SGDClassifier"
    "sklearn.linear_model.SGDRegressor"
    "sklearn.linear_model.PassiveAggressiveClassifier"
    "sklearn.linear_model.PassiveAggressiveRegressor"
    "sklearn.linear_model.Perceptron"
)

# Naive Bayes with partial_fit
algorithms+=(
    "sklearn.naive_bayes.MultinomialNB"
    "sklearn.naive_bayes.BernoulliNB"
    "sklearn.naive_bayes.ComplementNB"
)

# Neural networks with partial_fit
algorithms+=(
    "sklearn.neural_network.MLPClassifier"
    "sklearn.neural_network.MLPRegressor"
)

# Clustering with partial_fit
algorithms+=(
    "sklearn.cluster.MiniBatchKMeans"
    "sklearn.cluster.Birch"
)

# Decomposition with partial_fit
algorithms+=(
    "sklearn.decomposition.IncrementalPCA"
    "sklearn.decomposition.MiniBatchDictionaryLearning"
    "sklearn.decomposition.MiniBatchNMF"
)

for algo in "${algorithms[@]}"; do
    ((total++))
    echo "[$total] Generating FL: $algo"
    if python xlearn/_secretflow/algorithm_migrator_standalone.py --algorithm "$algo" --mode fl 2>&1 | grep -q "生成适配器"; then
        ((success++))
    fi
done

echo ""
echo "======================================================================"
echo "FL Generation Complete!"
echo "======================================================================"
echo "✅ Success: $success/$total algorithms"
echo ""
echo "📊 Current totals:"
echo "  SS algorithms: $(ls xlearn/_secretflow/generated/ss_*.py 2>/dev/null | wc -l)"
echo "  FL algorithms: $(ls xlearn/_secretflow/generated/fl_*.py 2>/dev/null | wc -l)"
echo "  Total: $(($(ls xlearn/_secretflow/generated/ss_*.py 2>/dev/null | wc -l) + $(ls xlearn/_secretflow/generated/fl_*.py 2>/dev/null | wc -l)))"
echo ""

