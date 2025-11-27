#!/bin/bash
# Expand more high-value algorithms

cd "$(dirname "$0")/.."

echo "Adding more high-value algorithms..."
echo ""

# SVM algorithms
python xlearn/_secretflow/algorithm_migrator_standalone.py --algorithm sklearn.svm.LinearSVC --mode ss
python xlearn/_secretflow/algorithm_migrator_standalone.py --algorithm sklearn.svm.LinearSVR --mode ss
python xlearn/_secretflow/algorithm_migrator_standalone.py --algorithm sklearn.svm.NuSVC --mode ss
python xlearn/_secretflow/algorithm_migrator_standalone.py --algorithm sklearn.svm.NuSVR --mode ss

# Linear models with CV
python xlearn/_secretflow/algorithm_migrator_standalone.py --algorithm sklearn.linear_model.RidgeCV --mode ss
python xlearn/_secretflow/algorithm_migrator_standalone.py --algorithm sklearn.linear_model.LassoCV --mode ss
python xlearn/_secretflow/algorithm_migrator_standalone.py --algorithm sklearn.linear_model.ElasticNetCV --mode ss

# Outlier detection
python xlearn/_secretflow/algorithm_migrator_standalone.py --algorithm sklearn.covariance.EllipticEnvelope --mode ss
python xlearn/_secretflow/algorithm_migrator_standalone.py --algorithm sklearn.neighbors.LocalOutlierFactor --mode ss

# Semi-supervised
python xlearn/_secretflow/algorithm_migrator_standalone.py --algorithm sklearn.semi_supervised.LabelPropagation --mode ss
python xlearn/_secretflow/algorithm_migrator_standalone.py --algorithm sklearn.semi_supervised.LabelSpreading --mode ss

# Feature selection
python xlearn/_secretflow/algorithm_migrator_standalone.py --algorithm sklearn.feature_selection.VarianceThreshold --mode ss
python xlearn/_secretflow/algorithm_migrator_standalone.py --algorithm sklearn.feature_selection.SelectKBest --mode ss

echo ""
echo "✅ Expansion complete!"
ls xlearn/_secretflow/generated/ss_*.py | wc -l

