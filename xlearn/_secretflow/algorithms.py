# Author: Chen Xingqiang
# SPDX-License-Identifier: BSD-3-Clause

"""
SecretFlow Algorithm Registry

Provides convenient access to all migrated algorithms.

Usage:
    from xlearn._secretflow.algorithms import get_algorithm, list_algorithms
    
    # Get algorithm by name
    PCA = get_algorithm("PCA")
    model = PCA(spu, n_components=10)
    
    # List all available algorithms
    algorithms = list_algorithms()
"""

import importlib
from typing import Dict, List, Optional


# ============================================================================
# Algorithm Registry
# ============================================================================

ALGORITHM_REGISTRY = {
    # ========== 降维算法 (Decomposition) ==========
    "PCA": {
        "module": "xlearn._secretflow.generated.ss_pca",
        "class": "SSPCA",
        "type": "decomposition",
        "mode": "ss",
        "description": "主成分分析 - 线性降维",
    },
    "TruncatedSVD": {
        "module": "xlearn._secretflow.generated.ss_truncatedsvd",
        "class": "SSTruncatedSVD",
        "type": "decomposition",
        "mode": "ss",
        "description": "截断奇异值分解 - 稀疏数据降维",
    },
    "NMF": {
        "module": "xlearn._secretflow.generated.ss_nmf",
        "class": "SSNMF",
        "type": "decomposition",
        "mode": "ss",
        "description": "非负矩阵分解 - 非负数据降维",
    },
    "FactorAnalysis": {
        "module": "xlearn._secretflow.generated.ss_factoranalysis",
        "class": "SSFactorAnalysis",
        "type": "decomposition",
        "mode": "ss",
        "description": "因子分析 - 隐变量建模",
    },
    "FastICA": {
        "module": "xlearn._secretflow.generated.ss_fastica",
        "class": "SSFastICA",
        "type": "decomposition",
        "mode": "ss",
        "description": "独立成分分析 - 信号分离",
    },
    
    # ========== 回归算法 (Regression) ==========
    "Ridge": {
        "module": "xlearn._secretflow.generated.ss_ridge",
        "class": "SSRidge",
        "type": "regression",
        "mode": "ss",
        "description": "岭回归 - L2 正则化线性回归",
    },
    "Lasso": {
        "module": "xlearn._secretflow.generated.ss_lasso",
        "class": "SSLasso",
        "type": "regression",
        "mode": "ss",
        "description": "Lasso 回归 - L1 正则化线性回归",
    },
    "ElasticNet": {
        "module": "xlearn._secretflow.generated.ss_elasticnet",
        "class": "SSElasticNet",
        "type": "regression",
        "mode": "ss",
        "description": "弹性网络 - L1+L2 正则化回归",
    },
    "HuberRegressor": {
        "module": "xlearn._secretflow.generated.ss_huberregressor",
        "class": "SSHuberRegressor",
        "type": "regression",
        "mode": "ss",
        "description": "Huber 回归 - 鲁棒线性回归",
    },
    "RANSACRegressor": {
        "module": "xlearn._secretflow.generated.ss_ransacregressor",
        "class": "SSRANSACRegressor",
        "type": "regression",
        "mode": "ss",
        "description": "RANSAC 回归 - 鲁棒拟合",
    },
    
    # ========== 聚类算法 (Clustering) ==========
    "MiniBatchKMeans": {
        "module": "xlearn._secretflow.generated.ss_minibatchkmeans",
        "class": "SSMiniBatchKMeans",
        "type": "clustering",
        "mode": "ss",
        "description": "小批量 K-Means - 大规模聚类",
    },
    "DBSCAN": {
        "module": "xlearn._secretflow.generated.ss_dbscan",
        "class": "SSDBSCAN",
        "type": "clustering",
        "mode": "ss",
        "description": "DBSCAN - 基于密度的聚类",
    },
    "AgglomerativeClustering": {
        "module": "xlearn._secretflow.generated.ss_agglomerativeclustering",
        "class": "SSAgglomerativeClustering",
        "type": "clustering",
        "mode": "ss",
        "description": "层次聚类 - 自底向上聚类",
    },
    "SpectralClustering": {
        "module": "xlearn._secretflow.generated.ss_spectralclustering",
        "class": "SSSpectralClustering",
        "type": "clustering",
        "mode": "ss",
        "description": "谱聚类 - 基于图的聚类",
    },
    "MeanShift": {
        "module": "xlearn._secretflow.generated.ss_meanshift",
        "class": "SSMeanShift",
        "type": "clustering",
        "mode": "ss",
        "description": "均值漂移 - 无需指定簇数",
    },
    "AffinityPropagation": {
        "module": "xlearn._secretflow.generated.ss_affinitypropagation",
        "class": "SSAffinityPropagation",
        "type": "clustering",
        "mode": "ss",
        "description": "亲和传播 - 基于消息传递",
    },
    "Birch": {
        "module": "xlearn._secretflow.generated.ss_birch",
        "class": "SSBirch",
        "type": "clustering",
        "mode": "ss",
        "description": "BIRCH - 增量聚类",
    },
    
    # ========== 分类算法 (Classification) ==========
    "MultinomialNB": {
        "module": "xlearn._secretflow.generated.ss_multinomialnb",
        "class": "SSMultinomialNB",
        "type": "classification",
        "mode": "ss",
        "description": "多项式朴素贝叶斯 - 文本分类",
    },
    "BernoulliNB": {
        "module": "xlearn._secretflow.generated.ss_bernoullinb",
        "class": "SSBernoulliNB",
        "type": "classification",
        "mode": "ss",
        "description": "伯努利朴素贝叶斯 - 二值特征分类",
    },
    "RidgeClassifier": {
        "module": "xlearn._secretflow.generated.ss_ridgeclassifier",
        "class": "SSRidgeClassifier",
        "type": "classification",
        "mode": "ss",
        "description": "岭分类器 - L2 正则化分类",
    },
    "LinearDiscriminantAnalysis": {
        "module": "xlearn._secretflow.generated.ss_lineardiscriminantanalysis",
        "class": "SSLinearDiscriminantAnalysis",
        "type": "classification",
        "mode": "ss",
        "description": "线性判别分析 - 监督降维+分类",
    },
    "QuadraticDiscriminantAnalysis": {
        "module": "xlearn._secretflow.generated.ss_quadraticdiscriminantanalysis",
        "class": "SSQuadraticDiscriminantAnalysis",
        "type": "classification",
        "mode": "ss",
        "description": "二次判别分析 - 非线性分类",
    },
    
    # ========== 预处理算法 (Preprocessing) ==========
    "RobustScaler": {
        "module": "xlearn._secretflow.generated.ss_robustscaler",
        "class": "SSRobustScaler",
        "type": "preprocessing",
        "mode": "ss",
        "description": "鲁棒标准化 - 抗异常值",
    },
    "MaxAbsScaler": {
        "module": "xlearn._secretflow.generated.ss_maxabsscaler",
        "class": "SSMaxAbsScaler",
        "type": "preprocessing",
        "mode": "ss",
        "description": "最大绝对值标准化 - 稀疏数据",
    },
    "QuantileTransformer": {
        "module": "xlearn._secretflow.generated.ss_quantiletransformer",
        "class": "SSQuantileTransformer",
        "type": "preprocessing",
        "mode": "ss",
        "description": "分位数转换 - 均匀/正态分布转换",
    },
    "PowerTransformer": {
        "module": "xlearn._secretflow.generated.ss_powertransformer",
        "class": "SSPowerTransformer",
        "type": "preprocessing",
        "mode": "ss",
        "description": "幂变换 - Yeo-Johnson/Box-Cox",
    },
    "Normalizer": {
        "module": "xlearn._secretflow.generated.ss_normalizer",
        "class": "SSNormalizer",
        "type": "preprocessing",
        "mode": "ss",
        "description": "样本归一化 - L1/L2/Max 范数",
    },
    "Binarizer": {
        "module": "xlearn._secretflow.generated.ss_binarizer",
        "class": "SSBinarizer",
        "type": "preprocessing",
        "mode": "ss",
        "description": "二值化 - 阈值转换",
    },
    
    # ========== 异常检测 (Anomaly Detection) ==========
    "IsolationForest": {
        "module": "xlearn._secretflow.generated.ss_isolationforest",
        "class": "SSIsolationForest",
        "type": "anomaly_detection",
        "mode": "ss",
        "description": "孤立森林 - 异常值检测",
    },
    
    # ========== 最近邻算法 (Neighbors) ==========
    "RadiusNeighborsClassifier": {
        "module": "xlearn._secretflow.generated.ss_radiusneighborsclassifier",
        "class": "SSRadiusNeighborsClassifier",
        "type": "neighbors",
        "mode": "ss",
        "description": "半径邻居分类器 - 基于半径的分类",
    },
    "RadiusNeighborsRegressor": {
        "module": "xlearn._secretflow.generated.ss_radiusneighborsregressor",
        "class": "SSRadiusNeighborsRegressor",
        "type": "neighbors",
        "mode": "ss",
        "description": "半径邻居回归器 - 基于半径的回归",
    },
    "NearestCentroid": {
        "module": "xlearn._secretflow.generated.ss_nearestcentroid",
        "class": "SSNearestCentroid",
        "type": "neighbors",
        "mode": "ss",
        "description": "最近质心 - 简单快速分类",
    },
}


# ============================================================================
# Public API
# ============================================================================

def get_algorithm(name: str):
    """
    Get algorithm class by name
    
    Parameters
    ----------
    name : str
        Algorithm name (e.g., 'PCA', 'Ridge', 'KMeans')
    
    Returns
    -------
    algorithm_class : Type
        SecretFlow-compatible algorithm class
    
    Examples
    --------
    >>> from xlearn._secretflow.algorithms import get_algorithm
    >>> PCA = get_algorithm("PCA")
    >>> model = PCA(spu, n_components=10)
    >>> model.fit(fed_data)
    """
    if name not in ALGORITHM_REGISTRY:
        available = ', '.join(sorted(ALGORITHM_REGISTRY.keys()))
        raise ValueError(
            f"Algorithm '{name}' not found. "
            f"Available algorithms: {available}"
        )
    
    info = ALGORITHM_REGISTRY[name]
    module = importlib.import_module(info["module"])
    return getattr(module, info["class"])


def list_algorithms(algorithm_type: Optional[str] = None) -> Dict[str, List[str]]:
    """
    List all available algorithms
    
    Parameters
    ----------
    algorithm_type : str, optional
        Filter by type: 'decomposition', 'regression', 'clustering',
        'classification', 'preprocessing', 'anomaly_detection', 'neighbors'
    
    Returns
    -------
    algorithms : dict or list
        If algorithm_type is None, returns dict grouped by type.
        Otherwise returns list of algorithm names.
    
    Examples
    --------
    >>> from xlearn._secretflow.algorithms import list_algorithms
    >>> 
    >>> # List all algorithms by category
    >>> all_algos = list_algorithms()
    >>> print(all_algos['decomposition'])
    ['PCA', 'TruncatedSVD', 'NMF', 'FactorAnalysis', 'FastICA']
    >>> 
    >>> # List algorithms of specific type
    >>> regression_algos = list_algorithms('regression')
    >>> print(regression_algos)
    ['Ridge', 'Lasso', 'ElasticNet', 'HuberRegressor', 'RANSACRegressor']
    """
    if algorithm_type is None:
        # Group by type
        result = {}
        for name, info in ALGORITHM_REGISTRY.items():
            algo_type = info["type"]
            if algo_type not in result:
                result[algo_type] = []
            result[algo_type].append(name)
        
        # Sort each list
        for key in result:
            result[key].sort()
        
        return result
    else:
        # Filter by type
        result = [
            name for name, info in ALGORITHM_REGISTRY.items()
            if info["type"] == algorithm_type
        ]
        return sorted(result)


def get_algorithm_info(name: str) -> Dict[str, str]:
    """
    Get algorithm information
    
    Parameters
    ----------
    name : str
        Algorithm name
    
    Returns
    -------
    info : dict
        Algorithm information including type, mode, description
    
    Examples
    --------
    >>> from xlearn._secretflow.algorithms import get_algorithm_info
    >>> info = get_algorithm_info("PCA")
    >>> print(info['description'])
    '主成分分析 - 线性降维'
    """
    if name not in ALGORITHM_REGISTRY:
        raise ValueError(f"Algorithm '{name}' not found")
    
    return ALGORITHM_REGISTRY[name].copy()


def print_algorithms_table():
    """
    Print a formatted table of all available algorithms
    
    Examples
    --------
    >>> from xlearn._secretflow.algorithms import print_algorithms_table
    >>> print_algorithms_table()
    """
    print("="*80)
    print(" SecretFlow 可用算法列表")
    print("="*80)
    print()
    
    algorithms_by_type = list_algorithms()
    
    type_names = {
        'decomposition': '降维算法',
        'regression': '回归算法',
        'clustering': '聚类算法',
        'classification': '分类算法',
        'preprocessing': '预处理算法',
        'anomaly_detection': '异常检测',
        'neighbors': '最近邻算法',
    }
    
    total_count = 0
    
    for algo_type, type_name in type_names.items():
        algos = algorithms_by_type.get(algo_type, [])
        if not algos:
            continue
        
        print(f"\n{type_name} ({len(algos)} 个)")
        print("-" * 80)
        
        for name in algos:
            info = ALGORITHM_REGISTRY[name]
            print(f"  ✅ {name:30s} - {info['description']}")
            total_count += 1
        
        print()
    
    print("="*80)
    print(f" 总计: {total_count} 个算法")
    print("="*80)


# ============================================================================
# Convenient imports
# ============================================================================

class AlgorithmFactory:
    """
    Factory class for convenient algorithm access
    
    Examples
    --------
    >>> from xlearn._secretflow.algorithms import algorithms
    >>> 
    >>> # Access by attribute
    >>> PCA = algorithms.PCA
    >>> Ridge = algorithms.Ridge
    >>> KMeans = algorithms.KMeans
    """
    
    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")
        
        try:
            return get_algorithm(name)
        except ValueError:
            available = ', '.join(sorted(ALGORITHM_REGISTRY.keys()))
            raise AttributeError(
                f"Algorithm '{name}' not found. "
                f"Available: {available}"
            )
    
    def __dir__(self):
        return list(ALGORITHM_REGISTRY.keys())


# Global factory instance
algorithms = AlgorithmFactory()


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    """Display all available algorithms"""
    print_algorithms_table()
    
    print("\n使用示例:")
    print("-" * 80)
    print("""
import secretflow as sf
from xlearn._secretflow.algorithms import algorithms, get_algorithm

# 方式 1: 使用 factory
PCA = algorithms.PCA
model = PCA(spu, n_components=10)

# 方式 2: 使用函数
Ridge = get_algorithm("Ridge")
model = Ridge(spu, alpha=1.0)

# 查看所有算法
from xlearn._secretflow.algorithms import list_algorithms
all_algos = list_algorithms()
print(all_algos['decomposition'])
""")

