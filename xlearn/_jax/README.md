# JAX Acceleration Module

这个模块为 JAX-sklearn 提供透明的 JAX 加速功能。

## 📁 文件结构

```
xlearn/_jax/
├── __init__.py              # 模块入口，JAX可用性检查
├── _config.py              # 配置管理系统
├── _data_conversion.py     # NumPy ↔ JAX 数据转换工具
├── _accelerator.py         # 加速器注册和管理系统
├── _proxy.py              # 智能代理系统
├── _universal_jax.py      # 通用JAX加速实现
└── README.md              # 本文档
```

## 🚀 核心架构

### 1. 智能代理模式 (`_proxy.py`)
- **EstimatorProxy**: 透明切换JAX和原版实现
- **create_intelligent_proxy**: 自动为任何算法创建JAX加速代理
- **自动回退**: JAX失败时自动使用原版实现

### 2. 通用JAX实现 (`_universal_jax.py`)
- **UniversalJAXMixin**: 基础JAX加速混入类
- **JAXLinearModelMixin**: 线性模型JAX加速
- **JAXClusterMixin**: 聚类算法JAX加速  
- **JAXDecompositionMixin**: 降维算法JAX加速
- **性能启发式**: 智能决定何时使用JAX

### 3. 配置系统 (`_config.py`)
```python
import xlearn._jax as jax_config

# 检查JAX状态
jax_config.get_config()

# 配置JAX设置
jax_config.set_config(enable_jax=True, jax_platform="gpu")

# 临时配置
with jax_config.config_context(enable_jax=False):
    # 强制使用NumPy实现
    pass
```

### 4. 数据转换 (`_data_conversion.py`)
- **to_jax()**: NumPy → JAX 数组转换
- **to_numpy()**: JAX → NumPy 数组转换
- **auto_convert_arrays**: 装饰器，自动处理数据转换

### 5. 注册系统 (`_accelerator.py`)
- **AcceleratorRegistry**: 管理JAX实现注册
- **@accelerated_estimator**: 装饰器注册JAX实现
- **create_accelerated_estimator**: 创建加速实例

## ⚡ 工作原理

1. **自动检测**: 系统启动时检查JAX可用性
2. **动态代理**: 为每个算法类创建智能代理
3. **性能决策**: 基于数据规模智能选择实现
4. **透明切换**: 用户无感知的JAX/NumPy切换
5. **错误回退**: JAX失败时自动使用原版

## 🎯 性能优化

### 启发式规则
```python
# 算法特定的阈值
thresholds = {
    'LinearRegression': {'min_complexity': 1e8, 'min_samples': 10000},
    'KMeans': {'min_complexity': 1e6, 'min_samples': 5000},
    'PCA': {'min_complexity': 1e7, 'min_samples': 5000},
    # ...
}
```

### JIT编译优化
- 静态函数编译: `@jax.jit` 装饰核心计算
- 函数缓存: 避免重复编译开销
- 数值稳定性: 添加正则化防止数值问题

## 🔧 扩展新算法

添加新算法的JAX支持：

```python
# 1. 在_universal_jax.py中添加专用mixin
class JAXNewAlgorithmMixin(UniversalJAXMixin):
    def jax_fit(self, X, y=None):
        # JAX实现
        pass

# 2. 在_proxy.py中添加算法检测
def create_universal_jax_class(original_class):
    if 'new_algorithm' in module_name:
        mixin_class = JAXNewAlgorithmMixin
    # ...
```

## 📊 使用示例

```python
import xlearn as sklearn  # JAX自动启用

# 正常使用，JAX在后台自动加速
model = sklearn.linear_model.LinearRegression()
model.fit(X, y)  # 大数据时自动使用JAX
predictions = model.predict(X_test)

# 检查是否使用了JAX
print(f"Using JAX: {getattr(model, 'is_using_jax', False)}")
```

## 🎉 特性

- ✅ **100% API兼容**: 完全兼容scikit-learn接口
- ✅ **透明加速**: 用户无需修改代码
- ✅ **智能回退**: 错误时自动使用原版
- ✅ **性能优化**: 基于数据规模智能决策
- ✅ **易于扩展**: 模块化设计便于添加新算法

这个架构确保了JAX-sklearn既能提供性能提升，又保持了完全的兼容性和稳定性。
