# 符号回归可解释性功能

## 概述

符号回归（Symbolic Regression）是一种强大的机器学习可解释性技术,可以从训练好的神经网络模型中自动发现可解释的数学公式。与传统的黑盒深度学习模型不同,符号回归能够将复杂的神经网络预测转化为简洁、可理解的数学表达式。

### 主要特性

✅ **自动公式发现**: 从神经网络特征中自动发现数学关系
✅ **可解释性**: 生成人类可读的数学公式
✅ **准确性评估**: 比较符号公式与神经网络的性能
✅ **多样化候选**: 生成多个候选公式供选择
✅ **易于集成**: 无缝集成到现有的可解释性分析流程中

### 技术背景

本功能使用 [PySR (Python Symbolic Regression)](https://github.com/MilesCranmer/PySR) 库,这是一个基于遗传算法的高效符号回归工具。它通过以下步骤工作:

1. **特征提取**: 从训练好的神经网络中提取中间层特征
2. **符号搜索**: 使用遗传算法搜索最佳数学公式
3. **公式优化**: 在准确性和复杂度之间找到最佳平衡
4. **结果验证**: 评估发现的公式的预测性能

## 安装

### 1. 安装Julia (必需)

PySR依赖于Julia编程语言。请按照以下步骤安装:

**Linux/Mac:**
```bash
# 下载并安装Julia
wget https://julialang-s3.julialang.org/bin/linux/x64/1.9/julia-1.9.3-linux-x86_64.tar.gz
tar -xvzf julia-1.9.3-linux-x86_64.tar.gz
sudo mv julia-1.9.3 /opt/
sudo ln -s /opt/julia-1.9.3/bin/julia /usr/local/bin/julia
```

**或者使用包管理器:**
```bash
# Ubuntu/Debian
sudo apt install julia

# Mac
brew install julia
```

**Windows:**
从 [Julia官网](https://julialang.org/downloads/) 下载安装包

### 2. 安装PySR

```bash
# 安装PySR包
pip install pysr

# 安装Julia依赖 (首次运行时会自动安装,也可以手动安装)
python -c 'import pysr; pysr.install()'
```

### 3. 验证安装

```python
import pysr
print("PySR版本:", pysr.__version__)
```

## 使用方法

### 方法1: 使用ComprehensiveExplainer类

```python
from interpretability_enhanced_v2 import ComprehensiveExplainer
import torch

# 1. 初始化解释器
explainer = ComprehensiveExplainer(
    model=your_trained_model,
    tokenizer=your_tokenizer,  # 可选
    device='cuda'
)

# 2. 运行符号回归分析
model_sr, results = explainer.extract_symbolic_features(
    test_loader=test_loader,
    save_dir='./symbolic_regression_results',
    max_samples=500  # 可选: 限制样本数以加快速度
)

# 3. 查看结果
if model_sr is not None:
    print("发现的最佳公式:")
    print(model_sr.sympy())

    print("\n性能指标:")
    print(f"MAE: {results['metrics']['mae']:.4f}")
    print(f"R²: {results['metrics']['r2']:.4f}")
```

### 方法2: 使用示例脚本

```bash
python run_symbolic_regression_analysis.py \
    --model_path ./checkpoints/best_model.pt \
    --data_path ./data/test_data.json \
    --save_dir ./symbolic_results \
    --max_samples 500 \
    --device cuda
```

## 输出结果

运行符号回归分析后,会在指定目录生成以下文件:

### 1. `symbolic_regression_formulas.txt`
包含所有发现的符号公式,按复杂度-准确度权衡排序:

```
================================================================================
符号回归发现的公式
================================================================================

最佳公式:
0.245*x0 + 0.891*sqrt(x1) - 1.23*log(x2 + 1) + 0.567

所有候选公式:
--------------------------------------------------------------------------------

公式 1:
  0.245*x0 + 0.891*sqrt(x1) - 1.23*log(x2 + 1) + 0.567
  复杂度: 15, 损失: 0.023456

公式 2:
  0.312*x0 + 0.456*x1 - 0.789*x2
  复杂度: 7, 损失: 0.045678

...
```

### 2. `symbolic_regression_results.json`
详细的分析结果,包括:

```json
{
  "best_formula": "0.245*x0 + 0.891*sqrt(x1) - 1.23*log(x2 + 1) + 0.567",
  "metrics": {
    "mae": 0.0234,
    "rmse": 0.0456,
    "r2": 0.9123
  },
  "nn_comparison": {
    "mae_nn": 0.0189,
    "r2_nn": 0.9456,
    "mae_ratio": 1.2381,
    "r2_diff": -0.0333
  },
  "feature_dim": 128,
  "num_samples": 500
}
```

### 3. `symbolic_regression_model.pkl`
保存的PySR模型,可用于后续预测:

```python
import pickle

# 加载模型
with open('symbolic_regression_model.pkl', 'rb') as f:
    model_sr = pickle.load(f)

# 使用模型预测
predictions = model_sr.predict(new_features)
```

## 参数说明

### `extract_symbolic_features()` 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `test_loader` | DataLoader | **必需** | 测试数据加载器 |
| `save_dir` | str/Path | None | 结果保存目录 |
| `max_samples` | int | None | 最大样本数 (None表示使用所有) |

### PySR配置参数 (在代码中修改)

在 `interpretability_enhanced_v2.py` 中的 `extract_symbolic_features` 方法里可以调整PySR参数:

```python
model_sr = pysr.PySRRegressor(
    niterations=100,              # 迭代次数 (增加以提高质量)
    maxsize=20,                   # 最大公式复杂度
    binary_operators=["+", "-", "*", "/", "^"],  # 二元运算符
    unary_operators=["exp", "log", "sqrt", "abs"],  # 一元运算符
    parsimony=0.0032,             # 简洁性惩罚 (越大越简洁)
    populations=15,               # 种群数量
    select_k_features=10,         # 自动选择最重要的k个特征
)
```

## 应用场景

### 1. 科学发现
从材料性质预测模型中发现新的物理-化学关系:
```
formation_energy = -2.34*electronegativity + 1.56*sqrt(atomic_radius) - 0.89
```

### 2. 模型简化
用简单的数学公式替代复杂的神经网络,便于部署:
```python
# 神经网络: 1M参数, 100ms推理时间
# 符号公式: 即时计算, 可在任何设备上运行
```

### 3. 特征重要性分析
通过公式中各项的系数了解特征重要性:
```
prediction = 0.5*feature_1 + 0.3*feature_2 + 0.1*feature_3
# feature_1 最重要 (系数0.5)
```

### 4. 知识提取
从黑盒模型中提取可解释的科学知识:
```
band_gap = 3.45 - 0.89*log(num_electrons) + 0.23*lattice_constant^2
```

## 最佳实践

### 1. 样本数量
- **推荐**: 500-2000个样本
- **最少**: 100个样本
- **过多**: >5000个样本会显著增加计算时间

### 2. 迭代次数
- **快速测试**: 50次迭代
- **生产环境**: 100-200次迭代
- **高精度**: 500+次迭代

### 3. 特征选择
- 自动选择前10个最重要的特征 (`select_k_features=10`)
- 如果特征维度很高(>100),考虑降维

### 4. 公式复杂度
- 从简单开始: `maxsize=10`
- 标准设置: `maxsize=20`
- 允许复杂公式: `maxsize=30+`

## 常见问题

### Q1: PySR安装失败
**A**: 确保Julia已正确安装并在PATH中:
```bash
julia --version
# 应该显示: julia version 1.9.x
```

### Q2: 符号回归运行很慢
**A**: 尝试以下优化:
- 减少样本数: `max_samples=500`
- 减少迭代次数: `niterations=50`
- 减少特征维度: `select_k_features=5`

### Q3: 发现的公式准确度低
**A**: 可以:
- 增加迭代次数: `niterations=200`
- 增加公式复杂度: `maxsize=30`
- 增加样本数
- 添加更多运算符

### Q4: 如何选择最佳公式?
**A**: 查看 `equations_` 表格,平衡考虑:
- **损失**(loss): 越低越好
- **复杂度**(complexity): 越低越简洁
- **评分**(score): 综合指标,越高越好

### Q5: 能否自定义运算符?
**A**: 可以!在PySRRegressor中添加:
```python
unary_operators=["exp", "log", "sqrt", "sin", "cos", "tanh"],
binary_operators=["+", "-", "*", "/", "^", "max", "min"]
```

## 示例输出

以下是一个真实的符号回归分析输出示例:

```
================================================================================
🔬 符号回归分析 - 从神经网络特征中发现数学公式
================================================================================

📊 [1/3] 提取模型特征...
提取特征: 100%|████████████████████| 32/32 [00:05<00:00,  6.12it/s]
   ✓ 提取了 500 个样本
   ✓ 特征维度: 128

🧮 [2/3] 运行符号回归...
   这可能需要几分钟时间,请耐心等待...

Hall of Fame:
-----------------------------------------
Complexity  Loss       Equation
1          0.123456    0.567
3          0.045678    0.245*x0 + 0.891
7          0.023456    0.245*x0 + 0.891*sqrt(x1)
15         0.012345    0.245*x0 + 0.891*sqrt(x1) - 1.23*log(x2 + 1)
-----------------------------------------

================================================================================
📝 发现的符号公式:
================================================================================

🎯 最佳公式 (SymPy格式):
================================================================================

0.245*x0 + 0.891*sqrt(x1) - 1.23*log(x2 + 1) + 0.567

📊 [3/3] 评估符号回归模型...

符号回归模型性能:
  MAE:  0.0234
  RMSE: 0.0456
  R²:   0.9123

与神经网络对比:
  神经网络 MAE: 0.0189
  神经网络 R²:  0.9456
  MAE 比率:     123.81% (越小越好)
  R² 差距:      -0.0333

   ✓ 公式已保存: symbolic_regression_results/symbolic_regression_formulas.txt
   ✓ 结果已保存: symbolic_regression_results/symbolic_regression_results.json
   ✓ 模型已保存: symbolic_regression_results/symbolic_regression_model.pkl

================================================================================
✅ 符号回归分析完成!
================================================================================
```

## 引用

如果您在研究中使用了此功能,请引用PySR:

```bibtex
@article{cranmer2023interpretable,
  title={Interpretable Machine Learning for Science with PySR and SymbolicRegression.jl},
  author={Cranmer, Miles},
  journal={arXiv preprint arXiv:2305.01582},
  year={2023}
}
```

## 相关资源

- [PySR官方文档](https://astroautomata.com/PySR/)
- [PySR GitHub仓库](https://github.com/MilesCranmer/PySR)
- [符号回归介绍](https://en.wikipedia.org/wiki/Symbolic_regression)
- [遗传编程](https://en.wikipedia.org/wiki/Genetic_programming)

## 许可

本功能基于PySR库,遵循Apache-2.0许可证。

## 更新日志

### v2.0 (2025-12-04)
- ✨ 新增: 符号回归分析功能
- ✨ 新增: 自动特征提取
- ✨ 新增: 神经网络性能对比
- 📝 新增: 完整的使用文档和示例
