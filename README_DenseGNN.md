# DenseGNN Integration Guide

## 概述

我们已经成功将 **DenseGNN** (Dense Graph Neural Network) 集成到多模态框架中，用于替代 ALIGNN 模型。DenseGNN 提供了更高效的图神经网络架构，支持文本-图跨模态学习。

## 主要特性

### 1. DenseGNN 架构
- **密集连接**：在图神经网络层之间使用密集连接，提高信息流动
- **分层更新**：节点、边特征的分层更新机制
- **残差连接**：每一层都有残差连接，避免梯度消失

### 2. 多模态支持
- **文本编码**：使用 MatSciBERT 编码材料描述文本
- **跨模态注意力**：图特征和文本特征之间的双向注意力机制
- **对比学习**（可选）：通过对比损失对齐图和文本表示

### 3. 灵活配置
- 可配置的层数、隐藏维度、注意力头数等
- 支持分类和回归任务
- 支持不同的输出激活函数（identity, log, logit）

## 使用方法

### 1. 基本使用

```python
from config import TrainingConfig
from models.densegnn import DenseGNNConfig
from train import train_dgl

# 创建 DenseGNN 配置
model_config = DenseGNNConfig(
    name="densegnn",
    densegnn_layers=4,
    hidden_features=256,
    use_cross_modal_attention=True
)

# 创建训练配置
config = TrainingConfig(
    dataset="dft_3d",
    target="formation_energy_peratom",
    model=model_config,
    epochs=300,
    batch_size=64,
    learning_rate=0.001
)

# 开始训练
history = train_dgl(config)
```

### 2. 使用配置文件

```bash
# 使用示例配置文件
python your_training_script.py --config config_densegnn_example.json
```

### 3. 配置参数说明

#### 模型参数
- `name`: 必须为 "densegnn"
- `densegnn_layers`: DenseGNN 卷积层数量（默认：4）
- `atom_input_features`: 原子输入特征维度（默认：92）
- `edge_input_features`: 边输入特征维度（默认：80）
- `embedding_features`: 嵌入特征维度（默认：64）
- `hidden_features`: 隐藏层特征维度（默认：256）
- `output_features`: 输出特征维度（默认：1）
- `graph_dropout`: Dropout 比率（默认：0.0）

#### 中期融合参数（Middle Fusion）
- `use_middle_fusion`: 是否使用中期融合（默认：False）
- `middle_fusion_layers`: 在哪些层应用融合，逗号分隔（默认："1,3"）
- `middle_fusion_hidden_dim`: 融合隐藏维度（默认：128）
- `middle_fusion_num_heads`: 注意力头数（默认：2）
- `middle_fusion_dropout`: 融合 Dropout（默认：0.1）

#### 跨模态注意力参数（Late Fusion）
- `use_cross_modal_attention`: 是否使用跨模态注意力（默认：False）
- `cross_modal_hidden_dim`: 注意力隐藏维度（默认：256）
- `cross_modal_num_heads`: 注意力头数（默认：4）
- `cross_modal_dropout`: 注意力 Dropout（默认：0.1）

#### 对比学习参数
- `use_contrastive_loss`: 是否使用对比损失（默认：False）
- `contrastive_temperature`: 温度参数（默认：0.1）
- `contrastive_loss_weight`: 对比损失权重（默认：0.1）

## 与 ALIGNN 的对比

| 特性 | ALIGNN | DenseGNN |
|------|--------|----------|
| 架构 | 原子线图神经网络 | 密集图神经网络 |
| 线图 | 需要 | 不需要 |
| 计算效率 | 中等 | 更高 |
| 参数量 | 较多 | 较少 |
| 多模态支持 | ✓ | ✓ |
| 跨模态注意力 | ✓ | ✓ |
| 对比学习 | ✓ | ✓ |

## 文件结构

```
SGA-V1.6/
├── models/
│   ├── densegnn.py              # DenseGNN 模型实现
│   ├── alignn.py                # ALIGNN 模型（原有）
│   └── __init__.py              # 模型导出
├── config.py                    # 配置类（已添加 DenseGNN）
├── train.py                     # 训练脚本（已支持 DenseGNN）
├── config_densegnn_example.json # DenseGNN 示例配置
└── README_DenseGNN.md          # 本文档
```

## 实现细节

### DenseGNN 卷积层

```python
class DenseGNNConv(nn.Module):
    """Dense GNN Convolution layer."""
    def forward(self, g, node_feats, edge_feats):
        # 1. 边更新：基于源节点、目标节点和边特征
        # 2. 节点更新：聚合邻居边特征
        # 3. 残差连接
        return updated_nodes, updated_edges
```

### 多模态融合

DenseGNN 支持**两种融合策略**，可以单独使用或组合使用：

#### 1. 中期融合（Middle Fusion）
在图卷积的中间层注入文本信息：
1. **文本编码**：MatSciBERT → CLS token → Projection Head
2. **图卷积**：在指定层（如第1层和第3层）：
   - DenseGNN 卷积更新节点特征
   - **门控融合**：将文本特征通过门控机制融合到节点特征中
   - 继续后续卷积
3. 优势：文本信息早期融入，影响后续所有层的表示学习

#### 2. 晚期融合（Late Fusion - Cross-Modal Attention）
在图池化后进行全局融合：
1. **文本编码**：MatSciBERT → CLS token → Projection Head
2. **图编码**：DenseGNN layers → Pooling → Projection Head
3. **跨模态注意力**：
   - 图特征 attend to 文本特征
   - 文本特征 attend to 图特征
4. **融合**：平均或拼接
5. **预测**：MLP → 输出

#### 3. 组合策略（推荐）
可以同时使用中期融合和晚期融合：
```python
model_config = DenseGNNConfig(
    use_middle_fusion=True,        # 在第1、3层注入文本
    middle_fusion_layers="1,3",
    use_cross_modal_attention=True  # 最后进行全局注意力融合
)
```

## 训练建议

1. **学习率**：建议从 0.001 开始
2. **Batch Size**：根据 GPU 显存调整（推荐：32-128）
3. **层数**：4-6 层效果较好
4. **Early Stopping**：建议设置 patience=50
5. **跨模态注意力**：对于有文本描述的数据集建议开启

## 性能优化

1. **不使用线图**：相比 ALIGNN 减少了计算开销
2. **密集连接**：提高特征重用，减少参数量
3. **残差连接**：允许训练更深的网络
4. **批归一化**：加速训练收敛

## 常见问题

### Q: DenseGNN 和 ALIGNN 哪个更好？
A: DenseGNN 通常在计算效率上更优，ALIGNN 在某些数据集上可能有更高的精度。建议都尝试。

### Q: 是否必须提供文本描述？
A: 不是必须的。如果没有文本，模型会只使用图特征。

### Q: 如何切换回 ALIGNN？
A: 在配置文件中将 `model.name` 改为 `"alignn"` 即可。

### Q: 能否同时使用对比学习和跨模态注意力？
A: 可以，但建议先单独尝试，再组合使用。

## 引用

如果使用 DenseGNN，请引用：

```bibtex
@article{densegnn2024,
  title={DenseGNN: universal and scalable deeper graph neural networks for high-performance property prediction in crystals and molecules},
  author={Du, Hongwei and Wang, Jian and Hui, Jielan and others},
  journal={npj Computational Materials},
  volume={10},
  number={292},
  year={2024},
  publisher={Nature}
}
```

## 联系方式

如有问题或建议，请提交 Issue 或 Pull Request。
