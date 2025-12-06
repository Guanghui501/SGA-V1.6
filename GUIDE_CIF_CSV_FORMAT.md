# CIF + CSV 数据格式训练指南

本指南说明如何使用 `train_local_cif_csv.py` 训练模型，该脚本支持 **CIF文件目录 + CSV元数据** 的数据格式。

**✨ 新功能**: 现已支持标准数据集（JARVIS, Material Project, 分类数据集等），完全照抄 `train_with_cross_modal_attention.py` 的数据处理方式！

---

## 🎯 支持的数据集类型

`train_local_cif_csv.py` 支持以下数据集：

1. **JARVIS-DFT**: JARVIS 数据库的标准格式
2. **Material Project (MP)**: MP 数据库格式
3. **Class**: 分类任务数据集
4. **Toy**: 玩具/测试数据集
5. **Custom**: 自定义数据集（原有功能）

---

## 📂 数据格式

### 目录结构示例

```
my_project/
├── structures/          # CIF文件目录
│   ├── sample_001.cif
│   ├── sample_002.cif
│   ├── sample_003.cif
│   └── ...
├── data.csv             # CSV元数据文件
└── results/             # 训练结果输出目录（自动创建）
```

### CIF 文件要求

- **格式**: 标准 CIF (Crystallographic Information File) 格式
- **命名**: 文件名需要与 CSV 中的 `id` 列对应
  - 例如: CSV 中的 `id=sample_001` 对应文件 `sample_001.cif`
- **内容**: 包含晶体结构的晶格参数、原子坐标、元素类型等信息

### CSV 文件格式（按数据集类型）

#### 标准数据集格式

不同数据集有不同的 CSV 格式（与 `train_with_cross_modal_attention.py` 完全一致）：

| 数据集 | CSV 列顺序 | 示例 |
|--------|-----------|------|
| **JARVIS** | `Id, Composition, prop, Description, File_Name` | `0, VSe2, 0.0, "VSe2 trigonal...", desc_mbj_bandgap0.csv` |
| **MP (formation_energy)** | `id, composition, formation_energy, band_gap, description, file_name` | `mp-1234, Fe2O3, -3.45, 2.1, "Iron oxide...", mat_text.csv` |
| **MP (band_gap)** | `id, composition, formation_energy, band_gap, description, file_name` | （band_gap 在第4列） |
| **Class** | `id, target, description` | `sample_001, 0, "Metal with BCC structure"` |
| **Toy** | 同 JARVIS | 用于测试 |

#### 自定义数据集格式（Custom）

对于自定义数据集，可以通过参数指定列名：

**必需列**:
| 列名（默认） | 类型 | 说明 | 示例 |
|------|------|------|------|
| `id` | 字符串 | 样本唯一标识符，对应CIF文件名 | `sample_001` |
| `target` | 浮点数 | 目标属性值（回归任务） | `-3.456` |

**可选列**:
| 列名（默认） | 类型 | 说明 | 示例 |
|------|------|------|------|
| `text_description` | 字符串 | 材料文本描述（用于多模态学习） | `Perovskite structure with high conductivity` |
| `composition` | 字符串 | 化学式 | `Ca2MnO4` |

#### CSV 示例（自定义格式）

```csv
id,target,text_description,composition
sample_001,-3.456,Stable perovskite with cubic symmetry,CaTiO3
sample_002,-2.123,Layered oxide with good ionic conductivity,Li2MnO3
sample_003,-4.789,High entropy alloy with FCC structure,CoCrFeNi
```

#### CSV 示例（分类任务）

对于分类任务，目标列应为整数类别标签：

```csv
id,label,text_description
sample_001,0,Metal with BCC structure
sample_002,1,Semiconductor with diamond structure
sample_003,2,Insulator with perovskite structure
```

---

## 🚀 快速开始

### 方式 1: 使用标准数据集（JARVIS/MP/Class）

```bash
# JARVIS 数据集 - 形成能预测
python train_local_cif_csv.py \
    --root_dir ../dataset/ \
    --dataset jarvis \
    --property formation_energy \
    --model densegnn \
    --use_middle_fusion \
    --use_cross_modal

# Material Project 数据集 - 带隙预测
python train_local_cif_csv.py \
    --root_dir ../dataset/ \
    --dataset mp \
    --property band_gap \
    --model densegnn \
    --use_cross_modal

# 分类数据集
python train_local_cif_csv.py \
    --root_dir ../dataset/ \
    --dataset class \
    --property syn \
    --classification \
    --num_classes 2
```

### 方式 2: 使用自定义数据集

#### 步骤 1: 准备数据

```bash
# 确保数据结构正确
ls structures/
# 输出: sample_001.cif  sample_002.cif  sample_003.cif  ...

head -5 data.csv
# 输出: CSV文件前5行
```

#### 步骤 2: 基础训练（回归）

```bash
python train_local_cif_csv.py \
    --dataset custom \
    --cif_dir ./structures/ \
    --csv_file ./data.csv \
    --output_dir ./results/
```

### 步骤 3: 查看结果

```bash
ls results/
# 输出:
# best_val_model.pt
# best_test_model.pt
# config.json
# history_val.json
# predictions_best_val_model_test.csv
```

---

## ⚙️ 配置选项

### 数据集参数

#### 标准数据集（JARVIS/MP/Class/Toy）

```bash
python train_local_cif_csv.py \
    --root_dir ../dataset/            # 数据集根目录
    --dataset jarvis                  # 数据集类型: jarvis, mp, class, toy
    --property formation_energy       # 预测属性
```

**支持的属性**:
- **JARVIS**: `formation_energy`, `mbj_bandgap`, `opt_bandgap`, `bulk_modulus`, `shear_modulus`, 等
- **MP**: `formation_energy`, `band_gap`, `bulk`, `shear`
- **Class**: 根据具体分类任务，如 `syn`, `metal_oxide` 等

#### 自定义数据集（Custom）

```bash
python train_local_cif_csv.py \
    --dataset custom \
    --cif_dir ./structures/           # CIF文件目录
    --csv_file ./data.csv             # CSV元数据文件
    --id_column id                    # CSV中ID列名（默认: id）
    --target_column target            # CSV中目标列名（默认: target）
    --text_column text_description    # CSV中文本列名（可选）
```

### 模型选择

```bash
# 使用 DenseGNN 模型（推荐）
python train_local_cif_csv.py \
    --model densegnn \
    --densegnn_layers 4 \
    --hidden_features 256

# 使用 ALIGNN 模型
python train_local_cif_csv.py \
    --model alignn \
    --alignn_layers 4 \
    --hidden_features 256
```

### 多模态学习

#### 中期融合 (Middle Fusion)

在图卷积的中间层注入文本特征：

```bash
python train_local_cif_csv.py \
    --cif_dir ./structures/ \
    --csv_file ./data.csv \
    --use_middle_fusion \
    --middle_fusion_layers 1,3        # 在第1和第3层融合
    --middle_fusion_hidden_dim 128 \
    --middle_fusion_num_heads 2
```

#### 后期融合 (Late Fusion / Cross-Modal Attention)

在图池化后使用交叉注意力：

```bash
python train_local_cif_csv.py \
    --cif_dir ./structures/ \
    --csv_file ./data.csv \
    --use_cross_modal \
    --cross_modal_hidden_dim 256 \
    --cross_modal_num_heads 4
```

#### 对比学习 (Contrastive Learning)

使用 InfoNCE 损失对齐图-文本表示：

```bash
python train_local_cif_csv.py \
    --cif_dir ./structures/ \
    --csv_file ./data.csv \
    --use_contrastive \
    --contrastive_temperature 0.1 \
    --contrastive_weight 0.1
```

#### 完整多模态配置

```bash
python train_local_cif_csv.py \
    --cif_dir ./structures/ \
    --csv_file ./data.csv \
    --model densegnn \
    --use_middle_fusion \
    --middle_fusion_layers 1,3 \
    --use_cross_modal \
    --use_contrastive \
    --epochs 500 \
    --batch_size 32 \
    --learning_rate 0.001
```

### 训练参数

```bash
python train_local_cif_csv.py \
    --epochs 500 \
    --batch_size 32 \
    --learning_rate 0.001 \
    --train_ratio 0.8 \
    --val_ratio 0.1 \
    --test_ratio 0.1 \
    --n_early_stopping 50
```

### 分类任务

```bash
python train_local_cif_csv.py \
    --cif_dir ./structures/ \
    --csv_file ./labels.csv \
    --classification \
    --target_column label \
    --num_classes 3
```

---

## 📊 完整示例

### 示例 1: 形成能预测（回归 + DenseGNN）

```bash
python train_local_cif_csv.py \
    --cif_dir ./formation_energy_data/cif/ \
    --csv_file ./formation_energy_data/targets.csv \
    --target_column formation_energy_peratom \
    --model densegnn \
    --densegnn_layers 4 \
    --hidden_features 256 \
    --epochs 500 \
    --batch_size 32 \
    --output_dir ./results_formation_energy/
```

**数据格式** (`targets.csv`):
```csv
id,formation_energy_peratom,composition
mp-1234,-3.456,Fe2O3
mp-5678,-2.123,CaTiO3
mp-9012,-4.789,Li2MnO3
```

### 示例 2: 带隙预测（多模态 + 文本描述）

```bash
python train_local_cif_csv.py \
    --cif_dir ./bandgap_data/cif/ \
    --csv_file ./bandgap_data/data.csv \
    --target_column band_gap \
    --text_column description \
    --model densegnn \
    --use_middle_fusion \
    --middle_fusion_layers 1,3 \
    --use_cross_modal \
    --epochs 500 \
    --output_dir ./results_bandgap/
```

**数据格式** (`data.csv`):
```csv
id,band_gap,description,composition
sample_001,2.3,Direct bandgap semiconductor with wurtzite structure,GaN
sample_002,3.4,Wide bandgap oxide with cubic structure,MgO
sample_003,0.0,Metallic conductor with FCC structure,Cu
```

### 示例 3: 材料分类（3类 + ALIGNN）

```bash
python train_local_cif_csv.py \
    --cif_dir ./classification_data/cif/ \
    --csv_file ./classification_data/labels.csv \
    --target_column material_class \
    --text_column properties \
    --model alignn \
    --classification \
    --num_classes 3 \
    --use_cross_modal \
    --epochs 300 \
    --output_dir ./results_classification/
```

**数据格式** (`labels.csv`):
```csv
id,material_class,properties
sample_001,0,Metal with high electrical conductivity
sample_002,1,Semiconductor with moderate bandgap
sample_003,2,Insulator with low thermal conductivity
```

---

## 🔧 常见问题

### Q1: CSV 中的 ID 找不到对应的 CIF 文件

**错误信息**:
```
FileNotFoundError: CIF file not found: ./structures/sample_001.cif
```

**解决方案**:
- 检查 CIF 文件名是否与 CSV 中的 `id` 完全匹配
- 确保文件扩展名为 `.cif`
- 检查 `--cif_dir` 路径是否正确

### Q2: 多模态训练但 CSV 中没有文本列

**错误信息**:
```
KeyError: 'text_description'
```

**解决方案**:
- 使用 `--text_column` 指定正确的文本列名
- 或者在 CSV 中添加 `text_description` 列
- 如果不使用文本，移除 `--use_middle_fusion` 和 `--use_cross_modal` 选项

### Q3: 内存不足

**解决方案**:
```bash
python train_local_cif_csv.py \
    --batch_size 8 \          # 减小批次大小
    --num_workers 0 \          # 禁用多进程
    --hidden_features 128      # 减小隐藏层维度
```

### Q4: 训练不收敛

**解决方案**:
```bash
python train_local_cif_csv.py \
    --learning_rate 0.0001 \   # 降低学习率
    --epochs 1000 \            # 增加训练轮数
    --n_early_stopping 100     # 放宽早停条件
```

---

## 📈 结果分析

### 训练历史

```python
import json
import matplotlib.pyplot as plt

# 读取训练历史
with open("results/history_val.json", "r") as f:
    history = json.load(f)

# 绘制学习曲线
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(history["mae"], label="Validation MAE")
plt.xlabel("Epoch")
plt.ylabel("MAE")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history["loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.savefig("training_curves.png")
```

### 预测结果

```python
import pandas as pd
import numpy as np

# 读取预测结果
predictions = pd.read_csv("results/predictions_best_val_model_test.csv")

# 计算误差指标
mae = np.abs(predictions["prediction"] - predictions["target"]).mean()
rmse = np.sqrt(((predictions["prediction"] - predictions["target"])**2).mean())

print(f"Test MAE: {mae:.4f}")
print(f"Test RMSE: {rmse:.4f}")

# 绘制预测 vs 真实值
plt.figure(figsize=(6, 6))
plt.scatter(predictions["target"], predictions["prediction"], alpha=0.5)
plt.plot([predictions["target"].min(), predictions["target"].max()],
         [predictions["target"].min(), predictions["target"].max()],
         'r--', label='Perfect prediction')
plt.xlabel("True values")
plt.ylabel("Predictions")
plt.legend()
plt.savefig("predictions.png")
```

---

## 📚 相关文档

- **DenseGNN 模型**: `README_DenseGNN.md`
- **自定义数据集（JSON格式）**: `GUIDE_CUSTOM_DATASET.md`
- **快速开始**: `QUICKSTART_CUSTOM_DATA.md`

---

## 💡 最佳实践

1. **数据准备**:
   - 使用描述性的样本 ID
   - 确保 CIF 文件结构正确（可使用晶体学软件验证）
   - 文本描述应简洁且信息丰富

2. **模型选择**:
   - 小数据集（<1000）: 使用较小模型（2-3层，128-256隐藏维度）
   - 大数据集（>10000）: 可使用更深模型（4-6层，256-512隐藏维度）
   - 有文本描述: 启用多模态融合

3. **训练策略**:
   - 从小学习率开始（1e-4）
   - 使用早停防止过拟合
   - 定期保存检查点

4. **验证**:
   - 训练前检查数据统计
   - 监控训练/验证曲线
   - 分析预测误差分布

---

**祝训练顺利！** 🎉

如有问题，请参考源代码中的详细注释或查阅其他文档。
