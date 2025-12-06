# 本地数据集训练指南

## 📚 概述

本指南介绍如何使用**本地自定义数据集**训练 DenseGNN/ALIGNN 模型，包括数据准备、配置和训练步骤。

---

## 📋 数据格式要求

### 1. JSON 格式（推荐）

数据必须是一个 **JSON 列表**，每个元素包含以下字段：

```json
[
  {
    "jid": "sample_001",           // 样本ID（必需）
    "atoms": {                      // 晶体结构（必需）
      "lattice_mat": [              // 晶格矩阵 3x3
        [5.0, 0.0, 0.0],
        [0.0, 5.0, 0.0],
        [0.0, 0.0, 5.0]
      ],
      "coords": [                   // 原子坐标（分数坐标）
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.5]
      ],
      "elements": ["Si", "Si"],     // 元素符号
      "abc": [5.0, 5.0, 5.0],      // 晶格参数（可选）
      "angles": [90, 90, 90]        // 晶格角度（可选）
    },
    "formation_energy_peratom": -3.5,  // 目标属性（根据任务命名）
    "text_description": "Silicon crystal with diamond structure"  // 文本描述（多模态可选）
  },
  {
    "jid": "sample_002",
    "atoms": { ... },
    "formation_energy_peratom": -2.1,
    "text_description": "..."
  }
]
```

### 2. 必需字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `jid` (或 `id`) | string | 唯一标识符 |
| `atoms` | dict | 晶体结构字典 |
| `atoms.lattice_mat` | 3x3 array | 晶格矩阵 |
| `atoms.coords` | Nx3 array | 原子分数坐标 |
| `atoms.elements` | list[string] | 元素符号列表 |
| `<target>` | float/list | 目标属性值 |

### 3. 可选字段

- `text_description`: 材料文本描述（多模态学习）
- `atoms.abc`: 晶格参数 `[a, b, c]`
- `atoms.angles`: 晶格角度 `[α, β, γ]`
- 其他自定义属性

---

## 🔧 准备数据文件

### 方法 1: 从 CIF 文件生成

如果你有 CIF 文件，可以使用以下脚本转换：

```python
import json
from jarvis.core.atoms import Atoms

def cif_to_json(cif_file, target_value, jid):
    """将 CIF 文件转换为数据集格式"""
    atoms = Atoms.from_cif(cif_file)
    return {
        "jid": jid,
        "atoms": atoms.to_dict(),
        "formation_energy_peratom": target_value,
        "text_description": f"Crystal structure from {cif_file}"
    }

# 批量转换
dataset = []
cif_files = [
    ("sample1.cif", -3.5, "sample_001"),
    ("sample2.cif", -2.1, "sample_002"),
]

for cif_file, target, jid in cif_files:
    dataset.append(cif_to_json(cif_file, target, jid))

# 保存为 JSON
with open("my_dataset.json", "w") as f:
    json.dump(dataset, f, indent=2)
```

### 方法 2: 从 POSCAR/VASP 文件生成

```python
from jarvis.core.atoms import Atoms

def poscar_to_json(poscar_file, target_value, jid):
    """将 POSCAR 文件转换为数据集格式"""
    atoms = Atoms.from_poscar(poscar_file)
    return {
        "jid": jid,
        "atoms": atoms.to_dict(),
        "formation_energy_peratom": target_value
    }

# 类似的批量转换...
```

### 方法 3: 手动构造（Python）

```python
from jarvis.core.atoms import Atoms
from jarvis.core.lattice import Lattice

# 创建晶格
lattice = Lattice([[5.0, 0, 0], [0, 5.0, 0], [0, 0, 5.0]])

# 创建原子结构
atoms = Atoms(
    lattice_mat=lattice.matrix,
    coords=[[0, 0, 0], [0.5, 0.5, 0.5]],
    elements=["Si", "Si"]
)

# 构造数据条目
data_entry = {
    "jid": "manual_001",
    "atoms": atoms.to_dict(),
    "formation_energy_peratom": -3.5,
    "text_description": "Silicon diamond structure"
}

dataset = [data_entry]
with open("my_dataset.json", "w") as f:
    json.dump(dataset, f, indent=2)
```

---

## ⚙️ 配置训练

### 1. 创建配置文件

创建 `config_custom_dataset.json`:

```json
{
  "dataset": "user_data",
  "target": "formation_energy_peratom",
  "atom_features": "cgcnn",
  "id_tag": "jid",

  "random_seed": 123,
  "train_ratio": 0.8,
  "val_ratio": 0.1,
  "test_ratio": 0.1,

  "epochs": 300,
  "batch_size": 32,
  "learning_rate": 0.001,
  "optimizer": "adamw",
  "scheduler": "onecycle",
  "n_early_stopping": 50,

  "cutoff": 8.0,
  "max_neighbors": 12,
  "output_dir": "./results_custom",

  "model": {
    "name": "densegnn",
    "densegnn_layers": 4,
    "hidden_features": 256,
    "use_middle_fusion": true,
    "middle_fusion_layers": "1,3",
    "use_cross_modal_attention": true
  }
}
```

### 2. 创建训练脚本

创建 `train_custom_data.py`:

```python
"""训练自定义数据集"""
import json
from config import TrainingConfig
from train import train_dgl
from data import get_train_val_loaders

# 1. 加载数据集
with open("my_dataset.json", "r") as f:
    dataset = json.load(f)

print(f"✅ 加载了 {len(dataset)} 个样本")

# 2. 加载配置
with open("config_custom_dataset.json", "r") as f:
    config_dict = json.load(f)

# 3. 创建训练配置
config = TrainingConfig(**config_dict)

# 4. 准备数据加载器
train_loader, val_loader, test_loader, prepare_batch = get_train_val_loaders(
    dataset="user_data",
    dataset_array=dataset,  # 传入自定义数据
    target=config.target,
    atom_features=config.atom_features,
    id_tag=config.id_tag,
    batch_size=config.batch_size,
    split_seed=config.random_seed,
    train_ratio=config.train_ratio,
    val_ratio=config.val_ratio,
    test_ratio=config.test_ratio,
    cutoff=config.cutoff,
    max_neighbors=config.max_neighbors,
    line_graph=False,  # DenseGNN 不需要线图
    workers=config.num_workers,
    output_dir=config.output_dir
)

# 5. 开始训练
print("\n🚀 开始训练...")
history = train_dgl(
    config=config,
    train_val_test_loaders=[train_loader, val_loader, test_loader, prepare_batch]
)

print("\n✅ 训练完成！")
print(f"结果保存在: {config.output_dir}")
```

---

## 🚀 开始训练

### 命令行运行

```bash
# 运行训练脚本
python train_custom_data.py
```

### 预期输出

```
✅ 加载了 100 个样本
Batch Size: 32
data range 0.5 -5.2
n_train: 80
n_val: 10
n_test: 10

🚀 开始训练...
Epoch: 1
Train_MAE: 0.5234
Val_MAE: 0.4821
Test_MAE: 0.4956
...
```

---

## 📊 数据集要求

### 最小样本数建议

| 任务类型 | 最小样本 | 推荐样本 |
|---------|---------|---------|
| 回归 | 100 | 1000+ |
| 分类 | 200 | 2000+ |
| 多模态 | 200 | 1000+ |

### 数据质量检查

```python
import json
import numpy as np

# 加载数据
with open("my_dataset.json", "r") as f:
    dataset = json.load(f)

# 检查
print(f"总样本数: {len(dataset)}")

# 检查目标值分布
targets = [d["formation_energy_peratom"] for d in dataset]
print(f"目标值范围: [{min(targets):.2f}, {max(targets):.2f}]")
print(f"目标值均值: {np.mean(targets):.2f}")
print(f"目标值标准差: {np.std(targets):.2f}")

# 检查缺失值
missing_count = sum(1 for d in dataset if d.get("formation_energy_peratom") is None)
print(f"缺失值数量: {missing_count}")

# 检查原子数分布
num_atoms = [len(d["atoms"]["elements"]) for d in dataset]
print(f"原子数范围: [{min(num_atoms)}, {max(num_atoms)}]")
print(f"平均原子数: {np.mean(num_atoms):.1f}")
```

---

## 🔍 常见问题

### Q1: 数据格式错误

**错误**: `KeyError: 'atoms'`

**解决**: 确保每个数据条目都有 `atoms` 字段，格式如上所示。

### Q2: 内存不足

**解决**: 减小 `batch_size`，或使用数据加载器缓存：

```json
{
  "batch_size": 16,  // 减小批次大小
  "save_dataloader": true,  // 缓存数据加载器
  "pin_memory": false  // 禁用 pin memory
}
```

### Q3: 训练不收敛

**解决**:
1. 检查数据范围，考虑归一化
2. 调整学习率（0.0001 - 0.01）
3. 增加训练样本数
4. 使用更深的模型（增加 `densegnn_layers`）

### Q4: 文本描述缺失

**解决**:
- 如果没有文本，确保模型配置中未启用多模态：
```json
{
  "use_middle_fusion": false,
  "use_cross_modal_attention": false
}
```

---

## 📁 完整示例

详细示例文件在：
- `train_custom_data.py` - 训练脚本
- `prepare_dataset.py` - 数据准备脚本
- `config_custom_dataset.json` - 配置文件

---

## 💡 进阶技巧

### 1. 数据增强

```python
# 在数据加载时添加噪声
def augment_structure(atoms_dict, noise_level=0.01):
    coords = np.array(atoms_dict["coords"])
    noise = np.random.randn(*coords.shape) * noise_level
    atoms_dict["coords"] = (coords + noise).tolist()
    return atoms_dict
```

### 2. 交叉验证

```python
from sklearn.model_selection import KFold

kf = KFold(n_splits=5, shuffle=True, random_seed=123)
for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
    print(f"训练 Fold {fold+1}/5")
    # 训练代码...
```

### 3. 多目标学习

如果有多个目标属性：

```json
{
  "target": ["formation_energy", "band_gap", "elastic_modulus"],
  "model": {
    "output_features": 3  // 多输出
  }
}
```

---

## 📞 获取帮助

如果遇到问题：
1. 检查数据格式是否正确
2. 查看 `output_dir` 中的日志文件
3. 参考 `README_DenseGNN.md` 了解模型配置
4. 提交 Issue 到 GitHub

---

**祝训练顺利！** 🎉
