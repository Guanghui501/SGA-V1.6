# 快速开始：自定义数据集训练

## 🚀 5分钟快速上手

### 步骤 1: 创建示例数据集

```bash
# 生成一个包含10个样本的示例数据集
python prepare_dataset.py example --num-samples 10 --output example_dataset.json
```

**输出**:
```
✅ 成功创建示例数据集: example_dataset.json
   总样本数: 10
```

### 步骤 2: 验证数据集（可选）

```bash
# 检查数据集格式是否正确
python train_custom_data.py --dataset example_dataset.json --validate-only
```

**输出**:
```
✅ 数据集格式正确
📊 数据集统计:
   总样本数: 10
   目标范围: [-4.2345, -1.8765]
   ...
```

### 步骤 3: 开始训练

```bash
# 使用示例数据集训练 DenseGNN 模型
python train_custom_data.py --dataset example_dataset.json --config config_custom_dataset.json
```

**训练过程**:
```
🚀 开始训练 DENSEGNN 模型...
   训练集: 8 样本
   验证集: 1 样本
   测试集: 1 样本

Epoch: 1
Train_MAE: 0.5234
Val_MAE: 0.4821
...
🎉 训练完成！
```

### 步骤 4: 查看结果

```bash
# 结果保存在 results_custom_dataset/ 目录
ls results_custom_dataset/

# 输出:
# best_val_model.pt
# best_test_model.pt
# predictions_*.csv
# history_*.json
```

---

## 📁 使用自己的数据

### 从 CIF 文件

```bash
# 1. 准备目标值文件 (targets.csv)
echo "filename,target_value" > targets.csv
echo "structure1.cif,-3.5" >> targets.csv
echo "structure2.cif,-2.1" >> targets.csv

# 2. 从 CIF 创建数据集
python prepare_dataset.py from-cif \
    --cif-dir ./my_cif_files/ \
    --target-file targets.csv \
    --output my_dataset.json

# 3. 训练
python train_custom_data.py --dataset my_dataset.json
```

### 从 POSCAR 文件

```bash
# 从 POSCAR 文件创建数据集
python prepare_dataset.py from-poscar \
    --poscar-dir ./my_poscar_files/ \
    --target-file targets.csv \
    --output my_dataset.json

# 训练
python train_custom_data.py --dataset my_dataset.json
```

### 手动创建 JSON

创建 `my_dataset.json`:

```json
[
  {
    "jid": "sample_001",
    "atoms": {
      "lattice_mat": [[5.0, 0, 0], [0, 5.0, 0], [0, 0, 5.0]],
      "coords": [[0, 0, 0], [0.5, 0.5, 0.5]],
      "elements": ["Si", "Si"]
    },
    "formation_energy_peratom": -3.5,
    "text_description": "Silicon crystal"
  }
]
```

```bash
python train_custom_data.py --dataset my_dataset.json
```

---

## ⚙️ 常用配置调整

### 修改模型架构

编辑 `config_custom_dataset.json`:

```json
{
  "model": {
    "name": "densegnn",
    "densegnn_layers": 6,        // 更深的模型
    "hidden_features": 512,      // 更大的隐藏维度
    "use_middle_fusion": true,
    "use_cross_modal_attention": true
  }
}
```

### 调整训练参数

```json
{
  "epochs": 500,              // 更多训练轮数
  "batch_size": 16,           // 更小的批次（内存不足时）
  "learning_rate": 0.0001,    // 更小的学习率
  "n_early_stopping": 100     // 更宽松的早停
}
```

### 修改数据分割

```json
{
  "train_ratio": 0.7,   // 70% 训练
  "val_ratio": 0.15,    // 15% 验证
  "test_ratio": 0.15    // 15% 测试
}
```

---

## 🔧 常见问题

### 问题 1: 内存不足

**解决**:
```json
{
  "batch_size": 8,        // 减小批次
  "num_workers": 0,       // 禁用多进程
  "pin_memory": false     // 禁用 pin memory
}
```

### 问题 2: 训练太慢

**解决**:
```json
{
  "num_workers": 4,       // 使用多进程
  "save_dataloader": true // 缓存数据加载器
}
```

### 问题 3: 数据格式错误

**检查**:
```bash
python train_custom_data.py --dataset my_dataset.json --validate-only
```

---

## 📊 评估模型

### 查看训练历史

```python
import json

with open("results_custom_dataset/history_val.json", "r") as f:
    history = json.load(f)

# 绘制学习曲线
import matplotlib.pyplot as plt

plt.plot(history["mae"])
plt.xlabel("Epoch")
plt.ylabel("MAE")
plt.show()
```

### 分析预测结果

```python
import pandas as pd

# 读取预测结果
predictions = pd.read_csv("results_custom_dataset/predictions_best_val_model_test.csv")

# 计算误差
predictions["error"] = predictions["prediction"] - predictions["target"]

# 统计
print(f"平均绝对误差: {predictions['error'].abs().mean():.4f}")
print(f"均方根误差: {(predictions['error']**2).mean()**0.5:.4f}")
```

---

## 📚 完整文档

- **详细指南**: `GUIDE_CUSTOM_DATASET.md`
- **DenseGNN 文档**: `README_DenseGNN.md`
- **模型配置**: `config_custom_dataset.json`

---

## 💡 提示

1. **从小数据集开始**: 先用少量样本测试流程
2. **验证数据**: 训练前务必验证数据格式
3. **监控训练**: 使用 TensorBoard 监控（如果启用）
4. **保存检查点**: 定期备份 `best_*_model.pt`
5. **记录配置**: 每次实验保存配置文件

---

**祝训练顺利！** 🎉
