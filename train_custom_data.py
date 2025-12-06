"""训练自定义数据集的示例脚本

使用方法:
    1. 准备数据集 JSON 文件 (my_dataset.json)
    2. 修改配置文件 (config_custom_dataset.json)
    3. 运行: python train_custom_data.py
"""

import json
import os
import argparse
from config import TrainingConfig
from train import train_dgl
from data import get_train_val_loaders


def load_dataset(dataset_file):
    """加载数据集 JSON 文件"""
    print(f"📂 加载数据集: {dataset_file}")

    if not os.path.exists(dataset_file):
        raise FileNotFoundError(f"❌ 数据集文件不存在: {dataset_file}")

    with open(dataset_file, "r") as f:
        dataset = json.load(f)

    print(f"✅ 成功加载 {len(dataset)} 个样本")
    return dataset


def validate_dataset(dataset, target_key):
    """验证数据集格式"""
    print("\n🔍 验证数据集格式...")

    required_keys = ["jid", "atoms"]
    errors = []

    for i, data in enumerate(dataset):
        # 检查必需字段
        for key in required_keys:
            if key not in data:
                errors.append(f"样本 {i}: 缺少字段 '{key}'")

        # 检查 atoms 结构
        if "atoms" in data:
            atoms = data["atoms"]
            required_atom_keys = ["lattice_mat", "coords", "elements"]
            for key in required_atom_keys:
                if key not in atoms:
                    errors.append(f"样本 {i}: atoms 缺少字段 '{key}'")

        # 检查目标值
        if target_key not in data:
            errors.append(f"样本 {i}: 缺少目标字段 '{target_key}'")
        elif data[target_key] is None:
            errors.append(f"样本 {i}: 目标值为 None")

    if errors:
        print("❌ 发现以下问题:")
        for error in errors[:10]:  # 只显示前10个错误
            print(f"   - {error}")
        if len(errors) > 10:
            print(f"   ... 还有 {len(errors) - 10} 个错误")
        raise ValueError("数据集验证失败")

    print("✅ 数据集格式正确")


def print_dataset_stats(dataset, target_key):
    """打印数据集统计信息"""
    import numpy as np

    print("\n📊 数据集统计:")
    print(f"   总样本数: {len(dataset)}")

    # 目标值统计
    targets = [d[target_key] for d in dataset if d.get(target_key) is not None]
    if targets:
        if isinstance(targets[0], list):
            print(f"   目标类型: 多输出 (维度={len(targets[0])})")
        else:
            print(f"   目标范围: [{min(targets):.4f}, {max(targets):.4f}]")
            print(f"   目标均值: {np.mean(targets):.4f}")
            print(f"   目标标准差: {np.std(targets):.4f}")

    # 原子数统计
    num_atoms = [len(d["atoms"]["elements"]) for d in dataset]
    print(f"   原子数范围: [{min(num_atoms)}, {max(num_atoms)}]")
    print(f"   平均原子数: {np.mean(num_atoms):.1f}")

    # 文本描述统计
    has_text = sum(1 for d in dataset if d.get("text_description"))
    print(f"   包含文本描述: {has_text}/{len(dataset)} ({has_text/len(dataset)*100:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="训练自定义数据集")
    parser.add_argument("--dataset", type=str, default="my_dataset.json",
                       help="数据集 JSON 文件路径")
    parser.add_argument("--config", type=str, default="config_custom_dataset.json",
                       help="配置文件路径")
    parser.add_argument("--validate-only", action="store_true",
                       help="仅验证数据集，不训练")
    args = parser.parse_args()

    # 1. 加载数据集
    dataset = load_dataset(args.dataset)

    # 2. 加载配置
    print(f"\n📋 加载配置: {args.config}")
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"❌ 配置文件不存在: {args.config}")

    with open(args.config, "r") as f:
        config_dict = json.load(f)

    # 3. 验证数据集
    target_key = config_dict.get("target", "formation_energy_peratom")
    validate_dataset(dataset, target_key)
    print_dataset_stats(dataset, target_key)

    if args.validate_only:
        print("\n✅ 验证完成，退出")
        return

    # 4. 创建训练配置
    print("\n⚙️  创建训练配置...")
    config = TrainingConfig(**config_dict)

    # 5. 准备数据加载器
    print("\n🔄 准备数据加载器...")

    # 检查是否使用 DenseGNN（不需要线图）
    use_line_graph = config.model.name not in ["densegnn"]

    train_loader, val_loader, test_loader, prepare_batch = get_train_val_loaders(
        dataset="user_data",
        dataset_array=dataset,
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
        line_graph=use_line_graph,
        workers=config.num_workers,
        output_dir=config.output_dir,
        use_canonize=config.use_canonize,
        save_dataloader=config.save_dataloader,
        keep_data_order=config.keep_data_order
    )

    print(f"\n✅ 数据加载器准备完成")
    print(f"   训练集: {len(train_loader.dataset)} 样本")
    print(f"   验证集: {len(val_loader.dataset)} 样本")
    print(f"   测试集: {len(test_loader.dataset)} 样本")

    # 6. 开始训练
    print(f"\n🚀 开始训练 {config.model.name.upper()} 模型...")
    print(f"   模型: {config.model.name}")
    print(f"   输出目录: {config.output_dir}")
    print(f"   训练轮数: {config.epochs}")
    print(f"   批次大小: {config.batch_size}")
    print(f"   学习率: {config.learning_rate}")
    print("=" * 80)

    history = train_dgl(
        config=config,
        train_val_test_loaders=[train_loader, val_loader, test_loader, prepare_batch]
    )

    # 7. 训练完成
    print("\n" + "=" * 80)
    print("🎉 训练完成！")
    print("=" * 80)
    print(f"\n📁 结果保存在: {config.output_dir}")
    print(f"   - best_val_model.pt: 最佳验证集模型")
    print(f"   - best_test_model.pt: 最佳测试集模型")
    print(f"   - config.json: 训练配置")
    print(f"   - history_*.json: 训练历史")
    print(f"   - predictions_*.csv: 预测结果")

    # 打印最终结果
    if history and "validation" in history:
        final_val_mae = history["validation"]["mae"][-1]
        final_test_mae = history["test"]["mae"][-1]
        print(f"\n📊 最终结果:")
        print(f"   验证集 MAE: {final_val_mae:.4f}")
        print(f"   测试集 MAE: {final_test_mae:.4f}")


if __name__ == "__main__":
    main()
