"""准备自定义数据集的示例脚本

支持从多种格式创建数据集:
- CIF 文件
- POSCAR 文件
- 手动构造

使用方法:
    python prepare_dataset.py --help
"""

import json
import argparse
import os
from pathlib import Path
from jarvis.core.atoms import Atoms


def from_cif_files(cif_dir, target_file, output_file):
    """从 CIF 文件目录创建数据集

    Args:
        cif_dir: CIF 文件目录
        target_file: 目标值 CSV 文件 (格式: filename,target_value)
        output_file: 输出 JSON 文件
    """
    import pandas as pd

    print(f"📂 从 CIF 文件创建数据集...")
    print(f"   CIF 目录: {cif_dir}")
    print(f"   目标文件: {target_file}")

    # 读取目标值
    targets_df = pd.read_csv(target_file)
    print(f"   读取 {len(targets_df)} 个目标值")

    dataset = []
    for idx, row in targets_df.iterrows():
        filename = row['filename']
        target_value = row['target_value']

        cif_path = os.path.join(cif_dir, filename)
        if not os.path.exists(cif_path):
            print(f"   ⚠️  跳过: {filename} (文件不存在)")
            continue

        try:
            # 从 CIF 加载结构
            atoms = Atoms.from_cif(cif_path)

            # 创建数据条目
            data_entry = {
                "jid": Path(filename).stem,  # 使用文件名（不含扩展名）作为 ID
                "atoms": atoms.to_dict(),
                "formation_energy_peratom": float(target_value),
                "text_description": f"Structure from {filename}"
            }

            dataset.append(data_entry)

        except Exception as e:
            print(f"   ❌ 错误: {filename} - {str(e)}")

    # 保存数据集
    with open(output_file, "w") as f:
        json.dump(dataset, f, indent=2)

    print(f"\n✅ 成功创建数据集: {output_file}")
    print(f"   总样本数: {len(dataset)}")


def from_poscar_files(poscar_dir, target_file, output_file):
    """从 POSCAR 文件目录创建数据集"""
    import pandas as pd

    print(f"📂 从 POSCAR 文件创建数据集...")
    print(f"   POSCAR 目录: {poscar_dir}")
    print(f"   目标文件: {target_file}")

    targets_df = pd.read_csv(target_file)
    print(f"   读取 {len(targets_df)} 个目标值")

    dataset = []
    for idx, row in targets_df.iterrows():
        filename = row['filename']
        target_value = row['target_value']

        poscar_path = os.path.join(poscar_dir, filename)
        if not os.path.exists(poscar_path):
            print(f"   ⚠️  跳过: {filename} (文件不存在)")
            continue

        try:
            atoms = Atoms.from_poscar(poscar_path)

            data_entry = {
                "jid": Path(filename).stem,
                "atoms": atoms.to_dict(),
                "formation_energy_peratom": float(target_value),
                "text_description": f"Structure from {filename}"
            }

            dataset.append(data_entry)

        except Exception as e:
            print(f"   ❌ 错误: {filename} - {str(e)}")

    with open(output_file, "w") as f:
        json.dump(dataset, f, indent=2)

    print(f"\n✅ 成功创建数据集: {output_file}")
    print(f"   总样本数: {len(dataset)}")


def create_example_dataset(output_file, num_samples=10):
    """创建示例数据集（用于测试）"""
    import numpy as np
    from jarvis.core.lattice import Lattice

    print(f"📝 创建示例数据集 ({num_samples} 个样本)...")

    dataset = []

    # 创建不同的示例结构
    structures = [
        # 简单立方
        {
            "lattice": [[4.0, 0, 0], [0, 4.0, 0], [0, 0, 4.0]],
            "coords": [[0, 0, 0]],
            "elements": ["Si"],
            "name": "simple_cubic_Si"
        },
        # 面心立方
        {
            "lattice": [[4.0, 0, 0], [0, 4.0, 0], [0, 0, 4.0]],
            "coords": [[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]],
            "elements": ["Al", "Al", "Al", "Al"],
            "name": "fcc_Al"
        },
        # 金刚石结构
        {
            "lattice": [[5.43, 0, 0], [0, 5.43, 0], [0, 0, 5.43]],
            "coords": [
                [0, 0, 0], [0.25, 0.25, 0.25],
                [0.5, 0.5, 0], [0.75, 0.75, 0.25],
                [0.5, 0, 0.5], [0.75, 0.25, 0.75],
                [0, 0.5, 0.5], [0.25, 0.75, 0.75]
            ],
            "elements": ["Si"] * 8,
            "name": "diamond_Si"
        }
    ]

    for i in range(num_samples):
        # 选择一个结构模板
        template = structures[i % len(structures)]

        # 稍微修改晶格参数（增加多样性）
        lattice_scale = 1.0 + np.random.randn() * 0.05
        lattice = Lattice(np.array(template["lattice"]) * lattice_scale)

        # 创建原子结构
        atoms = Atoms(
            lattice_mat=lattice.matrix,
            coords=template["coords"],
            elements=template["elements"]
        )

        # 生成随机目标值（示例）
        target_value = -3.0 + np.random.randn() * 1.0

        data_entry = {
            "jid": f"{template['name']}_{i:03d}",
            "atoms": atoms.to_dict(),
            "formation_energy_peratom": float(target_value),
            "text_description": f"Example {template['name']} structure with scale {lattice_scale:.3f}"
        }

        dataset.append(data_entry)

    # 保存数据集
    with open(output_file, "w") as f:
        json.dump(dataset, f, indent=2)

    print(f"\n✅ 成功创建示例数据集: {output_file}")
    print(f"   总样本数: {len(dataset)}")
    print(f"   结构类型: {len(structures)}")


def validate_dataset(dataset_file):
    """验证数据集文件"""
    print(f"🔍 验证数据集: {dataset_file}")

    with open(dataset_file, "r") as f:
        dataset = json.load(f)

    print(f"   总样本数: {len(dataset)}")

    # 检查格式
    required_keys = ["jid", "atoms"]
    errors = []

    for i, data in enumerate(dataset[:10]):  # 只检查前10个
        for key in required_keys:
            if key not in data:
                errors.append(f"样本 {i}: 缺少 '{key}'")

        if "atoms" in data:
            atoms_keys = ["lattice_mat", "coords", "elements"]
            for key in atoms_keys:
                if key not in data["atoms"]:
                    errors.append(f"样本 {i}: atoms 缺少 '{key}'")

    if errors:
        print("   ❌ 发现问题:")
        for error in errors:
            print(f"      - {error}")
    else:
        print("   ✅ 格式正确")


def main():
    parser = argparse.ArgumentParser(description="准备自定义数据集")
    subparsers = parser.add_subparsers(dest="command", help="命令")

    # from-cif 命令
    cif_parser = subparsers.add_parser("from-cif", help="从 CIF 文件创建数据集")
    cif_parser.add_argument("--cif-dir", required=True, help="CIF 文件目录")
    cif_parser.add_argument("--target-file", required=True, help="目标值 CSV 文件")
    cif_parser.add_argument("--output", default="my_dataset.json", help="输出 JSON 文件")

    # from-poscar 命令
    poscar_parser = subparsers.add_parser("from-poscar", help="从 POSCAR 文件创建数据集")
    poscar_parser.add_argument("--poscar-dir", required=True, help="POSCAR 文件目录")
    poscar_parser.add_argument("--target-file", required=True, help="目标值 CSV 文件")
    poscar_parser.add_argument("--output", default="my_dataset.json", help="输出 JSON 文件")

    # example 命令
    example_parser = subparsers.add_parser("example", help="创建示例数据集")
    example_parser.add_argument("--output", default="example_dataset.json", help="输出 JSON 文件")
    example_parser.add_argument("--num-samples", type=int, default=10, help="样本数量")

    # validate 命令
    validate_parser = subparsers.add_parser("validate", help="验证数据集")
    validate_parser.add_argument("dataset_file", help="数据集 JSON 文件")

    args = parser.parse_args()

    if args.command == "from-cif":
        from_cif_files(args.cif_dir, args.target_file, args.output)
    elif args.command == "from-poscar":
        from_poscar_files(args.poscar_dir, args.target_file, args.output)
    elif args.command == "example":
        create_example_dataset(args.output, args.num_samples)
    elif args.command == "validate":
        validate_dataset(args.dataset_file)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
