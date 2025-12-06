#!/usr/bin/env python
"""
本地数据训练脚本 - 使用CIF文件 + CSV描述
（照抄 train_with_cross_modal_attention.py 的数据集处理方式）

支持的数据集:
    - jarvis: JARVIS-DFT 数据集
    - mp: Material Project 数据集
    - class: 分类数据集
    - toy: 玩具数据集（测试用）
    - custom: 自定义数据集

CSV格式（按数据集类型）:
    JARVIS/TOY: Id, Composition, prop, Description, File_Name
    MP:         id, composition, formation_energy, band_gap, description, file_name
    Class:      id, target, description
    Custom:     由 --target_column, --text_column, --id_column 参数指定

使用示例:
    # 1. JARVIS 数据集
    python train_local_cif_csv.py \\
        --root_dir ../dataset/ \\
        --dataset jarvis \\
        --property formation_energy \\
        --model densegnn \\
        --use_middle_fusion

    # 2. Material Project 数据集
    python train_local_cif_csv.py \\
        --root_dir ../dataset/ \\
        --dataset mp \\
        --property band_gap \\
        --model densegnn \\
        --use_cross_modal

    # 3. 自定义数据集（与之前一样）
    python train_local_cif_csv.py \\
        --dataset custom \\
        --cif_dir ./my_structures/cif/ \\
        --csv_file ./my_structures/data.csv \\
        --model densegnn \\
        --use_middle_fusion \\
        --use_cross_modal

    # 4. 分类任务
    python train_local_cif_csv.py \\
        --dataset class \\
        --property syn \\
        --classification \\
        --num_classes 2
"""

import os
import sys
import csv
import time
import json
import argparse
import numpy as np
from tqdm import tqdm

import torch
from jarvis.core.atoms import Atoms
from transformers import AutoTokenizer
from tokenizers.normalizers import BertNormalizer

from data import get_train_val_loaders
from train import train_dgl
from config import TrainingConfig
from models.alignn import ALIGNNConfig
from models.densegnn import DenseGNNConfig


# ==================== 辅助函数 ====================

def str2bool(v):
    """字符串转布尔值"""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('布尔值应为 yes/no, true/false, t/f, y/n, 1/0')


# ==================== 文本归一化 ====================

def setup_text_normalizer():
    """设置文本归一化器"""
    # BERT归一化器
    norm = BertNormalizer(lowercase=False, strip_accents=True,
                         clean_text=True, handle_chinese_chars=True)

    # 加载词汇映射
    possible_paths = [
        'vocab_mappings.txt',
        './vocab_mappings.txt',
        os.path.join(os.path.dirname(__file__), 'vocab_mappings.txt'),
    ]

    vocab_file = None
    for path in possible_paths:
        if os.path.exists(path):
            vocab_file = path
            break

    if vocab_file is None:
        print("⚠️  未找到 vocab_mappings.txt，使用默认文本归一化")
        mappings = {}
    else:
        with open(vocab_file, 'r') as f:
            mappings_list = f.read().strip().split('\n')
        mappings = {m[0]: m[2:] for m in mappings_list}

    def normalize(text):
        """归一化文本"""
        text = [norm.normalize_str(s) for s in text.split('\n')]
        out = []
        for s in text:
            norm_s = ''
            for c in s:
                norm_s += mappings.get(c, ' ') if mappings else c
            out.append(norm_s)
        return '\n'.join(out)

    return normalize


# ==================== 数据集路径配置（照抄 train_with_cross_modal_attention.py）====================

def get_dataset_paths(root_dir, dataset, property_name):
    """根据数据集和性质获取数据路径

    Args:
        root_dir: 数据集根目录
        dataset: 数据集名称 (jarvis, mp, class, toy, custom)
        property_name: 属性名称

    Returns:
        cif_dir, csv_file: CIF目录路径和CSV文件路径
    """
    if dataset.lower() == 'jarvis':
        # JARVIS-DFT 数据集
        property_map = {
            'formation_energy': 'formation_energy_peratom',
            'fe': 'formation_energy_peratom',
            'total_energy': 'optb88vdw_total_energy',
            'opt_bandgap': 'optb88vdw_bandgap',
            'mbj_bandgap': 'mbj_bandgap',
            'bulk_modulus': 'bulk_modulus_kv',
            'bulk_modulus_kv': 'bulk_modulus_kv',
            'shear_modulus': 'shear_modulus_gv',
            'shear_modulus_gv': 'shear_modulus_gv',
        }

        prop_folder = property_map.get(property_name, property_name)
        cif_dir = os.path.join(root_dir, f'jarvis/{prop_folder}/cif/')
        csv_file = os.path.join(root_dir, f'jarvis/{prop_folder}/description.csv')

    elif dataset.lower() == 'mp':
        # Material Project 数据集
        if property_name in ['formation_energy', 'band_gap']:
            cif_dir = os.path.join(root_dir, 'mp_2018_new/')
            csv_file = os.path.join(root_dir, 'mp_2018_new/mat_text.csv')
        elif property_name in ['bulk', 'shear', 'bulk_modulus', 'shear_modulus']:
            cif_dir = os.path.join(root_dir, 'mp_2018_small/cif/')
            csv_file = os.path.join(root_dir, 'mp_2018_small/description.csv')
        else:
            raise ValueError(f"Unsupported property for MP dataset: {property_name}")

    elif dataset.lower() == 'class':
        # 分类数据集（类似jarvis结构）
        # 例如：class/syn, class/metal_oxide, 等
        cif_dir = os.path.join(root_dir, f'class/{property_name}/cif/')
        csv_file = os.path.join(root_dir, f'class/{property_name}/description.csv')

    elif dataset.lower() == 'toy':
        # 玩具数据集（用于测试）
        cif_dir = os.path.join(root_dir, 'toy/cif/')
        csv_file = os.path.join(root_dir, 'toy/description.csv')

    elif dataset.lower() == 'custom':
        # 自定义数据集：由命令行参数指定
        # 这种情况下，cif_dir 和 csv_file 会在 main() 中直接使用参数值
        return None, None

    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    return cif_dir, csv_file


# ==================== 数据加载 ====================

def load_dataset_from_cif_csv(cif_dir, csv_file, dataset='custom', property_name='formation_energy',
                               target_column='target', text_column='text_description',
                               id_column='id'):
    """从CIF目录和CSV文件加载数据集

    Args:
        cif_dir: CIF文件目录
        csv_file: CSV描述文件路径
        target_column: CSV中目标值列名
        text_column: CSV中文本描述列名（可选）
        id_column: CSV中ID列名

    Returns:
        dataset_array: 数据列表，每个元素是字典
    """
    print(f"\n{'='*80}")
    print(f"📂 加载本地数据集")
    print(f"{'='*80}")
    print(f"CIF目录: {cif_dir}")
    print(f"CSV文件: {csv_file}")
    print(f"目标列: {target_column}")
    print(f"文本列: {text_column}")
    print(f"{'='*80}\n")

    # 检查文件存在性
    if not os.path.exists(cif_dir):
        raise FileNotFoundError(f"CIF目录不存在: {cif_dir}")
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"CSV文件不存在: {csv_file}")

    # 读取CSV文件（照抄 train_with_cross_modal_attention.py 的方式）
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        headings = next(reader)
        data = [row for row in reader]

    print(f"CSV文件共 {len(data)} 行")
    print(f"CSV列名: {headings}")

    # 检查必需列
    if len(data) == 0:
        raise ValueError("CSV文件为空")

    # 设置文本归一化器
    normalize_text = setup_text_normalizer()

    # 加载数据
    dataset_array = []
    skipped = 0
    errors = []

    for j in tqdm(range(len(data)), desc="加载样本"):
        try:
            # 根据不同数据集解析CSV行（照抄 train_with_cross_modal_attention.py）
            if dataset.lower() == 'mp':
                if property_name == 'formation_energy':
                    sample_id, composition, target_value, _, crys_desc_full, _ = data[j]
                elif property_name == 'band_gap':
                    sample_id, composition, _, target_value, crys_desc_full, _ = data[j]
                elif property_name == 'shear':
                    sample_id, composition, target_value, _, crys_desc_full, _ = data[j]
                elif property_name in ['bulk', 'bulk_modulus']:
                    sample_id, composition, _, target_value, crys_desc_full, _ = data[j]
                else:
                    sample_id, composition, target_value, _, crys_desc_full, _ = data[j]

            elif dataset.lower() == 'jarvis':
                # JARVIS格式: Id, Composition, prop, Description, File_Name
                sample_id, composition, target_value, crys_desc_full, _ = data[j]

            elif dataset.lower() == 'class':
                # 分类数据集格式: id, target, description
                sample_id, target_value, crys_desc_full = data[j]
                composition = ''  # 分类任务不需要composition

            elif dataset.lower() == 'toy':
                sample_id, composition, target_value, crys_desc_full, _ = data[j]

            elif dataset.lower() == 'custom':
                # 自定义格式：使用 DictReader 解析
                row_dict = dict(zip(headings, data[j]))
                sample_id = row_dict[id_column].strip()
                target_value = float(row_dict[target_column])
                crys_desc_full = row_dict.get(text_column, f"Crystal structure {sample_id}")
            else:
                raise ValueError(f"Unsupported dataset: {dataset}")

            # 标准化文本描述
            text_desc = normalize_text(crys_desc_full)

            # CIF文件路径
            cif_file = os.path.join(cif_dir, f"{sample_id}.cif")
            if not os.path.exists(cif_file):
                skipped += 1
                if len(errors) < 5:
                    errors.append(f"CIF文件不存在: {cif_file}")
                continue

            # 加载结构
            atoms = Atoms.from_cif(cif_file)

            # 构建样本（与 train_with_cross_modal_attention.py 一致）
            sample = {
                "atoms": atoms.to_dict(),
                "jid": sample_id,
                "text": text_desc,  # 使用 "text" 而不是 "text_description"
                "target": float(target_value)
            }

            # MP数据集的特殊处理（对数变换）
            if dataset.lower() == 'mp' and property_name in ['shear', 'bulk', 'bulk_modulus', 'shear_modulus']:
                sample["target"] = np.log10(float(target_value))

            dataset_array.append(sample)

        except Exception as e:
            skipped += 1
            if len(errors) < 5:
                errors.append(f"样本 {j}: {str(e)}")

    # 打印统计
    print(f"\n✅ 成功加载: {len(dataset_array)} 样本")
    if skipped > 0:
        print(f"⚠️  跳过: {skipped} 样本")
        if errors:
            print("\n前几个错误:")
            for err in errors:
                print(f"   - {err}")

    if len(dataset_array) == 0:
        raise ValueError("没有成功加载任何样本！请检查CIF文件路径和CSV格式。")

    # 打印数据统计
    targets = [d['target'] for d in dataset_array]
    print(f"\n📊 数据统计:")
    print(f"   样本数: {len(dataset_array)}")
    print(f"   目标值范围: [{min(targets):.4f}, {max(targets):.4f}]")
    print(f"   目标值均值: {np.mean(targets):.4f}")
    print(f"   目标值标准差: {np.std(targets):.4f}")

    num_atoms = [len(d['atoms']['elements']) for d in dataset_array]
    print(f"   原子数范围: [{min(num_atoms)}, {max(num_atoms)}]")
    print(f"   平均原子数: {np.mean(num_atoms):.1f}\n")

    return dataset_array


# ==================== 配置生成 ====================

def create_config(args, dataset_array):
    """根据命令行参数创建训练配置"""

    # 选择模型配置
    if args.model == 'densegnn':
        model_config = DenseGNNConfig(
            name="densegnn",
            densegnn_layers=args.densegnn_layers,
            atom_input_features=92,
            edge_input_features=80,
            embedding_features=64,
            hidden_features=args.hidden_features,
            output_features=1,
            graph_dropout=args.graph_dropout,
            # 中期融合
            use_middle_fusion=args.use_middle_fusion,
            middle_fusion_layers=args.middle_fusion_layers,
            middle_fusion_hidden_dim=args.middle_fusion_hidden_dim,
            middle_fusion_num_heads=args.middle_fusion_num_heads,
            middle_fusion_dropout=args.middle_fusion_dropout,
            # 晚期融合
            use_cross_modal_attention=args.use_cross_modal,
            cross_modal_hidden_dim=args.cross_modal_hidden_dim,
            cross_modal_num_heads=args.cross_modal_num_heads,
            cross_modal_dropout=args.cross_modal_dropout,
            # 对比学习
            use_contrastive_loss=args.use_contrastive,
            contrastive_temperature=args.contrastive_temperature,
            contrastive_loss_weight=args.contrastive_weight,
            link="identity",
            classification=args.classification
        )
    else:  # alignn
        model_config = ALIGNNConfig(
            name="alignn",
            alignn_layers=args.alignn_layers,
            gcn_layers=args.gcn_layers,
            atom_input_features=92,
            edge_input_features=80,
            triplet_input_features=40,
            embedding_features=64,
            hidden_features=args.hidden_features,
            output_features=1,
            graph_dropout=args.graph_dropout,
            # 中期融合
            use_middle_fusion=args.use_middle_fusion,
            middle_fusion_layers=args.middle_fusion_layers,
            middle_fusion_hidden_dim=args.middle_fusion_hidden_dim,
            middle_fusion_num_heads=args.middle_fusion_num_heads,
            middle_fusion_dropout=args.middle_fusion_dropout,
            # 晚期融合
            use_cross_modal_attention=args.use_cross_modal,
            cross_modal_hidden_dim=args.cross_modal_hidden_dim,
            cross_modal_num_heads=args.cross_modal_num_heads,
            cross_modal_dropout=args.cross_modal_dropout,
            # 对比学习
            use_contrastive_loss=args.use_contrastive,
            contrastive_temperature=args.contrastive_temperature,
            contrastive_loss_weight=args.contrastive_weight,
            link="identity",
            classification=args.classification
        )

    # 创建训练配置
    config_dict = {
        "dataset": "user_data",
        "target": "target",
        "atom_features": "cgcnn",
        "neighbor_strategy": "k-nearest",
        "id_tag": "jid",

        "random_seed": args.random_seed,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
        "n_train": args.n_train,
        "n_val": args.n_val,
        "n_test": args.n_test,

        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "optimizer": "adamw",
        "scheduler": "onecycle",
        "criterion": "mse",

        "n_early_stopping": args.early_stopping_patience,
        "output_dir": args.output_dir,

        "cutoff": 8.0,
        "max_neighbors": 12,
        "num_workers": args.num_workers,
        "pin_memory": True,
        "save_dataloader": False,
        "use_canonize": True,
        "keep_data_order": False,

        "classification_threshold": args.classification_threshold if args.classification else None,

        "model": model_config
    }

    return TrainingConfig(**config_dict)


# ==================== 命令行参数 ====================

def get_parser():
    """构建命令行参数解析器"""
    parser = argparse.ArgumentParser(
        description='本地CIF+CSV数据训练脚本',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # ========== 数据集路径参数（照抄 train_with_cross_modal_attention.py）==========
    parser.add_argument('--root_dir', type=str, default='../dataset/',
                       help='数据集根目录（相对于当前目录或绝对路径）')
    parser.add_argument('--dataset', type=str, default='custom',
                       choices=['jarvis', 'mp', 'class', 'toy', 'custom'],
                       help='数据集名称: jarvis, mp, class (分类), toy, custom (自定义)')
    parser.add_argument('--property', type=str, default='formation_energy',
                       help='预测的性质 (回归: formation_energy, band_gap; 分类: syn, metal_oxide等)')

    # ========== 自定义数据路径（当 dataset=custom 时使用）==========
    parser.add_argument('--cif_dir', type=str, default=None,
                       help='CIF文件目录路径（当dataset=custom时必需）')
    parser.add_argument('--csv_file', type=str, default=None,
                       help='CSV描述文件路径（当dataset=custom时必需）')
    parser.add_argument('--target_column', type=str, default='target',
                       help='CSV中目标值列名')
    parser.add_argument('--text_column', type=str, default='text_description',
                       help='CSV中文本描述列名')
    parser.add_argument('--id_column', type=str, default='id',
                       help='CSV中样本ID列名')

    # 数据划分
    parser.add_argument('--train_ratio', type=float, default=0.8,
                       help='训练集比例')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                       help='验证集比例')
    parser.add_argument('--test_ratio', type=float, default=0.1,
                       help='测试集比例')
    parser.add_argument('--n_train', type=int, default=None,
                       help='训练样本数（覆盖train_ratio）')
    parser.add_argument('--n_val', type=int, default=None,
                       help='验证样本数')
    parser.add_argument('--n_test', type=int, default=None,
                       help='测试样本数')

    # 训练参数
    parser.add_argument('--batch_size', type=int, default=32,
                       help='批次大小')
    parser.add_argument('--epochs', type=int, default=300,
                       help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='学习率')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                       help='权重衰减')
    parser.add_argument('--early_stopping_patience', type=int, default=50,
                       help='Early stopping耐心值')

    # 模型选择
    parser.add_argument('--model', type=str, default='densegnn',
                       choices=['densegnn', 'alignn'],
                       help='模型类型')

    # DenseGNN参数
    parser.add_argument('--densegnn_layers', type=int, default=4,
                       help='DenseGNN层数')

    # ALIGNN参数
    parser.add_argument('--alignn_layers', type=int, default=4,
                       help='ALIGNN层数')
    parser.add_argument('--gcn_layers', type=int, default=4,
                       help='GCN层数')

    # 通用模型参数
    parser.add_argument('--hidden_features', type=int, default=256,
                       help='隐藏层特征维度')
    parser.add_argument('--graph_dropout', type=float, default=0.0,
                       help='图层dropout率')

    # 中期融合
    parser.add_argument('--use_middle_fusion', type=str2bool, default=True,
                       help='是否使用中期融合')
    parser.add_argument('--middle_fusion_layers', type=str, default='1,3',
                       help='中期融合层索引（逗号分隔）')
    parser.add_argument('--middle_fusion_hidden_dim', type=int, default=128,
                       help='中期融合隐藏维度')
    parser.add_argument('--middle_fusion_num_heads', type=int, default=2,
                       help='中期融合注意力头数')
    parser.add_argument('--middle_fusion_dropout', type=float, default=0.1,
                       help='中期融合dropout率')

    # 晚期融合
    parser.add_argument('--use_cross_modal', type=str2bool, default=True,
                       help='是否使用跨模态注意力（晚期融合）')
    parser.add_argument('--cross_modal_hidden_dim', type=int, default=256,
                       help='跨模态注意力隐藏维度')
    parser.add_argument('--cross_modal_num_heads', type=int, default=4,
                       help='跨模态注意力头数')
    parser.add_argument('--cross_modal_dropout', type=float, default=0.1,
                       help='跨模态注意力dropout率')

    # 对比学习
    parser.add_argument('--use_contrastive', type=str2bool, default=False,
                       help='是否使用对比学习')
    parser.add_argument('--contrastive_weight', type=float, default=0.1,
                       help='对比学习损失权重')
    parser.add_argument('--contrastive_temperature', type=float, default=0.1,
                       help='对比学习温度参数')

    # 分类任务
    parser.add_argument('--classification', type=str2bool, default=False,
                       help='是否为分类任务')
    parser.add_argument('--classification_threshold', type=float, default=0.5,
                       help='分类阈值')

    # 其他
    parser.add_argument('--output_dir', type=str, default='./results_local_cif_csv/',
                       help='输出目录')
    parser.add_argument('--random_seed', type=int, default=123,
                       help='随机种子')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='数据加载workers数量')

    return parser


# ==================== 主函数 ====================

def main():
    """主函数"""
    parser = get_parser()
    args = parser.parse_args()

    print("\n" + "="*80)
    print("🚀 本地CIF+CSV数据训练")
    print("="*80)
    print(f"数据集: {args.dataset}")
    print(f"属性: {args.property}")
    print(f"模型: {args.model.upper()}")
    print(f"输出目录: {args.output_dir}")
    print("="*80 + "\n")

    # 1. 确定数据路径（照抄 train_with_cross_modal_attention.py 的方式）
    if args.dataset == 'custom':
        # 自定义数据集：需要用户提供路径
        if args.cif_dir is None or args.csv_file is None:
            raise ValueError(
                "使用 custom 数据集时，必须提供 --cif_dir 和 --csv_file 参数\n"
                "示例: python train_local_cif_csv.py --dataset custom --cif_dir ./cifs/ --csv_file ./data.csv"
            )
        cif_dir = args.cif_dir
        csv_file = args.csv_file
    else:
        # 标准数据集（jarvis, mp, class, toy）：使用 get_dataset_paths
        cif_dir, csv_file = get_dataset_paths(args.root_dir, args.dataset, args.property)
        print(f"📂 使用标准数据集路径:")
        print(f"   CIF目录: {cif_dir}")
        print(f"   CSV文件: {csv_file}\n")

    # 2. 加载数据集
    dataset_array = load_dataset_from_cif_csv(
        cif_dir=cif_dir,
        csv_file=csv_file,
        dataset=args.dataset,
        property_name=args.property,
        target_column=args.target_column,
        text_column=args.text_column,
        id_column=args.id_column
    )

    # 2. 创建配置
    print("\n⚙️  创建训练配置...")
    config = create_config(args, dataset_array)

    # 保存配置
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(config.dict(), f, indent=2, default=str)
    print(f"✅ 配置已保存到: {os.path.join(args.output_dir, 'config.json')}")

    # 3. 准备数据加载器
    print("\n🔄 准备数据加载器...")

    # DenseGNN不需要线图
    use_line_graph = (args.model == 'alignn')

    train_loader, val_loader, test_loader, prepare_batch = get_train_val_loaders(
        dataset="user_data",
        dataset_array=dataset_array,
        target="target",
        atom_features="cgcnn",
        id_tag="jid",
        batch_size=args.batch_size,
        split_seed=args.random_seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        n_train=args.n_train,
        n_val=args.n_val,
        n_test=args.n_test,
        cutoff=8.0,
        max_neighbors=12,
        line_graph=use_line_graph,
        workers=args.num_workers,
        output_dir=args.output_dir,
        use_canonize=True,
        keep_data_order=False
    )

    print(f"\n✅ 数据加载器准备完成")
    print(f"   训练集: {len(train_loader.dataset)} 样本")
    print(f"   验证集: {len(val_loader.dataset)} 样本")
    print(f"   测试集: {len(test_loader.dataset)} 样本")

    # 4. 开始训练
    print(f"\n🎯 开始训练...")
    print(f"   模型: {args.model.upper()}")
    print(f"   中期融合: {'✓' if args.use_middle_fusion else '✗'}")
    print(f"   晚期融合: {'✓' if args.use_cross_modal else '✗'}")
    print(f"   对比学习: {'✓' if args.use_contrastive else '✗'}")
    print("="*80 + "\n")

    start_time = time.time()

    history = train_dgl(
        config=config,
        train_val_test_loaders=[train_loader, val_loader, test_loader, prepare_batch]
    )

    # 5. 训练完成
    elapsed_time = time.time() - start_time
    print("\n" + "="*80)
    print("🎉 训练完成！")
    print("="*80)
    print(f"总耗时: {elapsed_time/3600:.2f} 小时")
    print(f"结果保存在: {args.output_dir}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
