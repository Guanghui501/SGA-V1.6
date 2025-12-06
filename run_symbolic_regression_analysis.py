"""
符号回归分析示例脚本

演示如何使用interpretability_enhanced_v2模块中的符号回归功能
从训练好的神经网络模型中发现可解释的数学公式

使用方法:
    python run_symbolic_regression_analysis.py --model_path <path_to_model> --data_path <path_to_data>

依赖安装:
    1. 安装PySR: pip install pysr
    2. 安装Julia: https://github.com/MilesCranmer/PySR#installation
    3. 安装其他依赖: pip install torch dgl jarvis-tools transformers

作者: SGA-V1.6
日期: 2025-12-04
"""

import os
import sys
import argparse
import torch
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
from tqdm import tqdm

# 导入可解释性模块
from interpretability_enhanced_v2 import ComprehensiveExplainer


def main():
    parser = argparse.ArgumentParser(description='符号回归分析 - 从神经网络中发现数学公式')

    # 必需参数
    parser.add_argument('--model_path', type=str, required=True,
                        help='训练好的模型路径 (.pt 或 .pth 文件)')
    parser.add_argument('--data_path', type=str, required=True,
                        help='数据集路径 (JARVIS格式)')

    # 可选参数
    parser.add_argument('--save_dir', type=str, default='./symbolic_regression_results',
                        help='结果保存目录 (默认: ./symbolic_regression_results)')
    parser.add_argument('--max_samples', type=int, default=500,
                        help='最大样本数 (默认: 500, None表示使用所有样本)')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='批次大小 (默认: 16)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='设备 (cuda 或 cpu, 默认: cuda)')
    parser.add_argument('--property', type=str, default='formation_energy_peratom',
                        help='预测的材料属性 (默认: formation_energy_peratom)')

    # PySR参数
    parser.add_argument('--niterations', type=int, default=100,
                        help='符号回归迭代次数 (默认: 100)')
    parser.add_argument('--maxsize', type=int, default=20,
                        help='公式最大复杂度 (默认: 20)')

    args = parser.parse_args()

    # 检查设备
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("⚠️ CUDA不可用，切换到CPU")
        args.device = 'cpu'

    print("\n" + "="*80)
    print("🔬 符号回归可解释性分析")
    print("="*80)
    print(f"\n配置:")
    print(f"  模型路径: {args.model_path}")
    print(f"  数据路径: {args.data_path}")
    print(f"  保存目录: {args.save_dir}")
    print(f"  最大样本数: {args.max_samples}")
    print(f"  设备: {args.device}")
    print(f"  属性: {args.property}")

    # ==================== 1. 加载模型 ====================
    print("\n" + "="*80)
    print("📦 [1/4] 加载模型...")
    print("="*80)

    if not os.path.exists(args.model_path):
        print(f"❌ 模型文件不存在: {args.model_path}")
        sys.exit(1)

    try:
        # 加载模型检查点
        checkpoint = torch.load(args.model_path, map_location=args.device)

        # 如果checkpoint是字典，提取模型状态
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                model_state = checkpoint['model_state_dict']
            elif 'model' in checkpoint:
                model_state = checkpoint['model']
            else:
                model_state = checkpoint
        else:
            model_state = checkpoint

        print("   ✓ 模型检查点已加载")

        # 这里需要根据实际模型架构导入和初始化
        # 示例：假设使用ALIGNN模型
        print("   ⚠️ 注意: 需要根据实际模型架构初始化模型")
        print("   提示: 修改此脚本中的模型初始化部分")

        # TODO: 替换为实际的模型初始化代码
        # from models.alignn import ALIGNN
        # model = ALIGNN(...)
        # model.load_state_dict(model_state)
        # model = model.to(args.device)
        # model.eval()

        # 临时占位符
        model = None

        if model is None:
            print("   ❌ 请在脚本中配置正确的模型初始化代码")
            print("   提示: 查看 models/ 目录中的模型定义")
            sys.exit(1)

    except Exception as e:
        print(f"❌ 加载模型失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # ==================== 2. 加载数据 ====================
    print("\n" + "="*80)
    print("📊 [2/4] 加载数据...")
    print("="*80)

    if not os.path.exists(args.data_path):
        print(f"❌ 数据文件不存在: {args.data_path}")
        sys.exit(1)

    try:
        # TODO: 根据实际数据格式加载数据
        # from data import get_test_loader
        # test_loader = get_test_loader(
        #     data_path=args.data_path,
        #     batch_size=args.batch_size,
        #     property_name=args.property
        # )

        test_loader = None

        if test_loader is None:
            print("   ❌ 请在脚本中配置正确的数据加载代码")
            print("   提示: 查看 data.py 中的数据加载函数")
            sys.exit(1)

        print(f"   ✓ 数据集已加载")

    except Exception as e:
        print(f"❌ 加载数据失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # ==================== 3. 初始化解释器 ====================
    print("\n" + "="*80)
    print("🔧 [3/4] 初始化可解释性分析器...")
    print("="*80)

    try:
        # 初始化tokenizer (如果模型使用文本输入)
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
            print("   ✓ Tokenizer已加载")
        except Exception as e:
            print(f"   ⚠️ Tokenizer加载失败: {e}")
            tokenizer = None

        # 初始化解释器
        explainer = ComprehensiveExplainer(
            model=model,
            tokenizer=tokenizer,
            device=args.device
        )

        print("   ✓ 解释器初始化完成")

    except Exception as e:
        print(f"❌ 初始化解释器失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # ==================== 4. 运行符号回归分析 ====================
    print("\n" + "="*80)
    print("🧮 [4/4] 运行符号回归分析...")
    print("="*80)

    try:
        # 检查PySR是否安装
        try:
            import pysr
            print("   ✓ PySR已安装")
        except ImportError:
            print("   ❌ PySR未安装")
            print("   请运行以下命令安装:")
            print("   1. pip install pysr")
            print("   2. python -c 'import pysr; pysr.install()'")
            print("   或访问: https://github.com/MilesCranmer/PySR")
            sys.exit(1)

        # 运行符号回归
        model_sr, results = explainer.extract_symbolic_features(
            test_loader=test_loader,
            save_dir=args.save_dir,
            max_samples=args.max_samples
        )

        if model_sr is not None and results is not None:
            print("\n" + "="*80)
            print("✅ 符号回归分析成功完成!")
            print("="*80)

            print(f"\n📊 主要结果:")
            print(f"  样本数: {results['num_samples']}")
            print(f"  特征维度: {results['feature_dim']}")

            print(f"\n📈 性能指标:")
            metrics = results['metrics']
            print(f"  MAE:  {metrics['mae']:.4f}")
            print(f"  RMSE: {metrics['rmse']:.4f}")
            print(f"  R²:   {metrics['r2']:.4f}")

            print(f"\n🔬 与神经网络对比:")
            nn_comp = results['nn_comparison']
            print(f"  神经网络 MAE: {nn_comp['mae_nn']:.4f}")
            print(f"  MAE 比率: {nn_comp['mae_ratio']:.2%}")
            print(f"  R² 差距: {nn_comp['r2_diff']:+.4f}")

            if results['best_formula']:
                print(f"\n🎯 最佳符号公式:")
                print(f"  {results['best_formula']}")

            print(f"\n📁 结果已保存到: {args.save_dir}")
            print("   - symbolic_regression_formulas.txt  (所有公式)")
            print("   - symbolic_regression_results.json  (详细结果)")
            print("   - symbolic_regression_model.pkl     (PySR模型)")

        else:
            print("\n❌ 符号回归分析失败")
            sys.exit(1)

    except Exception as e:
        print(f"\n❌ 运行符号回归失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("\n" + "="*80)
    print("🎉 分析完成!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
