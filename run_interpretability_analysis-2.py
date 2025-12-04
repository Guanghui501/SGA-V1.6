#!/usr/bin/env python
"""
可解释性分析使用示例和集成指南

本文件展示如何将增强的可解释性模块集成到 SGA-Net 训练和推理流程中。

用法:
    python run_interpretability_analysis.py \
        --model_path ./output/best_val_model.pt \
        --data_dir ./dataset/jarvis/formation_energy_peratom \
        --output_dir ./interpretability_results \
        --num_samples 10
"""

import os
import sys
import argparse
import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

# 添加模块路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 导入可解释性模块
from interpretability_enhanced_v2 import (
    ComprehensiveExplainer,
    AtomImportanceAnalyzer,
    CrossModalInteractionAnalyzer,
    PhysicsCorrelationAnalyzer,
    UncertaintyEstimator,
)
from advanced_visualization import AdvancedVisualizer, quick_visualize


def load_model_and_config(model_path, device='cuda'):
    """
    加载训练好的模型和配置
    
    Args:
        model_path: 模型checkpoint路径
        device: 计算设备
        
    Returns:
        model: 加载好的模型
        config: 模型配置
    """
    print(f"\n📦 加载模型: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # 获取配置
    config = checkpoint.get('config', None)
    
    if config is None:
        print("⚠️  checkpoint中没有找到配置，使用默认配置")
        # 使用默认配置
        from models.alignn import ALIGNNConfig
        config = ALIGNNConfig(
            name="alignn",
            alignn_layers=4,
            gcn_layers=4,
            atom_input_features=92,
            hidden_features=256,
            output_features=1,
            use_cross_modal_attention=True,
            use_fine_grained_attention=True,
        )
        
    # 创建模型
    from models.alignn import ALIGNN
    model = ALIGNN(config)
    
    # 加载权重
    model.load_state_dict(checkpoint['model'])
    model = model.to(device)
    model.eval()
    
    print(f"✅ 模型加载成功")
    print(f"   - ALIGNN层数: {config.alignn_layers}")
    print(f"   - GCN层数: {config.gcn_layers}")
    print(f"   - 跨模态注意力: {config.use_cross_modal_attention}")
    print(f"   - 细粒度注意力: {getattr(config, 'use_fine_grained_attention', False)}")
    
    return model, config


def load_test_samples(data_dir, num_samples=10):
    """
    加载测试样本
    
    Args:
        data_dir: 数据目录
        num_samples: 样本数量
        
    Returns:
        samples: [(g, lg, text, target, atoms_object), ...]
    """
    import csv
    from jarvis.core.atoms import Atoms
    from graphs import Graph
    
    print(f"\n📂 加载测试数据: {data_dir}")
    
    cif_dir = os.path.join(data_dir, 'cif')
    desc_file = os.path.join(data_dir, 'description.csv')
    
    # 读取描述文件
    with open(desc_file, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        data = list(reader)[:num_samples]
        
    samples = []
    
    for row in tqdm(data, desc="加载样本"):
        try:
            jid = row[0]
            target = float(row[2])
            text = row[3]
            
            # 读取CIF
            cif_path = os.path.join(cif_dir, f'{jid}.cif')
            atoms = Atoms.from_cif(cif_path)
            
            # 构建图
            g, lg = Graph.atom_dgl_multigraph(
                atoms=atoms,
                cutoff=8.0,
                max_neighbors=12,
                atom_features="cgcnn",
                compute_line_graph=True,
                use_canonize=True
            )
            
            samples.append({
                'jid': jid,
                'g': g,
                'lg': lg,
                'text': [text],
                'target': target,
                'atoms': atoms
            })
            
        except Exception as e:
            print(f"⚠️  跳过样本 {row[0]}: {e}")
            
    print(f"✅ 加载了 {len(samples)} 个样本")
    return samples


def run_single_sample_analysis(explainer, sample, output_dir, visualizer=None):
    """
    对单个样本运行完整分析
    
    Args:
        explainer: ComprehensiveExplainer实例
        sample: 样本字典
        output_dir: 输出目录
        visualizer: AdvancedVisualizer实例（可选）
    """
    sample_dir = Path(output_dir) / sample['jid']
    sample_dir.mkdir(parents=True, exist_ok=True)
    
    # 运行综合分析
    explanation = explainer.explain_prediction(
        g=sample['g'],
        lg=sample['lg'],
        text=sample['text'],
        atoms_object=sample['atoms'],
        true_value=sample['target'],
        save_dir=sample_dir,
        sample_id=sample['jid']
    )
    
    # 高级可视化
    if visualizer:
        visualizer.create_comprehensive_report(
            explanation, sample['atoms'],
            save_path=sample_dir / 'comprehensive_report.png'
        )
        
        # 元素重要性周期表
        importance = explanation.get('atom_importance_integrated_gradients', [])
        if len(importance) > 0:
            from collections import defaultdict
            elements = list(sample['atoms'].elements)
            elem_imp = defaultdict(list)
            for elem, imp in zip(elements, importance):
                elem_imp[elem].append(imp)
            elem_mean = {k: np.mean(v) for k, v in elem_imp.items()}
            
            visualizer.plot_periodic_table_importance(
                elem_mean,
                save_path=sample_dir / 'periodic_table_importance.png'
            )
            
        # HTML报告
        image_paths = {
            'Atom Importance': 'atom_importance.png',
            'Modal Contribution': 'modal_contribution.png',
            'Physics Correlation': 'physics_correlation.png',
        }
        # 只包含存在的图像
        image_paths = {k: str(sample_dir / v) for k, v in image_paths.items() 
                      if (sample_dir / v).exists()}
        
        visualizer.generate_html_report(
            explanation, sample['atoms'],
            image_paths,
            save_path=str(sample_dir / 'report.html')
        )
        
    return explanation


def run_batch_analysis(explainer, samples, output_dir, visualizer=None):
    """
    批量分析
    
    Args:
        explainer: ComprehensiveExplainer实例
        samples: 样本列表
        output_dir: 输出目录
        visualizer: AdvancedVisualizer实例
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_explanations = []
    
    # 统计数据
    all_errors = []
    all_uncertainties = []
    all_graph_contrib = []
    all_text_contrib = []
    element_importance_agg = {}
    
    for sample in tqdm(samples, desc="批量分析"):
        explanation = run_single_sample_analysis(
            explainer, sample, output_dir, visualizer
        )
        all_explanations.append(explanation)
        
        # 收集统计数据
        if explanation.get('error') is not None:
            all_errors.append(explanation['error'])
        if explanation.get('uncertainty', {}).get('std') is not None:
            all_uncertainties.append(explanation['uncertainty']['std'])
            
        mc = explanation.get('modal_contribution', {})
        if mc.get('graph_contribution') is not None:
            all_graph_contrib.append(mc['graph_contribution'])
        if mc.get('text_contribution') is not None:
            all_text_contrib.append(mc['text_contribution'])
            
        # 聚合元素重要性
        sa = explanation.get('structure_analysis', {})
        elem_imp = sa.get('element_importance', {})
        for elem, imp in elem_imp.items():
            if elem not in element_importance_agg:
                element_importance_agg[elem] = []
            element_importance_agg[elem].append(imp)
            
    # 生成批量统计报告
    summary = {
        'num_samples': len(samples),
        'error_statistics': {
            'mean': float(np.mean(all_errors)) if all_errors else None,
            'std': float(np.std(all_errors)) if all_errors else None,
            'min': float(np.min(all_errors)) if all_errors else None,
            'max': float(np.max(all_errors)) if all_errors else None,
        },
        'uncertainty_statistics': {
            'mean': float(np.mean(all_uncertainties)) if all_uncertainties else None,
            'std': float(np.std(all_uncertainties)) if all_uncertainties else None,
        },
        'modal_contribution': {
            'graph_mean': float(np.mean(all_graph_contrib)) if all_graph_contrib else None,
            'text_mean': float(np.mean(all_text_contrib)) if all_text_contrib else None,
        },
        'element_importance': {
            elem: {
                'mean': float(np.mean(imps)),
                'std': float(np.std(imps)),
                'count': len(imps)
            }
            for elem, imps in element_importance_agg.items()
        }
    }
    
    # 保存汇总
    with open(output_dir / 'batch_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
        
    # 生成汇总可视化
    if visualizer and element_importance_agg:
        elem_mean = {k: np.mean(v) for k, v in element_importance_agg.items()}
        visualizer.plot_periodic_table_importance(
            elem_mean,
            save_path=output_dir / 'batch_periodic_table_importance.png'
        )
        
    print(f"\n{'='*80}")
    print("📊 批量分析统计")
    print(f"{'='*80}")
    print(f"样本数: {len(samples)}")
    if all_errors:
        print(f"平均误差: {np.mean(all_errors):.4f} ± {np.std(all_errors):.4f}")
    if all_uncertainties:
        print(f"平均不确定性: {np.mean(all_uncertainties):.4f}")
    if all_graph_contrib:
        print(f"图模态平均贡献: {np.mean(all_graph_contrib):.1%}")
        print(f"文本模态平均贡献: {np.mean(all_text_contrib):.1%}")
    print(f"\n结果保存在: {output_dir}")
    print(f"{'='*80}\n")
    
    return all_explanations, summary


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='运行可解释性分析')
    parser.add_argument('--model_path', type=str, required=True,
                        help='模型checkpoint路径')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='数据目录（包含cif/和description.csv）')
    parser.add_argument('--output_dir', type=str, default='./interpretability_results',
                        help='输出目录')
    parser.add_argument('--num_samples', type=int, default=10,
                        help='分析样本数')
    parser.add_argument('--device', type=str, default='cuda',
                        help='计算设备')
    parser.add_argument('--style', type=str, default='publication',
                        choices=['publication', 'presentation', 'report'],
                        help='可视化风格')
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("🔬 SGA-Net 可解释性分析")
    print("="*80)
    
    # 设置设备
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"\n使用设备: {device}")
    
    # 加载模型
    model, config = load_model_and_config(args.model_path, device)
    
    # 加载数据
    samples = load_test_samples(args.data_dir, args.num_samples)
    
    if not samples:
        print("❌ 没有可用的样本!")
        return
        
    # 初始化分析器
    print("\n🔧 初始化分析器...")
    
    # 尝试加载tokenizer（可选）
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained('m3rg-iitd/matscibert')
    except:
        tokenizer = None
        print("⚠️  未能加载tokenizer，部分功能可能受限")
        
    explainer = ComprehensiveExplainer(
        model=model,
        tokenizer=tokenizer,
        device=device
    )
    
    visualizer = AdvancedVisualizer(style=args.style)
    
    # 运行分析
    print(f"\n🚀 开始分析 {len(samples)} 个样本...")
    
    explanations, summary = run_batch_analysis(
        explainer, samples, args.output_dir, visualizer
    )
    
    print("\n✅ 分析完成!")


# ==================== 快捷使用接口 ====================

class QuickAnalyzer:
    """
    快捷分析接口 - 简化使用流程
    
    用法:
        analyzer = QuickAnalyzer(model)
        result = analyzer.analyze(g, lg, text, atoms)
        analyzer.visualize(result, atoms, './output')
    """
    
    def __init__(self, model, device='cuda'):
        """
        Args:
            model: 训练好的模型
            device: 计算设备
        """
        self.model = model
        self.device = device
        self.model.eval()
        self.model.to(device)
        
        # 初始化分析器
        self.explainer = ComprehensiveExplainer(model, device=device)
        self.visualizer = AdvancedVisualizer()
        
    def analyze(self, g, lg, text, atoms_object, true_value=None):
        """
        快速分析
        
        Args:
            g: DGL graph
            lg: Line graph
            text: 文本列表
            atoms_object: JARVIS Atoms对象
            true_value: 真实值（可选）
            
        Returns:
            explanation: 解释字典
        """
        return self.explainer.explain_prediction(
            g, lg, text, atoms_object,
            true_value=true_value,
            save_dir=None  # 不保存，只返回
        )
        
    def visualize(self, explanation, atoms_object, save_dir, sample_id='sample'):
        """
        可视化分析结果
        
        Args:
            explanation: 解释字典
            atoms_object: Atoms对象
            save_dir: 保存目录
            sample_id: 样本ID
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 综合报告
        self.visualizer.create_comprehensive_report(
            explanation, atoms_object,
            save_path=save_dir / f'{sample_id}_report.png'
        )
        
        # 周期表
        importance = explanation.get('atom_importance_integrated_gradients', [])
        if len(importance) > 0:
            from collections import defaultdict
            elements = list(atoms_object.elements)
            elem_imp = defaultdict(list)
            for elem, imp in zip(elements, importance):
                elem_imp[elem].append(imp)
            elem_mean = {k: np.mean(v) for k, v in elem_imp.items()}
            
            self.visualizer.plot_periodic_table_importance(
                elem_mean,
                save_path=save_dir / f'{sample_id}_periodic_table.png'
            )
            
        print(f"✅ 可视化结果已保存到: {save_dir}")
        
    def get_atom_importance(self, g, lg, text, method='integrated_gradients'):
        """
        快速获取原子重要性
        
        Args:
            g, lg, text: 模型输入
            method: 'gradient' 或 'integrated_gradients'
            
        Returns:
            importance: numpy数组
        """
        atom_analyzer = AtomImportanceAnalyzer(self.model, self.device)
        
        if method == 'gradient':
            importance, _ = atom_analyzer.gradient_importance(g, lg, text)
        else:
            importance, _ = atom_analyzer.integrated_gradients(g, lg, text)
            
        return importance
        
    def get_uncertainty(self, g, lg, text, n_samples=30):
        """
        快速获取预测不确定性
        
        Args:
            g, lg, text: 模型输入
            n_samples: MC采样数
            
        Returns:
            mean, std: 预测均值和标准差
        """
        uncertainty_estimator = UncertaintyEstimator(self.model, self.device)
        mean, std, _ = uncertainty_estimator.mc_dropout_uncertainty(g, lg, text, n_samples)
        return mean, std


# ==================== 演示函数 ====================

def demo():
    """演示用法"""
    print("""
╔════════════════════════════════════════════════════════════════════════════════╗
║                    SGA-Net 增强可解释性分析模块使用指南                        ║
╠════════════════════════════════════════════════════════════════════════════════╣
║                                                                                ║
║  1. 命令行使用:                                                                ║
║     python run_interpretability_analysis.py \\                                 ║
║         --model_path ./output/best_val_model.pt \\                             ║
║         --data_dir ./dataset/jarvis/formation_energy_peratom \\                ║
║         --output_dir ./interpretability_results \\                             ║
║         --num_samples 10                                                       ║
║                                                                                ║
║  2. Python代码使用:                                                            ║
║                                                                                ║
║     # 方式一：使用快捷分析器                                                    ║
║     from run_interpretability_analysis import QuickAnalyzer                    ║
║                                                                                ║
║     analyzer = QuickAnalyzer(model)                                            ║
║     explanation = analyzer.analyze(g, lg, text, atoms)                         ║
║     analyzer.visualize(explanation, atoms, './output')                         ║
║                                                                                ║
║     # 方式二：使用综合解释器                                                    ║
║     from interpretability_enhanced_v2 import ComprehensiveExplainer            ║
║                                                                                ║
║     explainer = ComprehensiveExplainer(model, tokenizer, device='cuda')        ║
║     explanation = explainer.explain_prediction(                                ║
║         g, lg, text, atoms,                                                    ║
║         true_value=1.5,                                                        ║
║         save_dir='./results',                                                  ║
║         sample_id='sample_001'                                                 ║
║     )                                                                          ║
║                                                                                ║
║     # 方式三：单独使用各分析器                                                  ║
║     from interpretability_enhanced_v2 import (                                 ║
║         AtomImportanceAnalyzer,                                                ║
║         CrossModalInteractionAnalyzer,                                         ║
║         PhysicsCorrelationAnalyzer,                                            ║
║         UncertaintyEstimator                                                   ║
║     )                                                                          ║
║                                                                                ║
║     # 原子重要性                                                                ║
║     atom_analyzer = AtomImportanceAnalyzer(model)                              ║
║     importance, gradients = atom_analyzer.integrated_gradients(g, lg, text)    ║
║                                                                                ║
║     # 跨模态分析                                                                ║
║     cross_modal = CrossModalInteractionAnalyzer(model)                         ║
║     contributions = cross_modal.analyze_modal_contribution(g, lg, text)        ║
║                                                                                ║
║     # 物理关联                                                                  ║
║     physics = PhysicsCorrelationAnalyzer()                                     ║
║     correlations = physics.correlate_importance_with_physics(elements, imp)    ║
║                                                                                ║
║     # 不确定性估计                                                              ║
║     uncertainty = UncertaintyEstimator(model)                                  ║
║     mean, std, samples = uncertainty.mc_dropout_uncertainty(g, lg, text)       ║
║                                                                                ║
║  3. 输出说明:                                                                  ║
║     - *_explanation.json: 完整解释数据                                         ║
║     - *_summary.txt: 文本摘要                                                  ║
║     - *_comprehensive_report.png: 综合报告图                                   ║
║     - *_periodic_table.png: 元素周期表重要性图                                 ║
║     - *_report.html: 交互式HTML报告                                            ║
║                                                                                ║
╚════════════════════════════════════════════════════════════════════════════════╝
    """)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main()
    else:
        demo()
