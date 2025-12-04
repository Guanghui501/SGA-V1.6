"""
高级可解释性可视化模块

专业级可视化功能：
1. 多面板综合报告图
2. 交互式HTML报告
3. 动态注意力流图
4. 元素周期表重要性映射
5. 3D分子结构+重要性叠加
6. 时序训练分析图

"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import json
import warnings
warnings.filterwarnings('ignore')

# 设置绘图风格
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 9


class AdvancedVisualizer:
    """高级可视化类"""
    
    # 自定义颜色方案
    COLORS = {
        'primary': '#3498db',
        'secondary': '#e74c3c',
        'success': '#2ecc71',
        'warning': '#f39c12',
        'info': '#9b59b6',
        'dark': '#2c3e50',
        'light': '#ecf0f1',
        'graph': '#3498db',
        'text': '#e74c3c',
        'physics': '#2ecc71',
    }
    
    # 元素颜色映射（CPK配色方案）
    ELEMENT_COLORS = {
        'H': '#FFFFFF', 'C': '#909090', 'N': '#3050F8', 'O': '#FF0D0D',
        'F': '#90E050', 'Cl': '#1FF01F', 'Br': '#A62929', 'I': '#940094',
        'S': '#FFFF30', 'P': '#FF8000', 'B': '#FFB5B5', 'Si': '#F0C8A0',
        'Fe': '#E06633', 'Cu': '#C88033', 'Zn': '#7D80B0', 'Ag': '#C0C0C0',
        'Au': '#FFD123', 'Ti': '#BFC2C7', 'Ni': '#50D050', 'Co': '#F090A0',
        'Mn': '#9C7AC7', 'Cr': '#8A99C7', 'V': '#A6A6AB', 'Ca': '#3DFF00',
        'K': '#8F40D4', 'Na': '#AB5CF2', 'Mg': '#8AFF00', 'Al': '#BFA6A6',
        'Li': '#CC80FF', 'Be': '#C2FF00', 'Ba': '#00C900', 'Sr': '#00FF00',
        'Pb': '#575961', 'Sn': '#668080', 'Ga': '#C28F8F', 'Ge': '#668F8F',
        'As': '#BD80E3', 'Se': '#FFA100', 'Te': '#D47A00', 'Sb': '#9E63B5',
        'Bi': '#9E4FB5', 'La': '#70D4FF', 'Y': '#94FFFF', 'Zr': '#94E0E0',
        'Nb': '#73C2C9', 'Mo': '#54B5B5', 'default': '#FF1493'
    }
    
    def __init__(self, style='publication'):
        """
        Args:
            style: 'publication' (适合论文), 'presentation' (适合PPT), 'report' (适合报告)
        """
        self.style = style
        self._setup_style()
        
    def _setup_style(self):
        """设置绘图风格"""
        if self.style == 'publication':
            plt.rcParams.update({
                'font.family': 'serif',
                'font.size': 10,
                'axes.linewidth': 1.2,
                'xtick.major.width': 1.0,
                'ytick.major.width': 1.0,
            })
        elif self.style == 'presentation':
            plt.rcParams.update({
                'font.family': 'sans-serif',
                'font.size': 14,
                'axes.linewidth': 2.0,
                'lines.linewidth': 2.5,
            })
        elif self.style == 'report':
            plt.rcParams.update({
                'font.family': 'sans-serif',
                'font.size': 11,
            })
            
    def create_comprehensive_report(self, explanation: Dict, atoms_object,
                                    save_path: Optional[str] = None,
                                    figsize: Tuple = (20, 16)):
        """
        创建综合可解释性报告图
        
        Args:
            explanation: 解释字典（来自ComprehensiveExplainer）
            atoms_object: JARVIS Atoms对象
            save_path: 保存路径
            figsize: 图像大小
        """
        fig = plt.figure(figsize=figsize)
        
        # 创建网格布局
        gs = gridspec.GridSpec(4, 4, figure=fig, hspace=0.35, wspace=0.35)
        
        # ==================== 第一行：预测和不确定性 ====================
        
        # 1.1 预测结果面板
        ax_pred = fig.add_subplot(gs[0, 0])
        self._plot_prediction_panel(ax_pred, explanation)
        
        # 1.2 不确定性分布
        ax_unc = fig.add_subplot(gs[0, 1])
        self._plot_uncertainty_panel(ax_unc, explanation)
        
        # 1.3 模态贡献
        ax_modal = fig.add_subplot(gs[0, 2])
        self._plot_modal_contribution_panel(ax_modal, explanation)
        
        # 1.4 物理相关性
        ax_phys = fig.add_subplot(gs[0, 3])
        self._plot_physics_correlation_panel(ax_phys, explanation)
        
        # ==================== 第二行：原子重要性空间分布 ====================
        
        # 2.1-2.3 三个投影
        coords = atoms_object.cart_coords
        elements = list(atoms_object.elements)
        importance = np.array(explanation.get('atom_importance_integrated_gradients', 
                                              explanation.get('atom_importance_gradient', [])))
        
        if len(importance) > 0:
            projections = [(0, 1, 'X-Y'), (0, 2, 'X-Z'), (1, 2, 'Y-Z')]
            for i, (xi, yi, title) in enumerate(projections):
                ax = fig.add_subplot(gs[1, i])
                self._plot_spatial_projection(ax, coords, elements, importance, xi, yi, title)
                
        # 2.4 元素重要性柱状图
        ax_elem = fig.add_subplot(gs[1, 3])
        self._plot_element_importance(ax_elem, elements, importance)
        
        # ==================== 第三行：注意力分析 ====================
        
        # 3.1-3.2 注意力热图
        ax_attn = fig.add_subplot(gs[2, :2])
        self._plot_attention_heatmap_detailed(ax_attn, explanation, elements)
        
        # 3.3-3.4 注意力流图
        ax_flow = fig.add_subplot(gs[2, 2:])
        self._plot_attention_flow(ax_flow, explanation)
        
        # ==================== 第四行：特征分析 ====================
        
        # 4.1 重要性分布直方图
        ax_hist = fig.add_subplot(gs[3, 0])
        self._plot_importance_distribution(ax_hist, importance)
        
        # 4.2 元素-重要性关系
        ax_elem_rel = fig.add_subplot(gs[3, 1])
        self._plot_element_importance_relationship(ax_elem_rel, elements, importance, explanation)
        
        # 4.3-4.4 文本摘要
        ax_summary = fig.add_subplot(gs[3, 2:])
        self._plot_text_summary(ax_summary, explanation)
        
        # 添加总标题
        fig.suptitle(
            f"Comprehensive Interpretability Report - Sample {explanation.get('sample_id', 'Unknown')}",
            fontsize=16, fontweight='bold', y=0.98
        )
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"✅ 综合报告已保存: {save_path}")
            
        plt.close()
        
    def _plot_prediction_panel(self, ax, explanation):
        """预测结果面板"""
        pred = explanation.get('prediction', 0)
        true_val = explanation.get('true_value')
        
        # 创建仪表盘效果
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-0.5, 1.5)
        
        # 预测值显示
        ax.text(0, 1.0, 'Prediction', ha='center', va='center', 
               fontsize=12, fontweight='bold', color=self.COLORS['dark'])
        ax.text(0, 0.6, f'{pred:.4f}', ha='center', va='center',
               fontsize=24, fontweight='bold', color=self.COLORS['primary'])
        
        if true_val is not None:
            error = explanation.get('error', abs(pred - true_val))
            ax.text(0, 0.2, f'True: {true_val:.4f}', ha='center', va='center',
                   fontsize=11, color=self.COLORS['dark'])
            
            # 误差指示器
            error_color = self.COLORS['success'] if error < 0.1 else \
                         (self.COLORS['warning'] if error < 0.5 else self.COLORS['secondary'])
            ax.text(0, -0.1, f'Error: {error:.4f}', ha='center', va='center',
                   fontsize=11, fontweight='bold', color=error_color)
                   
        ax.axis('off')
        ax.set_title('Prediction', fontsize=11, fontweight='bold')
        
    def _plot_uncertainty_panel(self, ax, explanation):
        """不确定性面板"""
        unc = explanation.get('uncertainty', {})
        
        if not unc:
            ax.text(0.5, 0.5, 'No uncertainty\ndata available', 
                   ha='center', va='center', fontsize=10)
            ax.axis('off')
            ax.set_title('Uncertainty', fontsize=11, fontweight='bold')
            return
            
        mean = unc.get('mean', 0)
        std = unc.get('std', 0)
        
        # 绘制正态分布
        x = np.linspace(mean - 4*std, mean + 4*std, 100)
        y = np.exp(-0.5 * ((x - mean) / std) ** 2) / (std * np.sqrt(2 * np.pi))
        
        ax.fill_between(x, y, alpha=0.3, color=self.COLORS['primary'])
        ax.plot(x, y, color=self.COLORS['primary'], linewidth=2)
        
        # 标注均值和置信区间
        ax.axvline(mean, color=self.COLORS['secondary'], linestyle='--', linewidth=2, label='Mean')
        ax.axvline(mean - 1.96*std, color=self.COLORS['warning'], linestyle=':', linewidth=1.5)
        ax.axvline(mean + 1.96*std, color=self.COLORS['warning'], linestyle=':', linewidth=1.5)
        
        ax.set_xlabel('Predicted Value')
        ax.set_ylabel('Density')
        ax.set_title(f'Uncertainty (σ={std:.4f})', fontsize=11, fontweight='bold')
        ax.legend(loc='upper right', fontsize=8)
        
    def _plot_modal_contribution_panel(self, ax, explanation):
        """模态贡献面板"""
        mc = explanation.get('modal_contribution', {})
        
        graph_contrib = mc.get('graph_contribution', 0.5)
        text_contrib = mc.get('text_contribution', 0.5)
        
        # 饼图
        sizes = [graph_contrib, text_contrib]
        labels = ['Graph', 'Text']
        colors = [self.COLORS['graph'], self.COLORS['text']]
        explode = (0.05, 0.05)
        
        wedges, texts, autotexts = ax.pie(
            sizes, explode=explode, labels=labels, colors=colors,
            autopct='%1.1f%%', startangle=90, 
            textprops={'fontsize': 10, 'fontweight': 'bold'}
        )
        
        # 添加对齐度信息
        alignment = mc.get('feature_alignment', None)
        if alignment is not None:
            ax.text(0, -1.4, f'Alignment: {alignment:.3f}', 
                   ha='center', fontsize=9, style='italic')
                   
        ax.set_title('Modal Contribution', fontsize=11, fontweight='bold')
        
    def _plot_physics_correlation_panel(self, ax, explanation):
        """物理相关性面板"""
        correlations = explanation.get('physics_correlations', {})
        
        if not correlations:
            ax.text(0.5, 0.5, 'No physics\ncorrelation data', 
                   ha='center', va='center', fontsize=10)
            ax.axis('off')
            ax.set_title('Physics Correlation', fontsize=11, fontweight='bold')
            return
            
        # 排序
        sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
        props = [x[0] for x in sorted_corr]
        values = [x[1] for x in sorted_corr]
        
        colors = [self.COLORS['secondary'] if v < 0 else self.COLORS['success'] for v in values]
        
        y_pos = np.arange(len(props))
        bars = ax.barh(y_pos, values, color=colors, edgecolor='black', linewidth=0.5)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(props, fontsize=9)
        ax.axvline(0, color='black', linewidth=0.8)
        ax.set_xlim(-1, 1)
        ax.set_xlabel('Correlation')
        ax.set_title('Physics Correlation', fontsize=11, fontweight='bold')
        
    def _plot_spatial_projection(self, ax, coords, elements, importance, xi, yi, title):
        """空间投影图"""
        if len(importance) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
            ax.set_title(title)
            return
            
        # 归一化重要性
        imp_norm = (importance - importance.min()) / (importance.max() - importance.min() + 1e-8)
        
        # 绘制原子
        scatter = ax.scatter(
            coords[:, xi], coords[:, yi],
            c=imp_norm, cmap='YlOrRd', s=200, alpha=0.8,
            edgecolors='black', linewidth=1
        )
        
        # 标注元素
        for i, (coord, elem) in enumerate(zip(coords, elements)):
            ax.annotate(
                elem, (coord[xi], coord[yi]),
                fontsize=8, fontweight='bold',
                ha='center', va='center',
                color='black' if imp_norm[i] < 0.5 else 'white'
            )
            
        ax.set_xlabel(['X', 'Y', 'Z'][xi] + ' (Å)', fontsize=10)
        ax.set_ylabel(['X', 'Y', 'Z'][yi] + ' (Å)', fontsize=10)
        ax.set_title(title, fontsize=11, fontweight='bold')
        
    def _plot_element_importance(self, ax, elements, importance):
        """元素重要性柱状图"""
        if len(importance) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
            ax.set_title('Element Importance')
            return
            
        # 按元素类型聚合
        from collections import defaultdict
        elem_imp = defaultdict(list)
        for elem, imp in zip(elements, importance):
            elem_imp[elem].append(imp)
            
        # 计算均值
        elem_mean = {k: np.mean(v) for k, v in elem_imp.items()}
        elem_std = {k: np.std(v) for k, v in elem_imp.items()}
        
        # 排序
        sorted_elems = sorted(elem_mean.keys(), key=lambda x: elem_mean[x], reverse=True)
        
        x_pos = np.arange(len(sorted_elems))
        means = [elem_mean[e] for e in sorted_elems]
        stds = [elem_std[e] for e in sorted_elems]
        colors = [self.ELEMENT_COLORS.get(e, self.ELEMENT_COLORS['default']) for e in sorted_elems]
        
        bars = ax.bar(x_pos, means, yerr=stds, color=colors, 
                     edgecolor='black', linewidth=1, capsize=3)
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(sorted_elems, fontsize=10, fontweight='bold')
        ax.set_ylabel('Importance', fontsize=10)
        ax.set_title('Element Importance', fontsize=11, fontweight='bold')
        
    def _plot_attention_heatmap_detailed(self, ax, explanation, elements):
        """详细注意力热图"""
        # 尝试获取细粒度注意力
        atom_focus = explanation.get('atom_text_focus', None)
        
        if atom_focus is None:
            ax.text(0.5, 0.5, 'No fine-grained attention data available\n'
                   '(Enable use_fine_grained_attention=True)',
                   ha='center', va='center', fontsize=10)
            ax.axis('off')
            ax.set_title('Atom-Text Attention', fontsize=11, fontweight='bold')
            return
            
        # 创建简化的注意力表示
        num_atoms = len(elements)
        num_tokens = max(atom_focus) + 1 if atom_focus else 10
        
        # 创建热图数据
        attention_matrix = np.zeros((min(num_atoms, 15), min(num_tokens, 20)))
        for i, focus in enumerate(atom_focus[:15]):
            if focus < attention_matrix.shape[1]:
                attention_matrix[i, focus] = 1.0
                
        sns.heatmap(
            attention_matrix, cmap='YlOrRd', ax=ax,
            xticklabels=[f'T{i}' for i in range(attention_matrix.shape[1])],
            yticklabels=[f'{elements[i]}_{i}' for i in range(min(num_atoms, 15))],
            cbar_kws={'label': 'Attention', 'shrink': 0.8}
        )
        
        ax.set_xlabel('Text Tokens', fontsize=10)
        ax.set_ylabel('Atoms', fontsize=10)
        ax.set_title('Atom-Text Attention Focus', fontsize=11, fontweight='bold')
        
    def _plot_attention_flow(self, ax, explanation):
        """注意力流图"""
        flow = explanation.get('attention_flow', {})
        
        if not flow:
            ax.text(0.5, 0.5, 'No attention flow data', ha='center', va='center')
            ax.axis('off')
            ax.set_title('Information Flow', fontsize=11, fontweight='bold')
            return
            
        # 创建简化的流图
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 6)
        
        # 绘制节点
        nodes = [
            (2, 4, 'Graph\nEncoder', self.COLORS['graph']),
            (2, 2, 'Text\nEncoder', self.COLORS['text']),
            (5, 3, 'Cross-Modal\nAttention', self.COLORS['info']),
            (8, 3, 'Prediction', self.COLORS['success']),
        ]
        
        for x, y, label, color in nodes:
            circle = Circle((x, y), 0.8, facecolor=color, edgecolor='black', linewidth=2, alpha=0.7)
            ax.add_patch(circle)
            ax.text(x, y, label, ha='center', va='center', fontsize=8, fontweight='bold')
            
        # 绘制箭头
        arrows = [
            (2.8, 4, 4.2, 3.3),  # Graph -> Cross-Modal
            (2.8, 2, 4.2, 2.7),  # Text -> Cross-Modal
            (5.8, 3, 7.2, 3),    # Cross-Modal -> Prediction
        ]
        
        for x1, y1, x2, y2 in arrows:
            ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                       arrowprops=dict(arrowstyle='->', color='black', lw=2))
                       
        # 添加流量标注
        if 'graph_change_from_text' in flow:
            ax.text(3.5, 3.8, f'Δ={flow["graph_change_from_text"]:.2f}', fontsize=8)
        if 'text_change_from_graph' in flow:
            ax.text(3.5, 2.2, f'Δ={flow["text_change_from_graph"]:.2f}', fontsize=8)
            
        ax.axis('off')
        ax.set_title('Information Flow', fontsize=11, fontweight='bold')
        
    def _plot_importance_distribution(self, ax, importance):
        """重要性分布直方图"""
        if len(importance) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
            ax.set_title('Importance Distribution')
            return
            
        ax.hist(importance, bins=20, color=self.COLORS['primary'], 
               edgecolor='black', alpha=0.7)
        ax.axvline(np.mean(importance), color=self.COLORS['secondary'], 
                  linestyle='--', linewidth=2, label=f'Mean: {np.mean(importance):.3f}')
        ax.axvline(np.median(importance), color=self.COLORS['success'],
                  linestyle=':', linewidth=2, label=f'Median: {np.median(importance):.3f}')
        
        ax.set_xlabel('Importance Score', fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.set_title('Importance Distribution', fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        
    def _plot_element_importance_relationship(self, ax, elements, importance, explanation):
        """元素-重要性关系图"""
        if len(importance) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
            ax.set_title('Element-Importance')
            return
            
        # 获取物理数据
        physics = explanation.get('physics_correlations', {})
        
        # 简单展示：元素与重要性的箱线图
        from collections import defaultdict
        elem_imp = defaultdict(list)
        for elem, imp in zip(elements, importance):
            elem_imp[elem].append(imp)
            
        unique_elems = list(elem_imp.keys())
        data = [elem_imp[e] for e in unique_elems]
        
        bp = ax.boxplot(data, labels=unique_elems, patch_artist=True)
        
        for patch, elem in zip(bp['boxes'], unique_elems):
            color = self.ELEMENT_COLORS.get(elem, self.ELEMENT_COLORS['default'])
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
            
        ax.set_xlabel('Element', fontsize=10)
        ax.set_ylabel('Importance', fontsize=10)
        ax.set_title('Element-wise Importance', fontsize=11, fontweight='bold')
        
    def _plot_text_summary(self, ax, explanation):
        """文本摘要面板"""
        ax.axis('off')
        
        # 构建摘要文本
        lines = []
        lines.append("═" * 50)
        lines.append("ANALYSIS SUMMARY")
        lines.append("═" * 50)
        
        # 预测信息
        pred = explanation.get('prediction', 'N/A')
        lines.append(f"Prediction: {pred:.4f}" if isinstance(pred, float) else f"Prediction: {pred}")
        
        true_val = explanation.get('true_value')
        if true_val is not None:
            lines.append(f"True Value: {true_val:.4f}")
            lines.append(f"Error: {explanation.get('error', 'N/A'):.4f}")
            
        # 不确定性
        unc = explanation.get('uncertainty', {})
        if unc:
            lines.append(f"\nUncertainty (σ): {unc.get('std', 'N/A'):.4f}")
            
        # 最重要元素
        struct = explanation.get('structure_analysis', {})
        if struct.get('most_important_element'):
            elem, imp = struct['most_important_element']
            lines.append(f"\nMost Important Element: {elem} ({imp:.4f})")
            
        # 模态贡献
        mc = explanation.get('modal_contribution', {})
        if mc:
            lines.append(f"\nGraph Contribution: {mc.get('graph_contribution', 0):.1%}")
            lines.append(f"Text Contribution: {mc.get('text_contribution', 0):.1%}")
            
        lines.append("\n" + "═" * 50)
        
        summary_text = '\n'.join(lines)
        
        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
               fontsize=9, fontfamily='monospace',
               verticalalignment='top', horizontalalignment='left',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
               
        ax.set_title('Summary', fontsize=11, fontweight='bold')
        
    def plot_periodic_table_importance(self, element_importance: Dict[str, float],
                                        save_path: Optional[str] = None):
        """
        在元素周期表上显示元素重要性
        
        Args:
            element_importance: {元素符号: 重要性} 字典
            save_path: 保存路径
        """
        fig, ax = plt.subplots(figsize=(18, 10))
        
        # 元素周期表位置 (row, col)
        positions = {
            'H': (0, 0), 'He': (0, 17),
            'Li': (1, 0), 'Be': (1, 1), 'B': (1, 12), 'C': (1, 13), 'N': (1, 14), 'O': (1, 15), 'F': (1, 16), 'Ne': (1, 17),
            'Na': (2, 0), 'Mg': (2, 1), 'Al': (2, 12), 'Si': (2, 13), 'P': (2, 14), 'S': (2, 15), 'Cl': (2, 16), 'Ar': (2, 17),
            'K': (3, 0), 'Ca': (3, 1), 'Sc': (3, 2), 'Ti': (3, 3), 'V': (3, 4), 'Cr': (3, 5), 'Mn': (3, 6), 'Fe': (3, 7),
            'Co': (3, 8), 'Ni': (3, 9), 'Cu': (3, 10), 'Zn': (3, 11), 'Ga': (3, 12), 'Ge': (3, 13), 'As': (3, 14),
            'Se': (3, 15), 'Br': (3, 16), 'Kr': (3, 17),
            'Rb': (4, 0), 'Sr': (4, 1), 'Y': (4, 2), 'Zr': (4, 3), 'Nb': (4, 4), 'Mo': (4, 5), 'Tc': (4, 6), 'Ru': (4, 7),
            'Rh': (4, 8), 'Pd': (4, 9), 'Ag': (4, 10), 'Cd': (4, 11), 'In': (4, 12), 'Sn': (4, 13), 'Sb': (4, 14),
            'Te': (4, 15), 'I': (4, 16), 'Xe': (4, 17),
            'Cs': (5, 0), 'Ba': (5, 1), 'La': (5, 2), 'Hf': (5, 3), 'Ta': (5, 4), 'W': (5, 5), 'Re': (5, 6), 'Os': (5, 7),
            'Ir': (5, 8), 'Pt': (5, 9), 'Au': (5, 10), 'Hg': (5, 11), 'Tl': (5, 12), 'Pb': (5, 13), 'Bi': (5, 14),
            'Po': (5, 15), 'At': (5, 16), 'Rn': (5, 17),
        }
        
        # 归一化重要性
        if element_importance:
            max_imp = max(element_importance.values())
            min_imp = min(element_importance.values())
            range_imp = max_imp - min_imp + 1e-8
        else:
            max_imp, min_imp, range_imp = 1, 0, 1
            
        # 颜色映射
        cmap = plt.cm.YlOrRd
        norm = Normalize(vmin=0, vmax=1)
        
        # 绘制元素格子
        for elem, (row, col) in positions.items():
            x, y = col, 6 - row
            
            # 获取重要性
            imp = element_importance.get(elem, None)
            
            if imp is not None:
                # 归一化
                imp_norm = (imp - min_imp) / range_imp
                color = cmap(norm(imp_norm))
                alpha = 0.9
            else:
                color = '#E8E8E8'
                alpha = 0.5
                
            # 绘制方格
            rect = FancyBboxPatch(
                (x - 0.45, y - 0.45), 0.9, 0.9,
                boxstyle="round,pad=0.02",
                facecolor=color, edgecolor='black',
                linewidth=1, alpha=alpha
            )
            ax.add_patch(rect)
            
            # 元素符号
            text_color = 'white' if imp is not None and (imp - min_imp) / range_imp > 0.5 else 'black'
            ax.text(x, y + 0.1, elem, ha='center', va='center',
                   fontsize=11, fontweight='bold', color=text_color)
                   
            # 重要性值
            if imp is not None:
                ax.text(x, y - 0.25, f'{imp:.2f}', ha='center', va='center',
                       fontsize=7, color=text_color)
                       
        ax.set_xlim(-1, 19)
        ax.set_ylim(-1, 8)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # 颜色条
        sm = ScalarMappable(cmap=cmap, norm=Normalize(vmin=min_imp, vmax=max_imp))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.6, label='Importance')
        
        ax.set_title('Element Importance on Periodic Table', fontsize=14, fontweight='bold', y=1.02)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"✅ 周期表可视化已保存: {save_path}")
            
        plt.close()
        
    def generate_html_report(self, explanation: Dict, atoms_object,
                            image_paths: Dict[str, str],
                            save_path: str):
        """
        生成交互式HTML报告
        
        Args:
            explanation: 解释字典
            atoms_object: Atoms对象
            image_paths: 图像路径字典 {'name': 'path'}
            save_path: HTML文件保存路径
        """
        html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Interpretability Report - {sample_id}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            padding: 30px;
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
        }}
        .metric-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .metric-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
        }}
        .metric-card.success {{
            background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        }}
        .metric-card.warning {{
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        }}
        .metric-value {{
            font-size: 2em;
            font-weight: bold;
        }}
        .metric-label {{
            font-size: 0.9em;
            opacity: 0.9;
        }}
        .image-container {{
            margin: 20px 0;
            text-align: center;
        }}
        .image-container img {{
            max-width: 100%;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.2);
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #3498db;
            color: white;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .footer {{
            text-align: center;
            margin-top: 30px;
            color: #7f8c8d;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🔬 Interpretability Analysis Report</h1>
        <p><strong>Sample ID:</strong> {sample_id}</p>
        
        <h2>📊 Prediction Results</h2>
        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-value">{prediction:.4f}</div>
                <div class="metric-label">Prediction</div>
            </div>
            {true_value_card}
            {error_card}
            {uncertainty_card}
        </div>
        
        <h2>🔗 Modal Contributions</h2>
        <div class="metric-grid">
            <div class="metric-card success">
                <div class="metric-value">{graph_contrib:.1%}</div>
                <div class="metric-label">Graph Contribution</div>
            </div>
            <div class="metric-card warning">
                <div class="metric-value">{text_contrib:.1%}</div>
                <div class="metric-label">Text Contribution</div>
            </div>
        </div>
        
        {images_section}
        
        <h2>📋 Top Important Atoms</h2>
        {atoms_table}
        
        <h2>🧪 Physics Correlations</h2>
        {physics_table}
        
        <div class="footer">
            Generated by Enhanced Interpretability Module v2.0
        </div>
    </div>
</body>
</html>
"""
        
        # 准备数据
        sample_id = explanation.get('sample_id', 'Unknown')
        prediction = explanation.get('prediction', 0)
        true_value = explanation.get('true_value')
        error = explanation.get('error')
        uncertainty = explanation.get('uncertainty', {})
        mc = explanation.get('modal_contribution', {})
        
        # 构建可选卡片
        true_value_card = ""
        if true_value is not None:
            true_value_card = f'''
            <div class="metric-card success">
                <div class="metric-value">{true_value:.4f}</div>
                <div class="metric-label">True Value</div>
            </div>'''
            
        error_card = ""
        if error is not None:
            error_card = f'''
            <div class="metric-card warning">
                <div class="metric-value">{error:.4f}</div>
                <div class="metric-label">Error</div>
            </div>'''
            
        uncertainty_card = ""
        if uncertainty.get('std'):
            uncertainty_card = f'''
            <div class="metric-card">
                <div class="metric-value">{uncertainty["std"]:.4f}</div>
                <div class="metric-label">Uncertainty (σ)</div>
            </div>'''
            
        # 图像部分
        images_section = ""
        for name, path in image_paths.items():
            images_section += f'''
        <h2>{name}</h2>
        <div class="image-container">
            <img src="{path}" alt="{name}">
        </div>'''
            
        # 原子表格
        elements = list(atoms_object.elements)
        importance = explanation.get('atom_importance_integrated_gradients', 
                                    explanation.get('atom_importance_gradient', []))
        
        atoms_rows = ""
        if importance:
            sorted_indices = np.argsort(importance)[::-1][:10]
            for rank, idx in enumerate(sorted_indices, 1):
                atoms_rows += f"<tr><td>{rank}</td><td>{elements[idx]}</td><td>{idx}</td><td>{importance[idx]:.4f}</td></tr>\n"
        atoms_table = f"""
        <table>
            <tr><th>Rank</th><th>Element</th><th>Index</th><th>Importance</th></tr>
            {atoms_rows}
        </table>"""
        
        # 物理相关性表格
        correlations = explanation.get('physics_correlations', {})
        physics_rows = ""
        for prop, corr in sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True):
            physics_rows += f"<tr><td>{prop}</td><td>{corr:+.4f}</td></tr>\n"
        physics_table = f"""
        <table>
            <tr><th>Property</th><th>Correlation</th></tr>
            {physics_rows}
        </table>""" if physics_rows else "<p>No physics correlation data available.</p>"
        
        # 填充模板
        html_content = html_template.format(
            sample_id=sample_id,
            prediction=prediction if isinstance(prediction, float) else prediction[0],
            true_value_card=true_value_card,
            error_card=error_card,
            uncertainty_card=uncertainty_card,
            graph_contrib=mc.get('graph_contribution', 0.5),
            text_contrib=mc.get('text_contribution', 0.5),
            images_section=images_section,
            atoms_table=atoms_table,
            physics_table=physics_table,
        )
        
        # 保存HTML
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
            
        print(f"✅ HTML报告已保存: {save_path}")


# ==================== 便捷函数 ====================

def quick_visualize(explanation, atoms_object, save_dir, sample_id='sample'):
    """
    快速可视化函数
    
    Args:
        explanation: 解释字典
        atoms_object: Atoms对象
        save_dir: 保存目录
        sample_id: 样本ID
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    viz = AdvancedVisualizer(style='publication')
    
    # 综合报告
    viz.create_comprehensive_report(
        explanation, atoms_object,
        save_path=save_dir / f'{sample_id}_comprehensive_report.png'
    )
    
    # 元素重要性（如果有）
    importance = explanation.get('atom_importance_integrated_gradients',
                                explanation.get('atom_importance_gradient', []))
    if len(importance) > 0:
        elements = list(atoms_object.elements)
        from collections import defaultdict
        elem_imp = defaultdict(list)
        for elem, imp in zip(elements, importance):
            elem_imp[elem].append(imp)
        elem_mean = {k: np.mean(v) for k, v in elem_imp.items()}
        
        viz.plot_periodic_table_importance(
            elem_mean,
            save_path=save_dir / f'{sample_id}_periodic_table.png'
        )
        
    print(f"✅ 可视化完成! 保存目录: {save_dir}")


if __name__ == "__main__":
    print("""
    高级可解释性可视化模块
    
    使用示例:
    
    from advanced_visualization import AdvancedVisualizer, quick_visualize
    
    # 方式1: 使用快捷函数
    quick_visualize(explanation, atoms_object, './output', 'sample_001')
    
    # 方式2: 使用可视化器类
    viz = AdvancedVisualizer(style='publication')
    viz.create_comprehensive_report(explanation, atoms_object, 'report.png')
    viz.plot_periodic_table_importance(element_importance, 'periodic_table.png')
    viz.generate_html_report(explanation, atoms_object, image_paths, 'report.html')
    """)
