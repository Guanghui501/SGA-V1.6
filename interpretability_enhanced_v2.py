"""
增强可解释性分析模块 v2.0

功能特性：
1. 多层次注意力分析（全局、细粒度、多头）
2. 原子重要性归因（梯度、积分梯度、SHAP近似）
3. 文本Token重要性分析
4. 跨模态交互可视化
5. 物理化学特征关联分析
6. 预测置信度与不确定性估计
7. 批量可解释性报告生成
8. 符号回归分析（从神经网络特征中发现可解释的数学公式）

作者: Enhanced Interpretability Module v2
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union, Any
import pandas as pd
from pathlib import Path
import json
from collections import defaultdict
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


def safe_to_float(value):
    """
    安全地将各种格式的数值转换为Python float
    
    处理：标量、0维数组、1维数组、tensor等
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return float(value.item())
        else:
            return float(value.flat[0])
    if torch.is_tensor(value):
        return float(value.detach().cpu().item()) if value.numel() == 1 else float(value.detach().cpu().flatten()[0].item())
    if hasattr(value, 'item'):
        return float(value.item())
    return float(value)


class AtomImportanceAnalyzer:
    """原子重要性分析器 - 多种归因方法"""
    
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.model.eval()
        
    def gradient_importance(self, g, lg, text, target_output=None):
        """
        梯度法计算原子重要性
        
        Args:
            g: DGL graph
            lg: Line graph
            text: 文本列表
            target_output: 目标输出索引（用于多输出任务）
            
        Returns:
            importance: [num_atoms] 原子重要性分数
            gradients: 原始梯度
        """
        g = g.to(self.device)
        lg = lg.to(self.device)
        
        # 获取原子特征并启用梯度
        atom_features = g.ndata['atom_features'].clone().detach().requires_grad_(True)
        original_features = g.ndata['atom_features']
        g.ndata['atom_features'] = atom_features
        
        # 前向传播
        self.model.train()  # 启用梯度计算
        output = self.model([g, lg, text])
        
        if isinstance(output, dict):
            prediction = output['predictions']
        else:
            prediction = output
            
        # 选择目标输出
        if target_output is not None and prediction.dim() > 1:
            prediction = prediction[:, target_output]
            
        # 反向传播
        loss = prediction.sum()
        loss.backward()
        
        # 计算重要性（梯度的L2范数）
        gradients = atom_features.grad.detach()
        importance = torch.norm(gradients, dim=1).cpu().numpy()
        
        # 恢复模型状态
        self.model.eval()
        g.ndata['atom_features'] = original_features
        
        return importance, gradients.cpu().numpy()
    
    def integrated_gradients(self, g, lg, text, steps=50, baseline='zero'):
        """
        积分梯度法 - 更可靠的归因方法
        
        Args:
            g, lg, text: 输入数据
            steps: 积分步数
            baseline: 基线类型 ('zero', 'random', 'mean')
            
        Returns:
            importance: 原子重要性分数
            attributions: 详细归因
        """
        g = g.to(self.device)
        lg = lg.to(self.device)
        
        original_features = g.ndata['atom_features'].clone()
        
        # 创建基线
        if baseline == 'zero':
            baseline_features = torch.zeros_like(original_features)
        elif baseline == 'random':
            baseline_features = torch.randn_like(original_features) * 0.1
        elif baseline == 'mean':
            baseline_features = original_features.mean(dim=0, keepdim=True).expand_as(original_features)
        else:
            baseline_features = torch.zeros_like(original_features)
            
        # 积分路径
        integrated_grads = torch.zeros_like(original_features)
        
        self.model.train()
        
        for alpha in torch.linspace(0, 1, steps):
            # 插值特征
            interpolated = baseline_features + alpha * (original_features - baseline_features)
            interpolated = interpolated.clone().detach().requires_grad_(True)
            g.ndata['atom_features'] = interpolated
            
            # 前向传播
            output = self.model([g, lg, text])
            if isinstance(output, dict):
                prediction = output['predictions']
            else:
                prediction = output
                
            loss = prediction.sum()
            loss.backward()
            
            integrated_grads += interpolated.grad
            
        # 平均并缩放
        integrated_grads = integrated_grads / steps
        attributions = integrated_grads * (original_features - baseline_features)
        importance = torch.norm(attributions, dim=1).cpu().numpy()
        
        # 恢复
        self.model.eval()
        g.ndata['atom_features'] = original_features
        
        return importance, attributions.cpu().numpy()
    
    def layer_wise_relevance(self, g, lg, text):
        """
        Layer-wise Relevance Propagation (LRP) 近似
        通过分析中间层激活来理解贡献
        """
        g = g.to(self.device)
        lg = lg.to(self.device)
        
        # 注册钩子收集中间激活
        activations = {}
        hooks = []
        
        def save_activation(name):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    activations[name] = output[0].detach()
                else:
                    activations[name] = output.detach()
            return hook
        
        # 在关键层注册钩子
        for name, module in self.model.named_modules():
            if 'alignn_layers' in name or 'gcn_layers' in name:
                hooks.append(module.register_forward_hook(save_activation(name)))
                
        # 前向传播
        with torch.no_grad():
            output = self.model([g, lg, text])
            
        # 移除钩子
        for hook in hooks:
            hook.remove()
            
        # 分析激活强度作为重要性代理
        layer_importance = {}
        for name, act in activations.items():
            if act.dim() >= 2:
                layer_importance[name] = torch.norm(act, dim=-1).cpu().numpy()
                
        return layer_importance, activations


class TextTokenAnalyzer:
    """文本Token重要性分析器"""
    
    def __init__(self, model, tokenizer, device='cuda'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        
    def analyze_token_importance(self, g, lg, text, method='attention'):
        """
        分析文本token的重要性
        
        Args:
            g, lg, text: 输入数据
            method: 分析方法 ('attention', 'gradient', 'occlusion')
            
        Returns:
            token_importance: {token: importance_score}
        """
        if method == 'attention':
            return self._attention_based_importance(g, lg, text)
        elif method == 'gradient':
            return self._gradient_based_importance(g, lg, text)
        elif method == 'occlusion':
            return self._occlusion_based_importance(g, lg, text)
        else:
            raise ValueError(f"Unknown method: {method}")
            
    def _attention_based_importance(self, g, lg, text):
        """基于注意力权重的token重要性"""
        g = g.to(self.device)
        lg = lg.to(self.device)
        
        with torch.no_grad():
            output = self.model(
                [g, lg, text],
                return_features=True,
                return_attention=True
            )
            
        # 获取细粒度注意力
        if 'fine_grained_attention_weights' in output:
            fg_attn = output['fine_grained_attention_weights']
            
            # text_to_atom: [batch, heads, seq_len, num_atoms]
            t2a = fg_attn.get('text_to_atom', None)
            
            if t2a is not None:
                # 平均跨头和原子，得到每个token的重要性
                token_importance = t2a[0].mean(dim=0).mean(dim=1).cpu().numpy()
                
                # 获取token文本
                tokens = self.tokenizer.tokenize(text[0])
                tokens = ['[CLS]'] + tokens + ['[SEP]']
                
                # 截断或填充到匹配长度
                seq_len = len(token_importance)
                if len(tokens) > seq_len:
                    tokens = tokens[:seq_len]
                elif len(tokens) < seq_len:
                    tokens = tokens + ['[PAD]'] * (seq_len - len(tokens))
                    
                return dict(zip(tokens, token_importance))
                
        return {}
    
    def _gradient_based_importance(self, g, lg, text):
        """基于梯度的token重要性"""
        # 需要访问文本编码器的嵌入层
        # 这里提供框架，具体实现取决于模型结构
        pass
    
    def _occlusion_based_importance(self, g, lg, text, batch_size=10):
        """
        遮挡法：逐个mask token观察预测变化
        """
        g = g.to(self.device)
        lg = lg.to(self.device)
        
        # 基准预测
        with torch.no_grad():
            base_output = self.model([g, lg, text])
            if isinstance(base_output, dict):
                base_pred = base_output['predictions'].item()
            else:
                base_pred = base_output.item()
                
        # 获取tokens
        tokens = self.tokenizer.tokenize(text[0])
        tokens = ['[CLS]'] + tokens + ['[SEP]']
        
        token_importance = {}
        
        for i, token in enumerate(tokens):
            if token in ['[CLS]', '[SEP]', '[PAD]']:
                token_importance[f"{token}_{i}"] = 0.0
                continue
                
            # 创建mask文本
            masked_tokens = tokens.copy()
            masked_tokens[i] = '[MASK]'
            masked_text = [self.tokenizer.convert_tokens_to_string(masked_tokens[1:-1])]
            
            # 预测
            with torch.no_grad():
                masked_output = self.model([g, lg, masked_text])
                if isinstance(masked_output, dict):
                    masked_pred = masked_output['predictions'].item()
                else:
                    masked_pred = masked_output.item()
                    
            # 重要性 = 预测变化的绝对值
            importance = abs(base_pred - masked_pred)
            token_importance[f"{token}_{i}"] = importance
            
        return token_importance


class CrossModalInteractionAnalyzer:
    """跨模态交互分析器"""
    
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        
    def analyze_modal_contribution(self, g, lg, text):
        """
        分析各模态对预测的贡献
        
        Returns:
            contributions: {
                'graph_only': prediction,
                'text_only': prediction,
                'combined': prediction,
                'graph_contribution': float,
                'text_contribution': float,
                'synergy': float  # 协同效应
            }
        """
        g = g.to(self.device)
        lg = lg.to(self.device)
        
        contributions = {}
        
        # 完整预测
        with torch.no_grad():
            full_output = self.model([g, lg, text], return_features=True)
            if isinstance(full_output, dict):
                contributions['combined'] = safe_to_float(full_output['predictions'].cpu().numpy())
                graph_feat = full_output.get('graph_features', None)
                text_feat = full_output.get('text_features', None)
            else:
                contributions['combined'] = safe_to_float(full_output.cpu().numpy())
                
        # 特征范数作为贡献度代理
        if graph_feat is not None and text_feat is not None:
            graph_norm = torch.norm(graph_feat).item()
            text_norm = torch.norm(text_feat).item()
            total_norm = graph_norm + text_norm
            
            contributions['graph_contribution'] = float(graph_norm / total_norm)
            contributions['text_contribution'] = float(text_norm / total_norm)
            
            # 计算特征相似度作为协同度量
            if graph_feat.shape == text_feat.shape:
                similarity = F.cosine_similarity(
                    graph_feat.flatten().unsqueeze(0),
                    text_feat.flatten().unsqueeze(0)
                ).item()
                contributions['feature_alignment'] = float(similarity)
                
        return contributions
    
    def attention_flow_analysis(self, g, lg, text):
        """
        注意力流分析：追踪信息如何在模态间流动
        """
        g = g.to(self.device)
        lg = lg.to(self.device)
        
        with torch.no_grad():
            output = self.model(
                [g, lg, text],
                return_features=True,
                return_attention=True,
                return_intermediate_features=True
            )
            
        flow_analysis = {}
        
        # 收集各阶段的特征
        if 'graph_base' in output:
            flow_analysis['graph_base_norm'] = float(torch.norm(output['graph_base']).item())
        if 'text_base' in output:
            flow_analysis['text_base_norm'] = float(torch.norm(output['text_base']).item())
        if 'graph_cross' in output:
            flow_analysis['graph_after_cross_norm'] = float(torch.norm(output['graph_cross']).item())
        if 'text_cross' in output:
            flow_analysis['text_after_cross_norm'] = float(torch.norm(output['text_cross']).item())
            
        # 计算特征变化
        if 'graph_base' in output and 'graph_cross' in output:
            change = output['graph_cross'] - output['graph_base']
            flow_analysis['graph_change_from_text'] = float(torch.norm(change).item())
            
        if 'text_base' in output and 'text_cross' in output:
            change = output['text_cross'] - output['text_base']
            flow_analysis['text_change_from_graph'] = float(torch.norm(change).item())
            
        return flow_analysis


class PhysicsCorrelationAnalyzer:
    """物理化学特征关联分析器"""
    
    def __init__(self):
        # 元素物理化学数据
        self.element_data = self._load_element_data()
        
    def _load_element_data(self):
        """加载元素物理化学数据"""
        # 基础元素数据（可扩展）
        data = {
            'H': {'electronegativity': 2.20, 'atomic_radius': 53, 'group': 1, 'period': 1},
            'Li': {'electronegativity': 0.98, 'atomic_radius': 167, 'group': 1, 'period': 2},
            'Be': {'electronegativity': 1.57, 'atomic_radius': 112, 'group': 2, 'period': 2},
            'B': {'electronegativity': 2.04, 'atomic_radius': 87, 'group': 13, 'period': 2},
            'C': {'electronegativity': 2.55, 'atomic_radius': 67, 'group': 14, 'period': 2},
            'N': {'electronegativity': 3.04, 'atomic_radius': 56, 'group': 15, 'period': 2},
            'O': {'electronegativity': 3.44, 'atomic_radius': 48, 'group': 16, 'period': 2},
            'F': {'electronegativity': 3.98, 'atomic_radius': 42, 'group': 17, 'period': 2},
            'Na': {'electronegativity': 0.93, 'atomic_radius': 190, 'group': 1, 'period': 3},
            'Mg': {'electronegativity': 1.31, 'atomic_radius': 145, 'group': 2, 'period': 3},
            'Al': {'electronegativity': 1.61, 'atomic_radius': 118, 'group': 13, 'period': 3},
            'Si': {'electronegativity': 1.90, 'atomic_radius': 111, 'group': 14, 'period': 3},
            'P': {'electronegativity': 2.19, 'atomic_radius': 98, 'group': 15, 'period': 3},
            'S': {'electronegativity': 2.58, 'atomic_radius': 88, 'group': 16, 'period': 3},
            'Cl': {'electronegativity': 3.16, 'atomic_radius': 79, 'group': 17, 'period': 3},
            'K': {'electronegativity': 0.82, 'atomic_radius': 243, 'group': 1, 'period': 4},
            'Ca': {'electronegativity': 1.00, 'atomic_radius': 194, 'group': 2, 'period': 4},
            'Ti': {'electronegativity': 1.54, 'atomic_radius': 176, 'group': 4, 'period': 4},
            'V': {'electronegativity': 1.63, 'atomic_radius': 171, 'group': 5, 'period': 4},
            'Cr': {'electronegativity': 1.66, 'atomic_radius': 166, 'group': 6, 'period': 4},
            'Mn': {'electronegativity': 1.55, 'atomic_radius': 161, 'group': 7, 'period': 4},
            'Fe': {'electronegativity': 1.83, 'atomic_radius': 156, 'group': 8, 'period': 4},
            'Co': {'electronegativity': 1.88, 'atomic_radius': 152, 'group': 9, 'period': 4},
            'Ni': {'electronegativity': 1.91, 'atomic_radius': 149, 'group': 10, 'period': 4},
            'Cu': {'electronegativity': 1.90, 'atomic_radius': 145, 'group': 11, 'period': 4},
            'Zn': {'electronegativity': 1.65, 'atomic_radius': 142, 'group': 12, 'period': 4},
            'Ga': {'electronegativity': 1.81, 'atomic_radius': 136, 'group': 13, 'period': 4},
            'Ge': {'electronegativity': 2.01, 'atomic_radius': 125, 'group': 14, 'period': 4},
            'As': {'electronegativity': 2.18, 'atomic_radius': 114, 'group': 15, 'period': 4},
            'Se': {'electronegativity': 2.55, 'atomic_radius': 103, 'group': 16, 'period': 4},
            'Br': {'electronegativity': 2.96, 'atomic_radius': 94, 'group': 17, 'period': 4},
            'Sr': {'electronegativity': 0.95, 'atomic_radius': 219, 'group': 2, 'period': 5},
            'Y': {'electronegativity': 1.22, 'atomic_radius': 212, 'group': 3, 'period': 5},
            'Zr': {'electronegativity': 1.33, 'atomic_radius': 206, 'group': 4, 'period': 5},
            'Nb': {'electronegativity': 1.60, 'atomic_radius': 198, 'group': 5, 'period': 5},
            'Mo': {'electronegativity': 2.16, 'atomic_radius': 190, 'group': 6, 'period': 5},
            'Ag': {'electronegativity': 1.93, 'atomic_radius': 165, 'group': 11, 'period': 5},
            'Cd': {'electronegativity': 1.69, 'atomic_radius': 161, 'group': 12, 'period': 5},
            'In': {'electronegativity': 1.78, 'atomic_radius': 156, 'group': 13, 'period': 5},
            'Sn': {'electronegativity': 1.96, 'atomic_radius': 145, 'group': 14, 'period': 5},
            'Sb': {'electronegativity': 2.05, 'atomic_radius': 133, 'group': 15, 'period': 5},
            'Te': {'electronegativity': 2.10, 'atomic_radius': 123, 'group': 16, 'period': 5},
            'I': {'electronegativity': 2.66, 'atomic_radius': 115, 'group': 17, 'period': 5},
            'Ba': {'electronegativity': 0.89, 'atomic_radius': 253, 'group': 2, 'period': 6},
            'La': {'electronegativity': 1.10, 'atomic_radius': 195, 'group': 3, 'period': 6},
            'Pb': {'electronegativity': 2.33, 'atomic_radius': 154, 'group': 14, 'period': 6},
            'Bi': {'electronegativity': 2.02, 'atomic_radius': 143, 'group': 15, 'period': 6},
        }
        return data
    
    def correlate_importance_with_physics(self, elements, importance_scores):
        """
        分析原子重要性与物理化学特征的关联
        
        Args:
            elements: 元素列表
            importance_scores: 重要性分数
            
        Returns:
            correlations: 物理量与重要性的相关性
        """
        # 收集物理化学特征
        physics_features = defaultdict(list)
        valid_importance = []
        
        for elem, imp in zip(elements, importance_scores):
            if elem in self.element_data:
                for key, value in self.element_data[elem].items():
                    physics_features[key].append(value)
                valid_importance.append(imp)
                
        if len(valid_importance) < 3:
            return {}
            
        # 计算相关性
        correlations = {}
        for key, values in physics_features.items():
            if len(values) == len(valid_importance):
                corr = np.corrcoef(values, valid_importance)[0, 1]
                correlations[key] = float(corr) if not np.isnan(corr) else 0.0
                
        return correlations
    
    def analyze_structure_property_relation(self, atoms_object, prediction, importance_scores):
        """
        分析结构-性质关系
        
        Args:
            atoms_object: JARVIS Atoms对象
            prediction: 模型预测值
            importance_scores: 原子重要性
            
        Returns:
            analysis: 结构-性质分析结果
        """
        analysis = {}
        
        elements = list(atoms_object.elements)
        coords = atoms_object.cart_coords
        
        # 元素统计
        unique_elements = list(set(elements))
        element_importance = {}
        for elem in unique_elements:
            indices = [i for i, e in enumerate(elements) if e == elem]
            element_importance[elem] = float(np.mean([importance_scores[i] for i in indices]))
        analysis['element_importance'] = element_importance
        
        # 按重要性排序的元素
        sorted_elements = sorted(element_importance.items(), key=lambda x: x[1], reverse=True)
        analysis['most_important_element'] = list(sorted_elements[0]) if sorted_elements else None
        
        # 空间分布分析
        if len(coords) > 0:
            # 重心
            center = coords.mean(axis=0)
            
            # 重要原子的空间分布
            top_k = min(5, len(importance_scores))
            top_indices = np.argsort(importance_scores)[-top_k:]
            top_coords = coords[top_indices]
            
            # 重要原子是否集中在某区域
            if len(top_coords) > 1:
                spread = np.std(top_coords, axis=0).mean()
                analysis['important_atoms_spread'] = float(spread)
                
        # 配位环境分析（简化版）
        analysis['num_atoms'] = len(elements)
        analysis['num_elements'] = len(unique_elements)
        analysis['prediction'] = safe_to_float(prediction)
        
        return analysis


class UncertaintyEstimator:
    """不确定性估计器"""
    
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        
    def mc_dropout_uncertainty(self, g, lg, text, n_samples=30):
        """
        Monte Carlo Dropout 不确定性估计
        
        Args:
            g, lg, text: 输入
            n_samples: MC采样次数
            
        Returns:
            mean: 预测均值
            std: 预测标准差（不确定性）
            samples: 所有采样预测
        """
        g = g.to(self.device)
        lg = lg.to(self.device)
        
        # 启用dropout（训练模式）
        self.model.train()
        
        predictions = []
        for _ in range(n_samples):
            with torch.no_grad():
                output = self.model([g, lg, text])
                if isinstance(output, dict):
                    pred = output['predictions'].cpu().numpy()
                else:
                    pred = output.cpu().numpy()
                predictions.append(pred)
                
        # 恢复评估模式
        self.model.eval()
        
        predictions = np.array(predictions)
        mean = predictions.mean(axis=0)
        std = predictions.std(axis=0)
        
        return mean, std, predictions
    
    def feature_space_uncertainty(self, g, lg, text, reference_features=None):
        """
        基于特征空间的不确定性估计
        检测输入是否在训练分布内
        """
        g = g.to(self.device)
        lg = lg.to(self.device)
        
        with torch.no_grad():
            output = self.model([g, lg, text], return_features=True)
            
        if isinstance(output, dict):
            graph_feat = output.get('graph_features', None)
            text_feat = output.get('text_features', None)
            
            if graph_feat is not None:
                feature_norm = float(torch.norm(graph_feat).item())
                
                # 如果有参考特征分布，计算马氏距离
                if reference_features is not None:
                    # 简化：使用欧氏距离到参考中心
                    ref_center = reference_features.mean(dim=0)
                    distance = float(torch.norm(graph_feat - ref_center).item())
                    return {'feature_norm': feature_norm, 'distance_to_center': distance}
                    
                return {'feature_norm': feature_norm}
                
        return {}


class InterpretabilityVisualizer:
    """可解释性可视化工具"""
    
    @staticmethod
    def plot_atom_importance_3d(atoms_object, importance_scores, save_path=None, 
                                 title="Atom Importance Visualization"):
        """
        3D原子重要性可视化
        """
        from mpl_toolkits.mplot3d import Axes3D
        
        coords = atoms_object.cart_coords
        elements = list(atoms_object.elements)
        
        # 归一化重要性
        imp_norm = (importance_scores - importance_scores.min()) / \
                   (importance_scores.max() - importance_scores.min() + 1e-8)
        
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # 颜色映射
        colors = plt.cm.YlOrRd(imp_norm)
        
        # 绘制原子
        scatter = ax.scatter(
            coords[:, 0], coords[:, 1], coords[:, 2],
            c=imp_norm, cmap='YlOrRd', s=500, alpha=0.8,
            edgecolors='black', linewidth=1
        )
        
        # 标注元素符号
        for i, (coord, elem) in enumerate(zip(coords, elements)):
            ax.text(coord[0], coord[1], coord[2], elem, 
                   fontsize=10, fontweight='bold', ha='center', va='center')
            
        ax.set_xlabel('X (Å)', fontsize=12)
        ax.set_ylabel('Y (Å)', fontsize=12)
        ax.set_zlabel('Z (Å)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # 颜色条
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.6, label='Importance')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ 3D可视化已保存: {save_path}")
            
        plt.close()
        
    @staticmethod
    def plot_attention_heatmap(attention_weights, row_labels, col_labels, 
                               title="Attention Heatmap", save_path=None):
        """
        注意力热图可视化
        """
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # 确保数据是2D
        if attention_weights.ndim > 2:
            attention_weights = attention_weights.mean(axis=tuple(range(attention_weights.ndim - 2)))
            
        sns.heatmap(
            attention_weights,
            xticklabels=col_labels,
            yticklabels=row_labels,
            cmap='YlOrRd',
            ax=ax,
            cbar_kws={'label': 'Attention Weight'}
        )
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right', fontsize=8)
        plt.yticks(fontsize=8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ 热图已保存: {save_path}")
            
        plt.close()
        
    @staticmethod
    def plot_modal_contribution(contributions, save_path=None):
        """
        模态贡献饼图
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # 左图：贡献比例
        if 'graph_contribution' in contributions and 'text_contribution' in contributions:
            ax1 = axes[0]
            sizes = [contributions['graph_contribution'], contributions['text_contribution']]
            labels = ['Graph\n(Structure)', 'Text\n(Description)']
            colors = ['#3498db', '#e74c3c']
            explode = (0.05, 0.05)
            
            wedges, texts, autotexts = ax1.pie(
                sizes, explode=explode, labels=labels, colors=colors,
                autopct='%1.1f%%', shadow=True, startangle=90,
                textprops={'fontsize': 12}
            )
            ax1.set_title('Modal Contribution', fontsize=14, fontweight='bold')
            
        # 右图：特征对齐度
        ax2 = axes[1]
        metrics = ['Feature\nAlignment', 'Graph\nNorm', 'Text\nNorm']
        values = [
            contributions.get('feature_alignment', 0),
            contributions.get('graph_contribution', 0),
            contributions.get('text_contribution', 0)
        ]
        colors = ['#2ecc71', '#3498db', '#e74c3c']
        
        bars = ax2.bar(metrics, values, color=colors, edgecolor='black', linewidth=1.5)
        ax2.set_ylabel('Value', fontsize=12)
        ax2.set_title('Feature Metrics', fontsize=14, fontweight='bold')
        ax2.set_ylim(0, 1.1)
        
        for bar, val in zip(bars, values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ 模态贡献图已保存: {save_path}")
            
        plt.close()
        
    @staticmethod
    def plot_physics_correlation(correlations, save_path=None):
        """
        物理特征与重要性的相关性
        """
        if not correlations:
            print("⚠️ 没有相关性数据")
            return
            
        fig, ax = plt.subplots(figsize=(10, 6))
        
        properties = list(correlations.keys())
        values = list(correlations.values())
        colors = ['#e74c3c' if v < 0 else '#2ecc71' for v in values]
        
        bars = ax.barh(properties, values, color=colors, edgecolor='black', linewidth=1)
        
        ax.axvline(x=0, color='black', linewidth=1)
        ax.set_xlabel('Correlation with Importance', fontsize=12)
        ax.set_title('Physics-Importance Correlation', fontsize=14, fontweight='bold')
        ax.set_xlim(-1, 1)
        
        for bar, val in zip(bars, values):
            ax.text(val + 0.02 if val >= 0 else val - 0.02,
                   bar.get_y() + bar.get_height()/2,
                   f'{val:.3f}', ha='left' if val >= 0 else 'right',
                   va='center', fontsize=10)
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ 物理相关性图已保存: {save_path}")
            
        plt.close()


class ComprehensiveExplainer:
    """综合解释器 - 整合所有分析功能"""
    
    def __init__(self, model, tokenizer=None, device='cuda'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        
        # 初始化各分析器
        self.atom_analyzer = AtomImportanceAnalyzer(model, device)
        self.text_analyzer = TextTokenAnalyzer(model, tokenizer, device) if tokenizer else None
        self.cross_modal_analyzer = CrossModalInteractionAnalyzer(model, device)
        self.physics_analyzer = PhysicsCorrelationAnalyzer()
        self.uncertainty_estimator = UncertaintyEstimator(model, device)
        self.visualizer = InterpretabilityVisualizer()
        
        # 尝试导入局部环境分析器
        try:
            from local_environment_analyzer import (
                LocalEnvironmentAnalyzer, 
                LocalEnvironmentVisualizer,
                EnhancedAttentionVisualizer
            )
            self.local_env_analyzer = LocalEnvironmentAnalyzer()
            self.local_env_visualizer = LocalEnvironmentVisualizer()
            self.enhanced_attn_visualizer = EnhancedAttentionVisualizer()
            self._has_local_env = True
        except ImportError:
            self._has_local_env = False
            print("⚠️ 局部环境分析模块未找到，跳过局部环境分析")
        
    def explain_prediction(self, g, lg, text, atoms_object, 
                          true_value=None, save_dir=None, sample_id='sample'):
        """
        为单个预测生成完整解释报告
        
        Args:
            g, lg, text: 模型输入
            atoms_object: JARVIS Atoms对象
            true_value: 真实值（可选）
            save_dir: 保存目录
            sample_id: 样本ID
            
        Returns:
            explanation: 完整解释字典
        """
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            
        print(f"\n{'='*80}")
        print(f"🔬 样本 {sample_id} 的综合可解释性分析")
        print(f"{'='*80}")
        
        explanation = {'sample_id': sample_id}
        
        # ==================== 1. 基础预测 ====================
        print("\n📊 [1/7] 获取预测结果...")
        
        g_device = g.to(self.device)
        lg_device = lg.to(self.device)
        
        with torch.no_grad():
            output = self.model(
                [g_device, lg_device, text],
                return_features=True,
                return_attention=True,
                return_intermediate_features=True
            )
            
        if isinstance(output, dict):
            prediction = output['predictions'].cpu().numpy()
        else:
            prediction = output.cpu().numpy()
        
        pred_value = safe_to_float(prediction)
        explanation['prediction'] = pred_value
        explanation['true_value'] = float(true_value) if true_value is not None else None
        
        if true_value is not None:
            error = abs(pred_value - float(true_value))
            explanation['error'] = error
            print(f"   预测值: {pred_value:.4f}")
            print(f"   真实值: {true_value:.4f}")
            print(f"   误差: {error:.4f}")
        else:
            print(f"   预测值: {pred_value:.4f}")
            
        # ==================== 2. 原子重要性分析 ====================
        print("\n⚛️  [2/7] 计算原子重要性...")
        
        # 梯度法
        importance_grad, gradients = self.atom_analyzer.gradient_importance(g, lg, text)
        explanation['atom_importance_gradient'] = importance_grad.tolist()
        
        # 积分梯度法
        importance_ig, attributions = self.atom_analyzer.integrated_gradients(g, lg, text, steps=30)
        explanation['atom_importance_integrated_gradients'] = importance_ig.tolist()
        
        # 选择积分梯度作为主要重要性
        primary_importance = importance_ig
        
        # 可视化
        elements = list(atoms_object.elements)
        print(f"   分析了 {len(elements)} 个原子")
        
        # 显示top-5重要原子
        top_k = min(5, len(elements))
        top_indices = np.argsort(primary_importance)[-top_k:][::-1]
        print(f"   Top-{top_k} 重要原子:")
        for rank, idx in enumerate(top_indices, 1):
            print(f"     {rank}. {elements[idx]} (index {idx}): {primary_importance[idx]:.4f}")
            
        if save_dir:
            # 2D可视化
            self._plot_atom_importance_2d(
                atoms_object, primary_importance,
                save_path=save_dir / f'{sample_id}_atom_importance.png'
            )
            
        # ==================== 3. 物理化学关联分析 ====================
        print("\n🧪 [3/7] 分析物理化学关联...")
        
        correlations = self.physics_analyzer.correlate_importance_with_physics(
            elements, primary_importance
        )
        explanation['physics_correlations'] = correlations
        
        if correlations:
            print("   重要性与物理量的相关性:")
            for prop, corr in sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True):
                print(f"     {prop}: {corr:+.3f}")
                
            if save_dir:
                self.visualizer.plot_physics_correlation(
                    correlations,
                    save_path=save_dir / f'{sample_id}_physics_correlation.png'
                )
                
        # 结构-性质分析
        struct_analysis = self.physics_analyzer.analyze_structure_property_relation(
            atoms_object, pred_value, primary_importance
        )
        explanation['structure_analysis'] = struct_analysis
        
        # ==================== 3.5 局部化学环境分析 ====================
        if self._has_local_env:
            print("\n🔬 [3.5/7] 分析局部化学环境...")
            
            local_env = self.local_env_analyzer.analyze_local_environment(
                atoms_object, primary_importance
            )
            explanation['local_environment'] = local_env
            
            coord_data = local_env.get('coordination', {})
            bond_data = local_env.get('bonds', {})
            
            print(f"   平均配位数: {coord_data.get('mean_coordination', 0):.2f}")
            print(f"   键类型数: {len(bond_data.get('bond_types', []))}")
            print(f"   总成键数: {bond_data.get('total_bonds', 0)}")
            
            # 配位数与重要性相关性
            env_corr = local_env.get('environment_importance_correlation', {})
            if 'coordination_importance_correlation' in env_corr:
                print(f"   配位数-重要性相关性: {env_corr['coordination_importance_correlation']:+.3f}")
                
            if save_dir:
                self.local_env_visualizer.plot_coordination_analysis(
                    local_env,
                    primary_importance,
                    save_path=save_dir / f'{sample_id}_local_environment.png'
                )
        
        # ==================== 4. 跨模态交互分析 ====================
        print("\n🔗 [4/7] 分析跨模态交互...")
        
        modal_contribution = self.cross_modal_analyzer.analyze_modal_contribution(g, lg, text)
        explanation['modal_contribution'] = modal_contribution
        
        print(f"   图模态贡献: {modal_contribution.get('graph_contribution', 0):.1%}")
        print(f"   文本模态贡献: {modal_contribution.get('text_contribution', 0):.1%}")
        if 'feature_alignment' in modal_contribution:
            print(f"   特征对齐度: {modal_contribution['feature_alignment']:.3f}")
            
        if save_dir:
            self.visualizer.plot_modal_contribution(
                modal_contribution,
                save_path=save_dir / f'{sample_id}_modal_contribution.png'
            )
            
        # 注意力流分析
        flow_analysis = self.cross_modal_analyzer.attention_flow_analysis(g, lg, text)
        explanation['attention_flow'] = flow_analysis
        
        # ==================== 5. 注意力权重分析 ====================
        print("\n👁️  [5/7] 分析注意力权重...")
        
        if isinstance(output, dict):
            # 全局注意力
            if 'attention_weights' in output and output['attention_weights']:
                attn = output['attention_weights']
                explanation['global_attention'] = {
                    k: v.cpu().numpy().tolist() if v is not None else None
                    for k, v in attn.items()
                }
                print("   ✓ 全局跨模态注意力已提取")
                
            # 细粒度注意力
            if 'fine_grained_attention_weights' in output and output['fine_grained_attention_weights']:
                fg_attn = output['fine_grained_attention_weights']
                
                # 分析原子-token注意力
                if 'atom_to_text' in fg_attn and fg_attn['atom_to_text'] is not None:
                    a2t = fg_attn['atom_to_text'][0].cpu().numpy()  # [heads, atoms, tokens]
                    
                    # 保存原始多头注意力
                    explanation['atom_to_text_attention'] = a2t.tolist()
                    
                    # 平均跨头
                    a2t_mean = a2t.mean(axis=0)  # [atoms, tokens]
                    
                    # 每个原子最关注的token位置
                    atom_focus = a2t_mean.argmax(axis=1)
                    explanation['atom_text_focus'] = atom_focus.tolist()
                    
                    print(f"   ✓ 细粒度注意力已提取 (shape: {a2t.shape})")
                    
                    if save_dir:
                        # 使用增强的注意力可视化
                        if self._has_local_env:
                            # 准备token标签
                            if self.tokenizer and len(text) > 0:
                                try:
                                    tokens = self.tokenizer.tokenize(text[0])[:a2t.shape[2]]
                                    token_labels = tokens
                                except:
                                    token_labels = [f'T{i}' for i in range(a2t.shape[2])]
                            else:
                                token_labels = [f'T{i}' for i in range(a2t.shape[2])]
                            
                            # 增强的注意力热图
                            self.enhanced_attn_visualizer.plot_atom_text_attention_enhanced(
                                a2t,  # 多头注意力
                                atom_labels=[f"{elements[i]}" for i in range(len(elements))],
                                token_labels=token_labels,
                                importance_scores=primary_importance,
                                save_path=save_dir / f'{sample_id}_atom_text_attention_enhanced.png'
                            )
                            
                            # 多头注意力分解图
                            if a2t.shape[0] > 1:  # 如果有多头
                                self.enhanced_attn_visualizer.plot_multi_head_attention(
                                    a2t,
                                    atom_labels=[f"{elements[i]}" for i in range(len(elements))],
                                    token_labels=token_labels,
                                    save_path=save_dir / f'{sample_id}_multi_head_attention.png'
                                )
                                
                            # 注意力流图
                            self.enhanced_attn_visualizer.plot_attention_flow_sankey(
                                a2t_mean,
                                atom_labels=[f"{elements[i]}_{i}" for i in range(len(elements))],
                                token_labels=token_labels,
                                save_path=save_dir / f'{sample_id}_attention_flow.png'
                            )
                        else:
                            # 使用原始的简化热图
                            top_atoms = min(10, len(elements))
                            top_atom_indices = np.argsort(primary_importance)[-top_atoms:]
                            
                            self.visualizer.plot_attention_heatmap(
                                a2t_mean[top_atom_indices],
                                row_labels=[f"{elements[i]}_{i}" for i in top_atom_indices],
                                col_labels=[f"T{i}" for i in range(a2t_mean.shape[1])],
                                title="Atom-to-Text Attention (Top Atoms)",
                                save_path=save_dir / f'{sample_id}_atom_text_attention.png'
                            )
                        
        # ==================== 6. 不确定性估计 ====================
        print("\n📉 [6/7] 估计预测不确定性...")
        
        mean_pred, std_pred, mc_samples = self.uncertainty_estimator.mc_dropout_uncertainty(
            g, lg, text, n_samples=20
        )
        
        mean_val = safe_to_float(mean_pred)
        std_val = safe_to_float(std_pred)
        
        explanation['uncertainty'] = {
            'mean': mean_val,
            'std': std_val,
            'confidence_interval_95': [
                mean_val - 1.96 * std_val,
                mean_val + 1.96 * std_val
            ] if mean_val is not None else None
        }
        
        print(f"   预测均值: {mean_val:.4f}")
        print(f"   预测标准差: {std_val:.4f}")
        print(f"   95%置信区间: [{mean_val - 1.96*std_val:.4f}, "
              f"{mean_val + 1.96*std_val:.4f}]")
        
        # ==================== 7. 生成报告 ====================
        print("\n📝 [7/7] 生成解释报告...")
        
        if save_dir:
            # 保存JSON报告
            report_path = save_dir / f'{sample_id}_explanation.json'
            
            # 转换numpy数组为列表（增强版，处理所有numpy类型）
            def convert_to_serializable(obj):
                if obj is None:
                    return None
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, (np.floating, np.float32, np.float64)):
                    return float(obj)
                elif isinstance(obj, (np.integer, np.int32, np.int64)):
                    return int(obj)
                elif isinstance(obj, np.bool_):
                    return bool(obj)
                elif isinstance(obj, dict):
                    return {k: convert_to_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, (list, tuple)):
                    return [convert_to_serializable(i) for i in obj]
                elif isinstance(obj, (int, float, str, bool)):
                    return obj
                elif hasattr(obj, 'item'):  # 处理0维数组或单元素tensor
                    return obj.item()
                else:
                    try:
                        return float(obj)
                    except (TypeError, ValueError):
                        return str(obj)
                    
            serializable_explanation = convert_to_serializable(explanation)
            
            with open(report_path, 'w') as f:
                json.dump(serializable_explanation, f, indent=2)
            print(f"   ✓ JSON报告已保存: {report_path}")
            
            # 生成文本摘要
            summary_path = save_dir / f'{sample_id}_summary.txt'
            self._generate_text_summary(explanation, summary_path)
            print(f"   ✓ 文本摘要已保存: {summary_path}")
            
        print(f"\n{'='*80}")
        print(f"✅ 可解释性分析完成!")
        print(f"{'='*80}\n")
        
        return explanation
    
    def _plot_atom_importance_2d(self, atoms_object, importance_scores, save_path=None):
        """2D原子重要性可视化"""
        coords = atoms_object.cart_coords
        elements = list(atoms_object.elements)
        
        # 归一化
        imp_norm = (importance_scores - importance_scores.min()) / \
                   (importance_scores.max() - importance_scores.min() + 1e-8)
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        projections = [
            (0, 1, 'X', 'Y', 'X-Y Projection'),
            (0, 2, 'X', 'Z', 'X-Z Projection'),
            (1, 2, 'Y', 'Z', 'Y-Z Projection')
        ]
        
        for ax, (i, j, xlabel, ylabel, title) in zip(axes, projections):
            scatter = ax.scatter(
                coords[:, i], coords[:, j],
                c=imp_norm, cmap='YlOrRd', s=300, alpha=0.8,
                edgecolors='black', linewidth=1.5
            )
            
            # 标注元素
            for k, (coord, elem) in enumerate(zip(coords, elements)):
                ax.annotate(
                    f"{elem}",
                    (coord[i], coord[j]),
                    fontsize=9, fontweight='bold',
                    ha='center', va='center'
                )
                
            ax.set_xlabel(f'{xlabel} (Å)', fontsize=11)
            ax.set_ylabel(f'{ylabel} (Å)', fontsize=11)
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='--')
            
        # 添加颜色条
        cbar = plt.colorbar(scatter, ax=axes, shrink=0.8, label='Importance')
        
        plt.suptitle('Atom Importance - Spatial Distribution', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        plt.close()
        
    def _generate_text_summary(self, explanation, save_path):
        """生成文本摘要"""
        lines = []
        lines.append("=" * 80)
        lines.append("可解释性分析报告")
        lines.append("=" * 80)
        lines.append("")
        
        # 预测信息
        lines.append("【预测结果】")
        lines.append(f"  预测值: {explanation.get('prediction', 'N/A')}")
        if explanation.get('true_value') is not None:
            lines.append(f"  真实值: {explanation['true_value']}")
            lines.append(f"  误差: {explanation.get('error', 'N/A')}")
        lines.append("")
        
        # 不确定性
        if 'uncertainty' in explanation:
            lines.append("【不确定性估计】")
            unc = explanation['uncertainty']
            lines.append(f"  预测均值: {unc.get('mean', 'N/A')}")
            lines.append(f"  标准差: {unc.get('std', 'N/A')}")
            if unc.get('confidence_interval_95'):
                ci = unc['confidence_interval_95']
                lines.append(f"  95%置信区间: [{ci[0]:.4f}, {ci[1]:.4f}]")
            lines.append("")
            
        # 模态贡献
        if 'modal_contribution' in explanation:
            lines.append("【模态贡献】")
            mc = explanation['modal_contribution']
            lines.append(f"  图模态: {mc.get('graph_contribution', 0):.1%}")
            lines.append(f"  文本模态: {mc.get('text_contribution', 0):.1%}")
            if 'feature_alignment' in mc:
                lines.append(f"  特征对齐度: {mc['feature_alignment']:.3f}")
            lines.append("")
            
        # 物理关联
        if 'physics_correlations' in explanation and explanation['physics_correlations']:
            lines.append("【物理化学关联】")
            for prop, corr in sorted(explanation['physics_correlations'].items(), 
                                    key=lambda x: abs(x[1]), reverse=True):
                lines.append(f"  {prop}: {corr:+.3f}")
            lines.append("")
            
        # 结构分析
        if 'structure_analysis' in explanation:
            lines.append("【结构分析】")
            sa = explanation['structure_analysis']
            lines.append(f"  原子数: {sa.get('num_atoms', 'N/A')}")
            lines.append(f"  元素种类: {sa.get('num_elements', 'N/A')}")
            if sa.get('most_important_element'):
                elem, imp = sa['most_important_element']
                lines.append(f"  最重要元素: {elem} (重要性: {imp:.4f})")
            lines.append("")
            
        lines.append("=" * 80)
        
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

    def extract_symbolic_features(self, test_loader, save_dir=None, max_samples=None):
        """
        提取特征用于符号回归分析

        使用PySR库从模型特征中发现可解释的符号公式，将深度学习模型的
        黑盒预测转化为可理解的数学表达式。

        Args:
            test_loader: 测试数据加载器
            save_dir: 保存目录（可选）
            max_samples: 最大样本数（None表示使用所有样本）

        Returns:
            model_sr: 训练好的PySR回归器
            results: 包含公式和评估指标的字典
        """
        try:
            import pysr
        except ImportError:
            print("⚠️ PySR未安装。请运行: pip install pysr")
            print("   注意: PySR需要Julia环境。详见: https://github.com/MilesCranmer/PySR")
            return None, None

        print("\n" + "="*80)
        print("🔬 符号回归分析 - 从神经网络特征中发现数学公式")
        print("="*80)

        all_features = []
        all_targets = []
        all_predictions = []

        print("\n📊 [1/3] 提取模型特征...")

        self.model.eval()
        sample_count = 0

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="提取特征"):
                if max_samples is not None and sample_count >= max_samples:
                    break

                g, lg, text, target = batch

                # 前向传播，获取特征
                output = self.model(
                    [g.to(self.device), lg.to(self.device), text],
                    return_features=True
                )

                # 提取图特征（最具代表性的结构特征）
                if isinstance(output, dict):
                    # 优先使用融合后的图特征
                    if 'graph_features' in output and output['graph_features'] is not None:
                        features = output['graph_features'].cpu().numpy()
                    elif 'graph_cross' in output and output['graph_cross'] is not None:
                        features = output['graph_cross'].cpu().numpy()
                    else:
                        print("⚠️ 未找到图特征，跳过此批次")
                        continue

                    # 获取预测
                    pred = output['predictions'].cpu().numpy()
                else:
                    print("⚠️ 模型输出格式不支持特征提取")
                    continue

                all_features.append(features)
                all_targets.append(target.numpy())
                all_predictions.append(pred)
                sample_count += len(target)

        if len(all_features) == 0:
            print("❌ 未提取到任何特征！")
            return None, None

        # 合并所有批次
        X = np.vstack(all_features)
        y = np.concatenate(all_targets)
        y_pred_nn = np.concatenate(all_predictions)

        print(f"   ✓ 提取了 {len(y)} 个样本")
        print(f"   ✓ 特征维度: {X.shape[1]}")

        # ==================== 符号回归 ====================
        print("\n🧮 [2/3] 运行符号回归...")
        print("   这可能需要几分钟时间，请耐心等待...")

        # 配置PySR
        model_sr = pysr.PySRRegressor(
            niterations=100,  # 迭代次数
            binary_operators=["+", "-", "*", "/", "^"],  # 二元运算符
            unary_operators=[
                "exp",   # 指数
                "log",   # 对数
                "sqrt",  # 平方根
                "abs",   # 绝对值
            ],
            maxsize=20,  # 最大公式复杂度
            populations=15,  # 种群数量
            population_size=33,  # 每个种群的大小
            ncyclesperiteration=550,  # 每次迭代的循环数
            # 损失函数：平衡准确性和复杂度
            parsimony=0.0032,  # 简洁性惩罚
            # 特征选择
            select_k_features=min(10, X.shape[1]),  # 自动选择最重要的k个特征
            # 输出设置
            verbosity=1,  # 显示进度
            progress=True,  # 显示进度条
            # 性能优化
            turbo=True,  # 加速模式
            precision=32,  # 使用32位精度
        )

        # 拟合符号回归模型
        try:
            model_sr.fit(X, y)

            print("\n" + "="*80)
            print("📝 发现的符号公式:")
            print("="*80)

            # 获取最佳公式
            equations = model_sr.equations_

            # 显示前5个最佳公式
            print("\n前5个候选公式（按复杂度-准确度权衡排序）:")
            print("-"*80)

            for i, row in equations.head(5).iterrows():
                print(f"\n公式 {i+1}:")
                print(f"  表达式: {row['equation']}")
                print(f"  复杂度: {row['complexity']}")
                print(f"  损失: {row['loss']:.6f}")
                if 'score' in row:
                    print(f"  评分: {row['score']:.6f}")

            # 使用sympy显示最佳公式
            print("\n" + "="*80)
            print("🎯 最佳公式 (SymPy格式):")
            print("="*80)
            try:
                best_formula = model_sr.sympy()
                print(f"\n{best_formula}\n")
            except Exception as e:
                print(f"⚠️ 无法转换为SymPy格式: {e}")

            # ==================== 评估符号回归模型 ====================
            print("\n📊 [3/3] 评估符号回归模型...")

            # 使用符号回归模型预测
            y_pred_sr = model_sr.predict(X)

            # 计算指标
            from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

            mae = mean_absolute_error(y, y_pred_sr)
            mse = mean_squared_error(y, y_pred_sr)
            rmse = np.sqrt(mse)
            r2 = r2_score(y, y_pred_sr)

            # 与神经网络比较
            mae_nn = mean_absolute_error(y, y_pred_nn)
            r2_nn = r2_score(y, y_pred_nn)

            print("\n符号回归模型性能:")
            print(f"  MAE:  {mae:.4f}")
            print(f"  RMSE: {rmse:.4f}")
            print(f"  R²:   {r2:.4f}")

            print("\n与神经网络对比:")
            print(f"  神经网络 MAE: {mae_nn:.4f}")
            print(f"  神经网络 R²:  {r2_nn:.4f}")
            print(f"  MAE 比率:     {mae/mae_nn:.2%} (越小越好)")
            print(f"  R² 差距:      {r2 - r2_nn:+.4f}")

            # 组装结果
            results = {
                'best_formula': str(best_formula) if 'best_formula' in locals() else None,
                'equations_df': equations.to_dict('records') if equations is not None else None,
                'metrics': {
                    'mae': float(mae),
                    'rmse': float(rmse),
                    'r2': float(r2),
                },
                'nn_comparison': {
                    'mae_nn': float(mae_nn),
                    'r2_nn': float(r2_nn),
                    'mae_ratio': float(mae/mae_nn),
                    'r2_diff': float(r2 - r2_nn),
                },
                'feature_dim': int(X.shape[1]),
                'num_samples': int(len(y)),
            }

            # ==================== 保存结果 ====================
            if save_dir:
                save_dir = Path(save_dir)
                save_dir.mkdir(parents=True, exist_ok=True)

                # 保存公式
                formula_path = save_dir / 'symbolic_regression_formulas.txt'
                with open(formula_path, 'w') as f:
                    f.write("="*80 + "\n")
                    f.write("符号回归发现的公式\n")
                    f.write("="*80 + "\n\n")

                    if 'best_formula' in locals():
                        f.write(f"最佳公式:\n{best_formula}\n\n")

                    f.write("所有候选公式:\n")
                    f.write("-"*80 + "\n")
                    for i, row in equations.iterrows():
                        f.write(f"\n公式 {i+1}:\n")
                        f.write(f"  {row['equation']}\n")
                        f.write(f"  复杂度: {row['complexity']}, 损失: {row['loss']:.6f}\n")

                print(f"\n   ✓ 公式已保存: {formula_path}")

                # 保存详细结果
                results_path = save_dir / 'symbolic_regression_results.json'
                with open(results_path, 'w') as f:
                    json.dump(results, f, indent=2, default=str)

                print(f"   ✓ 结果已保存: {results_path}")

                # 保存模型
                try:
                    model_path = save_dir / 'symbolic_regression_model.pkl'
                    import pickle
                    with open(model_path, 'wb') as f:
                        pickle.dump(model_sr, f)
                    print(f"   ✓ 模型已保存: {model_path}")
                except Exception as e:
                    print(f"   ⚠️ 模型保存失败: {e}")

            print("\n" + "="*80)
            print("✅ 符号回归分析完成!")
            print("="*80 + "\n")

            return model_sr, results

        except Exception as e:
            print(f"\n❌ 符号回归失败: {e}")
            import traceback
            traceback.print_exc()
            return None, None

    def batch_explain(self, data_loader, atoms_list, save_dir, max_samples=50):
        """
        批量解释
        
        Args:
            data_loader: 数据加载器
            atoms_list: Atoms对象列表
            save_dir: 保存目录
            max_samples: 最大样本数
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        all_explanations = []
        
        for i, (batch, atoms) in enumerate(tqdm(zip(data_loader, atoms_list), 
                                                  total=min(max_samples, len(atoms_list)),
                                                  desc="批量分析")):
            if i >= max_samples:
                break
                
            g, lg, text, target = batch
            
            explanation = self.explain_prediction(
                g, lg, text, atoms,
                true_value=target.item() if target.numel() == 1 else None,
                save_dir=save_dir / f'sample_{i}',
                sample_id=f'sample_{i}'
            )
            
            all_explanations.append(explanation)
            
        # 保存汇总
        summary_path = save_dir / 'batch_summary.json'
        with open(summary_path, 'w') as f:
            json.dump({
                'num_samples': len(all_explanations),
                'explanations': all_explanations
            }, f, indent=2, default=str)
            
        print(f"\n✅ 批量分析完成! 共 {len(all_explanations)} 个样本")
        print(f"   结果保存在: {save_dir}")
        
        return all_explanations


# ==================== 使用示例 ====================

def demo_usage():
    """演示用法"""
    print("""
    =========================================================
    增强可解释性分析模块 v2.0 使用示例
    =========================================================
    
    # 初始化
    from interpretability_enhanced_v2 import ComprehensiveExplainer
    
    explainer = ComprehensiveExplainer(
        model=trained_model,
        tokenizer=tokenizer,  # 可选
        device='cuda'
    )
    
    # 单样本分析
    explanation = explainer.explain_prediction(
        g=graph,
        lg=line_graph,
        text=["Material description..."],
        atoms_object=atoms,
        true_value=1.5,
        save_dir='./explanations',
        sample_id='sample_001'
    )
    
    # 批量分析
    explanations = explainer.batch_explain(
        data_loader=test_loader,
        atoms_list=test_atoms,
        save_dir='./batch_explanations',
        max_samples=100
    )
    
    # 单独使用各分析器
    
    ## 原子重要性
    atom_analyzer = AtomImportanceAnalyzer(model)
    importance, gradients = atom_analyzer.integrated_gradients(g, lg, text)
    
    ## 跨模态分析
    cross_modal = CrossModalInteractionAnalyzer(model)
    contributions = cross_modal.analyze_modal_contribution(g, lg, text)
    
    ## 物理关联分析
    physics = PhysicsCorrelationAnalyzer()
    correlations = physics.correlate_importance_with_physics(elements, importance)
    
    ## 不确定性估计
    uncertainty = UncertaintyEstimator(model)
    mean, std, samples = uncertainty.mc_dropout_uncertainty(g, lg, text)

    ## 符号回归分析 (新功能!)
    # 从神经网络特征中发现可解释的数学公式
    model_sr, results = explainer.extract_symbolic_features(
        test_loader=test_loader,
        save_dir='./symbolic_regression',
        max_samples=500  # 可选：限制样本数以加快速度
    )

    # 使用发现的符号公式进行预测
    if model_sr is not None:
        # 提取新样本的特征
        new_features = model.get_features(new_g, new_lg, new_text)
        # 使用符号公式预测
        symbolic_prediction = model_sr.predict(new_features)
        print(f"符号公式预测: {symbolic_prediction}")

    =========================================================
    """)


if __name__ == "__main__":
    demo_usage()
