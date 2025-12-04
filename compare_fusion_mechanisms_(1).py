#!/usr/bin/env python
"""
对比不同融合机制的效果
通过消融实验直观展示各个模块的作用
版本2: 使用return_intermediate_features参数，避免动态修改模型架构
"""

import os
import sys
import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, davies_bouldin_score, r2_score, mean_absolute_error, mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

from models.alignn import ALIGNN, ALIGNNConfig
from data import get_train_val_loaders

sns.set_style("whitegrid")
plt.rcParams['font.size'] = 10


def centered_kernel_alignment(X, Y):
    """
    计算 CKA (Centered Kernel Alignment) 相似度

    Args:
        X: 特征矩阵1 [N, D1]
        Y: 特征矩阵2 [N, D2]

    Returns:
        CKA score (0-1之间，越高越相似)
    """
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)
    K = X @ X.T
    L = Y @ Y.T
    hsic = np.sum(K * L)
    denom = np.sqrt(np.sum(K * K) * np.sum(L * L))
    return hsic / denom if denom > 0 else 0.0


class FusionComparator:
    """融合机制对比器"""

    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.model.eval()

    def extract_features_ablation(self, data_loader, max_samples=None):
        """
        提取不同阶段的特征（消融实验）

        Returns:
            features_dict: {
                'graph_base': 图基础特征（投影后，融合前）,
                'text_base': 文本基础特征（投影后，融合前）,
                'graph_cross': 应用全局注意力后的图特征（如果启用）,
                'text_cross': 应用全局注意力后的文本特征（如果启用）,
                'graph_final': 最终图特征,
                'text_final': 最终文本特征,
                'fused': 最终融合特征
            }
            targets: 目标值
            ids: 样本ID
        """
        print("🔄 提取不同阶段的特征（消融实验）...")

        # 检查模型配置
        has_middle = self.model.use_middle_fusion
        has_fine = self.model.use_fine_grained_attention
        has_cross = self.model.use_cross_modal_attention

        print(f"   模型配置: 中间融合={has_middle}, 细粒度注意力={has_fine}, 全局注意力={has_cross}")

        features_dict = {
            'graph_base': [],
            'text_base': [],
            'graph_middle': [],
            'graph_fine': [],
            'text_fine': [],
            'graph_cross': [],
            'text_cross': [],
            'graph_final': [],
            'text_final': [],
            'fused': []
        }
        targets = []
        ids = []

        sample_count = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(data_loader, desc="提取特征")):
                if len(batch) == 3:
                    g, text, target = batch
                    lg = None
                elif len(batch) == 4:
                    g, lg, text, target = batch
                else:
                    raise ValueError(f"不支持的batch格式: {len(batch)}个元素")

                g = g.to(self.device)
                if lg is not None:
                    lg = lg.to(self.device)

                # 处理text
                if isinstance(text, dict):
                    text = {k: v.to(self.device) for k, v in text.items()}
                elif isinstance(text, (list, tuple)):
                    # text是字符串列表，保持不动
                    pass
                elif torch.is_tensor(text):
                    text = text.to(self.device)

                batch_size = target.size(0)

                # 构建模型输入
                if lg is not None:
                    model_input = (g, lg, text)
                else:
                    model_input = (g, text)

                # 提取中间特征（使用新的return_intermediate_features参数）
                output = self.model(model_input, return_intermediate_features=True)

                # 基础特征（融合前）
                features_dict['graph_base'].append(output['graph_base'].cpu().numpy())
                features_dict['text_base'].append(output['text_base'].cpu().numpy())

                # 中间融合后的特征（如果启用）
                if has_middle and 'graph_middle' in output:
                    features_dict['graph_middle'].append(output['graph_middle'].cpu().numpy())

                # 细粒度注意力后的特征（如果启用）
                if has_fine and 'graph_fine' in output:
                    features_dict['graph_fine'].append(output['graph_fine'].cpu().numpy())
                    features_dict['text_fine'].append(output['text_fine'].cpu().numpy())

                # 全局注意力后的特征（如果启用）
                if has_cross and 'graph_cross' in output:
                    features_dict['graph_cross'].append(output['graph_cross'].cpu().numpy())
                    features_dict['text_cross'].append(output['text_cross'].cpu().numpy())

                # 最终特征
                features_dict['graph_final'].append(output['graph_features'].cpu().numpy())
                features_dict['text_final'].append(output['text_features'].cpu().numpy())

                # 融合特征
                fused = np.concatenate([
                    output['graph_features'].cpu().numpy(),
                    output['text_features'].cpu().numpy()
                ], axis=1)
                features_dict['fused'].append(fused)

                targets.append(target.cpu().numpy())

                # 记录样本ID（如果有）
                if hasattr(g, 'ndata') and 'jid' in g.ndata:
                    batch_ids = [g.ndata['jid'][i] for i in range(g.batch_size)]
                    ids.extend(batch_ids)

                sample_count += batch_size
                if max_samples and sample_count >= max_samples:
                    break

        # 转换为numpy数组
        for key in features_dict:
            if len(features_dict[key]) > 0:
                features_dict[key] = np.concatenate(features_dict[key], axis=0)
            else:
                features_dict[key] = None

        targets = np.concatenate(targets, axis=0)

        # 移除空特征
        features_dict = {k: v for k, v in features_dict.items() if v is not None}

        print(f"✅ 提取完成! 样本数: {len(targets)}, 特征类型: {list(features_dict.keys())}")

        return features_dict, targets, ids

    def visualize_tsne(self, features_dict, targets, save_dir):
        """使用t-SNE可视化不同阶段的特征"""
        print("\n📊 生成t-SNE可视化...")

        # 确定要可视化的特征
        feature_names = []
        feature_data = []

        if 'graph_base' in features_dict:
            feature_names.append('Graph Base')
            feature_data.append(features_dict['graph_base'])

        if 'text_base' in features_dict:
            feature_names.append('Text Base')
            feature_data.append(features_dict['text_base'])

        if 'graph_middle' in features_dict:
            feature_names.append('Graph + Middle Fusion')
            feature_data.append(features_dict['graph_middle'])

        if 'graph_fine' in features_dict:
            feature_names.append('Graph + Fine-grained Attn')
            feature_data.append(features_dict['graph_fine'])

        if 'text_fine' in features_dict:
            feature_names.append('Text + Fine-grained Attn')
            feature_data.append(features_dict['text_fine'])

        if 'graph_cross' in features_dict:
            feature_names.append('Graph + Cross-Modal')
            feature_data.append(features_dict['graph_cross'])

        if 'text_cross' in features_dict:
            feature_names.append('Text + Cross-Modal')
            feature_data.append(features_dict['text_cross'])

        if 'graph_final' in features_dict:
            feature_names.append('Graph Final')
            feature_data.append(features_dict['graph_final'])

        if 'text_final' in features_dict:
            feature_names.append('Text Final')
            feature_data.append(features_dict['text_final'])

        if 'fused' in features_dict:
            feature_names.append('Fused')
            feature_data.append(features_dict['fused'])

        n_features = len(feature_names)
        if n_features == 0:
            print("⚠️  没有可视化的特征!")
            return

        # 先对所有特征进行t-SNE，收集所有坐标用于统一坐标轴范围
        print("   第一步: 计算所有t-SNE嵌入...")
        all_features_2d = []
        for name, features in zip(feature_names, feature_data):
            tsne = TSNE(n_components=2, random_state=42, perplexity=30)
            features_2d = tsne.fit_transform(features)
            all_features_2d.append(features_2d)

        # 计算全局坐标范围
        all_coords = np.vstack(all_features_2d)
        x_min, x_max = all_coords[:, 0].min(), all_coords[:, 0].max()
        y_min, y_max = all_coords[:, 1].min(), all_coords[:, 1].max()

        # 添加边距
        x_margin = (x_max - x_min) * 0.05
        y_margin = (y_max - y_min) * 0.05
        x_lim = [x_min - x_margin, x_max + x_margin]
        y_lim = [y_min - y_margin, y_max + y_margin]

        print(f"   全局坐标范围: x=[{x_lim[0]:.1f}, {x_lim[1]:.1f}], y=[{y_lim[0]:.1f}, {y_lim[1]:.1f}]")

        # 创建网格布局
        n_cols = min(3, n_features)
        n_rows = (n_features + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
        if n_features == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if n_features > 1 else [axes]

        # 第二步: 绘制每个特征
        print("   第二步: 绘制可视化...")
        for idx, (name, features_2d) in enumerate(zip(feature_names, all_features_2d)):
            ax = axes[idx]
            scatter = ax.scatter(features_2d[:, 0], features_2d[:, 1],
                                c=targets, cmap='viridis', alpha=0.6, s=20)
            ax.set_title(name, fontsize=12, fontweight='bold')
            ax.set_xlabel('t-SNE 1')
            ax.set_ylabel('t-SNE 2')

            # 设置统一的坐标范围
            ax.set_xlim(x_lim)
            ax.set_ylim(y_lim)

            # 设置白色网格线
            ax.grid(True, color='white', linewidth=0.8, alpha=0.7)

            plt.colorbar(scatter, ax=ax, label='Target Value')

        # 隐藏多余的子图
        for idx in range(n_features, len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()
        save_path = os.path.join(save_dir, 'tsne_comparison.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ t-SNE可视化已保存: {save_path}")
        plt.close()

    def compute_metrics(self, features_dict, targets, save_dir):
        """计算不同特征的质量指标"""
        print("\n📈 计算特征质量指标...")

        metrics_list = []

        for name, features in features_dict.items():
            if features is None or len(features) == 0:
                continue

            print(f"   分析 {name}...")

            # Silhouette Score (轮廓系数, 越大越好)
            try:
                sil_score = silhouette_score(features, targets)
            except:
                sil_score = np.nan

            # Davies-Bouldin Index (越小越好)
            try:
                db_score = davies_bouldin_score(features, targets)
            except:
                db_score = np.nan

            # Intra-class similarity (类内相似度, 越大越好)
            intra_sim = self._compute_intra_class_similarity(features, targets)

            # Inter-class separation (类间分离度, 越大越好)
            inter_sep = self._compute_inter_class_separation(features, targets)

            metrics_list.append({
                'Feature': name,
                'Silhouette Score': sil_score,
                'Davies-Bouldin Index': db_score,
                'Intra-class Similarity': intra_sim,
                'Inter-class Separation': inter_sep
            })

        # 创建DataFrame
        df = pd.DataFrame(metrics_list)
        save_path = os.path.join(save_dir, 'feature_metrics.csv')
        df.to_csv(save_path, index=False)
        print(f"\n✅ 指标已保存: {save_path}")
        print("\n" + df.to_string(index=False))

        # 可视化指标
        self._plot_metrics(df, save_dir)

        return df

    def _compute_intra_class_similarity(self, features, targets):
        """计算类内相似度"""
        unique_targets = np.unique(targets)
        if len(unique_targets) < 2:
            return 1.0

        sims = []
        for target in unique_targets[:10]:  # 只取前10个类别避免计算过慢
            mask = targets == target
            if np.sum(mask) < 2:
                continue
            class_features = features[mask]
            sim_matrix = cosine_similarity(class_features)
            # 取上三角（不包括对角线）
            upper_tri = sim_matrix[np.triu_indices_from(sim_matrix, k=1)]
            sims.append(np.mean(upper_tri))

        return np.mean(sims) if len(sims) > 0 else 0.0

    def _compute_inter_class_separation(self, features, targets):
        """计算类间分离度"""
        unique_targets = np.unique(targets)
        if len(unique_targets) < 2:
            return 0.0

        # 计算每个类别的中心
        centroids = []
        for target in unique_targets[:10]:  # 只取前10个类别
            mask = targets == target
            if np.sum(mask) == 0:
                continue
            centroids.append(np.mean(features[mask], axis=0))

        if len(centroids) < 2:
            return 0.0

        centroids = np.array(centroids)
        # 计算中心之间的平均距离
        distances = []
        for i in range(len(centroids)):
            for j in range(i+1, len(centroids)):
                dist = np.linalg.norm(centroids[i] - centroids[j])
                distances.append(dist)

        return np.mean(distances)

    def _plot_metrics(self, df, save_dir):
        """可视化指标对比"""
        metrics = ['Silhouette Score', 'Davies-Bouldin Index',
                   'Intra-class Similarity', 'Inter-class Separation']

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()

        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            data = df[['Feature', metric]].dropna()

            if len(data) == 0:
                continue

            x = range(len(data))
            y = data[metric].values
            labels = data['Feature'].values

            bars = ax.bar(x, y, alpha=0.7, color=sns.color_palette("husl", len(data)))
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=45, ha='right')
            ax.set_ylabel(metric)
            ax.set_title(f'{metric} Comparison', fontweight='bold')
            ax.grid(axis='y', alpha=0.3)

            # 标注数值
            for i, v in enumerate(y):
                ax.text(i, v + 0.01*max(y), f'{v:.3f}', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        save_path = os.path.join(save_dir, 'metrics_comparison.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ 指标对比图已保存: {save_path}")
        plt.close()

    def compute_regression_metrics(self, features_dict, targets, save_dir):
        """计算回归任务的特征质量指标"""
        print("\n📊 计算回归任务指标...")

        metrics_list = []

        for name, features in features_dict.items():
            if features is None or len(features) == 0:
                continue

            print(f"   分析 {name}...")

            # 1. 特征-目标相关性 (Pearson)
            feature_dim = features.shape[1]
            pearson_correlations = []
            for i in range(feature_dim):
                try:
                    corr, _ = pearsonr(features[:, i], targets)
                    if not np.isnan(corr):
                        pearson_correlations.append(abs(corr))
                except:
                    pass

            avg_pearson = np.mean(pearson_correlations) if len(pearson_correlations) > 0 else 0.0
            max_pearson = np.max(pearson_correlations) if len(pearson_correlations) > 0 else 0.0

            # 2. 特征-目标相关性 (Spearman)
            spearman_correlations = []
            for i in range(feature_dim):
                try:
                    corr, _ = spearmanr(features[:, i], targets)
                    if not np.isnan(corr):
                        spearman_correlations.append(abs(corr))
                except:
                    pass

            avg_spearman = np.mean(spearman_correlations) if len(spearman_correlations) > 0 else 0.0

            # 3. 特征方差 (表示特征的表达能力)
            feature_variance = np.mean(np.var(features, axis=0))

            # 4. 特征标准差
            feature_std = np.mean(np.std(features, axis=0))

            # 5. 特征范数
            feature_norm = np.mean(np.linalg.norm(features, axis=1))

            metrics_list.append({
                'Feature': name,
                'Avg Pearson Corr': avg_pearson,
                'Max Pearson Corr': max_pearson,
                'Avg Spearman Corr': avg_spearman,
                'Feature Variance': feature_variance,
                'Feature Std': feature_std,
                'Feature Norm': feature_norm
            })

        # 创建DataFrame
        df = pd.DataFrame(metrics_list)
        save_path = os.path.join(save_dir, 'regression_metrics.csv')
        df.to_csv(save_path, index=False)
        print(f"\n✅ 回归指标已保存: {save_path}")
        print("\n" + df.to_string(index=False))

        # 可视化回归指标
        self._plot_regression_metrics(df, save_dir)

        return df

    def _plot_regression_metrics(self, df, save_dir):
        """可视化回归指标对比"""
        metrics = ['Avg Pearson Corr', 'Max Pearson Corr',
                   'Avg Spearman Corr', 'Feature Variance']

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()

        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            data = df[['Feature', metric]].dropna()

            if len(data) == 0:
                continue

            x = range(len(data))
            y = data[metric].values
            labels = data['Feature'].values

            # 使用颜色区分性能
            colors = sns.color_palette("RdYlGn", len(data))
            if metric in ['Avg Pearson Corr', 'Max Pearson Corr', 'Avg Spearman Corr']:
                # 相关性越高越好，排序后上色
                sorted_indices = np.argsort(y)
                bar_colors = [colors[np.where(sorted_indices == i)[0][0]] for i in range(len(y))]
            else:
                bar_colors = sns.color_palette("husl", len(data))

            bars = ax.bar(x, y, alpha=0.7, color=bar_colors)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=45, ha='right')
            ax.set_ylabel(metric)
            ax.set_title(f'{metric} Comparison', fontweight='bold')
            ax.grid(axis='y', alpha=0.3, color='white', linewidth=0.8)

            # 标注数值
            for i, v in enumerate(y):
                ax.text(i, v + 0.01*max(abs(y)), f'{v:.4f}',
                       ha='center', va='bottom', fontsize=9)

            # 添加参考线
            if metric in ['Avg Pearson Corr', 'Max Pearson Corr', 'Avg Spearman Corr']:
                ax.axhline(y=0.3, color='orange', linestyle='--',
                          linewidth=1, alpha=0.5, label='Moderate (0.3)')
                ax.axhline(y=0.5, color='red', linestyle='--',
                          linewidth=1, alpha=0.5, label='Strong (0.5)')
                ax.legend(fontsize=8)

        plt.tight_layout()
        save_path = os.path.join(save_dir, 'regression_metrics_comparison.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ 回归指标对比图已保存: {save_path}")
        plt.close()

    def compute_cka_matrix(self, features_dict, save_dir):
        """
        计算所有特征对之间的 CKA 相似度矩阵

        Args:
            features_dict: 特征字典
            save_dir: 保存目录

        Returns:
            CKA 矩阵 DataFrame
        """
        print("\n🔍 计算 CKA 相似度矩阵...")

        # 获取所有有效特征名
        feature_names = [name for name, feats in features_dict.items()
                        if feats is not None and len(feats) > 0]

        if len(feature_names) < 2:
            print("⚠️  特征数量不足，无法计算 CKA 矩阵")
            return None

        # 初始化 CKA 矩阵
        n_features = len(feature_names)
        cka_matrix = np.zeros((n_features, n_features))

        # 计算所有特征对的 CKA
        for i, name_i in enumerate(feature_names):
            for j, name_j in enumerate(feature_names):
                if i == j:
                    cka_matrix[i, j] = 1.0
                elif i < j:
                    print(f"   计算 CKA: {name_i} vs {name_j}")
                    cka_score = centered_kernel_alignment(
                        features_dict[name_i],
                        features_dict[name_j]
                    )
                    cka_matrix[i, j] = cka_score
                    cka_matrix[j, i] = cka_score  # 对称矩阵

        # 创建 DataFrame
        cka_df = pd.DataFrame(cka_matrix,
                             index=feature_names,
                             columns=feature_names)

        # 保存为 CSV
        save_path = os.path.join(save_dir, 'cka_similarity_matrix.csv')
        cka_df.to_csv(save_path)
        print(f"\n✅ CKA 矩阵已保存: {save_path}")
        print("\n" + cka_df.to_string())

        return cka_df

    def visualize_cka_matrix(self, cka_df, save_dir):
        """
        可视化 CKA 相似度矩阵

        Args:
            cka_df: CKA 矩阵 DataFrame
            save_dir: 保存目录
        """
        print("\n📊 生成 CKA 相似度热图...")

        if cka_df is None or len(cka_df) == 0:
            print("⚠️  没有可视化的 CKA 数据")
            return

        # 创建图形
        fig, ax = plt.subplots(figsize=(10, 8))

        # 绘制热图
        sns.heatmap(cka_df,
                   annot=True,  # 显示数值
                   fmt='.3f',   # 保留3位小数
                   cmap='RdYlGn',  # 红黄绿配色
                   vmin=0.0,
                   vmax=1.0,
                   center=0.5,
                   square=True,
                   linewidths=0.5,
                   cbar_kws={'label': 'CKA Similarity'},
                   ax=ax)

        ax.set_title('CKA Similarity Matrix Between Different Fusion Stages',
                    fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel('Features', fontsize=12, fontweight='bold')
        ax.set_ylabel('Features', fontsize=12, fontweight='bold')

        # 旋转标签
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)

        plt.tight_layout()
        save_path = os.path.join(save_dir, 'cka_similarity_heatmap.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ CKA 热图已保存: {save_path}")
        plt.close()

        # 生成 CKA 分数摘要
        self._generate_cka_summary(cka_df, save_dir)

    def _generate_cka_summary(self, cka_df, save_dir):
        """
        生成 CKA 分数摘要报告

        Args:
            cka_df: CKA 矩阵 DataFrame
            save_dir: 保存目录
        """
        print("\n📝 生成 CKA 分数摘要...")

        report_lines = []
        report_lines.append("=" * 70)
        report_lines.append("CKA Similarity Score Summary Report")
        report_lines.append("=" * 70)
        report_lines.append("")

        # 1. 整体统计
        # 提取上三角（不包括对角线）
        n = len(cka_df)
        upper_tri_indices = np.triu_indices(n, k=1)
        upper_tri_values = cka_df.values[upper_tri_indices]

        report_lines.append("📊 Overall Statistics:")
        report_lines.append(f"  • Mean CKA Score: {np.mean(upper_tri_values):.4f}")
        report_lines.append(f"  • Median CKA Score: {np.median(upper_tri_values):.4f}")
        report_lines.append(f"  • Min CKA Score: {np.min(upper_tri_values):.4f}")
        report_lines.append(f"  • Max CKA Score: {np.max(upper_tri_values):.4f}")
        report_lines.append(f"  • Std CKA Score: {np.std(upper_tri_values):.4f}")
        report_lines.append("")

        # 2. 最相似的特征对（Top 5）
        report_lines.append("🔝 Top 5 Most Similar Feature Pairs:")
        similar_pairs = []
        for i in range(n):
            for j in range(i+1, n):
                similar_pairs.append((
                    cka_df.index[i],
                    cka_df.columns[j],
                    cka_df.iloc[i, j]
                ))
        similar_pairs.sort(key=lambda x: x[2], reverse=True)

        for rank, (feat1, feat2, score) in enumerate(similar_pairs[:5], 1):
            report_lines.append(f"  {rank}. {feat1} ↔ {feat2}: {score:.4f}")
        report_lines.append("")

        # 3. 最不相似的特征对（Top 5）
        report_lines.append("🔻 Top 5 Most Dissimilar Feature Pairs:")
        for rank, (feat1, feat2, score) in enumerate(similar_pairs[-5:][::-1], 1):
            report_lines.append(f"  {rank}. {feat1} ↔ {feat2}: {score:.4f}")
        report_lines.append("")

        # 4. 融合阶段的影响分析
        report_lines.append("🔬 Fusion Stage Impact Analysis:")

        # 检查特定的融合阶段对
        stage_pairs = [
            ('graph_base', 'graph_middle', '中期融合的影响'),
            ('graph_middle', 'graph_fine', '细粒度注意力的影响'),
            ('graph_fine', 'graph_cross', '全局注意力的影响'),
            ('graph_cross', 'graph_final', '最终融合的影响'),
            ('graph_base', 'graph_final', '整体融合效果'),
            ('text_base', 'text_final', '文本模态的变化'),
        ]

        for feat1, feat2, description in stage_pairs:
            if feat1 in cka_df.index and feat2 in cka_df.columns:
                score = cka_df.loc[feat1, feat2]
                report_lines.append(f"  • {description}")
                report_lines.append(f"    {feat1} → {feat2}: {score:.4f}")

                # 解释分数
                if score > 0.9:
                    interpretation = "极高相似度 - 融合影响较小"
                elif score > 0.7:
                    interpretation = "高相似度 - 融合保留了主要信息"
                elif score > 0.5:
                    interpretation = "中等相似度 - 融合带来了显著变化"
                else:
                    interpretation = "低相似度 - 融合大幅改变了特征空间"
                report_lines.append(f"    解释: {interpretation}")
                report_lines.append("")

        # 5. 建议
        report_lines.append("💡 Insights and Recommendations:")
        avg_cka = np.mean(upper_tri_values)
        if avg_cka > 0.85:
            report_lines.append("  • 特征空间整体相似度很高，可能存在过度融合")
            report_lines.append("  • 建议: 考虑减少融合层数或调整融合强度")
        elif avg_cka > 0.65:
            report_lines.append("  • 特征空间保持了适度的相似性和差异性")
            report_lines.append("  • 建议: 当前融合机制较为合理")
        else:
            report_lines.append("  • 不同阶段的特征差异较大")
            report_lines.append("  • 建议: 分析是否有过度变换导致信息损失")

        report_lines.append("")
        report_lines.append("=" * 70)

        # 保存报告
        report_text = "\n".join(report_lines)
        save_path = os.path.join(save_dir, 'cka_summary_report.txt')
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(report_text)

        print(f"✅ CKA 摘要报告已保存: {save_path}")
        print("\n" + report_text)


def main():
    parser = argparse.ArgumentParser(description='对比不同融合机制的效果 (v2)')
    parser.add_argument('--checkpoint', type=str, required=True, help='模型checkpoint路径')
    parser.add_argument('--dataset', type=str, required=True,
                        help='数据集类型 (jarvis/mp/class等)')
    parser.add_argument('--property', type=str, required=True,
                        help='目标属性 (如 formation_energy_peratom, bandgap等)')
    parser.add_argument('--root_dir', type=str, default='/public/home/ghzhang/crysmmnet-main/dataset',
                        help='数据集根目录')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--max_samples', type=int, default=500, help='最大样本数（用于快速测试）')
    parser.add_argument('--save_dir', type=str, default='./fusion_comparison',
                        help='结果保存目录')
    args = parser.parse_args()

    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)

    # 加载模型
    print(f"🔄 加载模型: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)

    if 'config' in checkpoint:
        config = checkpoint['config']
    else:
        raise ValueError("Checkpoint中没有找到config")

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   使用设备: {device}")

    # 创建模型
    model = ALIGNN(config)
    model.load_state_dict(checkpoint['model'])
    model = model.to(device)
    model.eval()

    print(f"   模型配置:")
    print(f"     - 中间融合: {model.use_middle_fusion}")
    print(f"     - 细粒度注意力: {model.use_fine_grained_attention}")
    print(f"     - 全局注意力: {model.use_cross_modal_attention}")

    # 加载数据集（支持本地数据）
    print(f"\n🔄 加载数据集: {args.dataset} - {args.property}")
    try:
        from train_with_cross_modal_attention import load_dataset, get_dataset_paths

        # 获取数据集路径
        cif_dir, id_prop_file = get_dataset_paths(args.root_dir, args.dataset, args.property)

        # 加载数据集
        df = load_dataset(cif_dir, id_prop_file, args.dataset, args.property)
        print(f"✅ 加载数据集: {len(df)} 样本")

        # 如果设置了max_samples，进行采样
        if args.max_samples and len(df) > args.max_samples:
            print(f"⚠️  数据集过大，随机采样 {args.max_samples} 样本")
            import random
            random.seed(42)
            df = random.sample(df, args.max_samples)

        # 创建数据加载器（使用本地数据）
        train_loader, val_loader, test_loader, _ = get_train_val_loaders(
            dataset='user_data',  # 使用user_data避免dataset限制
            dataset_array=df,
            target='target',
            n_train=None,
            n_val=None,
            n_test=None,
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1,
            batch_size=args.batch_size,
            atom_features=config.atom_features if hasattr(config, 'atom_features') else 'cgcnn',
            neighbor_strategy='k-nearest',
            line_graph=config.line_graph if hasattr(config, 'line_graph') else True,
            split_seed=42,
            workers=0,
            pin_memory=False,
            save_dataloader=False,
            filename='temp_comparison',
            id_tag='jid',
            use_canonize=True,
            cutoff=8.0,
            max_neighbors=12,
            output_dir=args.save_dir
        )
    except Exception as e:
        print(f"❌ 加载数据集失败: {e}")
        print("请确保:")
        print(f"  1. 数据集路径正确: {args.root_dir}")
        print(f"  2. 数据集类型正确: {args.dataset}")
        print(f"  3. 属性名称正确: {args.property}")
        raise

    print(f"   测试集样本数: {len(test_loader.dataset)}")

    # 创建对比器
    comparator = FusionComparator(model, device=device)

    # 提取特征
    features_dict, targets, ids = comparator.extract_features_ablation(
        test_loader, max_samples=args.max_samples
    )

    # 可视化
    comparator.visualize_tsne(features_dict, targets, args.save_dir)

    # 计算聚类指标
    metrics_df = comparator.compute_metrics(features_dict, targets, args.save_dir)

    # 计算回归指标
    regression_metrics_df = comparator.compute_regression_metrics(features_dict, targets, args.save_dir)

    # 计算 CKA 相似度矩阵
    cka_df = comparator.compute_cka_matrix(features_dict, args.save_dir)

    # 可视化 CKA 矩阵
    if cka_df is not None:
        comparator.visualize_cka_matrix(cka_df, args.save_dir)

    print(f"\n🎉 分析完成! 结果保存在: {args.save_dir}")


if __name__ == '__main__':
    main()
