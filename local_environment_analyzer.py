"""
局部化学环境分析模块

功能：
1. 配位数分析
2. 键长分布分析
3. 键角分析
4. 局部对称性分析
5. Voronoi多面体分析
6. 原子环境指纹

"""

import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class LocalEnvironmentAnalyzer:
    """局部化学环境分析器"""
    
    # 共价半径数据 (pm)
    COVALENT_RADII = {
        'H': 31, 'He': 28, 'Li': 128, 'Be': 96, 'B': 84, 'C': 76, 'N': 71, 'O': 66,
        'F': 57, 'Ne': 58, 'Na': 166, 'Mg': 141, 'Al': 121, 'Si': 111, 'P': 107,
        'S': 105, 'Cl': 102, 'Ar': 106, 'K': 203, 'Ca': 176, 'Sc': 170, 'Ti': 160,
        'V': 153, 'Cr': 139, 'Mn': 139, 'Fe': 132, 'Co': 126, 'Ni': 124, 'Cu': 132,
        'Zn': 122, 'Ga': 122, 'Ge': 120, 'As': 119, 'Se': 120, 'Br': 120, 'Kr': 116,
        'Rb': 220, 'Sr': 195, 'Y': 190, 'Zr': 175, 'Nb': 164, 'Mo': 154, 'Tc': 147,
        'Ru': 146, 'Rh': 142, 'Pd': 139, 'Ag': 145, 'Cd': 144, 'In': 142, 'Sn': 139,
        'Sb': 139, 'Te': 138, 'I': 139, 'Xe': 140, 'Cs': 244, 'Ba': 215, 'La': 207,
        'Ce': 204, 'Pr': 203, 'Nd': 201, 'Pm': 199, 'Sm': 198, 'Eu': 198, 'Gd': 196,
        'Tb': 194, 'Dy': 192, 'Ho': 192, 'Er': 189, 'Tm': 190, 'Yb': 187, 'Lu': 187,
        'Hf': 175, 'Ta': 170, 'W': 162, 'Re': 151, 'Os': 144, 'Ir': 141, 'Pt': 136,
        'Au': 136, 'Hg': 132, 'Tl': 145, 'Pb': 146, 'Bi': 148, 'Po': 140, 'At': 150,
        'Rn': 150, 'Fr': 260, 'Ra': 221, 'Ac': 215, 'Th': 206, 'Pa': 200, 'U': 196,
        'Np': 190, 'Pu': 187, 'Am': 180, 'Cm': 169,
    }
    
    def __init__(self, cutoff: float = 4.0, bond_tolerance: float = 0.2):
        """
        Args:
            cutoff: 搜索邻居的截断距离 (Å)
            bond_tolerance: 键长容差因子（相对于共价半径之和）
        """
        self.cutoff = cutoff
        self.bond_tolerance = bond_tolerance
        
    def analyze_local_environment(self, atoms_object, importance_scores: np.ndarray = None) -> Dict:
        """
        完整的局部化学环境分析
        
        Args:
            atoms_object: JARVIS Atoms对象
            importance_scores: 原子重要性分数（可选）
            
        Returns:
            分析结果字典
        """
        elements = list(atoms_object.elements)
        coords = atoms_object.cart_coords
        lattice = atoms_object.lattice_mat
        
        results = {
            'num_atoms': len(elements),
            'elements': elements,
            'coordination': {},
            'bonds': {},
            'bond_angles': {},
            'local_symmetry': {},
            'environment_fingerprint': {},
        }
        
        # 1. 计算距离矩阵和邻居
        distances, neighbors = self._compute_neighbors(coords, lattice)
        
        # 2. 配位数分析
        coord_analysis = self._analyze_coordination(elements, distances, neighbors)
        results['coordination'] = coord_analysis
        
        # 3. 键长分析
        bond_analysis = self._analyze_bonds(elements, distances, neighbors)
        results['bonds'] = bond_analysis
        
        # 4. 键角分析
        angle_analysis = self._analyze_bond_angles(coords, elements, neighbors)
        results['bond_angles'] = angle_analysis
        
        # 5. 局部对称性分析
        symmetry_analysis = self._analyze_local_symmetry(coords, neighbors)
        results['local_symmetry'] = symmetry_analysis
        
        # 6. 环境指纹
        fingerprint = self._compute_environment_fingerprint(elements, distances, neighbors)
        results['environment_fingerprint'] = fingerprint
        
        # 7. 如果有重要性分数，分析环境与重要性的关系
        if importance_scores is not None:
            env_importance = self._correlate_environment_importance(
                coord_analysis, bond_analysis, importance_scores
            )
            results['environment_importance_correlation'] = env_importance
            
        return results
    
    def _compute_neighbors(self, coords: np.ndarray, lattice: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        计算考虑周期性边界条件的邻居
        
        Returns:
            distances: 距离矩阵
            neighbors: {atom_idx: [(neighbor_idx, distance, image), ...]}
        """
        n_atoms = len(coords)
        distances = np.zeros((n_atoms, n_atoms))
        neighbors = defaultdict(list)
        
        # 考虑周期性边界 - 检查相邻晶胞
        images = []
        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                for k in [-1, 0, 1]:
                    images.append(np.array([i, j, k]))
        
        for i in range(n_atoms):
            for j in range(n_atoms):
                min_dist = float('inf')
                best_image = np.array([0, 0, 0])
                
                for image in images:
                    # 计算周期性镜像的距离
                    shift = image @ lattice
                    dist = np.linalg.norm(coords[i] - coords[j] - shift)
                    
                    if dist < min_dist and (i != j or np.any(image != 0)):
                        min_dist = dist
                        best_image = image.copy()
                        
                if min_dist < self.cutoff and min_dist > 0.1:  # 排除自身和过近的原子
                    distances[i, j] = min_dist
                    neighbors[i].append((j, min_dist, best_image))
                    
        # 按距离排序
        for i in neighbors:
            neighbors[i] = sorted(neighbors[i], key=lambda x: x[1])
            
        return distances, dict(neighbors)
    
    def _analyze_coordination(self, elements: List[str], distances: np.ndarray, 
                             neighbors: Dict) -> Dict:
        """配位数分析"""
        n_atoms = len(elements)
        
        coordination_numbers = []
        coordination_by_element = defaultdict(list)
        first_shell_distances = []
        
        for i in range(n_atoms):
            elem = elements[i]
            
            # 使用共价半径判断成键
            r_i = self.COVALENT_RADII.get(elem, 150) / 100  # 转换为 Å
            
            bonded_neighbors = []
            for j, dist, _ in neighbors.get(i, []):
                r_j = self.COVALENT_RADII.get(elements[j], 150) / 100
                max_bond_length = (r_i + r_j) * (1 + self.bond_tolerance)
                
                if dist <= max_bond_length:
                    bonded_neighbors.append((j, dist))
                    
            cn = len(bonded_neighbors)
            coordination_numbers.append(cn)
            coordination_by_element[elem].append(cn)
            
            if bonded_neighbors:
                first_shell_distances.append(np.mean([d for _, d in bonded_neighbors]))
                
        # 统计
        result = {
            'coordination_numbers': coordination_numbers,
            'mean_coordination': float(np.mean(coordination_numbers)),
            'std_coordination': float(np.std(coordination_numbers)),
            'coordination_by_element': {
                elem: {
                    'mean': float(np.mean(cns)),
                    'std': float(np.std(cns)) if len(cns) > 1 else 0.0,
                    'values': [int(cn) for cn in cns]
                }
                for elem, cns in coordination_by_element.items()
            },
            'mean_first_shell_distance': float(np.mean(first_shell_distances)) if first_shell_distances else 0.0,
        }
        
        return result
    
    def _analyze_bonds(self, elements: List[str], distances: np.ndarray,
                      neighbors: Dict) -> Dict:
        """键长分析"""
        n_atoms = len(elements)
        
        bond_lengths = defaultdict(list)
        all_bonds = []
        
        for i in range(n_atoms):
            elem_i = elements[i]
            r_i = self.COVALENT_RADII.get(elem_i, 150) / 100
            
            for j, dist, _ in neighbors.get(i, []):
                if j > i:  # 避免重复计算
                    elem_j = elements[j]
                    r_j = self.COVALENT_RADII.get(elem_j, 150) / 100
                    max_bond_length = (r_i + r_j) * (1 + self.bond_tolerance)
                    
                    if dist <= max_bond_length:
                        # 生成有序的键类型标签
                        bond_type = '-'.join(sorted([elem_i, elem_j]))
                        bond_lengths[bond_type].append(dist)
                        all_bonds.append({
                            'atoms': (i, j),
                            'elements': (elem_i, elem_j),
                            'length': float(dist),
                            'type': bond_type
                        })
                        
        # 统计各类型键长
        bond_statistics = {}
        for bond_type, lengths in bond_lengths.items():
            bond_statistics[bond_type] = {
                'count': len(lengths),
                'mean': float(np.mean(lengths)),
                'std': float(np.std(lengths)) if len(lengths) > 1 else 0.0,
                'min': float(np.min(lengths)),
                'max': float(np.max(lengths)),
                'lengths': [float(l) for l in lengths]
            }
            
        result = {
            'total_bonds': len(all_bonds),
            'bond_types': list(bond_statistics.keys()),
            'bond_statistics': bond_statistics,
            'all_bonds': all_bonds[:50],  # 只保留前50个键的详细信息
        }
        
        return result
    
    def _analyze_bond_angles(self, coords: np.ndarray, elements: List[str],
                            neighbors: Dict) -> Dict:
        """键角分析"""
        n_atoms = len(elements)
        
        all_angles = []
        angles_by_center = defaultdict(list)
        
        for i in range(n_atoms):
            elem_i = elements[i]
            r_i = self.COVALENT_RADII.get(elem_i, 150) / 100
            
            # 获取成键的邻居
            bonded = []
            for j, dist, image in neighbors.get(i, []):
                r_j = self.COVALENT_RADII.get(elements[j], 150) / 100
                max_bond_length = (r_i + r_j) * (1 + self.bond_tolerance)
                if dist <= max_bond_length:
                    bonded.append((j, dist, image))
                    
            # 计算键角
            for idx1, (j1, d1, img1) in enumerate(bonded):
                for j2, d2, img2 in bonded[idx1+1:]:
                    # 计算向量
                    vec1 = coords[j1] - coords[i]  # 简化，未考虑周期性
                    vec2 = coords[j2] - coords[i]
                    
                    # 计算夹角
                    cos_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-8)
                    cos_angle = np.clip(cos_angle, -1, 1)
                    angle = np.degrees(np.arccos(cos_angle))
                    
                    angle_type = f"{elements[j1]}-{elem_i}-{elements[j2]}"
                    all_angles.append({
                        'center': i,
                        'atoms': (j1, i, j2),
                        'angle': float(angle),
                        'type': angle_type
                    })
                    angles_by_center[elem_i].append(angle)
                    
        # 统计
        angle_distribution = {}
        for elem, angles in angles_by_center.items():
            if angles:
                angle_distribution[elem] = {
                    'count': len(angles),
                    'mean': float(np.mean(angles)),
                    'std': float(np.std(angles)) if len(angles) > 1 else 0.0,
                    'angles': [float(a) for a in angles]
                }
                
        # 识别典型几何构型
        geometry_hints = []
        for elem, stats in angle_distribution.items():
            mean_angle = stats['mean']
            if 105 < mean_angle < 115:
                geometry_hints.append(f"{elem}: 可能是四面体配位 (~109.5°)")
            elif 85 < mean_angle < 95:
                geometry_hints.append(f"{elem}: 可能是八面体配位 (~90°)")
            elif 115 < mean_angle < 125:
                geometry_hints.append(f"{elem}: 可能是三角双锥或平面三角形")
            elif 175 < mean_angle < 185:
                geometry_hints.append(f"{elem}: 可能是线性配位 (~180°)")
                
        result = {
            'total_angles': len(all_angles),
            'angle_distribution_by_element': angle_distribution,
            'geometry_hints': geometry_hints,
            'all_angles': all_angles[:50],  # 只保留前50个
        }
        
        return result
    
    def _analyze_local_symmetry(self, coords: np.ndarray, neighbors: Dict) -> Dict:
        """局部对称性分析（简化版）"""
        n_atoms = len(coords)
        
        sphericity_scores = []
        
        for i in range(n_atoms):
            neighbor_coords = []
            for j, dist, _ in neighbors.get(i, [])[:12]:  # 最近的12个邻居
                neighbor_coords.append(coords[j] - coords[i])
                
            if len(neighbor_coords) >= 4:
                neighbor_coords = np.array(neighbor_coords)
                
                # 计算协方差矩阵的特征值
                cov = np.cov(neighbor_coords.T)
                eigenvalues = np.linalg.eigvalsh(cov)
                eigenvalues = np.sort(eigenvalues)[::-1]
                
                # 球形度：特征值的相似程度
                if eigenvalues[0] > 1e-6:
                    sphericity = eigenvalues[2] / eigenvalues[0]
                    sphericity_scores.append(sphericity)
                else:
                    sphericity_scores.append(0.0)
            else:
                sphericity_scores.append(0.0)
                
        result = {
            'sphericity_scores': [float(s) for s in sphericity_scores],
            'mean_sphericity': float(np.mean(sphericity_scores)) if sphericity_scores else 0.0,
            'std_sphericity': float(np.std(sphericity_scores)) if len(sphericity_scores) > 1 else 0.0,
        }
        
        return result
    
    def _compute_environment_fingerprint(self, elements: List[str], distances: np.ndarray,
                                        neighbors: Dict) -> Dict:
        """计算原子环境指纹"""
        n_atoms = len(elements)
        
        fingerprints = []
        
        for i in range(n_atoms):
            elem = elements[i]
            
            # 统计邻居类型
            neighbor_types = defaultdict(int)
            neighbor_distances = defaultdict(list)
            
            for j, dist, _ in neighbors.get(i, [])[:12]:
                neighbor_elem = elements[j]
                neighbor_types[neighbor_elem] += 1
                neighbor_distances[neighbor_elem].append(dist)
                
            # 生成指纹
            fp = {
                'center_element': elem,
                'neighbor_counts': dict(neighbor_types),
                'mean_neighbor_distances': {
                    k: float(np.mean(v)) for k, v in neighbor_distances.items()
                },
                'total_neighbors': sum(neighbor_types.values()),
            }
            fingerprints.append(fp)
            
        # 统计不同环境类型
        env_types = defaultdict(int)
        for fp in fingerprints:
            # 创建环境类型签名
            signature = f"{fp['center_element']}_CN{fp['total_neighbors']}"
            env_types[signature] += 1
            
        result = {
            'atom_fingerprints': fingerprints,
            'environment_types': dict(env_types),
            'num_unique_environments': len(env_types),
        }
        
        return result
    
    def _correlate_environment_importance(self, coord_analysis: Dict, bond_analysis: Dict,
                                         importance_scores: np.ndarray) -> Dict:
        """分析局部环境与原子重要性的关系"""
        coord_numbers = coord_analysis.get('coordination_numbers', [])
        
        result = {}
        
        # 配位数与重要性的关系
        if len(coord_numbers) == len(importance_scores) and len(coord_numbers) >= 3:
            corr = np.corrcoef(coord_numbers, importance_scores)[0, 1]
            result['coordination_importance_correlation'] = float(corr) if not np.isnan(corr) else 0.0
            
            # 按配位数分组的平均重要性
            cn_importance = defaultdict(list)
            for cn, imp in zip(coord_numbers, importance_scores):
                cn_importance[cn].append(imp)
                
            result['importance_by_coordination'] = {
                int(cn): {
                    'mean': float(np.mean(imps)),
                    'count': len(imps)
                }
                for cn, imps in cn_importance.items()
            }
            
        return result


class EnhancedAttentionVisualizer:
    """增强的注意力可视化器"""
    
    def __init__(self):
        # 设置绘图风格
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.titlesize'] = 12
        
    def plot_atom_text_attention_enhanced(self, attention_weights: np.ndarray,
                                          atom_labels: List[str],
                                          token_labels: List[str] = None,
                                          importance_scores: np.ndarray = None,
                                          save_path: str = None,
                                          figsize: Tuple = (14, 10)):
        """
        增强的原子-文本注意力热图
        
        Args:
            attention_weights: [atoms, tokens] 或 [heads, atoms, tokens] 注意力权重
            atom_labels: 原子标签列表
            token_labels: token标签列表（可选）
            importance_scores: 原子重要性分数（可选，用于排序）
            save_path: 保存路径
            figsize: 图像大小
        """
        # 处理多头注意力
        if attention_weights.ndim == 3:
            # 取所有头的平均
            attn_mean = attention_weights.mean(axis=0)
            attn_std = attention_weights.std(axis=0)
            has_multi_head = True
            n_heads = attention_weights.shape[0]
        else:
            attn_mean = attention_weights
            attn_std = None
            has_multi_head = False
            n_heads = 1
            
        n_atoms, n_tokens = attn_mean.shape
        
        # 创建和清理token标签
        if token_labels is None:
            token_labels = [f'T{i}' for i in range(n_tokens)]
        else:
            # 截断或填充
            if len(token_labels) < n_tokens:
                token_labels = token_labels + [f'T{i}' for i in range(len(token_labels), n_tokens)]
            else:
                token_labels = token_labels[:n_tokens]
        
        # 清理WordPiece token标签（移除##前缀，合并显示）
        token_labels = self._clean_token_labels(token_labels)
                
        # 如果有重要性分数，按重要性排序原子
        if importance_scores is not None and len(importance_scores) == n_atoms:
            sort_idx = np.argsort(importance_scores)[::-1]
            attn_mean = attn_mean[sort_idx]
            if attn_std is not None:
                attn_std = attn_std[sort_idx]
            atom_labels = [atom_labels[i] for i in sort_idx]
            importance_scores = importance_scores[sort_idx]
            
        # 创建图形
        if has_multi_head:
            fig = plt.figure(figsize=(figsize[0] + 4, figsize[1]))
            gs = plt.GridSpec(2, 3, width_ratios=[1, 3, 0.15], height_ratios=[3, 1],
                            hspace=0.3, wspace=0.3)
        else:
            fig = plt.figure(figsize=figsize)
            gs = plt.GridSpec(2, 3, width_ratios=[1, 3, 0.15], height_ratios=[3, 1],
                            hspace=0.3, wspace=0.3)
            
        # ==================== 主热图 ====================
        ax_main = fig.add_subplot(gs[0, 1])
        
        # 使用改进的颜色映射
        # 对注意力权重进行归一化
        attn_normalized = self._normalize_attention(attn_mean)
        
        # 绘制热图
        im = ax_main.imshow(attn_normalized, cmap='RdYlBu_r', aspect='auto',
                           vmin=0, vmax=1)
        
        # 智能设置x轴刻度（token太多时只显示部分）
        if n_tokens > 40:
            # 选择要显示的token索引（均匀分布 + 高注意力token）
            step = max(1, n_tokens // 20)
            display_indices = list(range(0, n_tokens, step))
            
            # 添加高注意力的token
            token_attention_sum = attn_normalized.sum(axis=0)
            top_attn_indices = np.argsort(token_attention_sum)[-5:]
            display_indices = sorted(set(display_indices) | set(top_attn_indices))
            
            ax_main.set_xticks(display_indices)
            ax_main.set_xticklabels([token_labels[i] for i in display_indices], 
                                    rotation=45, ha='right', fontsize=8)
        else:
            ax_main.set_xticks(np.arange(n_tokens))
            ax_main.set_xticklabels(token_labels, rotation=45, ha='right', fontsize=8)
        
        ax_main.set_yticks(np.arange(n_atoms))
        ax_main.set_yticklabels(atom_labels, fontsize=9)
        
        # 添加网格
        ax_main.set_xticks(np.arange(-0.5, n_tokens, 1), minor=True)
        ax_main.set_yticks(np.arange(-0.5, n_atoms, 1), minor=True)
        ax_main.grid(which='minor', color='white', linestyle='-', linewidth=0.5)
        
        ax_main.set_xlabel('Text Tokens', fontsize=11)
        ax_main.set_ylabel('Atoms (sorted by importance)', fontsize=11)
        ax_main.set_title('Atom-Text Cross-Modal Attention', fontsize=13, fontweight='bold')
        
        # 颜色条
        ax_cbar = fig.add_subplot(gs[0, 2])
        cbar = plt.colorbar(im, cax=ax_cbar)
        cbar.set_label('Attention Weight (normalized)', fontsize=10)
        
        # ==================== 左侧：原子重要性柱状图 ====================
        ax_left = fig.add_subplot(gs[0, 0])
        
        if importance_scores is not None:
            # 归一化重要性
            imp_norm = (importance_scores - importance_scores.min()) / \
                      (importance_scores.max() - importance_scores.min() + 1e-8)
            
            colors = plt.cm.YlOrRd(imp_norm)
            y_pos = np.arange(n_atoms)
            
            ax_left.barh(y_pos, importance_scores, color=colors, edgecolor='black', linewidth=0.5)
            ax_left.set_yticks(y_pos)
            ax_left.set_yticklabels([])
            ax_left.set_xlabel('Importance', fontsize=10)
            ax_left.set_title('Atom\nImportance', fontsize=10, fontweight='bold')
            ax_left.invert_xaxis()
            ax_left.set_ylim(-0.5, n_atoms - 0.5)
        else:
            ax_left.axis('off')
            
        # ==================== 底部：Token注意力汇总 ====================
        ax_bottom = fig.add_subplot(gs[1, 1])
        
        # 计算每个token接收的总注意力
        token_attention_sum = attn_normalized.sum(axis=0)
        token_attention_sum = token_attention_sum / token_attention_sum.max()  # 归一化
        
        x_pos = np.arange(n_tokens)
        colors = plt.cm.Blues(token_attention_sum)
        
        ax_bottom.bar(x_pos, token_attention_sum, color=colors, edgecolor='black', linewidth=0.5)
        ax_bottom.set_xticks(x_pos)
        ax_bottom.set_xticklabels(token_labels, rotation=45, ha='right', fontsize=8)
        ax_bottom.set_ylabel('Total\nAttention', fontsize=10)
        ax_bottom.set_title('Token Attention Summary', fontsize=10, fontweight='bold')
        ax_bottom.set_xlim(-0.5, n_tokens - 0.5)
        
        # 标注高注意力的token
        top_k = min(3, n_tokens)
        top_indices = np.argsort(token_attention_sum)[-top_k:]
        for idx in top_indices:
            ax_bottom.annotate(
                f'{token_attention_sum[idx]:.2f}',
                xy=(idx, token_attention_sum[idx]),
                xytext=(0, 5),
                textcoords='offset points',
                ha='center',
                fontsize=8,
                fontweight='bold'
            )
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"✅ 增强注意力热图已保存: {save_path}")
            
        plt.close()
        
        return fig
    
    def _clean_token_labels(self, token_labels: List[str]) -> List[str]:
        """
        清理WordPiece token标签，将子词合并成完整词
        
        处理规则：
        1. 将 ## 前缀的子词与前一个词合并
        2. 处理特殊token（[CLS], [SEP], [PAD]等）
        3. 截断过长的token
        
        Args:
            token_labels: 原始token标签列表
            
        Returns:
            cleaned_labels: 清理后的标签列表（长度与输入相同）
        """
        if not token_labels:
            return token_labels
            
        # 第一步：合并WordPiece子词
        merged_tokens = []
        current_word = ""
        word_start_indices = []  # 记录每个合并词的起始索引
        
        for i, token in enumerate(token_labels):
            if token is None:
                if current_word:
                    merged_tokens.append(current_word)
                    current_word = ""
                merged_tokens.append(f'T{i}')
                word_start_indices.append(i)
                continue
                
            token = str(token)
            
            # 处理特殊token
            if token in ['[CLS]', '[SEP]', '[PAD]', '[MASK]', '[UNK]']:
                if current_word:
                    merged_tokens.append(current_word)
                    current_word = ""
                merged_tokens.append(token)
                word_start_indices.append(i)
                continue
            
            # 处理WordPiece子词
            if token.startswith('##'):
                # 续词，追加到当前词
                current_word += token[2:]
            else:
                # 新词开始
                if current_word:
                    merged_tokens.append(current_word)
                current_word = token
                word_start_indices.append(i)
        
        # 处理最后一个词
        if current_word:
            merged_tokens.append(current_word)
            
        # 第二步：创建与原始长度相同的标签列表
        # 对于合并的词，只在第一个位置显示完整词，其他位置留空或用短横线
        cleaned = [''] * len(token_labels)
        
        word_idx = 0
        for i, token in enumerate(token_labels):
            if token is None:
                cleaned[i] = f'T{i}'
                continue
                
            token = str(token)
            
            if token.startswith('##'):
                # 子词位置显示为空或连接符
                cleaned[i] = ''
            else:
                # 词首位置显示合并后的完整词
                if word_idx < len(merged_tokens):
                    full_word = merged_tokens[word_idx]
                    # 截断过长的词
                    if len(full_word) > 15:
                        full_word = full_word[:12] + '..'
                    cleaned[i] = full_word
                    word_idx += 1
                else:
                    cleaned[i] = token
                    
        return cleaned
    
    def _normalize_attention(self, attention: np.ndarray) -> np.ndarray:
        """
        归一化注意力权重，使其更易于可视化
        """
        # 方法1: Min-Max归一化
        attn_min = attention.min()
        attn_max = attention.max()
        
        if attn_max - attn_min > 1e-8:
            normalized = (attention - attn_min) / (attn_max - attn_min)
        else:
            normalized = np.zeros_like(attention)
            
        return normalized
    
    def plot_multi_head_attention(self, attention_weights: np.ndarray,
                                  atom_labels: List[str],
                                  token_labels: List[str] = None,
                                  save_path: str = None,
                                  max_heads: int = 8):
        """
        可视化多头注意力的各个头
        
        Args:
            attention_weights: [heads, atoms, tokens] 注意力权重
            atom_labels: 原子标签
            token_labels: token标签
            save_path: 保存路径
            max_heads: 最多显示的头数
        """
        if attention_weights.ndim != 3:
            print("警告: 输入不是多头注意力格式")
            return
            
        n_heads = min(attention_weights.shape[0], max_heads)
        n_atoms, n_tokens = attention_weights.shape[1:]
        
        # 创建和清理token标签
        if token_labels is None:
            token_labels = [f'T{i}' for i in range(n_tokens)]
        else:
            token_labels = self._clean_token_labels(token_labels)
            
        # 计算布局
        n_cols = min(4, n_heads)
        n_rows = (n_heads + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3.5*n_rows))
        if n_heads == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)
            
        for h in range(n_heads):
            row, col = h // n_cols, h % n_cols
            ax = axes[row, col]
            
            attn = self._normalize_attention(attention_weights[h])
            
            im = ax.imshow(attn, cmap='RdYlBu_r', aspect='auto', vmin=0, vmax=1)
            
            # 简化标签
            if n_atoms <= 10:
                ax.set_yticks(range(n_atoms))
                ax.set_yticklabels(atom_labels, fontsize=7)
            else:
                ax.set_yticks([0, n_atoms//2, n_atoms-1])
                ax.set_yticklabels([atom_labels[0], '...', atom_labels[-1]], fontsize=7)
                
            if n_tokens <= 15:
                ax.set_xticks(range(n_tokens))
                clean_labels = token_labels[:n_tokens] if len(token_labels) >= n_tokens else token_labels
                ax.set_xticklabels(clean_labels, rotation=45, ha='right', fontsize=6)
            else:
                # 选择关键位置显示
                display_idx = [0, n_tokens//4, n_tokens//2, 3*n_tokens//4, n_tokens-1]
                ax.set_xticks(display_idx)
                ax.set_xticklabels([token_labels[i] if i < len(token_labels) else f'T{i}' 
                                   for i in display_idx], fontsize=7)
                
            ax.set_title(f'Head {h+1}', fontsize=10, fontweight='bold')
            
            # 添加颜色条
            plt.colorbar(im, ax=ax, shrink=0.6)
            
        # 隐藏多余的子图
        for h in range(n_heads, n_rows * n_cols):
            row, col = h // n_cols, h % n_cols
            axes[row, col].axis('off')
            
        plt.suptitle('Multi-Head Cross-Modal Attention Analysis', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"✅ 多头注意力图已保存: {save_path}")
            
        plt.close()
        
    def plot_attention_flow_sankey(self, atom_attention: np.ndarray,
                                   atom_labels: List[str],
                                   token_labels: List[str],
                                   save_path: str = None,
                                   top_k: int = 5):
        """
        使用简化的流图显示注意力流动
        
        Args:
            atom_attention: [atoms, tokens] 注意力权重
            atom_labels: 原子标签
            token_labels: token标签
            save_path: 保存路径
            top_k: 显示前k个最强的连接
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        n_atoms, n_tokens = atom_attention.shape
        
        # 清理token标签
        token_labels = self._clean_token_labels(token_labels)
        
        # 归一化
        attn_norm = self._normalize_attention(atom_attention)
        
        # 计算每个token的总注意力，选择最重要的token显示
        token_attention_sum = attn_norm.sum(axis=0)
        max_display_tokens = 20
        
        if n_tokens > max_display_tokens:
            # 选择注意力最高的token
            top_token_indices = np.argsort(token_attention_sum)[-max_display_tokens:]
            top_token_indices = sorted(top_token_indices)  # 保持顺序
        else:
            top_token_indices = list(range(n_tokens))
            
        display_tokens = len(top_token_indices)
        
        # 找到最强的连接
        flat_indices = np.argsort(attn_norm.flatten())[::-1][:top_k * n_atoms]
        
        # 设置布局
        atom_y = np.linspace(0.9, 0.1, n_atoms)
        token_y = np.linspace(0.9, 0.1, display_tokens)
        
        # 创建token索引映射
        token_idx_to_display = {idx: i for i, idx in enumerate(top_token_indices)}
        
        # 绘制原子节点
        for i, (label, y) in enumerate(zip(atom_labels, atom_y)):
            ax.scatter(0.2, y, s=200, c='steelblue', zorder=3, edgecolors='black')
            ax.text(0.1, y, label, ha='right', va='center', fontsize=9, fontweight='bold')
            
        # 绘制token节点
        for i, token_idx in enumerate(top_token_indices):
            label = token_labels[token_idx] if token_idx < len(token_labels) else f'T{token_idx}'
            ax.scatter(0.8, token_y[i], s=150, c='coral', zorder=3, edgecolors='black')
            ax.text(0.9, token_y[i], label, ha='left', va='center', fontsize=8)
            
        # 绘制连接线
        for flat_idx in flat_indices:
            atom_idx = flat_idx // n_tokens
            token_idx = flat_idx % n_tokens
            
            if atom_idx < n_atoms and token_idx in token_idx_to_display:
                weight = attn_norm[atom_idx, token_idx]
                display_idx = token_idx_to_display[token_idx]
                
                if weight > 0.1:  # 只显示较强的连接
                    ax.plot([0.2, 0.8], [atom_y[atom_idx], token_y[display_idx]],
                           color='gray', alpha=weight, linewidth=weight*5,
                           zorder=1)
                    
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title('Atom-Token Attention Flow', fontsize=14, fontweight='bold')
        
        # 添加图例
        ax.text(0.2, 0.98, 'Atoms', ha='center', fontsize=11, fontweight='bold', color='steelblue')
        ax.text(0.8, 0.98, 'Tokens', ha='center', fontsize=11, fontweight='bold', color='coral')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"✅ 注意力流图已保存: {save_path}")
            
        plt.close()


class LocalEnvironmentVisualizer:
    """局部化学环境可视化器"""
    
    def plot_coordination_analysis(self, env_analysis: Dict, 
                                   importance_scores: np.ndarray = None,
                                   save_path: str = None):
        """
        可视化配位数分析结果
        
        Args:
            env_analysis: LocalEnvironmentAnalyzer的分析结果
            importance_scores: 原子重要性分数
            save_path: 保存路径
        """
        coord_data = env_analysis.get('coordination', {})
        bond_data = env_analysis.get('bonds', {})
        angle_data = env_analysis.get('bond_angles', {})
        
        fig = plt.figure(figsize=(16, 12))
        gs = plt.GridSpec(3, 3, hspace=0.35, wspace=0.3)
        
        # ==================== 1. 配位数分布 ====================
        ax1 = fig.add_subplot(gs[0, 0])
        
        coord_numbers = coord_data.get('coordination_numbers', [])
        if coord_numbers:
            unique_cn, counts = np.unique(coord_numbers, return_counts=True)
            colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(unique_cn)))
            
            bars = ax1.bar(unique_cn, counts, color=colors, edgecolor='black', linewidth=1)
            ax1.set_xlabel('Coordination Number', fontsize=11)
            ax1.set_ylabel('Count', fontsize=11)
            ax1.set_title('Coordination Number Distribution', fontsize=12, fontweight='bold')
            ax1.set_xticks(unique_cn)
            
            # 添加数值标签
            for bar, count in zip(bars, counts):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        str(count), ha='center', va='bottom', fontweight='bold')
                
        # ==================== 2. 按元素的配位数 ====================
        ax2 = fig.add_subplot(gs[0, 1])
        
        coord_by_elem = coord_data.get('coordination_by_element', {})
        if coord_by_elem:
            elements = list(coord_by_elem.keys())
            means = [coord_by_elem[e]['mean'] for e in elements]
            stds = [coord_by_elem[e]['std'] for e in elements]
            
            x_pos = np.arange(len(elements))
            colors = plt.cm.Set2(np.linspace(0, 1, len(elements)))
            
            ax2.bar(x_pos, means, yerr=stds, color=colors, edgecolor='black',
                   linewidth=1, capsize=5, error_kw={'linewidth': 2})
            ax2.set_xticks(x_pos)
            ax2.set_xticklabels(elements, fontsize=11, fontweight='bold')
            ax2.set_ylabel('Coordination Number', fontsize=11)
            ax2.set_title('Coordination by Element', fontsize=12, fontweight='bold')
            
        # ==================== 3. 配位数与重要性关系 ====================
        ax3 = fig.add_subplot(gs[0, 2])
        
        env_imp_corr = env_analysis.get('environment_importance_correlation', {})
        imp_by_cn = env_imp_corr.get('importance_by_coordination', {})
        
        if imp_by_cn:
            cns = sorted(imp_by_cn.keys())
            means = [imp_by_cn[cn]['mean'] for cn in cns]
            counts = [imp_by_cn[cn]['count'] for cn in cns]
            
            # 气泡图
            sizes = [c * 100 for c in counts]
            scatter = ax3.scatter(cns, means, s=sizes, c=means, cmap='YlOrRd',
                                 edgecolors='black', linewidth=1, alpha=0.7)
            
            ax3.set_xlabel('Coordination Number', fontsize=11)
            ax3.set_ylabel('Mean Importance', fontsize=11)
            ax3.set_title('Importance vs Coordination', fontsize=12, fontweight='bold')
            
            # 添加相关系数
            corr = env_imp_corr.get('coordination_importance_correlation', 0)
            ax3.text(0.95, 0.95, f'r = {corr:.3f}', transform=ax3.transAxes,
                    ha='right', va='top', fontsize=10, 
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        else:
            ax3.text(0.5, 0.5, 'No importance data', ha='center', va='center')
            ax3.set_title('Importance vs Coordination', fontsize=12, fontweight='bold')
            
        # ==================== 4. 键长分布 ====================
        ax4 = fig.add_subplot(gs[1, 0])
        
        bond_stats = bond_data.get('bond_statistics', {})
        if bond_stats:
            bond_types = list(bond_stats.keys())
            all_lengths = []
            labels = []
            
            for bt in bond_types:
                lengths = bond_stats[bt]['lengths']
                all_lengths.append(lengths)
                labels.append(f"{bt}\n(n={len(lengths)})")
                
            if all_lengths:
                bp = ax4.boxplot(all_lengths, labels=labels, patch_artist=True)
                
                colors = plt.cm.Pastel1(np.linspace(0, 1, len(all_lengths)))
                for patch, color in zip(bp['boxes'], colors):
                    patch.set_facecolor(color)
                    
                ax4.set_ylabel('Bond Length (Å)', fontsize=11)
                ax4.set_title('Bond Length Distribution', fontsize=12, fontweight='bold')
                plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')
                
        # ==================== 5. 键角分布 ====================
        ax5 = fig.add_subplot(gs[1, 1])
        
        angle_dist = angle_data.get('angle_distribution_by_element', {})
        if angle_dist:
            all_angles = []
            for elem, data in angle_dist.items():
                all_angles.extend(data['angles'])
                
            if all_angles:
                ax5.hist(all_angles, bins=20, color='steelblue', edgecolor='black',
                        alpha=0.7, density=True)
                
                # 标注典型角度
                typical_angles = [90, 109.5, 120, 180]
                for angle in typical_angles:
                    ax5.axvline(angle, color='red', linestyle='--', alpha=0.5)
                    ax5.text(angle, ax5.get_ylim()[1]*0.9, f'{angle}°',
                            ha='center', fontsize=8, color='red')
                    
                ax5.set_xlabel('Bond Angle (°)', fontsize=11)
                ax5.set_ylabel('Density', fontsize=11)
                ax5.set_title('Bond Angle Distribution', fontsize=12, fontweight='bold')
                
        # ==================== 6. 几何构型提示 ====================
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.axis('off')
        
        geometry_hints = angle_data.get('geometry_hints', [])
        
        text = "═══ Geometry Analysis ═══\n\n"
        if geometry_hints:
            for hint in geometry_hints:
                text += f"• {hint}\n"
        else:
            text += "No specific geometry identified.\n"
            
        # 添加统计摘要
        text += f"\n═══ Summary ═══\n\n"
        text += f"Total atoms: {env_analysis.get('num_atoms', 'N/A')}\n"
        text += f"Mean coordination: {coord_data.get('mean_coordination', 0):.2f}\n"
        text += f"Total bonds: {bond_data.get('total_bonds', 0)}\n"
        text += f"Bond types: {len(bond_stats)}\n"
        text += f"Total angles: {angle_data.get('total_angles', 0)}\n"
        
        ax6.text(0.1, 0.95, text, transform=ax6.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        ax6.set_title('Geometry & Summary', fontsize=12, fontweight='bold')
        
        # ==================== 7. 环境指纹 ====================
        ax7 = fig.add_subplot(gs[2, :2])
        
        fingerprint = env_analysis.get('environment_fingerprint', {})
        env_types = fingerprint.get('environment_types', {})
        
        if env_types:
            types = list(env_types.keys())
            counts = list(env_types.values())
            
            colors = plt.cm.Set3(np.linspace(0, 1, len(types)))
            
            wedges, texts, autotexts = ax7.pie(
                counts, labels=types, colors=colors,
                autopct='%1.1f%%', startangle=90,
                explode=[0.02] * len(types),
                textprops={'fontsize': 9}
            )
            
            ax7.set_title('Atomic Environment Types', fontsize=12, fontweight='bold')
            
        # ==================== 8. 球形度分布 ====================
        ax8 = fig.add_subplot(gs[2, 2])
        
        symmetry = env_analysis.get('local_symmetry', {})
        sphericity = symmetry.get('sphericity_scores', [])
        
        if sphericity:
            ax8.hist(sphericity, bins=15, color='mediumseagreen', edgecolor='black', alpha=0.7)
            ax8.axvline(np.mean(sphericity), color='red', linestyle='--', linewidth=2,
                       label=f'Mean: {np.mean(sphericity):.3f}')
            ax8.set_xlabel('Sphericity Score', fontsize=11)
            ax8.set_ylabel('Count', fontsize=11)
            ax8.set_title('Local Symmetry (Sphericity)', fontsize=12, fontweight='bold')
            ax8.legend()
            
        plt.suptitle('Local Chemical Environment Analysis', fontsize=14, fontweight='bold', y=1.02)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"✅ 局部环境分析图已保存: {save_path}")
            
        plt.close()
        
        return fig


def integrate_with_explainer(explainer_class):
    """
    装饰器：将局部环境分析集成到ComprehensiveExplainer
    
    用法:
        from local_environment_analyzer import integrate_with_explainer
        
        @integrate_with_explainer
        class MyExplainer(ComprehensiveExplainer):
            pass
    """
    original_explain = explainer_class.explain_prediction
    
    def enhanced_explain(self, g, lg, text, atoms_object, true_value=None,
                        save_dir=None, sample_id='sample', 
                        analyze_local_env=True, **kwargs):
        # 调用原始方法
        explanation = original_explain(
            self, g, lg, text, atoms_object, true_value,
            save_dir, sample_id, **kwargs
        )
        
        # 添加局部环境分析
        if analyze_local_env:
            env_analyzer = LocalEnvironmentAnalyzer()
            importance = np.array(explanation.get('atom_importance_integrated_gradients', []))
            
            env_analysis = env_analyzer.analyze_local_environment(
                atoms_object, 
                importance if len(importance) > 0 else None
            )
            explanation['local_environment'] = env_analysis
            
            # 可视化
            if save_dir:
                save_dir = Path(save_dir)
                env_viz = LocalEnvironmentVisualizer()
                env_viz.plot_coordination_analysis(
                    env_analysis,
                    importance if len(importance) > 0 else None,
                    save_path=save_dir / f'{sample_id}_local_environment.png'
                )
                
        return explanation
        
    explainer_class.explain_prediction = enhanced_explain
    return explainer_class


# ==================== 便捷函数 ====================

def analyze_local_environment(atoms_object, importance_scores=None, save_dir=None, sample_id='sample'):
    """
    便捷函数：快速分析局部化学环境
    
    Args:
        atoms_object: JARVIS Atoms对象
        importance_scores: 原子重要性分数（可选）
        save_dir: 保存目录（可选）
        sample_id: 样本ID
        
    Returns:
        分析结果字典
    """
    analyzer = LocalEnvironmentAnalyzer()
    results = analyzer.analyze_local_environment(atoms_object, importance_scores)
    
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        viz = LocalEnvironmentVisualizer()
        viz.plot_coordination_analysis(
            results, importance_scores,
            save_path=save_dir / f'{sample_id}_local_environment.png'
        )
        
    return results


if __name__ == "__main__":
    print("""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                    局部化学环境分析模块使用指南                               ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║                                                                               ║
║  1. 基本使用:                                                                 ║
║                                                                               ║
║     from local_environment_analyzer import LocalEnvironmentAnalyzer           ║
║                                                                               ║
║     analyzer = LocalEnvironmentAnalyzer(cutoff=4.0)                           ║
║     results = analyzer.analyze_local_environment(atoms_object, importance)    ║
║                                                                               ║
║  2. 可视化:                                                                   ║
║                                                                               ║
║     from local_environment_analyzer import LocalEnvironmentVisualizer         ║
║                                                                               ║
║     viz = LocalEnvironmentVisualizer()                                        ║
║     viz.plot_coordination_analysis(results, importance, 'output.png')         ║
║                                                                               ║
║  3. 增强注意力可视化:                                                         ║
║                                                                               ║
║     from local_environment_analyzer import EnhancedAttentionVisualizer        ║
║                                                                               ║
║     viz = EnhancedAttentionVisualizer()                                       ║
║     viz.plot_atom_text_attention_enhanced(                                    ║
║         attention_weights, atom_labels, token_labels,                         ║
║         importance_scores, 'attention.png'                                    ║
║     )                                                                         ║
║                                                                               ║
║  4. 快捷函数:                                                                 ║
║                                                                               ║
║     from local_environment_analyzer import analyze_local_environment          ║
║                                                                               ║
║     results = analyze_local_environment(                                      ║
║         atoms_object, importance_scores,                                      ║
║         save_dir='./output', sample_id='sample_001'                           ║
║     )                                                                         ║
║                                                                               ║
║  5. 分析结果包含:                                                             ║
║     - coordination: 配位数分析                                                ║
║     - bonds: 键长分析                                                         ║
║     - bond_angles: 键角分析                                                   ║
║     - local_symmetry: 局部对称性                                              ║
║     - environment_fingerprint: 环境指纹                                       ║
║     - environment_importance_correlation: 环境与重要性相关性                  ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
    """)
