"""DenseGNN: Dense Graph Neural Network for Crystal Property Prediction.

A PyTorch implementation of DenseGNN with multimodal fusion capabilities.
Based on the TensorFlow DenseGNN implementation from kgcnn.literature.DenseGNN.
"""
from typing import Tuple, Union
import dgl
import dgl.function as fn
import numpy as np
import torch
from dgl.nn import AvgPooling
from pydantic import root_validator
from pydantic.typing import Literal
from torch import nn
from torch.nn import functional as F

from models.utils import RBFExpansion
from utils import BaseSettings

from transformers import AutoTokenizer, AutoModel
from tokenizers.normalizers import BertNormalizer

# VoCab Mapping and Normalizer
f = open('vocab_mappings.txt', 'r')
mappings = f.read().strip().split('\n')
f.close()

mappings = {m[0]: m[2:] for m in mappings}
norm = BertNormalizer(lowercase=False, strip_accents=True, clean_text=True, handle_chinese_chars=True)

def normalize(text):
    text = [norm.normalize_str(s) for s in text.split('\n')]
    out = []
    for s in text:
        norm_s = ''
        for c in s:
            norm_s += mappings.get(c, ' ')
        out.append(norm_s)
    return '\n'.join(out)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
tokenizer = AutoTokenizer.from_pretrained('m3rg-iitd/matscibert', model_max_length=512)
text_model = AutoModel.from_pretrained('m3rg-iitd/matscibert')
text_model.to(device)


class ProjectionHead(nn.Module):
    """Projection head for contrastive learning."""
    def __init__(self, embedding_dim, projection_dim=64, dropout=0.1):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)

    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x


class ContrastiveLoss(nn.Module):
    """Contrastive loss for aligning graph and text representations."""
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, graph_features, text_features):
        batch_size = graph_features.size(0)
        graph_features = F.normalize(graph_features, dim=1)
        text_features = F.normalize(text_features, dim=1)

        logits = torch.matmul(graph_features, text_features.T) / self.temperature
        labels = torch.arange(batch_size, device=graph_features.device)

        loss_g2t = F.cross_entropy(logits, labels)
        loss_t2g = F.cross_entropy(logits.T, labels)

        return (loss_g2t + loss_t2g) / 2


class CrossModalAttention(nn.Module):
    """Cross-modal attention between graph and text features."""
    def __init__(self, graph_dim=256, text_dim=64, hidden_dim=256, num_heads=4, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        assert hidden_dim % num_heads == 0

        self.g2t_query = nn.Linear(graph_dim, hidden_dim)
        self.g2t_key = nn.Linear(text_dim, hidden_dim)
        self.g2t_value = nn.Linear(text_dim, hidden_dim)

        self.t2g_query = nn.Linear(text_dim, hidden_dim)
        self.t2g_key = nn.Linear(graph_dim, hidden_dim)
        self.t2g_value = nn.Linear(graph_dim, hidden_dim)

        self.graph_output = nn.Linear(hidden_dim, graph_dim)
        self.text_output = nn.Linear(hidden_dim, text_dim)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm_graph = nn.LayerNorm(graph_dim)
        self.layer_norm_text = nn.LayerNorm(text_dim)

        self.scale = self.head_dim ** -0.5

    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)

    def forward(self, graph_feat, text_feat, return_attention=False):
        batch_size = graph_feat.size(0)

        if graph_feat.dim() == 2:
            graph_feat_seq = graph_feat.unsqueeze(1)
        else:
            graph_feat_seq = graph_feat

        if text_feat.dim() == 2:
            text_feat_seq = text_feat.unsqueeze(1)
        else:
            text_feat_seq = text_feat

        # Graph-to-Text Attention
        Q_g2t = self.g2t_query(graph_feat_seq)
        K_g2t = self.g2t_key(text_feat_seq)
        V_g2t = self.g2t_value(text_feat_seq)

        Q_g2t = self.split_heads(Q_g2t, batch_size)
        K_g2t = self.split_heads(K_g2t, batch_size)
        V_g2t = self.split_heads(V_g2t, batch_size)

        attn_g2t = torch.matmul(Q_g2t, K_g2t.transpose(-2, -1)) * self.scale
        attn_g2t = F.softmax(attn_g2t, dim=-1)
        attn_g2t = self.dropout(attn_g2t)

        context_g2t = torch.matmul(attn_g2t, V_g2t)
        context_g2t = context_g2t.permute(0, 2, 1, 3).contiguous()
        context_g2t = context_g2t.view(batch_size, 1, self.hidden_dim)
        context_g2t = self.graph_output(context_g2t).squeeze(1)

        # Text-to-Graph Attention
        Q_t2g = self.t2g_query(text_feat_seq)
        K_t2g = self.t2g_key(graph_feat_seq)
        V_t2g = self.t2g_value(graph_feat_seq)

        Q_t2g = self.split_heads(Q_t2g, batch_size)
        K_t2g = self.split_heads(K_t2g, batch_size)
        V_t2g = self.split_heads(V_t2g, batch_size)

        attn_t2g = torch.matmul(Q_t2g, K_t2g.transpose(-2, -1)) * self.scale
        attn_t2g = F.softmax(attn_t2g, dim=-1)
        attn_t2g = self.dropout(attn_t2g)

        context_t2g = torch.matmul(attn_t2g, V_t2g)
        context_t2g = context_t2g.permute(0, 2, 1, 3).contiguous()
        context_t2g = context_t2g.view(batch_size, 1, self.hidden_dim)
        context_t2g = self.text_output(context_t2g).squeeze(1)

        enhanced_graph = self.layer_norm_graph(graph_feat + context_g2t)
        enhanced_text = self.layer_norm_text(text_feat + context_t2g)

        if return_attention:
            return enhanced_graph, enhanced_text, {'graph_to_text': attn_g2t, 'text_to_graph': attn_t2g}
        return enhanced_graph, enhanced_text


class MLPLayer(nn.Module):
    """MLP layer with normalization."""
    def __init__(self, in_features: int, out_features: int, bias: bool = True, activation=F.silu):
        super().__init__()
        self.layer = nn.Linear(in_features, out_features, bias=bias)
        self.activation = activation
        self.bn = nn.BatchNorm1d(out_features)

    def forward(self, x):
        return self.activation(self.bn(self.layer(x)))


class DenseGNNConv(nn.Module):
    """Dense GNN Convolution layer.

    Based on the DenseGNN architecture with dense connectivity and
    hierarchical node-edge-graph updates.
    """
    def __init__(self, node_features: int, edge_features: int,
                 output_features: int, dropout: float = 0.0):
        super().__init__()

        # Node update
        self.node_mlp = nn.Sequential(
            nn.Linear(node_features * 2 + edge_features, output_features),
            nn.BatchNorm1d(output_features),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(output_features, output_features),
            nn.BatchNorm1d(output_features)
        )

        # Edge update
        self.edge_mlp = nn.Sequential(
            nn.Linear(node_features * 2 + edge_features, output_features),
            nn.BatchNorm1d(output_features),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(output_features, output_features),
            nn.BatchNorm1d(output_features)
        )

        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, g: dgl.DGLGraph, node_feats: torch.Tensor,
                edge_feats: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Dense GNN convolution forward pass."""
        g = g.local_var()

        # Gather node features for edges
        g.ndata['h'] = node_feats
        g.apply_edges(fn.copy_u('h', 'h_src'))
        g.apply_edges(fn.copy_v('h', 'h_dst'))

        # Edge update: concat(src_node, dst_node, edge)
        edge_input = torch.cat([
            g.edata['h_src'],
            g.edata['h_dst'],
            edge_feats
        ], dim=-1)

        updated_edges = self.edge_mlp(edge_input)
        updated_edges = self.activation(updated_edges + edge_feats)  # Residual connection

        # Node update: aggregate from edges
        g.edata['m'] = updated_edges
        g.update_all(fn.copy_e('m', 'm'), fn.mean('m', 'h_agg'))

        # Concat aggregated features with original node features
        node_input = torch.cat([
            node_feats,
            g.ndata['h_agg'],
            torch.zeros(node_feats.size(0), edge_feats.size(1),
                       device=node_feats.device)  # Placeholder for consistency
        ], dim=-1)

        updated_nodes = self.node_mlp(node_input)
        updated_nodes = self.activation(updated_nodes + node_feats)  # Residual connection

        return updated_nodes, updated_edges


class DenseGNNConfig(BaseSettings):
    """Hyperparameter schema for DenseGNN model."""
    name: Literal["densegnn"]
    densegnn_layers: int = 4
    atom_input_features: int = 92
    edge_input_features: int = 80
    embedding_features: int = 64
    hidden_features: int = 256
    output_features: int = 1
    graph_dropout: float = 0.0

    # Cross-modal attention settings
    use_cross_modal_attention: bool = False
    cross_modal_hidden_dim: int = 256
    cross_modal_num_heads: int = 4
    cross_modal_dropout: float = 0.1

    # Contrastive learning settings
    use_contrastive_loss: bool = False
    contrastive_temperature: float = 0.1
    contrastive_loss_weight: float = 0.1

    link: Literal["identity", "log", "logit"] = "identity"
    classification: bool = False

    class Config:
        """Configure model settings behavior."""
        env_prefix = "jv_model"


class DenseGNN(nn.Module):
    """DenseGNN model for crystal property prediction with multimodal support.

    Implements dense connectivity graph neural network with support for
    text-graph multimodal fusion.
    """
    def __init__(self, config: DenseGNNConfig = DenseGNNConfig(name="densegnn")):
        super().__init__()
        self.classification = config.classification

        # Atom and edge embeddings
        self.atom_embedding = MLPLayer(config.atom_input_features, config.hidden_features)

        self.edge_embedding = nn.Sequential(
            RBFExpansion(vmin=0, vmax=8.0, bins=config.edge_input_features),
            MLPLayer(config.edge_input_features, config.embedding_features),
            MLPLayer(config.embedding_features, config.hidden_features),
        )

        # DenseGNN layers
        self.densegnn_layers = nn.ModuleList([
            DenseGNNConv(
                node_features=config.hidden_features,
                edge_features=config.hidden_features,
                output_features=config.hidden_features,
                dropout=config.graph_dropout
            )
            for _ in range(config.densegnn_layers)
        ])

        # Graph pooling
        self.readout = AvgPooling()

        # Projection heads for multimodal learning
        self.graph_projection = ProjectionHead(embedding_dim=config.hidden_features)
        self.text_projection = ProjectionHead(embedding_dim=768)

        # Cross-modal attention
        self.use_cross_modal_attention = config.use_cross_modal_attention
        if self.use_cross_modal_attention:
            self.cross_modal_attention = CrossModalAttention(
                graph_dim=64,
                text_dim=64,
                hidden_dim=config.cross_modal_hidden_dim,
                num_heads=config.cross_modal_num_heads,
                dropout=config.cross_modal_dropout
            )
            self.fc1 = nn.Linear(64, 64)
            self.fc = nn.Linear(64, config.output_features)
        else:
            self.fc1 = nn.Linear(128, 64)
            self.fc = nn.Linear(64, config.output_features)

        # Contrastive learning
        self.use_contrastive_loss = config.use_contrastive_loss
        if self.use_contrastive_loss:
            self.contrastive_loss_fn = ContrastiveLoss(temperature=config.contrastive_temperature)
            self.contrastive_loss_weight = config.contrastive_loss_weight

        # Output activation
        self.link = None
        self.link_name = config.link
        if config.link == "identity":
            self.link = lambda x: x
        elif config.link == "log":
            self.link = torch.exp
            avg_gap = 0.7
            self.fc.bias.data = torch.tensor(np.log(avg_gap), dtype=torch.float)
        elif config.link == "logit":
            self.link = torch.sigmoid

    def forward(self, g: Union[Tuple[dgl.DGLGraph, dgl.DGLGraph], dgl.DGLGraph],
                return_features=False, return_attention=False):
        """Forward pass of DenseGNN model.

        Args:
            g: Tuple of (graph, line_graph, text) or just graph
            return_features: Whether to return intermediate features
            return_attention: Whether to return attention weights

        Returns:
            predictions or dict with predictions and features
        """
        # Parse input
        if isinstance(g, tuple) and len(g) >= 2:
            g, _, text = g  # DenseGNN doesn't use line graph
        else:
            g, text = g, None

        g = g.local_var()

        # Text encoding (if provided)
        if text is not None:
            norm_sents = [normalize(s) for s in text]
            encodings = tokenizer(norm_sents, return_tensors='pt', padding=True, truncation=True)
            if torch.cuda.is_available():
                encodings = {k: v.to(device) for k, v in encodings.items()}

            with torch.no_grad():
                last_hidden_state = text_model(**encodings)[0]

            cls_emb = last_hidden_state[:, 0, :]
            text_emb = self.text_projection(cls_emb)
        else:
            text_emb = None

        # Initial node features
        x = g.ndata.pop("atom_features")
        x = self.atom_embedding(x)

        # Initial edge features
        bondlength = torch.norm(g.edata.pop("r"), dim=1)
        y = self.edge_embedding(bondlength)

        # DenseGNN updates
        for layer in self.densegnn_layers:
            x, y = layer(g, x, y)

        # Graph-level pooling
        graph_emb = self.readout(g, x)
        graph_emb_projected = self.graph_projection(graph_emb)

        # Multimodal fusion
        if text_emb is not None:
            if self.use_cross_modal_attention:
                enhanced_graph, enhanced_text = self.cross_modal_attention(
                    graph_emb_projected, text_emb
                )
                fused = (enhanced_graph + enhanced_text) / 2
            else:
                fused = torch.cat([graph_emb_projected, text_emb], dim=-1)

            h = F.silu(self.fc1(fused))
            out = self.fc(h)

            # Contrastive loss
            if self.use_contrastive_loss and self.training:
                contrastive_loss = self.contrastive_loss_fn(graph_emb_projected, text_emb)
                if return_features:
                    return {
                        'predictions': self.link(out),
                        'contrastive_loss': contrastive_loss,
                        'graph_features': graph_emb_projected,
                        'text_features': text_emb
                    }
                return {
                    'predictions': self.link(out),
                    'contrastive_loss': contrastive_loss
                }
        else:
            # No text, use graph features only
            h = F.silu(self.fc1(torch.cat([graph_emb_projected,
                                           torch.zeros_like(graph_emb_projected)], dim=-1)))
            out = self.fc(h)

        if return_features or return_attention:
            return {
                'predictions': self.link(out),
                'graph_features': graph_emb_projected,
                'text_features': text_emb if text_emb is not None else None
            }

        return self.link(out)
