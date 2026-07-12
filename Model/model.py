import random
from typing import Optional, Tuple, List
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
import pandas as pd
from pathlib import Path
import numpy as np
import os
import json

class Cfg:
    epochs     = 100
    batch_size = 8
    seed       = 66
    device     = "cuda" if torch.cuda.is_available() else "cpu"
    # Data sources
    csv_path: Optional[str] = '/root/AbMSPN/Model/csv/Pt_Mapping_S645.csv'  
    root_dir: Optional[str] = '/root/AbMSPN/Data/graphs'     
    split_json_path: Optional[str] = "/root/AbMSPN/Model/jsonl/CV10_S645.jsonl" 
    json_indices_one_based: bool = True
    dropout = 0.25 
    # Early stopping
    early_stop_patience: int = 40
    # Training hyperparameters
    learning_rate: float = 1e-4  
    weight_decay: float = 1e-3   
    # number of MacroBlocks in the encoder
    n_blocks: int = 4 
    # hidden scalar dim for atoms/residues
    sA: int = 128 
    # Vector channel configuration
    atom_vdim: int = 4  
    res_vdim: int = 3   
    # Encoder backend configuration
    atom_encoder: str = "gvp"
    res_encoder: str = "gvp" 
    # Graph neural network configuration
    use_graph: bool = True  
    # ESM configuration
    use_esm: bool = False 

def set_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8" 
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def pearsonr_torch(x, y):
    x = x - x.mean(); y = y - y.mean()
    denom = x.norm() * y.norm() + 1e-8
    return (x*y).sum() / denom

def spearmanr_torch(x, y):
    x_cpu = x.detach().to(torch.float32).cpu().view(-1).numpy()
    y_cpu = y.detach().to(torch.float32).cpu().view(-1).numpy()
    rx = pd.Series(x_cpu).rank(method='average').to_numpy()
    ry = pd.Series(y_cpu).rank(method='average').to_numpy()
    rx_t = torch.from_numpy(rx).float()
    ry_t = torch.from_numpy(ry).float()
    return pearsonr_torch(rx_t, ry_t)

def r2_torch(y_pred, y_true):
    y_true_mean = y_true.mean()
    ss_tot = ((y_true - y_true_mean) ** 2).sum()
    ss_res = ((y_true - y_pred) ** 2).sum()
    r2 = 1 - (ss_res / (ss_tot + 1e-8))
    return r2

# ---------- GVP-GNN Implementation ----------
def _safe_norm(v: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    """数值稳定的向量范数。v: [..., 3]"""
    return torch.sqrt(torch.clamp((v * v).sum(dim=dim), min=eps))

def vector_rms_norm(V, eps=1e-8):
    """
    向量RMS归一化
    V: [N, v_dim, 3]
    每个样本一行，对 v_dim 根向量的平方范数取均值再开方
    """
    rms = torch.sqrt((V.pow(2).sum(-1).mean(-1, keepdim=True)).clamp_min(eps))
    return V / rms.unsqueeze(-1)

class VectorDropout(nn.Module):
    """
    向量通道级Dropout
    对形状 [N, v_dim, 3] 的向量,按通道进行dropout
    """
    def __init__(self, p=0.1):
        super().__init__()
        self.p = p
    
    def forward(self, v):
        if v is None or self.p == 0 or not self.training:
            return v
        mask_shape = v.shape[:-1] + (1,)  # [N, v_dim, 1]
        mask = torch.bernoulli(torch.full(mask_shape, 1 - self.p, device=v.device, dtype=v.dtype))
        mask = mask / (1 - self.p)         
        return v * mask


class LayerNormSV(nn.Module):
    """
    - 标量通道:标准LayerNorm
    - 向量通道:RMS归一化
    """
    def __init__(self, s_dim: int, v_dim: int, eps: float = 1e-5):
        super().__init__()
        self.s_ln = nn.LayerNorm(s_dim) if s_dim > 0 else nn.Identity()
        self.v_dim = v_dim
        self.eps = eps

    def forward(self,
                s: Optional[torch.Tensor],     # [N, s_dim]
                v: Optional[torch.Tensor]      # [N, v_dim, 3]
                ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        s = self.s_ln(s) if (s is not None and isinstance(self.s_ln, nn.LayerNorm)) else s
        if v is None or self.v_dim == 0:
            return s, v
        v = vector_rms_norm(v, eps=self.eps)  # RMS归一化
        return s, v

class GVP(nn.Module):
    """
      输入/输出: (s, v),v 的最后维为 3(xyz)
      - 向量通道: v_in -> Vh -> Vo(方向归一化)
      - 标量分支: concat(s, ||Vh||) -> MLP -> s_out
      - 向量分支: v_out = (W_g s_out) * (Vo / ||Vo||)
      - 向量通道级 dropout
    """
    def __init__(self,
                 in_dims: Tuple[int, int],
                 out_dims: Tuple[int, int],
                 h_dim_s: Optional[int] = None,
                 h_dim_v: Optional[int] = None,
                 s_act = F.silu,
                 dropout: float = 0.1,
                 eps: float = 1e-8):
        super().__init__()
        s_in, v_in = in_dims
        s_out, v_out = out_dims
        self.s_in, self.v_in, self.s_out, self.v_out = s_in, v_in, s_out, v_out
        self.eps = eps
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()
        self.vector_dropout = VectorDropout(dropout) if dropout and dropout > 0 else nn.Identity()
        self.s_act = s_act
        # 向量线性（对 xyz 共享权重）
        self.vh = max(v_in, 1) if h_dim_v is None else h_dim_v
        if v_in > 0 and self.vh > 0:
            self.W_vh = nn.Parameter(torch.empty(v_in, self.vh))
            nn.init.xavier_uniform_(self.W_vh)
        else:
            self.register_parameter('W_vh', None)
        if self.vh > 0 and v_out > 0:
            self.W_hv = nn.Parameter(torch.empty(self.vh, v_out))
            nn.init.xavier_uniform_(self.W_hv)
        else:
            self.register_parameter('W_hv', None)
        # 标量分支
        s_concat = s_in + (self.vh if self.vh > 0 else 0)
        h_dim_s = h_dim_s or max(s_out, 1)
        self.W_s_1 = nn.Linear(s_concat, h_dim_s) if s_out > 0 else nn.Identity()
        self.W_s_2 = nn.Linear(h_dim_s, s_out)     if s_out > 0 else nn.Identity()
        # 向量门控
        if s_out > 0 and v_out > 0:
            self.W_g = nn.Linear(s_out, v_out)
        else:
            self.W_g = None

    def forward(self,
                x: Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]
                ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        s, v = x  # s:[N,s_in], v:[N,v_in,3]
        # 向量 -> 中间维
        if self.W_vh is not None and v is not None:
            Vh = torch.einsum('n i c, i h -> n h c', v, self.W_vh)   # [N, vh, 3]
            vhn = _safe_norm(Vh, dim=-1, eps=self.eps)               # [N, vh]
        else:
            Vh, vhn = None, None
        # 标量分支
        if isinstance(self.W_s_2, nn.Linear):  # s_out > 0
            s_inp = s if vhn is None else torch.cat([s, vhn], dim=-1)
            h = self.s_act(self.W_s_1(s_inp))
            h = self.dropout(h)
            s_out = self.W_s_2(h)                                   # [N, s_out]
        else:
            s_out = None
        # 向量分支
        if self.W_hv is not None and Vh is not None:
            Vo = torch.einsum('n h c, h o -> n o c', Vh, self.W_hv)  # [N, v_out, 3]
            Vo_norm = _safe_norm(Vo, dim=-1, eps=self.eps)           # [N, v_out]
            Vo_dir  = Vo / (Vo_norm.unsqueeze(-1) + self.eps)
            if self.W_g is not None and s_out is not None:
                gate = torch.sigmoid(self.W_g(s_out)).unsqueeze(-1)  # [N, v_out, 1]
            else:
                gate = 1.0
            v_out = Vo_dir * gate
            v_out = self.vector_dropout(v_out)
        else:
            v_out = None
        return s_out, v_out


class GVPConvLayer(nn.Module):
    def __init__(self,
                 s_dim: int = 100,
                 v_dim: int = 16,
                 edge_dim: int = 32,
                 dropout: float = 0.1,
                 aggr: str = 'sum',
                 msg_vh: Optional[int] = None,
                 upd_vh: Optional[int] = None,
                 skip_edge_dims: int = 0):
        super().__init__()
        assert aggr in ['mean', 'sum']
        self.s_dim, self.v_dim = s_dim, v_dim
        self.edge_dim = edge_dim
        self.skip_edge_dims = skip_edge_dims
        self.aggr = aggr
        self.dropout = nn.Dropout(dropout)
        hid = max(128, s_dim)
        # 边标量编码
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_dim - skip_edge_dims, hid),
            nn.SiLU(),
            nn.Linear(hid, s_dim)
        )
        # 消息：2层GVP，处理 (s_j ⊕ e_s, v_j ⊕ e_v) -> (m_s, m_v)
        self.msg_gvp1 = GVP(in_dims=(s_dim + s_dim, v_dim + 1),
                            out_dims=(s_dim, v_dim),
                            h_dim_s=s_dim, h_dim_v=msg_vh, dropout=dropout)
        self.msg_gvp2 = GVP(in_dims=(s_dim, v_dim),
                            out_dims=(s_dim, v_dim),
                            h_dim_s=s_dim, h_dim_v=msg_vh, dropout=dropout)
        # 更新：2层GVP，输入 (s_i ⊕ agg(m_s), v_i ⊕ agg(m_v)) -> (s', v')
        self.upd_gvp1 = GVP(in_dims=(s_dim + s_dim, v_dim + v_dim),
                            out_dims=(s_dim, v_dim),
                            h_dim_s=s_dim, h_dim_v=upd_vh, dropout=dropout)
        self.upd_gvp2 = GVP(in_dims=(s_dim, v_dim),
                            out_dims=(s_dim, v_dim),
                            h_dim_s=s_dim, h_dim_v=upd_vh, dropout=dropout)
        self.norm = LayerNormSV(s_dim, v_dim)

    def forward(self,
                s: torch.Tensor,                 # [N, s_dim]
                v: torch.Tensor,                 # [N, v_dim, 3]
                pos: torch.Tensor,               # [N, 3]
                edge_index: torch.Tensor,        # [2, E]  (j -> i)
                edge_attr: Optional[torch.Tensor] = None  # [E, edge_dim]
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if edge_index.size(1) == 0:
            s_out, v_out = self.norm(s, v)
            return s_out, v_out, pos      
        if edge_attr is None:
            edge_attr = torch.zeros((edge_index.size(1), self.edge_dim),
                                    device=pos.device, dtype=pos.dtype)

        i, j = edge_index  # j -> i
        rij = pos[i] - pos[j]                         # [E,3]
        dij = torch.linalg.norm(rij, dim=-1) + 1e-8   # [E]
        uij = rij / dij.unsqueeze(-1)                 # [E,3]

        if edge_attr.size(1) >= self.skip_edge_dims:
            edge_s_features = edge_attr[:, self.skip_edge_dims:]  # [E, edge_dim-skip_edge_dims]
        else:
            edge_s_features = edge_attr
        e_s = self.edge_mlp(edge_s_features)  # [E, s_dim]
        e_v = uij.unsqueeze(1)  # [E, 1, 3]
        s_src = torch.cat([s[j], e_s], dim=-1)  # [E, s_dim + s_dim]
        v_src = torch.cat([v[j], e_v], dim=-2)  # [E, v_dim + 1, 3]

        m_s, m_v = self.msg_gvp1((s_src, v_src))       # [E, s_dim], [E, v_dim, 3]
        m_s, m_v = self.msg_gvp2((m_s, m_v))           # [E, s_dim], [E, v_dim, 3]
        if self.aggr == 'sum':
            reduce = 'sum'
        else:
            reduce = 'mean'

        ms = torch.zeros_like(s)
        ms.index_add_(0, i, m_s)
        if reduce == 'mean':
            deg = torch.zeros(s.size(0), device=s.device, dtype=s.dtype)
            deg.index_add_(0, i, torch.ones(m_s.size(0), device=s.device, dtype=s.dtype))
            ms = ms / (deg.clamp_min(1.0).unsqueeze(-1))

        mv = torch.zeros_like(v)
        mv.index_add_(0, i, m_v)
        if reduce == 'mean':
            deg_v = torch.zeros(v.size(0), device=v.device, dtype=v.dtype)
            deg_v.index_add_(0, i, torch.ones(m_v.size(0), device=v.device, dtype=v.dtype))
            mv = mv / (deg_v.clamp_min(1.0).unsqueeze(-1).unsqueeze(-1))

        s_inp = torch.cat([s, ms], dim=-1)
        v_inp = torch.cat([v, mv], dim=-2)
        s_upd, v_upd = self.upd_gvp1((s_inp, v_inp))    # 第一层更新GVP
        s_upd, v_upd = self.upd_gvp2((s_upd, v_upd))    # 第二层更新GVP
        s_out = s + self.dropout(s_upd)
        v_out = v + v_upd
        s_out, v_out = self.norm(s_out, v_out)

        return s_out, v_out, pos

class MacroBlock(nn.Module):
    def __init__(self, sA=128, sR=128, edgeA_dim=128, edgeR_dim=128,
                 mode='atom_only', atom_encoder='gvp', res_encoder='gvp'):
        super().__init__()
        self.mode = mode
        self.atom_encoder = atom_encoder
        self.res_encoder = res_encoder
            
        if mode == 'atom_only':
            if atom_encoder == 'gvp':
                self.atom1 = GVPConvLayer(s_dim=sA, v_dim=Cfg.atom_vdim,
                                         edge_dim=edgeA_dim, dropout=Cfg.dropout,
                                         skip_edge_dims=8)
            else:
                raise ValueError(f"Unsupported atom_encoder: {atom_encoder}. Must be 'gvp'")
        elif mode == 'residue_only':
            if res_encoder == 'gvp':
                self.res1 = GVPConvLayer(s_dim=sR, v_dim=Cfg.res_vdim,
                                        edge_dim=edgeR_dim, dropout=Cfg.dropout,skip_edge_dims=4)
            else:
                raise ValueError(f"Unsupported res_encoder: {res_encoder}. Must be 'gvp'")
        
    def forward(self, A, R):
        if self.mode == 'atom_only':
            A['s'], A['v'], A['pos'] = self.atom1(A['s'], A['v'], A['pos'], A['edge_index'], A['edge_attr'])
        elif self.mode == 'residue_only':
            sR, vR, posR = self.res1(R['s'], R['v'], R['pos'], R['edge_index'], R['edge_attr_s'])
            R['s'], R['pos'], R['v'] = sR, posR, vR
        
        return A, R

class MeanPool(nn.Module):

    def __init__(self, in_dim):
        super().__init__()
        self.in_dim = in_dim

    def forward(self, s, pos, batch, mutation_mask=None):

        if batch.numel() == 0:
            return (torch.zeros(0, self.in_dim, device=s.device, dtype=s.dtype),
                    [])
        B = int(batch.max().item() + 1)
        pooled_list = []
        attn_all = []

        for b in range(B):
            mask_b = (batch == b)
            s_b = s[mask_b]  # [N_g, F]           
            if s_b.size(0) > 0:
                pooled_b = s_b.mean(dim=0)  # [F]
            else:
                pooled_b = torch.zeros(self.in_dim, device=s.device, dtype=s.dtype)            
            pooled_list.append(pooled_b)
            attn_all.append([])
        pooled = torch.stack(pooled_list, dim=0)  # [B, F]
        return pooled, attn_all

# ---------- Siamese 对比式 ddG 头 ----------
class ContrastiveDDGHead(nn.Module):
    """
    先把 WT/MUT 各自投影到 latent,再做 [mut, wt, mut-wt] 融合预测 ddG
    """
    def __init__(self, in_dim, hidden=256, dropout=0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.SiLU()
        )
        fuse_dim = hidden * 3
        self.pred = nn.Sequential(
            nn.LayerNorm(fuse_dim),
            nn.Linear(fuse_dim, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1)
        )

    def forward(self, wt_emb, mut_emb):

        z_wt  = self.proj(wt_emb)   # [B,H]
        z_mut = self.proj(mut_emb)  # [B,H]
        z_diff = z_mut - z_wt
        z_fuse = torch.cat([z_mut, z_wt, z_diff], dim=-1)  # [B, 3H]
        ddg = self.pred(z_fuse).squeeze(-1)  # [B]
        return ddg


class SeparatedDDGHead(nn.Module):
    """
    分离式 ddG 预测头：
    - 图嵌入:Siamese 对比方式处理
    - ESM 嵌入：使用和图一样的 Siamese 对比方式处理
    - 最后相加得到 ddG 预测
    """
    def __init__(self, graph_dim, esm_dim, hidden=256, dropout=0.1):
        super().__init__()
        # 统一图嵌入投影
        self.graph_proj = nn.Sequential(
            nn.LayerNorm(graph_dim),
            nn.Linear(graph_dim, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.SiLU()
        )
        # 图嵌入预测头
        fuse_dim = hidden * 3
        self.pred = nn.Sequential(
            nn.LayerNorm(fuse_dim),
            nn.Linear(fuse_dim, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1)
        )       
        # ESM 嵌入投影
        self.esm_proj = nn.Sequential(
            nn.LayerNorm(esm_dim),
            nn.Linear(esm_dim, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.SiLU()
        )        
        # ESM 预测头
        fuse_dim_esm = hidden * 3
        self.esm_pred = nn.Sequential(
            nn.LayerNorm(fuse_dim_esm),
            nn.Linear(fuse_dim_esm, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1)
        )

    def forward(self, wt_graph_emb, mut_graph_emb, wt_esm_emb, mut_esm_emb):
        # 图嵌入：Siamese 对比方式
        z_wt_graph = self.graph_proj(wt_graph_emb)   # [B,H]
        z_mut_graph = self.graph_proj(mut_graph_emb)  # [B,H]
        z_diff_graph = z_mut_graph - z_wt_graph
        z_fuse_graph = torch.cat([z_mut_graph, z_wt_graph, z_diff_graph], dim=-1)  # [B, 3*H]
        ddg_graph = self.pred(z_fuse_graph).squeeze(-1)  # [B]        
        # ESM 嵌入：Siamese 对比方式
        z_wt_esm = self.esm_proj(wt_esm_emb)   # [B,H]
        z_mut_esm = self.esm_proj(mut_esm_emb)  # [B,H]
        z_diff_esm = z_mut_esm - z_wt_esm
        z_fuse_esm = torch.cat([z_mut_esm, z_wt_esm, z_diff_esm], dim=-1)  # [B, 3*H]
        ddg_esm = self.esm_pred(z_fuse_esm).squeeze(-1)  # [B]        
        # 相加得到最终预测
        ddg = ddg_graph + ddg_esm  # [B]
        return ddg

class ESMFusion(nn.Module):
    """
    ESM 残基级融合：
    - 各侧先做自注意力（残基级）+ FFN
    - 然后用双向交叉注意力(ab->ag 与 ag->ab),分别用独立的 MHA 层
    - 所有注意力采用 Pre-LN
    - 对每侧的查询输出使用 mean 池化，得到两个向量并拼接输出 [B, 2*out_dim]
    """
    def __init__(self, esm_in_dim: int = 1280, out_dim: int = 128, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.out_dim = out_dim
        self.proj = nn.Linear(esm_in_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.token_norm = nn.LayerNorm(out_dim)        
        # 第一层：自注意力层
        self.mha_self_ab_1 = nn.MultiheadAttention(out_dim, num_heads, dropout=dropout, batch_first=True)
        self.mha_self_ag_1 = nn.MultiheadAttention(out_dim, num_heads, dropout=dropout, batch_first=True)        
        # 第一层：双向交叉注意力层
        self.mha_cross_ab_to_ag_1 = nn.MultiheadAttention(out_dim, num_heads, dropout=dropout, batch_first=True)
        self.mha_cross_ag_to_ab_1 = nn.MultiheadAttention(out_dim, num_heads, dropout=dropout, batch_first=True)        
        # 第二层：自注意力层
        self.mha_self_ab_2 = nn.MultiheadAttention(out_dim, num_heads, dropout=dropout, batch_first=True)
        self.mha_self_ag_2 = nn.MultiheadAttention(out_dim, num_heads, dropout=dropout, batch_first=True)        
        # 第二层：双向交叉注意力层
        self.mha_cross_ab_to_ag_2 = nn.MultiheadAttention(out_dim, num_heads, dropout=dropout, batch_first=True)
        self.mha_cross_ag_to_ab_2 = nn.MultiheadAttention(out_dim, num_heads, dropout=dropout, batch_first=True)        
        # Pre-Norm 层
        self.ln_q_ab_1 = nn.LayerNorm(out_dim)
        self.ln_kv_ag_1 = nn.LayerNorm(out_dim)
        self.ln_q_ag_1 = nn.LayerNorm(out_dim)
        self.ln_kv_ab_1 = nn.LayerNorm(out_dim)
        self.ln_q_ab_2 = nn.LayerNorm(out_dim)
        self.ln_kv_ag_2 = nn.LayerNorm(out_dim)
        self.ln_q_ag_2 = nn.LayerNorm(out_dim)
        self.ln_kv_ab_2 = nn.LayerNorm(out_dim)        
        # FFN
        self.ffn_ab_1 = nn.Sequential(
            nn.Linear(out_dim, 4 * out_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(4 * out_dim, out_dim)
        )
        self.ffn_ag_1 = nn.Sequential(
            nn.Linear(out_dim, 4 * out_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(4 * out_dim, out_dim)
        )
        self.ffn_ab_2 = nn.Sequential(
            nn.Linear(out_dim, 4 * out_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(4 * out_dim, out_dim)
        )
        self.ffn_ag_2 = nn.Sequential(
            nn.Linear(out_dim, 4 * out_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(4 * out_dim, out_dim)
        )        
        # 残差后的 LayerNorm
        self.ln_res_ab_1 = nn.LayerNorm(out_dim)
        self.ln_res_ag_1 = nn.LayerNorm(out_dim)
        self.ln_res_ab_2 = nn.LayerNorm(out_dim)
        self.ln_res_ag_2 = nn.LayerNorm(out_dim)

    def forward(self, esm_ab, esm_ab_pad, esm_ag, esm_ag_pad) -> torch.Tensor:
        # 1) 预处理：直接投影和标准化
        ab_seq = self.token_norm(self.dropout(self.proj(esm_ab)))
        ag_seq = self.token_norm(self.dropout(self.proj(esm_ag)))
        # 2) 第一层：自注意力 + FFN（Pre-LN）
        # ab 自注意力（第一层）
        B, L, D = ab_seq.shape
        has_valid_ab = (~esm_ab_pad).any(dim=1)
        idx_ab = has_valid_ab.nonzero(as_tuple=False).view(-1)
        qkv_ab = ab_seq.index_select(0, idx_ab)
        m_sel_ab = esm_ab_pad.index_select(0, idx_ab)
        qkv_ab_norm = self.ln_res_ab_1(qkv_ab)
        o_ab, _ = self.mha_self_ab_1(qkv_ab_norm, qkv_ab_norm, qkv_ab_norm, key_padding_mask=m_sel_ab)
        o_ab = qkv_ab + self.dropout(o_ab)
        o_ab_norm = self.ln_res_ab_1(o_ab)
        o_ab = o_ab + self.dropout(self.ffn_ab_1(o_ab_norm))
        ab_seq.index_copy_(0, idx_ab, o_ab)        
        # ag 自注意力（第一层）
        has_valid_ag = (~esm_ag_pad).any(dim=1)
        idx_ag = has_valid_ag.nonzero(as_tuple=False).view(-1)
        qkv_ag = ag_seq.index_select(0, idx_ag)
        m_sel_ag = esm_ag_pad.index_select(0, idx_ag)
        qkv_ag_norm = self.ln_res_ag_1(qkv_ag)
        o_ag, _ = self.mha_self_ag_1(qkv_ag_norm, qkv_ag_norm, qkv_ag_norm, key_padding_mask=m_sel_ag)
        o_ag = qkv_ag + self.dropout(o_ag)
        o_ag_norm = self.ln_res_ag_1(o_ag)
        o_ag = o_ag + self.dropout(self.ffn_ag_1(o_ag_norm))
        ag_seq.index_copy_(0, idx_ag, o_ag)
        # 3) 第一层：双向交叉注意力
        # ab -> ag（第一层）
        has_valid_ag = (~esm_ag_pad).any(dim=1)
        idx_ab = has_valid_ag.nonzero(as_tuple=False).view(-1)
        q_ab = ab_seq.index_select(0, idx_ab)
        kv_ag = ag_seq.index_select(0, idx_ab)
        pad_k_ag = esm_ag_pad.index_select(0, idx_ab)        
        q_ab_norm = self.ln_q_ab_1(q_ab)
        kv_ag_norm = self.ln_kv_ag_1(kv_ag)
        out_q, _ = self.mha_cross_ab_to_ag_1(q_ab_norm, kv_ag_norm, kv_ag_norm, key_padding_mask=pad_k_ag)
        out_q = q_ab + self.dropout(out_q)        
        out_q_norm = self.ln_res_ab_1(out_q)
        out_q = out_q + self.dropout(self.ffn_ab_1(out_q_norm))
        ab_seq.index_copy_(0, idx_ab, out_q)
        # ag -> ab（第一层）
        has_valid_ab = (~esm_ab_pad).any(dim=1)
        idx_ag = has_valid_ab.nonzero(as_tuple=False).view(-1)
        q_ag = ag_seq.index_select(0, idx_ag)
        kv_ab = ab_seq.index_select(0, idx_ag)
        pad_k_ab = esm_ab_pad.index_select(0, idx_ag)        
        q_ag_norm = self.ln_q_ag_1(q_ag)
        kv_ab_norm = self.ln_kv_ab_1(kv_ab)
        out_q, _ = self.mha_cross_ag_to_ab_1(q_ag_norm, kv_ab_norm, kv_ab_norm, key_padding_mask=pad_k_ab)
        out_q = q_ag + self.dropout(out_q)        
        out_q_norm = self.ln_res_ag_1(out_q)
        out_q = out_q + self.dropout(self.ffn_ag_1(out_q_norm))
        ag_seq.index_copy_(0, idx_ag, out_q)
        # 4) 第二层：自注意力 + FFN（Pre-LN）
        # ab 自注意力（第二层）
        has_valid_ab = (~esm_ab_pad).any(dim=1)
        idx_ab = has_valid_ab.nonzero(as_tuple=False).view(-1)
        qkv_ab = ab_seq.index_select(0, idx_ab)
        m_sel_ab = esm_ab_pad.index_select(0, idx_ab)
        qkv_ab_norm = self.ln_res_ab_2(qkv_ab)
        o_ab, _ = self.mha_self_ab_2(qkv_ab_norm, qkv_ab_norm, qkv_ab_norm, key_padding_mask=m_sel_ab)
        o_ab = qkv_ab + self.dropout(o_ab)
        o_ab_norm = self.ln_res_ab_2(o_ab)
        o_ab = o_ab + self.dropout(self.ffn_ab_2(o_ab_norm))
        ab_seq.index_copy_(0, idx_ab, o_ab)        
        # ag 自注意力（第二层）
        has_valid_ag = (~esm_ag_pad).any(dim=1)
        idx_ag = has_valid_ag.nonzero(as_tuple=False).view(-1)
        qkv_ag = ag_seq.index_select(0, idx_ag)
        m_sel_ag = esm_ag_pad.index_select(0, idx_ag)
        qkv_ag_norm = self.ln_res_ag_2(qkv_ag)
        o_ag, _ = self.mha_self_ag_2(qkv_ag_norm, qkv_ag_norm, qkv_ag_norm, key_padding_mask=m_sel_ag)
        o_ag = qkv_ag + self.dropout(o_ag)
        o_ag_norm = self.ln_res_ag_2(o_ag)
        o_ag = o_ag + self.dropout(self.ffn_ag_2(o_ag_norm))
        ag_seq.index_copy_(0, idx_ag, o_ag)
        # 5) 第二层：双向交叉注意力（最终输出）
        # ab -> ag（第二层，最终输出）
        has_valid_ag = (~esm_ag_pad).any(dim=1)
        idx_ab = has_valid_ag.nonzero(as_tuple=False).view(-1)
        q_ab = ab_seq.index_select(0, idx_ab)
        kv_ag = ag_seq.index_select(0, idx_ab)
        pad_k_ag = esm_ag_pad.index_select(0, idx_ab)        
        q_ab_norm = self.ln_q_ab_2(q_ab)
        kv_ag_norm = self.ln_kv_ag_2(kv_ag)
        out_ab, _ = self.mha_cross_ab_to_ag_2(q_ab_norm, kv_ag_norm, kv_ag_norm, key_padding_mask=pad_k_ag)
        out_ab = q_ab + self.dropout(out_ab)        
        out_ab_norm = self.ln_res_ab_2(out_ab)
        out_ab = out_ab + self.dropout(self.ffn_ab_2(out_ab_norm))
        # ag -> ab（第二层，最终输出）
        has_valid_ab = (~esm_ab_pad).any(dim=1)
        idx_ag = has_valid_ab.nonzero(as_tuple=False).view(-1)
        q_ag = ag_seq.index_select(0, idx_ag)
        kv_ab = ab_seq.index_select(0, idx_ag)
        pad_k_ab = esm_ab_pad.index_select(0, idx_ag)        
        q_ag_norm = self.ln_q_ag_2(q_ag)
        kv_ab_norm = self.ln_kv_ab_2(kv_ab)
        out_ag, _ = self.mha_cross_ag_to_ab_2(q_ag_norm, kv_ab_norm, kv_ab_norm, key_padding_mask=pad_k_ab)
        out_ag = q_ag + self.dropout(out_ag)        
        out_ag_norm = self.ln_res_ag_2(out_ag)
        out_ag = out_ag + self.dropout(self.ffn_ag_2(out_ag_norm))
        # 使用 mean 池化
        # ab 侧：将out_ab写回ab_seq
        if idx_ab.numel() > 0:
            ab_seq.index_copy_(0, idx_ab, out_ab)
        # ag 侧：将out_ag写回ag_seq
        if idx_ag.numel() > 0:
            ag_seq.index_copy_(0, idx_ag, out_ag)       
        # 使用 mean 池化
        B = ab_seq.size(0)
        device = ab_seq.device
        pooled_ab = torch.zeros(B, self.out_dim, device=device, dtype=ab_seq.dtype)
        pooled_ag = torch.zeros(B, self.out_dim, device=device, dtype=ag_seq.dtype)       
        for b in range(B):
            # ab 侧 mean 池化
            pad_ab_b = esm_ab_pad[b]  # [L_ab]
            valid_ab = ~pad_ab_b  # [L_ab]
            if valid_ab.any():
                ab_seq_b = ab_seq[b]  # [L_ab, D]
                pooled_ab[b] = ab_seq_b[valid_ab].mean(dim=0)  # [D]            
            # ag 侧 mean 池化
            pad_ag_b = esm_ag_pad[b]  # [L_ag]
            valid_ag = ~pad_ag_b  # [L_ag]
            if valid_ag.any():
                ag_seq_b = ag_seq[b]  # [L_ag, D]
                pooled_ag[b] = ag_seq_b[valid_ag].mean(dim=0)  # [D]        
        return torch.cat([pooled_ab, pooled_ag], dim=-1)  # [B, 2*out_dim]

class DDGPredictor(nn.Module):

    def __init__(self, sA=128, sR=128, n_blocks=3,
                 edgeA_dim=128, edgeR_dim=128,
                 sA_in_dim=19, sR_in_dim=47, hidden_dim=256, mode='residue_only',
                 atom_encoder='gvp', res_encoder='gvp',
                 dropout=0.1,
                 use_esm=None,
                 use_graph=None):
        super().__init__()
        self.sA, self.sR = sA, sR
        self.mode = mode
        self.atom_encoder = atom_encoder
        self.res_encoder = res_encoder
        self.n_blocks = n_blocks
        self.use_esm = use_esm if use_esm is not None else Cfg.use_esm
        self.use_graph = use_graph if use_graph is not None else Cfg.use_graph
        # 输入嵌入
        if mode == 'atom_only':
            self.atom_embed = nn.Sequential(nn.Linear(sA_in_dim, sA), nn.SiLU(), nn.Linear(sA, sA))
        elif mode == 'residue_only':
            self.res_embed  = nn.Sequential(nn.Linear(sR_in_dim, sR), nn.SiLU(), nn.Linear(sR, sR))
        elif mode == 'dual':
            self.atom_embed = nn.Sequential(nn.Linear(sA_in_dim, sA), nn.SiLU(), nn.Linear(sA, sA))
            self.res_embed  = nn.Sequential(nn.Linear(sR_in_dim, sR), nn.SiLU(), nn.Linear(sR, sR))
        else:
            raise ValueError(f"Unsupported mode: {mode}. Must be 'atom_only', 'residue_only', or 'dual'")
        if self.use_graph:
            if mode == 'dual':
                self.atom_blocks = nn.ModuleList([
                    MacroBlock(sA, sR, edgeA_dim, edgeR_dim,
                              mode='atom_only', atom_encoder=atom_encoder, res_encoder=res_encoder)
                    for _ in range(n_blocks)
                ])
                self.residue_blocks = nn.ModuleList([
                    MacroBlock(sA, sR, edgeA_dim, edgeR_dim,
                              mode='residue_only', atom_encoder=atom_encoder, res_encoder=res_encoder)
                    for _ in range(n_blocks)
                ])
            else:
                self.blocks = nn.ModuleList([
                    MacroBlock(sA, sR, edgeA_dim, edgeR_dim,
                              mode=mode, atom_encoder=atom_encoder, res_encoder=res_encoder)
                    for _ in range(n_blocks)
                ])
        else:
            if mode == 'dual':
                self.atom_blocks = nn.ModuleList()
                self.residue_blocks = nn.ModuleList()
            else:
                self.blocks = nn.ModuleList()
        if mode == 'atom_only':
            pool_in_dim = sA
            graph_emb_dim = pool_in_dim
            self.pool_in_dim = pool_in_dim
            self.pool = MeanPool(pool_in_dim)
        elif mode == 'residue_only':
            pool_in_dim = sR
            graph_emb_dim = pool_in_dim
            self.pool_in_dim = pool_in_dim
            self.pool = MeanPool(pool_in_dim)
        elif mode == 'dual':
            atom_pool_in_dim = sA
            res_pool_in_dim = sR
            atom_graph_emb_dim = atom_pool_in_dim
            res_graph_emb_dim = res_pool_in_dim
            self.pool_in_dim = res_pool_in_dim
            self.atom_pool = MeanPool(atom_pool_in_dim)
            self.res_pool = MeanPool(res_pool_in_dim)
        else:
            raise ValueError(f"Unsupported mode: {mode}")
        if mode == 'dual':
            esm_out_dim = res_pool_in_dim
        else:
            esm_out_dim = pool_in_dim      
        if self.use_esm:
            self.esm_fusion = ESMFusion(
                esm_in_dim=1280, 
                out_dim=esm_out_dim, 
                num_heads=4,
                dropout=dropout
            )
            esm_fusion_out_dim = 2 * esm_out_dim
        else:
            self.esm_fusion = None
            esm_fusion_out_dim = 0       
        # 预测头
        if mode == 'dual':
            self.atom_pred = nn.Sequential(
                nn.LayerNorm(3 * atom_graph_emb_dim),
                nn.Linear(3 * atom_graph_emb_dim, hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1)
            )
            self.res_pred = nn.Sequential(
                nn.LayerNorm(3 * res_graph_emb_dim),
                nn.Linear(3 * res_graph_emb_dim, hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1)
            )
            if self.use_esm:
                self.esm_proj = nn.Sequential(
                    nn.LayerNorm(esm_fusion_out_dim),
                    nn.Linear(esm_fusion_out_dim, hidden_dim),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.SiLU()
                )
                fuse_dim_esm = hidden_dim * 3
                self.esm_pred = nn.Sequential(
                    nn.LayerNorm(fuse_dim_esm),
                    nn.Linear(fuse_dim_esm, hidden_dim),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, 1)
                )
            else:
                self.esm_proj = None
                self.esm_pred = None           
        else:
            if self.use_esm:
                self.head = SeparatedDDGHead(
                    graph_dim=graph_emb_dim, esm_dim=esm_fusion_out_dim,
                    hidden=hidden_dim, dropout=dropout
                )
            else:
                self.head = ContrastiveDDGHead(
                    in_dim=graph_emb_dim,
                    hidden=hidden_dim,
                    dropout=dropout
                )           

    def encode_single_graph(self, A, R):
        if self.mode == 'atom_only':
            A['s'] = self.atom_embed(A['x_scalar'])
            if self.use_graph and self.atom_encoder == 'gvp':
                A['v'] = A['x_vector']
            else:
                A['v'] = torch.zeros(A['s'].size(0), A['s'].size(1), 3, device=A['s'].device, dtype=A['s'].dtype)
            if self.use_graph:
                for blk in self.blocks:
                    A, R = blk(A, R)
        elif self.mode == 'residue_only':
            R['s'] = self.res_embed(R['x_scalar'])
            if self.use_graph and self.res_encoder == 'gvp':
                R['v'] = R['x_vector']
            else:
                R['v'] = torch.zeros(R['s'].size(0), R['s'].size(1), 3, device=R['s'].device, dtype=R['s'].dtype)
            if self.use_graph:
                for blk in self.blocks:
                    A, R = blk(A, R)
        elif self.mode == 'dual':
            A['s'] = self.atom_embed(A['x_scalar'])
            if self.use_graph and self.atom_encoder == 'gvp':
                A['v'] = A['x_vector']
            else:
                A['v'] = torch.zeros(A['s'].size(0), A['s'].size(1), 3, device=A['s'].device, dtype=A['s'].dtype)
            R['s'] = self.res_embed(R['x_scalar'])
            if self.use_graph and self.res_encoder == 'gvp':
                R['v'] = R['x_vector']
            else:
                R['v'] = torch.zeros(R['s'].size(0), R['s'].size(1), 3, device=R['s'].device, dtype=R['s'].dtype)
            if self.use_graph:
                for atom_blk, res_blk in zip(self.atom_blocks, self.residue_blocks):
                    A, _ = atom_blk(A, R)
                    _, R = res_blk(A, R)
        return A, R

    def _pool_batch(self, A, R):
        if self.mode == 'atom_only':
            return self.pool(A['s'], A['pos'], A['batch'], A.get('mutation_mask', None))
        elif self.mode == 'residue_only':
            return self.pool(R['s'], R['pos'], R['batch'], R.get('mutation_mask', None))
        elif self.mode == 'dual':
            # 原子图和残基图分别池化
            atom_emb, atom_attn = self.atom_pool(A['s'], A['pos'], A['batch'], A.get('mutation_mask', None))
            res_emb, res_attn = self.res_pool(R['s'], R['pos'], R['batch'], R.get('mutation_mask', None))
            return (atom_emb, res_emb), (atom_attn, res_attn)
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")
    
    def forward(self, A_wt, R_wt, A_mut, R_mut):
        # 1) 编码 WT / MUT
        A_wt, R_wt = self.encode_single_graph(A_wt, R_wt)
        A_mut, R_mut = self.encode_single_graph(A_mut, R_mut)
        # 2) 池化
        if self.mode == 'dual':
            (wt_atom_emb, wt_res_emb), (wt_atom_attn, wt_res_attn) = self._pool_batch(A_wt, R_wt)
            (mut_atom_emb, mut_res_emb), (mut_atom_attn, mut_res_attn) = self._pool_batch(A_mut, R_mut)
            batch_size = wt_atom_emb.size(0)
            device = wt_atom_emb.device
            dtype = wt_atom_emb.dtype
        else:
            wt_emb, wt_attn = self._pool_batch(A_wt, R_wt)   # [B, 2*D]
            mut_emb, mut_attn = self._pool_batch(A_mut, R_mut)
            batch_size = wt_emb.size(0)
            device = wt_emb.device
            dtype = wt_emb.dtype
        # 3) ESM 融合
        if self.use_esm:
            def _esm_vec(R_batch, batch_size, device, dtype):
                esm_ab = R_batch.get('esm_ab')
                esm_ab_pad = R_batch.get('esm_ab_pad')
                esm_ag = R_batch.get('esm_ag')
                esm_ag_pad = R_batch.get('esm_ag_pad')
                if esm_ab is None or esm_ag is None:
                    esm_fusion_out_dim = 2 * (self.pool_in_dim if self.mode != 'dual' else getattr(self, 'pool_in_dim', 128))
                    return torch.zeros(batch_size, esm_fusion_out_dim, device=device, dtype=dtype)                
                fused = self.esm_fusion(esm_ab, esm_ab_pad, esm_ag, esm_ag_pad)
                return fused  # [B, esm_fusion_out_dim]
            esm_wt = _esm_vec(R_wt, batch_size, device, dtype)
            esm_mut = _esm_vec(R_mut, batch_size, device, dtype)
        else:
            esm_wt = None
            esm_mut = None
        # 4) 预测
        if self.mode == 'dual':
            atom_wt_emb = wt_atom_emb
            atom_mut_emb = mut_atom_emb
            atom_diff_emb = atom_mut_emb - atom_wt_emb
            atom_fuse = torch.cat([atom_mut_emb, atom_wt_emb, atom_diff_emb], dim=-1)
            ddg_atom = self.atom_pred(atom_fuse)
            ddg_atom = ddg_atom.squeeze()
            if ddg_atom.dim() == 0:
                ddg_atom = ddg_atom.unsqueeze(0)            
            res_wt_emb = wt_res_emb
            res_mut_emb = mut_res_emb
            res_diff_emb = res_mut_emb - res_wt_emb
            res_fuse = torch.cat([res_mut_emb, res_wt_emb, res_diff_emb], dim=-1)
            ddg_res = self.res_pred(res_fuse)
            ddg_res = ddg_res.squeeze()
            if ddg_res.dim() == 0:
                ddg_res = ddg_res.unsqueeze(0)            
            if self.use_esm and self.esm_pred is not None:
                # ESM Siamese对比方式
                z_wt_esm = self.esm_proj(esm_wt)   # [B, H]
                z_mut_esm = self.esm_proj(esm_mut)  # [B, H]
                z_diff_esm = z_mut_esm - z_wt_esm
                z_fuse_esm = torch.cat([z_mut_esm, z_wt_esm, z_diff_esm], dim=-1)  # [B, 3*H]
                ddg_esm = self.esm_pred(z_fuse_esm)
                ddg_esm = ddg_esm.squeeze()
                if ddg_esm.dim() == 0:
                    ddg_esm = ddg_esm.unsqueeze(0)
            else:
                ddg_esm = torch.zeros_like(ddg_atom)
            ddg_pred = ddg_atom + ddg_res + ddg_esm
            if ddg_pred.dim() == 0:
                ddg_pred = ddg_pred.unsqueeze(0)            
            aux = {
                'wt_atom_features': A_wt['s'], 'mut_atom_features': A_mut['s'],
                'wt_res_features': R_wt['s'], 'mut_res_features': R_mut['s'],
                'wt_atom_attn': wt_atom_attn, 'mut_atom_attn': mut_atom_attn,
                'wt_res_attn': wt_res_attn, 'mut_res_attn': mut_res_attn,
                'ddg_atom': ddg_atom, 'ddg_res': ddg_res, 'ddg_esm': ddg_esm
            }
            if self.use_esm:
                aux.update({'esm_wt': esm_wt, 'esm_mut': esm_mut})
        else:
            if self.use_esm:
                ddg_pred = self.head(wt_emb, mut_emb, esm_wt, esm_mut)
                z_wt_graph = self.head.graph_proj(wt_emb)   # [B,H]
                z_mut_graph = self.head.graph_proj(mut_emb)  # [B,H]
                z_diff_graph = z_mut_graph - z_wt_graph
                z_fuse_graph = torch.cat([z_mut_graph, z_wt_graph, z_diff_graph], dim=-1)  # [B, 3*H]
                ddg_graph = self.head.pred(z_fuse_graph)
                ddg_graph = ddg_graph.squeeze()
                if ddg_graph.dim() == 0:
                    ddg_graph = ddg_graph.unsqueeze(0)
                
                z_wt_esm = self.head.esm_proj(esm_wt)   # [B,H]
                z_mut_esm = self.head.esm_proj(esm_mut)  # [B,H]
                z_diff_esm = z_mut_esm - z_wt_esm
                z_fuse_esm = torch.cat([z_mut_esm, z_wt_esm, z_diff_esm], dim=-1)  # [B, 3*H]
                ddg_esm = self.head.esm_pred(z_fuse_esm)
                ddg_esm = ddg_esm.squeeze()
                if ddg_esm.dim() == 0:
                    ddg_esm = ddg_esm.unsqueeze(0)                
                if ddg_pred.dim() == 0:
                    ddg_pred = ddg_pred.unsqueeze(0)
            else:
                ddg_pred = self.head(wt_emb, mut_emb)
                ddg_graph = ddg_pred  # 占位符
                ddg_esm = torch.zeros_like(ddg_pred)  # 占位符
                if ddg_pred.dim() == 0:
                    ddg_pred = ddg_pred.unsqueeze(0)
                    ddg_graph = ddg_graph.unsqueeze(0)
                    ddg_esm = ddg_esm.unsqueeze(0)
            aux = {}
            if self.mode == 'atom_only':
                aux.update({'wt_atom_features': A_wt['s'], 'mut_atom_features': A_mut['s'],
                            'wt_attn': wt_attn, 'mut_attn': mut_attn})
            elif self.mode == 'residue_only':
                aux.update({'wt_res_features': R_wt['s'], 'mut_res_features': R_mut['s'],
                            'wt_attn': wt_attn, 'mut_attn': mut_attn})
            aux.update({
                'ddg_graph': ddg_graph, 'ddg_esm': ddg_esm
            })
            if self.use_esm:
                aux.update({'esm_wt': esm_wt, 'esm_mut': esm_mut})       
        return ddg_pred, aux

def _to_tensor(x, dtype=None, device=None):
    t = torch.as_tensor(x)
    if dtype is not None:
        t = t.to(dtype)
    if device is not None:
        t = t.to(device)
    return t

def _ensure_edge_index(ei, device, dtype=torch.long):
    ei = _to_tensor(ei, dtype=dtype, device=device)
    if ei.dim() != 2:
        raise ValueError(f"edge_index must be rank-2, got shape {tuple(ei.shape)}")
    if ei.shape[0] == 2:
        return ei.contiguous()
    if ei.shape[1] == 2:
        return ei.t().contiguous()
    raise ValueError(f"edge_index bad shape {tuple(ei.shape)}, expected [2,E] or [E,2]")

def build_AR_from_data(data):
    device = data.x_atom.device if isinstance(data.x_atom, torch.Tensor) else 'cpu'
    x_atom  = _to_tensor(data.x_atom,  torch.float32, device)
    x_res   = _to_tensor(data.x_residues, torch.float32, device)
    pos_atom= _to_tensor(data.atom_coord, torch.float32, device)
    pos_res = _to_tensor(data.residues_coord, torch.float32, device)
    edge_index_atom = _ensure_edge_index(data.edge_index_atom, device, torch.long)
    edge_index_res  = _ensure_edge_index(data.edge_index_residues, device, torch.long)
    edge_attr_atom = _to_tensor(data.edge_attr_atom, torch.float32, device)
    edge_attr_res  = _to_tensor(data.edge_attr_residues, torch.float32, device)
    atom_vector = _to_tensor(data.atom_vector, torch.float32, device) 
    residue_vector = _to_tensor(data.residue_vector, torch.float32, device) 
    edge_vector_atom = _to_tensor(data.edge_vector_atom, torch.float32, device)
    edge_vector_residues = _to_tensor(data.edge_vector_residues, torch.float32, device)   
    mutation_mask = _to_tensor(data.mutation_mask, torch.long, device)
    atom2res       = _to_tensor(data.residue_indices, torch.long, device).view(-1)
    batch_atom = torch.zeros(x_atom.size(0), dtype=torch.long, device=device)
    batch_res  = torch.zeros(x_res.size(0),  dtype=torch.long, device=device)

    A = dict(
        x_scalar = x_atom, pos = pos_atom,
        edge_index = edge_index_atom, edge_attr = edge_attr_atom,
        atom2res = atom2res,
        batch = batch_atom,
        x_vector = atom_vector,
        edge_vector = edge_vector_atom
    )
    R = dict(
        x_scalar = x_res, pos = pos_res,
        edge_index = edge_index_res, edge_attr_s = edge_attr_res,
        batch = batch_res,
        x_vector = residue_vector,
        edge_vector = edge_vector_residues,
        mutation_mask = mutation_mask
    )
    
    if mutation_mask.numel() > 0 and atom2res.numel() > 0:
        valid_mask = (atom2res >= 0) & (atom2res < mutation_mask.size(0))
        if valid_mask.any():
            A['mutation_mask'] = torch.zeros_like(atom2res, dtype=mutation_mask.dtype, device=mutation_mask.device)
            A['mutation_mask'][valid_mask] = mutation_mask[atom2res[valid_mask]]
        else:
            A['mutation_mask'] = torch.zeros_like(atom2res, dtype=mutation_mask.dtype, device=mutation_mask.device)
    else:
        A['mutation_mask'] = torch.zeros(atom2res.size(0), dtype=torch.long, device=atom2res.device) if atom2res.numel() > 0 else torch.zeros(0, dtype=torch.long, device=atom2res.device)
    return A, R

def batch_AR(A_list: List[dict], R_list: List[dict]):
    assert len(A_list) == len(R_list)
    batch_size = len(A_list)
    atom_offset = 0
    res_offset  = 0

    A_cat = dict(
        x_scalar=[], pos=[], edge_index=[], edge_attr=[],
        atom2res=[], batch=[],
        v=[], x_vector=[], edge_vector=[], mutation_mask=[]
    )
    R_cat = dict(
        x_scalar=[], pos=[], edge_index=[], edge_attr_s=[],
        batch=[], v=[],
        x_vector=[], edge_vector=[], mutation_mask=[]
    )

    for b_idx, (A, R) in enumerate(zip(A_list, R_list)):
        Na = A['x_scalar'].size(0)
        Nr = R['x_scalar'].size(0)
        Ei_atom = A['edge_index'].size(1)
        Ei_res = R['edge_index'].size(1)
        A_cat['x_scalar'].append(A['x_scalar'])
        A_cat['pos'].append(A['pos'])
        edge_attr = A['edge_attr']
        if edge_attr.numel() == 0 and len(edge_attr.shape) == 1:
            edge_attr = torch.empty((0, 22), dtype=edge_attr.dtype, device=edge_attr.device)
        A_cat['edge_attr'].append(edge_attr)
        if 'x_vector' in A and A['x_vector'] is not None:
            A_cat['x_vector'].append(A['x_vector'])
        if 'edge_vector' in A and A['edge_vector'] is not None:
            A_cat['edge_vector'].append(A['edge_vector'])
        eiA = A['edge_index'] + atom_offset
        A_cat['edge_index'].append(eiA)
        A_cat['atom2res'].append(A['atom2res'] + res_offset)
        A_cat['batch'].append(torch.full((Na,), b_idx, dtype=torch.long, device=A['x_scalar'].device))

        if 'mutation_mask' in A and A['mutation_mask'] is not None:
            A_cat['mutation_mask'].append(A['mutation_mask'])
        else:
            A_cat['mutation_mask'].append(torch.zeros(Na, dtype=torch.long, device=A['x_scalar'].device))
        R_cat['x_scalar'].append(R['x_scalar'])
        R_cat['pos'].append(R['pos'])
        edge_attr_s = R['edge_attr_s']
        if edge_attr_s.numel() == 0:
            if len(edge_attr_s.shape) == 1:
                edge_attr_s = torch.empty((0, 22), dtype=edge_attr_s.dtype, device=edge_attr_s.device)
            elif len(edge_attr_s.shape) == 2 and edge_attr_s.size(1) != 22:
                edge_attr_s = torch.empty((0, 22), dtype=edge_attr_s.dtype, device=edge_attr_s.device)
        R_cat['edge_attr_s'].append(edge_attr_s)
        R_cat['x_vector'].append(R['x_vector'])
        R_cat['edge_vector'].append(R['edge_vector'])
        R_cat['mutation_mask'].append(R['mutation_mask'])
        eiR = R['edge_index'] + res_offset
        R_cat['edge_index'].append(eiR)
        R_cat['batch'].append(torch.full((Nr,), b_idx, dtype=torch.long, device=R['x_scalar'].device))
        atom_offset += Na
        res_offset  += Nr

    A_out = {}
    for k, v in A_cat.items():
        if k == 'edge_index':
            A_out[k] = torch.cat(v, dim=1)
        elif k in ['v', 'x_vector', 'edge_vector']:
            if len(v) > 0:
                A_out[k] = torch.cat(v, dim=0)  # vector features: [N, F, 3]
            else:
                A_out[k] = torch.empty((0,), device=A_cat['x_scalar'][0].device)  # 空tensor
        else:
            A_out[k] = torch.cat(v, dim=0)
    
    R_out = {}
    for k, v in R_cat.items():
        if k == 'edge_index':
            R_out[k] = torch.cat(v, dim=1)
        elif k in ['v', 'x_vector', 'edge_vector']:
            if len(v) > 0:
                R_out[k] = torch.cat(v, dim=0)  # vector features: [N, F, 3]
            else:
                R_out[k] = torch.empty((0,), device=R_cat['x_scalar'][0].device)  # 空tensor
        else:
            R_out[k] = torch.cat(v, dim=0)
    return A_out, R_out

def collate_pairs_fast(batch):

    filtered = [b for b in batch if (b is not None and b[0] is not None)]
    if len(filtered) == 0:
        A_wt_empty = {'x_scalar': torch.empty(0, 19), 'pos': torch.empty(0, 3), 'edge_index': torch.empty(2, 0, dtype=torch.long)}
        return A_wt_empty, {}, A_wt_empty, {}, torch.empty(0)

    A_wt_list, R_wt_list, A_mut_list, R_mut_list, ys = [], [], [], [], []
    for wt, mut, y in filtered:
        A_wt, R_wt = build_AR_from_data(wt)
        A_mut, R_mut = build_AR_from_data(mut)
        A_wt_list.append(A_wt)
        R_wt_list.append(R_wt)
        A_mut_list.append(A_mut)
        R_mut_list.append(R_mut)
        ys.append(y)
    A_wt_b, R_wt_b = batch_AR(A_wt_list, R_wt_list)
    A_mut_b, R_mut_b = batch_AR(A_mut_list, R_mut_list)

    if Cfg.use_esm:
        def _collect_side(graph, name1, name2):
            toks = []
            t = getattr(graph, name1, None)
            if t is not None and isinstance(t, torch.Tensor) and t.numel() > 0:
                toks.append(t.to(torch.float32))
            t = getattr(graph, name2, None)
            if t is not None and isinstance(t, torch.Tensor) and t.numel() > 0:
                toks.append(t.to(torch.float32))
            if toks:
                return torch.cat(toks, dim=0)
            return None

        def _pad_batch(graphs, side_prefix):
            seqs, chains = [], []
            for g in graphs:
                if side_prefix == 'ab':
                    s = _collect_side(g, 'esm_Ab1', 'esm_Ab2')
                else:
                    s = _collect_side(g, 'esm_Ag1', 'esm_Ag2')
                seqs.append(s)
            Ls = [int(s.size(0)) if s is not None else 0 for s in seqs]
            Lmax = max(Ls) if Ls else 0
            if Lmax == 0:
                return None, None
            B = len(graphs)
            tok = torch.zeros(B, Lmax, 1280, dtype=torch.float32)
            pad = torch.ones(B, Lmax, dtype=torch.bool)
            for i, s in enumerate(seqs):
                if s is None:
                    continue
                L = s.size(0)
                tok[i, :L] = s
                pad[i, :L] = False
            return tok, pad
        wt_graphs = [wt for wt, _, _ in filtered]
        mut_graphs = [mut for _, mut, _ in filtered]
        # WT
        ab_tok, ab_pad = _pad_batch(wt_graphs, 'ab')
        ag_tok, ag_pad = _pad_batch(wt_graphs, 'ag')
        if ab_tok is not None:
            R_wt_b['esm_ab'] = ab_tok
            R_wt_b['esm_ab_pad'] = ab_pad
        if ag_tok is not None:
            R_wt_b['esm_ag'] = ag_tok
            R_wt_b['esm_ag_pad'] = ag_pad
        # MUT
        ab_tok, ab_pad = _pad_batch(mut_graphs, 'ab')
        ag_tok, ag_pad = _pad_batch(mut_graphs, 'ag')
        if ab_tok is not None:
            R_mut_b['esm_ab'] = ab_tok
            R_mut_b['esm_ab_pad'] = ab_pad
        if ag_tok is not None:
            R_mut_b['esm_ag'] = ag_tok
            R_mut_b['esm_ag_pad'] = ag_pad
    y_tensor = torch.stack(ys, dim=0)
    
    return A_wt_b, R_wt_b, A_mut_b, R_mut_b, y_tensor

def to_device(d, device, non_blocking=True):
    return {k: (v.to(device, non_blocking=non_blocking) if torch.is_tensor(v) else v) for k, v in d.items()}

def load_split_indices(json_path: str, fold_id: int, one_based: bool = True) -> Tuple[List[int], List[int]]:
    train_idx: List[int] = []
    val_idx: List[int] = []
    p = Path(json_path)
    if not p.exists():
        raise FileNotFoundError(f"Split file not found: {json_path}")
    lines = p.read_text().strip().splitlines()
    line_obj = None
    if len(lines) == 1:
        line_obj = json.loads(lines[0])
    else:
        pick = (fold_id - 1) % len(lines)
        line_obj = json.loads(lines[pick])
    t_raw = line_obj.get('train', [])
    v_raw = line_obj.get('val', [])
    def norm_list(x):
        out = []
        for e in x:
            if isinstance(e, str) and e.isdigit():
                idx = int(e)
            else:
                try:
                    idx = int(e)
                except Exception:
                    continue
            if one_based:
                idx -= 1
            out.append(idx)
        return out
    train_idx = norm_list(t_raw)
    val_idx = norm_list(v_raw)
    return train_idx, val_idx

def load_graph_pt(path: str) -> Optional[Data]:
    try:
        obj = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(obj, Data):
            return obj
        if isinstance(obj, dict) and 'data' in obj and isinstance(obj['data'], Data):
            return obj['data']
        print(f"Warning: Unexpected data format in {path}")
        return None
    except Exception as e:
        print(f"Warning: Failed to load {path}: {e}")
        return None

class DeltaGDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path: str, root_dir: str, split: str, explicit_indices: Optional[List[int]] = None, filter_missing: bool = True):
        super().__init__()
        self.root_dir = Path(root_dir)
        df = pd.read_csv(csv_path)
        if explicit_indices is not None:
            self.df = df.iloc[explicit_indices].reset_index(drop=True)
        else:
            raise ValueError("explicit_indices must be provided; external split JSON is required.")
        self.split = split
        if filter_missing:
            self._filter_missing_data()
        self._graph_cache = OrderedDict()
        self.cache_capacity = 2000
    
    def __len__(self): return len(self.df)
    
    def _get_graph(self, path: str):
        path_str = str(path)
        g = self._graph_cache.get(path_str)
        if g is not None:
            self._graph_cache.move_to_end(path_str, last=True)
            return g
        g = load_graph_pt(path_str)
        if g is not None:
            if len(self._graph_cache) >= self.cache_capacity:
                self._graph_cache.popitem(last=False)
            self._graph_cache[path_str] = g
        return g

    def _filter_missing_data(self):
        original_len = len(self.df)
        valid_indices = []
        missing_files = []
        for idx, row in self.df.iterrows():
            wt_name = str(row['wild_pt'])
            mut_name = str(row['mutant_pt'])
            wt_path = self._resolve_path(wt_name)
            mut_path = self._resolve_path(mut_name)
            if wt_path.exists() and mut_path.exists():
                valid_indices.append(idx)
            else:
                missing_files.append((wt_name, mut_name))
        self.df = self.df.iloc[valid_indices].reset_index(drop=True)
        if missing_files:
            print(f"[{self.split}] Filtered out {len(missing_files)} samples with missing files:")
            for wt, mut in missing_files[:5]:
                print(f"  - {wt} or {mut}")
            if len(missing_files) > 5:
                print(f"  ... and {len(missing_files) - 5} more")
            print(f"[{self.split}] Dataset size: {original_len} -> {len(self.df)}", flush=True)

    def _resolve_path(self, name: str) -> Path:
        p = Path(name)
        if p.is_absolute() and p.exists():
            return p
        p2 = self.root_dir / name
        if p2.exists():
            return p2
        p3 = self.root_dir / "graphs" / name
        if p3.exists():
            return p3
        return p2

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        wt_name  = str(row['wild_pt'])
        mut_name = str(row['mutant_pt'])
        ddg = float(row['ddG'])
        wt_path  = self._resolve_path(wt_name)
        mut_path = self._resolve_path(mut_name)
        wt = self._get_graph(wt_path)
        mut = self._get_graph(mut_path)
        if wt is None:
            raise ValueError(f"Failed to load wild-type graph: {wt_path}")
        if mut is None:
            raise ValueError(f"Failed to load mutant graph: {mut_path}")
        return wt, mut, torch.tensor(ddg, dtype=torch.float32)

def create_model(mode, A_tmp, R_tmp, device, atom_encoder=None, res_encoder=None):
    if mode == 'atom_only':
        edgeA_dim = int(A_tmp['edge_attr'].size(-1))
        sA_in = int(A_tmp['x_scalar'].size(-1))
        model = DDGPredictor(
            sA=Cfg.sA, sR=Cfg.sA, n_blocks=Cfg.n_blocks,
            edgeA_dim=edgeA_dim, edgeR_dim=0,
            sA_in_dim=sA_in, sR_in_dim=0, hidden_dim=256, mode='atom_only',
            atom_encoder=atom_encoder or Cfg.atom_encoder, res_encoder=res_encoder
        ).to(device)

    elif mode == 'residue_only':
        edgeR_dim = int(R_tmp['edge_attr_s'].size(-1))
        sR_in = int(R_tmp['x_scalar'].size(-1))
        model = DDGPredictor(
            sA=Cfg.sA, sR=Cfg.sA, n_blocks=Cfg.n_blocks,
            edgeA_dim=0, edgeR_dim=edgeR_dim,
            sA_in_dim=0, sR_in_dim=sR_in, hidden_dim=256, mode='residue_only',
            atom_encoder=atom_encoder, res_encoder=res_encoder or Cfg.res_encoder
        ).to(device)

    elif mode == 'dual':
        edgeA_dim = int(A_tmp['edge_attr'].size(-1)) if A_tmp['edge_attr'].size(1) > 0 else 0
        edgeR_dim = int(R_tmp['edge_attr_s'].size(-1)) if R_tmp['edge_attr_s'].size(1) > 0 else 0
        sA_in = int(A_tmp['x_scalar'].size(-1))
        sR_in = int(R_tmp['x_scalar'].size(-1))
        model = DDGPredictor(
            sA=Cfg.sA, sR=Cfg.sA, n_blocks=Cfg.n_blocks,
            edgeA_dim=edgeA_dim, edgeR_dim=edgeR_dim,
            sA_in_dim=sA_in, sR_in_dim=sR_in, hidden_dim=256, mode='dual',
            atom_encoder=atom_encoder or Cfg.atom_encoder, res_encoder=res_encoder or Cfg.res_encoder
        ).to(device)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return model