import numpy as np
import torch
from torch_geometric.data import Data
from scipy.spatial import cKDTree
from collections import defaultdict
import logging

from config import (
    ANTIBODY_CHAINS, ANTIGEN_CHAINS, INTERACTION_DISTANCE,
    RESIDUE_TOPOLOGY, amino_acid_to_idx,
    NONCOV_CROSS_CUTOFF, NONCOV_WITHIN_CUTOFF,INCLUDE_WITHIN_CHAIN_NONCOV,RESIDUE_WITHIN_CHAIN_CUTOFF,
    RESIDUE_RADIUS_CUTOFF, RESIDUE_SEQ_K,
    MUTATION_EXTENSION_DISTANCE, ATOM_INTERFACE_CUTOFF
)
from utils import (
    is_vdw_interaction,
    is_hydrophobic_interaction, 
)
from features import (
    get_atom_features, get_residue_features,
    is_hbond_pair, is_ionic_pair,
    get_esm2_embeddings, get_atom_edge_features, get_residue_edge_features,
    virtual_cb_direction,
    compute_dssp_cache
)
# 导入突变感知模块
import sys
sys.path.append('/root')
#from mutation_aware_modules import MutationDetector

logger = logging.getLogger(__name__)


def parse_mutation_from_filename(filename):
    """
    从文件名解析突变信息
    
    Args:
        filename: PDB文件名，例如 "1A22_DA160A.pdb" 或 "1A22_DW160A.pdb"
        
    Returns:
        list: 突变信息列表，每个元素为dict包含 'chain', 'position', 'from_aa', 'to_aa'
    """
    import re
    from pathlib import Path
    from config import AA_3TO1
    
    mutations = []
    stem = Path(filename).stem
    
    # 匹配模式: PDBID_ChainFromPosTo.pdb
    # 例如: 1A22_DA160A.pdb 表示链D的160位从A变为A
    pattern = r'^([0-9A-Za-z]{4})_([A-Z])([A-Z]{1,3})(\d+)([A-Z]{1,3})$'
    match = re.match(pattern, stem)
    
    if match:
        pdb_id, chain, from_aa_raw, position, to_aa_raw = match.groups()
        
        # 转换三字母氨基酸名为单字母
        if len(from_aa_raw) == 3:
            from_aa = AA_3TO1.get(from_aa_raw, from_aa_raw[0])
        else:
            from_aa = from_aa_raw
            
        if len(to_aa_raw) == 3:
            to_aa = AA_3TO1.get(to_aa_raw, to_aa_raw[0])
        else:
            to_aa = to_aa_raw
        
        mutations.append({
            'chain': chain,
            'position': int(position),
            'from_aa': from_aa,
            'to_aa': to_aa
        })
    
    return mutations

def parse_mutation_from_repair_filename(filename):
    """
    从Repair格式的文件名解析突变信息
    
    Args:
        filename: PDB文件名，例如 "1ak4_Repair_1.pdb" 或 "1ak4_Repair.pdb"
        
    Returns:
        list: 突变信息列表，每个元素为dict包含 'chain', 'position', 'from_aa', 'to_aa'
    """
    import re
    from pathlib import Path
    
    mutations = []
    stem = Path(filename).stem
    
    # 匹配模式: PDBID_Repair_[MUTATION_ID].pdb
    # 例如: 1ak4_Repair_1.pdb 表示突变体1
    pattern = r'^([0-9A-Za-z]{4})_Repair(?:_(\d+))?$'
    match = re.match(pattern, stem)
    
    if match:
        pdb_id, mutation_id = match.groups()
        
        # 如果有mutation_id，说明这是突变体文件
        if mutation_id:
            # 对于Repair格式，我们无法从文件名直接获取突变信息
            # 这里返回一个占位符，实际的突变信息需要通过其他方式获取
            mutations.append({
                'chain': 'UNKNOWN',  # 需要从PDB文件内容中获取
                'position': 0,       # 需要从PDB文件内容中获取
                'from_aa': 'UNKNOWN', # 需要从PDB文件内容中获取
                'to_aa': 'UNKNOWN',   # 需要从PDB文件内容中获取
                'mutation_id': mutation_id,
                'pdb_id': pdb_id
            })
    
    return mutations

def parse_mutation_from_skempi_format(mutation_str):
    """
    从SKEMPI格式的突变字符串解析突变信息
    
    Args:
        mutation_str: SKEMPI格式的突变字符串，例如 "A:W50A" 或 "A:TRP50ALA"
        
    Returns:
        list: 突变信息列表
    """
    import re
    from config import AA_3TO1
    
    mutations = []
    if not mutation_str or not isinstance(mutation_str, str):
        return mutations
    
    # 分割多个突变
    mutation_parts = re.split(r'[;,]\s*', mutation_str.strip())
    
    for part in mutation_parts:
        part = part.strip()
        if not part:
            continue
            
        # 匹配模式: Chain:FromPosTo 或 Chain:From3PosTo3
        # 例如: "A:W50A" 或 "A:TRP50ALA"
        pattern = r'^([A-Z]):([A-Z]{1,3})(\d+)([A-Z]{1,3})$'
        match = re.match(pattern, part)
        
        if match:
            chain, from_aa, position, to_aa = match.groups()
            
            # 转换三字母氨基酸名为单字母（如果适用）
            if len(from_aa) == 3:
                from_aa = AA_3TO1.get(from_aa, from_aa[0])
            if len(to_aa) == 3:
                to_aa = AA_3TO1.get(to_aa, to_aa[0])
            
            mutations.append({
                'chain': chain,
                'position': int(position),
                'from_aa': from_aa,
                'to_aa': to_aa
            })
    
    return mutations

def parse_mutation_from_abbind_format(mutation_str):
    """
    从AB-Bind格式的突变字符串解析突变信息
    
    Args:
        mutation_str: AB-Bind格式的突变字符串，例如 "D160A" 或 "DW160A"
        
    Returns:
        list: 突变信息列表
    """
    import re
    from config import AA_3TO1
    
    mutations = []
    if not mutation_str or not isinstance(mutation_str, str):
        return mutations
    
    # 匹配模式: ChainFromPosTo 或 ChainFrom3PosTo3
    # 例如: "D160A" 或 "DW160A"
    pattern = r'^([A-Z])([A-Z]{1,3})(\d+)([A-Z]{1,3})$'
    match = re.match(pattern, mutation_str.strip())
    
    if match:
        chain, from_aa, position, to_aa = match.groups()
        
        # 转换三字母氨基酸名为单字母（如果适用）
        if len(from_aa) == 3:
            from_aa = AA_3TO1.get(from_aa, from_aa[0])
        if len(to_aa) == 3:
            to_aa = AA_3TO1.get(to_aa, to_aa[0])
        
        mutations.append({
            'chain': chain,
            'position': int(position),
            'from_aa': from_aa,
            'to_aa': to_aa
        })
    
    return mutations

def parse_mutation_from_standard_format(mutation_str):
    """
    从标准格式的突变字符串解析突变信息
    
    Args:
        mutation_str: 标准格式的突变字符串，例如 "A50W" 或 "A50TRP"
        
    Returns:
        list: 突变信息列表
    """
    import re
    from config import AA_3TO1
    
    mutations = []
    if not mutation_str or not isinstance(mutation_str, str):
        return mutations
    
    # 匹配模式: ChainPosFromTo 或 ChainPosFrom3To3
    # 例如: "A50W" 或 "A50TRP"
    pattern = r'^([A-Z])(\d+)([A-Z]{1,3})([A-Z]{1,3})$'
    match = re.match(pattern, mutation_str.strip())
    
    if match:
        chain, position, from_aa, to_aa = match.groups()
        
        # 转换三字母氨基酸名为单字母（如果适用）
        if len(from_aa) == 3:
            from_aa = AA_3TO1.get(from_aa, from_aa[0])
        if len(to_aa) == 3:
            to_aa = AA_3TO1.get(to_aa, to_aa[0])
        
        mutations.append({
            'chain': chain,
            'position': int(position),
            'from_aa': from_aa,
            'to_aa': to_aa
        })
    
    return mutations

def parse_mutation_info(mutation_input):
    """
    通用突变信息解析函数，支持多种格式
    
    Args:
        mutation_input: 突变信息，可以是：
            - 字符串：自动检测格式
            - 列表：包含多个突变信息
            - 字典：单个突变信息
            
    Returns:
        list: 突变信息列表
    """
    mutations = []
    
    if isinstance(mutation_input, dict):
        # 单个突变字典
        mutations.append(mutation_input)
    elif isinstance(mutation_input, list):
        # 多个突变列表
        for item in mutation_input:
            if isinstance(item, dict):
                mutations.append(item)
            elif isinstance(item, str):
                mutations.extend(parse_mutation_info(item))
    elif isinstance(mutation_input, str):
        # 字符串格式，尝试多种解析方法
        mutation_str = mutation_input.strip()
        
        # 尝试不同的解析格式
        parsers = [
            parse_mutation_from_filename,
            parse_mutation_from_repair_filename,
            parse_mutation_from_skempi_format,
            parse_mutation_from_abbind_format,
            parse_mutation_from_standard_format
        ]
        
        for parser in parsers:
            try:
                parsed = parser(mutation_str)
                if parsed:
                    mutations.extend(parsed)
                    break
            except Exception:
                continue
    
    return mutations

def _edge_pair_geom(i, j, edge_index, pos):
    """
    计算两条边的几何关系（距离和角度）
    
    Args:
        i, j: 边索引
        edge_index: [2, E] 边索引
        pos: [N, 3] 节点坐标
        
    Returns:
        dist: 边间距离
        angle: 边间角度
    """
    si, di = edge_index[:, i]
    sj, dj = edge_index[:, j]
    vi = pos[di] - pos[si]
    vj = pos[dj] - pos[sj]
    vi = vi / (vi.norm() + 1e-8)
    vj = vj / (vj.norm() + 1e-8)
    angle = torch.acos(torch.clamp(torch.dot(vi, vj), -1.0, 1.0))

    # 检查是否共享节点
    share = (si==sj) or (si==dj) or (di==sj) or (di==dj)
    if share:
        # 取两条边的非共享端点
        def other(s, d, x):
            return d if x==s else s
        if   si==sj: oi, oj = di, other(sj, dj, si)
        elif si==dj: oi, oj = di, other(sj, dj, si)
        elif di==sj: oi, oj = si, other(sj, dj, di)
        else:        oi, oj = si, other(sj, dj, di)
        dist = (pos[oi] - pos[oj]).norm()
    else:
        # 不共享节点时，使用边中心点距离
        ci = (pos[si] + pos[di]) / 2.0
        cj = (pos[sj] + pos[dj]) / 2.0
        dist = (ci - cj).norm()
    return dist, angle

def build_node_sharing_edge_graph(edge_index, pos, n_rbf=16, cutoff=10.0, max_neighbors=8):
    """
    构建节点共享边图：连接共享节点的边（连接所有共享节点的边）
    
    Args:
        edge_index: [2, E] - 原始图的边索引
        pos: [N, 3] - 节点坐标
        n_rbf: RBF基函数数量
        cutoff: 距离截断值
        max_neighbors: 每个边节点的最大邻居数
        
    Returns:
        edge_graph_index: [2, E_edge] - 边图的边索引
        edge_graph_attr: [E_edge, n_rbf+1] - 边图边属性 [距离RBF(n_rbf), 角度(1)]
    """
    if edge_index.size(1) == 0:
        return (torch.empty((2, 0), dtype=torch.long, device=edge_index.device),
                torch.empty((0, n_rbf+1), dtype=torch.float32, device=edge_index.device))
    
    num_edges = edge_index.size(1)
    
    # 计算边向量和中心点
    src_nodes = edge_index[0]  # [E]
    dst_nodes = edge_index[1]  # [E]
    edge_vecs = pos[dst_nodes] - pos[src_nodes]  # [E, 3] 边向量
    edge_centers = (pos[src_nodes] + pos[dst_nodes]) / 2.0  # [E, 3] 边中心点
    edge_lengths = torch.norm(edge_vecs, dim=1)  # [E] 边长度
    
    # 构建节点到边的映射
    max_node = edge_index.max().item() + 1
    node_to_edges = [[] for _ in range(max_node)]
    
    # 构建节点到边的映射 - O(E)
    for edge_idx in range(num_edges):
        src, dst = edge_index[:, edge_idx]
        node_to_edges[src.item()].append(edge_idx)
        node_to_edges[dst.item()].append(edge_idx)
    
    # 收集共享节点的边对 - 连接所有共享节点的边
    # 注意：edge_graph_index[i, j] 表示 j -> i，即j的信息聚合到i
    edge_pairs = []
    distances = []
    angles = []
    
    for node_edges in node_to_edges:
        if len(node_edges) > 1:
            # 该节点连接的所有边都相互连接
            # 收集所有边对及其几何信息
            edge_pair_info = []
            for i in range(len(node_edges)):
                for j in range(i + 1, len(node_edges)):
                    e1_idx = node_edges[i]  # 目标边 i
                    e2_idx = node_edges[j]  # 源边 j
                    
                    # 使用统一的几何计算函数
                    dist, angle = _edge_pair_geom(e1_idx, e2_idx, edge_index, pos)
                    edge_pair_info.append((dist.item(), angle.item(), e1_idx, e2_idx))
            
            # 限制每个节点的边对数量
            if len(edge_pair_info) > max_neighbors:
                # 按距离排序，选择最近的边对
                edge_pair_info.sort(key=lambda x: x[0])
                edge_pair_info = edge_pair_info[:max_neighbors]
            
            # 添加边对（不添加反向边）
            for dist_val, angle_val, e1_idx, e2_idx in edge_pair_info:
                # j -> i：将边j的信息聚合到边i
                edge_pairs.append((e1_idx, e2_idx))
                distances.append(dist_val)
                angles.append(angle_val)
    
    if not edge_pairs:
        return (torch.empty((2, 0), dtype=torch.long, device=edge_index.device),
                torch.empty((0, n_rbf+1), dtype=torch.float32, device=edge_index.device))
    
    # 构建边图索引（单向）
    edge_pairs_tensor = torch.tensor(edge_pairs, device=edge_index.device)
    edge_graph_index = edge_pairs_tensor.t().contiguous()
    
    # 将距离转换为RBF编码（使用与节点图一致的参数）
    distances_tensor = torch.tensor(distances, device=edge_index.device)  # [E_edge]
    angles_tensor = torch.tensor(angles, device=edge_index.device)  # [E_edge]
    
    # 计算RBF编码（与features.py中的实现一致）
    # 使用16个基函数，mu从0到7.5，步长0.5，sigma=0.5
    dist_rbf = []
    for i in range(16):
        mu = i * 0.5  # 从0到7.5，步长0.5
        sigma = 0.5
        dist_rbf_i = torch.exp(-((distances_tensor - mu) ** 2) / (2 * sigma ** 2))
        dist_rbf.append(dist_rbf_i)
    dist_rbf = torch.stack(dist_rbf, dim=1)  # [E_edge, 16]
    
    # 构建边图属性 [距离RBF(n_rbf), 角度(1)]
    num_edge_graph_edges = edge_graph_index.size(1)
    edge_graph_attr = torch.zeros((num_edge_graph_edges, n_rbf+1), dtype=torch.float32, device=edge_index.device)
    edge_graph_attr[:, :n_rbf] = dist_rbf  # RBF编码的距离
    edge_graph_attr[:, n_rbf] = angles_tensor  # 角度
    
    return edge_graph_index, edge_graph_attr

def build_spatial_edge_graph(edge_index, pos, distance_threshold=5.0, max_neighbors=8, n_rbf=16, cutoff=10.0):
    """
    构建空间边图：基于边中心点距离的边连接（O(n)优化版本）
    
    Args:
        edge_index: [2, E] - 原始图的边索引
        pos: [N, 3] - 节点坐标
        distance_threshold: 距离阈值
        max_neighbors: 每个边节点的最大邻居数
        n_rbf: RBF基函数数量
        cutoff: 距离截断值
        
    Returns:
        edge_graph_index: [2, E_edge] - 边图的边索引
        edge_graph_attr: [E_edge, n_rbf+1] - 边图边属性 [距离RBF(n_rbf), 角度(1)]
    """
    if edge_index.size(1) == 0:
        return (torch.empty((2, 0), dtype=torch.long, device=edge_index.device),
                torch.empty((0, n_rbf+1), dtype=torch.float32, device=edge_index.device))
    
    num_edges = edge_index.size(1)
    
    # O(n)优化：向量化计算边中心点和边向量
    src_nodes = edge_index[0]  # [E]
    dst_nodes = edge_index[1]  # [E]
    edge_centers = (pos[src_nodes] + pos[dst_nodes]) / 2.0  # [E, 3]
    edge_vecs = pos[dst_nodes] - pos[src_nodes]  # [E, 3] 边向量
    edge_lengths = torch.norm(edge_vecs, dim=1)  # [E] 边长度
    
    # 使用KDTree进行O(n log n)邻域搜索，限制每个边的邻居数
    from scipy.spatial import cKDTree
    import numpy as np
    
    # 转换为numpy进行KDTree搜索
    edge_centers_np = edge_centers.cpu().numpy()
    tree = cKDTree(edge_centers_np)
    
    # 查找每个边中心点的邻居，限制邻居数
    edge_pairs = []
    distances = []
    angles = []
    
    for i in range(num_edges):
        # 使用query_ball_point找到距离内的所有邻居
        neighbors = tree.query_ball_point(edge_centers_np[i], r=distance_threshold)
        
        # 移除自连接
        neighbors = [j for j in neighbors if j != i]
        
        # 为每个邻居计算距离和角度
        neighbor_dist_angle = []
        for j in neighbors:
            # 使用统一的几何计算函数
            dist, angle = _edge_pair_geom(i, j, edge_index, pos)
            neighbor_dist_angle.append((dist.item(), angle.item(), j))
        
        # 限制邻居数量，避免O(n²)复杂度
        if len(neighbor_dist_angle) > max_neighbors:
            # 按距离排序，选择最近的邻居
            neighbor_dist_angle.sort(key=lambda x: x[0])
            neighbor_dist_angle = neighbor_dist_angle[:max_neighbors]
        
        # 添加边对（避免重复）
        for dist_val, angle_val, j in neighbor_dist_angle:
            if j > i:  # 避免重复和自连接
                edge_pairs.append([i, j])
                distances.append(dist_val)
                angles.append(angle_val)
    
    if not edge_pairs:
        return (torch.empty((2, 0), dtype=torch.long, device=edge_index.device),
                torch.empty((0, n_rbf+1), dtype=torch.float32, device=edge_index.device))
    
    edge_graph_edges = torch.tensor(edge_pairs, device=edge_index.device).t().contiguous()
    
    # 将距离转换为RBF编码（使用与节点图一致的参数）
    distances_tensor = torch.tensor(distances, device=edge_index.device)  # [E_edge]
    angles_tensor = torch.tensor(angles, device=edge_index.device)  # [E_edge]
    
    # 计算RBF编码（与features.py中的实现一致）
    # 使用16个基函数，mu从0到7.5，步长0.5，sigma=0.5
    dist_rbf = []
    for k in range(n_rbf):
        mu = k * 0.5  # 从0到7.5，步长0.5
        sigma = 0.5
        dist_rbf_k = torch.exp(-((distances_tensor - mu) ** 2) / (2 * sigma ** 2))
        dist_rbf.append(dist_rbf_k)
    dist_rbf = torch.stack(dist_rbf, dim=1)  # [E_edge, n_rbf]
    
    # 构建边图属性 [距离RBF(n_rbf), 角度(1)]
    num_edge_graph_edges = edge_graph_edges.size(1)
    edge_graph_attr = torch.zeros((num_edge_graph_edges, n_rbf+1), dtype=torch.float32, device=edge_index.device)
    edge_graph_attr[:, :n_rbf] = dist_rbf  # RBF编码的距离
    edge_graph_attr[:, n_rbf] = angles_tensor  # 角度
    
    return edge_graph_edges, edge_graph_attr

def build_angle_edge_graph(edge_index, pos, angle_threshold=30.0, max_neighbors=8, n_rbf=16, cutoff=10.0):
    """
    构建角度边图：基于边之间角度的边连接（O(n)优化版本）
    
    Args:
        edge_index: [2, E] - 原始图的边索引
        pos: [N, 3] - 节点坐标
        angle_threshold: 角度阈值（度）
        max_neighbors: 每个边节点的最大邻居数
        n_rbf: RBF基函数数量
        cutoff: 距离截断值
        
    Returns:
        edge_graph_index: [2, E_edge] - 边图的边索引
        edge_graph_attr: [E_edge, n_rbf+1] - 边图边属性 [距离RBF(n_rbf), 角度(1)]
    """
    if edge_index.size(1) == 0:
        return (torch.empty((2, 0), dtype=torch.long, device=edge_index.device),
                torch.empty((0, n_rbf+1), dtype=torch.float32, device=edge_index.device))
    
    num_edges = edge_index.size(1)
    
    # O(n)优化：向量化计算边向量
    src_nodes = edge_index[0]  # [E]
    dst_nodes = edge_index[1]  # [E]
    edge_vectors = pos[dst_nodes] - pos[src_nodes]  # [E, 3]
    edge_lengths = torch.norm(edge_vectors, dim=1)  # [E] 边长度
    
    # 使用空间哈希或KDTree进行O(n log n)邻域搜索
    from scipy.spatial import cKDTree
    import numpy as np
    
    # 将边向量转换为numpy进行KDTree搜索
    edge_vectors_np = edge_vectors.cpu().numpy()
    
    # 使用边向量的方向进行邻域搜索
    # 归一化边向量
    norms = np.linalg.norm(edge_vectors_np, axis=1, keepdims=True)
    norms = np.where(norms > 1e-8, norms, 1.0)  # 避免除零
    normalized_vectors = edge_vectors_np / norms
    
    # 构建KDTree进行快速邻域搜索
    tree = cKDTree(normalized_vectors)
    
    # 查找每个边向量的邻居，限制邻居数
    edge_pairs = []
    distances = []
    angles = []
    
    for i in range(num_edges):
        # 使用query_ball_point找到角度相近的邻居
        # 在单位球面上搜索相近方向的向量
        neighbors = tree.query_ball_point(normalized_vectors[i], r=2.0)  # 使用较大的半径
        
        # 移除自连接
        neighbors = [j for j in neighbors if j != i]
        
        # 计算实际角度并筛选
        valid_neighbors = []
        for j in neighbors:
            # 计算余弦值
            cos_angle = np.dot(normalized_vectors[i], normalized_vectors[j])
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle_deg = np.arccos(cos_angle) * 180.0 / np.pi  # 转换为角度
            
            if angle_deg < angle_threshold:
                # 计算两条边的节点索引
                e1_src, e1_dst = src_nodes[i].item(), dst_nodes[i].item()
                e2_src, e2_dst = src_nodes[j].item(), dst_nodes[j].item()
                
                # 找到共享节点和独立节点
                if e1_src == e2_src or e1_src == e2_dst:
                    e1_other = e1_dst
                    e2_other = e2_dst if e1_src == e2_src else e2_src
                else:  # e1_dst == e2_src or e1_dst == e2_dst
                    e1_other = e1_src
                    e2_other = e2_dst if e1_dst == e2_src else e2_src
                
                # 使用统一的几何计算函数
                dist, angle = _edge_pair_geom(i, j, edge_index, pos)
                valid_neighbors.append((angle_deg, dist.item(), angle.item(), j))
        
        # 限制邻居数量，避免O(n²)复杂度
        if len(valid_neighbors) > max_neighbors:
            # 按角度排序，选择角度最小的邻居
            valid_neighbors.sort(key=lambda x: x[0])
            valid_neighbors = valid_neighbors[:max_neighbors]
        
        # 添加边对（避免重复）
        for _, dist_val, angle_val, j in valid_neighbors:
            if j > i:  # 避免重复和自连接
                edge_pairs.append([i, j])
                distances.append(dist_val)
                angles.append(angle_val)
    
    if not edge_pairs:
        return (torch.empty((2, 0), dtype=torch.long, device=edge_index.device),
                torch.empty((0, n_rbf+1), dtype=torch.float32, device=edge_index.device))
    
    edge_graph_edges = torch.tensor(edge_pairs, device=edge_index.device).t().contiguous()
    
    # 将距离转换为RBF编码（使用与节点图一致的参数）
    distances_tensor = torch.tensor(distances, device=edge_index.device)  # [E_edge]
    angles_tensor = torch.tensor(angles, device=edge_index.device)  # [E_edge]
    
    # 计算RBF编码（与features.py中的实现一致）
    # 使用16个基函数，mu从0到7.5，步长0.5，sigma=0.5
    dist_rbf = []
    for k in range(n_rbf):
        mu = k * 0.5  # 从0到7.5，步长0.5
        sigma = 0.5
        dist_rbf_k = torch.exp(-((distances_tensor - mu) ** 2) / (2 * sigma ** 2))
        dist_rbf.append(dist_rbf_k)
    dist_rbf = torch.stack(dist_rbf, dim=1)  # [E_edge, n_rbf]
    
    # 构建边图属性 [距离RBF(n_rbf), 角度(1)]
    num_edge_graph_edges = edge_graph_edges.size(1)
    edge_graph_attr = torch.zeros((num_edge_graph_edges, n_rbf+1), dtype=torch.float32, device=edge_index.device)
    edge_graph_attr[:, :n_rbf] = dist_rbf  # RBF编码的距离
    edge_graph_attr[:, n_rbf] = angles_tensor  # 角度
    
    return edge_graph_edges, edge_graph_attr

def build_hybrid_edge_graph(edge_index, pos,
                          use_node_sharing=True,
                          use_spatial=True,
                          use_angle=True,
                          spatial_threshold=5.0,
                          angle_threshold=30.0,
                          max_neighbors=8,
                          n_rbf=16,
                          cutoff=10.0):
    """
    构建混合边图：结合节点共享、空间和角度三种连接方式（O(n)优化版本）
    
    Args:
        edge_index: [2, E] - 原始图的边索引
        pos: [N, 3] - 节点坐标
        use_node_sharing: 是否使用节点共享连接
        use_spatial: 是否使用空间连接
        use_angle: 是否使用角度连接
        spatial_threshold: 空间距离阈值
        angle_threshold: 角度阈值
        max_neighbors: 每个边节点的最大邻居数
        n_rbf: RBF基函数数量
        cutoff: 距离截断值
        
    Returns:
        edge_graph_index: [2, E_edge] - 边图的边索引
        edge_graph_attr: [E_edge, n_rbf+1] - 边图边属性 [距离RBF(n_rbf), 角度(1)]
    """
    edge_graph_edges = []
    edge_graph_attrs = []
    
    if use_node_sharing:
        node_sharing_edges, node_sharing_attrs = build_node_sharing_edge_graph(edge_index, pos, n_rbf, cutoff, max_neighbors)
        if node_sharing_edges.size(1) > 0:
            edge_graph_edges.append(node_sharing_edges)
            edge_graph_attrs.append(node_sharing_attrs)
    
    if use_spatial:
        spatial_edges, spatial_attrs = build_spatial_edge_graph(
            edge_index, pos, spatial_threshold, max_neighbors, n_rbf, cutoff
        )
        if spatial_edges.size(1) > 0:
            edge_graph_edges.append(spatial_edges)
            edge_graph_attrs.append(spatial_attrs)
    
    if use_angle:
        angle_edges, angle_attrs = build_angle_edge_graph(
            edge_index, pos, angle_threshold, max_neighbors, n_rbf, cutoff
        )
        if angle_edges.size(1) > 0:
            edge_graph_edges.append(angle_edges)
            edge_graph_attrs.append(angle_attrs)
    
    if edge_graph_edges:
        all_edges = torch.cat(edge_graph_edges, dim=1)
        all_attrs = torch.cat(edge_graph_attrs, dim=0)
        
        # 去重：对于重复的边，合并其属性（取最大值）
        unique_edges, inverse_indices = torch.unique(all_edges, dim=1, return_inverse=True)
        unique_attrs = torch.zeros((unique_edges.size(1), n_rbf + 1), dtype=torch.float32, device=edge_index.device)
        
        # 对于每条唯一边，合并所有对应边的属性
        for i in range(unique_edges.size(1)):
            # 找到所有映射到这条唯一边的原始边
            mask = (inverse_indices == i)
            if mask.any():
                # 取所有对应属性的最大值（包括one-hot编码和RBF特征）
                unique_attrs[i] = all_attrs[mask].max(dim=0)[0]
        
        return unique_edges, unique_attrs
    else:
        return (torch.empty((2, 0), dtype=torch.long, device=edge_index.device),
                torch.empty((0, n_rbf + 1), dtype=torch.float32, device=edge_index.device))

def _is_mut_residue(mutation_sites, chain_id, residue):
    """
    宽松的突变残基匹配函数，支持多种残基ID格式
    
    Args:
        mutation_sites: 突变位点信息 {chain_id: {residue_id: True/False}}
        chain_id: 链ID
        residue: 残基对象
        
    Returns:
        bool: 是否为突变残基
    """
    rid = residue.get_id()           # (' ', resseq, icode)
    resseq = rid[1]
    icode  = rid[2]
    m = mutation_sites.get(chain_id, {})
    # 依次尝试多种key格式
    return bool(
        (rid in m and m[rid]) or
        ((resseq, icode) in m and m[(resseq, icode)]) or
        ((resseq, ' ') in m and m[(resseq, ' ')]) or
        (resseq in m and m[resseq])
    )

def identify_mutation_sites_and_extend_interface(antibody_atoms, antigen_atoms,
                                                antibody_coords, antigen_coords,
                                                antibody_info, antigen_info,
                                                mutation_sites=None):
    """
    识别突变位点并扩展周围区域，确保突变位点及其周围MUTATION_EXTENSION_DISTANCE半径内的原子被包含在图中
    
    Args:
        antibody_atoms: 抗体原子列表
        antigen_atoms: 抗原原子列表
        antibody_coords: 抗体原子坐标
        antigen_coords: 抗原原子坐标
        antibody_info: 抗体原子信息 [(atom, residue), ...]
        antigen_info: 抗原原子信息 [(atom, residue), ...]
        mutation_sites: 突变位点信息 {chain_id: {residue_id: True/False}}
        
    Returns:
        tuple: (extended_antibody_atoms, extended_antigen_atoms, 
                extended_antibody_coords, extended_antigen_coords,
                extended_antibody_info, extended_antigen_info,
                mutation_extension_mask)
    """
    if mutation_sites is None:
        # 如果没有突变位点信息，直接返回原始数据
        return (antibody_atoms, antigen_atoms, antibody_coords, antigen_coords,
                antibody_info, antigen_info, np.array([False] * len(antibody_atoms + antigen_atoms)))
    
    # 收集所有原子和坐标
    all_atoms = antibody_atoms + antigen_atoms
    all_coords = np.vstack([antibody_coords, antigen_coords])
    all_info = antibody_info + antigen_info
    
    # 识别突变位点的原子
    mutation_atom_indices = set()
    
    for i, (atom, residue) in enumerate(all_info):
        chain_id = atom.get_parent().get_parent().get_id()
        if _is_mut_residue(mutation_sites, chain_id, residue):
            mutation_atom_indices.add(i)
    
    if not mutation_atom_indices:
        # 如果没有突变位点，直接返回原始数据
        return (antibody_atoms, antigen_atoms, antibody_coords, antigen_coords,
                antibody_info, antigen_info, np.array([False] * len(all_atoms)))
    
    # 构建KD树用于快速邻域搜索
    kd_tree = cKDTree(all_coords)
    
    # 找到突变位点周围MUTATION_EXTENSION_DISTANCE内的所有原子
    extended_atom_indices = set()
    for mutation_idx in mutation_atom_indices:
        # 找到该突变位点周围的所有原子
        neighbors = kd_tree.query_ball_point(all_coords[mutation_idx], r=MUTATION_EXTENSION_DISTANCE)
        extended_atom_indices.update(neighbors)
    
    # 创建扩展掩码
    mutation_extension_mask = np.array([i in extended_atom_indices for i in range(len(all_atoms))])
    
    # 分离扩展后的抗体和抗原原子
    extended_antibody_atoms = []
    extended_antigen_atoms = []
    extended_antibody_coords = []
    extended_antigen_coords = []
    extended_antibody_info = []
    extended_antigen_info = []
    
    antibody_count = len(antibody_atoms)
    
    for i in range(len(all_atoms)):
        if mutation_extension_mask[i]:
            if i < antibody_count:
                # 抗体原子
                extended_antibody_atoms.append(all_atoms[i])
                extended_antibody_coords.append(all_coords[i])
                extended_antibody_info.append(all_info[i])
            else:
                # 抗原原子
                extended_antigen_atoms.append(all_atoms[i])
                extended_antigen_coords.append(all_coords[i])
                extended_antigen_info.append(all_info[i])
    
    # 转换为numpy数组
    extended_antibody_coords = np.array(extended_antibody_coords) if extended_antibody_coords else np.array([])
    extended_antigen_coords = np.array(extended_antigen_coords) if extended_antigen_coords else np.array([])
    
    return (extended_antibody_atoms, extended_antigen_atoms,
            extended_antibody_coords, extended_antigen_coords,
            extended_antibody_info, extended_antigen_info,
            mutation_extension_mask)

def extract_interface_subgraph(antibody_atoms, antigen_atoms, 
                              antibody_coords, antigen_coords,
                              antibody_info, antigen_info,
                              interaction_stats,
                              distance_threshold=None):
    """提取相互作用子图
    
    Args:
        distance_threshold: 接触面距离阈值，如果为None则使用INTERACTION_DISTANCE
    """
    if len(antibody_atoms) == 0 or len(antigen_atoms) == 0:
        return [], []
    
    # 使用传入的距离阈值，如果没有则使用默认值
    cutoff = distance_threshold if distance_threshold is not None else INTERACTION_DISTANCE
    
    # 构建抗原原子的KD树
    antigen_tree = cKDTree(antigen_coords)
    
    # 查找每个抗体原子的最近抗原原子
    distances, nearest_indices = antigen_tree.query(antibody_coords)
    
    # 找出在相互作用距离内的抗体原子
    antibody_interface_mask = distances < cutoff
    
    # 构建抗原原子的KD树（反向查找）
    antibody_tree = cKDTree(antibody_coords)
    distances_reverse, nearest_indices_reverse = antibody_tree.query(antigen_coords)
    antigen_interface_mask = distances_reverse < cutoff
    
    # 合并两个方向的相互作用原子
    antibody_interface_atoms = []
    antigen_interface_atoms = []
    
    # 收集抗体相互作用原子
    for i, (atom, residue) in enumerate(antibody_info):
        if antibody_interface_mask[i]:
            antibody_interface_atoms.append((atom, residue))
    
    # 收集抗原相互作用原子
    for i, (atom, residue) in enumerate(antigen_info):
        if antigen_interface_mask[i]:
            antigen_interface_atoms.append((atom, residue))
    
    # 合并所有相互作用原子
    interface_atoms = antibody_interface_atoms + antigen_interface_atoms
    
    # 更新统计信息
    interaction_stats['interface_atoms'] = len(interface_atoms)
    
    # 统计相互作用残基
    interface_residues = set()
    for atom, residue in interface_atoms:
        interface_residues.add(id(residue))
    interaction_stats['interface_residues'] = len(interface_residues)
    
    return interface_atoms, antibody_interface_mask


def extract_residue_node_metadata(
    pdb_file,
    mutation_sites=None,
    ab_chains=None,
    ag_chains=None,
):
    """提取残基图节点元数据，顺序与 build_molecular_graph 的 residues_coord 一致。

    仅解析 PDB 与突变扩展/接触面逻辑，不计算特征，用于可解释性导出时的 node_index 映射。
    """
    from Bio.PDB import PDBParser

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("metadata", pdb_file)

    antibody_chains = list(ab_chains) if ab_chains is not None else ANTIBODY_CHAINS.copy()
    antigen_chains = list(ag_chains) if ag_chains is not None else ANTIGEN_CHAINS.copy()
    antibody_chains_upper = {c.upper() for c in antibody_chains}
    antigen_chains_upper = {c.upper() for c in antigen_chains}

    antibody_atoms = []
    antigen_atoms = []
    antibody_coords = []
    antigen_coords = []
    antibody_info = []
    antigen_info = []

    for model in structure:
        for chain in model:
            chain_id = chain.get_id()
            chain_id_upper = chain_id.upper()
            is_antibody_chain = chain_id_upper in antibody_chains_upper
            is_antigen_chain = chain_id_upper in antigen_chains_upper
            if not is_antibody_chain and not is_antigen_chain:
                continue

            for residue in chain:
                if residue.get_id()[0] != " ":
                    continue
                for atom in residue:
                    if atom.element == "H":
                        continue
                    if is_antibody_chain:
                        antibody_atoms.append(atom)
                        antibody_coords.append(atom.get_coord())
                        antibody_info.append((atom, residue))
                    else:
                        antigen_atoms.append(atom)
                        antigen_coords.append(atom.get_coord())
                        antigen_info.append((atom, residue))

    antibody_coords = np.array(antibody_coords) if antibody_coords else np.empty((0, 3))
    antigen_coords = np.array(antigen_coords) if antigen_coords else np.empty((0, 3))

    interaction_stats = {
        "interface_atoms": 0,
        "interface_residues": 0,
    }

    if mutation_sites is not None:
        _, _, _, _, _, _, mutation_extension_mask = identify_mutation_sites_and_extend_interface(
            antibody_atoms,
            antigen_atoms,
            antibody_coords,
            antigen_coords,
            antibody_info,
            antigen_info,
            mutation_sites,
        )
        all_atoms = antibody_atoms + antigen_atoms
        all_info = antibody_info + antigen_info
        residue_graph_atoms = [
            (all_atoms[i], all_info[i][1])
            for i, flag in enumerate(mutation_extension_mask)
            if flag
        ]
    else:
        residue_graph_atoms, _ = extract_interface_subgraph(
            antibody_atoms,
            antigen_atoms,
            antibody_coords,
            antigen_coords,
            antibody_info,
            antigen_info,
            interaction_stats,
            distance_threshold=INTERACTION_DISTANCE,
        )

    residue_entries = []
    residue_coords = []
    residue_to_residue_idx = {}

    for _atom, residue in residue_graph_atoms:
        residue_id = id(residue)
        if residue_id in residue_to_residue_idx:
            continue

        ca_atom = residue["CA"] if "CA" in residue else None
        if ca_atom is not None:
            residue_coord = ca_atom.get_coord()
        else:
            coords = [a.get_coord() for a in residue if a.element != "H"]
            if not coords:
                continue
            residue_coord = np.mean(coords, axis=0)

        residue_to_residue_idx[residue_id] = len(residue_entries)
        chain_id = residue.get_parent().get_id()
        hetflag, resseq, icode = residue.get_id()
        residue_entries.append({
            "node_index": len(residue_entries),
            "chain_id": chain_id,
            "residue_number": int(resseq),
            "insertion_code": str(icode).strip(),
            "residue_name": residue.get_resname().strip(),
        })
        residue_coords.append(residue_coord)

    return residue_entries, np.asarray(residue_coords, dtype=np.float64)


def build_molecular_graph(pdb_file, molecule_type,
                         # 原子特征开关
                         use_atom_type=True, use_hbond=True, use_charge=True, use_hydrophobic=True, use_asa=True, use_residue_type=True, use_atom_name=True,
                         # 残基特征开关
                         use_aa_type=True, use_polar=True, use_aromatic=True, use_residue_hydrophobic=True,
                         use_pssm=True, use_esm2=False, use_dssp=False,
                         # 原子边构建模式
                         atom_edge_mode="interaction",
                         # 突变位点扩展选项
                         mutation_sites=None):
    """构建分离的原子图和残基图
    
    构图策略：
    - 如果有突变位点信息：仅以突变位置为中心，在MUTATION_EXTENSION_DISTANCE半径内构建图
    - 如果无突变位点信息：回退到传统的接触面构图策略
    """
    atom_edge_mode = (atom_edge_mode or "interaction").lower()
    if atom_edge_mode not in {"interaction", "radius"}:
        raise ValueError(f"Unsupported atom_edge_mode: {atom_edge_mode}")
    
    from Bio.PDB import PDBParser
    from features import compute_atom_asa, get_pssm_features
    
    # 解析PDB文件
    parser = PDBParser()
    structure = parser.get_structure(molecule_type, pdb_file)
    
    # 提取序列和计算特征
    from pdb_processor import extract_sequence_from_structure
    sequences, residue_to_seq_idx = extract_sequence_from_structure(structure)
    #print(f"build_molecular_graph - 文件: {pdb_file}")
    #print(f"build_molecular_graph - 序列: {sequences}")
    # 计算序列特征：优先 ESM2，可选 PSSM
    pssm_features = None
    if use_pssm and not use_esm2:
        #logger.info("计算PSSM特征...")
        pssm_features = get_pssm_features(sequences, pdb_file)
    else:
        logger.info("跳过PSSM特征计算")

    esm2_reps = {}
    if use_esm2:
        logger.info("计算ESM2序列嵌入...")
        try:
            esm2_reps = get_esm2_embeddings(sequences)
            # 记录维度
            _esm_dim = 0
            for _cid, _rep in esm2_reps.items():
                if _rep is not None and len(_rep) > 0:
                    _esm_dim = int(_rep.shape[1]); break
            #logger.info(f"ESM2嵌入维度: {_esm_dim}")
        except Exception as e:
            logger.warning(f"ESM2嵌入计算失败: {e}")
            esm2_reps = {}

        # --- 始终为保存目的计算完整链级 ESM（与特征拼装解耦） ---
    esm2_reps_for_save = {}
    try:
        # 复用已计算结果，避免重复前向
        if len(esm2_reps) > 0:
            esm2_reps_for_save = esm2_reps
        else:
            esm2_reps_for_save = get_esm2_embeddings(sequences) or {}
    except Exception as _e:
        logger.warning(f"ESM2 保存用嵌入计算失败: {_e}")
        esm2_reps_for_save = {}
    
    # 只在需要原子ASA特征时才计算
    atom_asa_dict = None
    if use_asa:
        original_pdb = pdb_file.replace('_withH.pdb', '.pdb') if '_withH.pdb' in pdb_file else pdb_file
        atom_asa_dict = compute_atom_asa(original_pdb)
        logger.info("计算ASA特征完成")
    #else:
    #    #logger.info("跳过ASA特征计算")
    #    pass
    
    # 计算DSSP特征（可选）
    dssp_cache = None
    if use_dssp:
        try:
            original_pdb_for_dssp = pdb_file.replace('_withH.pdb', '.pdb') if '_withH.pdb' in pdb_file else pdb_file
            dssp_cache = compute_dssp_cache(structure, original_pdb_for_dssp, dssp_exe='mkdssp', acc_array='Sander')
            logger.info("DSSP 特征缓存完成")
        except Exception as e:
            logger.warning(f"DSSP 特征计算失败: {e}")
            dssp_cache = None
        
    # 统计信息
    interaction_stats = {
        'antibody_atoms': 0,
        'antigen_atoms': 0,
        'antibody_residues': 0,
        'antigen_residues': 0,
        'interface_atoms': 0,
        'interface_residues': 0,
        'interaction_pairs': 0
    }
    
    # 分别收集抗体和抗原的原子
    antibody_atoms = []
    antigen_atoms = []
    # 直接使用配置文件中的链映射，不使用自动检测
    antibody_chains = ANTIBODY_CHAINS.copy()
    antigen_chains = ANTIGEN_CHAINS.copy()
    # 转换为大写集合以便大小写不敏感匹配
    antibody_chains_upper = {c.upper() for c in antibody_chains}
    antigen_chains_upper = {c.upper() for c in antigen_chains}
    logger.info(f"使用配置的抗体链: {antibody_chains} (匹配时大小写不敏感)")
    logger.info(f"使用配置的抗原链: {antigen_chains} (匹配时大小写不敏感)")
    
    antibody_coords = []
    antigen_coords = []
    antibody_info = []  # 保存(atom, residue)元组
    antigen_info = []

    # 遍历结构收集原子
    for model in structure:
        for chain in model:
            chain_id = chain.get_id()
            
            # 统计链类型（大小写不敏感匹配）
            chain_id_upper = chain_id.upper()
            is_antibody_chain = chain_id_upper in antibody_chains_upper
            is_antigen_chain = chain_id_upper in antigen_chains_upper
            
            for residue in chain:
                # 🔧 修复问题2：过滤非标准蛋白残基（HETATM：水、配体、糖基等）
                hetflag = residue.get_id()[0]
                if hetflag != ' ':  # 只保留标准蛋白残基
                    continue
                
                # 统计残基数量
                if is_antibody_chain:
                    interaction_stats['antibody_residues'] += 1
                elif is_antigen_chain:
                    interaction_stats['antigen_residues'] += 1
                else:
                    continue
                
                for atom in residue:
                    # 保留所有重原子：包括主链原子和侧链原子，排除氢原子
                    if atom.element == 'H':
                        continue                    
                    # 统计原子数量
                    if is_antibody_chain:
                        interaction_stats['antibody_atoms'] += 1
                        antibody_atoms.append(atom)
                        antibody_coords.append(atom.get_coord())
                        antibody_info.append((atom, residue))
                    elif is_antigen_chain:
                        interaction_stats['antigen_atoms'] += 1
                        antigen_atoms.append(atom)
                        antigen_coords.append(atom.get_coord())
                        antigen_info.append((atom, residue))

    # 转换为numpy数组
    antibody_coords = np.array(antibody_coords)
    antigen_coords = np.array(antigen_coords)
    
    # ========= 分离原子图和残基图的原子集合 =========
    # 1) 原子图：使用接触面6Å以内的原子
    atom_graph_atoms, _ = extract_interface_subgraph(
        antibody_atoms, antigen_atoms,
        antibody_coords, antigen_coords,
        antibody_info, antigen_info,
        interaction_stats,
        distance_threshold=ATOM_INTERFACE_CUTOFF
    )
    logger.info(f"原子图接触面原子数（{ATOM_INTERFACE_CUTOFF}Å）: {len(atom_graph_atoms)}")
    
    # 2) 残基图：使用突变位点10Å内的残基（通过识别包含的原子来获取残基）
    residue_graph_atoms = []
    if mutation_sites is not None:
        # 构造突变局部补丁（半径 = MUTATION_EXTENSION_DISTANCE）
        _, _, _, _, _, _, mutation_extension_mask = identify_mutation_sites_and_extend_interface(
            antibody_atoms, antigen_atoms,
            antibody_coords, antigen_coords,
            antibody_info, antigen_info,
            mutation_sites
        )
        all_atoms = antibody_atoms + antigen_atoms
        all_info  = antibody_info + antigen_info
        residue_graph_atoms = [(all_atoms[i], all_info[i][1]) 
                              for i, flag in enumerate(mutation_extension_mask) if flag]
        logger.info(f"残基图突变中心扩展原子数（{MUTATION_EXTENSION_DISTANCE}Å）: {len(residue_graph_atoms)}")
    else:
        # 如果没有突变位点信息，使用接触面构图
        residue_graph_atoms, _ = extract_interface_subgraph(
            antibody_atoms, antigen_atoms,
            antibody_coords, antigen_coords,
            antibody_info, antigen_info,
            interaction_stats,
            distance_threshold=INTERACTION_DISTANCE
        )
        logger.info(f"残基图接触面原子数（{INTERACTION_DISTANCE}Å）: {len(residue_graph_atoms)}")
    
    # 使用不同的原子集合：原子图用atom_graph_atoms，残基图用residue_graph_atoms
    interface_atoms = atom_graph_atoms  # 用于原子图特征和边构建
    
    
    # RDKit 特征已移除，不再读取PDB文本用于RDKit

    # 计算分离的节点特征
    atom_scalar_features = []
    atom_vectors = []  # 原子向量特征：前一个CA、后一个CA、与CA的方向向量、到本残基CB的向量
    residue_features = []
    residue_features_cache = {}
    
    # 为每个原子计算特征
    for atom, residue in interface_atoms:
        # 获取原子标量特征
        # 仅使用 BioPython 特征提取
        atom_scalar_feature = get_atom_features(
            atom, residue, atom_asa_dict,
            use_atom_type=use_atom_type, use_hbond=use_hbond, use_charge=use_charge, 
            use_hydrophobic=use_hydrophobic, use_asa=use_asa, 
            use_residue_type=use_residue_type, use_atom_name=use_atom_name,
            pssm_features=pssm_features, residue_to_seq_idx=residue_to_seq_idx
        )
        atom_scalar_features.append(atom_scalar_feature)
        
        # 计算原子向量特征：前一个CA、后一个CA、与CA的方向向量、到本残基CB的向量
        try:
            ca_atom = residue['CA']
            atom_coord = atom.get_coord()
            ca_coord = ca_atom.get_coord()
            
            # 1. 与CA原子的方向向量
            direction_vector = atom_coord - ca_coord
            # 归一化
            norm = np.linalg.norm(direction_vector)
            if norm > 1e-8:
                direction_vector = direction_vector / norm
            else:
                direction_vector = np.zeros(3, dtype=np.float32)
            
            # 2. 到本残基CB的向量
            cb_vector = np.zeros(3, dtype=np.float32)
            try:
                if 'CB' in residue:
                    cb_atom = residue['CB']
                    cb_coord = cb_atom.get_coord()
                    cb_vector = cb_coord - ca_coord
                else:
                    # 对于GLY，构建虚拟CB
                    n = residue['N']
                    c = residue['C']
                    n_coord = torch.tensor(n.get_coord(), dtype=torch.float32)
                    ca_coord_tensor = torch.tensor(ca_coord, dtype=torch.float32)
                    c_coord = torch.tensor(c.get_coord(), dtype=torch.float32)
                    vcb = virtual_cb_direction(n_coord, ca_coord_tensor, c_coord)
                    cb_vector = vcb.numpy() - ca_coord
                
                # 归一化
                norm = np.linalg.norm(cb_vector)
                if norm > 1e-8:
                    cb_vector = cb_vector / norm
            except Exception:
                cb_vector = np.zeros(3, dtype=np.float32)
            
            # 3. 前一个残基的CA向量
            prev_ca_vec = np.zeros(3, dtype=np.float32)
            try:
                chain = residue.get_parent()
                resnum = residue.get_id()[1]
                prev_residue = None
                for r in chain:
                    if r.get_id()[1] == resnum - 1:
                        prev_residue = r
                        break
                if prev_residue is not None and 'CA' in prev_residue:
                    prev_ca = prev_residue['CA']
                    prev_ca_vec = prev_ca.get_coord() - ca_coord
                    # 归一化
                    norm = np.linalg.norm(prev_ca_vec)
                    if norm > 1e-8:
                        prev_ca_vec = prev_ca_vec / norm
            except Exception:
                pass
            
            # 4. 后一个残基的CA向量
            next_ca_vec = np.zeros(3, dtype=np.float32)
            try:
                chain = residue.get_parent()
                resnum = residue.get_id()[1]
                next_residue = None
                for r in chain:
                    if r.get_id()[1] == resnum + 1:
                        next_residue = r
                        break
                if next_residue is not None and 'CA' in next_residue:
                    next_ca = next_residue['CA']
                    next_ca_vec = next_ca.get_coord() - ca_coord
                    # 归一化
                    norm = np.linalg.norm(next_ca_vec)
                    if norm > 1e-8:
                        next_ca_vec = next_ca_vec / norm
            except Exception:
                pass
            
        except (KeyError, AttributeError):
            direction_vector = np.zeros(3, dtype=np.float32)
            cb_vector = np.zeros(3, dtype=np.float32)
            prev_ca_vec = np.zeros(3, dtype=np.float32)
            next_ca_vec = np.zeros(3, dtype=np.float32)
        
        # 原子向量特征：形状为 [4, 3] = [向量类型, XYZ]
        # 向量类型：0=与CA的方向向量, 1=到本残基CB的向量, 2=前一个CA, 3=后一个CA
        atom_vectors.append(np.array([direction_vector, cb_vector, prev_ca_vec, next_ca_vec], dtype=np.float32))
    
    # 为每个唯一的残基计算特征（基于残基图的原子集合）
    unique_residues = {}
    residue_vectors = {}  # 残基向量特征：前一个、后一个、CB向量
    for atom, residue in residue_graph_atoms:
        residue_id = id(residue)
        if residue_id not in unique_residues:
            residue_name = residue.get_resname()
            # 获取残基特征（缓存结果，包含ASA和PSSM特征）
            residue_key = (residue_name, residue_id)  # 使用残基名称和ID作为缓存键
            
            if residue_key not in residue_features_cache:
                residue_features_cache[residue_key] = get_residue_features(
                    residue_name, residue, atom_asa_dict, pssm_features, residue_to_seq_idx,
                    use_aa_type, use_polar, use_aromatic, use_residue_hydrophobic, use_pssm,
                    esm2_reps=esm2_reps, use_esm2=use_esm2,
                    use_dssp=use_dssp, dssp_cache=dssp_cache
                )
            
            unique_residues[residue_id] = residue_features_cache[residue_key]
            
            # 计算残基向量特征：前一个、后一个、CB向量
            try:
                ca = residue['CA'] if 'CA' in residue else None
                cb = residue['CB'] if 'CB' in residue else None
                
                # CB向量
                if ca is not None and cb is not None:
                    cb_vec = cb.get_coord() - ca.get_coord()
                elif ca is not None:
                    # 对于GLY，构建虚拟CB
                    try:
                        n = residue['N']
                        c = residue['C']
                        n_coord = torch.tensor(n.get_coord(), dtype=torch.float32)
                        ca_coord = torch.tensor(ca.get_coord(), dtype=torch.float32)
                        c_coord = torch.tensor(c.get_coord(), dtype=torch.float32)
                        vcb = virtual_cb_direction(n_coord, ca_coord, c_coord)
                        cb_vec = vcb.numpy()
                    except Exception:
                        cb_vec = np.zeros(3, dtype=np.float32)
                else:
                    cb_vec = np.zeros(3, dtype=np.float32)
                
                # 前一个残基的CA向量
                prev_vec = np.zeros(3, dtype=np.float32)
                try:
                    chain = residue.get_parent()
                    resnum = residue.get_id()[1]
                    prev_residue = None
                    for r in chain:
                        if r.get_id()[1] == resnum - 1:
                            prev_residue = r
                            break
                    if prev_residue is not None and 'CA' in prev_residue:
                        prev_ca = prev_residue['CA']
                        prev_vec = prev_ca.get_coord() - ca.get_coord()
                        # 归一化
                        norm = np.linalg.norm(prev_vec)
                        if norm > 1e-8:
                            prev_vec = prev_vec / norm
                except Exception:
                    pass
                
                # 后一个残基的CA向量
                next_vec = np.zeros(3, dtype=np.float32)
                try:
                    chain = residue.get_parent()
                    resnum = residue.get_id()[1]
                    next_residue = None
                    for r in chain:
                        if r.get_id()[1] == resnum + 1:
                            next_residue = r
                            break
                    if next_residue is not None and 'CA' in next_residue:
                        next_ca = next_residue['CA']
                        next_vec = next_ca.get_coord() - ca.get_coord()
                        # 归一化
                        norm = np.linalg.norm(next_vec)
                        if norm > 1e-8:
                            next_vec = next_vec / norm
                except Exception:
                    pass
                
                # 组合向量特征：形状为 [3, 3] = [向量类型, XYZ]
                # 向量类型：0=前一个, 1=后一个, 2=CB
                residue_vectors[residue_id] = np.array([prev_vec, next_vec, cb_vec], dtype=np.float32)
                
            except Exception:
                residue_vectors[residue_id] = np.zeros((3, 3), dtype=np.float32)
    
    # 将残基特征转换为列表，保持与残基信息相同的顺序
    residue_features = list(unique_residues.values())
    # 需要根据 residue_info 顺序创建 vectors，故延后用 residue_info 重建
    
    atom_scalar_features = np.array(atom_scalar_features)
    residue_features = np.array(residue_features)
    # 创建相互作用原子的坐标数组
    if len(interface_atoms) > 0:
        interface_coords = np.array([atom.get_coord() for atom, _ in interface_atoms])
    else:
        # 如果为空，创建一个形状为 (0, 3) 的空数组，避免 cKDTree 初始化错误
        interface_coords = np.empty((0, 3), dtype=np.float32)
    atom_vectors = np.asarray(atom_vectors, dtype=np.float32)  # 形状: [原子数, 4, 3]
    
    # 创建原子到索引的映射
    atom_to_index = {atom: i for i, (atom, _) in enumerate(interface_atoms)}
    
    # ========= 原子图边构建 =========
    atom_edge_list = []
    atom_edge_scalar_features = []  # 原子边标量特征
    atom_edge_vector_features = []  # 原子边向量特征
    edge_set_directed = set()
    # 统计信息（与旧接口兼容字段名）
    edge_stats = {
        'hydrogen': 0,
        'ionic': 0,
        'hydrophobic': 0,
        'vdw': 0,
        'radius': 0,
        'covalent': 0,
        'knn': 0,
        'total': 0,
    }

    # 1) 为每个残基建立 原子名 -> 索引 映射（用于共价边）
    residue_to_atomname_index = defaultdict(dict)
    for idx, (atom, residue) in enumerate(interface_atoms):
        residue_to_atomname_index[id(residue)][atom.get_name().strip()] = idx

    # 2) 共价边（残基内部，基于 RESIDUE_TOPOLOGY）- 先构建共价边
    # 首先需要构建残基信息，以便建立肽键
    residue_info_temp = []
    residue_to_residue_idx_temp = {}
    for atom, residue in interface_atoms:
        residue_id = id(residue)
        if residue_id not in residue_to_residue_idx_temp:
            actual_residue_id = residue.get_id()
            residue_info_temp.append((actual_residue_id, residue))
            residue_to_residue_idx_temp[residue_id] = len(residue_info_temp) - 1
    
    # 2a) 残基内共价边
    for _, residue in residue_info_temp:
        bonds = RESIDUE_TOPOLOGY.get(residue.get_resname().strip(), [])
        atomname_to_idx = residue_to_atomname_index.get(id(residue), {})
        for a_name, b_name in bonds:
            ia = atomname_to_idx.get(a_name)
            ib = atomname_to_idx.get(b_name)
            if ia is None or ib is None or ia == ib:
                continue
            ai, aj = interface_atoms[ia][0], interface_atoms[ib][0]
            dist = float(np.linalg.norm(ai.get_coord() - aj.get_coord()))
            # 共价边基于拓扑定义，不需要距离限制
            scalar_ef0, vector_ef0 = get_atom_edge_features(ai, aj, ai.get_parent(), aj.get_parent(), dist, is_covalent=True)
            scalar_ef1, vector_ef1 = get_atom_edge_features(aj, ai, aj.get_parent(), ai.get_parent(), dist, is_covalent=True)
            
            if (ia, ib) not in edge_set_directed:
                atom_edge_list.append((ia, ib))
                atom_edge_scalar_features.append(scalar_ef0)
                atom_edge_vector_features.append(vector_ef0.reshape(1, 3))  # 形状: [1, 3]
                edge_set_directed.add((ia, ib))
                edge_stats['covalent'] += 1
            if (ib, ia) not in edge_set_directed:
                atom_edge_list.append((ib, ia))
                atom_edge_scalar_features.append(scalar_ef1)
                atom_edge_vector_features.append(vector_ef1.reshape(1, 3))  # 形状: [1, 3]
                edge_set_directed.add((ib, ia))
                edge_stats['covalent'] += 1

    # 2b) 肽键 C(i)-N(i+1)（同一链的相邻残基）
    chain_to_ordered_indices = defaultdict(list)
    for idx_res, (_, res) in enumerate(residue_info_temp):
        chain_id = res.get_parent().get_id()
        chain_to_ordered_indices[chain_id].append(idx_res)
    for chain_id, idx_list in chain_to_ordered_indices.items():
        idx_list_sorted = sorted(idx_list, key=lambda k: residue_info_temp[k][1].get_id()[1])
        for a_i, b_i in zip(idx_list_sorted, idx_list_sorted[1:]):
            res_a = residue_info_temp[a_i][1]
            res_b = residue_info_temp[b_i][1]
            ia = residue_to_atomname_index.get(id(res_a), {}).get('C')
            ib = residue_to_atomname_index.get(id(res_b), {}).get('N')
            if ia is None or ib is None:
                continue
            ai, aj = interface_atoms[ia][0], interface_atoms[ib][0]
            dist = float(np.linalg.norm(ai.get_coord() - aj.get_coord()))
            # 肽键边基于序列相邻关系定义，不需要距离限制
            scalar_ef0, vector_ef0 = get_atom_edge_features(ai, aj, res_a, res_b, dist, is_covalent=True)
            scalar_ef1, vector_ef1 = get_atom_edge_features(aj, ai, res_b, res_a, dist, is_covalent=True)
            
            if (ia, ib) not in edge_set_directed:
                atom_edge_list.append((ia, ib))
                atom_edge_scalar_features.append(scalar_ef0)
                atom_edge_vector_features.append(vector_ef0.reshape(1, 3))  # 形状: [1, 3]
                edge_set_directed.add((ia, ib))
                edge_stats['covalent'] += 1
            if (ib, ia) not in edge_set_directed:
                atom_edge_list.append((ib, ia))
                atom_edge_scalar_features.append(scalar_ef1)
                atom_edge_vector_features.append(vector_ef1.reshape(1, 3))  # 形状: [1, 3]
                edge_set_directed.add((ib, ia))
                edge_stats['covalent'] += 1

    # 3) 非共价边（只保留相互作用边：氢键、离子、疏水、vdw）
    if len(interface_coords) > 0:
        kd_tree = cKDTree(interface_coords)
        
        # 一次性获取半径内所有无向原子对
        if INCLUDE_WITHIN_CHAIN_NONCOV:
            r_max = max(NONCOV_CROSS_CUTOFF, NONCOV_WITHIN_CUTOFF)
        else:
            r_max = NONCOV_CROSS_CUTOFF
        pairs = kd_tree.query_pairs(r_max)
        for i, j in pairs:
            atom_i, residue_i = interface_atoms[i]
            atom_j, residue_j = interface_atoms[j]
            # 跳过同一残基内原子
            if residue_i == residue_j:
                continue
            chain_i = atom_i.get_parent().get_parent().get_id()
            chain_j = atom_j.get_parent().get_parent().get_id()
            
            dist = float(np.linalg.norm(interface_coords[i] - interface_coords[j]))

            # 判定跨链/同链与对应半径阈值
            if chain_i != chain_j:
                if dist > NONCOV_CROSS_CUTOFF:
                    continue
                edge_type_flag = np.array([0.0, 1.0, 0.0], dtype=np.float32)  # 跨链
            else:
                if not INCLUDE_WITHIN_CHAIN_NONCOV or dist > NONCOV_WITHIN_CUTOFF:
                    continue
                edge_type_flag = np.array([0.0, 0.0, 1.0], dtype=np.float32)  # 同链

            add_edge = False
            if atom_edge_mode == "interaction":
                has_interaction = False
                
                if dist <= 3.5:
                    try:
                        if is_hbond_pair(atom_i, atom_j, residue_i, residue_j):
                            edge_stats['hydrogen'] += 1
                            has_interaction = True
                    except Exception:
                        pass
                if dist <= 5.0:
                    try:
                        if is_ionic_pair(atom_i, atom_j, residue_i, residue_j):
                            edge_stats['ionic'] += 1
                            has_interaction = True
                    except Exception:
                        pass
                try:
                    if is_hydrophobic_interaction(atom_i, atom_j, residue_i, residue_j, max_distance=5.0):
                        edge_stats['hydrophobic'] += 1
                        has_interaction = True
                except Exception:
                    pass
                try:
                    if is_vdw_interaction(atom_i, atom_j, epsilon=0.3, lower_tolerance_ratio=0.0, heavy_only=True):
                        edge_stats['vdw'] += 1
                        has_interaction = True
                except Exception:
                    pass
                
                add_edge = has_interaction
            else:
                add_edge = True
            
            if not add_edge:
                continue
            
            if atom_edge_mode == "radius":
                edge_stats['radius'] += 1
            
            # 获取边特征（标量和向量分离）
            scalar_ef0, vector_ef0 = get_atom_edge_features(atom_i, atom_j, residue_i, residue_j, dist, is_covalent=False)
            scalar_ef1, vector_ef1 = get_atom_edge_features(atom_j, atom_i, residue_j, residue_i, dist, is_covalent=False)
            
            if (i, j) not in edge_set_directed:
                atom_edge_list.append((i, j))
                atom_edge_scalar_features.append(scalar_ef0)
                atom_edge_vector_features.append(vector_ef0.reshape(1, 3))  # 形状: [1, 3]
                edge_set_directed.add((i, j))

    # 构建残基图边
    residue_edge_list = []
    residue_edge_scalar_features = []  # 残基边标量特征
    residue_edge_vector_features = []  # 残基边向量特征
    
    # 收集残基信息，保持与残基特征数组相同的顺序（基于残基图的原子集合）
    residue_info = []
    residue_coords = []
    residue_to_atoms = defaultdict(list)
    residue_to_residue_idx = {}  # 残基ID到残基图索引的映射
    
    # 首先收集所有唯一的残基，保持顺序（从residue_graph_atoms中收集）
    for atom, residue in residue_graph_atoms:
        residue_id = id(residue)
        if residue_id not in residue_to_residue_idx:
            # 计算残基中心坐标（使用CA原子）
            ca_atom = residue['CA'] if 'CA' in residue else None
            
            if ca_atom is not None:
                residue_coord = ca_atom.get_coord()
            else:
                # 如果没有CA原子，使用所有重原子的平均坐标
                coords = [a.get_coord() for a in residue if a.element != 'H']
                if coords:
                    residue_coord = np.mean(coords, axis=0)
                else:
                    continue
            
            residue_idx = len(residue_info)
            # 存储实际的残基ID（来自PDB文件）和残基对象
            actual_residue_id = residue.get_id()
            residue_info.append((actual_residue_id, residue))
            residue_coords.append(residue_coord)
            residue_to_residue_idx[residue_id] = residue_idx

    # 然后收集每个残基包含的原子索引（基于residue_graph_atoms）
    # 注意：这里的索引是相对于residue_graph_atoms的，不是interface_atoms的
    for i, (atom, residue) in enumerate(residue_graph_atoms):
        residue_id = id(residue)
        residue_to_atoms[residue_id].append(i)
    
    residue_coords = np.array(residue_coords)
    # 构建与 residue_info 对齐的残基向量数组
    residue_vectors_list = []
    for _, residue in residue_info:
        rid = id(residue)
        vec = residue_vectors.get(rid, np.zeros((3, 3), dtype=np.float32))
        residue_vectors_list.append(vec)
    residue_vectors = np.array(residue_vectors_list, dtype=np.float32)  # 形状: [残基数, 向量类型, 3]
    
    # 2) 残基半径图（Cα–Cα，区分同链和跨链半径）
    if len(residue_coords) > 1:
        residue_kd_tree = cKDTree(residue_coords)
        # 调试计数器
        dbg_cand = 0
        dbg_no_ca = 0
        dbg_added = 0
        for i, (residue_id, residue) in enumerate(residue_info):
            residue_chain_id = residue.get_parent().get_id()
            
            # 根据同链或跨链选择不同的半径
            r_max = max(RESIDUE_RADIUS_CUTOFF, RESIDUE_WITHIN_CHAIN_CUTOFF)
            neighbor_idx_list = residue_kd_tree.query_ball_point(residue_coords[i], r=r_max)

            for j in neighbor_idx_list:
                if j == i:
                    continue
                dist = float(np.linalg.norm(residue_coords[i] - residue_coords[j]))

                dbg_cand += 1
                
                neighbor_residue = residue_info[j][1]
                neighbor_chain_id = neighbor_residue.get_parent().get_id()
                
                # 判断是同链还是跨链，使用不同的半径阈值
                is_same_chain = (residue_chain_id == neighbor_chain_id)
                
                if is_same_chain:
                    if dist > RESIDUE_WITHIN_CHAIN_CUTOFF:
                        continue
                else:
                    if dist > RESIDUE_RADIUS_CUTOFF:
                        continue
                
                # 统一使用CA原子
                ca1 = residue['CA'] if 'CA' in residue else None
                ca2 = neighbor_residue['CA'] if 'CA' in neighbor_residue else None
                if ca1 is None or ca2 is None:
                    dbg_no_ca += 1
                    continue
                
                scalar_ef, vector_ef = get_residue_edge_features(ca1, ca2, residue, neighbor_residue, dist, is_covalent=False, edge_type_hint="radius")
                
                # 添加所有半径内的边
                residue_edge_list.append((i, j))
                residue_edge_scalar_features.append(scalar_ef)
                residue_edge_vector_features.append(vector_ef.reshape(1, 3))  # 形状: [1, 3]
                dbg_added += 1
    
    # 3) 序列边：同一链内，连接序列上前后 RESIDUE_SEQ_K 个邻居
    # 构建 chain -> seq_idx -> 残基图索引 的映射
    chain_to_seq_to_idx = defaultdict(dict)
    residue_chain_and_seq = []  # [(chain_id, seq_idx, residue), ...] 顺序与 residue_info 对齐
    for idx, (residue_id, residue) in enumerate(residue_info):
        chain_id = residue.get_parent().get_id()
        resnum = residue.get_id()[1]
        seq_idx = None
        try:
            if chain_id in residue_to_seq_idx and resnum in residue_to_seq_idx[chain_id]:
                seq_idx = residue_to_seq_idx[chain_id][resnum]
                chain_to_seq_to_idx[chain_id][seq_idx] = idx
        except Exception:
            pass
        residue_chain_and_seq.append((chain_id, seq_idx, residue))

    for i, (chain_id, seq_idx, residue) in enumerate(residue_chain_and_seq):
        if seq_idx is None:
            continue
        for offset in range(1, RESIDUE_SEQ_K + 1):
            # 前向连接
            neighbor_seq_idx = seq_idx + offset
            j = chain_to_seq_to_idx.get(chain_id, {}).get(neighbor_seq_idx)
            if j is not None:
                neighbor_residue = residue_info[j][1]
                # 使用CA原子
                ca1 = residue['CA'] if 'CA' in residue else None
                ca2 = neighbor_residue['CA'] if 'CA' in neighbor_residue else None
                if ca1 is not None and ca2 is not None:
                    # 序列边不受空间半径限制，直接连接序列相邻的残基
                    dist = float(np.linalg.norm(ca1.get_coord() - ca2.get_coord()))
                    scalar_ef, vector_ef = get_residue_edge_features(ca1, ca2, residue, neighbor_residue, dist, is_covalent=False, edge_type_hint="sequence")
                    
                    residue_edge_list.append((i, j))
                    residue_edge_scalar_features.append(scalar_ef)
                    residue_edge_vector_features.append(vector_ef.reshape(1, 3))  # 形状: [1, 3]
            
            # 后向连接
            neighbor_seq_idx_back = seq_idx - offset
            if neighbor_seq_idx_back >= 0:
                j_back = chain_to_seq_to_idx.get(chain_id, {}).get(neighbor_seq_idx_back)
                if j_back is not None:
                    neighbor_residue_back = residue_info[j_back][1]
                    # 使用CA原子
                    ca1_back = residue['CA'] if 'CA' in residue else None
                    ca2_back = neighbor_residue_back['CA'] if 'CA' in neighbor_residue_back else None
                    if ca1_back is not None and ca2_back is not None:
                        # 序列边不受空间半径限制，直接连接序列相邻的残基
                        dist_back = float(np.linalg.norm(ca1_back.get_coord() - ca2_back.get_coord()))
                        scalar_ef_back, vector_ef_back = get_residue_edge_features(ca1_back, ca2_back, residue, neighbor_residue_back, dist_back, is_covalent=False, edge_type_hint="sequence")
                        
                        residue_edge_list.append((i, j_back))
                        residue_edge_scalar_features.append(scalar_ef_back)
                        residue_edge_vector_features.append(vector_ef_back.reshape(1, 3))  # 形状: [1, 3]
    
    # 统计总边数
    atom_edge_count = len(atom_edge_list)
    residue_edge_count = len(residue_edge_list)
    edge_stats['total'] = atom_edge_count + residue_edge_count   
    # 创建PyG Data对象
    if len(atom_edge_list) > 0:
        atom_edge_index = torch.tensor(atom_edge_list, dtype=torch.long).t().contiguous()
    else:
        atom_edge_index = torch.empty((2, 0), dtype=torch.long)
    
    if len(residue_edge_list) > 0:
        residue_edge_index = torch.tensor(residue_edge_list, dtype=torch.long).t().contiguous()
    else:
        residue_edge_index = torch.empty((2, 0), dtype=torch.long)

    x_atom = torch.tensor(atom_scalar_features, dtype=torch.float32)
    x_residues = torch.tensor(residue_features, dtype=torch.float32)
    
    # 原子边特征
    if atom_edge_scalar_features:
        atom_edge_scalar_features_np = np.asarray(atom_edge_scalar_features, dtype=np.float32)
        atom_edge_attr = torch.from_numpy(atom_edge_scalar_features_np)
    else:
        atom_edge_attr = torch.empty((0, 0), dtype=torch.float32)
    
    if atom_edge_vector_features:
        atom_edge_vector_features_np = np.asarray(atom_edge_vector_features, dtype=np.float32)
        atom_edge_vector_attr = torch.from_numpy(atom_edge_vector_features_np)  # 形状: [边数, 1, 3]
    else:
        atom_edge_vector_attr = torch.empty((0, 1, 3), dtype=torch.float32)
    
    # 残基边特征
    if residue_edge_scalar_features:
        residue_edge_scalar_features_np = np.asarray(residue_edge_scalar_features, dtype=np.float32)
        residue_edge_attr = torch.from_numpy(residue_edge_scalar_features_np)
    else:
        residue_edge_attr = torch.empty((0, 0), dtype=torch.float32)
    
    if residue_edge_vector_features:
        residue_edge_vector_features_np = np.asarray(residue_edge_vector_features, dtype=np.float32)
        residue_edge_vector_attr = torch.from_numpy(residue_edge_vector_features_np)  # 形状: [边数, 1, 3]
    else:
        residue_edge_vector_attr = torch.empty((0, 1, 3), dtype=torch.float32)
    
    # 重新构建相互作用子图的数据（仅保留必要映射）
    # residue_indices: 原子图中的每个原子对应残基图中的残基索引
    # 原子图和残基图使用不同的原子集合，需要映射原子图的原子到残基图的残基
    interface_residue_indices = []
    for i, (atom, residue) in enumerate(interface_atoms):  # interface_atoms = atom_graph_atoms
        residue_id = id(residue)
        if residue_id in residue_to_residue_idx:
            # 该原子所属的残基在残基图中存在
            interface_residue_indices.append(residue_to_residue_idx[residue_id])
        else:
            # 该原子所属的残基不在残基图中（因为残基图使用的是不同的原子集合）
            interface_residue_indices.append(-1)
        
    residue_indices_arr = np.array(interface_residue_indices)
    
    # 构建基础图数据
    graph_data = {
        'x_atom': x_atom,
        'x_residues': x_residues,
        'atom_coord': torch.tensor(interface_coords, dtype=torch.float32),
        'residues_coord': torch.tensor(residue_coords, dtype=torch.float32),
        'atom_vector': torch.tensor(atom_vectors, dtype=torch.float32),
        'residue_vector': torch.tensor(residue_vectors, dtype=torch.float32),
        'edge_index_atom': atom_edge_index,
        'edge_index_residues': residue_edge_index,
        'edge_attr_atom': atom_edge_attr,
        'edge_attr_residues': residue_edge_attr,
        # 新的边向量特征
        'edge_vector_atom': atom_edge_vector_attr,
        'edge_vector_residues': residue_edge_vector_attr,
        'residue_indices': residue_indices_arr,
    }

    # --- 将链级 ESM 嵌入按结构中的标准残基顺序对齐，并保存为 Ab1/Ab2 与 Ag1/Ag2 ---
    try:
        def _align_chain_esm(chain_id: str):
            """返回与结构中该链标准残基顺序对齐的 ESM 表示 [N_chain_res, 1280]。"""
            rep = esm2_reps_for_save.get(chain_id)
            if rep is None or (hasattr(rep, '__len__') and len(rep) == 0):
                return None
            # 取该链的残基到序列索引映射
            seq_map = residue_to_seq_idx.get(chain_id, {})
            # 迭代结构中该链的标准残基（保持顺序），用映射取到对应的 seq_idx
            try:
                model0 = next(structure.get_models())
                chain = model0[chain_id]
            except Exception:
                return None
            idx_list = []
            from features import parse_residue_number  # 局部导入以避免循环
            for residue in chain:
                if residue.id[0] != ' ':  # 仅标准蛋白残基
                    continue
                resseq = parse_residue_number(residue.get_id()[1])
                icode  = residue.get_id()[2]
                seq_idx = None
                if len(seq_map) > 0:
                    sample_key = next(iter(seq_map.keys()))
                    if isinstance(sample_key, tuple):
                        seq_idx = seq_map.get((resseq, icode)) or seq_map.get((resseq, ' ')) or seq_map.get((resseq, ''))
                    else:
                        seq_idx = seq_map.get(resseq)
                if seq_idx is None:
                    continue
                # 边界检查
                if 0 <= int(seq_idx) < int(rep.shape[0]):
                    idx_list.append(int(seq_idx))
            if not idx_list:
                return None
            rep_t = torch.from_numpy(rep.astype(np.float32)) if isinstance(rep, np.ndarray) else torch.as_tensor(rep, dtype=torch.float32)
            return rep_t.index_select(0, torch.tensor(idx_list, dtype=torch.long))

        # 收集按配置存在于当前结构且有ESM的链（对齐后保存）
        ab_chains_available = [cid for cid in antibody_chains if _align_chain_esm(cid) is not None]
        ag_chains_available = [cid for cid in antigen_chains if _align_chain_esm(cid) is not None]

        # 最多保存两条（Ab1/Ab2，Ag1/Ag2）
        if len(ab_chains_available) >= 1:
            graph_data['esm_Ab1'] = _align_chain_esm(ab_chains_available[0])
        if len(ab_chains_available) >= 2:
            graph_data['esm_Ab2'] = _align_chain_esm(ab_chains_available[1])

        if len(ag_chains_available) >= 1:
            graph_data['esm_Ag1'] = _align_chain_esm(ag_chains_available[0])
        if len(ag_chains_available) >= 2:
            graph_data['esm_Ag2'] = _align_chain_esm(ag_chains_available[1])
    except Exception as _e:
        logger.warning(f"保存ESM嵌入到图数据时出错: {_e}")
    
    graph = Data(**graph_data)
    
    # 注意：不保存调试/统计信息到图中，避免文件过大
    # edge_stats, residue_info, residue_to_seq_idx 仅作为返回值用于调试
    # 这些信息不会被保存到图对象中
    
    return graph, edge_stats, residue_info, residue_to_seq_idx

def find_mutation_positions(wt_sequence, mt_sequence):
    """
    通过序列比对找到突变位点
    
    Args:
        wt_sequence: 野生型序列
        mt_sequence: 突变型序列
        
    Returns:
        list: 突变位点索引列表（0-based）
    """
    if len(wt_sequence) != len(mt_sequence):
        raise ValueError(f"序列长度不匹配: WT={len(wt_sequence)}, MT={len(mt_sequence)}")
    
    mutation_positions = []
    for i, (wt_aa, mt_aa) in enumerate(zip(wt_sequence, mt_sequence)):
        if wt_aa != mt_aa and wt_aa != 'X' and mt_aa != 'X':
            mutation_positions.append(i)
    
    return mutation_positions

def build_mut_vector_from_seqdiff(graph, wt_sequences, mt_sequences, residue_info, residue_to_seq_idx):
    """
    返回与 graph.x_residues 对齐的突变指示向量（float32, 0/1）。
    依赖 residue_to_seq_idx: {chain_id: {resseq(int) 或 (resseq, icode): seq_idx(int)}}
    
    Args:
        graph: PyTorch Geometric图对象
        wt_sequences: 野生型序列字典
        mt_sequences: 突变型序列字典
        residue_info: 残基信息列表 [(residue_id, residue), ...]
        residue_to_seq_idx: 残基到序列索引的映射
    """
    # 1) 先把每条链的突变位置放到"序列索引集合"里
    mut_pos_by_chain = {}
    for ch in wt_sequences:
        if ch in mt_sequences and len(wt_sequences[ch]) == len(mt_sequences[ch]):
            mut_pos_by_chain[ch] = set(find_mutation_positions(wt_sequences[ch], mt_sequences[ch]))
        else:
            logger.warning(f"[build_mut_vector_from_seqdiff] chain {ch}: WT/MT 序列长度不等或缺失，跳过突变标注")
            mut_pos_by_chain[ch] = set()

    # 2) 遍历图里的残基节点：残基 -> (chain, resseq, icode) -> seq_idx -> 是否突变
    N = graph.x_residues.size(0)
    vec = torch.zeros(N, dtype=torch.float32)
    missed = 0
    for i, (rid_tuple, residue) in enumerate(residue_info):
        ch = residue.get_parent().get_id()
        hetflag, resseq, icode = residue.get_id()   # 注意：PDB三元组
        
        # 兼容有插入码的映射：允许两种key格式
        seq_map = residue_to_seq_idx.get(ch, {})
        seq_idx = None
        
        # 优先使用 (resseq, icode) 作为键
        if len(seq_map) > 0:
            sample_key = next(iter(seq_map.keys()))
            if isinstance(sample_key, tuple):
                # 键格式为 (resseq, icode)
                seq_idx = seq_map.get((resseq, icode), None) \
                          or seq_map.get((resseq, ' '), None) \
                          or seq_map.get((resseq, ''), None)
        
        # 如果还没找到，尝试直接用 resseq 作为键
        if seq_idx is None:
            seq_idx = seq_map.get(resseq, None)
        
        if seq_idx is None:
            missed += 1
            continue
        
        if seq_idx in mut_pos_by_chain.get(ch, set()):
            vec[i] = 1.0

    #graph.mut_res_vector = vec
    #graph.mut_res_mask = vec.bool()
    #graph.mut_missed_count = missed  # 诊断用：多少残基没映射到序列索引
    return vec

def build_wt_mt_graph_pair(wt_pdb_file, mt_pdb_file, molecule_type, 
                          # 原子特征开关
                          use_atom_type=True, use_hbond=True, use_charge=True, 
                          use_hydrophobic=True, use_asa=True, 
                          use_residue_type=True, use_atom_name=True,
                          # 残基特征开关
                          use_aa_type=True, use_polar=True, use_aromatic=True, 
                          use_residue_hydrophobic=True,
                          use_pssm=True, use_esm2=False, use_dssp=False,
                          atom_edge_mode="interaction"):
    """
    构建WT-MT图对
    
    Args:
        wt_pdb_file: 野生型PDB文件路径
        mt_pdb_file: 突变型PDB文件路径
        molecule_type: 分子类型标识符
        其他参数: 与build_molecular_graph相同的参数
        
    Returns:
        tuple: (wt_graph, mt_graph, mutation_positions, mutation_vector)
            - wt_graph: 野生型图
            - mt_graph: 突变型图  
            - mutation_positions: 突变位点索引列表
            - mutation_vector: 突变位点向量（1表示突变位点，0表示非突变位点）
    """
    from Bio.PDB import PDBParser
    
    # 解析WT和MT的PDB文件
    parser = PDBParser()
    wt_structure = parser.get_structure("WT", wt_pdb_file)
    mt_structure = parser.get_structure("MT", mt_pdb_file)
    
    # 提取序列
    from pdb_processor import extract_sequence_from_structure
    wt_sequences, wt_residue_to_seq_idx = extract_sequence_from_structure(wt_structure)
    mt_sequences, mt_residue_to_seq_idx = extract_sequence_from_structure(mt_structure)
    
    # 找到突变位点，记录链ID和残基ID
    mutation_sites = {}  # {chain_id: {residue_id: True/False}}
    
    # 对每个链进行序列比对
    for chain_id in wt_sequences:
        if chain_id in mt_sequences:
            wt_seq = wt_sequences[chain_id]
            mt_seq = mt_sequences[chain_id]
            
            # 找到该链的突变位点
            chain_mutations = find_mutation_positions(wt_seq, mt_seq)
            
            # 记录突变位点的链ID和残基ID
            mutation_sites[chain_id] = {}
            
            # 获取该链的残基ID到序列索引的映射
            if chain_id in wt_residue_to_seq_idx:
                for residue_id, seq_idx in wt_residue_to_seq_idx[chain_id].items():
                    mutation_sites[chain_id][residue_id] = seq_idx in chain_mutations
    
    # 构建WT图
    wt_graph, wt_edge_stats, wt_residue_info, wt_residue_to_seq_idx = build_molecular_graph(
        wt_pdb_file, molecule_type,
        use_atom_type=use_atom_type, use_hbond=use_hbond, use_charge=use_charge,
        use_hydrophobic=use_hydrophobic, use_asa=use_asa,
        use_residue_type=use_residue_type, use_atom_name=use_atom_name,
        use_aa_type=use_aa_type, use_polar=use_polar, use_aromatic=use_aromatic,
        use_residue_hydrophobic=use_residue_hydrophobic,
        use_pssm=use_pssm, use_esm2=use_esm2, use_dssp=use_dssp,
        atom_edge_mode=atom_edge_mode,
        mutation_sites=mutation_sites
    )
    
    # 构建MT图
    mt_graph, mt_edge_stats, mt_residue_info, mt_residue_to_seq_idx = build_molecular_graph(
        mt_pdb_file, molecule_type,
        use_atom_type=use_atom_type, use_hbond=use_hbond, use_charge=use_charge,
        use_hydrophobic=use_hydrophobic, use_asa=use_asa,
        use_residue_type=use_residue_type, use_atom_name=use_atom_name,
        use_aa_type=use_aa_type, use_polar=use_polar, use_aromatic=use_aromatic,
        use_residue_hydrophobic=use_residue_hydrophobic,
        use_pssm=use_pssm, use_esm2=use_esm2, use_dssp=use_dssp,
        atom_edge_mode=atom_edge_mode,
        mutation_sites=mutation_sites
    )
    
    # 构完图后，直接用序列差异在图里生成向量
    wt_vec = build_mut_vector_from_seqdiff(wt_graph, wt_sequences, mt_sequences, wt_residue_info, wt_residue_to_seq_idx)
    mt_vec = build_mut_vector_from_seqdiff(mt_graph, wt_sequences, mt_sequences, mt_residue_info, mt_residue_to_seq_idx)
    
    wt_graph.mutation_mask = wt_vec.long()
    mt_graph.mutation_mask = mt_vec.long()
    
    return wt_graph, mt_graph, mutation_sites, wt_vec

def build_wt_mt_graphs_from_folder(folder_path, molecule_type="protein", **kwargs):
    """
    从文件夹中构建所有WT-MT图对
    
    Args:
        folder_path: 包含PDB文件的文件夹路径
        molecule_type: 分子类型标识符
        **kwargs: 传递给build_wt_mt_graph_pair的其他参数
        
    Returns:
        list: WT-MT图对列表，每个元素为(wt_graph, mt_graph, mutation_positions, mutation_vector)
    """
    import os
    import glob
    
    # 查找所有WT文件（pdbid_Repair.pdb）
    wt_files = glob.glob(os.path.join(folder_path, "*_Repair.pdb"))
    
    graph_pairs = []
    
    for wt_file in wt_files:
        # 提取PDB ID
        basename = os.path.basename(wt_file)
        pdb_id = basename.split('_')[0]
        
        # 查找对应的MT文件（pdbid_Repair_*.pdb）
        mt_pattern = os.path.join(folder_path, f"{pdb_id}_Repair_*.pdb")
        mt_files = glob.glob(mt_pattern)
        
        #print(f"找到WT文件: {wt_file}")
        #print(f"找到 {len(mt_files)} 个MT文件: {mt_files}")
        
        # 为每个MT文件构建图对
        for mt_file in mt_files:
            try:
                #print(f"\n处理图对: {basename} <-> {os.path.basename(mt_file)}")
                wt_graph, mt_graph, mutation_positions, mutation_vector = build_wt_mt_graph_pair(
                    wt_file, mt_file, molecule_type, **kwargs
                )
                
                graph_pairs.append((wt_graph, mt_graph, mutation_positions, mutation_vector))
                #print(f"成功构建图对，突变位点数: {len(mutation_positions)}")
                
            except Exception as e:
                #print(f"构建图对失败: {wt_file} <-> {mt_file}, 错误: {e}")
                continue
    
    #print(f"\n总共构建了 {len(graph_pairs)} 个WT-MT图对")
    return graph_pairs