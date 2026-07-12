import os
# 抗体和抗原链标识符
ANTIBODY_CHAINS = ['D']  # 抗体链标识符
ANTIGEN_CHAINS = ['A']   # 抗原链标识符
# 相互作用参数
INTERACTION_DISTANCE = 10.0  # 抗体-抗原相互作用距离阈值 (Å) #
# 突变位点扩展参数
MUTATION_EXTENSION_DISTANCE = 10.0  # 突变位点周围扩展距离 (Å)，确保突变位点及其周围区域被包含在图中  #
# 非共价边（原子图）参数
NONCOV_CROSS_CUTOFF = 5.0      # 跨链非共价半径 (Å) - 与INTERACTION_DISTANCE保持一致  #
NONCOV_WITHIN_CUTOFF = 5.0     # 同链非共价半径 (Å) - 与INTERACTION_DISTANCE保持一致
INCLUDE_WITHIN_CHAIN_NONCOV = True  # 是否加入同链非共价边
# 原子图接触面参数
ATOM_INTERFACE_CUTOFF = 6.0    # 原子图接触面距离阈值 (Å) - 用于提取接触面原子
# 残基图参数（Cα 半径 + 序列）
RESIDUE_RADIUS_CUTOFF = 10.0   # Cα–Cα 跨链半径 (Å) #
RESIDUE_WITHIN_CHAIN_CUTOFF = 10.0  # Cα–Cα 同链半径 (Å) - 用于残基图 #
RESIDUE_SEQ_K = 1             # 序列边 K（1 或 2）
# PSSM相关配置
SWISSPROT_FASTA_PATH = "/root/AbMSPN/Data/swiss/uniprot_sprot.fasta"  # SwissProt数据库路径
SWISSPROT_DB_PATH = "/root/AbMSPN/uniref/uniref90"  # BLAST数据库路径
ENABLE_PSSM = True  # 是否启用PSSM计算（False可大幅提升速度）

# 氨基酸类型编码 (20种标准氨基酸)
AMINO_ACIDS = [
    'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
    'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL'
]
# 原子类型编码（精简为蛋白质中实际存在的原子类型）
ATOM_TYPES = ['C', 'N', 'O', 'S', 'OTHER']
# 原子名称编码（基于GearBind的37种标准原子名称）
ATOM_NAMES = [
    'N', 'CA', 'C', 'CB', 'O', 'CG', 'CG1', 'CG2', 'OG', 'OG1', 'SG', 'CD',
    'CD1', 'CD2', 'ND1', 'ND2', 'OD1', 'OD2', 'SD', 'CE', 'CE1', 'CE2', 'CE3',
    'NE', 'NE1', 'NE2', 'OE1', 'OE2', 'CH2', 'NH1', 'NH2', 'OH', 'CZ', 'CZ2',
    'CZ3', 'NZ', 'OXT'
]
# 范德华半径 (单位: Å)
VDW_RADII = {
    'C': 1.7, 'N': 1.55, 'O': 1.52, 'S': 1.8,
    'H': 1.2, 'P': 1.8, 'F': 1.47, 'Cl': 1.75,
    'Br': 1.85, 'I': 1.98
}
# ESM2 本地权重与加速配置
# 若提供本地 .pt 模型路径，将优先使用本地权重加载；否则按模型名在线/缓存加载
ESM2_MODEL_NAME = 'esm2_t33_650M_UR50D'  # 默认使用 650M 模型
ESM2_MODEL_PATH = '/root/AbMSPN/Data/esm/esm2_t33_650M_UR50D.pt'  # 若存在则使用本地权重
# 加速与缓存
ESM2_USE_AUTOCUDA = False            # 强制使用 CPU（不使用 CUDA）
ESM2_USE_AMP = False                 # 仅在 CUDA 使用，CPU 下关闭
ESM2_BATCH_SIZE = 4                  # 每批处理的链条数量
ESM2_CACHE_DIR = os.path.join(os.path.dirname(__file__), 'esm_cache')  # ESM 嵌入缓存目录
ESM2_ENABLE_CACHE = True             # 启用磁盘缓存
# CPU 专用加速项
ESM2_CPU_THREADS = 5                 # CPU 推理使用的线程数（None 表示保持默认）
ESM2_USE_QUANTIZATION = True         # CPU 上对 Linear 层使用动态量化（int8）
ESM2_ONLY_ANTIBODY = False            # 仅为抗体链计算 ESM（加速）
ESM2_SORT_BY_LENGTH = True           # 按序列长度排序后再批处理，减少填充
# 氨基酸内部连接模板 (以键名存储)
RESIDUE_TOPOLOGY = {
    "ALA": [("N", "CA"), ("CA", "C"), ("C", "O"), ("CA", "CB")],
    "ARG": [("N", "CA"), ("CA", "C"), ("C", "O"), ("CA", "CB"), 
            ("CB", "CG"), ("CG", "CD"), ("CD", "NE"), ("NE", "CZ"), ("CZ", "NH1"), ("CZ", "NH2")],
    "ASN": [("N", "CA"), ("CA", "C"), ("C", "O"), ("CA", "CB"), ("CB", "CG"), ("CG", "OD1"), ("CG", "ND2")],
    "ASP": [("N", "CA"), ("CA", "C"), ("C", "O"), ("CA", "CB"), ("CB", "CG"), ("CG", "OD1"), ("CG", "OD2")],
    "CYS": [("N", "CA"), ("CA", "C"), ("C", "O"), ("CA", "CB"), ("CB", "SG")],
    "GLN": [("N", "CA"), ("CA", "C"), ("C", "O"), ("CA", "CB"), 
            ("CB", "CG"), ("CG", "CD"), ("CD", "OE1"), ("CD", "NE2")],
    "GLU": [("N", "CA"), ("CA", "C"), ("C", "O"), ("CA", "CB"), 
            ("CB", "CG"), ("CG", "CD"), ("CD", "OE1"), ("CD", "OE2")],
    "GLY": [("N", "CA"), ("CA", "C"), ("C", "O")],
    "HIS": [("N", "CA"), ("CA", "C"), ("C", "O"), ("CA", "CB"), ("CB", "CG"), ("CG", "ND1"), ("CG", "CD2")],
    "ILE": [("N", "CA"), ("CA", "C"), ("C", "O"), ("CA", "CB"), ("CB", "CG1"), ("CB", "CG2"), ("CG1", "CD1")],
    "LEU": [("N", "CA"), ("CA", "C"), ("C", "O"), ("CA", "CB"), ("CB", "CG"), ("CG", "CD1"), ("CG", "CD2")],
    "LYS": [("N", "CA"), ("CA", "C"), ("C", "O"), ("CA", "CB"), 
            ("CB", "CG"), ("CG", "CD"), ("CD", "CE"), ("CE", "NZ")],
    "MET": [("N", "CA"), ("CA", "C"), ("C", "O"), ("CA", "CB"), 
            ("CB", "CG"), ("CG", "SD"), ("SD", "CE")],
    "PHE": [("N", "CA"), ("CA", "C"), ("C", "O"), ("CA", "CB"), ("CB", "CG"), ("CG", "CD1"), ("CG", "CD2")],
    "PRO": [("N", "CA"), ("CA", "C"), ("C", "O"), ("CA", "CB"), ("CB", "CG"), ("CG", "CD")],
    "SER": [("N", "CA"), ("CA", "C"), ("C", "O"), ("CA", "CB"), ("CB", "OG")],
    "THR": [("N", "CA"), ("CA", "C"), ("C", "O"), ("CA", "CB"), ("CB", "OG1"), ("CB", "CG2")],
    "TRP": [("N", "CA"), ("CA", "C"), ("C", "O"), ("CA", "CB"), ("CB", "CG"), ("CG", "CD1"), ("CG", "CD2")],
    "TYR": [("N", "CA"), ("CA", "C"), ("C", "O"), ("CA", "CB"), ("CB", "CG"), ("CG", "CD1"), ("CG", "CD2")],
    "VAL": [("N", "CA"), ("CA", "C"), ("C", "O"), ("CA", "CB"), ("CB", "CG1"), ("CB", "CG2")]
}
# 氨基酸3字母到1字母的映射
AA_3TO1 = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
}
# 创建索引映射
amino_acid_to_idx = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
atom_type_to_idx = {atype: i for i, atype in enumerate(ATOM_TYPES)}
atom_name_to_idx = {aname: i for i, aname in enumerate(ATOM_NAMES)}
# WT/MT文件路径工具函数
def get_wt_file_path(pdb_id=None, base_dir=None):
    if pdb_id is None:
        pdb_id = DEFAULT_PDB_ID
    if base_dir is None:
        base_dir = BASE_DATA_DIR
    
    filename = WT_FILE_PATTERN.format(pdb_id=pdb_id)
    # 添加子目录路径
    return os.path.join(base_dir, pdb_id.lower(), filename)

def get_mt_file_path(pdb_id=None, mutation_id=None, base_dir=None):
    if pdb_id is None:
        pdb_id = DEFAULT_PDB_ID
    if mutation_id is None:
        mutation_id = DEFAULT_MUTATION_ID
    if base_dir is None:
        base_dir = BASE_DATA_DIR
    
    filename = MT_FILE_PATTERN.format(pdb_id=pdb_id, mutation_id=mutation_id)
    # 添加子目录路径
    return os.path.join(base_dir, pdb_id.lower(), filename)

def get_wt_mt_file_pair(pdb_id=None, mutation_id=None, base_dir=None):
    wt_path = get_wt_file_path(pdb_id, base_dir)
    mt_path = get_mt_file_path(pdb_id, mutation_id, base_dir)
    return wt_path, mt_path
