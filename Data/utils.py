import numpy as np
import logging
from config import VDW_RADII

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_vdw_radius(element):
    """获取原子的范德华半径"""
    return VDW_RADII.get(element, 1.7)  # 默认值

def is_vdw_interaction(atom1, atom2, epsilon=0.5, lower_tolerance_ratio=0.1, heavy_only=True):
    """更科学地判断两个原子是否形成范德华接触（LJ 近似的接触窗口）

    判定为接触需满足：
    - 若 heavy_only=True，则忽略任一为氢原子的配对；
    - 距离位于 [ (r1+r2)*(1-lower_tolerance_ratio), (r1+r2)+epsilon ] 的窗口内；
      其中 r1、r2 为元素的范德华半径。

    说明：
    - 上界 epsilon 表示允许的微小接触富余（默认 0.5Å，更贴近常见“接触”宽容）。
    - 下界按半径和成比例（默认 10%），避免将明显立体冲突计作接触。
    """
    e1 = getattr(atom1, 'element', '').strip()
    e2 = getattr(atom2, 'element', '').strip()
    if heavy_only and (e1 == 'H' or e2 == 'H'):
        return False

    vdw1 = get_vdw_radius(e1)
    vdw2 = get_vdw_radius(e2)
    sum_r = vdw1 + vdw2
    distance = np.linalg.norm(atom1.get_coord() - atom2.get_coord())

    lower_bound = max(0.0, sum_r * (1.0 - lower_tolerance_ratio))
    upper_bound = sum_r + epsilon

    return lower_bound <= distance <= upper_bound

def should_skip_residue_pair(residue1, residue2, min_res_diff=2):
    """判断是否应该跳过残基对（同一链内距离太近的残基）"""
    # 同一链内距离太近的残基
    if (residue1.get_parent().get_id() == residue2.get_parent().get_id() and
        abs(residue1.get_id()[1] - residue2.get_id()[1]) < min_res_diff):
        return True
    return False

def is_polar(residue_name):
    """判断氨基酸是否为极性"""
    polar_residues = ['SER', 'THR', 'ASN', 'GLN', 'CYS', 'TYR', 'HIS', 'TRP']
    return residue_name in polar_residues

def is_aromatic(residue_name):
    """判断氨基酸是否为芳香族"""
    aromatic_residues = ['PHE', 'TYR', 'TRP', 'HIS']
    return residue_name in aromatic_residues

def is_hydrophobic(residue_name):
    """判断氨基酸是否为疏水性"""
    hydrophobic_residues = ['ALA', 'VAL', 'LEU', 'ILE', 'MET', 'PHE', 'TRP', 'PRO', 'TYR'] 
    return residue_name in hydrophobic_residues

def get_residue_sequence_position(residue):
    """获取残基在序列中的位置"""
    return residue.get_id()[1]  # PDB中残基编号

def get_residue_sequence_distance(residue1, residue2):
    """计算两个残基在序列中的距离"""
    pos1 = get_residue_sequence_position(residue1)
    pos2 = get_residue_sequence_position(residue2)
    return abs(pos1 - pos2)

def calculate_bond_angle(atom1, atom2, atom3):
    """计算三个原子之间的键角 (atom1-atom2-atom3)
    返回角度（度）
    """
    try:
        v1 = atom1.get_coord() - atom2.get_coord()
        v2 = atom3.get_coord() - atom2.get_coord()
        
        # 检查向量长度
        v1_norm_val = np.linalg.norm(v1)
        v2_norm_val = np.linalg.norm(v2)
        
        if v1_norm_val == 0 or v2_norm_val == 0:
            return 0.0
        
        # 归一化向量
        v1_norm = v1 / v1_norm_val
        v2_norm = v2 / v2_norm_val
        
        # 计算夹角
        cos_angle = np.dot(v1_norm, v2_norm)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)  # 避免数值误差
        angle = np.degrees(np.arccos(cos_angle))
        
        return angle
    except Exception as e:
        return 0.0

def calculate_dihedral_angle(atom1, atom2, atom3, atom4):
    """计算四个原子之间的二面角 (atom1-atom2-atom3-atom4)
    返回角度（度）
    """
    try:
        # 计算三个向量
        v1 = atom1.get_coord() - atom2.get_coord()
        v2 = atom3.get_coord() - atom2.get_coord()
        v3 = atom4.get_coord() - atom3.get_coord()
        
        # 计算法向量
        n1 = np.cross(v1, v2)
        n2 = np.cross(v2, v3)
        
        # 检查法向量长度
        n1_norm_val = np.linalg.norm(n1)
        n2_norm_val = np.linalg.norm(n2)
        
        if n1_norm_val == 0 or n2_norm_val == 0:
            return 0.0
        
        # 归一化法向量
        n1_norm = n1 / n1_norm_val
        n2_norm = n2 / n2_norm_val
        
        # 计算二面角
        cos_dihedral = np.dot(n1_norm, n2_norm)
        cos_dihedral = np.clip(cos_dihedral, -1.0, 1.0)  # 避免数值误差
        
        # 确定二面角的符号
        sign = np.sign(np.dot(np.cross(n1_norm, n2_norm), v2))
        dihedral_angle = np.degrees(np.arccos(cos_dihedral)) * sign
        
        return dihedral_angle
    except Exception as e:
        return 0.0

def get_backbone_ca_atom(residue):
    """获取残基的主链CA原子"""
    try:
        return residue['CA']
    except KeyError:
        return None

def normalize_distance_features(atom_distance, bond_angle, dihedral_angle):
    """归一化距离和角度特征
    原子距离: 0-10Å → 0-1
    键角: 0-180° → 0-1  
    二面角: -180-180° → 0-1
    """
    # 原子距离归一化 (假设最大距离为10Å)
    normalized_distance = min(atom_distance / 10.0, 1.0)
    
    # 键角归一化 (0-180度)
    normalized_bond_angle = bond_angle / 180.0
    
    # 二面角归一化 (-180到180度)
    normalized_dihedral_angle = (dihedral_angle + 180.0) / 360.0
    
    return normalized_distance, normalized_bond_angle, normalized_dihedral_angle

def calculate_edge_angles(atom1, atom2, residue1, residue2):
    """计算两个原子之间的键角和二面角
    键角: 主链CA-atom1-atom2 或 atom1-atom2-主链CA
    二面角: 主链CA1-atom1-atom2-主链CA2
    """
    try:
        # 获取主链CA原子
        ca1 = get_backbone_ca_atom(residue1)
        ca2 = get_backbone_ca_atom(residue2)
        
        if ca1 is None or ca2 is None:
            return 0.0, 0.0
        
        # 计算键角: CA-atom1-atom2
        bond_angle1 = calculate_bond_angle(ca1, atom1, atom2)
        
        # 计算键角: atom1-atom2-CA
        bond_angle2 = calculate_bond_angle(atom1, atom2, ca2)
        
        # 使用平均键角
        bond_angle = (bond_angle1 + bond_angle2) / 2.0
        
        # 计算二面角: CA1-atom1-atom2-CA2
        dihedral_angle = calculate_dihedral_angle(ca1, atom1, atom2, ca2)
        
        return bond_angle, dihedral_angle
        
    except Exception as e:
        return 0.0, 0.0

def find_attached_hydrogens(atom):
    """查找与某原子共价连接的氢原子（距离阈值过滤，默认≤1.2Å）"""
    residue = atom.get_parent()
    pos = atom.get_coord()
    hydrogens = []
    for neighbor in residue:
        if getattr(neighbor, 'element', '').strip() == 'H':
            if np.linalg.norm(neighbor.get_coord() - pos) <= 1.2:
                hydrogens.append(neighbor)
    return hydrogens

def is_hydrophobic_interaction(atom1, atom2, residue1, residue2, max_distance=5.0):
    """检测两个原子之间是否形成疏水性相互作用"""
    # 计算距离
    distance = np.linalg.norm(atom1.get_coord() - atom2.get_coord())
    if distance > max_distance:
        return False
    
    # 获取原子和残基信息
    atom_name1, element1 = atom1.get_name().strip(), atom1.element
    atom_name2, element2 = atom2.get_name().strip(), atom2.element
    residue_name1 = residue1.get_resname()
    residue_name2 = residue2.get_resname()
    
    # 判断原子1是否疏水
    is_hydrophobic1 = False
    if element1 in ['C', 'S']:
        # 主链原子不疏水
        if atom_name1 not in ['N', 'CA', 'C', 'O']:
            # 疏水氨基酸的侧链原子
            if is_hydrophobic(residue_name1):
                is_hydrophobic1 = True
    
    # 判断原子2是否疏水
    is_hydrophobic2 = False
    if element2 in ['C', 'S']:
        # 主链原子不疏水
        if atom_name2 not in ['N', 'CA', 'C', 'O']:
            # 疏水氨基酸的侧链原子
            if is_hydrophobic(residue_name2):
                is_hydrophobic2 = True
    
    # 两个原子都必须是疏水的
    return is_hydrophobic1 and is_hydrophobic2
