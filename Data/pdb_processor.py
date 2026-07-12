"""
PDB文件处理模块 - 包含PDB文件处理相关函数
"""

import os
import subprocess
import logging
import warnings
from Bio.PDB import PDBParser
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from config import AA_3TO1, amino_acid_to_idx

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=PDBConstructionWarning)

def extract_sequence_from_structure(structure):
    """从结构中提取氨基酸序列和残基映射，只考虑抗体和抗原链"""
    from config import ANTIBODY_CHAINS, ANTIGEN_CHAINS
    
    sequences = {}
    residue_to_seq_idx = {}
    
    for model in structure:
        for chain in model:
            chain_id = chain.get_id()
            
            # 只处理抗体和抗原链
            if chain_id not in ANTIBODY_CHAINS and chain_id not in ANTIGEN_CHAINS:
                continue
                
            sequence = ""
            chain_mapping = {}
            seq_idx = 0
            
            for residue in chain:
                residue_id = residue.get_id()[1]
                if residue.get_resname() in amino_acid_to_idx:
                    sequence += AA_3TO1.get(residue.get_resname(), 'X')
                    chain_mapping[residue_id] = seq_idx
                    seq_idx += 1
                else:
                    sequence += 'X'
                    chain_mapping[residue_id] = seq_idx
                    seq_idx += 1
            
            sequences[chain_id] = sequence
            residue_to_seq_idx[chain_id] = chain_mapping
    
    return sequences, residue_to_seq_idx

def add_hydrogens_with_reduce(input_pdb, output_pdb):
    """用Reduce自动加氢"""
    cmd = ["reduce", input_pdb]
    with open(output_pdb, "w") as out:
        subprocess.run(cmd, stdout=out, stderr=subprocess.DEVNULL, check=True)

def check_hydrogens_in_pdb(pdb_file):
    """检查PDB文件是否包含氢原子"""
    with open(pdb_file) as f:
        has_hydrogens = any(line.startswith('ATOM') and line[76:78].strip() == 'H' for line in f)
    return has_hydrogens

def process_pdb_file(pdb_file, add_hydrogens=True):
    """处理PDB文件，包括加氢等预处理步骤"""
    temp_pdb_file = None
    
    try:
        # 检查是否需要加氢
        has_hydrogens = check_hydrogens_in_pdb(pdb_file)
        
        if not has_hydrogens and add_hydrogens:
            # 需要加氢，创建临时文件
            temp_pdb_file = pdb_file.replace('.pdb', '_withH.pdb')
            add_hydrogens_with_reduce(pdb_file, temp_pdb_file)
            pdb_file_to_use = temp_pdb_file
            logger.debug(f"已为 {pdb_file} 添加氢原子")
        else:
            pdb_file_to_use = pdb_file
            logger.debug(f"使用原始PDB文件: {pdb_file}")
        
        return pdb_file_to_use, temp_pdb_file
        
    except Exception as e:
        logger.error(f"处理PDB文件 {pdb_file} 时出错: {e}")
        raise

def cleanup_temp_file(temp_file):
    """清理临时文件"""
    if temp_file and os.path.exists(temp_file):
        try:
            os.remove(temp_file)
            logger.info(f"已清理临时文件: {temp_file}")
        except OSError as e:
            logger.warning(f"清理临时文件失败: {e}")

def validate_pdb_structure(pdb_file):
    """验证PDB文件结构的有效性"""
    try:
        parser = PDBParser()
        structure = parser.get_structure("validation", pdb_file)
        
        # 检查是否有模型
        if len(structure) == 0:
            logger.warning(f"PDB文件 {pdb_file} 没有模型")
            return False
        
        # 检查是否有链
        model = structure[0]
        if len(model) == 0:
            logger.warning(f"PDB文件 {pdb_file} 没有链")
            return False
        
        # 检查是否有残基
        total_residues = 0
        total_atoms = 0
        
        for chain in model:
            chain_residues = len(list(chain))
            chain_atoms = sum(len(list(residue)) for residue in chain)
            total_residues += chain_residues
            total_atoms += chain_atoms
            
            if chain_residues == 0:
                logger.warning(f"链 {chain.get_id()} 没有残基")
                return False
        
        if total_residues == 0:
            logger.warning(f"PDB文件 {pdb_file} 没有残基")
            return False
        
        if total_atoms == 0:
            logger.warning(f"PDB文件 {pdb_file} 没有原子")
            return False
        
        logger.info(f"PDB文件 {pdb_file} 验证通过: {len(model)} 链, {total_residues} 残基, {total_atoms} 原子")
        return True
        
    except Exception as e:
        logger.error(f"验证PDB文件 {pdb_file} 时出错: {e}")
        return False

def get_pdb_info(pdb_file):
    """获取PDB文件的基本信息"""
    try:
        parser = PDBParser()
        structure = parser.get_structure("info", pdb_file)
        
        info = {
            'num_models': len(structure),
            'chains': {},
            'total_residues': 0,
            'total_atoms': 0
        }
        
        for model in structure:
            for chain in model:
                chain_id = chain.get_id()
                residues = list(chain)
                atoms = sum(len(list(residue)) for residue in residues)
                
                info['chains'][chain_id] = {
                    'num_residues': len(residues),
                    'num_atoms': atoms,
                    'residue_types': list(set(residue.get_resname() for residue in residues))
                }
                
                info['total_residues'] += len(residues)
                info['total_atoms'] += atoms
        
        return info
        
    except Exception as e:
        logger.error(f"获取PDB文件信息时出错: {e}")
        return None
