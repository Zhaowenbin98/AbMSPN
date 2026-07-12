#!/usr/bin/env python3
"""
python build_graphs_S1131.py --num_processes 2
"""
import os
import pandas as pd
import logging
import argparse
import torch
from multiprocessing import Pool, cpu_count
import time
from graph_builder_mutation import build_wt_mt_graph_pair
from config import ANTIBODY_CHAINS, ANTIGEN_CHAINS

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
BASE_DATA_PATH = "/root/AbMSPN_Data/S1131-SKEMPI/FoldX_Output/Mutants_repaired_only"  # PDB文件所在目录
CSV_FILE_PATH = "/root/AbMSPN/Data/S1131_with_pdbid.csv"  # CSV文件路径（需从 S1131.txt 准备）
OUTPUT_BASE_DIR = "/root/AbMSPN/Data/S1131_graphs"  # 输出目录

CHAIN_MAPPING = {
    'A_B': {'antibody': ['A'], 'antigen': ['B']},    
    'A_BC': {'antibody': ['B','C'], 'antigen': ['A']},
    'A_D': {'antibody': ['A'], 'antigen': ['D']},
    'A_HL': {'antibody': ['H','L'], 'antigen': ['A']},    
    'AB_C': {'antibody': ['A','B'], 'antigen': ['C']},    
    'AB_CD': {'antibody': ['A', 'B'], 'antigen': ['C', 'D']},
    'C_HL': {'antibody': ['H', 'L'], 'antigen': ['C']},
    'E_HL': {'antibody': ['H', 'L'], 'antigen': ['E']},
    'G_HL': {'antibody': ['H', 'L'], 'antigen': ['G']},      
    'HL_I': {'antibody': ['H', 'L'], 'antigen': ['I']},
    'HL_P': {'antibody': ['H', 'L'], 'antigen': ['P']},
    'HL_V': {'antibody': ['H', 'L'], 'antigen': ['V']},
    'HL_Y': {'antibody': ['H', 'L'], 'antigen': ['Y']},          
    'HL_VW': {'antibody': ['H', 'L'], 'antigen': ['V', 'W']},
}

def parse_chain_mapping(partners_str):
    if partners_str in CHAIN_MAPPING:
        return CHAIN_MAPPING[partners_str]
    else:
        chains = partners_str.split('_')
        if len(chains) >= 2:
            return {'antibody': [chains[0]], 'antigen': [chains[1]]}
        else:
            logger.warning(f"无法解析链映射: {partners_str}")
            return {'antibody': ['H', 'L'], 'antigen': ['A']}

def find_wt_pdb_file(pdb_id, base_path):
    name_variants = [
        pdb_id.lower(),
        pdb_id.upper(),
        pdb_id.capitalize(),
        pdb_id
    ]
    
    for variant in name_variants:
        wt_pattern = os.path.join(base_path, f"{variant}_Repair.pdb")
        if os.path.exists(wt_pattern):
            return wt_pattern
    for root, _, files in os.walk(base_path):
        for variant in name_variants:
            filename = f"{variant}_Repair.pdb"
            if filename in files:
                return os.path.join(root, filename)
    return None

def find_mt_pdb_file(pdbid, base_path):
    pdb_file = os.path.join(base_path, pdbid)
    if os.path.exists(pdb_file):
        return pdb_file
    pdb_id = pdbid.split('_')[0]
    name_variants = [
        pdb_id.lower(),
        pdb_id.upper(),
        pdb_id.capitalize(),
        pdb_id
    ]
    for variant in name_variants:
        pdb_file = os.path.join(base_path, variant, pdbid)
        if os.path.exists(pdb_file):
            return pdb_file
    for root, _, files in os.walk(base_path):
        if pdbid in files:
            return os.path.join(root, pdbid)
    return None

def process_wt_mt_complex(wt_file, mt_file, output_folder, complex_name, mutation, ddG, antibody_chains=None, antigen_chains=None, atom_graph_mode="interaction"):
    try:
        wt_basename = os.path.splitext(os.path.basename(wt_file))[0]
        mt_basename = os.path.splitext(os.path.basename(mt_file))[0]
        wt_graph_path = os.path.join(output_folder, f'{mt_basename}_wt_graph.pt')
        mt_graph_path = os.path.join(output_folder, f'{mt_basename}_graph.pt')
        if os.path.exists(wt_graph_path) and os.path.exists(mt_graph_path):
            logger.info(f"跳过已存在的图文件: {wt_basename} <-> {mt_basename}")
            return {
                "complex_name": complex_name,
                "mutation": mutation,
                "ddG": ddG,
                "wt_file": wt_file,
                "mt_file": mt_file,
                "wt_graph_path": wt_graph_path,
                "mt_graph_path": mt_graph_path,
                "status": "cached"
            }
        
        logger.info(f"处理WT-MT图对: {wt_basename} <-> {mt_basename} (突变: {mutation}, ddG: {ddG})")

        if antibody_chains is not None and antigen_chains is not None:
            original_antibody_chains = ANTIBODY_CHAINS.copy()
            original_antigen_chains = ANTIGEN_CHAINS.copy()
            ANTIBODY_CHAINS.clear()
            ANTIBODY_CHAINS.extend(antibody_chains)
            ANTIGEN_CHAINS.clear()
            ANTIGEN_CHAINS.extend(antigen_chains)
            logger.info(f"使用链映射: 抗体链={antibody_chains}, 抗原链={antigen_chains}")
        
        try:
            wt_graph, mt_graph, mutation_sites, mutation_mask = build_wt_mt_graph_pair(
                wt_file, mt_file, "protein",
                # 原子特征开关
                use_atom_type=True, use_hbond=False, use_charge=False, 
                use_hydrophobic=False, use_asa=False, 
                use_residue_type=True, use_atom_name=True,
                # 残基特征开关
                use_aa_type=True, use_polar=False, use_aromatic=False, 
                use_residue_hydrophobic=False,
                use_pssm=True, use_esm2=False, use_dssp=False,
                atom_edge_mode=atom_graph_mode
            )
            torch.save(wt_graph, wt_graph_path)
            torch.save(mt_graph, mt_graph_path)
            metadata = {
                "complex_name": complex_name,
                "mutation": mutation,
                "ddG": ddG,
                "wt_file": wt_file,
                "mt_file": mt_file,
                "wt_graph_path": wt_graph_path,
                "mt_graph_path": mt_graph_path,
                # WT图信息
                "wt_num_atom_nodes": len(wt_graph.x_atom) if hasattr(wt_graph, 'x_atom') else 0,
                "wt_num_residue_nodes": len(wt_graph.x_residues) if hasattr(wt_graph, 'x_residues') else 0,
                "wt_num_atom_edges": wt_graph.edge_index_atom.size(1) if hasattr(wt_graph, 'edge_index_atom') else 0,
                "wt_num_residue_edges": wt_graph.edge_index_residues.size(1) if hasattr(wt_graph, 'edge_index_residues') else 0,
                # MT图信息
                "mt_num_atom_nodes": len(mt_graph.x_atom) if hasattr(mt_graph, 'x_atom') else 0,
                "mt_num_residue_nodes": len(mt_graph.x_residues) if hasattr(mt_graph, 'x_residues') else 0,
                "mt_num_atom_edges": mt_graph.edge_index_atom.size(1) if hasattr(mt_graph, 'edge_index_atom') else 0,
                "mt_num_residue_edges": mt_graph.edge_index_residues.size(1) if hasattr(mt_graph, 'edge_index_residues') else 0,
                # 突变信息
                "mutation_sites_count": sum(sum(1 for is_mutation in chain_mutations.values() if is_mutation) for chain_mutations in mutation_sites.values()),
                "status": "new"
            }
            logger.info(f"完成WT-MT图对: {wt_basename} <-> {mt_basename}")
            return metadata
            
        finally:
            if antibody_chains is not None and antigen_chains is not None:
                ANTIBODY_CHAINS.clear()
                ANTIBODY_CHAINS.extend(original_antibody_chains)
                ANTIGEN_CHAINS.clear()
                ANTIGEN_CHAINS.extend(original_antigen_chains)
        
    except Exception as e:
        error_msg = f"处理WT-MT图对 {wt_file} <-> {mt_file} 时出错: {str(e)}"
        logger.error(error_msg)
        return {
            "complex_name": complex_name,
            "mutation": mutation,
            "ddG": ddG,
            "wt_file": wt_file,
            "mt_file": mt_file,
            "status": "error",
            "error": error_msg
        }

def process_single_row(args):
    row, base_data_path, output_base_dir, atom_graph_mode = args
    
    try:
        pdb_id = row['PDB']
        partners = row['Partners']
        mutation = row['mutation']
        ddG = row['ddG']
        pdbid = row['pdbid']
        chain_config = parse_chain_mapping(partners)
        antibody_chains = chain_config['antibody']
        antigen_chains = chain_config['antigen']
        wt_file = find_wt_pdb_file(pdb_id, base_data_path)
        if not wt_file:
            logger.warning(f"未找到复合物 {pdb_id} 的WT文件")
            return {
                "pdb_id": pdb_id,
                "mutation": mutation,
                "ddG": ddG,
                "pdbid": pdbid,
                "wt_file": "not_found",
                "mt_file": "not_found",
                "status": "not_found",
                "error": "WT file not found"
            }
        mt_file = find_mt_pdb_file(pdbid, base_data_path)
        if not mt_file:
            logger.warning(f"未找到MT文件: {pdbid}")
            return {
                "pdb_id": pdb_id,
                "mutation": mutation,
                "ddG": ddG,
                "pdbid": pdbid,
                "wt_file": wt_file,
                "mt_file": "not_found",
                "status": "not_found",
                "error": f"MT file not found: {pdbid}"
            }
        complex_output_dir = os.path.join(output_base_dir, pdb_id.lower())
        os.makedirs(complex_output_dir, exist_ok=True)
        result = process_wt_mt_complex(
            wt_file, mt_file, complex_output_dir, pdb_id, mutation, ddG,
            antibody_chains=antibody_chains, antigen_chains=antigen_chains,
            atom_graph_mode=atom_graph_mode
        )
        return result
    except Exception as e:
        logger.error(f"处理行数据时出错: {str(e)}")
        return {
            "pdb_id": row.get('PDB', 'unknown'),
            "mutation": row.get('mutation', 'unknown'),
            "ddG": row.get('ddG', 0.0),
            "pdbid": row.get('pdbid', 'unknown'),
            "status": "error",
            "error": str(e)
        }

def process_s1131_graphs(csv_file_path, base_data_path, output_base_dir, num_processes=None, atom_graph_mode="interaction"):
    if num_processes is None:
        num_processes = min(cpu_count(), 4)
    logger.info(f"使用 {num_processes} 个进程进行并行处理（S1131数据集，原子图={atom_graph_mode}）")
    logger.info(f"读取CSV文件: {csv_file_path}")
    df = pd.read_csv(csv_file_path)
    logger.info(f"找到 {len(df)} 条数据记录")
    os.makedirs(output_base_dir, exist_ok=True)
    tasks = []
    for idx, row in df.iterrows():
        tasks.append((row, base_data_path, output_base_dir, atom_graph_mode))
    start_time = time.time()
    all_results = []
    successful_count = 0
    error_count = 0
    cached_count = 0
    not_found_count = 0
    with Pool(processes=num_processes) as pool:
        for i, result in enumerate(pool.imap(process_single_row, tasks)):
            if (i + 1) % 100 == 0:
                logger.info(f"完成 {i+1}/{len(tasks)} 条记录")
            all_results.append(result)
            if result['status'] == 'new':
                successful_count += 1
            elif result['status'] == 'cached':
                cached_count += 1
            elif result['status'] == 'not_found':
                not_found_count += 1
            else:
                error_count += 1
    end_time = time.time()
    results_df = pd.DataFrame(all_results)
    results_csv_path = os.path.join(output_base_dir, "s1131_graph_results.csv")
    results_df.to_csv(results_csv_path, index=False)
    logger.info(f"处理完成!")
    logger.info(f"总耗时: {end_time - start_time:.2f} 秒")
    logger.info(f"成功处理: {successful_count}")
    logger.info(f"缓存文件: {cached_count}")
    logger.info(f"文件未找到: {not_found_count}")
    logger.info(f"错误: {error_count}")
    logger.info(f"结果保存到: {results_csv_path}")
    
    return all_results

def main():
    parser = argparse.ArgumentParser(description='S1131数据集图构建脚本')
    parser.add_argument('--num_processes', type=int, default=2,
                        help='并行进程数（默认为2）')
    parser.add_argument('--base_data_path', type=str, default=BASE_DATA_PATH,
                        help='PDB文件所在目录')
    parser.add_argument('--csv_file_path', type=str, default=CSV_FILE_PATH,
                        help='CSV文件路径')
    parser.add_argument('--output_dir', type=str, default=OUTPUT_BASE_DIR,
                        help='输出目录')
    parser.add_argument('--atom_graph_mode', type=str, choices=['interaction', 'radius'], default='radius',
                        help='原子图连边方式: interaction=仅包含物化作用边, radius=纯半径图')
    
    args = parser.parse_args()
    logger.info("启动S1131数据集图构建处理")
    # 检查路径是否存在
    if not os.path.exists(args.base_data_path):
        logger.error(f"数据路径不存在: {args.base_data_path}")
        return
    if not os.path.exists(args.csv_file_path):
        logger.error(f"CSV文件不存在: {args.csv_file_path}")
        return
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    # 处理数据
    results = process_s1131_graphs(
        csv_file_path=args.csv_file_path,
        base_data_path=args.base_data_path,
        output_base_dir=args.output_dir,
        num_processes=args.num_processes,
        atom_graph_mode=args.atom_graph_mode
    )
    # 打印最终统计
    if results:
        successful = sum(1 for r in results if r.get('status') == 'new')
        cached = sum(1 for r in results if r.get('status') == 'cached')
        not_found = sum(1 for r in results if r.get('status') == 'not_found')
        errors = sum(1 for r in results if r.get('status') == 'error')
        logger.info(f"最终统计:")
        logger.info(f"  成功处理: {successful}")
        logger.info(f"  缓存文件: {cached}")
        logger.info(f"  文件未找到: {not_found}")
        logger.info(f"  错误: {errors}")
        logger.info(f"  总计: {len(results)}")
    logger.info("S1131数据集处理完成")

if __name__ == "__main__":
    main()

