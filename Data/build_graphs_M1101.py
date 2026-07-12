#!/usr/bin/env python3

import os
import pandas as pd
import logging
import glob
import argparse
import torch
from multiprocessing import Pool, cpu_count
import time
from pdb_processor import process_pdb_file, cleanup_temp_file
from graph_builder_mutation import build_molecular_graph, build_wt_mt_graph_pair
from config import ANTIBODY_CHAINS, ANTIGEN_CHAINS, INTERACTION_DISTANCE

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
BASE_DATA_PATH = "/root/AbMSPN_Data/M1101-Ab-Bind"  # PDB文件所在目录（Zenodo）
CSV_FILE_PATH = "/root/AbMSPN/Data/AB-Bind_experimental_data.csv"  # CSV文件路径

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

def find_pdb_files(complex_name, base_path):
    name_variants = [
        complex_name.lower(),
        complex_name.upper(),
        complex_name.capitalize(),
        complex_name
    ]
    possible_patterns = []
    for variant in name_variants:
        possible_patterns.extend([
            f"{base_path}/{variant}/*.pdb",
            f"{base_path}/*{variant}*.pdb",
        ])
    for pattern in possible_patterns:
        files = glob.glob(pattern)
        if files:
            return files
    return []

def find_wt_mt_pdb_files(complex_name, base_path):
    name_variants = [
        complex_name.lower(),
        complex_name.upper(),
        complex_name.capitalize(),
        complex_name
    ]
    wt_files = []
    mt_files = []
    for variant in name_variants:
        wt_pattern = f"{base_path}/{variant}/*_Repair.pdb"
        wt_found = glob.glob(wt_pattern)
        if wt_found:
            wt_files.extend(wt_found)
        mt_pattern = f"{base_path}/{variant}/*_Repair_*.pdb"
        mt_found = glob.glob(mt_pattern)
        if mt_found:
            mt_files.extend(mt_found)
    return wt_files, mt_files

def load_chain_mappings_from_csv(csv_file_path):
    import pandas as pd
    try:
        df = pd.read_csv(csv_file_path, comment='#')
        complex_info = df[['PDB', 'Partners(A_B)']].drop_duplicates()
        chain_mappings = {}
        for idx, row in complex_info.iterrows():
            pdb_id = row['PDB']
            partners = row['Partners(A_B)']
            chain_config = parse_chain_mapping(partners)
            chain_mappings[pdb_id] = chain_config
            chain_mappings[pdb_id.lower()] = chain_config
        logger.info(f"从CSV文件加载了 {len(chain_mappings)} 个复合物的链映射信息")
        return chain_mappings
    except Exception as e:
        logger.error(f"从CSV文件加载链映射信息失败: {e}")
        return {}

def process_complex_with_chain_config(pdb_file, output_folder, antibody_chains, antigen_chains, atom_graph_mode="interaction"):
    try:
        complex_name = os.path.splitext(os.path.basename(pdb_file))[0]
        graph_save_path = os.path.join(output_folder, f'{complex_name}_graph.pt')
        if os.path.exists(graph_save_path):
            logger.info(f"跳过已存在的相互作用图文件: {complex_name}")
            return {
                "complex_name": complex_name,
                "pdb_file": pdb_file,
                "antibody_chains": antibody_chains,
                "antigen_chains": antigen_chains,
                "status": "cached",
                "file_path": graph_save_path
            }
        logger.info(f"处理: {complex_name} (抗体链: {antibody_chains}, 抗原链: {antigen_chains})")
        original_antibody_chains = ANTIBODY_CHAINS.copy()
        original_antigen_chains = ANTIGEN_CHAINS.copy()
        ANTIBODY_CHAINS.clear()
        ANTIBODY_CHAINS.extend(antibody_chains)
        ANTIGEN_CHAINS.clear()
        ANTIGEN_CHAINS.extend(antigen_chains)
        pdb_file_to_use, temp_pdb_file = process_pdb_file(pdb_file)
        try:
            interaction_graph, edge_stats, residue_info, residue_to_seq_idx = build_molecular_graph(
                pdb_file_to_use, 
                complex_name, 
                # 原子特征开关
                use_atom_type=True, use_hbond=True, use_charge=True, use_hydrophobic=True, use_asa=True, use_residue_type=True, use_atom_name=True,
                # 残基特征开关
                use_aa_type=True, use_polar=True, use_aromatic=True, use_residue_hydrophobic=True, use_pssm=False, use_esm2=False, use_dssp=True,
                atom_edge_mode=atom_graph_mode
            )
            torch.save(interaction_graph, graph_save_path)
            metadata = {
                "complex_name": complex_name,
                "pdb_file": pdb_file,
                "antibody_chains": antibody_chains,
                "antigen_chains": antigen_chains,
                "num_atom_nodes": len(interaction_graph.x_atom) if hasattr(interaction_graph, 'x_atom') else 0,
                "num_residue_nodes": len(interaction_graph.x_residues) if hasattr(interaction_graph, 'x_residues') else 0,
                "num_atom_edges": interaction_graph.edge_index_atom.size(1) if hasattr(interaction_graph, 'edge_index_atom') else 0,
                "num_residue_edges": interaction_graph.edge_index_residues.size(1) if hasattr(interaction_graph, 'edge_index_residues') else 0,
                "atom_feature_dim": interaction_graph.x_atom.shape[1] if hasattr(interaction_graph, 'x_atom') and interaction_graph.x_atom is not None else 0,
                "residue_feature_dim": interaction_graph.x_residues.shape[1] if hasattr(interaction_graph, 'x_residues') and interaction_graph.x_residues is not None else 0,
                "atom_edge_feature_dim": interaction_graph.edge_attr_atom.shape[1] if hasattr(interaction_graph, 'edge_attr_atom') and interaction_graph.edge_attr_atom is not None else 0,
                "residue_edge_feature_dim": interaction_graph.edge_attr_residues.shape[1] if hasattr(interaction_graph, 'edge_attr_residues') and interaction_graph.edge_attr_residues is not None else 0,
                "atom_vector_shape": list(interaction_graph.atom_vector.shape) if hasattr(interaction_graph, 'atom_vector') and interaction_graph.atom_vector is not None else [0, 0, 0],
                "residue_vector_shape": list(interaction_graph.residue_vector.shape) if hasattr(interaction_graph, 'residue_vector') and interaction_graph.residue_vector is not None else [0, 0, 0],
                "atom_edge_vector_shape": list(interaction_graph.edge_vector_atom.shape) if hasattr(interaction_graph, 'edge_vector_atom') and interaction_graph.edge_vector_atom is not None else [0, 0, 0],
                "residue_edge_vector_shape": list(interaction_graph.edge_vector_residues.shape) if hasattr(interaction_graph, 'edge_vector_residues') and interaction_graph.edge_vector_residues is not None else [0, 0, 0],
                "interaction_distance": INTERACTION_DISTANCE,
                "hydrogen_bonds": edge_stats.get('hydrogen', 0),
                "ionic_bonds": edge_stats.get('ionic', 0),
                "hydrophobic_interactions": edge_stats.get('hydrophobic', 0),
                "vdw_edges": edge_stats.get('vdw', 0),
                "radius_edges": edge_stats.get('radius', 0),
                "knn_edges": edge_stats.get('knn', 0),
                "total_edges": edge_stats.get('total', 0),
                "atom_graph_mode": atom_graph_mode,
                "file_path": graph_save_path,
                "status": "new"
            }
            logger.info(f"完成: {complex_name} - 原子节点数: {metadata['num_atom_nodes']}, 残基节点数: {metadata['num_residue_nodes']}")
            return metadata
        finally:
            ANTIBODY_CHAINS.clear()
            ANTIBODY_CHAINS.extend(original_antibody_chains)
            ANTIGEN_CHAINS.clear()
            ANTIGEN_CHAINS.extend(original_antigen_chains)
            cleanup_temp_file(temp_pdb_file)     
    except Exception as e:
        error_msg = f"处理文件 {pdb_file} 时出错: {str(e)}"
        logger.error(error_msg)
        return {
            "complex_name": complex_name if 'complex_name' in locals() else "unknown",
            "pdb_file": pdb_file,
            "antibody_chains": antibody_chains,
            "antigen_chains": antigen_chains,
            "status": "error",
            "error": error_msg
        }

def process_wt_mt_complex(wt_file, mt_file, output_folder, complex_name, antibody_chains=None, antigen_chains=None, atom_graph_mode="interaction"):
    try:
        wt_basename = os.path.splitext(os.path.basename(wt_file))[0]
        mt_basename = os.path.splitext(os.path.basename(mt_file))[0]
        wt_graph_path = os.path.join(output_folder, f'{mt_basename}_wt_graph.pt')
        mt_graph_path = os.path.join(output_folder, f'{mt_basename}_graph.pt')
        if os.path.exists(wt_graph_path) and os.path.exists(mt_graph_path):
            logger.info(f"跳过已存在的图文件: {wt_basename} <-> {mt_basename}")
            return {
                "complex_name": complex_name,
                "wt_file": wt_file,
                "mt_file": mt_file,
                "wt_graph_path": wt_graph_path,
                "mt_graph_path": mt_graph_path,
                "status": "cached"
            }
        logger.info(f"处理WT-MT图对: {wt_basename} <-> {mt_basename}")

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
                use_hydrophobic=False, use_asa=True, 
                use_residue_type=True, use_atom_name=True,
                # 残基特征开关
                use_aa_type=True, use_polar=False, use_aromatic=False, 
                use_residue_hydrophobic=False,
                use_pssm=True, use_esm2=False, use_dssp=True,
                atom_edge_mode=atom_graph_mode
            )
            torch.save(wt_graph, wt_graph_path)
            torch.save(mt_graph, mt_graph_path)
            metadata = {
                "complex_name": complex_name,
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
            # 恢复原始链配置
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
            "wt_file": wt_file,
            "mt_file": mt_file,
            "status": "error",
            "error": error_msg
        }

def process_single_complex_normal(args):
    """普通模式：单个复合物处理任务，用于多进程"""
    pdb_id, partners, base_data_path, output_base_dir, atom_graph_mode = args
    try:
        chain_config = parse_chain_mapping(partners)
        antibody_chains = chain_config['antibody']
        antigen_chains = chain_config['antigen']
        pdb_files = find_pdb_files(pdb_id, base_data_path)
        if not pdb_files:
            logger.warning(f"未找到复合物 {pdb_id} 的PDB文件")
            return [{
                "complex_name": pdb_id,
                "pdb_file": "not_found",
                "antibody_chains": antibody_chains,
                "antigen_chains": antigen_chains,
                "status": "not_found",
                "error": "PDB file not found"
            }]
        results = []
        for pdb_file in pdb_files:
            complex_output_dir = os.path.join(output_base_dir, pdb_id.lower())
            os.makedirs(complex_output_dir, exist_ok=True)
            result = process_complex_with_chain_config(
                pdb_file, 
                complex_output_dir, 
                antibody_chains, 
                antigen_chains,
                atom_graph_mode=atom_graph_mode
            )
            results.append(result)
        return results
    except Exception as e:
        logger.error(f"处理复合物 {pdb_id} 时出错: {str(e)}")
        return [{
            "complex_name": pdb_id,
            "pdb_file": "error",
            "antibody_chains": [],
            "antigen_chains": [],
            "status": "error",
            "error": str(e)
        }]

def process_single_complex_wt_mt(args):
    complex_name, base_data_path, output_base_dir, chain_mappings, atom_graph_mode = args
    try:
        wt_files, mt_files = find_wt_mt_pdb_files(complex_name, base_data_path)
        if not wt_files:
            logger.warning(f"未找到复合物 {complex_name} 的WT文件")
            return [{
                "complex_name": complex_name,
                "wt_file": "not_found",
                "mt_file": "not_found",
                "status": "not_found",
                "error": "WT file not found"
            }]
        if not mt_files:
            logger.warning(f"未找到复合物 {complex_name} 的MT文件")
            return [{
                "complex_name": complex_name,
                "wt_file": wt_files[0] if wt_files else "not_found",
                "mt_file": "not_found",
                "status": "not_found",
                "error": "MT file not found"
            }]
        results = []
        wt_file = wt_files[0] if wt_files else None
        if not wt_file:
            logger.warning(f"未找到复合物 {complex_name} 的WT文件")
            return [{
                "complex_name": complex_name,
                "wt_file": "not_found",
                "mt_file": "not_found",
                "status": "not_found",
                "error": "WT file not found"
            }]
        complex_output_dir = os.path.join(output_base_dir, complex_name.lower())
        os.makedirs(complex_output_dir, exist_ok=True)
        chain_config = chain_mappings.get(complex_name, parse_chain_mapping(complex_name))
        antibody_chains = chain_config['antibody']
        antigen_chains = chain_config['antigen']
        for mt_file in mt_files:
            result = process_wt_mt_complex(
                wt_file, mt_file, complex_output_dir, complex_name,
                antibody_chains=antibody_chains, antigen_chains=antigen_chains,
                atom_graph_mode=atom_graph_mode
            )
            results.append(result)
        return results
    except Exception as e:
        logger.error(f"处理复合物 {complex_name} 时出错: {str(e)}")
        return [{
            "complex_name": complex_name,
            "wt_file": "error",
            "mt_file": "error",
            "status": "error",
            "error": str(e)
        }]

def process_normal_graphs(csv_file_path, base_data_path, output_base_dir, num_processes=None, atom_graph_mode="interaction"):
    if num_processes is None:
        num_processes = min(cpu_count(), 16)
    logger.info(f"使用 {num_processes} 个进程进行并行处理（普通图模式，原子图={atom_graph_mode}）")
    logger.info(f"读取CSV文件: {csv_file_path}")
    df = pd.read_csv(csv_file_path, comment='#')
    complex_info = df[['PDB', 'Partners(A_B)']].drop_duplicates()
    logger.info(f"找到 {len(complex_info)} 个唯一复合物")
    os.makedirs(output_base_dir, exist_ok=True)
    tasks = []

    for idx, row in complex_info.iterrows():
        pdb_id = row['PDB']
        partners = row['Partners(A_B)']
        tasks.append((pdb_id, partners, base_data_path, output_base_dir, atom_graph_mode))

    start_time = time.time()
    all_results = []
    successful_count = 0
    error_count = 0
    cached_count = 0
    with Pool(processes=num_processes) as pool:
        for i, result_list in enumerate(pool.imap(process_single_complex_normal, tasks)):
            logger.info(f"完成复合物 {i+1}/{len(tasks)}")
            for result in result_list:
                all_results.append(result)
                if result['status'] == 'new':
                    successful_count += 1
                elif result['status'] == 'cached':
                    cached_count += 1
                else:
                    error_count += 1
    end_time = time.time()

    results_df = pd.DataFrame(all_results)
    results_csv_path = os.path.join(output_base_dir, "graph_results.csv")
    results_df.to_csv(results_csv_path, index=False)
    logger.info(f"处理完成!")
    logger.info(f"总耗时: {end_time - start_time:.2f} 秒")
    logger.info(f"成功处理: {successful_count}")
    logger.info(f"缓存文件: {cached_count}")
    logger.info(f"错误/未找到: {error_count}")
    logger.info(f"结果保存到: {results_csv_path}")
    return all_results

def process_wt_mt_graphs(base_data_path, output_base_dir, csv_file_path, num_processes=None, atom_graph_mode="interaction"):
    if num_processes is None:
        num_processes = min(cpu_count(), 6)
    logger.info(f"使用 {num_processes} 个进程进行并行处理（WT-MT图模式，原子图={atom_graph_mode}）")
    chain_mappings = load_chain_mappings_from_csv(csv_file_path)
    complex_dirs = [d for d in os.listdir(base_data_path) if os.path.isdir(os.path.join(base_data_path, d))]
    logger.info(f"找到 {len(complex_dirs)} 个复合物目录")
    os.makedirs(output_base_dir, exist_ok=True)

    tasks = []
    for complex_name in complex_dirs:
        tasks.append((complex_name, base_data_path, output_base_dir, chain_mappings, atom_graph_mode))
    start_time = time.time()

    all_results = []
    successful_count = 0
    error_count = 0
    cached_count = 0
    with Pool(processes=num_processes) as pool:
        for i, result_list in enumerate(pool.imap(process_single_complex_wt_mt, tasks)):
            logger.info(f"完成复合物 {i+1}/{len(tasks)}")
            for result in result_list:
                all_results.append(result)
                if result['status'] == 'new':
                    successful_count += 1
                elif result['status'] == 'cached':
                    cached_count += 1
                else:
                    error_count += 1
    end_time = time.time()
    results_df = pd.DataFrame(all_results)
    results_csv_path = os.path.join(output_base_dir, "wt_mt_graph_results.csv")
    results_df.to_csv(results_csv_path, index=False)
    logger.info(f"处理完成!")
    logger.info(f"总耗时: {end_time - start_time:.2f} 秒")
    logger.info(f"成功处理: {successful_count}")
    logger.info(f"缓存文件: {cached_count}")
    logger.info(f"错误/未找到: {error_count}")
    logger.info(f"结果保存到: {results_csv_path}")
    return all_results

def main():
    parser = argparse.ArgumentParser(description='统一的分子图构建脚本')
    parser.add_argument('--mode', type=str, choices=['normal', 'wt_mt'], default='wt_mt',
                        help='构建模式: normal(普通分子图) 或 wt_mt(WT-MT图对)')
    parser.add_argument('--num_processes', type=int, default=3,
                        help='并行进程数（默认为CPU核心数）')
    parser.add_argument('--atom_graph_mode', type=str, choices=['interaction', 'radius'], default='radius',
                        help='原子图连边方式: interaction=仅包含物化作用边, radius=纯半径图')
    args = parser.parse_args()
    if args.mode == 'normal':
        OUTPUT_BASE_DIR = "/root/AbMSPN/Data/graphs"
        logger.info("启动普通分子图构建处理")
    else:
        OUTPUT_BASE_DIR = "/root/AbMSPN/Data/wt_mt_graphs"
        logger.info("启动WT-MT图对构建处理")
    if not os.path.exists(BASE_DATA_PATH):
        logger.error(f"数据路径不存在: {BASE_DATA_PATH}")
        return
    if args.mode == 'normal':
        if not os.path.exists(CSV_FILE_PATH):
            logger.error(f"CSV文件不存在: {CSV_FILE_PATH}")
            return
            
    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)

    if args.mode == 'normal':
        results = process_normal_graphs(
            csv_file_path=CSV_FILE_PATH,
            base_data_path=BASE_DATA_PATH,
            output_base_dir=OUTPUT_BASE_DIR,
            num_processes=args.num_processes,
            atom_graph_mode=args.atom_graph_mode
        )
    else:
        results = process_wt_mt_graphs(
            base_data_path=BASE_DATA_PATH,
            output_base_dir=OUTPUT_BASE_DIR,
            csv_file_path=CSV_FILE_PATH,
            num_processes=args.num_processes,
            atom_graph_mode=args.atom_graph_mode
        )
    
    if results:
        successful = sum(1 for r in results if r.get('status') == 'new')
        cached = sum(1 for r in results if r.get('status') == 'cached')
        errors = sum(1 for r in results if r.get('status') in ['error', 'not_found'])
        logger.info(f"最终统计:")
        logger.info(f"  成功处理: {successful}")
        logger.info(f"  缓存文件: {cached}")
        logger.info(f"  错误/未找到: {errors}")
        logger.info(f"  总计: {len(results)}")
    logger.info(f"{args.mode}模式处理完成")

if __name__ == "__main__":
    main()

