#!/usr/bin/env python3
"""
优化的PSSM预计算脚本
基于序列去重分析，避免重复序列的重复PSSM计算

功能特点:
- 利用序列去重分析结果
- 只为唯一序列计算PSSM
- 为重复序列创建符号链接或复制
- 大幅减少计算时间和存储空间
"""

import os
import sys
import pandas as pd
import logging
import pickle
import time
import shutil
from pathlib import Path
from multiprocessing import Pool, cpu_count
import numpy as np
from features import get_pssm_features, _check_blast_database
from config import SWISSPROT_DB_PATH, SWISSPROT_FASTA_PATH, ENABLE_PSSM
import argparse
from textwrap import dedent

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 配置参数
BASE_DATA_PATH = "/root/AbMSPN_Data/S1131-SKEMPI/FoldX_Output/Mutants_repaired_only"
PSSM_CACHE_DIR = "/root/AbMSPN/Data/pssm_S1131_cache"
SEQUENCE_INDEX_FILE = os.path.join(PSSM_CACHE_DIR, "sequence_index.pkl")
PSSM_INDEX_FILE = os.path.join(PSSM_CACHE_DIR, "pssm_index.pkl")
OPTIMIZATION_REPORT_FILE = os.path.join(PSSM_CACHE_DIR, "pssm_optimization_report.csv")

# 多进程配置
NUM_PROCESSES = 4  # 进程数：2个进程
BLAST_THREADS_PER_PROCESS = 4  # 每个进程的BLAST线程数：8个线程
# 或者使用 4×4 配置：
# NUM_PROCESSES = 4
# BLAST_THREADS_PER_PROCESS = 4

def load_sequence_analysis():
    """加载序列分析结果"""
    if os.path.exists(SEQUENCE_INDEX_FILE):
        with open(SEQUENCE_INDEX_FILE, 'rb') as f:
            return pickle.load(f)
    return None

def load_pssm_index():
    """加载PSSM索引文件"""
    if os.path.exists(PSSM_INDEX_FILE):
        with open(PSSM_INDEX_FILE, 'rb') as f:
            return pickle.load(f)
    return {}

def save_pssm_index(index):
    """保存PSSM索引文件"""
    os.makedirs(PSSM_CACHE_DIR, exist_ok=True)
    with open(PSSM_INDEX_FILE, 'wb') as f:
        pickle.dump(index, f)

def get_sequence_hash(sequence):
    """获取序列的哈希值"""
    import hashlib
    return hashlib.md5(sequence.encode()).hexdigest()

def compute_pssm_for_sequence(sequence, chain_id, sequence_hash, blast_threads=None):
    """为单个序列计算PSSM
    
    Args:
        sequence: 序列字符串
        chain_id: 链ID
        sequence_hash: 序列哈希值
        blast_threads: BLAST使用的线程数，如果为None则使用默认值
    """
    try:
        # 创建临时序列字典
        sequences = {chain_id: sequence}
        
        # 计算PSSM（传入线程数参数）
        pssm_features = get_pssm_features(sequences, f"temp_{chain_id}.pdb", blast_threads=blast_threads)
        
        if chain_id in pssm_features and pssm_features[chain_id].size > 0:
            return pssm_features[chain_id]
        else:
            logger.warning(f"链 {chain_id} 的PSSM计算失败，返回零矩阵")
            return np.zeros((len(sequence), 20))
            
    except Exception as e:
        logger.error(f"计算链 {chain_id} 的PSSM失败: {e}")
        return np.zeros((len(sequence), 20))

def process_single_sequence(args):
    """处理单个序列的包装函数，用于多进程"""
    seq_hash, chain_info, pssm_index_dict, blast_threads = args
    
    pdb_chain_key = chain_info['pdb_chain_key']
    sequence = chain_info['sequence']
    chain_id = chain_info['chain_id']
    
    # 直接检查npy文件是否存在（不依赖索引）
    default_pssm_file = os.path.join(PSSM_CACHE_DIR, f"{pdb_chain_key}.npy")
    if os.path.exists(default_pssm_file):
        logger.info(f"[进程 {os.getpid()}] 链 {pdb_chain_key} 的PSSM文件已存在，跳过计算")
        return {
            'pdb_chain_key': pdb_chain_key,
            'sequence_hash': seq_hash,
            'sequence_length': len(sequence),
            'status': 'cached',
            'pssm_file': default_pssm_file
        }
    
    # 如果索引中有其他路径，也检查一下
    if pdb_chain_key in pssm_index_dict:
        pssm_file = pssm_index_dict[pdb_chain_key].get('pssm_file')
        if pssm_file and os.path.exists(pssm_file) and pssm_file != default_pssm_file:
            logger.info(f"[进程 {os.getpid()}] 链 {pdb_chain_key} 的PSSM文件已存在（索引路径），跳过计算")
            return {
                'pdb_chain_key': pdb_chain_key,
                'sequence_hash': seq_hash,
                'sequence_length': len(sequence),
                'status': 'cached',
                'pssm_file': pssm_file
            }
    
    # 计算PSSM
    logger.info(f"[进程 {os.getpid()}] 计算链 {pdb_chain_key} 的PSSM (长度: {len(sequence)}, 线程数: {blast_threads})")
    start_time = time.time()
    
    pssm_matrix = compute_pssm_for_sequence(sequence, chain_id, seq_hash, blast_threads=blast_threads)
    
    end_time = time.time()
    duration = end_time - start_time
    
    # 保存PSSM矩阵
    pssm_file = os.path.join(PSSM_CACHE_DIR, f"{pdb_chain_key}.npy")
    np.save(pssm_file, pssm_matrix)
    
    logger.info(f"[进程 {os.getpid()}] 链 {pdb_chain_key} PSSM计算完成，耗时: {duration:.2f}秒")
    
    return {
        'pdb_chain_key': pdb_chain_key,
        'sequence_hash': seq_hash,
        'sequence_length': len(sequence),
        'status': 'new',
        'pssm_file': pssm_file,
        'computation_time': duration,
        'pdb_name': chain_info['pdb_name'],
        'chain_id': chain_id
    }

def process_representative_sequences():
    """处理代表序列，计算PSSM（支持多进程）"""
    logger.info("开始处理代表序列")
    logger.info(f"多进程配置: {NUM_PROCESSES} 个进程，每个进程使用 {BLAST_THREADS_PER_PROCESS} 个BLAST线程")
    
    # 加载序列分析结果
    analysis_data = load_sequence_analysis()
    if not analysis_data:
        logger.error("未找到序列分析数据，请先运行序列去重分析")
        return []
    
    sequence_info = analysis_data['sequence_info']
    pdb_chain_info = analysis_data['pdb_chain_info']
    
    # 加载现有PSSM索引
    pssm_index = load_pssm_index()
    
    # 为每个唯一序列选择一个代表链进行PSSM计算
    representative_chains = {}
    computation_plan = {}
    
    for seq_hash, info in sequence_info.items():
        # 选择第一个链作为代表
        representative_pdb, representative_chain = info['pdb_chains'][0]
        representative_key = f"{representative_pdb}_{representative_chain}"
        
        representative_chains[seq_hash] = {
            'pdb_name': representative_pdb,
            'chain_id': representative_chain,
            'pdb_chain_key': representative_key,
            'sequence': info['sequence'],
            'sequence_length': info['sequence_length']
        }
        
        # 为所有相同序列的链分配相同的PSSM文件
        for pdb_name, chain_id in info['pdb_chains']:
            pdb_chain_key = f"{pdb_name}_{chain_id}"
            computation_plan[pdb_chain_key] = {
                'representative_key': representative_key,
                'pssm_file': f"{representative_key}.npy",
                'needs_computation': pdb_chain_key == representative_key
            }
    
    logger.info(f"优化计算计划:")
    logger.info(f"  需要计算的唯一序列数: {len(representative_chains)}")
    logger.info(f"  总链数: {len(computation_plan)}")
    logger.info(f"  计算节省: {len(computation_plan) - len(representative_chains)} 个链")
    
    # 过滤出需要计算的序列（排除已缓存的）
    tasks_to_compute = []
    cached_results = []
    
    for seq_hash, chain_info in representative_chains.items():
        pdb_chain_key = chain_info['pdb_chain_key']
        # 直接检查npy文件是否存在（不依赖索引）
        default_pssm_file = os.path.join(PSSM_CACHE_DIR, f"{pdb_chain_key}.npy")
        if os.path.exists(default_pssm_file):
            cached_results.append({
                'pdb_chain_key': pdb_chain_key,
                'sequence_hash': seq_hash,
                'sequence_length': len(chain_info['sequence']),
                'status': 'cached',
                'pssm_file': default_pssm_file
            })
        else:
            # 如果索引中有其他路径，也检查一下
            if pdb_chain_key in pssm_index:
                pssm_file = pssm_index[pdb_chain_key].get('pssm_file')
                if pssm_file and os.path.exists(pssm_file):
                    cached_results.append({
                        'pdb_chain_key': pdb_chain_key,
                        'sequence_hash': seq_hash,
                        'sequence_length': len(chain_info['sequence']),
                        'status': 'cached',
                        'pssm_file': pssm_file
                    })
                else:
                    # 文件不存在，需要重新计算
                    tasks_to_compute.append((seq_hash, chain_info, pssm_index.copy(), BLAST_THREADS_PER_PROCESS))
            else:
                # 索引中也没有，需要重新计算
                tasks_to_compute.append((seq_hash, chain_info, pssm_index.copy(), BLAST_THREADS_PER_PROCESS))
    
    logger.info(f"  已缓存: {len(cached_results)}")
    logger.info(f"  需要计算: {len(tasks_to_compute)}")
    
    # 使用多进程处理
    results = cached_results.copy()
    successful_count = len(cached_results)
    error_count = 0
    
    if tasks_to_compute:
        if NUM_PROCESSES > 1:
            logger.info(f"使用 {NUM_PROCESSES} 个进程并行计算")
            with Pool(processes=NUM_PROCESSES) as pool:
                computed_results = pool.map(process_single_sequence, tasks_to_compute)
        else:
            logger.info("使用单进程计算")
            computed_results = [process_single_sequence(task) for task in tasks_to_compute]
        
        # 更新索引并收集结果
        for result in computed_results:
            if result['status'] == 'new':
                pdb_chain_key = result['pdb_chain_key']
                pssm_index[pdb_chain_key] = {
                    'pdb_name': result['pdb_name'],
                    'chain_id': result['chain_id'],
                    'sequence_length': result['sequence_length'],
                    'pssm_file': result['pssm_file'],
                    'computed_time': time.time(),
                    'sequence_hash': result['sequence_hash']
                }
                successful_count += 1
            results.append(result)
    
    # 保存更新的索引（重新加载以确保包含所有更新）
    pssm_index = load_pssm_index()
    for result in computed_results:
        if result['status'] == 'new':
            pdb_chain_key = result['pdb_chain_key']
            if pdb_chain_key not in pssm_index:
                pssm_index[pdb_chain_key] = {
                    'pdb_name': result['pdb_name'],
                    'chain_id': result['chain_id'],
                    'sequence_length': result['sequence_length'],
                    'pssm_file': result['pssm_file'],
                    'computed_time': time.time(),
                    'sequence_hash': result['sequence_hash']
                }
    save_pssm_index(pssm_index)
    
    logger.info(f"代表序列处理完成:")
    logger.info(f"  新计算: {successful_count - len(cached_results)}")
    logger.info(f"  缓存命中: {len(cached_results)}")
    logger.info(f"  错误: {error_count}")
    
    return results, computation_plan

def create_duplicate_links(computation_plan):
    """为重复序列创建符号链接或复制PSSM文件"""
    logger.info("开始为重复序列创建链接")
    
    # 加载PSSM索引
    pssm_index = load_pssm_index()
    
    link_count = 0
    error_count = 0
    
    for pdb_chain_key, plan_info in computation_plan.items():
        if not plan_info['needs_computation']:
            # 这是一个重复序列，需要创建链接
            representative_key = plan_info['representative_key']
            target_pssm_file = plan_info['pssm_file']
            source_pssm_file = os.path.join(PSSM_CACHE_DIR, target_pssm_file)
            
            if os.path.exists(source_pssm_file):
                # 创建符号链接
                link_file = os.path.join(PSSM_CACHE_DIR, f"{pdb_chain_key}.npy")
                
                try:
                    if os.path.exists(link_file):
                        os.remove(link_file)
                    
                    # 创建符号链接
                    os.symlink(target_pssm_file, link_file)
                    
                    # 更新索引
                    pdb_name, chain_id = pdb_chain_key.rsplit('_', 1)
                    pssm_index[pdb_chain_key] = {
                        'pdb_name': pdb_name,
                        'chain_id': chain_id,
                        'pssm_file': link_file,
                        'linked_from': representative_key,
                        'linked_time': time.time()
                    }
                    
                    link_count += 1
                    logger.debug(f"为 {pdb_chain_key} 创建链接到 {representative_key}")
                    
                except Exception as e:
                    logger.error(f"为 {pdb_chain_key} 创建链接失败: {e}")
                    error_count += 1
            else:
                logger.warning(f"源PSSM文件不存在: {source_pssm_file}")
                error_count += 1
    
    # 保存更新的索引
    save_pssm_index(pssm_index)
    
    logger.info(f"重复序列链接创建完成:")
    logger.info(f"  成功创建链接: {link_count}")
    logger.info(f"  错误: {error_count}")
    
    return link_count, error_count

def generate_optimization_report(computation_plan, representative_results, link_count, error_count):
    """生成优化报告"""
    logger.info("生成优化报告")
    
    # 统计信息
    total_chains = len(computation_plan)
    unique_sequences = len([p for p in computation_plan.values() if p['needs_computation']])
    duplicate_chains = total_chains - unique_sequences
    
    # 计算节省
    computation_savings = duplicate_chains
    time_savings_percent = (computation_savings / total_chains) * 100
    
    # 生成报告数据
    report_data = []
    
    for pdb_chain_key, plan_info in computation_plan.items():
        is_representative = plan_info['needs_computation']
        representative_key = plan_info['representative_key']
        
        report_data.append({
            'pdb_chain_key': pdb_chain_key,
            'is_representative': is_representative,
            'representative_key': representative_key if not is_representative else pdb_chain_key,
            'pssm_file': plan_info['pssm_file'],
            'optimization_type': 'computed' if is_representative else 'linked'
        })
    
    # 保存报告
    report_df = pd.DataFrame(report_data)
    report_df.to_csv(OPTIMIZATION_REPORT_FILE, index=False)
    
    # 打印统计信息
    logger.info(f"优化报告:")
    logger.info(f"  总链数: {total_chains}")
    logger.info(f"  唯一序列数: {unique_sequences}")
    logger.info(f"  重复链数: {duplicate_chains}")
    logger.info(f"  计算节省: {computation_savings} 个链")
    logger.info(f"  时间节省: {time_savings_percent:.1f}%")
    logger.info(f"  成功创建链接: {link_count}")
    logger.info(f"  链接错误: {error_count}")
    logger.info(f"  报告保存到: {OPTIMIZATION_REPORT_FILE}")
    
    return {
        'total_chains': total_chains,
        'unique_sequences': unique_sequences,
        'duplicate_chains': duplicate_chains,
        'computation_savings': computation_savings,
        'time_savings_percent': time_savings_percent,
        'link_count': link_count,
        'error_count': error_count
    }

def main():
    """主函数"""
    logger.info("开始优化的PSSM预计算")
    
    # 检查配置
    if not ENABLE_PSSM:
        logger.warning("PSSM计算已禁用，请检查配置")
        return
    
    # 检查数据库
    if not _check_blast_database():
        logger.error("BLAST数据库不存在，请先构建数据库")
        return
    
    # 检查序列分析数据
    analysis_data = load_sequence_analysis()
    if not analysis_data:
        logger.error("未找到序列分析数据，请先运行序列去重分析")
        return
    
    start_time = time.time()
    
    # 处理代表序列
    representative_results, computation_plan = process_representative_sequences()
    
    # 为重复序列创建链接
    link_count, error_count = create_duplicate_links(computation_plan)
    
    # 生成优化报告
    optimization_stats = generate_optimization_report(
        computation_plan, representative_results, link_count, error_count
    )
    
    end_time = time.time()
    total_time = end_time - start_time
    
    logger.info(f"优化的PSSM预计算完成!")
    logger.info(f"总耗时: {total_time:.2f} 秒")
    logger.info(f"计算节省: {optimization_stats['computation_savings']} 个链")
    logger.info(f"时间节省: {optimization_stats['time_savings_percent']:.1f}%")
    
    return optimization_stats

if __name__ == "__main__":
    main()
