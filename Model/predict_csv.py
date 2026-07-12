#!/usr/bin/env python3
import os
import sys
from pathlib import Path
import json
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import (
    Cfg, set_seed, build_AR_from_data, to_device,
    DeltaGDataset, create_model, collate_pairs_fast,
    pearsonr_torch, spearmanr_torch, r2_torch
)

def load_test_indices(json_path: str, fold_id: int = None, one_based: bool = True):
    """
    从JSONL文件中加载test集索引
    优先使用'test'键,如果没有则使用'val'键
    
    Args:
        json_path: JSONL文件路径
        fold_id: fold编号（如果为None,则读取第一行；如果指定,则读取对应fold的行）
        one_based: 索引是否从1开始（需要转换为0-based）
    
    Returns:
        test集索引列表
    """
    p = Path(json_path)
    if not p.exists():
        raise FileNotFoundError(f"Split file not found: {json_path}")
    
    lines = p.read_text().strip().splitlines()
    # 过滤空行
    lines = [line for line in lines if line.strip()]
    if len(lines) == 0:
        raise ValueError("JSONL file is empty")
    
    # 根据fold_id选择读取的行
    if fold_id is None:
        # 如果没有指定fold_id,读取第一行（通常是整个数据集的分割）
        line_obj = json.loads(lines[0])
    else:
        # 如果指定了fold_id,读取对应fold的行（fold_id从1开始,数组索引从0开始）
        pick = (fold_id - 1) % len(lines)
        line_obj = json.loads(lines[pick])
    
    # 优先使用'test'键,如果没有则使用'val'键
    test_raw = line_obj.get('test', line_obj.get('val', []))
    
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
    
    test_idx = norm_list(test_raw)
    return test_idx

def extract_pdbid_from_filename(filename: str) -> str:
    """
    从文件名提取pdbid
    
    Args:
        filename: 文件名,如 '1ak4_1_wt_graph.pt' 或 'hm_3bn9_8_wt_graph.pt'
    
    Returns:
        pdbid,如 '1AK4' 或 'HM_3BN9'
    """
    # 移除扩展名
    name = filename.replace('.pt', '')
    # 按 '_' 分割
    parts = name.split('_')
    
    if len(parts) >= 2 and parts[0].lower() == 'hm':
        # 处理 hm_ 前缀的情况,如 'hm_3bn9_8_wt_graph' -> 'HM_3BN9'
        pdbid = f"{parts[0].upper()}_{parts[1].upper()}"
    else:
        # 普通情况,如 '1ak4_1_wt_graph' -> '1AK4'
        pdbid = parts[0].upper()
    
    return pdbid

def load_model_from_folder(model_dir: str, fold_id: int, mode: str = 'dual', 
                           atom_encoder: str = 'rgcn', res_encoder: str = 'gvp'):
    """
    从文件夹中按fold编号加载模型
    
    Args:
        model_dir: 模型文件夹路径
        fold_id: fold编号（1, 2, 3...）
        mode: 模型模式 ('atom_only', 'residue_only', 'dual')
        atom_encoder: 原子编码器类型
        res_encoder: 残基编码器类型
    
    Returns:
        加载的模型
    """
    device = Cfg.device
    model_dir = Path(model_dir)
    
    # 尝试多种可能的文件名格式
    possible_paths = [
        model_dir / f"{fold_id}.pt",
        model_dir / f"fold_{fold_id}.pt",
        model_dir / f"best_e2e_cv_fold_{fold_id}_val_params.pt",
        model_dir / f"best_e2e_cv_fold_{fold_id}_val.pt",
    ]
    
    model_path = None
    for path in possible_paths:
        if path.exists():
            model_path = path
            break
    
    if model_path is None:
        raise FileNotFoundError(
            f"Model file not found for fold {fold_id}. Tried:\n" +
            "\n".join([f"  - {p}" for p in possible_paths])
        )
    
    print(f"Loading model from: {model_path}")
    
    # 创建临时数据来获取模型结构
    dataset = DeltaGDataset(
        csv_path=Cfg.csv_path,
        root_dir=Cfg.root_dir,
        split='test',
        explicit_indices=[0]  # 只取一个样本用于获取模型结构
    )
    
    # 获取一个样本用于创建模型
    wt_data, mut_data, _ = dataset[0]
    A_tmp, R_tmp = build_AR_from_data(wt_data)
    
    # 创建模型
    model = create_model(mode, A_tmp, R_tmp, device, atom_encoder, res_encoder)
    
    # 加载模型权重
    checkpoint = torch.load(str(model_path), map_location=device, weights_only=False)
    
    if isinstance(checkpoint, dict):
        # 检查不同的可能键名
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded model from checkpoint (epoch {checkpoint.get('epoch', 'unknown')})")
        elif 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
            print(f"Loaded model from checkpoint (epoch {checkpoint.get('epoch', 'unknown')})")
        else:
            # 尝试直接加载整个字典作为state_dict
            try:
                model.load_state_dict(checkpoint)
                print("Loaded model state dict directly")
            except Exception as e:
                print(f"Error loading model: {e}")
                print(f"Available keys in checkpoint: {list(checkpoint.keys())}")
                raise
    else:
        # 假设checkpoint直接是state_dict（params文件通常是这种情况）
        model.load_state_dict(checkpoint)
        print("Loaded model state dict directly")
    
    model.eval()
    return model

def load_model_from_path(model_path: str, mode: str = 'dual', 
                         atom_encoder: str = 'rgcn', res_encoder: str = 'gvp'):
    """
    从指定路径加载模型（保持向后兼容）
    
    Args:
        model_path: 模型参数文件路径
        mode: 模型模式 ('atom_only', 'residue_only', 'dual')
        atom_encoder: 原子编码器类型
        res_encoder: 残基编码器类型
    
    Returns:
        加载的模型
    """
    device = Cfg.device
    model_path = Path(model_path)
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    print(f"Loading model from: {model_path}")
    
    # 创建临时数据来获取模型结构
    dataset = DeltaGDataset(
        csv_path=Cfg.csv_path,
        root_dir=Cfg.root_dir,
        split='test',
        explicit_indices=[0]  # 只取一个样本用于获取模型结构
    )
    
    # 获取一个样本用于创建模型
    wt_data, mut_data, _ = dataset[0]
    A_tmp, R_tmp = build_AR_from_data(wt_data)
    
    # 创建模型
    model = create_model(mode, A_tmp, R_tmp, device, atom_encoder, res_encoder)
    
    # 加载模型权重
    checkpoint = torch.load(str(model_path), map_location=device, weights_only=False)
    
    if isinstance(checkpoint, dict):
        # 检查不同的可能键名
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded model from checkpoint (epoch {checkpoint.get('epoch', 'unknown')})")
        elif 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
            print(f"Loaded model from checkpoint (epoch {checkpoint.get('epoch', 'unknown')})")
        else:
            # 尝试直接加载整个字典作为state_dict
            try:
                model.load_state_dict(checkpoint)
                print("Loaded model state dict directly")
            except Exception as e:
                print(f"Error loading model: {e}")
                print(f"Available keys in checkpoint: {list(checkpoint.keys())}")
                raise
    else:
        # 假设checkpoint直接是state_dict（params文件通常是这种情况）
        model.load_state_dict(checkpoint)
        print("Loaded model state dict directly")
    
    model.eval()
    return model

def predict_on_test_set(model, test_indices, batch_size=16):
    """
    在test集上进行预测
    
    Args:
        model: 加载的模型
        test_indices: test集的索引列表
        batch_size: 批次大小
    
    Returns:
        (indices, predictions, targets, pdbids, mutations) - 原始索引、预测值、真实值、pdbid列表、mutation列表
    """
    device = Cfg.device
    
    # 先创建一个临时数据集来获取过滤后的索引映射
    # 我们需要手动跟踪哪些原始索引被保留了
    original_df = pd.read_csv(Cfg.csv_path)
    test_df_before_filter = original_df.iloc[test_indices].copy()
    test_df_before_filter['original_index'] = test_indices
    
    # 创建test数据集
    test_dataset = DeltaGDataset(
        csv_path=Cfg.csv_path,
        root_dir=Cfg.root_dir,
        split='test',
        explicit_indices=test_indices,
        filter_missing=True
    )
    
    print(f"Test set size: {len(test_dataset)} (after filtering missing files from {len(test_indices)} original samples)")
    
    # 构建过滤后的索引映射和数据信息
    # test_dataset.df是过滤后的数据,但索引已经被重置了
    # 我们需要通过匹配数据来找出原始索引
    filtered_original_indices = []
    filtered_pdbids = []
    filtered_mutations = []
    
    for i in range(len(test_dataset.df)):
        # 通过匹配行数据来找到原始索引
        row = test_dataset.df.iloc[i]
        # 在原始test_df中查找匹配的行
        matches = test_df_before_filter[
            (test_df_before_filter['wild_pt'] == row['wild_pt']) &
            (test_df_before_filter['mutant_pt'] == row['mutant_pt'])
        ]
        if len(matches) > 0:
            match_row = matches.iloc[0]
            filtered_original_indices.append(match_row['original_index'])
            # 提取pdbid和mutation
            pdbid = extract_pdbid_from_filename(match_row['wild_pt'])
            mutation = match_row.get('Mutation', '')
            filtered_pdbids.append(pdbid)
            filtered_mutations.append(mutation)
        else:
            # 如果找不到匹配,使用索引位置（可能不准确,但至少不会报错）
            filtered_original_indices.append(test_indices[i] if i < len(test_indices) else i)
            # 从当前行提取pdbid和mutation
            pdbid = extract_pdbid_from_filename(row['wild_pt'])
            mutation = row.get('Mutation', '')
            filtered_pdbids.append(pdbid)
            filtered_mutations.append(mutation)
    
    # 创建数据加载器
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_pairs_fast,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False
    )
    
    # 记录预测结果
    all_predictions = []
    all_targets = []
    
    model.eval()
    
    print("Running prediction on test set...")
    with torch.no_grad():
        for batch_idx, (A_wt_b, R_wt_b, A_mut_b, R_mut_b, y_true_batch) in enumerate(tqdm(test_loader, desc="Predicting")):
            if A_wt_b['x_scalar'].size(0) == 0:
                continue
            
            # 将数据移动到设备
            A_wt_b = to_device(A_wt_b, device, non_blocking=True)
            R_wt_b = to_device(R_wt_b, device, non_blocking=True)
            A_mut_b = to_device(A_mut_b, device, non_blocking=True)
            R_mut_b = to_device(R_mut_b, device, non_blocking=True)
            y_true_batch = y_true_batch.to(device, non_blocking=True)
            
            # 前向传播
            ddg_pred, _ = model(A_wt_b, R_wt_b, A_mut_b, R_mut_b)
            
            # 确保形状一致
            if ddg_pred.dim() == 0:
                ddg_pred = ddg_pred.unsqueeze(0)
            if y_true_batch.dim() == 0:
                y_true_batch = y_true_batch.unsqueeze(0)
            
            all_predictions.append(ddg_pred.detach().cpu())
            all_targets.append(y_true_batch.detach().cpu())
    
    # 合并所有预测和真值
    if len(all_predictions) == 0:
        print("Warning: No valid predictions made!")
        return [], torch.empty(0), torch.empty(0)
    
    all_pred_tensor = torch.cat(all_predictions, dim=0)
    all_target_tensor = torch.cat(all_targets, dim=0)
    
    # 确保索引数量匹配
    if len(filtered_original_indices) != len(all_pred_tensor):
        print(f"Warning: Index count mismatch ({len(filtered_original_indices)} vs {len(all_pred_tensor)}). Using sequential indices.")
        # 如果数量不匹配,使用顺序索引
        filtered_original_indices = [filtered_original_indices[i] if i < len(filtered_original_indices) else test_indices[i] if i < len(test_indices) else i 
                                     for i in range(len(all_pred_tensor))]
        # 同样调整pdbid和mutation列表
        filtered_pdbids = [filtered_pdbids[i] if i < len(filtered_pdbids) else '' 
                           for i in range(len(all_pred_tensor))]
        filtered_mutations = [filtered_mutations[i] if i < len(filtered_mutations) else '' 
                              for i in range(len(all_pred_tensor))]
    
    return filtered_original_indices, all_pred_tensor, all_target_tensor, filtered_pdbids, filtered_mutations

def count_folds_in_jsonl(json_path: str) -> int:
    """
    统计JSONL文件中的fold数量（行数）
    
    Args:
        json_path: JSONL文件路径
    
    Returns:
        fold数量
    """
    p = Path(json_path)
    if not p.exists():
        raise FileNotFoundError(f"JSONL file not found: {json_path}")
    
    lines = p.read_text().strip().splitlines()
    # 过滤空行
    lines = [line for line in lines if line.strip()]
    return len(lines)

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Test all fold models on test set and output CSV with pdbid and mutation')
    parser.add_argument('--model_dir', type=str, 
                       default="/root/AbMSPN/Model/CROSS_VALIDATION/dual_S645",
                       help='Directory containing model files')
    parser.add_argument('--mode', type=str, default='dual',
                       choices=['atom_only', 'residue_only', 'dual'],
                       help='Model mode')
    parser.add_argument('--atom_encoder', type=str, default='gvp',
                       help='Atom encoder type')
    parser.add_argument('--res_encoder', type=str, default='gvp',
                       help='Residue encoder type')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for prediction')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for CSV files (default: current directory)')
    parser.add_argument('--merge_output', action='store_true',
                       help='Merge all fold predictions into a single CSV file')
    args = parser.parse_args()
    
    model_dir = args.model_dir
    if not Path(model_dir).exists():
        print(f"Error: Model directory not found: {model_dir}")
        print("Please check the path and try again.")
        return
    print("=" * 80)
    print("Test Set Prediction with All Fold Models")
    print("=" * 80)
    print(f"Model directory: {model_dir}")
    print(f"Mode: {args.mode}")
    print(f"Atom encoder: {args.atom_encoder}")
    print(f"Residue encoder: {args.res_encoder}")
    print("=" * 80)
    
    try:
        set_seed(Cfg.seed)
        json_path = Cfg.split_json_path
        if not json_path:
            raise ValueError("Cfg.split_json_path must be set for testing.")
        print(f"\nLoading JSONL file: {json_path}")
        n_folds = count_folds_in_jsonl(json_path)
        print(f"Found {n_folds} folds in JSONL file")
        output_dir = Path(args.output_dir) if args.output_dir else Path(".")
        output_dir.mkdir(parents=True, exist_ok=True)
        all_results = []
        all_fold_metrics = []
        for fold_id in range(1, n_folds + 1):
            print(f"\n{'=' * 80}")
            print(f"Processing Fold {fold_id}/{n_folds}")
            print(f"{'=' * 80}")
            
            try:
                print(f"Loading test indices for fold {fold_id} from: {json_path}")
                test_indices = load_test_indices(
                    json_path, 
                    fold_id=fold_id, 
                    one_based=Cfg.json_indices_one_based
                )
                print(f"Test set size for fold {fold_id}: {len(test_indices)} samples")
                model = load_model_from_folder(
                    model_dir=model_dir,
                    fold_id=fold_id,
                    mode=args.mode,
                    atom_encoder=args.atom_encoder,
                    res_encoder=args.res_encoder
                )

                indices, predictions, targets, pdbids, mutations = predict_on_test_set(
                    model=model,
                    test_indices=test_indices,
                    batch_size=args.batch_size
                )
                if len(predictions) == 0:
                    print(f"Warning: No valid predictions for fold {fold_id}")
                    continue
                results_df = pd.DataFrame({
                    'index': indices,
                    'fold': fold_id,
                    'pdbid': pdbids,
                    'mutation': mutations,
                    'predicted_value': predictions.numpy(),
                    'true_value': targets.numpy()
                })
                output_path = output_dir / f"test_predictions_fold_{fold_id}.csv"
                results_df.to_csv(output_path, index=False)
                print(f"\nFold {fold_id} completed!")
                print(f"Results saved to: {output_path}")
                print(f"Total samples: {len(results_df)}")
                mae = torch.mean(torch.abs(predictions - targets)).item()
                mse = torch.mean((predictions - targets) ** 2).item()
                rmse = torch.sqrt(torch.tensor(mse)).item()
                pearson = pearsonr_torch(predictions, targets).item()
                spearman = spearmanr_torch(predictions, targets).item()
                r2 = r2_torch(predictions, targets).item()
                print(f"Statistics for Fold {fold_id}:")
                print(f"  MAE:      {mae:.4f}")
                print(f"  RMSE:     {rmse:.4f}")
                print(f"  R²:       {r2:.4f}")
                print(f"  Pearson:  {pearson:.4f}")
                print(f"  Spearman: {spearman:.4f}")
                fold_metrics = {
                    'fold': fold_id,
                    'mae': mae,
                    'rmse': rmse,
                    'r2': r2,
                    'pearson': pearson,
                    'spearman': spearman,
                    'samples': len(results_df)
                }
                all_fold_metrics.append(fold_metrics)
                if args.merge_output:
                    all_results.append(results_df)
            except FileNotFoundError as e:
                print(f"Warning: Model file not found for fold {fold_id}: {e}")
                continue
            except Exception as e:
                print(f"Error processing fold {fold_id}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        if args.merge_output and all_results:
            merged_df = pd.concat(all_results, ignore_index=True)
            merged_output_path = output_dir / "test_predictions_all_folds.csv"
            merged_df.to_csv(merged_output_path, index=False)
            all_pred = torch.tensor(merged_df['predicted_value'].values)
            all_true = torch.tensor(merged_df['true_value'].values)
            overall_mae = torch.mean(torch.abs(all_pred - all_true)).item()
            overall_mse = torch.mean((all_pred - all_true) ** 2).item()
            overall_rmse = torch.sqrt(torch.tensor(overall_mse)).item()
            overall_pearson = pearsonr_torch(all_pred, all_true).item()
            overall_spearman = spearmanr_torch(all_pred, all_true).item()
            overall_r2 = r2_torch(all_pred, all_true).item()
            print(f"\n{'=' * 80}")
            print(f"Merged results saved to: {merged_output_path}")
            print(f"Total samples across all folds: {len(merged_df)}")
            print(f"\nOverall Statistics (All Folds Combined):")
            print(f"  MAE:      {overall_mae:.4f}")
            print(f"  RMSE:     {overall_rmse:.4f}")
            print(f"  R²:       {overall_r2:.4f}")
            print(f"  Pearson:  {overall_pearson:.4f}")
            print(f"  Spearman: {overall_spearman:.4f}")
            print(f"{'=' * 80}")
        
        if all_fold_metrics:
            metrics_df = pd.DataFrame(all_fold_metrics)
            metrics_output_path = output_dir / "test_metrics_summary.csv"
            metrics_df.to_csv(metrics_output_path, index=False)
            avg_mae = metrics_df['mae'].mean()
            avg_rmse = metrics_df['rmse'].mean()
            avg_r2 = metrics_df['r2'].mean()
            avg_pearson = metrics_df['pearson'].mean()
            avg_spearman = metrics_df['spearman'].mean()
            n_folds_count = len(metrics_df)
            if n_folds_count == 1:
                std_mae = 0.0
                std_rmse = 0.0
                std_r2 = 0.0
                std_pearson = 0.0
                std_spearman = 0.0
            else:
                std_mae = metrics_df['mae'].std()
                std_rmse = metrics_df['rmse'].std()
                std_r2 = metrics_df['r2'].std()
                std_pearson = metrics_df['pearson'].std()
                std_spearman = metrics_df['spearman'].std()
            print(f"\n{'=' * 80}")
            print(f"Metrics Summary saved to: {metrics_output_path}")
            print(f"\nAverage Statistics Across All Folds:")
            print(f"  MAE:      {avg_mae:.4f} ± {std_mae:.4f}")
            print(f"  RMSE:     {avg_rmse:.4f} ± {std_rmse:.4f}")
            print(f"  R²:       {avg_r2:.4f} ± {std_r2:.4f}")
            print(f"  Pearson:  {avg_pearson:.4f} ± {std_pearson:.4f}")
            print(f"  Spearman: {avg_spearman:.4f} ± {std_spearman:.4f}")
            print(f"{'=' * 80}")
        print(f"\n{'=' * 80}")
        print(f"All predictions completed!")
        print(f"Output directory: {output_dir}")
        print(f"Processed {n_folds} folds")
        print(f"{'=' * 80}")
    except Exception as e:
        print(f"Error during prediction: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
