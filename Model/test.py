import torch, os
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
import argparse
import torch
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
from tqdm import tqdm
import os
import json
from typing import Dict, Tuple
from model import (
    Cfg, set_seed, build_AR_from_data, collate_pairs_fast, to_device,
    DeltaGDataset, pearsonr_torch, spearmanr_torch, r2_torch,
    DDGPredictor, create_model, load_split_indices, 
)

def load_cv_model(fold_id: int, model_dir: str, mode: str = 'residue_only', 
                  atom_encoder: str = 'gvp', res_encoder: str = 'gvp',
                  use_val_model: bool = True) -> DDGPredictor:

    device = Cfg.device  
    model_type = 'val' if use_val_model else 'train'
    model_path = Path(model_dir) / f"best_e2e_cv_fold_{fold_id}_{model_type}.pt"
    params_path = Path(model_dir) / f"best_e2e_cv_fold_{fold_id}_{model_type}_params.pt"
    
    if model_path.exists():
        load_path = model_path
        print(f"Loading model from checkpoint: {model_path}")
    elif params_path.exists():
        load_path = params_path
        print(f"Loading model from params file: {params_path}")
    else:
        raise FileNotFoundError(
            f"Model file not found. Tried:\n"
            f"  - {model_path}\n"
            f"  - {params_path}"
        )
    dataset = DeltaGDataset(
        csv_path=Cfg.csv_path,
        root_dir=Cfg.root_dir,
        split='train',
        explicit_indices=[0]
    )
    wt_data, mut_data, _ = dataset[0]
    A_tmp, R_tmp = build_AR_from_data(wt_data)
    model = create_model(mode, A_tmp, R_tmp, device, atom_encoder, res_encoder)
    checkpoint = torch.load(load_path, map_location=device, weights_only=False)
    
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print("Loaded model from checkpoint")
        elif 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
            print("Loaded model from checkpoint")
        else:
            try:
                model.load_state_dict(checkpoint)
                print("Loaded model state dict directly")
            except Exception as e:
                print(f"Error loading model: {e}")
                print(f"Available keys in checkpoint: {list(checkpoint.keys())}")
                raise
    else:
        model.load_state_dict(checkpoint)
        print("Loaded model state dict directly")
    model.eval()
    return model

def test_single_fold(fold_id: int, model_dir: str, mode: str = 'residue_only',
                    atom_encoder: str = 'gvp', res_encoder: str = 'gvp',
                    batch_size: int = 8, use_val_model: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    set_seed(Cfg.seed)
    device = Cfg.device
    print(f"\n=== Testing Fold {fold_id} ===")
    model = load_cv_model(fold_id, model_dir, mode, atom_encoder, res_encoder, use_val_model=use_val_model)
    json_path = Cfg.split_json_path
    if not json_path:
        raise ValueError("Cfg.split_json_path must be set for testing.")
    tr_idx, va_idx = load_split_indices(json_path, fold_id, one_based=Cfg.json_indices_one_based)
    print(f"Validation set size: {len(va_idx)}")
    val_dataset = DeltaGDataset(
        csv_path=Cfg.csv_path,
        root_dir=Cfg.root_dir,
        split='val',
        explicit_indices=va_idx,
        filter_missing=True
    )
    print(f"Filtered validation set size: {len(val_dataset)}")
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_pairs_fast,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False
    )
    model.eval()
    all_predictions = []
    all_targets = []
    print("Running validation...")
    with torch.no_grad():
        for batch_idx, (A_wt_b, R_wt_b, A_mut_b, R_mut_b, y_true_batch) in enumerate(tqdm(val_loader, desc="Testing")):
            if A_wt_b['x_scalar'].size(0) == 0:
                continue
            A_wt_b = to_device(A_wt_b, device, non_blocking=True)
            R_wt_b = to_device(R_wt_b, device, non_blocking=True)
            A_mut_b = to_device(A_mut_b, device, non_blocking=True)
            R_mut_b = to_device(R_mut_b, device, non_blocking=True)
            y_true_batch = y_true_batch.to(device, non_blocking=True)
            ddg_pred, _ = model(A_wt_b, R_wt_b, A_mut_b, R_mut_b)
            if ddg_pred.dim() == 0:
                ddg_pred = ddg_pred.unsqueeze(0)
            if y_true_batch.dim() == 0:
                y_true_batch = y_true_batch.unsqueeze(0)
            all_predictions.append(ddg_pred.detach().cpu())
            all_targets.append(y_true_batch.detach().cpu())

    if len(all_predictions) == 0:
        print("Warning: No valid predictions made!")
        return torch.empty(0), torch.empty(0)
    all_pred_tensor = torch.cat(all_predictions, dim=0)
    all_target_tensor = torch.cat(all_targets, dim=0)
    print(f"Fold {fold_id} completed: {len(all_pred_tensor)} samples")
    return all_pred_tensor, all_target_tensor

def test_single_fold_with_metrics(fold_id: int, model_dir: str, mode: str = 'residue_only',
                                 atom_encoder: str = 'gvp', res_encoder: str = 'gvp',
                                 batch_size: int = 8, use_val_model: bool = False) -> Dict[str, float]:
    all_pred_tensor, all_target_tensor = test_single_fold(
        fold_id, model_dir, mode, atom_encoder, res_encoder, batch_size, use_val_model
    )
    
    if len(all_pred_tensor) == 0:
        return {
            'mae': float('inf'),
            'rmse': float('inf'),
            'pearson': 0.0,
            'spearman': 0.0,
            'r2': 0.0
        }

    mae = torch.mean(torch.abs(all_pred_tensor - all_target_tensor)).item()
    mse = torch.mean((all_pred_tensor - all_target_tensor) ** 2).item()
    rmse = torch.sqrt(torch.tensor(mse)).item()
    pearson = pearsonr_torch(all_pred_tensor, all_target_tensor).item()
    spearman = spearmanr_torch(all_pred_tensor, all_target_tensor).item()
    r2 = r2_torch(all_pred_tensor, all_target_tensor).item()
    results = {
        'mae': mae,
        'rmse': rmse,
        'pearson': pearson,
        'spearman': spearman,
        'r2': r2
    }
    print(f"Fold {fold_id} Results:")
    print(f"  MAE:  {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  Pearson:  {pearson:.4f}")
    print(f"  Spearman: {spearman:.4f}")
    print(f"  R²:       {r2:.4f}")
    return results

def test_all_folds(model_dir: str, mode: str = 'dual',
                  atom_encoder: str = 'rgcn', res_encoder: str = 'gvp',
                  n_folds: int = 10, batch_size: int = 8, use_val_model: bool = True) -> Dict[str, Dict[str, float]]:
    print(f"Testing all {n_folds} folds...")
    print(f"Model directory: {model_dir}")
    print(f"Mode: {mode}")
    print(f"Atom encoder: {atom_encoder}")
    print(f"Residue encoder: {res_encoder}")
    print(f"Using {'validation' if use_val_model else 'training'} best model")
    all_predictions = []
    all_targets = []
    fold_results = {}
    
    for fold_id in range(1, n_folds + 1):
        try:
            print(f"\n=== Testing Fold {fold_id} ===")
            pred_tensor, target_tensor = test_single_fold(
                fold_id, model_dir, mode, atom_encoder, res_encoder, batch_size, use_val_model
            )
            if len(pred_tensor) > 0:
                all_predictions.append(pred_tensor)
                all_targets.append(target_tensor)
                mse = torch.mean((pred_tensor - target_tensor) ** 2).item()
                rmse = torch.sqrt(torch.tensor(mse)).item()
                pearson = pearsonr_torch(pred_tensor, target_tensor).item()
                spearman = spearmanr_torch(pred_tensor, target_tensor).item()
                r2 = r2_torch(pred_tensor, target_tensor).item()
                fold_results[str(fold_id)] = {
                    'test': {
                        'loss': float(mse),
                        'criterion_loss': float(mse),
                        'mse_loss': float(mse),
                        'rmse': float(rmse),
                        'pearson': float(pearson),
                        'spearman': float(spearman),
                        'r2': float(r2),
                    }
                }
                print(f"Fold {fold_id} Results:")
                print(f"  RMSE: {rmse:.4f}")
                print(f"  Pearson:  {pearson:.4f}")
                print(f"  Spearman: {spearman:.4f}")
                print(f"  R²:       {r2:.4f}")
                print(f"  Loss (MSE): {mse:.4f}")
            else:
                print(f"Warning: No valid predictions for fold {fold_id}")
                fold_results[str(fold_id)] = {
                    'test': {
                        'loss': 0.0,
                        'criterion_loss': 0.0,
                        'mse_loss': 0.0,
                        'rmse': 0.0,
                        'pearson': 0.0,
                        'spearman': 0.0,
                        'r2': 0.0,
                    }
                }
        except Exception as e:
            print(f"Error testing fold {fold_id}: {e}")
            fold_results[str(fold_id)] = {
                'test': {
                    'loss': 0.0,
                    'criterion_loss': 0.0,
                    'mse_loss': 0.0,
                    'rmse': 0.0,
                    'pearson': 0.0,
                    'spearman': 0.0,
                    'r2': 0.0,
                }
            }
    if len(all_predictions) == 0:
        print("Warning: No valid predictions from any fold!")
        avg_results = {
            'test': {
                'loss': 0.0,
                'loss_std': 0.0,
                'rmse': 0.0,
                'rmse_std': 0.0,
                'pearson': 0.0,
                'pearson_std': 0.0,
                'spearman': 0.0,
                'spearman_std': 0.0,
                'r2': 0.0,
                'r2_std': 0.0,
            }
        }
        return {
            'per_fold': fold_results,
            'avg_results': avg_results
        }
    combined_predictions = torch.cat(all_predictions, dim=0)
    combined_targets = torch.cat(all_targets, dim=0)
    print(f"\n=== Overall Results (Combined from All Folds) ===")
    print(f"Total samples: {len(combined_predictions)}")
    overall_mse = torch.mean((combined_predictions - combined_targets) ** 2).item()
    overall_rmse = torch.sqrt(torch.tensor(overall_mse)).item()
    overall_pearson = pearsonr_torch(combined_predictions, combined_targets).item()
    overall_spearman = spearmanr_torch(combined_predictions, combined_targets).item()
    overall_r2 = r2_torch(combined_predictions, combined_targets).item()
    print(f"Overall Results:")
    print(f"  RMSE: {overall_rmse:.4f}")
    print(f"  Pearson:  {overall_pearson:.4f}")
    print(f"  Spearman: {overall_spearman:.4f}")
    print(f"  R²:       {overall_r2:.4f}")
    print(f"  Loss (MSE): {overall_mse:.4f}")
    valid_folds = [fold_result['test'] for fold_id, fold_result in fold_results.items()
                   if fold_result['test']['rmse'] > 0]
    if valid_folds:
        avg_results = {
            'test': {}
        }
        for metric in ['loss', 'rmse', 'pearson', 'spearman', 'r2']:
            values = [fold[metric] for fold in valid_folds]
            if values:
                if metric in ['pearson', 'spearman', 'r2']:
                    filtered_values = [v for v in values if not (np.isnan(v) or np.isinf(v))]
                else:
                    filtered_values = [v for v in values if v > 0 and not (np.isnan(v) or np.isinf(v))]
                if filtered_values:
                    avg_results['test'][metric] = float(np.mean(filtered_values))
                    avg_results['test'][f'{metric}_std'] = float(np.std(filtered_values))
                else:
                    avg_results['test'][metric] = 0.0
                    avg_results['test'][f'{metric}_std'] = 0.0
            else:
                avg_results['test'][metric] = 0.0
                avg_results['test'][f'{metric}_std'] = 0.0
        print(f"\n=== Per-Fold Statistics ===")
        print(f"Valid folds: {len(valid_folds)}")
        print(f"Loss:  {avg_results['test']['loss']:.4f} ± {avg_results['test']['loss_std']:.4f}")
        print(f"RMSE: {avg_results['test']['rmse']:.4f} ± {avg_results['test']['rmse_std']:.4f}")
        print(f"Pearson:  {avg_results['test']['pearson']:.4f} ± {avg_results['test']['pearson_std']:.4f}")
        print(f"Spearman: {avg_results['test']['spearman']:.4f} ± {avg_results['test']['spearman_std']:.4f}")
        print(f"R²:       {avg_results['test']['r2']:.4f} ± {avg_results['test']['r2_std']:.4f}")
        return {
            'per_fold': fold_results,
            'avg_results': avg_results
        }
    else:
        avg_results = {
            'test': {
                'loss': 0.0,
                'loss_std': 0.0,
                'rmse': 0.0,
                'rmse_std': 0.0,
                'pearson': 0.0,
                'pearson_std': 0.0,
                'spearman': 0.0,
                'spearman_std': 0.0,
                'r2': 0.0,
                'r2_std': 0.0,
            }
        }
        return {
            'per_fold': fold_results,
            'avg_results': avg_results
        }

def save_test_results(results: Dict, output_path: str, mode: str = 'dual',
                     atom_encoder: str = 'gvp', res_encoder: str = 'gvp'):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def convert_numpy(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(v) for v in obj]
        else:
            return obj

    final_results = {
        "method": "end_to_end_unified",
        "atom_encoder": atom_encoder,
        "res_encoder": res_encoder,
        "mode": mode,
        "avg_results": results.get('avg_results', {}),
        "per_fold": results.get('per_fold', {}),
        "config": {
            "epochs": getattr(Cfg, 'epochs', 100),
            "lr": getattr(Cfg, 'learning_rate', 0.0001),
            "batch_size": getattr(Cfg, 'batch_size', 16),
            "weight_decay": getattr(Cfg, 'weight_decay', 0.001),
            "sA": getattr(Cfg, 'sA', 128),
            "n_blocks": getattr(Cfg, 'n_blocks', 4),
            "atom_encoder": atom_encoder,
            "res_encoder": res_encoder,
            "mode": mode,
        },
    }
    results_converted = convert_numpy(final_results)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results_converted, f, indent=2, ensure_ascii=False)
    print(f"Test results saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Test cross-validation models')
    parser.add_argument('--model_dir', type=str, required=True,
                       help='Directory containing the trained models')
    parser.add_argument('--mode', type=str, default='dual',
                       choices=['atom_only', 'residue_only', 'dual'],
                       help='Model mode')
    parser.add_argument('--atom_encoder', type=str, default='gvp',
                       help='Atom encoder type')
    parser.add_argument('--res_encoder', type=str, default='gvp',
                       help='Residue encoder type')
    parser.add_argument('--n_folds', type=int, default=10,
                       help='Number of folds')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for testing')
    parser.add_argument('--fold_id', type=int, default=None,
                       help='Test specific fold (if None, test all folds)')
    parser.add_argument('--output_path', type=str, default=None,
                       help='Output path for test results JSON file')
    parser.add_argument('--use_val_model', action='store_true', default=True,
                       help='Use validation best model (default: True)')
    parser.add_argument('--use_train_model', action='store_false', dest='use_val_model',
                       help='Use training best model instead of validation best model')
    
    args = parser.parse_args()
    device = Cfg.device
    print(f"Using device: {device}")
    if args.fold_id is not None:
        results = test_single_fold_with_metrics(
            args.fold_id, args.model_dir, args.mode,
            args.atom_encoder, args.res_encoder, args.batch_size, args.use_val_model
        )
        mse = results.get('rmse', 0.0) ** 2 if 'rmse' in results else 0.0
        per_fold = {
            str(args.fold_id): {
                'test': {
                    'loss': float(mse),
                    'criterion_loss': float(mse),
                    'mse_loss': float(mse),
                    'rmse': results.get('rmse', 0.0),
                    'pearson': results.get('pearson', 0.0),
                    'spearman': results.get('spearman', 0.0),
                    'r2': results.get('r2', 0.0),
                }
            }
        }
        avg_results = {
            'test': {
                'loss': float(mse),
                'loss_std': 0.0,
                'rmse': results.get('rmse', 0.0),
                'rmse_std': 0.0,
                'pearson': results.get('pearson', 0.0),
                'pearson_std': 0.0,
                'spearman': results.get('spearman', 0.0),
                'spearman_std': 0.0,
                'r2': results.get('r2', 0.0),
                'r2_std': 0.0,
            }
        }
        all_results = {
            'per_fold': per_fold,
            'avg_results': avg_results
        }
    else:
        all_results = test_all_folds(
            args.model_dir, args.mode, args.atom_encoder,
            args.res_encoder, args.n_folds, args.batch_size, args.use_val_model
        )
    if args.output_path:
        save_test_results(all_results, args.output_path, args.mode, args.atom_encoder, args.res_encoder)
    else:
        model_dir_name = Path(args.model_dir).name
        output_path = f"test_results_{model_dir_name}_{args.mode}.json"
        save_test_results(all_results, output_path, args.mode, args.atom_encoder, args.res_encoder)

if __name__ == "__main__":
    main()
