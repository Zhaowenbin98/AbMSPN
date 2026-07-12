import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
from tqdm import tqdm
import os
from model import (
    Cfg, set_seed, build_AR_from_data, collate_pairs_fast, to_device,
    DeltaGDataset, pearsonr_torch, spearmanr_torch, r2_torch,
    DDGPredictor, create_model, load_split_indices
)

def train_end_to_end(fold_id=6, mode='residue_only'):
    set_seed(Cfg.seed)
    device = Cfg.device
    json_path = Cfg.split_json_path
    if not json_path:
        raise ValueError("Cfg.split_json_path must be set for training.")
    tr_idx_json, val_idx_json = load_split_indices(json_path, fold_id, one_based=Cfg.json_indices_one_based)
    print(f"[E2E] JSON split: train={len(tr_idx_json)} val={len(val_idx_json)}", flush=True)
    train_ds = DeltaGDataset(Cfg.csv_path, Cfg.root_dir, split='train', explicit_indices=tr_idx_json)
    val_ds = DeltaGDataset(Cfg.csv_path, Cfg.root_dir, split='test', explicit_indices=val_idx_json)
    test_ds = DeltaGDataset(Cfg.csv_path, Cfg.root_dir, split='test', explicit_indices=val_idx_json)
    g = torch.Generator()
    g.manual_seed(Cfg.seed)
    num_w = 0
    train_loader = DataLoader(train_ds, batch_size=Cfg.batch_size, shuffle=True,
                              collate_fn=collate_pairs_fast, generator=g, num_workers=num_w,
                              pin_memory=False, persistent_workers=True if num_w > 0 else False, 
                              prefetch_factor=2 if num_w > 0 else None)
    val_loader = DataLoader(val_ds, batch_size=Cfg.batch_size, shuffle=False,
                            collate_fn=collate_pairs_fast, generator=g, num_workers=num_w,
                            pin_memory=False, persistent_workers=True if num_w > 0 else False,
                            prefetch_factor=2 if num_w > 0 else None)
    test_loader = DataLoader(test_ds, batch_size=Cfg.batch_size, shuffle=False,
                             collate_fn=collate_pairs_fast, generator=g, num_workers=num_w,
                             pin_memory=False, persistent_workers=True if num_w > 0 else False,
                             prefetch_factor=2 if num_w > 0 else None)
    sample_graph_wt, sample_graph_mut, _ = train_ds[0]
    A_tmp, R_tmp = build_AR_from_data(sample_graph_wt)
    edgeR_dim = int(R_tmp['edge_attr_s'].size(-1)) if 'edge_attr_s' in R_tmp else 0
    sR_in = int(R_tmp['x_scalar'].size(-1)) if 'x_scalar' in R_tmp else 0
    model = create_model(mode, A_tmp, R_tmp, device, 
                        atom_encoder=Cfg.atom_encoder, res_encoder=Cfg.res_encoder)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params | model: {trainable_params}/{total_params} trainable/total", flush=True)
    def is_no_decay(n: str, p: nn.Parameter) -> bool:
        if p.ndim == 1:
            return True
        no_decay_keywords = ["bias", "LayerNorm", "layernorm", "ln", "norm"]
        return any(k.lower() in n.lower() for k in no_decay_keywords)
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if is_no_decay(name, param):
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": Cfg.weight_decay},
            {"params": no_decay_params, "weight_decay":0},
        ],
        lr=Cfg.learning_rate,
    )

    scheduler = None
    criterion = nn.MSELoss()
    best_val_rmse = float('inf')
    patience_counter = 0
    es_patience = int(getattr(Cfg, 'early_stop_patience', 10))
    
    def run_epoch(loader, train: bool):
        if train: 
            model.train()
        else: 
            model.eval()
        all_predictions = []
        all_targets = []
        all_criterion_losses = []
        step_pbar = tqdm(enumerate(loader, start=1), total=len(loader), 
                        desc=f"{'Training' if train else 'Validation'}", 
                        leave=False, unit="batch")
        for step, (A_wt_b, R_wt_b, A_mut_b, R_mut_b, y_true_batch) in step_pbar:
            if A_wt_b['x_scalar'].size(0) == 0:
                continue
            with torch.set_grad_enabled(train):
                A_wt_b = to_device(A_wt_b, device, non_blocking=True)
                R_wt_b = to_device(R_wt_b, device, non_blocking=True)
                A_mut_b = to_device(A_mut_b, device, non_blocking=True)
                R_mut_b = to_device(R_mut_b, device, non_blocking=True)
                y_true_batch = y_true_batch.to(device, non_blocking=True)
                ddg_pred, attention_info = model(A_wt_b, R_wt_b, A_mut_b, R_mut_b)
                if ddg_pred.dim() == 0:
                    ddg_pred = ddg_pred.unsqueeze(0)
                if y_true_batch.dim() == 0:
                    y_true_batch = y_true_batch.unsqueeze(0)
                loss = criterion(ddg_pred, y_true_batch)
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"Warning: NaN/Inf loss detected: {loss.item()}")
                    print(f"ddg_pred: {ddg_pred}")
                    print(f"y_true: {y_true_batch}")
                    continue
                if train:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    # 添加梯度裁剪
                    #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  
                    optimizer.step()
                # 计算指标
                with torch.no_grad():
                    if ddg_pred.dim() == 0:
                        ddg_pred = ddg_pred.unsqueeze(0)
                    if y_true_batch.dim() == 0:
                        y_true_batch = y_true_batch.unsqueeze(0) 
                    all_predictions.append(ddg_pred.detach().cpu())
                    all_targets.append(y_true_batch.detach().cpu())
                    all_criterion_losses.append(loss.detach().cpu())  # 收集MSE损失

        if len(all_predictions) > 0:
            all_pred_tensor = torch.cat(all_predictions, dim=0)
            all_target_tensor = torch.cat(all_targets, dim=0)
            epoch_criterion_loss = torch.stack(all_criterion_losses).mean()
            epoch_mse = ((all_pred_tensor - all_target_tensor) ** 2).mean()
            epoch_rmse = torch.sqrt(epoch_mse)
            epoch_pearson = pearsonr_torch(all_pred_tensor, all_target_tensor)
            epoch_spearman = spearmanr_torch(all_pred_tensor, all_target_tensor)
            epoch_r2 = r2_torch(all_pred_tensor, all_target_tensor)
            
            return {
                'loss': float(epoch_criterion_loss),  # 使用MSE损失进行早停
                'criterion_loss': float(epoch_criterion_loss),  # MSE损失
                'mse_loss': float(epoch_mse),  # MSE用于RMSE计算
                'rmse': float(epoch_rmse),
                'pearson': float(epoch_pearson),
                'spearman': float(epoch_spearman),
                'r2': float(epoch_r2),
            }
        else:
            return {
                'loss': 0.0,
                'criterion_loss': 0.0,
                'mse_loss': 0.0,
                'rmse': 0.0,
                'pearson': 0.0,
                'spearman': 0.0,
                'r2': 0.0,
            }

    epoch_pbar = tqdm(range(1, Cfg.epochs + 1), desc="Epochs", unit="epoch")
    for epoch in epoch_pbar:
        epoch_pbar.set_description(f"Epoch {epoch}/{Cfg.epochs}")
        print(f"\n[E2E] Epoch {epoch}/{Cfg.epochs}", flush=True)
        print(f"  LR={optimizer.param_groups[0]['lr']:.2e}", flush=True)
        tr_stats = run_epoch(train_loader, train=True)
        val_stats = run_epoch(val_loader, train=False)
        print(f"  Train     | Loss: {tr_stats['criterion_loss']:.4f} RMSE: {tr_stats['rmse']:.3f} Pearson: {tr_stats['pearson']:.3f} Spearman: {tr_stats['spearman']:.3f} R²: {tr_stats['r2']:.3f}", flush=True)
        print(f"  Validation| Loss: {val_stats['criterion_loss']:.4f} RMSE: {val_stats['rmse']:.3f} Pearson: {val_stats['pearson']:.3f} Spearman: {val_stats['spearman']:.3f} R²: {val_stats['r2']:.3f}", flush=True)
        if val_stats['rmse'] < best_val_rmse - 1e-5:
            best_val_rmse = val_stats['rmse']
            patience_counter = 0
            save_dict = {
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'train_loss': tr_stats['loss'],
                'train_rmse': tr_stats['rmse'],
                'train_pearson': tr_stats['pearson'],
                'train_spearman': tr_stats['spearman'],
                'train_r2': tr_stats['r2'],
                'val_loss': val_stats['loss'],
                'val_rmse': val_stats['rmse'],
                'val_pearson': val_stats['pearson'],
                'val_spearman': val_stats['spearman'],
                'val_r2': val_stats['r2'],
            }
            if scheduler is not None:
                save_dict['scheduler'] = scheduler.state_dict()
            torch.save(save_dict, f"best_e2e_model_fold_{fold_id}_val.pt")
            torch.save(model.state_dict(), f"best_e2e_model_fold_{fold_id}_val_params.pt")
            print(f"Saved best val model -> best_e2e_model_fold_{fold_id}_val.pt", flush=True)
            print(f"Val RMSE: {val_stats['rmse']:.4f} -> {best_val_rmse:.4f}", flush=True)
            print(f"Val Pearson: {val_stats['pearson']:.3f} Loss: {val_stats['loss']:.4f}", flush=True)
        
        if val_stats['rmse'] >= best_val_rmse - 1e-5:
            patience_counter += 1
            if patience_counter >= es_patience:
                print(f"Early stopping at epoch {epoch} (no validation RMSE improvement for {patience_counter} epochs)", flush=True)
                break

    best_val_checkpoint = None
    test_stats = None
    best_val_model_path = f"best_e2e_model_fold_{fold_id}_val_params.pt"
    best_val_checkpoint_path = f"best_e2e_model_fold_{fold_id}_val.pt"
    if os.path.exists(best_val_model_path):
        model.load_state_dict(torch.load(best_val_model_path, map_location=device))
        if os.path.exists(best_val_checkpoint_path):
            best_val_checkpoint = torch.load(best_val_checkpoint_path, map_location=device)
            print(f"[E2E] Loaded best validation model from epoch {best_val_checkpoint['epoch']}", flush=True)
            print(f"[E2E] Best validation metrics:", flush=True)
            print(f"  Loss: {best_val_checkpoint['val_loss']:.4f}", flush=True)
            print(f"  RMSE: {best_val_checkpoint['val_rmse']:.3f}", flush=True)
            print(f"  Pearson: {best_val_checkpoint['val_pearson']:.3f}", flush=True)
            print(f"  Spearman: {best_val_checkpoint['val_spearman']:.3f}", flush=True)
            print(f"  R²: {best_val_checkpoint['val_r2']:.3f}", flush=True)
        else:
            print(f"[E2E] Loaded best validation model from {best_val_model_path}", flush=True)
        print(f"\n[E2E] Evaluating test set with best validation model...", flush=True)
        test_stats = run_epoch(test_loader, train=False)
        print(f"[E2E] Test set metrics:", flush=True)
        print(f"  Loss: {test_stats['criterion_loss']:.4f} RMSE: {test_stats['rmse']:.3f} Pearson: {test_stats['pearson']:.3f} Spearman: {test_stats['spearman']:.3f} R²: {test_stats['r2']:.3f}", flush=True)
    else:
        print(f"[E2E] Warning: Best validation model not found at {best_val_model_path}", flush=True)
        print(f"[E2E] Evaluating test set with current model...", flush=True)
        test_stats = run_epoch(test_loader, train=False)
        print(f"[E2E] Test set metrics:", flush=True)
        print(f"  Loss: {test_stats['criterion_loss']:.4f} RMSE: {test_stats['rmse']:.3f} Pearson: {test_stats['pearson']:.3f} Spearman: {test_stats['spearman']:.3f} R²: {test_stats['r2']:.3f}", flush=True)

    validation_results = {
        "test_metrics": {
            "loss": test_stats['criterion_loss'],
            "rmse": test_stats['rmse'],
            "pearson": test_stats['pearson'],
            "spearman": test_stats['spearman'],
            "r2": test_stats['r2'],
        },
        "model_paths": {
            "val_model": f"best_e2e_model_fold_{fold_id}_val.pt",
            "val_params": f"best_e2e_model_fold_{fold_id}_val_params.pt"
        }
    }
    save_validation_results(fold_id, validation_results)
    return {
        "test_metrics": {
            "loss": test_stats['criterion_loss'],
            "rmse": test_stats['rmse'],
            "pearson": test_stats['pearson'],
            "spearman": test_stats['spearman'],
            "r2": test_stats['r2'],
        }
    }

def save_validation_results(fold_id, results_dict, output_dir="validation_results"):
    import json
    from pathlib import Path
    from datetime import datetime

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    save_data = {
        "fold_id": fold_id,
        "timestamp": datetime.now().isoformat(),
        "validation_results": results_dict,
        "config": {
            "epochs": Cfg.epochs,
            "lr": Cfg.learning_rate,
            "batch_size": Cfg.batch_size,
            "weight_decay": Cfg.weight_decay,
            "sA": Cfg.sA,
            "n_blocks": Cfg.n_blocks,
            "atom_encoder": Cfg.atom_encoder,
            "res_encoder": Cfg.res_encoder,
        }
    }
    
    json_path = output_path / f"validation_results_fold_{fold_id}.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    
    print(f"[Validation] Results saved to {json_path}", flush=True)
    return json_path

def load_validation_results(fold_id, input_dir="validation_results"):
    import json
    from pathlib import Path
    
    json_path = Path(input_dir) / f"validation_results_fold_{fold_id}.json"
    if json_path.exists():
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None

def cross_validate_end_to_end_unified(mode='residue_only', output_dir=None, atom_encoder=None, res_encoder=None, 
                                     csv_path=None, jsonl_path=None):
    set_seed(Cfg.seed)
    device = Cfg.device
    csv_path = csv_path if csv_path is not None else Cfg.csv_path
    json_path = jsonl_path if jsonl_path is not None else Cfg.split_json_path
    if not json_path:
        raise ValueError("split_json_path must be set for cross-validation.")
    if not csv_path:
        raise ValueError("csv_path must be set for cross-validation.")
    if output_dir is None:
        output_dir = f"CROSS_VALIDATION/{mode}"
    os.makedirs(output_dir, exist_ok=True)
    try:
        K = len(Path(json_path).read_text().strip().splitlines())
    except Exception:
        K = 10
    print(f"[E2E-CV][{mode}] Using splits from {json_path} | K={K}", flush=True)
    fold_results = {}

    for fold in tqdm(range(K), desc=f"Cross-Validation Folds", unit="fold"):
        fold_id = fold + 1
        print(f"\n[E2E-CV][{mode}] Fold {fold_id}/{K}", flush=True)
        best_model_path = os.path.join(output_dir, f"best_e2e_cv_fold_{fold_id}_val_params.pt")
        best_checkpoint_path = os.path.join(output_dir, f"best_e2e_cv_fold_{fold_id}_val.pt")
        if os.path.exists(best_model_path) and os.path.exists(best_checkpoint_path):
            print(f"[E2E-CV][{mode}] Fold {fold_id} already exists, loading existing results...", flush=True)
            try:
                tr_idx_json, val_idx_json = load_split_indices(json_path, fold_id, one_based=Cfg.json_indices_one_based)
                test_ds = DeltaGDataset(csv_path, Cfg.root_dir, split='test', explicit_indices=val_idx_json)
                g = torch.Generator()
                g.manual_seed(Cfg.seed)
                num_w = 0
                test_loader = DataLoader(test_ds, batch_size=Cfg.batch_size, shuffle=False,
                                      collate_fn=collate_pairs_fast, generator=g, num_workers=num_w,
                                      pin_memory=False, persistent_workers=True if num_w > 0 else False,
                                      prefetch_factor=2 if num_w > 0 else None)
                sample_graph_wt, sample_graph_mut, _ = test_ds[0]
                A_tmp, R_tmp = build_AR_from_data(sample_graph_wt)
                if mode == 'atom_only':
                    edgeA_dim = int(A_tmp['edge_attr'].size(-1))
                    sA_in = int(A_tmp['x_scalar'].size(-1))
                    atom_enc = atom_encoder if atom_encoder is not None else Cfg.atom_encoder
                    model = DDGPredictor(
                        sA=Cfg.sA, sR=Cfg.sA, n_blocks=Cfg.n_blocks,
                        edgeA_dim=edgeA_dim, edgeR_dim=0,
                        sA_in_dim=sA_in, sR_in_dim=0, hidden_dim=256, mode='atom_only',
                        atom_encoder=atom_enc, res_encoder='gat'
                    ).to(device)
                elif mode == 'residue_only':
                    edgeR_dim = int(R_tmp['edge_attr_s'].size(-1))
                    sR_in = int(R_tmp['x_scalar'].size(-1))
                    res_enc = res_encoder if res_encoder is not None else Cfg.res_encoder
                    model = DDGPredictor(
                        sA=Cfg.sA, sR=Cfg.sA, n_blocks=Cfg.n_blocks,
                        edgeA_dim=0, edgeR_dim=edgeR_dim,
                        sA_in_dim=0, sR_in_dim=sR_in, hidden_dim=256, mode='residue_only',
                        atom_encoder='egnn', res_encoder=res_enc
                    ).to(device)
                elif mode == 'dual':
                    edgeA_dim = int(A_tmp['edge_attr'].size(-1))
                    edgeR_dim = int(R_tmp['edge_attr_s'].size(-1))
                    sA_in = int(A_tmp['x_scalar'].size(-1))
                    sR_in = int(R_tmp['x_scalar'].size(-1))
                    atom_enc = atom_encoder if atom_encoder is not None else Cfg.atom_encoder
                    res_enc = res_encoder if res_encoder is not None else Cfg.res_encoder
                    model = DDGPredictor(
                        sA=Cfg.sA, sR=Cfg.sA, n_blocks=Cfg.n_blocks,
                        edgeA_dim=edgeA_dim, edgeR_dim=edgeR_dim,
                        sA_in_dim=sA_in, sR_in_dim=sR_in, hidden_dim=256, mode='dual',
                        atom_encoder=atom_enc, res_encoder=res_enc
                    ).to(device)
                model.load_state_dict(torch.load(best_model_path, map_location=device))

                def evaluate_existing_model(loader):
                    model.eval()
                    all_predictions = []
                    all_targets = []
                    all_criterion_losses = []
                    criterion = nn.MSELoss()
                    
                    with torch.no_grad():
                        for A_wt_b, R_wt_b, A_mut_b, R_mut_b, y_true_batch in loader:
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
                            loss = criterion(ddg_pred, y_true_batch)
                            if not (torch.isnan(loss) or torch.isinf(loss)):
                                all_predictions.append(ddg_pred.detach().cpu())
                                all_targets.append(y_true_batch.detach().cpu())
                                all_criterion_losses.append(loss.detach().cpu())
                    
                    if len(all_predictions) > 0:
                        all_pred_tensor = torch.cat(all_predictions, dim=0)
                        all_target_tensor = torch.cat(all_targets, dim=0)
                        epoch_criterion_loss = torch.stack(all_criterion_losses).mean()
                        epoch_mse = ((all_pred_tensor - all_target_tensor) ** 2).mean()
                        epoch_rmse = torch.sqrt(epoch_mse)
                        epoch_pearson = pearsonr_torch(all_pred_tensor, all_target_tensor)
                        epoch_spearman = spearmanr_torch(all_pred_tensor, all_target_tensor)
                        epoch_r2 = r2_torch(all_pred_tensor, all_target_tensor)
                        return {
                            'loss': float(epoch_criterion_loss),
                            'criterion_loss': float(epoch_criterion_loss),
                            'mse_loss': float(epoch_mse),
                            'rmse': float(epoch_rmse),
                            'pearson': float(epoch_pearson),
                            'spearman': float(epoch_spearman),
                            'r2': float(epoch_r2),
                        }
                    else:
                        return {
                            'loss': 0.0, 'criterion_loss': 0.0, 'mse_loss': 0.0,
                            'rmse': 0.0, 'pearson': 0.0, 'spearman': 0.0, 'r2': 0.0,
                        }
                test_stats = evaluate_existing_model(test_loader)
                print(f"[E2E-CV][{mode}] Loaded existing model test metrics (fold {fold_id}):", flush=True)
                print(f"  Loss: {test_stats['criterion_loss']:.4f} RMSE: {test_stats['rmse']:.3f} Pearson: {test_stats['pearson']:.3f} Spearman: {test_stats['spearman']:.3f} R²: {test_stats['r2']:.3f}", flush=True)
                fold_results[fold_id] = {'test': test_stats}
                continue
            except Exception as e:
                print(f"[E2E-CV][{mode}] Error loading existing model for fold {fold_id}: {e}", flush=True)
                print(f"[E2E-CV][{mode}] Will retrain fold {fold_id}...", flush=True)
        tr_idx_json, val_idx_json = load_split_indices(json_path, fold_id, one_based=Cfg.json_indices_one_based)
        print(f"[E2E-CV][{mode}] Training fold {fold_id} - JSON split: train={len(tr_idx_json)} val={len(val_idx_json)}", flush=True)

        train_ds = DeltaGDataset(csv_path, Cfg.root_dir, split='train', explicit_indices=tr_idx_json)
        val_ds = DeltaGDataset(csv_path, Cfg.root_dir, split='test', explicit_indices=val_idx_json)
        test_ds = DeltaGDataset(csv_path, Cfg.root_dir, split='test', explicit_indices=val_idx_json)
        g = torch.Generator()
        g.manual_seed(Cfg.seed)
        num_w = 0 
        train_loader = DataLoader(train_ds, batch_size=Cfg.batch_size, shuffle=True,
                                  collate_fn=collate_pairs_fast, generator=g, num_workers=num_w,
                                  pin_memory=False, persistent_workers=True if num_w > 0 else False,
                                  prefetch_factor=2 if num_w > 0 else None)
        val_loader = DataLoader(val_ds, batch_size=Cfg.batch_size, shuffle=False,
                                collate_fn=collate_pairs_fast, generator=g, num_workers=num_w,
                                pin_memory=False, persistent_workers=True if num_w > 0 else False,
                                prefetch_factor=2 if num_w > 0 else None)
        test_loader = DataLoader(test_ds, batch_size=Cfg.batch_size, shuffle=False,
                                 collate_fn=collate_pairs_fast, generator=g, num_workers=num_w,
                                 pin_memory=False, persistent_workers=True if num_w > 0 else False,
                                 prefetch_factor=2 if num_w > 0 else None)
        sample_graph_wt, sample_graph_mut, _ = train_ds[0]
        A_tmp, R_tmp = build_AR_from_data(sample_graph_wt)

        if mode == 'atom_only':
            edgeA_dim = int(A_tmp['edge_attr'].size(-1))
            sA_in = int(A_tmp['x_scalar'].size(-1))
            atom_enc = atom_encoder if atom_encoder is not None else Cfg.atom_encoder
            model = DDGPredictor(
                sA=Cfg.sA, sR=Cfg.sA, n_blocks=Cfg.n_blocks,
                edgeA_dim=edgeA_dim, edgeR_dim=0,  # edgeR_dim设为0，不使用
                sA_in_dim=sA_in, sR_in_dim=0, hidden_dim=256, mode='atom_only',
                atom_encoder=atom_enc, res_encoder='gat'  # sR_in_dim设为0，不使用
            ).to(device)
        elif mode == 'residue_only':
            edgeR_dim = int(R_tmp['edge_attr_s'].size(-1))
            sR_in = int(R_tmp['x_scalar'].size(-1))
            res_enc = res_encoder if res_encoder is not None else Cfg.res_encoder
            model = DDGPredictor(
                sA=Cfg.sA, sR=Cfg.sA, n_blocks=Cfg.n_blocks,
                edgeA_dim=0, edgeR_dim=edgeR_dim,  # edgeA_dim设为0，不使用
                sA_in_dim=0, sR_in_dim=sR_in, hidden_dim=256, mode='residue_only',
                atom_encoder='egnn', res_encoder=res_enc  # sA_in_dim设为0，不使用
            ).to(device)
        elif mode == 'dual':
            edgeA_dim = int(A_tmp['edge_attr'].size(-1))
            edgeR_dim = int(R_tmp['edge_attr_s'].size(-1))
            sA_in = int(A_tmp['x_scalar'].size(-1))
            sR_in = int(R_tmp['x_scalar'].size(-1))
            atom_enc = atom_encoder if atom_encoder is not None else Cfg.atom_encoder
            res_enc = res_encoder if res_encoder is not None else Cfg.res_encoder
            model = DDGPredictor(
                sA=Cfg.sA, sR=Cfg.sA, n_blocks=Cfg.n_blocks,
                edgeA_dim=edgeA_dim, edgeR_dim=edgeR_dim,
                sA_in_dim=sA_in, sR_in_dim=sR_in, hidden_dim=256, mode='dual',
                atom_encoder=atom_enc, res_encoder=res_enc
            ).to(device)
            
        def is_no_decay(n: str, p: nn.Parameter) -> bool:
            if p.ndim == 1:
                return True
            no_decay_keywords = ["bias", "LayerNorm", "layernorm", "ln", "norm"]
            return any(k.lower() in n.lower() for k in no_decay_keywords)
        
        decay_params = []
        no_decay_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if is_no_decay(name, param):
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        optimizer = torch.optim.AdamW(
            [
                {"params": decay_params, "weight_decay": Cfg.weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=Cfg.learning_rate,
        )
        scheduler = None
        criterion = nn.MSELoss()
        best_val_rmse = float('inf')
        patience_counter = 0
        es_patience = int(getattr(Cfg, 'early_stop_patience', 10))

        def run_epoch(loader, train: bool):
            if train:
                model.train()
            else:
                model.eval()
            all_predictions = []
            all_targets = []
            all_criterion_losses = []  # 收集MSE损失
            step_pbar = tqdm(enumerate(loader, start=1), total=len(loader),
                             desc=f"{'Training' if train else 'Validation'}",
                             leave=False, unit="batch")
            for step, (A_wt_b, R_wt_b, A_mut_b, R_mut_b, y_true_batch) in step_pbar:
                if A_wt_b['x_scalar'].size(0) == 0:
                    continue
                with torch.set_grad_enabled(train):
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
                    loss = criterion(ddg_pred, y_true_batch)
                    if torch.isnan(loss) or torch.isinf(loss):
                        continue
                    if train:
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        optimizer.step()
                    with torch.no_grad():
                        if ddg_pred.dim() == 0:
                            ddg_pred = ddg_pred.unsqueeze(0)
                        if y_true_batch.dim() == 0:
                            y_true_batch = y_true_batch.unsqueeze(0)
                        all_predictions.append(ddg_pred.detach().cpu())
                        all_targets.append(y_true_batch.detach().cpu())
                        all_criterion_losses.append(loss.detach().cpu())
            if len(all_predictions) > 0:
                all_pred_tensor = torch.cat(all_predictions, dim=0)
                all_target_tensor = torch.cat(all_targets, dim=0)
                epoch_criterion_loss = torch.stack(all_criterion_losses).mean()
                epoch_mse = ((all_pred_tensor - all_target_tensor) ** 2).mean()
                epoch_rmse = torch.sqrt(epoch_mse)
                epoch_pearson = pearsonr_torch(all_pred_tensor, all_target_tensor)
                epoch_spearman = spearmanr_torch(all_pred_tensor, all_target_tensor)
                epoch_r2 = r2_torch(all_pred_tensor, all_target_tensor)
                return {
                    'loss': float(epoch_criterion_loss),
                    'criterion_loss': float(epoch_criterion_loss),
                    'mse_loss': float(epoch_mse),
                    'rmse': float(epoch_rmse),
                    'pearson': float(epoch_pearson),
                    'spearman': float(epoch_spearman),
                    'r2': float(epoch_r2),
                }
            else:
                return {
                    'loss': 0.0,
                    'criterion_loss': 0.0,
                    'mse_loss': 0.0,
                    'rmse': 0.0,
                    'pearson': 0.0,
                    'spearman': 0.0,
                    'r2': 0.0,
                }

        for epoch in tqdm(range(1, Cfg.epochs + 1), desc=f"Fold {fold_id} Epochs", unit="epoch"):
            print(f"\n[E2E-CV][{mode}] Fold {fold_id}/{K} | Epoch {epoch}/{Cfg.epochs}", flush=True)
            tr_stats = run_epoch(train_loader, train=True)
            val_stats = run_epoch(val_loader, train=False)
            print(f"  Train     | Loss: {tr_stats['criterion_loss']:.4f} RMSE: {tr_stats['rmse']:.3f} Pearson: {tr_stats['pearson']:.3f} Spearman: {tr_stats['spearman']:.3f} R²: {tr_stats['r2']:.3f}", flush=True)
            print(f"  Validation| Loss: {val_stats['criterion_loss']:.4f} RMSE: {val_stats['rmse']:.3f} Pearson: {val_stats['pearson']:.3f} Spearman: {val_stats['spearman']:.3f} R²: {val_stats['r2']:.3f}", flush=True)
            if val_stats['rmse'] < best_val_rmse - 1e-5:
                best_val_rmse = val_stats['rmse']
                patience_counter = 0
                save_dict = {
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict() if optimizer else None,
                    'scheduler': scheduler.state_dict() if scheduler else None,
                    'train_loss': tr_stats['loss'],
                    'train_rmse': tr_stats['rmse'],
                    'train_pearson': tr_stats['pearson'],
                    'train_spearman': tr_stats['spearman'],
                    'train_r2': tr_stats['r2'],
                    'val_loss': val_stats['loss'],
                    'val_rmse': val_stats['rmse'],
                    'val_pearson': val_stats['pearson'],
                    'val_spearman': val_stats['spearman'],
                    'val_r2': val_stats['r2'],
                    'epoch': epoch
                }
                fold_val_path = os.path.join(output_dir, f"best_e2e_cv_fold_{fold_id}_val.pt")
                torch.save(save_dict, fold_val_path)
                torch.save(model.state_dict(), fold_val_path.replace('.pt', '_params.pt'))
                print(f"Saved best val model -> {fold_val_path}", flush=True)
                print(f"Val RMSE: {val_stats['rmse']:.4f} -> {best_val_rmse:.4f}", flush=True)
                print(f"Val Pearson: {val_stats['pearson']:.3f} Loss: {val_stats['loss']:.4f}", flush=True)
            if val_stats['rmse'] >= best_val_rmse - 1e-5:
                patience_counter += 1
                if patience_counter >= es_patience:
                    print(f"Early stopping at epoch {epoch} (no validation RMSE improvement for {patience_counter} epochs)", flush=True)
                    break
        best_model_path = os.path.join(output_dir, f"best_e2e_cv_fold_{fold_id}_val_params.pt")
        best_checkpoint_path = os.path.join(output_dir, f"best_e2e_cv_fold_{fold_id}_val.pt")
        test_stats = None
        
        if os.path.exists(best_model_path):
            model.load_state_dict(torch.load(best_model_path, map_location=device))
            if os.path.exists(best_checkpoint_path):
                checkpoint = torch.load(best_checkpoint_path, map_location=device)
                print(f"[E2E-CV][{mode}] Loaded best validation model from epoch {checkpoint['epoch']}", flush=True)
            else:
                print(f"[E2E-CV][{mode}] Loaded best validation model from {best_model_path}", flush=True)
            print(f"[E2E-CV][{mode}] Evaluating test set with best validation model (fold {fold_id})...", flush=True)
            test_stats = run_epoch(test_loader, train=False)
            print(f"[E2E-CV][{mode}] Test set metrics (fold {fold_id}):", flush=True)
            print(f"Loss: {test_stats['criterion_loss']:.4f} RMSE: {test_stats['rmse']:.3f} Pearson: {test_stats['pearson']:.3f} Spearman: {test_stats['spearman']:.3f} R²: {test_stats['r2']:.3f}", flush=True)
        else:
            print(f"[E2E-CV][{mode}] Warning: Best validation model not found at {best_model_path}", flush=True)
            print(f"[E2E-CV][{mode}] Evaluating test set with current model (fold {fold_id})...", flush=True)
            test_stats = run_epoch(test_loader, train=False)
            print(f"[E2E-CV][{mode}] Test set metrics (fold {fold_id}):", flush=True)
            print(f"Loss: {test_stats['criterion_loss']:.4f} RMSE: {test_stats['rmse']:.3f} Pearson: {test_stats['pearson']:.3f} Spearman: {test_stats['spearman']:.3f} R²: {test_stats['r2']:.3f}", flush=True)
        fold_results[fold_id] = {
            'test': test_stats
        }

    avg_results = {
        'test': {}
    }
    for metric in ['loss', 'rmse', 'pearson', 'spearman', 'r2']:
        values = [fold_results[f]['test'][metric] for f in fold_results]
        avg_results['test'][metric] = float(np.mean(values))
        avg_results['test'][f'{metric}_std'] = float(np.std(values))

    print(f"\n[E2E-CV][{mode}] Average test metrics across {K} folds:", flush=True)
    print(f"Test:", flush=True)
    print(f"Loss: {avg_results['test']['loss']:.4f} ± {avg_results['test']['loss_std']:.4f}", flush=True)
    print(f"RMSE: {avg_results['test']['rmse']:.3f} ± {avg_results['test']['rmse_std']:.3f}", flush=True)
    print(f"Pearson: {avg_results['test']['pearson']:.3f} ± {avg_results['test']['pearson_std']:.3f}", flush=True)
    print(f"Spearman: {avg_results['test']['spearman']:.3f} ± {avg_results['test']['spearman_std']:.3f}", flush=True)
    print(f"R²: {avg_results['test']['r2']:.3f} ± {avg_results['test']['r2_std']:.3f}", flush=True)

    import json as _json
    results = {
        "method": "end_to_end_unified",
        "atom_encoder": atom_encoder or Cfg.atom_encoder,
        "res_encoder": res_encoder or Cfg.res_encoder,
        "mode": mode,
        "avg_results": avg_results,
        "per_fold": fold_results,
        "config": {
            "epochs": Cfg.epochs,
            "lr": Cfg.learning_rate,
            "batch_size": Cfg.batch_size,
            "weight_decay": Cfg.weight_decay,
            "sA": Cfg.sA,
            "n_blocks": Cfg.n_blocks,
            "atom_encoder": atom_encoder or Cfg.atom_encoder,
            "res_encoder": res_encoder or Cfg.res_encoder,
            "mode": mode,
        },
    }
    results_path = f"{output_dir}/e2e_cv_results.json"
    Path(results_path).write_text(_json.dumps(results, indent=2))
    print(f"[E2E-CV][{mode}] Results saved -> {results_path}", flush=True)
    return {"per_fold": fold_results, "avg": avg_results}

def train_dual_datasets_m1101(mode='residue_only', atom_encoder=None, res_encoder=None):
    print("=" * 80, flush=True)
    print("Starting Sequential Training for M1101: S645 → M1101", flush=True)
    print("=" * 80, flush=True)
    datasets_config = [
        {
            'name': 'S645',
            'csv_path': '/root/AbMSPN/Model/csv/Pt_Mapping_S645.csv',
            'jsonl_path': '/root/AbMSPN/Model/jsonl/CV10_S645.jsonl',
            'root_dir': Cfg.root_dir,  # 使用相同的root_dir
        },
        {
            'name': 'M1101',
            'csv_path': '/root/AbMSPN/Model/csv/Pt_Mapping_M1101.csv',
            'jsonl_path': '/root/AbMSPN/Model/jsonl/CV5_M1101.jsonl',
            'root_dir': Cfg.root_dir,  # 使用相同的root_dir
        }
    ]
    all_results = {}
    for idx, dataset_config in enumerate(datasets_config, 1):
        dataset_name = dataset_config['name']
        print(f"\n{'=' * 80}", flush=True)
        print(f"Step {idx}/2: Training on {dataset_name} Dataset", flush=True)
        print(f"{'=' * 80}", flush=True)
        print(f"CSV: {dataset_config['csv_path']}", flush=True)
        print(f"JSONL: {dataset_config['jsonl_path']}", flush=True)
        print(f"Root Dir: {dataset_config['root_dir']}", flush=True)
        try:
            output_dir = f"CROSS_VALIDATION/{mode}_{dataset_name}"
            print(f"Output Dir: {output_dir}", flush=True)
            results = cross_validate_end_to_end_unified(
                mode=mode,
                output_dir=output_dir,
                atom_encoder=atom_encoder,
                res_encoder=res_encoder,
                csv_path=dataset_config['csv_path'],
                jsonl_path=dataset_config['jsonl_path']
            )
            
            all_results[dataset_name] = results
            print(f"{dataset_name} dataset training completed!", flush=True)  
        except Exception as e:
            print(f"Error training {dataset_name} dataset: {e}", flush=True)
            import traceback
            traceback.print_exc()
            continue
    print(f"\n{'=' * 80}", flush=True)
    print("Summary of Sequential Training (S645 → M1101)", flush=True)
    print(f"{'=' * 80}", flush=True)
    
    for dataset_name, results in all_results.items():
        if results and 'avg' in results and 'test' in results['avg']:
            test_avg = results['avg']['test']
            print(f"\n{dataset_name} Dataset Test Metrics:", flush=True)
            print(f"Loss: {test_avg.get('loss', 0):.4f} ± {test_avg.get('loss_std', 0):.4f}", flush=True)
            print(f"RMSE: {test_avg.get('rmse', 0):.3f} ± {test_avg.get('rmse_std', 0):.3f}", flush=True)
            print(f"Pearson: {test_avg.get('pearson', 0):.3f} ± {test_avg.get('pearson_std', 0):.3f}", flush=True)
            print(f"Spearman: {test_avg.get('spearman', 0):.3f} ± {test_avg.get('spearman_std', 0):.3f}", flush=True)
            print(f"R²: {test_avg.get('r2', 0):.3f} ± {test_avg.get('r2_std', 0):.3f}", flush=True)
    
    return all_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ddG predictor with end-to-end method")
    parser.add_argument("--method", type=str, choices=["e2e", "e2e-cv", "dual-m1101"], default="e2e-cv", 
                       help="Training method: e2e (end-to-end), e2e-cv (end-to-end CV), dual-m1101 (sequential training: S645 then M1101)")
    parser.add_argument("--fold", type=int, default=1, 
                       help="Fold ID to use for training (0-9, default: 5)")
    parser.add_argument("--mode", type=str, choices=["atom_only", "residue_only",'dual'], default="dual",
                       help="Graph mode: atom_only or residue_only")
    parser.add_argument("--force-dual", action="store_true",
                       help="Force dual dataset training even if not M1101")
    args = parser.parse_args()

    is_m1101 = "M1101" in str(Cfg.root_dir) or args.force_dual
    if args.method == "e2e":
        print(f"Starting End-to-End Training with Fold {args.fold}", flush=True)
        train_end_to_end(fold_id=args.fold, mode=args.mode)
        
    elif args.method == "e2e-cv":
        if is_m1101:
            print("Detected M1101 data, switching to sequential training (S645 → M1101)...", flush=True)
            train_dual_datasets_m1101(mode=args.mode, atom_encoder=Cfg.atom_encoder, res_encoder=Cfg.res_encoder)
        else:
            print("Starting End-to-End Cross-Validation", flush=True)
            cross_validate_end_to_end_unified(mode=args.mode, atom_encoder=Cfg.atom_encoder, res_encoder=Cfg.res_encoder)
    
    elif args.method == "dual-m1101":
        print("Starting Sequential Training (S645 → M1101)", flush=True)
        train_dual_datasets_m1101(mode=args.mode, atom_encoder=Cfg.atom_encoder, res_encoder=Cfg.res_encoder)
    else:
        print("Invalid method. Use --method e2e, --method e2e-cv, or --method dual-m1101", flush=True)