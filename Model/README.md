# AbMSPN Model — Training and Prediction

GNN-based antibody–antigen ΔΔG prediction using GVP encoders on atom- and residue-level graphs, with optional ESM fusion.

## Setup

1. Activate environment: `conda activate AbMSPN`
2. Edit paths in `model.py` → `class Cfg`:

```python
csv_path = '/root/AbMSPN/Model/csv/Pt_Mapping_S645.csv'
root_dir = '/root/AbMSPN/Data/graphs'   # directory of *.pt graph files (not bundled)
split_json_path = '/root/AbMSPN/Model/jsonl/CV10_S645.jsonl'
json_indices_one_based = True
use_esm = False
```

## Training

```bash
cd /root/AbMSPN/Model

python train.py --method e2e --fold 1 --mode dual      # single fold
python train.py --method e2e-cv --mode dual            # full CV
python train.py --method dual-m1101 --mode dual        # S645 → M1101 sequential
```

| `--mode` | Description |
|----------|-------------|
| `atom_only` | Atom graph encoder only |
| `residue_only` | Residue graph encoder only |
| `dual` | Both encoders (default) |

Outputs: `best_e2e_model_fold_{k}_val.pt`, `best_e2e_model_fold_{k}_val_params.pt`, `e2e_cv_results.json`

## Testing

```bash
python test.py --model_dir . --fold_id 1 --mode dual
python test.py --model_dir . --n_folds 10 --mode dual --use_val_model
```

## Prediction

Runs on the test split defined in `Cfg.split_json_path` (not a custom CSV):

```bash
python predict_csv.py --model_dir . --mode dual --merge_output --output_dir ./predictions
```

## Key `Cfg` Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `epochs` | 100 | Max training epochs |
| `batch_size` | 8 | Batch size |
| `learning_rate` | 1e-4 | AdamW learning rate |
| `weight_decay` | 1e-3 | L2 regularization |
| `dropout` | 0.25 | Dropout rate |
| `early_stop_patience` | 40 | Early stopping (val RMSE) |
| `n_blocks` | 4 | GVP macro-block count |
| `sA` | 128 | Hidden scalar dimension |
| `use_esm` | False | Enable ESM fusion branch |

## Data Files

- `csv/Pt_Mapping_{S645,S1131,M1101}.csv` — graph filename ↔ ΔΔG mapping
- `jsonl/CV10_S645.jsonl` etc. — train/val index splits (1-based)

Mapping CSV columns: `wild_pt`, `Partners`, `mutant_pt`, `Mutation`, `ddG`
