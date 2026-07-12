# AbMSPN: Antibody–Antigen ΔΔG Prediction

End-to-end pipeline for predicting binding affinity changes (ΔΔG) on antibody–antigen complexes. The workflow builds atom/residue-level graphs from PDB structures, extracts structural and sequence features (optional ESM/PSSM), and trains GNN models with contrastive WT/MT learning.

**Code repository:** `AbMSPN/` (this repo)  
**Companion data archive (Zenodo):** `AbMSPN_Data/` — raw structures, FoldX outputs, and trained model checkpoints (~13 GB). Download separately and place next to the code repo (e.g. `/root/AbMSPN_Data`).

![AbMSPN pipeline](picture.jpg)

## Repository Structure

```
AbMSPN/                  # source code (this repository)
├── Data/                # graph construction scripts
├── Model/               # training / testing / prediction
├── Results/             # pre-computed benchmark predictions
├── environment.yml
├── picture.jpg
└── README.md

AbMSPN_Data/             # companion Zenodo archive (download separately)
├── M1101-Ab-Bind/       # AB-Bind PDB structures
├── S1131-SKEMPI/        # SKEMPI S1131 PDB + FoldX outputs
└── Model_params/        # trained checkpoints (AbMSPN + baselines)
```

### What is bundled vs external

| In **AbMSPN** (code repo) | In **AbMSPN_Data** (Zenodo) | Still external |
|---------------------------|-----------------------------|----------------|
| `Model/csv/Pt_Mapping_*.csv` | M1101 PDBs (`M1101-Ab-Bind/`) | Graph `.pt` files (`Cfg.root_dir`) |
| `Model/jsonl/*.jsonl` (CV splits) | S1131 PDB + FoldX (`S1131-SKEMPI/`) | ESM2 weights, UniRef90 BLAST DB |
| `Data/AB-Bind_experimental_data.csv` | Trained checkpoints (`Model_params/`) | S645 raw PDB (not in archive) |
| `Results/**/predictions_*.csv` | | |
| All Python scripts | | |

Training still requires **graph `.pt` files** (build locally from Zenodo PDB/FoldX, or use your own copy). Pre-trained weights for reproduction are under `AbMSPN_Data/Model_params/`. See [Companion Dataset (Zenodo)](#companion-dataset-zenodo-abmspn_data).

### Data/

| File | Purpose |
|------|---------|
| `build_graphs_S1131.py` | Build WT/MT graph pairs for S1131 (CLI path overrides) |
| `build_graphs_M1101.py` | Build graphs for M1101 / AB-Bind (paths in script header) |
| `graph_builder_mutation.py` | Core graph construction (atom/residue graphs, WT/MT pairs) |
| `features.py` | Feature extraction (atom/residue, ASA, PSSM, optional ESM) |
| `pdb_processor.py` | PDB parsing helpers |
| `utils.py` | Geometry / VdW helpers used by graph builder |
| `pssm_precompute.py` | Optional deduplicated PSSM cache for S1131 |
| `config.py` | Global graph/feature config (cutoffs, ESM/PSSM toggles) |
| `AB-Bind_experimental_data.csv` | M1101 mutation table (bundled) |

### Model/

| File / Dir | Purpose |
|------------|---------|
| `model.py` | GVP-based model + `Cfg` defaults |
| `train.py` | End-to-end training and cross-validation |
| `test.py` | Single- or multi-fold evaluation |
| `predict_csv.py` | Run all CV folds on the test split and export predictions |
| `csv/` | `Pt_Mapping_S645.csv`, `Pt_Mapping_S1131.csv`, `Pt_Mapping_M1101.csv` |
| `jsonl/` | CV splits (`CV10_S645.jsonl`, `CV5_M1101.jsonl`, `SquenceIdentity_*.jsonl`, …) |
| `README.md` | Model-level documentation |

### Results/

Pre-computed benchmark outputs only (evaluation scripts are **not** included in this repo):

| Path | Purpose |
|------|---------|
| `AbMSPN_Resluts/` | AbMSPN ablation predictions (see below) |
| `MpbPPI_Results/` | MpbPPI baseline predictions |
| `GearBind_Results/` | GearBind baseline predictions |
| `abCAN_Resluts/` | abCAN baseline predictions |

Each dataset folder contains `predictions_all_folds.csv` and `metrics_summary.csv`. Folder names `AbMSPN_Resluts` / `abCAN_Resluts` retain the original spelling used in filenames.

---

## Companion Dataset (Zenodo): AbMSPN_Data

Archive uploaded to Zenodo alongside this repository. Typical layout after download:

```
AbMSPN_Data/                          (~13 GB total)
├── M1101-Ab-Bind/                    (~464 MB)
│   ├── 1ak4/                         # one folder per complex
│   │   ├── 1ak4_Repair.pdb
│   │   ├── 1ak4_Repair_1.pdb           # mutant structures
│   │   └── ...
│   ├── 1bj1/
│
├── S1131-SKEMPI/                     (~1.2 GB)
│   ├── PDB/                            # 158 WT PDB structures
│   ├── S1131.txt                       # 1131 mutation records (SKEMPI format)
│   ├── allExpData                      # experimental metadata
│   ├── selectExpDDG                    # selected ΔΔG subset
│   ├── foldx_lists/                    # per-complex FoldX mutation lists
│   ├── make.py / mutant.py             # FoldX list generation helpers
│   └── FoldX_Output/
│       ├── Mutants_repaired_only/      # repaired mutant PDBs (112 complexes)
│       ├── Mutants/ Repaired/ Renumbered/ Logs/ TempLists/
│
└── Model_params/                     (~12 GB)
    ├── AbMSPN_Resluts/                 # 12 ablation variants × CV splits
    │   ├── Dual/CV10_S645/
    │   │   ├── best_e2e_cv_fold_{k}_val.pt
    │   │   ├── best_e2e_cv_fold_{k}_val_params.pt
    │   │   └── e2e_cv_results.json
    │   └── AtomGraph/ ... No_PSY/ ...
    ├── GearBind_Results/             # GearBind checkpoints + metrics JSON
    ├── MpbPPI_Results/
    └── abCAN_Resluts/
```

**Not included in AbMSPN_Data:** pre-built graph `.pt` files, S645 PDB structures, ESM2 weights, UniRef90 BLAST database.

### Mapping Zenodo paths → code defaults

After downloading, either symlink into `AbMSPN/Data/` or point scripts / `Cfg` at the archive directly:

| Purpose | AbMSPN_Data path | Used by |
|---------|------------------|---------|
| M1101 PDB input | `M1101-Ab-Bind/` | `build_graphs_M1101.py` → `BASE_DATA_PATH` |
| S1131 FoldX mutants | `S1131-SKEMPI/FoldX_Output/Mutants_repaired_only/` | `build_graphs_S1131.py` → `--base_data_path` |
| S1131 mutation table | `S1131-SKEMPI/S1131.txt` (convert to CSV as needed) | graph builder CSV input |
| S1131 WT PDBs | `S1131-SKEMPI/PDB/` | reference structures |
| Pre-trained AbMSPN | `Model_params/AbMSPN_Resluts/Dual/CV10_S645/` | `test.py` / `predict_csv.py` → `--model_dir` |
| Baseline models | `Model_params/{GearBind,MpbPPI,abCAN}_*/` | external baseline reproduction |

Example symlinks (optional, from repo root):

```bash
ln -s /path/to/AbMSPN_Data/M1101-Ab-Bind /root/AbMSPN/Data/abbind
ln -s /path/to/AbMSPN_Data/S1131-SKEMPI/FoldX_Output/Mutants_repaired_only \
      /root/AbMSPN/Data/S1131/FoldX_Output/Mutants_repaired_only
```

### Run inference with Zenodo checkpoints

```bash
conda activate AbMSPN
cd /root/AbMSPN/Model

python test.py \
  --model_dir /root/AbMSPN_Data/Model_params/AbMSPN_Resluts/Dual/CV10_S645 \
  --n_folds 10 --mode dual

python predict_csv.py \
  --model_dir /root/AbMSPN_Data/Model_params/AbMSPN_Resluts/Dual/CV10_S645 \
  --mode dual --merge_output --output_dir ./predictions
```

Ensure `Cfg.csv_path`, `Cfg.root_dir`, and `Cfg.split_json_path` match the dataset folder you are evaluating (e.g. S645 graphs + `CV10_S645.jsonl`).

---

## Requirements

**Recommended:** create the Conda environment from `environment.yml` (Python 3.10, PyTorch / PyTorch Geometric, fair-esm, freesasa, etc.).

```bash
cd /root/AbMSPN
conda env create -f environment.yml
conda activate AbMSPN
```

**Minimal manual setup** (if not using Conda):

- Python 3.10 (matches `environment.yml`)
- PyTorch 1.8+ (CUDA supported)
- PyTorch Geometric
- NumPy, Pandas, tqdm, SciPy
- Biopython (Data pipeline)

```bash
pip install torch torch-geometric numpy pandas tqdm scipy biopython
```

**Optional** (enable only when used):

| Component | Install |
|-----------|---------|
| ESM | `pip install fair-esm` |
| freesasa (ASA) | `pip install freesasa` or `conda install -c conda-forge freesasa-c` |
| BLAST+ (PSSM) | `conda install -c bioconda blast` |
| DSSP | `conda install -c conda-forge dssp` |
| FoldX (S1131 mutants) | https://foldxsuite.crg.eu |

---

## Paths and Data

### Layout (relative to repository)

- Graph **inputs/outputs** → `Data/` (build scripts) and `Model/model.py` → `Cfg.root_dir`
- Training **mapping CSVs** and **CV splits** → `Model/csv/`, `Model/jsonl/`
- Benchmark **predictions** → `Results/`

```
AbMSPN/
├── Data/
│   ├── abbind/              # M1101 PDB input (external)
│   ├── graphs/              # example graph .pt output (M1101 normal mode)
│   ├── wt_mt_graphs/        # example WT/MT output (M1101 wt_mt mode)
│   └── S1131_graphs/        # S1131 graph output
├── Model/csv/               # Pt_Mapping_*.csv (bundled)
├── Model/jsonl/             # CV*.jsonl (bundled)
└── uniref/                  # optional PSSM BLAST database
    ├── uniref90.fasta
    └── uniref90.*
```

Scripts default to `/root/AbMSPN/...`. Override via CLI flags (S1131 builder) or edit the config / script headers.

### UniRef90 / PSSM database setup (optional)

Required only when `ENABLE_PSSM = True` in `Data/config.py`. For a quick test, set `ENABLE_PSSM = False` to skip BLAST entirely.

```bash
mkdir -p /root/AbMSPN/uniref && cd /root/AbMSPN/uniref
wget https://ftp.uniprot.org/pub/databases/uniprot/uniref/uniref90/uniref90.fasta.gz
gunzip uniref90.fasta.gz
makeblastdb -in uniref90.fasta -dbtype prot -out uniref90
```

Then in `Data/config.py`:

```python
SWISSPROT_FASTA_PATH = "../uniref/uniref90.fasta"
SWISSPROT_DB_PATH    = "../uniref/uniref90"
ENABLE_PSSM = True
```

### ESM2 local weights (optional)

In `Data/config.py`:

```python
ESM2_MODEL_NAME = 'esm2_t33_650M_UR50D'
ESM2_MODEL_PATH = '/root/AbMSPN/Data/esm/esm2_t33_650M_UR50D.pt'
ESM2_ENABLE_CACHE = True
ESM2_CACHE_DIR = os.path.join(os.path.dirname(__file__), 'esm_cache')
```

Set `Cfg.use_esm = True` in `Model/model.py` when graphs include ESM embeddings.

---

## Before You Run: Configure Paths

### 1) Training — `Model/model.py` → `class Cfg`

Default values (S645 10-fold CV):

```python
class Cfg:
    csv_path = '/root/AbMSPN/Model/csv/Pt_Mapping_S645.csv'
    root_dir = '/root/AbMSPN/Data/graphs'          # must contain *.pt graphs
    split_json_path = '/root/AbMSPN/Model/jsonl/CV10_S645.jsonl'
    json_indices_one_based = True
    use_esm = False
```

**Important:** `root_dir` must point to an existing directory of graph `.pt` files referenced by the mapping CSV. The repo does not ship these files.

| Dataset | `csv_path` | `split_json_path` | Folds |
|---------|------------|-------------------|-------|
| S645 | `Model/csv/Pt_Mapping_S645.csv` | `Model/jsonl/CV10_S645.jsonl` | 10 |
| S1131 | `Model/csv/Pt_Mapping_S1131.csv` | `Model/jsonl/CV10_S1131.jsonl` | 10 |
| M1101 | `Model/csv/Pt_Mapping_M1101.csv` | `Model/jsonl/CV5_M1101.jsonl` | 5 |
| DeepMutant | — | `Model/jsonl/DeepMutant.jsonl` | — |
| Seq-identity split | — | `Model/jsonl/SquenceIdentity_S645.jsonl` | 10 |

`train.py --method e2e-cv` runs **M1101 sequential training (S645 → M1101)** automatically when `"M1101"` appears in `Cfg.root_dir`, or when `--force-dual` is set.

### 2) S1131 graph builder — `Data/build_graphs_S1131.py`

With Zenodo data at `/root/AbMSPN_Data` (all overridable on CLI):

```python
BASE_DATA_PATH  = '/root/AbMSPN_Data/S1131-SKEMPI/FoldX_Output/Mutants_repaired_only'
CSV_FILE_PATH   = '/root/AbMSPN/Data/S1131_with_pdbid.csv'   # prepare from S1131.txt
OUTPUT_BASE_DIR = '/root/AbMSPN/Data/S1131_graphs'
```

### 3) M1101 graph builder — `Data/build_graphs_M1101.py`

With Zenodo data:

```python
BASE_DATA_PATH = '/root/AbMSPN_Data/M1101-Ab-Bind'
CSV_FILE_PATH  = '/root/AbMSPN/Data/AB-Bind_experimental_data.csv'  # bundled in code repo
# OUTPUT: Data/graphs (normal) or Data/wt_mt_graphs (wt_mt, default)
```

### 4) Feature paths — `Data/config.py`

```python
SWISSPROT_FASTA_PATH = '/root/AbMSPN/Data/swiss/uniprot_sprot.fasta'
SWISSPROT_DB_PATH    = '/root/AbMSPN/uniref/uniref90'
ESM2_MODEL_PATH      = '/root/AbMSPN/Data/esm/esm2_t33_650M_UR50D.pt'
ENABLE_PSSM = True    # False = much faster graph building
```

---

## Building Graphs

Run from `Data/` with the environment activated. Download [AbMSPN_Data from Zenodo](#companion-dataset-zenodo-abmspn_data) first for M1101 / S1131 PDB inputs.

```bash
conda activate AbMSPN
cd /root/AbMSPN/Data
```

### S1131

```bash
python build_graphs_S1131.py \
  --num_processes 4 \
  --base_data_path /root/AbMSPN_Data/S1131-SKEMPI/FoldX_Output/Mutants_repaired_only \
  --csv_file_path /path/to/S1131_with_pdbid.csv \
  --output_dir /root/AbMSPN/Data/S1131_graphs \
  --atom_graph_mode radius
```

| `--atom_graph_mode` | Description |
|---------------------|-------------|
| `radius` (default) | Radius-based atom graph |
| `interaction` | Physics-based interaction edges only |

Outputs per mutation: `{mt}_wt_graph.pt`, `{mt}_graph.pt`, plus `s1131_graph_results.csv`. Chain mapping comes from the CSV `Partners` column.

### M1101 (AB-Bind)

```bash
python build_graphs_M1101.py --mode wt_mt --num_processes 3 --atom_graph_mode radius
```

| `--mode` | Output directory |
|----------|------------------|
| `wt_mt` (default) | `Data/wt_mt_graphs/` |
| `normal` | `Data/graphs/` |

Requires PDB files under `AbMSPN_Data/M1101-Ab-Bind/` (or symlink to `Data/abbind/`). After building, set `Cfg.root_dir` to the output directory and switch `csv_path` / `split_json_path` to M1101 presets.

### Optional: PSSM precompute (S1131)

```bash
python pssm_precompute.py
```

Edit `BASE_DATA_PATH` and `PSSM_CACHE_DIR` at the top of `pssm_precompute.py`. Requires a built BLAST database and prior sequence-index files under the cache directory.

---

## Training

```bash
conda activate AbMSPN
cd /root/AbMSPN/Model

# Single fold
python train.py --method e2e --fold 1 --mode dual

# Full CV on current Cfg dataset
python train.py --method e2e-cv --mode dual

# Force S645 → M1101 sequential training
python train.py --method dual-m1101 --mode dual
```

| Argument | Values | Description |
|----------|--------|-------------|
| `--method` | `e2e`, `e2e-cv`, `dual-m1101` | Training mode |
| `--fold` | 1–10 (1–5 for M1101) | Fold for `e2e` mode |
| `--mode` | `atom_only`, `residue_only`, `dual` | Encoder mode |

Checkpoints (current working directory):

- `best_e2e_model_fold_{k}_val.pt`
- `best_e2e_model_fold_{k}_val_params.pt`
- `e2e_cv_results.json`

Default hyperparameters (`Cfg`): `epochs=100`, `batch_size=8`, `lr=1e-4`, `weight_decay=1e-3`, `dropout=0.25`, `early_stop_patience=40`, `n_blocks=4`, `sA=128`.

---

## Testing

```bash
cd /root/AbMSPN/Model

python test.py --model_dir . --fold_id 1 --mode dual
python test.py --model_dir . --n_folds 10 --mode dual --use_val_model
```

Metrics: MAE, RMSE, Pearson, Spearman, R² → `e2e_cv_results.json`.

---

## Prediction on Test Split

`predict_csv.py` uses indices from `Cfg.split_json_path` (no custom input CSV):

```bash
python predict_csv.py \
  --model_dir . \
  --mode dual \
  --output_dir ./predictions \
  --merge_output
```

Outputs: `predictions_fold_{k}.csv`, `predictions_all_folds.csv`.

---

## AbMSPN Ablation Experiments

Under `Results/AbMSPN_Resluts/`:

| Variant | Description |
|---------|-------------|
| `Dual` | Atom + residue graphs (full model) |
| `AtomGraph` | Atom graph only |
| `ResidueGraph` | Residue graph only |
| `AtomGraph+ResidueGraph` | Dual without ESM |
| `ESMFusion` | ESM sequence fusion |
| `AtomGraph+ESMFusion` | Atom + ESM |
| `ResidueGraph+ESMFusion` | Residue + ESM |
| `No_DSSP` | Without DSSP features |
| `No_Features` | Minimal feature set |
| `No_Interaction` | Without interaction edges |
| `No_PSSM` | Without PSSM features |
| `No_PSY` | Without PSY features |

**Dataset coverage:** `Dual/` includes all seven benchmark splits (`CV10_S645`, `CV10_S1131`, `CV5_M1101`, `DeepMutant`, `SquenceIdentity_*`). Most other ablation variants only include the three CV splits (`CV10_S645`, `CV10_S1131`, `CV5_M1101`).

Each dataset folder contains:

- `predictions_all_folds.csv`
- `metrics_summary.csv`
- optional per-fold `predictions_fold_*.csv`

---

## Data Formats

### Graph `.pt` files

PyTorch Geometric `Data` objects:

- **Atom**: `x_scalar`, `x_vector`, `edge_index`, `edge_attr`, `pos`
- **Residue**: `x_scalar`, `x_vector`, `edge_index_s`, `edge_index_r`, `edge_attr_s`, `edge_attr_r`, `pos`
- **ESM (optional)**: `esm_Ab1`, `esm_Ab2`, `esm_Ag1`, …

### Mapping CSV (`Pt_Mapping_*.csv`)

| Column | Description |
|--------|-------------|
| `wild_pt` | WT graph filename |
| `Partners` | Chain code (e.g. `A_D`, `HL_P`) |
| `mutant_pt` | MUT graph filename |
| `Mutation` | Mutation label |
| `ddG` | Experimental ΔΔG (kcal/mol) |

### JSONL splits

One JSON object per line; indices are **1-based** when `Cfg.json_indices_one_based = True`:

```json
{"train": ["1", "2", ...], "val": ["9", "11", ...]}
```

---

## Quick Start (S645, dual mode)

```bash
# 1. Clone code + download AbMSPN_Data from Zenodo
#    Place AbMSPN_Data next to AbMSPN (e.g. /root/AbMSPN_Data)

# 2. Environment
cd /root/AbMSPN
conda env create -f environment.yml
conda activate AbMSPN

# 3. Option A — reproduce with pre-trained weights (needs graph .pt for test/predict):
python test.py \
  --model_dir /root/AbMSPN_Data/Model_params/AbMSPN_Resluts/Dual/CV10_S645 \
  --n_folds 10 --mode dual

# 3. Option B — train from scratch:
#    Build/obtain graph .pt files, edit Model/model.py → Cfg.root_dir, then:
python train.py --method e2e-cv --mode dual

# 4. Export predictions
python predict_csv.py --model_dir . --mode dual --merge_output --output_dir ./predictions
```

Pre-computed benchmark metrics: `Results/AbMSPN_Resluts/Dual/CV10_S645/metrics_summary.csv`.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError: utils` | Ensure `Data/utils.py` is present (required by graph builder) |
| `FileNotFoundError` for graphs | Build graphs or copy `.pt` files; set `Cfg.root_dir` accordingly |
| `Cfg` paths wrong for your setup | Update `csv_path`, `root_dir`, `split_json_path` in `Model/model.py` |
| M1101 build fails on data path | Download Zenodo archive; PDBs live under `AbMSPN_Data/M1101-Ab-Bind/` |
| S1131 build finds no PDBs | Use `AbMSPN_Data/S1131-SKEMPI/FoldX_Output/Mutants_repaired_only/` |
| PSSM very slow / BLAST missing | Set `ENABLE_PSSM = False` in `Data/config.py` |
| ESM not used in training | Set `Cfg.use_esm = True` and build graphs with ESM cache enabled |
| `predict_csv.py` wrong split | Match `Cfg.split_json_path` to the dataset you trained on |
| Unexpected M1101 sequential training | Default `root_dir` no longer contains `M1101`; use `--force-dual` to enable |

---

## Notes

- Conda environment name: **AbMSPN**; repository folder: **AbMSPN**.
- **AbMSPN** (code) and **AbMSPN_Data** (Zenodo) are separate downloads; together they cover PDB/FoldX/checkpoints, but not graph `.pt` or S645 PDBs.
- Graph/PDB raw data: Zenodo `AbMSPN_Data/`; graph `.pt` outputs: build locally under `Data/`.
- Default antibody/antigen chains in `Data/config.py` are `D` / `A`; per-complex overrides come from the `Partners` column in mapping CSVs.
- See `Model/README.md` for additional model-level details.
