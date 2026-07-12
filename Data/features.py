import numpy as np
import logging
import subprocess
import os
import uuid
import re
import torch
import math
import pickle
from config import (
    ATOM_TYPES, ATOM_NAMES, AMINO_ACIDS, amino_acid_to_idx, atom_type_to_idx, atom_name_to_idx,
    SWISSPROT_FASTA_PATH, SWISSPROT_DB_PATH,
    ANTIBODY_CHAINS, ESM2_MODEL_PATH,
    ESM2_USE_AUTOCUDA, ESM2_USE_AMP, ESM2_BATCH_SIZE, ESM2_CACHE_DIR, ESM2_ENABLE_CACHE,
    ESM2_CPU_THREADS, ESM2_USE_QUANTIZATION, ESM2_ONLY_ANTIBODY, ESM2_SORT_BY_LENGTH,
    ENABLE_PSSM,
)
from utils import (
    is_polar, is_aromatic, is_hydrophobic, 
    calculate_edge_angles, find_attached_hydrogens
)

logger = logging.getLogger(__name__)
try:
    from Bio.PDB.DSSP import DSSP
    _DSSP_AVAILABLE = True
except Exception:
    _DSSP_AVAILABLE = False
    DSSP = None
    logger.warning("DSSP 未可用(缺少 Biopython DSSP 或 mkdssp)")

def _unit(v, eps=1e-8):
    """单位向量归一化"""
    return v / (v.norm(dim=-1, keepdim=True) + eps)

def _normalize(v, eps=1e-8):
    """向量归一化(numpy版本)"""
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.clip(n, eps, None)

def _dihedral(p0, p1, p2, p3):
    b0 = p0 - p1
    b1 = p2 - p1
    b2 = p3 - p2
    b1n = _normalize(b1)
    v = b0 - (b0 * b1n).sum(-1, keepdims=True) * b1n
    w = b2 - (b2 * b1n).sum(-1, keepdims=True) * b1n
    x = (v * w).sum(-1)
    y = (np.cross(b1n, v) * w).sum(-1)
    return np.arctan2(y, x)

def backbone_dihedrals_sincos(N, CA, C):
    def _normalize(v, eps=1e-8):
        n = np.linalg.norm(v, axis=-1, keepdims=True)
        return v / np.clip(n, eps, None)

    def _dihedral(p0, p1, p2, p3):
        b0, b1, b2 = p0-p1, p2-p1, p3-p2
        b1n = _normalize(b1)
        v = b0 - (b0*b1n).sum(-1, keepdims=True)*b1n
        w = b2 - (b2*b1n).sum(-1, keepdims=True)*b1n
        x = (v*w).sum(-1)
        y = (np.cross(b1n, v)*w).sum(-1)
        return np.arctan2(y, x)

    L = CA.shape[0]
    phi   = np.zeros(L)
    psi   = np.zeros(L)
    omega = np.zeros(L)
    if L >= 2:
        phi[1:]    = _dihedral(C[:-1],  N[1:],  CA[1:],  C[1:])        # C_{i-1},N_i,CA_i,C_i
        psi[:-1]   = _dihedral(N[:-1],  CA[:-1],C[:-1], N[1:])        # N_i,CA_i,C_i,N_{i+1}
        omega[:-1] = _dihedral(CA[:-1], C[:-1], N[1:],  CA[1:])       # CA_i,C_i,N_{i+1},CA_{i+1}
    feats = np.stack([np.sin(phi),np.cos(phi),
                      np.sin(psi),np.cos(psi),
                      np.sin(omega),np.cos(omega)], axis=-1).astype(np.float32)
    return feats

def chain_dihedral_feats(structure, model_id=0, chain_id="A"):
    model = structure[model_id]
    chain = model[chain_id]
    # 只保留标准残基且主链原子齐全
    residues = [r for r in chain if r.id[0] == ' ' and all(a in r for a in ('N','CA','C'))]
    N  = np.array([r['N'].get_coord()  for r in residues], dtype=float)
    CA = np.array([r['CA'].get_coord() for r in residues], dtype=float)
    C  = np.array([r['C'].get_coord()  for r in residues], dtype=float)
    feats = backbone_dihedrals_sincos(N, CA, C)   #[L,6]
    return residues, feats

def virtual_cb_direction(xN: torch.Tensor,
                         xCA: torch.Tensor,
                         xC: torch.Tensor,
                         eps: float = 1e-8) -> torch.Tensor:
    n = xN - xCA
    c = xC - xCA
    w = _unit(torch.cross(n, c, dim=-1), eps=eps)   # 主链平面法向
    u = _unit(n + c, eps=eps)                       # 角平分方向
    inv_sqrt3      = 1.0 / math.sqrt(3.0)
    sqrt_2_over_3  = math.sqrt(2.0/3.0)
    vcb = _unit(w * inv_sqrt3 - u * sqrt_2_over_3, eps=eps)
    return vcb  # [...,3]

_SS8_ORDER = ['H', 'B', 'E', 'G', 'I', 'T', 'S', '-']
def _ss8_one_hot(ch: str):
    """将 DSSP 二级结构字符转换为 8 维 one-hot 向量"""
    vec = np.zeros(8, dtype=np.float32)
    if ch not in _SS8_ORDER:
        ch = '-'
    vec[_SS8_ORDER.index(ch)] = 1.0
    return vec

from Bio.PDB import Select
class OnlyStdProtein(Select):
    def accept_model(self, model):
        return 1
    def accept_chain(self, chain):
        return 1
    def accept_residue(self, residue):
        return 1 if residue.id[0] == ' ' else 0
    def accept_atom(self, atom):
        try:
            return 0 if atom.element == 'H' else 1
        except Exception:
            return 0 if str(atom.get_name()).startswith('H') else 1

def _format_cryst1(a=100.0, b=100.0, c=100.0,
                   alpha=90.0, beta=90.0, gamma=90.0,
                   sg="P 1", z=1) -> str:
    return f"CRYST1{a:9.3f}{b:9.3f}{c:9.3f}{alpha:7.2f}{beta:7.2f}{gamma:7.2f} {sg:<11s}{z:4d}\n"

def _ensure_cryst1(pdb_path):
    with open(pdb_path, 'r') as f:
        lines = [ln for ln in f.readlines() if ln.strip() != ""]
    if not lines or not lines[0].startswith("CRYST1"):
        cryst1 = _format_cryst1()
        lines.insert(0, cryst1)
    else:
        lines[0] = _format_cryst1()
    with open(pdb_path, 'w') as f:
        f.writelines(lines)

def run_dssp_safe(structure, prefer_prog='mkdssp', acc_array='Sander'):
    if not _DSSP_AVAILABLE or DSSP is None:
        return None
    dssp_cmd = None
    for cmd in ['mkdssp', 'dssp', 'dsspcmbi']:
        result = subprocess.run(['which', cmd], capture_output=True, timeout=1, text=True)
        if result.returncode == 0 and result.stdout.strip():
            dssp_cmd = cmd
            break
    if dssp_cmd is None:
        logger.warning(f"DSSP 可执行文件未找到(已尝试 mkdssp/dssp/dsspcmbi)")
        return None
    
    model = next(structure.get_models())
    import tempfile
    from Bio.PDB import PDBIO
    io = PDBIO()
    io.set_structure(model)
    tmp = tempfile.NamedTemporaryFile(suffix='.pdb', delete=False)
    tmp.close()
    clean_pdb = tmp.name
    io.save(clean_pdb, select=OnlyStdProtein())
    _ensure_cryst1(clean_pdb)
    try:
        dssp = DSSP(model, clean_pdb, dssp=dssp_cmd, acc_array=acc_array, file_type='PDB')
        return dssp
    except Exception as e:
        logger.warning(f"DSSP 运行失败: {e}")
        return None
    finally:
        try:
            os.remove(clean_pdb)
        except Exception:
            pass

def compute_dssp_cache(structure, pdb_path, dssp_exe='mkdssp', acc_array='Sander'):
    """
    [SS8 onehot(8), rASA(1), sin(phi),cos(phi),sin(psi),cos(psi)(4), HB energies(4)]
    """
    cache = {}
    dssp = run_dssp_safe(structure, prefer_prog=dssp_exe, acc_array=acc_array)
    if dssp is None:
        logger.warning("DSSP 计算失败,DSSP 特征将为 0")
        return cache
    for key in dssp.keys():
        chain_id, res_id = key
        row = dssp[key]
        ss_vec = _ss8_one_hot(row[2])
        rsa = float(row[3]); rsa = float(min(max(rsa, 0.0), 1.0))
        phi = math.radians(float(row[4]));  psi = math.radians(float(row[5]))
        sphi, cphi = math.sin(phi), math.cos(phi)
        spsi, cpsi = math.sin(psi), math.cos(psi)
        hb = [float(row[7]), float(row[9]), float(row[11]), float(row[13])]
        feat = np.concatenate([ss_vec, [rsa, sphi, cphi, spsi, cpsi], hb]).astype(np.float32)
        cache.setdefault(chain_id, {})[res_id] = feat
    
    return cache

try:
    import freesasa
    FREESASA_AVAILABLE = True
except ImportError:
    FREESASA_AVAILABLE = False
    logger.warning("freesasa未安装,ASA特征将被禁用")

try:
    from Bio import Blast
    PSSM_AVAILABLE = True
except ImportError:
    PSSM_AVAILABLE = False
    logger.warning("Bio.Blast未安装,PSSM特征将被禁用")

_ESM2_MODEL_CACHE = {
    'initialized': False,
    'model': None,
    'alphabet': None,
    'batch_converter': None,
    'device': None,
}
def _load_esm2_model(model_name=None):
    if _ESM2_MODEL_CACHE['initialized']:
        return (
            _ESM2_MODEL_CACHE['model'],
            _ESM2_MODEL_CACHE['alphabet'],
            _ESM2_MODEL_CACHE['batch_converter'],
            _ESM2_MODEL_CACHE['device'],
        )
    try:
        import torch
        import esm
        if not (ESM2_MODEL_PATH and os.path.exists(ESM2_MODEL_PATH)):
            raise FileNotFoundError(f"未找到本地ESM2权重: {ESM2_MODEL_PATH}")
        model, alphabet = esm.pretrained.load_model_and_alphabet_local(ESM2_MODEL_PATH)
        model.eval()
        device = torch.device('cpu')
        if ESM2_USE_AUTOCUDA and torch.cuda.is_available():
            try:
                _ = torch.cuda.get_device_properties(0)
                device = torch.device('cuda')
                model = model.to(device)
            except Exception as _cuda_e:
                logger.warning(f"CUDA 初始化失败,回退至CPU: {_cuda_e}")
                device = torch.device('cpu')
                model = model.to(device)
        try:
            if device.type == 'cpu':
                if ESM2_CPU_THREADS is not None:
                    import torch as _t
                    _t.set_num_threads(int(ESM2_CPU_THREADS))
                    logger.info(f"ESM2 CPU 线程数设为 {int(ESM2_CPU_THREADS)}")
                if ESM2_USE_QUANTIZATION:
                    import torch.ao.quantization as tq
                    model = tq.quantize_dynamic(
                        model, {torch.nn.Linear}, dtype=torch.qint8
                    )
                    logger.info("已对 ESM2 Linear 层应用动态量化(int8)")
        except Exception as _qe:
            logger.warning(f"CPU 优化未应用: {_qe}")
        batch_converter = alphabet.get_batch_converter()
        _ESM2_MODEL_CACHE.update({
            'initialized': True,
            'model': model,
            'alphabet': alphabet,
            'batch_converter': batch_converter,
            'device': device,
        })
        return model, alphabet, batch_converter, device
    except Exception as e:
        logger.warning(f"ESM2加载失败: {e}")
        return None, None, None, None

def _esm_cache_key(chain_id: str, seq: str):
    """生成ESM2缓存键,优化键生成策略"""
    import hashlib
    max_length = 1024
    seq_truncated = seq[:max_length] if len(seq) > max_length else seq
    h = hashlib.sha256(seq_truncated.encode('utf-8')).hexdigest()[:16]
    safe_chain = ''.join(ch for ch in chain_id if ch.isalnum()) or 'NA'
    return f"{safe_chain}_{h}.pt"

def _esm_cache_load(chain_id: str, seq: str):
    if not ESM2_ENABLE_CACHE:
        return None
    try:
        os.makedirs(ESM2_CACHE_DIR, exist_ok=True)
        path = os.path.join(ESM2_CACHE_DIR, _esm_cache_key(chain_id, seq))
        if os.path.exists(path):
            arr = torch.load(path, map_location='cpu',weights_only=False)
            if isinstance(arr, torch.Tensor):
                return arr.detach().cpu().numpy()
            return arr
    except Exception:
        pass
    return None

def _esm_cache_save(chain_id: str, seq: str, rep_np):
    if not ESM2_ENABLE_CACHE:
        return
    try:
        os.makedirs(ESM2_CACHE_DIR, exist_ok=True)
        path = os.path.join(ESM2_CACHE_DIR, _esm_cache_key(chain_id, seq))
        # 存成 tensor 以便高效加载
        torch.save(torch.from_numpy(rep_np), path)
    except Exception:
        pass


def get_esm2_embeddings(sequences, model_name=None):
    model, alphabet, batch_converter, device = _load_esm2_model(model_name)
    if model is None:
        return {}

    to_compute = []  # list of (cid, seq)
    results = {}
    for cid, seq in sequences.items():
        if seq is None:
            continue
        s = str(seq).strip()
        if len(s) == 0:
            continue
        if ESM2_ONLY_ANTIBODY:
            if cid not in ANTIBODY_CHAINS:
                continue
        cached = _esm_cache_load(cid, s)
        if cached is not None:
            results[cid] = cached
        else:
            to_compute.append((cid, s))

    if not to_compute:
        return results
    if ESM2_SORT_BY_LENGTH and len(to_compute) > 1:
        to_compute.sort(key=lambda x: len(x[1]), reverse=True)

    model_layer = 33
    model_return_contacts = False
    batch_size = max(1, int(ESM2_BATCH_SIZE))
    with torch.no_grad():
        for start in range(0, len(to_compute), batch_size):
            chunk = to_compute[start:start+batch_size]
            _, batch_strs, batch_tokens = batch_converter(chunk)
            batch_tokens = batch_tokens.to(device)
            try:
                if ESM2_USE_AMP and device.type == 'cuda':
                    with torch.cuda.amp.autocast():
                        out = model(batch_tokens, repr_layers=[model_layer], return_contacts=model_return_contacts)
                else:
                    out = model(batch_tokens, repr_layers=[model_layer], return_contacts=model_return_contacts)
            except Exception as e:
                # 若 CUDA 失败,自动回退 CPU 重算本批次
                logger.warning(f"ESM2 前向在 {device} 失败,回退CPU: {e}")
                cpu_device = torch.device('cpu')
                _ESM2_MODEL_CACHE['model'] = _ESM2_MODEL_CACHE['model'].to(cpu_device)
                _ESM2_MODEL_CACHE['device'] = cpu_device
                device = cpu_device
                batch_tokens = batch_tokens.to(cpu_device)
                out = _ESM2_MODEL_CACHE['model'](batch_tokens, repr_layers=[model_layer], return_contacts=model_return_contacts)
            reps = out["representations"][model_layer]
            for i, (cid, s) in enumerate(chunk):
                rep = reps[i, 1:-1].detach().cpu().numpy()
                results[cid] = rep
                _esm_cache_save(cid, s, rep)

    return results

def parse_residue_number(res_number):
    res_num_str = str(res_number).strip()
    match = re.match(r'(\d+)([A-Za-z]*)', res_num_str)
    
    if match:
        base_number = int(match.group(1))  # 基础编号
        insertion_code = match.group(2).upper()  # 插入代码(转换为大写)
        if not insertion_code:
            return base_number
        else:
            insertion_offset = ord(insertion_code) - ord('A') + 1
            actual_number = base_number + insertion_offset
            return actual_number
    else:
        logger.warning(f"无法解析残基编号: {res_num_str},使用0")
        return 0

def _calculate_hbond_properties(atom, residue):
    atom_name = atom.get_name().strip()
    element = atom.element.strip()
    residue_name = residue.get_resname().strip()
    is_acceptor = False
    is_donor = False
    if element == 'O':
        is_acceptor = True

    if residue_name in ['HIS', 'HID', 'HIE', 'HIP'] and atom_name in ['ND1', 'NE2']:
        if residue_name == 'HIP':
            his_acceptor = False
        else:
            his_acceptor = (len(find_attached_hydrogens(atom)) == 0)
        is_acceptor = is_acceptor or his_acceptor
    has_H = len(find_attached_hydrogens(atom)) > 0
    if atom_name == 'N' and residue_name != 'PRO' and has_H:
        is_donor = True
    elif has_H:
        if residue_name == 'SER' and atom_name == 'OG':
            is_donor = True
        elif residue_name == 'THR' and atom_name == 'OG1':
            is_donor = True
        elif residue_name == 'TYR' and atom_name == 'OH':
            is_donor = True
        elif residue_name == 'ASN' and atom_name == 'ND2':
            is_donor = True
        elif residue_name == 'GLN' and atom_name == 'NE2':
            is_donor = True
        elif residue_name == 'LYS' and atom_name == 'NZ':
            is_donor = True
        elif residue_name == 'ARG' and atom_name in ['NE', 'NH1', 'NH2']:
            is_donor = True
        elif residue_name in ['HIS', 'HID', 'HIE', 'HIP'] and atom_name in ['ND1', 'NE2']:
            is_donor = True
        elif residue_name == 'TRP' and atom_name == 'NE1':
            is_donor = True

    return is_acceptor, is_donor

def _calculate_charge_properties(atom, residue):
    atom_name = atom.get_name().strip()
    residue_name = residue.get_resname().strip()
    def _is_protonated_histidine(residue_):
        name = residue_.get_resname().strip()
        if name in ['HIP', 'HSP']:
            return True
        present_names = set(a.get_name().strip() for a in residue_.get_atoms())
        for n in ['ND1', 'NE2']:
            if n in present_names:
                a = residue_[n]
                from utils import find_attached_hydrogens
                if len(find_attached_hydrogens(a)) >= 1:
                    return True
        return False

    is_positive = False
    is_negative = False

    if residue_name == 'LYS' and atom_name == 'NZ':
        is_positive = True
    elif residue_name == 'ARG' and atom_name in ['NH1', 'NH2']:
        is_positive = True
    elif residue_name == 'HIS' and atom_name in ['ND1', 'NE2'] and _is_protonated_histidine(residue):
        is_positive = True
    if residue_name == 'ASP' and atom_name in ['OD1', 'OD2']:
        is_negative = True
    elif residue_name == 'GLU' and atom_name in ['OE1', 'OE2']:
        is_negative = True
    if residue_name in ['ASH', 'GLH']:
        is_negative = False
    if atom_name == 'OXT':
        is_negative = True
    if atom_name == 'N':
        from utils import find_attached_hydrogens
        if len(find_attached_hydrogens(atom)) >= 2:
            is_positive = True

    return is_positive, is_negative

def is_hbond_pair(atom1, atom2, residue1, residue2,
                  da_max=3.6, ha_max=2.6, angle_min=120.0):
    """判断原子对是否形成氢键
    重原子距离 D···A ≤ da_max(默认 3.5 Å)
    若供体氢存在：同时要求 H···A ≤ ha_max(默认 2.5 Å)且 ∠D–H…A ≥ angle_min(默认 120°)
    若供体氢不存在：退化判定,仅使用更严格的 D···A ≤ 3.2 Å 近似
    识别分子内和分子间氢键
    """
    is_acceptor1, is_donor1 = _calculate_hbond_properties(atom1, residue1)
    is_acceptor2, is_donor2 = _calculate_hbond_properties(atom2, residue2)
    if is_donor1 and is_acceptor2:
        donor, acceptor, donor_res, acceptor_res = atom1, atom2, residue1, residue2
    elif is_donor2 and is_acceptor1:
        donor, acceptor, donor_res, acceptor_res = atom2, atom1, residue2, residue1
    else:
        return False
    d = np.linalg.norm(donor.get_coord() - acceptor.get_coord())
    if not (d <= da_max):
        return False
    # 角度与 H···A 距离判断(以H为顶点)
    hydrogens = find_attached_hydrogens(donor)
    for h in hydrogens:
        # H···A 距离
        ha = np.linalg.norm(acceptor.get_coord() - h.get_coord())
        if ha > ha_max:
            continue
        # ∠D–H…A
        v1 = donor.get_coord() - h.get_coord()     # H->D
        v2 = acceptor.get_coord() - h.get_coord()  # H->A
        v1_norm = np.linalg.norm(v1)
        v2_norm = np.linalg.norm(v2)
        if v1_norm == 0 or v2_norm == 0:
            continue
        cosine_angle = np.dot(v1, v2) / (v1_norm * v2_norm)
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
        angle = np.degrees(np.arccos(cosine_angle))
        if angle >= angle_min:
            return True
    return False

def is_hbond_pair_directed(atom1, atom2, residue1, residue2,
                           da_max=3.6, ha_max=2.6, angle_min=120.0):
    is_acceptor1, is_donor1 = _calculate_hbond_properties(atom1, residue1)
    is_acceptor2, is_donor2 = _calculate_hbond_properties(atom2, residue2)

    if is_donor1 and is_acceptor2:
        donor, acceptor = atom1, atom2
        direction = 1
    elif is_donor2 and is_acceptor1:
        donor, acceptor = atom2, atom1
        direction = -1
    else:
        return False, 0
    d = np.linalg.norm(donor.get_coord() - acceptor.get_coord())
    if not (d <= da_max):
        return False, 0
    
    # 角度与 H···A 距离判断
    hydrogens = find_attached_hydrogens(donor)
    for h in hydrogens:
        ha = np.linalg.norm(acceptor.get_coord() - h.get_coord())
        if ha > ha_max:
            continue
        # 以H为顶点的角(与 is_hbond_pair 一致)
        v1 = donor.get_coord() - h.get_coord()     # H->D
        v2 = acceptor.get_coord() - h.get_coord()  # H->A
        v1n = np.linalg.norm(v1); v2n = np.linalg.norm(v2)
        if v1n == 0 or v2n == 0:
            continue
        cosine_angle = np.dot(v1, v2) / (v1n * v2n)
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
        angle = np.degrees(np.arccos(cosine_angle))
        if angle >= angle_min:
            return True, direction

    return False, 0

def is_ionic_pair(atom1, atom2, residue1, residue2):
    is_positive1, is_negative1 = _calculate_charge_properties(atom1, residue1)
    is_positive2, is_negative2 = _calculate_charge_properties(atom2, residue2)

    return (is_positive1 and is_negative2) or (is_positive2 and is_negative1)

def is_ionic_pair_directed(atom1, atom2, residue1, residue2):

    is_positive1, is_negative1 = _calculate_charge_properties(atom1, residue1)
    is_positive2, is_negative2 = _calculate_charge_properties(atom2, residue2)
    
    if is_positive1 and is_negative2:
        return True, 1
    elif is_positive2 and is_negative1:
        return True, -1
    
    return False, 0

def get_atom_features(atom, residue, atom_asa_dict=None, 
                     use_atom_type=True, use_hbond=True, use_charge=True, 
                     use_hydrophobic=True, use_asa=True,
                     use_residue_type=True, use_atom_name=True,
                     pssm_features=None, residue_to_seq_idx=None):
    """获取原子特征向量
    
    参数:
        atom: 原子对象
        residue: 残基对象
        atom_asa_dict: ASA字典
        use_atom_type: 是否使用原子类型特征
        use_hbond: 是否使用氢键特征
        use_charge: 是否使用电荷特征
        use_hydrophobic: 是否使用疏水性特征
        use_asa: 是否使用ASA特征
        use_residue_type: 是否使用残基类型特征
        use_atom_name: 是否使用原子名称特征
        pssm_features: PSSM特征字典
        residue_to_seq_idx: 残基到序列索引映射
    """
    features = []
    # 原子名称和元素
    atom_name = atom.get_name()
    element = atom.element.strip()
    
    # 1. 原子类型 (one-hot)
    if use_atom_type:
        atom_type_vec = np.zeros(len(ATOM_TYPES))
        atom_type_idx = atom_type_to_idx.get(element, len(ATOM_TYPES) - 1)
        atom_type_vec[atom_type_idx] = 1
        features.extend(atom_type_vec)
    
    # 2. 原子名称 (one-hot)
    if use_atom_name:
        atom_name_vec = np.zeros(len(ATOM_NAMES))
        atom_name_idx = atom_name_to_idx.get(atom_name, len(ATOM_NAMES) - 1)
        atom_name_vec[atom_name_idx] = 1
        features.extend(atom_name_vec)
    
    # 3. 残基PSSM特征 (替代残基one-hot)
    if use_residue_type:
        if pssm_features is not None:
            # 使用PSSM特征
            pssm_feature = get_residue_pssm_features(residue, pssm_features, residue_to_seq_idx)
            features.extend(pssm_feature)
        else:
            # 回退到残基类型one-hot
            residue_name = residue.get_resname().strip()
            residue_type_vec = np.zeros(len(AMINO_ACIDS))
            if residue_name in amino_acid_to_idx:
                residue_type_vec[amino_acid_to_idx[residue_name]] = 1
            else:
                # 对于非标准氨基酸,使用平均值
                residue_type_vec[:] = 1 / len(AMINO_ACIDS)
            features.extend(residue_type_vec)
    
    # 4. 氢键受供体 (2维one-hot)
    if use_hbond:
        hbond_feature = [0, 0]  # [受体, 供体]
        is_acceptor, is_donor = _calculate_hbond_properties(atom, residue)
        
        # 设置特征向量
        if is_acceptor:
            hbond_feature[0] = 1
        if is_donor:
            hbond_feature[1] = 1
        features.extend(hbond_feature)
    
    # 5. 带正负电基团 (one-hot)
    if use_charge:
        charge_feature = [0, 0, 0]
        is_positive, is_negative = _calculate_charge_properties(atom, residue)
        
        if is_positive:
            charge_feature[1] = 1  # 带正电
        elif is_negative:
            charge_feature[2] = 1  # 带负电
        else:
            charge_feature[0] = 1  # 中性
        features.extend(charge_feature)
    
    # 6. 疏水性特征 (2维one-hot: [非疏水, 疏水])
    if use_hydrophobic:
        hydrophobic_feature = [0, 0]
        
        # 主链原子永远不疏水
        if atom_name in ['N', 'CA', 'C', 'O']:
            hydrophobic_feature[0] = 1  # 非疏水
        else:
            if element in ['C', 'S']:
                # 疏水氨基酸的侧链原子为疏水
                if is_hydrophobic(residue.get_resname()):
                    hydrophobic_feature[1] = 1  # 疏水
            else:
                hydrophobic_feature[0] = 1  # 非疏水
        
        features.extend(hydrophobic_feature)

    # 6.1 B 因子与占有率(实验置信度与构象异质性)
    try:
        bfactor = float(atom.get_bfactor())
    except Exception:
        bfactor = 0.0
    try:
        occupancy = atom.get_occupancy()
        occupancy = 1.0 if occupancy is None else float(occupancy)
    except Exception:
        occupancy = 1.0
    # 轻量归一化：B 因子用 tanh 压缩到 [-1,1] 再映射到 [-1,1](直接用 tanh 输出)
    # 占有率裁剪到 [0,1]
    bfactor_feat = np.tanh(bfactor / 50.0)
    occupancy_feat = float(min(max(occupancy, 0.0), 1.0))
    features.extend([bfactor_feat, occupancy_feat])

    # 7. 原子 rASA 特征 (1维: 相对溶剂可及面积)
    if use_asa:
        if atom_asa_dict is not None:
            asa_feature = get_atom_asa_feature(atom, residue, atom_asa_dict)
        else:
            asa_feature = 0.0
        features.append(asa_feature)

    # 8. 与CA原子的RBF展开距离特征
    # 某些残基可能没有CA原子（如特殊残基或结构不完整），需要处理
    try:
        ca_atom = residue['CA']
        atom_coord = atom.get_coord()
        ca_coord = ca_atom.get_coord()
        distance = np.linalg.norm(atom_coord - ca_coord)
        
        # RBF展开 (使用16个高斯函数)
        rbf_features = []
        for i in range(16):
            mu = i * 0.5  # 从0到7.5,步长0.5
            sigma = 0.5
            rbf_val = np.exp(-((distance - mu) ** 2) / (2 * sigma ** 2))
            rbf_features.append(rbf_val)
        features.extend(rbf_features)
    except (KeyError, AttributeError):
        # 如果没有CA原子，使用默认值（距离为0的RBF特征）
        rbf_features = [1.0] + [0.0] * 15  # 第一个RBF为1（距离=0），其他为0
        features.extend(rbf_features)

    return np.array(features)

def get_residue_features(residue_name, residue=None, atom_asa_dict=None, pssm_features=None, residue_to_seq_idx=None,
                        use_aa_type=True, use_polar=True, use_aromatic=True, use_hydrophobic=True, 
                        use_pssm=True,
                        esm2_reps=None, use_esm2=False,
                        use_dssp=False, dssp_cache=None):
    """获取残基特征向量
    
    参数:
        residue_name: 残基名称
        residue: 残基对象
        atom_asa_dict: ASA字典(已弃用,保留以兼容旧代码)
        pssm_features: PSSM特征
        residue_to_seq_idx: 残基到序列索引映射
        use_aa_type: 是否使用氨基酸类型特征
        use_polar: 是否使用极性特征
        use_aromatic: 是否使用芳香族特征
        use_hydrophobic: 是否使用疏水性特征
        use_pssm: 是否使用PSSM特征
        esm2_reps: ESM2 per-residue embeddings dict(chain_id -> [L, D])
        use_esm2: 是否使用ESM2嵌入(优先级高于PSSM)
        use_dssp: 是否使用DSSP特征
        dssp_cache: DSSP缓存字典
    """
    features = []
    
    # 1. 氨基酸类型 (one-hot)
    if use_aa_type:
        aa_vec = np.zeros(len(AMINO_ACIDS))
        if residue_name in amino_acid_to_idx:
            aa_vec[amino_acid_to_idx[residue_name]] = 1
        else:
            # 对于非标准氨基酸,使用平均值
            aa_vec[:] = 1 / len(AMINO_ACIDS)
        features.extend(aa_vec)
    
    # 2. 极性 (one-hot)
    if use_polar:
        polar_vec = [0, 0]
        polar_vec[1 if is_polar(residue_name) else 0] = 1
        features.extend(polar_vec)
    
    # 3. 芳香族 (one-hot)
    if use_aromatic:
        aromatic_vec = [0, 0]
        aromatic_vec[1 if is_aromatic(residue_name) else 0] = 1
        features.extend(aromatic_vec)
    
    # 4. 疏水性 (one-hot)
    if use_hydrophobic:
        hydrophobic_vec = [0, 0]
        hydrophobic_vec[1 if is_hydrophobic(residue_name) else 0] = 1
        features.extend(hydrophobic_vec)

    # 5. DSSP 残基特征 (17维: SS8 onehot(8) + rASA(1) + sin/cos(phi/psi)(4) + HB energies(4))
    # 注：DSSP已经包含了rASA和主链二面角(phi/psi),所以不需要单独计算残基ASA和二面角特征
    if use_dssp and residue is not None and dssp_cache:
        chain_id = residue.get_parent().get_id()
        res_id = residue.get_id()            # (' ', resseq, icode)
        if chain_id in dssp_cache and res_id in dssp_cache[chain_id]:
            dssp_vec = dssp_cache[chain_id][res_id]
            features.extend(dssp_vec.tolist())         # 8+1+4+4 = 17
        else:
            features.extend([0.0] * 17)

    # 6. 序列嵌入：优先使用 ESM2,否则可退回 PSSM
    if use_esm2:
        if esm2_reps is not None and residue is not None:
            chain_id = residue.get_parent().get_id()
            resnum = parse_residue_number(residue.get_id()[1])
            chain_rep = esm2_reps.get(chain_id)
            if chain_rep is not None and len(chain_rep) > 0:
                # 使用 residue_to_seq_idx 对齐序列索引
                if residue_to_seq_idx is not None and chain_id in residue_to_seq_idx:
                    seq_map = residue_to_seq_idx[chain_id]
                    if resnum in seq_map:
                        si = seq_map[resnum]
                        if 0 <= si < chain_rep.shape[0]:
                            features.extend(chain_rep[si].tolist())
                        else:
                            features.extend([0.0] * int(chain_rep.shape[1]))
                    else:
                        features.extend([0.0] * int(chain_rep.shape[1]))
                else:
                    # 无映射时按最小编号对齐
                    chain = residue.get_parent()
                    min_id = min([parse_residue_number(r.get_id()[1]) for r in chain])
                    si = resnum - min_id
                    if 0 <= si < chain_rep.shape[0]:
                        features.extend(chain_rep[si].tolist())
                    else:
                        features.extend([0.0] * int(chain_rep.shape[1]))
            else:
                features.extend([0.0] * 1280)
        else:
            features.extend([0.0] * 1280)
    elif use_pssm:
        if pssm_features is not None:
            pssm_feature = get_residue_pssm_features(residue, pssm_features, residue_to_seq_idx)
        else:
            pssm_feature = np.zeros(20)
        features.extend(pssm_feature)
    
    # 7. 残基位置归一化特征
    try:
        pos_feat = 0.0
        if residue is not None and residue_to_seq_idx is not None:
            chain_id = residue.get_parent().get_id()
            if chain_id in residue_to_seq_idx:
                seq_map = residue_to_seq_idx[chain_id]
                seq_idx = None
                res_id_tuple = residue.get_id()  # (' ', resseq, icode)
                resseq = parse_residue_number(res_id_tuple[1])
                icode = res_id_tuple[2]
                # 兼容键型
                if len(seq_map) > 0:
                    sample_key = next(iter(seq_map.keys()))
                    if isinstance(sample_key, tuple):
                        seq_idx = seq_map.get((resseq, icode)) or seq_map.get((resseq, ' ')) or seq_map.get((resseq, ''))
                    else:
                        seq_idx = seq_map.get(resseq)
                if seq_idx is not None:
                    try:
                        chain_len = max(seq_map.values()) + 1 if len(seq_map) > 0 else 1
                    except Exception:
                        chain_len = max(len(seq_map), 1)
                    if chain_len <= 0:
                        chain_len = 1
                    pos_feat = float((int(seq_idx) + 1) / float(chain_len))
        features.append(pos_feat)
    except Exception:
        features.append(0.0)
    
    return np.array(features)

def get_atom_edge_features(atom1, atom2, residue1, residue2, distance, is_covalent=False):
    """获取原子边特征 - 分为标量特征和向量特征
    
    标量特征: [相互作用类型(2), 边类型(2), 作用类型(4), 距离RBF(16), 键角(1), 二面角(1)]
    向量特征: [方向单位向量(3)]
    """
    # 标量特征
    
    # 1. 相互作用类型 (2维): [同链, 跨链] - 放到最前面
    chain1 = atom1.get_parent().get_parent().get_id()
    chain2 = atom2.get_parent().get_parent().get_id()
    is_same_chain = (chain1 == chain2)
    interaction_type = [0, 0]
    if is_same_chain:
        interaction_type[0] = 1  # 同链
    else:
        interaction_type[1] = 1  # 跨链
    
    # 2. 边类型标记 (2维): [共价, 非共价]
    edge_type = [0, 0]
    if is_covalent:
        edge_type[0] = 1  # 共价键
    else:
        edge_type[1] = 1  # 非共价键
    
    # 3. 作用类型标记 (4维): [氢键, 离子键, 疏水性, 范德华力]
    bond_type = [0, 0, 0, 0]
    is_hbond = is_hbond_pair(atom1, atom2, residue1, residue2)
    is_ionic = is_ionic_pair(atom1, atom2, residue1, residue2)
    
    if is_hbond and distance <= 3.5:
        bond_type[0] = 1  # 氢键
    elif is_ionic and distance <= 5.0:
        bond_type[1] = 1  # 离子键
    else:
        # 检查疏水性和范德华力
        from utils import is_hydrophobic_interaction, is_vdw_interaction
        if is_hydrophobic_interaction(atom1, atom2, residue1, residue2):
            bond_type[2] = 1  # 疏水性
        elif is_vdw_interaction(atom1, atom2, epsilon=0.5, lower_tolerance_ratio=0.1, heavy_only=True):
            bond_type[3] = 1  # 范德华力
    
    # 4. 距离RBF展开 (16维)
    rbf_features = []
    for i in range(16):
        mu = i * 0.5  # 从0到7.5,步长0.5
        sigma = 0.5
        rbf_val = np.exp(-((distance - mu) ** 2) / (2 * sigma ** 2))
        rbf_features.append(rbf_val)
    
    # 5. 键角和二面角 (2维)
    if is_covalent:
        bond_angle = 0.0
        dihedral_angle = 0.0
    else:
        bond_angle, dihedral_angle = calculate_edge_angles(atom1, atom2, residue1, residue2)
    
    # 标量特征组合(相互作用类型放到最前面)
    scalar_features = interaction_type + edge_type + bond_type + rbf_features + [bond_angle, dihedral_angle]
    
    # 向量特征
    coord1 = atom1.get_coord()
    coord2 = atom2.get_coord()
    relative_position = coord2 - coord1
    distance_magnitude = np.linalg.norm(relative_position)
    epsilon = 1e-8
    direction_unit_vector = relative_position / (distance_magnitude + epsilon)
    
    return np.array(scalar_features, dtype=np.float32), np.array(direction_unit_vector, dtype=np.float32)

def get_residue_edge_features(atom1, atom2, residue1, residue2, distance, is_covalent=False, edge_type_hint="radius"):
    """获取残基边特征 - 分为标量特征和向量特征
    
    标量特征: [相互作用类型(2), 边类型(2), 距离RBF(16), 键角(1), 二面角(1)]
    向量特征: [方向单位向量(3)]
    
    参数:
        edge_type_hint: 边类型提示,可选 "radius", "sequence"
    """
    # 标量特征
    
    # 1. 相互作用类型 (2维): [同链, 跨链] - 放到最前面
    chain1 = atom1.get_parent().get_parent().get_id()
    chain2 = atom2.get_parent().get_parent().get_id()
    is_same_chain = (chain1 == chain2)
    interaction_type = [0, 0]
    if is_same_chain:
        interaction_type[0] = 1  # 同链
    else:
        interaction_type[1] = 1  # 跨链
    
    # 2. 边类型标记 (2维): [半径边, 序列边]
    edge_type = [0, 0]
    if edge_type_hint == "radius":
        edge_type[0] = 1  # 半径边
    elif edge_type_hint == "sequence":
        edge_type[1] = 1  # 序列边
    else:
        edge_type[0] = 1  # 默认为半径边
    
    # 3. 距离RBF展开 (16维)
    rbf_features = []
    for i in range(16):
        mu = i * 0.5  # 从0到7.5,步长0.5
        sigma = 0.5
        rbf_val = np.exp(-((distance - mu) ** 2) / (2 * sigma ** 2))
        rbf_features.append(rbf_val)
    
    # 4. 键角和二面角 (2维)
    if is_covalent:
        bond_angle = 0.0
        dihedral_angle = 0.0
    else:
        bond_angle, dihedral_angle = calculate_edge_angles(atom1, atom2, residue1, residue2)
    
    # 标量特征组合(相互作用类型放到最前面)
    scalar_features = interaction_type + edge_type + rbf_features + [bond_angle, dihedral_angle]
    
    # 向量特征
    coord1 = atom1.get_coord()
    coord2 = atom2.get_coord()
    relative_position = coord2 - coord1
    distance_magnitude = np.linalg.norm(relative_position)
    epsilon = 1e-8
    direction_unit_vector = relative_position / (distance_magnitude + epsilon)
    
    return np.array(scalar_features, dtype=np.float32), np.array(direction_unit_vector, dtype=np.float32)

# ASA相关函数
def compute_atom_asa(pdb_file):

    if not FREESASA_AVAILABLE:
        return {}
    try:
        structure = freesasa.Structure(pdb_file)
        result = freesasa.calc(structure)
        abs_asa = {}
        max_asa_by_type = {}  # (resname, atom_name) -> max ASA
        for i in range(structure.nAtoms()):
            atom_name = structure.atomName(i).strip()
            res_name = structure.residueName(i).strip()
            res_number = structure.residueNumber(i)
            chain = structure.chainLabel(i)
            asa = float(result.atomArea(i))
            res_num_int = parse_residue_number(res_number)
            key = (chain, res_num_int, res_name, atom_name)
            abs_asa[key] = asa
            type_key = (res_name, atom_name)
            prev = max_asa_by_type.get(type_key, 0.0)
            if asa > prev:
                max_asa_by_type[type_key] = asa

        rasa_dict = {}
        for key, asa in abs_asa.items():
            _, _, res_name, atom_name = key
            denom = max(max_asa_by_type.get((res_name, atom_name), 0.0), 1e-6)
            rasa = float(min(max(asa / denom, 0.0), 1.0))
            rasa_dict[key] = rasa
        return rasa_dict
        
    except Exception as e:
        logger.warning(f"计算ASA时出错: {e}")
        return {}

def get_atom_asa_feature(atom, residue, atom_asa_dict):

    if not FREESASA_AVAILABLE or not atom_asa_dict:
        return 0.0
    key = (atom.get_parent().get_parent().get_id(), 
           parse_residue_number(residue.get_id()[1]), 
           residue.get_resname(), 
           atom.get_name().strip())
    asa_value = atom_asa_dict.get(key, 0.0)
    
    return asa_value

def get_residue_asa_feature(residue, atom_asa_dict):
    if not FREESASA_AVAILABLE or not atom_asa_dict or residue is None:
        return 0.0
    total_asa = 0.0
    atom_count = 0
    
    for atom in residue:
        if atom.element == 'H':
            continue 
        key = (atom.get_parent().get_parent().get_id(), 
               parse_residue_number(residue.get_id()[1]), 
               residue.get_resname(), 
               atom.get_name().strip())
        asa_value = atom_asa_dict.get(key, 0.0)
        total_asa += asa_value
        atom_count += 1
    
    return total_asa / max(atom_count, 1)

# 全局变量用于缓存数据库检查结果
_DB_CHECKED = False
_DB_EXISTS = False

# PSSM缓存
_PSSM_CACHE = {}

# PSSM预计算缓存配置
PSSM_CACHE_DIR = "/root/AbMSPN/Data/pssm_cache"
PSSM_INDEX_FILE = os.path.join(PSSM_CACHE_DIR, "pssm_index.pkl")

def _check_blast_database():
    """检查BLAST数据库是否存在(支持分卷数据库)"""
    global _DB_CHECKED, _DB_EXISTS
    if _DB_CHECKED:
        return _DB_EXISTS
    
    # 首先检查是否是分卷数据库
    db_dir = os.path.dirname(SWISSPROT_DB_PATH)
    db_base = os.path.basename(SWISSPROT_DB_PATH)
    
    # 检查分卷数据库(uniref90.00.phr, uniref90.01.phr等)
    volume_files = []
    for i in range(100):  # 最多检查100个分卷
        volume_phr = os.path.join(db_dir, f"{db_base}.{i:02d}.phr")
        if os.path.exists(volume_phr):
            volume_files.append(i)
        else:
            break
    
    if volume_files:
        # 发现分卷数据库
        logger.info(f"发现分卷BLAST数据库: {SWISSPROT_DB_PATH} (共{len(volume_files)}个分卷)")
        _DB_EXISTS = True
    else:
        # 检查标准数据库文件
        db_files = [f"{SWISSPROT_DB_PATH}.phr", f"{SWISSPROT_DB_PATH}.pin", f"{SWISSPROT_DB_PATH}.psq"]
        _DB_EXISTS = all(os.path.exists(f) for f in db_files)
        
        if _DB_EXISTS:
            logger.info(f"发现标准BLAST数据库: {SWISSPROT_DB_PATH}")
        else:
            logger.info(f"BLAST数据库不存在: {SWISSPROT_DB_PATH}")
    
    _DB_CHECKED = True
    return _DB_EXISTS

def _load_pssm_index():
    """加载PSSM索引文件"""
    if os.path.exists(PSSM_INDEX_FILE):
        try:
            with open(PSSM_INDEX_FILE, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.warning(f"加载PSSM索引失败: {e}")
            return {}
    return {}

def _get_sequence_hash(sequence):
    """获取序列的哈希值,用于唯一标识"""
    import hashlib
    return hashlib.md5(sequence.encode()).hexdigest()

def _get_pdb_chain_key(pdb_name, chain_id):
    """获取PDB链的唯一标识"""
    return f"{pdb_name}_{chain_id}"

def _get_pssm_from_cache(pdb_name, chain_id):
    """从缓存中获取PSSM矩阵 - 简化版本
    直接使用 pdb_name + "_" + chain_id + ".npy" 格式查询
    支持大小写不敏感匹配
    """
    # 转换为大写以匹配索引格式，同时为 WT_ 前缀准备别名，便于复用突变体缓存
    pdb_name_upper = pdb_name.upper()
    chain_id_upper = chain_id.upper()
    pdb_name_aliases = [pdb_name_upper]
    if pdb_name_upper.startswith("WT_"):
        pdb_name_aliases.append(pdb_name_upper[3:])  # 去掉 WT_ 前缀，匹配突变体缓存命名
    # 尝试多种文件名格式（大小写变体）
    filename_variants = [
        f"{pdb_name_upper}_{chain_id_upper}.npy",  # 全大写
        f"{pdb_name}_{chain_id}.npy",  # 原始格式（可能包含大小写）
        f"{pdb_name.upper()}_{chain_id.upper()}.npy",  # 明确转换
        f"{pdb_name.lower()}_{chain_id.lower()}.npy",  # 全小写
    ]
    # 如果原始pdb_name和chain_id已经是大写，避免重复添加
    if pdb_name != pdb_name_upper:
        filename_variants.append(f"{pdb_name}_{chain_id}.npy")
    
    for filename in filename_variants:
        cache_file = os.path.join(PSSM_CACHE_DIR, filename)
        if os.path.exists(cache_file):
            try:
                # 检查是否是符号链接,如果是则检查目标文件是否存在
                if os.path.islink(cache_file):
                    target_file = os.readlink(cache_file)
                    # 如果是相对路径,转换为绝对路径
                    if not os.path.isabs(target_file):
                        target_file = os.path.join(PSSM_CACHE_DIR, target_file)
                    if not os.path.exists(target_file):
                        logger.warning(f"符号链接目标文件不存在: {cache_file} -> {target_file}")
                        continue
                
                logger.info(f"直接读取PSSM缓存文件: {cache_file}")
                return np.load(cache_file)
            except Exception as e:
                logger.warning(f"直接加载PSSM文件失败: {cache_file}, 错误: {e}")
                continue
    
    # 如果直接读取失败，尝试遍历目录查找匹配的文件（大小写不敏感）
    if os.path.exists(PSSM_CACHE_DIR):
        target_pattern_lower = f"{pdb_name_upper}_{chain_id_upper}.npy".lower()
        for file in os.listdir(PSSM_CACHE_DIR):
            if file.lower() == target_pattern_lower and file.endswith('.npy'):
                cache_file = os.path.join(PSSM_CACHE_DIR, file)
                try:
                    logger.info(f"通过大小写不敏感匹配找到PSSM文件: {cache_file}")
                    return np.load(cache_file)
                except Exception as e:
                    logger.warning(f"加载PSSM文件失败: {cache_file}, 错误: {e}")
                    break
    
    # 如果直接读取失败,尝试通过索引文件读取
    pssm_index = _load_pssm_index()
    # 尝试大小写匹配的键
    pdb_chain_key = _get_pdb_chain_key(pdb_name_upper, chain_id_upper)
    
    if pdb_chain_key in pssm_index:
        pssm_file = pssm_index[pdb_chain_key]['pssm_file']
        if os.path.exists(pssm_file):
            try:
                logger.info(f"通过索引读取PSSM缓存文件: {pssm_file}")
                return np.load(pssm_file)
            except Exception as e:
                logger.warning(f"通过索引加载PSSM文件失败: {pssm_file}, 错误: {e}")
                return None
        else:
            logger.warning(f"索引中的PSSM文件不存在: {pssm_file}")
            return None
    
    # 如果索引中找不到，尝试大小写不敏感搜索
    pdb_chain_key_lower = f"{pdb_name_upper}_{chain_id_upper}".lower()
    for key in pssm_index.keys():
        if key.lower() == pdb_chain_key_lower:
            pssm_file = pssm_index[key]['pssm_file']
            if os.path.exists(pssm_file):
                try:
                    logger.info(f"通过大小写不敏感匹配找到PSSM: {key} -> {pssm_file}")
                    return np.load(pssm_file)
                except Exception as e:
                    logger.warning(f"加载PSSM文件失败: {pssm_file}, 错误: {e}")
                    return None
    
    logger.debug(f"PDB链未在缓存中找到: {pdb_name_upper}_{chain_id_upper}")
    return None

# PSSM相关函数
def get_pssm_features(sequences, pdb_file, blast_threads=None):
    """获取PSSM特征
    使用本地SwissProt数据库和PSI-BLAST计算PSSM矩阵特征
    只计算抗体和抗原链的PSSM特征
    
    Args:
        sequences: 序列字典 {chain_id: sequence}
        pdb_file: PDB文件路径
        blast_threads: BLAST使用的线程数,如果为None则使用默认值18
    """
    if not PSSM_AVAILABLE:
        logger.warning("Bio.Blast未安装,跳过PSSM特征提取")
        return {}
    
    if not ENABLE_PSSM:
        logger.info("PSSM计算已禁用,跳过PSSM特征提取")
        return {}
    
    pdb_name = os.path.splitext(os.path.basename(pdb_file))[0]
    
    if '_' in pdb_name:
        parts = pdb_name.split('_')
        if len(parts[-1]) == 1 and parts[-1].isalpha():
            pdb_name = '_'.join(parts[:-1])
    
    if pdb_name.endswith('_withH'):
        pdb_name = pdb_name[:-6]
    
    if '_Repair_' in pdb_name:
        pass
    elif '_Repair' in pdb_name and not pdb_name.endswith('_Repair'):
        pdb_name = pdb_name.rsplit('_', 1)[0]
    else:
        pass
    
    # 转换为大写以匹配PSSM缓存索引格式
    pdb_name = pdb_name.upper()
    
    pssm_features = {}
    
    for chain_id, sequence in sequences.items():
        if len(sequence.strip()) == 0:
            logger.warning(f"链 {chain_id} 序列为空,跳过PSSM计算")
            continue
        cached_pssm = _get_pssm_from_cache(pdb_name, chain_id)
        if cached_pssm is not None:
            logger.info(f"使用预计算缓存的PSSM结果: {pdb_name}_{chain_id}, 形状={cached_pssm.shape}")
            pssm_features[chain_id] = cached_pssm
            continue

        sequence_hash = hash(sequence)
        if sequence_hash in _PSSM_CACHE:
            logger.debug(f"使用内存缓存的PSSM结果: 链{chain_id}")
            pssm_features[chain_id] = _PSSM_CACHE[sequence_hash]
            continue
            
        temp_files = []
        try:
            if not _check_blast_database():
                logger.info(f"BLAST数据库不存在,开始构建: {SWISSPROT_DB_PATH}")
                if not os.path.exists(SWISSPROT_FASTA_PATH):
                    logger.error(f"FASTA文件不存在: {SWISSPROT_FASTA_PATH}")
                    raise FileNotFoundError(f"FASTA文件不存在: {SWISSPROT_FASTA_PATH}")
                subprocess.run([
                    "makeblastdb", 
                    "-in", SWISSPROT_FASTA_PATH, 
                    "-dbtype", "prot", 
                    "-out", SWISSPROT_DB_PATH
                ], check=True)
                logger.info(f"BLAST数据库构建完成: {SWISSPROT_DB_PATH}")
                # 更新全局状态
                global _DB_EXISTS
                _DB_EXISTS = True
            else:
                logger.debug(f"使用现有BLAST数据库: {SWISSPROT_DB_PATH}")
            
            # 创建唯一的临时文件名(避免多进程冲突)
            unique_id = str(uuid.uuid4())[:8]
            temp_fasta = f"/tmp/{chain_id}_{unique_id}_temp.fasta"
            temp_files.append(temp_fasta)
            with open(temp_fasta, 'w') as f:
                f.write(f">{chain_id}\n{sequence}\n")
            
            # 运行PSI-BLAST生成PSSM矩阵
            temp_pssm = f"/tmp/{chain_id}_{unique_id}_pssm.pssm"
            temp_out = f"/tmp/{chain_id}_{unique_id}_psiblast.out"
            temp_files.extend([temp_pssm, temp_out])
            
            # 确定BLAST线程数
            if blast_threads is None:
                blast_threads = 18  # 默认值
            else:
                blast_threads = int(blast_threads)
            
            logger.info(f"运行PSI-BLAST: 链{chain_id}, 序列长度{len(sequence)}, 线程数: {blast_threads}")
            
            subprocess.run([
                "psiblast",
                "-query", temp_fasta,
                "-db", SWISSPROT_DB_PATH,
                "-num_iterations", "2",  # 减少迭代次数从3到1
                "-evalue", "0.01",       # 放宽evalue阈值从0.001到0.01
                "-out_ascii_pssm", temp_pssm,
                "-out", temp_out,
                "-num_threads", str(blast_threads),  # 使用传入的线程数
                "-matrix", "BLOSUM62",    # 使用BLOSUM62替换矩阵
                "-gapopen", "11",         # BLOSUM62推荐的缺口开放罚分
                "-gapextend", "1",        # BLOSUM62推荐的缺口扩展罚分
                "-comp_based_stats", "0", # 禁用基于组成的统计
                "-max_target_seqs", "100", # 限制目标序列数量
                "-word_size", "3",        # 使用更小的word size
            ], check=True)
            
            # 解析PSSM矩阵
            pssm = parse_pssm(temp_pssm)
            pssm_features[chain_id] = pssm
            
            # 缓存结果
            _PSSM_CACHE[sequence_hash] = pssm
            
            logger.info(f"成功计算链 {chain_id} 的PSSM特征")
            
        except Exception as e:
            logger.warning(f"计算链 {chain_id} 的PSSM特征失败: {e}")
            # 如果PSSM计算失败,使用零矩阵
            pssm_features[chain_id] = np.zeros((len(sequence), 20))
        finally:
            # 清理所有临时文件
            for temp_file in temp_files:
                try:
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                        logger.debug(f"删除临时文件: {temp_file}")
                except Exception as e:
                    logger.debug(f"清理临时文件 {temp_file} 失败: {e}")
    
    return pssm_features


def parse_pssm(pssm_file):
    """解析PSI-BLAST生成的PSSM矩阵文件"""
    with open(pssm_file) as f:
        lines = f.readlines()

    # 找到PSSM矩阵起始行(跳过注释)
    header_idx = None
    for i, line in enumerate(lines):
        if line.startswith("  ") and "A" in line:
            header_idx = i + 1
            break
    
    if header_idx is None:
        logger.warning(f"无法找到PSSM矩阵头部: {pssm_file}")
        return np.array([])

    pssm = []
    for line in lines[header_idx:]:
        if not line.strip(): 
            break
        values = line.strip().split()
        if len(values) < 22:  # 确保有足够的列
            continue
        try:
            scores = list(map(int, values[2:22]))  # 20 amino acids
            if len(scores) == 20:  # 确保正好20个分数
                pssm.append(scores)
        except (ValueError, IndexError) as e:
            logger.warning(f"解析PSSM行失败: {line.strip()}, 错误: {e}")
            continue

    if not pssm:
        logger.warning(f"PSSM矩阵为空: {pssm_file}")
        return np.array([])
    
    try:
        return np.array(pssm, dtype=np.int32)
    except Exception as e:
        logger.warning(f"创建PSSM数组失败: {e}")
        return np.array([])

def get_residue_pssm_features(residue, pssm_features, residue_to_seq_idx=None):
    """获取残基的PSSM特征"""
    if not pssm_features:
        return np.zeros(20)  # 20维PSSM特征
    
    chain_id = residue.get_parent().get_id()
    residue_id = parse_residue_number(residue.get_id()[1])  # 残基编号
    
    chain_pssm = pssm_features.get(chain_id)
    if chain_pssm is None or len(chain_pssm) == 0:
        return np.zeros(20)
    
    # 使用残基到序列索引的映射
    if residue_to_seq_idx is not None and chain_id in residue_to_seq_idx:
        chain_mapping = residue_to_seq_idx[chain_id]
        if residue_id in chain_mapping:
            seq_idx = chain_mapping[residue_id]
            if seq_idx < len(chain_pssm):
                pssm_feature = chain_pssm[seq_idx]
                if len(pssm_feature) == 20:  # 确保特征维度正确
                    return pssm_feature
    
    # 如果没有映射,尝试直接使用残基编号(假设从1开始连续)
    # 找到链中残基编号的最小值
    chain = residue.get_parent()
    min_residue_id = min([parse_residue_number(r.get_id()[1]) for r in chain])
    seq_idx = residue_id - min_residue_id
    
    if 0 <= seq_idx < len(chain_pssm):
        pssm_feature = chain_pssm[seq_idx]
        if len(pssm_feature) == 20:  # 确保特征维度正确
            return pssm_feature
    
    # 如果都不匹配,返回零向量
    return np.zeros(20)