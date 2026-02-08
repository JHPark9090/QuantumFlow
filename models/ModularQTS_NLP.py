"""
Modular QTS-NLP: QSVT + ANO for GLUE Benchmarks
==================================================

Discriminative NLP classifier combining:
  - QTSTransformer_v2's positional encoding + temporal chunking
  - ModularQTS's QSVT via LCU (PREPARE-SELECT-PREPARE†) + ANO measurement

Architecture:
  GLUE batch: {input_ids: (B,L), attention_mask: (B,L)}
    -> nn.Embedding(vocab_size, embed_dim) -> (B, L, embed_dim)
    -> + Sinusoidal PE -> (B, L, embed_dim)
    -> mask padding tokens to zero
    -> Linear(embed_dim, n_rots) + Sigmoid -> (B, L, n_rots)
    -> Chunk into windows of chunk_size
    -> Each chunk: QSVT via LCU (PREPARE-SELECT-PREPARE†) + QFF -> ANO
    -> Mean pool across chunks -> (B, n_obs)
    -> Linear(n_obs, output_dim) -> logits

Usage:
  python ModularQTS_NLP.py --task=sst2 --n-qubits=6 --degree=2 \
      --n-layers=2 --k-local=2 --embed-dim=64 --chunk-size=32 \
      --max-length=128 --n-epochs=50 --batch-size=32

References:
  - Sim et al. (2019). Expressibility and Entangling Capability of PQCs.
  - Chen et al. (2025). Learning to Measure QNNs. ICASSP 2025.
  - Lin et al. (2025). Adaptive Non-local Observable on QNNs. IEEE QCE 2025.
"""

import argparse
import os
import sys
import random
import copy
import math
import time
import csv
from math import ceil, log2
from itertools import combinations

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import scipy.constants  # noqa: F401 -- pre-import for PennyLane/scipy compat
import pennylane as qml

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


# ---------------------------------------------------------------------------
# 1. Argparse
# ---------------------------------------------------------------------------
def get_args():
    p = argparse.ArgumentParser(description="Modular QTS-NLP: QSVT + ANO for GLUE")

    # Model
    p.add_argument("--n-qubits", type=int, default=6)
    p.add_argument("--n-layers", type=int, default=2,
                   help="sim14 ansatz layers per timestep")
    p.add_argument("--degree", type=int, default=2,
                   help="QSVT polynomial degree")
    p.add_argument("--dropout", type=float, default=0.1)

    # Observable
    p.add_argument("--k-local", type=int, default=2,
                   help="Locality of ANO measurement. 0 = fixed PauliZ")
    p.add_argument("--obs-scheme", type=str, default="sliding",
                   choices=["sliding", "pairwise"])

    # NLP-specific
    p.add_argument("--task", type=str, default="sst2",
                   choices=["cola", "sst2", "mrpc", "qqp", "stsb",
                            "mnli", "qnli", "rte", "wnli"])
    p.add_argument("--max-length", type=int, default=128,
                   help="Max token sequence length")
    p.add_argument("--embed-dim", type=int, default=64,
                   help="Embedding dimension")
    p.add_argument("--chunk-size", type=int, default=32,
                   help="Temporal chunk size for QSVT")
    p.add_argument("--vocab-size", type=int, default=30522,
                   help="Vocabulary size (30522 for bert-base-uncased)")

    # Training
    p.add_argument("--n-epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3,
                   help="LR for circuit/head params")
    p.add_argument("--lr-H", type=float, default=1e-1,
                   help="LR for observable (ANO) params (Chen 2025: 100x)")
    p.add_argument("--wd", type=float, default=0.0)
    p.add_argument("--patience", type=int, default=20,
                   help="Early stopping patience (0 = disabled)")

    # Debug / data limits
    p.add_argument("--max-train-samples", type=int, default=None)
    p.add_argument("--max-val-samples", type=int, default=None)

    # I/O
    p.add_argument("--seed", type=int, default=2025)
    p.add_argument("--job-id", type=str, default="mqts_nlp_001")
    p.add_argument("--base-path", type=str, default=".")
    p.add_argument("--resume", action="store_true", default=False)

    return p.parse_args()


# ---------------------------------------------------------------------------
# 2. Utilities
# ---------------------------------------------------------------------------
def set_all_seeds(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    qml.numpy.random.seed(seed)


def create_Hermitian(N, A, B, D):
    """Build N x N Hermitian from learnable real params (Lin et al., 2025)."""
    h = torch.zeros((N, N), dtype=torch.complex128)
    count = 0
    for i in range(1, N):
        h[i - 1, i - 1] = D[i].clone()
        for j in range(i):
            h[i, j] = A[count + j].clone() + 1j * B[count + j].clone()
        count += i
    return h.clone() + h.clone().conj().T


def get_wire_groups(n_qubits, k_local, obs_scheme):
    """Wire groups for k-local observable measurement."""
    if k_local <= 0:
        return [[q] for q in range(n_qubits)]
    if obs_scheme == "sliding":
        return [list(range(s, s + k_local))
                for s in range(n_qubits - k_local + 1)]
    elif obs_scheme == "pairwise":
        return [list(c) for c in combinations(range(n_qubits), k_local)]
    raise ValueError(f"Unknown obs_scheme: {obs_scheme}")


def sim14_circuit(params, wires, layers=1):
    """sim14 ansatz: RY -> CRX(ring) -> RY -> CRX(counter-ring)."""
    is_batched = params.ndim == 2
    param_idx = 0
    for _ in range(layers):
        for i in range(wires):
            angle = params[:, param_idx] if is_batched else params[param_idx]
            qml.RY(angle, wires=i)
            param_idx += 1
        for i in range(wires - 1, -1, -1):
            angle = params[:, param_idx] if is_batched else params[param_idx]
            qml.CRX(angle, wires=[i, (i + 1) % wires])
            param_idx += 1
        for i in range(wires):
            angle = params[:, param_idx] if is_batched else params[param_idx]
            qml.RY(angle, wires=i)
            param_idx += 1
        wire_order = [wires - 1] + list(range(wires - 1))
        for i in wire_order:
            angle = params[:, param_idx] if is_batched else params[param_idx]
            qml.CRX(angle, wires=[i, (i - 1) % wires])
            param_idx += 1


# ---------------------------------------------------------------------------
# 3. ModularQTS_NLP Model
# ---------------------------------------------------------------------------
class ModularQTS_NLP(nn.Module):
    """
    Modular QTS for NLP: QSVT via LCU + ANO with token embeddings.

    Combines QTSTransformer_v2's PE + chunking with ModularQTS's QSVT + ANO
    pipeline, replacing EEG inputs with token embeddings for GLUE tasks.
    """

    def __init__(self,
                 n_qubits: int,
                 embed_dim: int,
                 max_length: int,
                 degree: int,
                 n_ansatz_layers: int,
                 output_dim: int,
                 dropout: float,
                 k_local: int,
                 obs_scheme: str,
                 vocab_size: int,
                 chunk_size: int,
                 device):
        super().__init__()

        self.n_qubits = n_qubits
        self.max_length = max_length
        self.degree = degree
        self.n_ansatz_layers = n_ansatz_layers
        self.k_local = k_local
        self.device = device

        # -- Token embedding --
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # -- Sinusoidal Positional Encoding (Vaswani et al., 2017) --
        pe = torch.zeros(max_length, embed_dim)
        pos = torch.arange(max_length).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, embed_dim, 2).float()
            * -(math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[:embed_dim // 2])
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_length, embed_dim)

        # -- Temporal chunking --
        self.chunk_size = chunk_size or max_length
        self.n_chunks = ceil(max_length / self.chunk_size)

        # -- Qubit registers (sized by chunk) --
        self.n_ancilla = ceil(log2(max(self.chunk_size, 2)))
        self.main_wires = list(range(n_qubits))
        self.anc_wires = list(range(n_qubits, n_qubits + self.n_ancilla))
        self.total_wires = n_qubits + self.n_ancilla
        self.n_select_ops = 2 ** self.n_ancilla

        # -- Parameter counts --
        self.n_rots = 4 * n_qubits * n_ansatz_layers
        self.qff_n_rots = 4 * n_qubits * 1  # single QFF layer

        # -- Classical layers --
        self.feature_projection = nn.Linear(embed_dim, self.n_rots)
        self.dropout = nn.Dropout(dropout)
        self.rot_sigm = nn.Sigmoid()

        # -- Trainable quantum parameters --
        self.n_prep_layers = self.n_ancilla
        self.prepare_params = nn.Parameter(
            0.1 * torch.randn(self.n_prep_layers, self.n_ancilla, 2))
        self.signal_angles = nn.Parameter(
            0.1 * torch.randn(degree + 1))
        self.qff_params = nn.Parameter(torch.rand(self.qff_n_rots))

        # -- ANO parameters (Chen et al., 2025) --
        self.wire_groups = get_wire_groups(n_qubits, k_local, obs_scheme)
        self.n_obs = len(self.wire_groups)

        if k_local > 0:
            K = 2 ** k_local
            self.obs_dim = K
            n_off = (K * (K - 1)) // 2
            self.A = nn.ParameterList(
                [nn.Parameter(torch.empty(n_off)) for _ in range(self.n_obs)])
            self.B = nn.ParameterList(
                [nn.Parameter(torch.empty(n_off)) for _ in range(self.n_obs)])
            self.D = nn.ParameterList(
                [nn.Parameter(torch.empty(K)) for _ in range(self.n_obs)])
            for w in range(self.n_obs):
                nn.init.normal_(self.A[w], std=2.0)
                nn.init.normal_(self.B[w], std=2.0)
                nn.init.normal_(self.D[w], std=2.0)
        else:
            self.obs_dim = 0

        # -- Classification head --
        self.head = nn.Linear(self.n_obs, output_dim)

        # -- PennyLane device and QNode --
        self.dev = qml.device("default.qubit", wires=self.total_wires)

        # Capture instance attributes for QNode closure
        _n_qubits = n_qubits
        _chunk_size = self.chunk_size
        _n_ancilla = self.n_ancilla
        _n_ansatz_layers = n_ansatz_layers
        _n_prep_layers = self.n_prep_layers
        _main_wires = self.main_wires
        _anc_wires = self.anc_wires
        _degree = degree
        _n_select_ops = self.n_select_ops
        _pcphase_wires = self.anc_wires + self.main_wires
        _pcphase_dim = 2 ** n_qubits
        _kl = k_local
        _wg = self.wire_groups
        _no = self.n_obs

        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def _circuit(ts_params, prep_p, sig_ang, qff_p, H_mats):
            """
            QSVT via LCU + ANO measurement on a single chunk.

            Args:
                ts_params: (B, chunk_size, n_rots) -- rotation params for chunk
                prep_p:    (n_prep_layers, n_ancilla, 2) -- PREPARE params
                sig_ang:   (degree+1,) -- signal processing angles
                qff_p:     (qff_n_rots,) -- QFF circuit params
                H_mats:    list of Hermitian matrices for ANO measurement
            """

            def prepare():
                for ly in range(_n_prep_layers):
                    for qi, q in enumerate(_anc_wires):
                        qml.RY(prep_p[ly, qi, 0], wires=q)
                        qml.RZ(prep_p[ly, qi, 1], wires=q)
                    for i in range(_n_ancilla - 1):
                        qml.CNOT(wires=[_anc_wires[i], _anc_wires[i + 1]])

            def build_select_ops():
                select_ops = []
                for t in range(_chunk_size):
                    gates = []
                    param_idx = 0
                    for _ in range(_n_ansatz_layers):
                        for i in range(_n_qubits):
                            gates.append(qml.RY(
                                ts_params[..., t, param_idx],
                                wires=_main_wires[i]))
                            param_idx += 1
                        for i in range(_n_qubits - 1, -1, -1):
                            gates.append(qml.CRX(
                                ts_params[..., t, param_idx],
                                wires=[_main_wires[i],
                                       _main_wires[(i + 1) % _n_qubits]]))
                            param_idx += 1
                        for i in range(_n_qubits):
                            gates.append(qml.RY(
                                ts_params[..., t, param_idx],
                                wires=_main_wires[i]))
                            param_idx += 1
                        wire_order = [_n_qubits - 1] + list(range(_n_qubits - 1))
                        for i in wire_order:
                            gates.append(qml.CRX(
                                ts_params[..., t, param_idx],
                                wires=[_main_wires[i],
                                       _main_wires[(i - 1) % _n_qubits]]))
                            param_idx += 1
                    select_ops.append(qml.prod(*reversed(gates)))
                while len(select_ops) < _n_select_ops:
                    select_ops.append(qml.Identity(wires=_main_wires[0]))
                return select_ops

            # QSVT: alternating signal processing and LCU block encoding
            select_ops = build_select_ops()  # build once, reuse

            qml.PCPhase(sig_ang[0], dim=_pcphase_dim, wires=_pcphase_wires)

            for k in range(_degree):
                prepare()
                if k % 2 == 0:
                    qml.Select(select_ops, control=_anc_wires)
                else:
                    qml.adjoint(qml.Select)(select_ops, control=_anc_wires)
                qml.adjoint(prepare)()
                qml.PCPhase(sig_ang[k + 1], dim=_pcphase_dim,
                            wires=_pcphase_wires)

            # QFF on main register
            sim14_circuit(qff_p, wires=_n_qubits, layers=1)

            # Measurement: ANO or fixed PauliZ
            if _kl > 0:
                return [qml.expval(qml.Hermitian(H_mats[w], wires=_wg[w]))
                        for w in range(_no)]
            else:
                return [qml.expval(qml.PauliZ(q)) for q in _main_wires]

        self._circuit = _circuit

    def _build_H_matrices(self):
        if self.k_local <= 0:
            return []
        return [create_Hermitian(self.obs_dim, self.A[w], self.B[w], self.D[w])
                for w in range(self.n_obs)]

    def forward(self, input_ids, attention_mask):
        """
        Args:
            input_ids:      (B, L) int token IDs
            attention_mask: (B, L) binary mask (1=real, 0=pad)
        Returns:
            logits: (B, output_dim)
        """
        B, L = input_ids.shape

        # Embed + PE + mask
        x = self.embedding(input_ids)                      # (B, L, embed_dim)
        x = x + self.pe[:, :L]                             # + sinusoidal PE
        x = x * attention_mask.unsqueeze(-1).float()       # zero out padding

        # Project to rotation params
        x = self.feature_projection(self.dropout(x))       # (B, L, n_rots)
        ts_params = self.rot_sigm(x) * (2 * math.pi)        # sigmoid -> [0, 2pi]

        # Build Hermitian matrices for ANO
        H_mats = self._build_H_matrices()

        # Process chunks (from QTSTransformer_v2)
        chunk_results = []
        for start in range(0, L, self.chunk_size):
            chunk = ts_params[:, start:start + self.chunk_size]
            pad_len = self.chunk_size - chunk.size(1)
            if pad_len > 0:
                chunk = nn.functional.pad(chunk, (0, 0, 0, pad_len))
            exps = self._circuit(
                chunk, self.prepare_params,
                self.signal_angles, self.qff_params, H_mats)
            chunk_results.append(torch.stack(exps, dim=1).float())

        # Mean pool across chunks -> (B, n_obs)
        exps = torch.stack(chunk_results, dim=0).mean(dim=0)
        return self.head(exps)                              # (B, output_dim)

    def get_eigenvalue_range(self):
        if self.k_local <= 0:
            return 0.0, 0.0
        H_mats = self._build_H_matrices()
        lo, hi = float("inf"), float("-inf")
        for H in H_mats:
            eigs = torch.linalg.eigvalsh(
                H.detach().cpu().to(torch.complex128)).real
            lo = min(lo, eigs.min().item())
            hi = max(hi, eigs.max().item())
        return lo, hi


# ---------------------------------------------------------------------------
# 4. Metrics
# ---------------------------------------------------------------------------
def compute_metric(metric_name, preds, labels, probs=None):
    """Compute GLUE task-specific metric."""
    if metric_name == "accuracy":
        return (np.array(preds) == np.array(labels)).mean()

    elif metric_name == "f1":
        from sklearn.metrics import f1_score
        return f1_score(labels, preds, average="binary")

    elif metric_name == "matthews_correlation":
        from sklearn.metrics import matthews_corrcoef
        return matthews_corrcoef(labels, preds)

    elif metric_name == "pearson":
        from scipy.stats import pearsonr
        r, _ = pearsonr(preds, labels)
        return r

    return 0.0


# ---------------------------------------------------------------------------
# 5. Main
# ---------------------------------------------------------------------------
def main():
    args = get_args()
    set_all_seeds(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"PennyLane {qml.__version__}  |  PyTorch {torch.__version__}  |  {device}")

    # -- Load GLUE data --
    from data.Load_GLUE import load_glue_task, GLUE_TASKS

    task_config = GLUE_TASKS[args.task]
    is_regression = task_config.get("is_regression", False)
    metric_name = task_config["metric"]

    train_loader, val_loader, test_loader, metadata = load_glue_task(
        task_name=args.task,
        max_length=args.max_length,
        batch_size=args.batch_size,
        num_workers=0,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
    )

    vocab_size = metadata["vocab_size"]
    if args.vocab_size != vocab_size:
        print(f"Note: using tokenizer vocab_size={vocab_size} "
              f"(CLI was {args.vocab_size})")

    print(f"Task: {args.task} ({task_config['name']})  |  "
          f"metric: {metric_name}")
    print(f"  num_labels={metadata['num_labels']}  "
          f"is_regression={is_regression}")
    print(f"  train={metadata['train_size']}  val={metadata['val_size']}")

    # -- Output dim --
    num_labels = metadata["num_labels"]
    if is_regression:
        output_dim = 1
    elif num_labels <= 2:
        output_dim = 1   # binary: BCEWithLogitsLoss
    else:
        output_dim = num_labels  # multi-class: CrossEntropyLoss

    # -- Model --
    model = ModularQTS_NLP(
        n_qubits=args.n_qubits,
        embed_dim=args.embed_dim,
        max_length=args.max_length,
        degree=args.degree,
        n_ansatz_layers=args.n_layers,
        output_dim=output_dim,
        dropout=args.dropout,
        k_local=args.k_local,
        obs_scheme=args.obs_scheme,
        vocab_size=vocab_size,
        chunk_size=args.chunk_size,
        device=device,
    ).to(device)

    print(f"Model: ModularQTS_NLP")
    print(f"  qubits={args.n_qubits}  ancilla={model.n_ancilla}  "
          f"total_wires={model.total_wires}")
    print(f"  degree={args.degree}  layers={args.n_layers}  "
          f"n_rots={model.n_rots}")
    print(f"  chunk_size={model.chunk_size}  n_chunks(max)={model.n_chunks}")
    print(f"  ANO: k_local={args.k_local}  scheme={args.obs_scheme}  "
          f"n_obs={model.n_obs}")
    print(f"  embed_dim={args.embed_dim}  vocab_size={vocab_size}  "
          f"output_dim={output_dim}")

    # -- Dual optimizer (Chen 2025) --
    H_params, circuit_params = [], []
    for name, param in model.named_parameters():
        if name.startswith(("A.", "B.", "D.")):
            H_params.append(param)
        else:
            circuit_params.append(param)

    circuit_opt = torch.optim.Adam(circuit_params, lr=args.lr,
                                   weight_decay=args.wd)
    H_opt = (torch.optim.Adam(H_params, lr=args.lr_H, weight_decay=args.wd)
             if H_params else None)

    # -- Loss --
    if is_regression:
        criterion = nn.MSELoss()
    elif output_dim == 1:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    total_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    h_p = sum(p.numel() for p in H_params)
    c_p = sum(p.numel() for p in circuit_params)
    print(f"Params: total={total_p}  circuit={c_p}  observable={h_p}")

    # -- Checkpoint / CSV --
    ckpt_dir = os.path.join(args.base_path, "checkpoints")
    results_dir = os.path.join(args.base_path, "results")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f"ckpt_mqts_nlp_{args.job_id}.pt")
    csv_path = os.path.join(results_dir, f"log_mqts_nlp_{args.job_id}.csv")
    csv_fields = ["epoch", "train_loss", f"train_{metric_name}",
                  "val_loss", f"val_{metric_name}",
                  "eig_min", "eig_max", "time_s"]

    start_epoch, best_val_metric, best_state, best_epoch = 0, -1e9, None, 0
    history = []

    if args.resume and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, weights_only=False)
        model.load_state_dict(ckpt["model"])
        circuit_opt.load_state_dict(ckpt["circuit_opt"])
        if H_opt and "H_opt" in ckpt:
            H_opt.load_state_dict(ckpt["H_opt"])
        start_epoch = ckpt["epoch"] + 1
        best_val_metric = ckpt.get("best_val_metric", -1e9)
        best_epoch = ckpt.get("best_epoch", 0)
        history = ckpt.get("history", [])
        print(f"Resumed from epoch {start_epoch}")

    if start_epoch == 0:
        with open(csv_path, "w", newline="") as f:
            csv.DictWriter(f, csv_fields).writeheader()

    # -- Helpers --
    def run_epoch(loader, training=True):
        if training:
            model.train()
        else:
            model.eval()

        total_loss, total_n = 0.0, 0
        all_preds, all_labels, all_probs = [], [], []

        ctx = torch.no_grad() if not training else torch.enable_grad()
        with ctx:
            for batch in (tqdm(loader,
                               desc=f"Epoch {epoch+1}/{args.n_epochs}",
                               leave=False) if training else loader):
                ids = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                if training:
                    circuit_opt.zero_grad()
                    if H_opt:
                        H_opt.zero_grad()

                logits = model(ids, mask)

                if is_regression:
                    logits = logits.squeeze(-1)
                    loss = criterion(logits, labels.float())
                    preds = logits.detach().cpu().numpy().tolist()
                elif output_dim == 1:
                    logits = logits.squeeze(-1)
                    loss = criterion(logits, labels.float())
                    preds = (logits > 0).long().cpu().numpy().tolist()
                    probs = torch.sigmoid(logits).detach().cpu().numpy()
                    all_probs.extend(probs.tolist())
                else:
                    loss = criterion(logits, labels.long())
                    preds = logits.argmax(1).cpu().numpy().tolist()

                if training:
                    loss.backward()
                    circuit_opt.step()
                    if H_opt:
                        H_opt.step()

                bs = ids.size(0)
                total_loss += loss.item() * bs
                total_n += bs
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy().tolist())

        avg_loss = total_loss / total_n
        metric_val = compute_metric(
            metric_name, all_preds, all_labels,
            probs=all_probs if all_probs else None)
        return avg_loss, metric_val

    # -- Training loop --
    no_improve = 0
    for epoch in range(start_epoch, args.n_epochs):
        t0 = time.time()

        tr_loss, tr_metric = run_epoch(train_loader, training=True)
        vl_loss, vl_metric = run_epoch(val_loader, training=False)

        eig_lo, eig_hi = model.get_eigenvalue_range()
        dt = time.time() - t0

        row = dict(epoch=epoch + 1, train_loss=f"{tr_loss:.6f}",
                   val_loss=f"{vl_loss:.6f}",
                   eig_min=f"{eig_lo:.4f}", eig_max=f"{eig_hi:.4f}",
                   time_s=f"{dt:.1f}")
        row[f"train_{metric_name}"] = f"{tr_metric:.4f}"
        row[f"val_{metric_name}"] = f"{vl_metric:.4f}"
        history.append(row)

        with open(csv_path, "a", newline="") as f:
            csv.DictWriter(f, csv_fields).writerow(row)

        print(f"Ep {epoch+1:3d} | Train loss={tr_loss:.4f} "
              f"{metric_name}={tr_metric:.4f} | "
              f"Val loss={vl_loss:.4f} {metric_name}={vl_metric:.4f} | "
              f"Eig [{eig_lo:.2f}, {eig_hi:.2f}] | {dt:.1f}s")

        if vl_metric > best_val_metric:
            best_val_metric = vl_metric
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
            no_improve = 0
        else:
            no_improve += 1

        # Checkpoint
        ckpt_data = dict(
            epoch=epoch, model=model.state_dict(),
            circuit_opt=circuit_opt.state_dict(),
            best_val_metric=best_val_metric, best_epoch=best_epoch,
            history=history)
        if H_opt:
            ckpt_data["H_opt"] = H_opt.state_dict()
        torch.save(ckpt_data, ckpt_path)

        if args.patience > 0 and no_improve >= args.patience:
            print(f"Early stopping at epoch {epoch+1} "
                  f"(no improvement for {args.patience} epochs)")
            break

    # -- Final test --
    if best_state is not None:
        model.load_state_dict(best_state)

    te_loss, te_metric = run_epoch(test_loader, training=False)
    print(f"\nTest loss={te_loss:.4f}  {metric_name}={te_metric:.4f}  "
          f"(best val epoch: {best_epoch})")

    w_path = os.path.join(ckpt_dir, f"weights_mqts_nlp_{args.job_id}.pt")
    torch.save(model.state_dict(), w_path)
    print(f"Saved to {w_path}")


if __name__ == "__main__":
    main()
