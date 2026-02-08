"""
Modular QTS-QFM for Spatio-Temporal Classification
====================================================

Combines:
  - QTSTransformer's QSVT via LCU (hardware-compatible block encoding)
  - ModularQFM's Adaptive Non-Local Observables (ANO, Chen et al., 2025)

Architecture:
  Input (batch, C, T) -> permute -> (batch, T, C)
    -> Linear(C, n_rots) + Sigmoid
    -> Single QNode:
        PCPhase(phi_0)
        for k in range(degree):
            PREPARE (learnable V on ancilla)
            SELECT([U_0(x_0), ..., U_T(x_T)])
            PREPARE_dag
            PCPhase(phi_{k+1})
        QFF sim14 on main register
        -> ANO: learnable k-local Hermitian observables on main register
    -> Linear(n_obs, output_dim) -> class logits

Usage:
  python ModularQTS.py --dataset=physionet --n-qubits=6 --degree=2 \
      --n-layers=2 --k-local=2 --n-epochs=50 --batch-size=32
"""

import argparse
import os
import sys
import random
import copy
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

# Import data loaders from data/ subdirectory
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


# ---------------------------------------------------------------------------
# 1. Argparse
# ---------------------------------------------------------------------------
def get_args():
    p = argparse.ArgumentParser(description="Modular QTS-QFM: QSVT + ANO")

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

    # Training
    p.add_argument("--n-epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3,
                   help="LR for circuit/head params")
    p.add_argument("--lr-H", type=float, default=1e-1,
                   help="LR for observable (ANO) params (Chen 2025: 100x)")
    p.add_argument("--wd", type=float, default=0.0,
                   help="Weight decay")
    p.add_argument("--patience", type=int, default=20,
                   help="Early stopping patience (0 = disabled)")

    # Data
    p.add_argument("--dataset", type=str, default="physionet",
                   help="Dataset to use (physionet or custom)")
    p.add_argument("--num-classes", type=int, default=2,
                   help="Number of classes (determines output_dim and loss)")
    p.add_argument("--sampling-freq", type=int, default=32,
                   help="PhysioNet: target sampling frequency")
    p.add_argument("--sample-size", type=int, default=50,
                   help="PhysioNet: number of subjects")

    # I/O
    p.add_argument("--seed", type=int, default=2025)
    p.add_argument("--job-id", type=str, default="mqts_001")
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


# ---------------------------------------------------------------------------
# 3. sim14 circuit (Sim et al., 2019)
# ---------------------------------------------------------------------------
def sim14_circuit(params, wires, layers=1):
    """
    sim14 ansatz: RY -> CRX(ring) -> RY -> CRX(counter-ring).
    Supports both batched (2D) and unbatched (1D) parameter tensors.
    Parameters per layer: 4 * wires.
    """
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
# 4. ModularQTS Model
# ---------------------------------------------------------------------------
class ModularQTS(nn.Module):
    """
    Modular QTS: QSVT via LCU + Adaptive Non-Local Observables.

    Combines QTSTransformer's hardware-compatible QSVT pipeline with
    ModularQFM's learnable k-local Hermitian measurement.
    """

    def __init__(self,
                 n_qubits: int,
                 n_timesteps: int,
                 degree: int,
                 n_ansatz_layers: int,
                 feature_dim: int,
                 output_dim: int,
                 dropout: float,
                 k_local: int,
                 obs_scheme: str,
                 device):
        super().__init__()

        self.n_qubits = n_qubits
        self.n_timesteps = n_timesteps
        self.degree = degree
        self.n_ansatz_layers = n_ansatz_layers
        self.k_local = k_local
        self.device = device

        # -- Qubit registers --
        self.n_ancilla = ceil(log2(max(n_timesteps, 2)))
        self.main_wires = list(range(n_qubits))
        self.anc_wires = list(range(n_qubits, n_qubits + self.n_ancilla))
        self.total_wires = n_qubits + self.n_ancilla
        self.n_select_ops = 2 ** self.n_ancilla

        # -- Parameter counts --
        self.n_rots = 4 * n_qubits * n_ansatz_layers
        self.qff_n_rots = 4 * n_qubits * 1  # single QFF layer

        # -- Classical layers --
        self.feature_projection = nn.Linear(feature_dim, self.n_rots)
        self.dropout = nn.Dropout(dropout)
        self.rot_sigm = nn.Sigmoid()

        # -- Trainable quantum parameters --
        # PREPARE ansatz on ancilla
        self.n_prep_layers = self.n_ancilla
        self.prepare_params = nn.Parameter(
            0.1 * torch.randn(self.n_prep_layers, self.n_ancilla, 2))

        # QSVT signal processing angles
        self.signal_angles = nn.Parameter(
            0.1 * torch.randn(degree + 1))

        # QFF parameters (single sim14 layer on main register)
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
        _n_timesteps = n_timesteps
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
            QSVT via LCU + ANO measurement.

            Args:
                ts_params: (B, T, n_rots) -- data-dependent sim14 params
                prep_p:    (n_prep_layers, n_ancilla, 2) -- PREPARE params
                sig_ang:   (degree+1,) -- signal processing angles
                qff_p:     (qff_n_rots,) -- QFF circuit params
                H_mats:    list of Hermitian matrices for ANO measurement
            """

            def prepare():
                """Learnable PREPARE unitary V on ancilla register."""
                for ly in range(_n_prep_layers):
                    for qi, q in enumerate(_anc_wires):
                        qml.RY(prep_p[ly, qi, 0], wires=q)
                        qml.RZ(prep_p[ly, qi, 1], wires=q)
                    for i in range(_n_ancilla - 1):
                        qml.CNOT(wires=[_anc_wires[i], _anc_wires[i + 1]])

            def build_select_ops():
                """Build sim14 unitaries for qml.Select, padded to 2^n_ancilla."""
                select_ops = []
                for t in range(_n_timesteps):
                    gates = []
                    param_idx = 0
                    for _ in range(_n_ansatz_layers):
                        # RY layer
                        for i in range(_n_qubits):
                            gates.append(qml.RY(
                                ts_params[..., t, param_idx],
                                wires=_main_wires[i]))
                            param_idx += 1
                        # CRX ring (reverse order)
                        for i in range(_n_qubits - 1, -1, -1):
                            gates.append(qml.CRX(
                                ts_params[..., t, param_idx],
                                wires=[_main_wires[i],
                                       _main_wires[(i + 1) % _n_qubits]]))
                            param_idx += 1
                        # RY layer
                        for i in range(_n_qubits):
                            gates.append(qml.RY(
                                ts_params[..., t, param_idx],
                                wires=_main_wires[i]))
                            param_idx += 1
                        # CRX counter-ring
                        wire_order = [_n_qubits - 1] + list(range(_n_qubits - 1))
                        for i in wire_order:
                            gates.append(qml.CRX(
                                ts_params[..., t, param_idx],
                                wires=[_main_wires[i],
                                       _main_wires[(i - 1) % _n_qubits]]))
                            param_idx += 1
                    select_ops.append(qml.prod(*reversed(gates)))
                # Pad to 2^n_ancilla with Identity
                while len(select_ops) < _n_select_ops:
                    select_ops.append(qml.Identity(wires=_main_wires[0]))
                return select_ops

            # -- QSVT: alternating signal processing and LCU block encoding --
            qml.PCPhase(sig_ang[0], dim=_pcphase_dim, wires=_pcphase_wires)

            for k in range(_degree):
                prepare()
                if k % 2 == 0:
                    qml.Select(build_select_ops(), control=_anc_wires)
                else:
                    qml.adjoint(
                        qml.Select(build_select_ops(), control=_anc_wires))
                qml.adjoint(prepare)()
                qml.PCPhase(sig_ang[k + 1], dim=_pcphase_dim,
                            wires=_pcphase_wires)

            # -- QFF on main register --
            sim14_circuit(qff_p, wires=_n_qubits, layers=1)

            # -- Measurement: ANO or fixed PauliZ --
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

    def forward(self, x):
        # x: (batch, n_channels, n_timesteps)
        x = x.permute(0, 2, 1)                           # (batch, T, C)
        x = self.feature_projection(self.dropout(x))      # (batch, T, n_rots)
        ts_params = self.rot_sigm(x)                      # sigmoid activation

        H_mats = self._build_H_matrices()
        exps = self._circuit(
            ts_params, self.prepare_params,
            self.signal_angles, self.qff_params, H_mats)
        exps = torch.stack(exps, dim=1).float()           # (batch, n_obs)
        return self.head(exps)                             # (batch, output_dim)

    def get_eigenvalue_range(self):
        """Diagnostic: eigenvalue range of learned observables."""
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
# 5. Main -- Data, Training, Evaluation
# ---------------------------------------------------------------------------
def main():
    args = get_args()
    set_all_seeds(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"PennyLane {qml.__version__}  |  PyTorch {torch.__version__}  |  {device}")

    # -- Data --
    # Dataset loader registry: each returns (train, val, test, input_dim)
    # input_dim = (n_trials, n_channels, n_timesteps)
    DATASET_LOADERS = {}

    # PhysioNet EEG Motor Imagery
    def load_physionet():
        from data.Load_PhysioNet_EEG import load_eeg_ts_revised
        return load_eeg_ts_revised(
            seed=args.seed, device=device, batch_size=args.batch_size,
            sampling_freq=args.sampling_freq, sample_size=args.sample_size)

    DATASET_LOADERS["physionet"] = load_physionet

    if args.dataset not in DATASET_LOADERS:
        raise ValueError(f"Unknown dataset: {args.dataset}. "
                         f"Available: {list(DATASET_LOADERS.keys())}")

    train_loader, val_loader, test_loader, input_dim = DATASET_LOADERS[args.dataset]()
    n_trials, n_channels, n_timesteps = input_dim[0], input_dim[1], input_dim[2]
    print(f"Dataset: {args.dataset}  |  input_dim={input_dim}")
    print(f"  n_channels={n_channels}  n_timesteps={n_timesteps}")

    # -- Output dim --
    if args.num_classes <= 2:
        output_dim = 1   # binary: BCEWithLogitsLoss
    else:
        output_dim = args.num_classes  # multi-class: CrossEntropyLoss

    # -- Model --
    model = ModularQTS(
        n_qubits=args.n_qubits,
        n_timesteps=n_timesteps,
        degree=args.degree,
        n_ansatz_layers=args.n_layers,
        feature_dim=n_channels,
        output_dim=output_dim,
        dropout=args.dropout,
        k_local=args.k_local,
        obs_scheme=args.obs_scheme,
        device=device,
    ).to(device)

    print(f"Model: ModularQTS")
    print(f"  qubits={args.n_qubits}  ancilla={model.n_ancilla}  "
          f"total_wires={model.total_wires}")
    print(f"  degree={args.degree}  layers={args.n_layers}  "
          f"n_rots={model.n_rots}")
    print(f"  ANO: k_local={args.k_local}  scheme={args.obs_scheme}  "
          f"n_obs={model.n_obs}")
    print(f"  output_dim={output_dim}")

    # -- Separate optimizers (Chen 2025: 100x ratio) --
    H_params, circuit_params = [], []
    for name, param in model.named_parameters():
        if name.startswith(("A.", "B.", "D.")):
            H_params.append(param)
        else:
            circuit_params.append(param)

    circuit_opt = torch.optim.Adam(circuit_params, lr=args.lr, weight_decay=args.wd)
    H_opt = (torch.optim.Adam(H_params, lr=args.lr_H, weight_decay=args.wd)
             if H_params else None)

    # Loss: binary vs multi-class
    if output_dim == 1:
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
    ckpt_path = os.path.join(ckpt_dir, f"ckpt_mqts_{args.job_id}.pt")
    csv_path = os.path.join(results_dir, f"log_mqts_{args.job_id}.csv")
    csv_fields = ["epoch", "train_loss", "train_acc", "train_auc",
                  "val_loss", "val_acc", "val_auc",
                  "eig_min", "eig_max", "time_s"]

    start_epoch, best_val_acc, best_state, best_epoch = 0, 0.0, None, 0
    history = []

    if args.resume and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, weights_only=False)
        model.load_state_dict(ckpt["model"])
        circuit_opt.load_state_dict(ckpt["circuit_opt"])
        if H_opt and "H_opt" in ckpt:
            H_opt.load_state_dict(ckpt["H_opt"])
        start_epoch = ckpt["epoch"] + 1
        best_val_acc = ckpt.get("best_val_acc", 0.0)
        best_epoch = ckpt.get("best_epoch", 0)
        history = ckpt.get("history", [])
        print(f"Resumed from epoch {start_epoch}")

    if start_epoch == 0:
        with open(csv_path, "w", newline="") as f:
            csv.DictWriter(f, csv_fields).writeheader()

    # -- AUC helper --
    def compute_auc(all_probs, all_labels):
        """Compute binary AUC. Returns 0.0 if single class or multi-class."""
        if output_dim != 1:
            return 0.0
        try:
            from sklearn.metrics import roc_auc_score
            return roc_auc_score(all_labels, all_probs)
        except Exception:
            return 0.0

    # -- Training loop --
    no_improve = 0
    for epoch in range(start_epoch, args.n_epochs):
        t0 = time.time()

        # Train
        model.train()
        tr_loss, tr_ok, tr_n = 0.0, 0, 0
        tr_probs, tr_labels = [], []
        for xb, yb in tqdm(train_loader,
                           desc=f"Epoch {epoch+1}/{args.n_epochs}",
                           leave=False):
            xb, yb = xb.to(device), yb.to(device)
            circuit_opt.zero_grad()
            if H_opt:
                H_opt.zero_grad()

            logits = model(xb)

            if output_dim == 1:
                logits = logits.squeeze(-1)
                loss = criterion(logits, yb.float())
                preds = (logits > 0).long()
                probs = torch.sigmoid(logits).detach().cpu().numpy()
                tr_probs.extend(probs.tolist())
                tr_labels.extend(yb.cpu().numpy().tolist())
            else:
                loss = criterion(logits, yb.long())
                preds = logits.argmax(1)

            loss.backward()
            circuit_opt.step()
            if H_opt:
                H_opt.step()

            tr_loss += loss.item() * len(yb)
            tr_ok += (preds == yb.long()).sum().item()
            tr_n += len(yb)

        tr_loss /= tr_n
        tr_acc = tr_ok / tr_n
        tr_auc = compute_auc(tr_probs, tr_labels)

        # Validate
        model.eval()
        vl_loss, vl_ok, vl_n = 0.0, 0, 0
        vl_probs, vl_labels = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)

                if output_dim == 1:
                    logits = logits.squeeze(-1)
                    loss = criterion(logits, yb.float())
                    preds = (logits > 0).long()
                    probs = torch.sigmoid(logits).detach().cpu().numpy()
                    vl_probs.extend(probs.tolist())
                    vl_labels.extend(yb.cpu().numpy().tolist())
                else:
                    loss = criterion(logits, yb.long())
                    preds = logits.argmax(1)

                vl_loss += loss.item() * len(yb)
                vl_ok += (preds == yb.long()).sum().item()
                vl_n += len(yb)

        vl_loss /= vl_n
        vl_acc = vl_ok / vl_n
        vl_auc = compute_auc(vl_probs, vl_labels)

        eig_lo, eig_hi = model.get_eigenvalue_range()
        dt = time.time() - t0

        # Log
        row = dict(epoch=epoch + 1, train_loss=f"{tr_loss:.6f}",
                   train_acc=f"{tr_acc:.4f}", train_auc=f"{tr_auc:.4f}",
                   val_loss=f"{vl_loss:.6f}", val_acc=f"{vl_acc:.4f}",
                   val_auc=f"{vl_auc:.4f}",
                   eig_min=f"{eig_lo:.4f}", eig_max=f"{eig_hi:.4f}",
                   time_s=f"{dt:.1f}")
        history.append(row)
        with open(csv_path, "a", newline="") as f:
            csv.DictWriter(f, csv_fields).writerow(row)

        print(f"Ep {epoch+1:3d} | Train {100*tr_acc:.1f}% ({tr_loss:.4f}) "
              f"AUC {tr_auc:.3f} | "
              f"Val {100*vl_acc:.1f}% ({vl_loss:.4f}) AUC {vl_auc:.3f} | "
              f"Eig [{eig_lo:.2f}, {eig_hi:.2f}] | {dt:.1f}s")

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
            no_improve = 0
        else:
            no_improve += 1

        # Checkpoint
        ckpt_data = dict(
            epoch=epoch, model=model.state_dict(),
            circuit_opt=circuit_opt.state_dict(),
            best_val_acc=best_val_acc, best_epoch=best_epoch,
            history=history)
        if H_opt:
            ckpt_data["H_opt"] = H_opt.state_dict()
        torch.save(ckpt_data, ckpt_path)

        # Early stopping
        if args.patience > 0 and no_improve >= args.patience:
            print(f"Early stopping at epoch {epoch+1} "
                  f"(no improvement for {args.patience} epochs)")
            break

    # -- Final test --
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    te_loss, te_ok, te_n = 0.0, 0, 0
    te_probs, te_labels = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)

            if output_dim == 1:
                logits = logits.squeeze(-1)
                loss = criterion(logits, yb.float())
                preds = (logits > 0).long()
                probs = torch.sigmoid(logits).detach().cpu().numpy()
                te_probs.extend(probs.tolist())
                te_labels.extend(yb.cpu().numpy().tolist())
            else:
                loss = criterion(logits, yb.long())
                preds = logits.argmax(1)

            te_loss += loss.item() * len(yb)
            te_ok += (preds == yb.long()).sum().item()
            te_n += len(yb)

    te_acc = te_ok / te_n
    te_auc = compute_auc(te_probs, te_labels)

    print(f"\nTest Accuracy: {100 * te_acc:.2f}%  AUC: {te_auc:.4f}  "
          f"(best val epoch: {best_epoch})")

    w_path = os.path.join(ckpt_dir, f"weights_mqts_{args.job_id}.pt")
    torch.save(model.state_dict(), w_path)
    print(f"Saved to {w_path}")


if __name__ == "__main__":
    main()
