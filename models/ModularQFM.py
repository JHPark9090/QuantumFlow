"""
Modular Quantum Flow Matching for Image Classification
=======================================================

Three composable stages:
  1. Encoding:    SU(4) exponential map  (Wiersema et al., 2024)
                  — or angle embedding baseline
  2. Processing:  Swappable VQC (QCNN / hardware-efficient / none)
  3. Measurement: Adaptive Non-Local Observables (Lin et al., 2025)
                  — learnable k-local Hermitians (Chen et al., 2025)

Architecture:
  Input x → [Encoding] → [VQC] → [ANO Measurement] → nn.Linear → logits

Usage:
  python ModularQFM.py --dataset=mnist --n-qubits=10 --encoding-type=sun \
      --vqc-type=qcnn --k-local=2 --epochs=30
"""

import argparse
import os
import sys
import random
import copy
import math
import time
import csv
from itertools import combinations

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import scipy.constants  # noqa: F401 — pre-import for PennyLane/scipy compat
import pennylane as qml

# Import data loaders from data/ subdirectory
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from data.Load_Image_Datasets import load_mnist, load_fashion, load_cifar


# ---------------------------------------------------------------------------
# 1. Argparse
# ---------------------------------------------------------------------------
def get_args():
    p = argparse.ArgumentParser(description="Modular Quantum Flow Matching")

    # Encoding
    p.add_argument("--n-qubits", type=int, default=10)
    p.add_argument("--n-blocks", type=int, default=0,
                   help="Encoding blocks. 0 = auto-compute from input_dim")
    p.add_argument("--encoding-type", type=str, default="sun",
                   choices=["sun", "angle"])
    p.add_argument("--encoding-mode", type=str, default="direct",
                   choices=["direct", "projected"],
                   help="'direct': raw data as encoding params (no FC); "
                        "'projected': classical FC layer before encoding")
    p.add_argument("--encoding-scale", type=float, default=math.pi,
                   help="Scale factor for data in direct encoding mode")

    # VQC
    p.add_argument("--vqc-type", type=str, default="qcnn",
                   choices=["qcnn", "hardware_efficient", "su4_reupload", "none"])
    p.add_argument("--vqc-depth", type=int, default=2,
                   help="Number of VQC layers (conv+pool for QCNN)")

    # Observable
    p.add_argument("--k-local", type=int, default=2,
                   help="Locality of ANO measurement. 0 = fixed PauliZ")
    p.add_argument("--obs-scheme", type=str, default="sliding",
                   choices=["sliding", "pairwise"])

    # Training
    p.add_argument("--lr", type=float, default=1e-3,
                   help="LR for encoding/VQC/head params")
    p.add_argument("--lr-H", type=float, default=1e-1,
                   help="LR for observable (ANO) params (Chen 2025: 100x)")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--seed", type=int, default=2025)

    # Data
    p.add_argument("--dataset", type=str, default="mnist",
                   choices=["mnist", "fashion", "cifar10"])
    p.add_argument("--n-train", type=int, default=1000)
    p.add_argument("--n-valtest", type=int, default=500)
    p.add_argument("--num-classes", type=int, default=10)

    # I/O
    p.add_argument("--job-id", type=str, default="mqfm_001")
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
    """Build N×N Hermitian from learnable real params (Lin et al., 2025)."""
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
# 3. ModularQFM Model
# ---------------------------------------------------------------------------
class ModularQFM(nn.Module):
    """
    Modular Quantum Flow Matching model.

    Three composable stages:
      1. Encoding  — SU(4) exponential map (direct/projected) or angle embedding
      2. VQC       — QCNN, hardware-efficient, or none
      3. Measurement — k-local ANO or fixed PauliZ
    """

    def __init__(self, input_dim, output_dim, n_qubits, n_blocks,
                 encoding_type, encoding_mode, encoding_scale,
                 vqc_type, vqc_depth, k_local, obs_scheme):
        super().__init__()

        self.n_qubits = n_qubits
        self.encoding_type = encoding_type
        self.encoding_mode = encoding_mode
        self.encoding_scale = encoding_scale
        self.vqc_type = vqc_type
        self.vqc_depth = vqc_depth
        self.k_local = k_local

        # ── Encoding setup ──
        if encoding_type == "sun":
            n_even = n_qubits // 2
            n_odd = (n_qubits - 1) // 2
            gates_per_block = n_even + n_odd
            params_per_block = gates_per_block * 15  # SU(4) Lie algebra dim

            if encoding_mode == "direct":
                if n_blocks <= 0:
                    n_blocks = math.ceil(input_dim / params_per_block)
                total_enc = n_blocks * params_per_block
                self.encoding_proj = None
            else:  # projected
                if n_blocks <= 0:
                    n_blocks = 2
                total_enc = n_blocks * params_per_block
                self.encoding_proj = nn.Linear(input_dim, total_enc)

        else:  # angle
            if encoding_mode == "direct":
                if n_blocks <= 0:
                    n_blocks = math.ceil(input_dim / n_qubits)
                total_enc = n_blocks * n_qubits
                self.encoding_proj = None
            else:  # projected
                n_blocks = 1
                total_enc = n_qubits
                self.encoding_proj = nn.Linear(input_dim, total_enc)

        self.n_blocks = n_blocks
        self.total_enc = total_enc

        # ── VQC parameters ──
        if vqc_type == "qcnn":
            self.conv_params = nn.Parameter(
                0.01 * torch.randn(vqc_depth, n_qubits, 15))
            self.pool_params = nn.Parameter(
                0.01 * torch.randn(vqc_depth, n_qubits // 2, 3))
        elif vqc_type == "hardware_efficient":
            self.var_params = nn.Parameter(
                0.01 * torch.randn(vqc_depth, n_qubits))
        elif vqc_type == "su4_reupload":
            # Each re-upload block: re-encode data + trainable SU(4) brick-wall.
            # Trainable params: one SU(4) brick-wall per block (same layout as encoding).
            n_even = n_qubits // 2
            n_odd = (n_qubits - 1) // 2
            gates_per_block = n_even + n_odd
            self.reupload_params = nn.Parameter(
                0.01 * torch.randn(vqc_depth, gates_per_block, 15))
        # vqc_type == "none": no parameters

        # ── ANO parameters ──
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

        # ── Classification head ──
        self.head = nn.Linear(self.n_obs, output_dim)

        # ── QNode ──
        # No wires= arg: lets PennyLane auto-allocate ancillas for
        # deferred mid-circuit measurements (needed by QCNN pooling).
        dev = qml.device("default.qubit")

        # Capture for closure (self not accessible inside QNode)
        _wg = self.wire_groups
        _nq = n_qubits
        _nb = self.n_blocks
        _et = encoding_type
        _vt = vqc_type
        _vd = vqc_depth
        _kl = k_local
        _no = self.n_obs

        @qml.qnode(dev, interface="torch", diff_method="best")
        def _circuit(enc, vqc_p1, vqc_p2, H_mats):
            """Encoding → VQC → Measurement.  Supports parameter broadcasting."""

            # ── Stage 1: Encoding ──
            if _et == "sun":
                idx = 0
                for _ in range(_nb):
                    for q in range(0, _nq - 1, 2):
                        qml.SpecialUnitary(enc[..., idx:idx + 15],
                                           wires=[q, q + 1])
                        idx += 15
                    for q in range(1, _nq - 1, 2):
                        qml.SpecialUnitary(enc[..., idx:idx + 15],
                                           wires=[q, q + 1])
                        idx += 15
            else:  # angle — re-upload layers with entangling gates
                for layer in range(_nb):
                    for q in range(_nq):
                        qml.RY(enc[..., layer * _nq + q], wires=q)
                    if layer < _nb - 1:
                        for q in range(0, _nq - 1, 2):
                            qml.CNOT(wires=[q, q + 1])
                        for q in range(1, _nq - 1, 2):
                            qml.CNOT(wires=[q, q + 1])

            # ── Stage 2: VQC ──
            if _vt == "qcnn":
                wires = list(range(_nq))
                for ly in range(_vd):
                    nw = len(wires)
                    if nw < 2:
                        break
                    # Convolution (staggered 15-param 2-qubit blocks)
                    for parity in [0, 1]:
                        for i in range(len(wires)):
                            if i % 2 == parity and i < nw - 1:
                                w1, w2 = wires[i], wires[i + 1]
                                qml.U3(*vqc_p1[ly, i, :3], wires=w1)
                                qml.U3(*vqc_p1[ly, i + 1, 3:6], wires=w2)
                                qml.IsingZZ(vqc_p1[ly, i, 6],
                                            wires=[w1, w2])
                                qml.IsingYY(vqc_p1[ly, i, 7],
                                            wires=[w1, w2])
                                qml.IsingXX(vqc_p1[ly, i, 8],
                                            wires=[w1, w2])
                                qml.U3(*vqc_p1[ly, i, 9:12], wires=w1)
                                qml.U3(*vqc_p1[ly, i + 1, 12:15], wires=w2)
                    # Pooling (mid-circuit measure + conditional U3)
                    for i in range(len(wires)):
                        if i % 2 == 1 and i < nw:
                            m = qml.measure(wires[i])
                            qml.cond(m, qml.U3)(
                                *vqc_p2[ly, i // 2],
                                wires=wires[i - 1])
                    wires = wires[::2]

            elif _vt == "hardware_efficient":
                for ly in range(_vd):
                    for q in range(_nq):
                        qml.RY(vqc_p1[ly, q], wires=q)
                    for q in range(0, _nq - 1, 2):
                        qml.CNOT(wires=[q, q + 1])
                    for q in range(1, _nq - 1, 2):
                        qml.CNOT(wires=[q, q + 1])

            elif _vt == "su4_reupload":
                # Re-upload: re-encode same data via SU(4) + trainable SU(4)
                # vqc_p1 = reupload_params [vqc_depth, gates_per_block, 15]
                for ly in range(_vd):
                    # Re-encode data (same enc every block)
                    idx_re = 0
                    for q in range(0, _nq - 1, 2):
                        qml.SpecialUnitary(enc[..., idx_re:idx_re + 15],
                                           wires=[q, q + 1])
                        idx_re += 15
                    for q in range(1, _nq - 1, 2):
                        qml.SpecialUnitary(enc[..., idx_re:idx_re + 15],
                                           wires=[q, q + 1])
                        idx_re += 15
                    # Trainable SU(4) processing layer
                    gi = 0
                    for q in range(0, _nq - 1, 2):
                        qml.SpecialUnitary(vqc_p1[ly, gi], wires=[q, q + 1])
                        gi += 1
                    for q in range(1, _nq - 1, 2):
                        qml.SpecialUnitary(vqc_p1[ly, gi], wires=[q, q + 1])
                        gi += 1

            # ── Stage 3: Measurement ──
            if _kl > 0:
                return [qml.expval(qml.Hermitian(H_mats[w], wires=_wg[w]))
                        for w in range(_no)]
            else:
                return [qml.expval(qml.PauliZ(q)) for q in range(_nq)]

        self._circuit = _circuit

    def _build_H_matrices(self):
        if self.k_local <= 0:
            return []
        return [create_Hermitian(self.obs_dim, self.A[w], self.B[w], self.D[w])
                for w in range(self.n_obs)]

    def forward(self, x):
        # ── Encode ──
        if self.encoding_proj is not None:
            encoded = self.encoding_proj(x)
        else:
            enc = x * self.encoding_scale
            diff = self.total_enc - enc.shape[1]
            if diff > 0:
                pad = torch.zeros(enc.shape[0], diff, device=enc.device)
                encoded = torch.cat([enc, pad], dim=1)
            elif diff < 0:
                encoded = enc[:, :self.total_enc]
            else:
                encoded = enc

        # ── Hermitians ──
        H_mats = self._build_H_matrices()

        # ── VQC params ──
        if self.vqc_type == "qcnn":
            p1, p2 = self.conv_params, self.pool_params
        elif self.vqc_type == "hardware_efficient":
            p1, p2 = self.var_params, torch.zeros(1)
        elif self.vqc_type == "su4_reupload":
            p1, p2 = self.reupload_params, torch.zeros(1)
        else:
            p1, p2 = torch.zeros(1), torch.zeros(1)

        # ── Quantum forward (batched via PennyLane broadcasting) ──
        q_out = self._circuit(encoded, p1, p2, H_mats)
        q_out = torch.stack(q_out, dim=1).float()   # (batch, n_obs)
        return self.head(q_out)                      # (batch, output_dim)

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
# 4. Main — Data, Training, Evaluation
# ---------------------------------------------------------------------------
def main():
    args = get_args()
    set_all_seeds(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"PennyLane {qml.__version__}  |  PyTorch {torch.__version__}  |  {device}")

    # ── Data ──
    loader_fn = {
        "mnist": load_mnist,
        "fashion": load_fashion,
        "cifar10": load_cifar,
    }[args.dataset]

    train_loader, val_loader, test_loader, input_dim = loader_fn(
        seed=args.seed, n_train=args.n_train, n_valtest=args.n_valtest,
        device=device, batch_size=args.batch_size)

    print(f"Dataset: {args.dataset}  |  input_dim={input_dim}  |  "
          f"train={args.n_train}  val+test={args.n_valtest}")

    # ── Model ──
    model = ModularQFM(
        input_dim=input_dim, output_dim=args.num_classes,
        n_qubits=args.n_qubits, n_blocks=args.n_blocks,
        encoding_type=args.encoding_type,
        encoding_mode=args.encoding_mode,
        encoding_scale=args.encoding_scale,
        vqc_type=args.vqc_type, vqc_depth=args.vqc_depth,
        k_local=args.k_local, obs_scheme=args.obs_scheme,
    ).to(device)

    print(f"Encoding:  {args.encoding_type} ({args.encoding_mode}), "
          f"{model.n_blocks} blocks, {model.total_enc} encoding slots")
    print(f"VQC:       {args.vqc_type}, depth={args.vqc_depth}")
    print(f"Obs:       k_local={args.k_local}, scheme={args.obs_scheme}, "
          f"n_obs={model.n_obs}")

    if (args.encoding_mode == "direct" and args.encoding_type == "angle"
            and model.n_blocks > 20):
        print(f"WARNING: angle+direct needs {model.n_blocks} encoding layers. "
              f"Consider --encoding-mode=projected for speed.")

    # ── Separate optimizers (Chen 2025: 100× ratio) ──
    H_params, circuit_params = [], []
    for name, param in model.named_parameters():
        if name.startswith(("A.", "B.", "D.")):
            H_params.append(param)
        else:
            circuit_params.append(param)

    circuit_opt = torch.optim.Adam(circuit_params, lr=args.lr)
    H_opt = (torch.optim.Adam(H_params, lr=args.lr_H)
             if H_params else None)
    criterion = nn.CrossEntropyLoss()

    total_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    h_p = sum(p.numel() for p in H_params)
    c_p = sum(p.numel() for p in circuit_params)
    print(f"Params:    total={total_p}  circuit={c_p}  observable={h_p}")

    # ── Checkpoint / CSV ──
    ckpt_dir = os.path.join(args.base_path, "checkpoints")
    results_dir = os.path.join(args.base_path, "results")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f"ckpt_mqfm_{args.job_id}.pt")
    csv_path = os.path.join(results_dir, f"log_mqfm_{args.job_id}.csv")
    csv_fields = ["epoch", "train_loss", "train_acc", "val_loss", "val_acc",
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

    # ── Training loop ──
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        # Train
        model.train()
        tr_loss, tr_ok, tr_n = 0.0, 0, 0
        for xb, yb in tqdm(train_loader,
                           desc=f"Epoch {epoch+1}/{args.epochs}",
                           leave=False):
            xb, yb = xb.to(device), yb.to(device)
            circuit_opt.zero_grad()
            if H_opt:
                H_opt.zero_grad()

            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()

            circuit_opt.step()
            if H_opt:
                H_opt.step()

            tr_loss += loss.item() * len(yb)
            tr_ok += (logits.argmax(1) == yb).sum().item()
            tr_n += len(yb)

        tr_loss /= tr_n
        tr_acc = tr_ok / tr_n

        # Validate
        model.eval()
        vl_loss, vl_ok, vl_n = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                vl_loss += loss.item() * len(yb)
                vl_ok += (logits.argmax(1) == yb).sum().item()
                vl_n += len(yb)

        vl_loss /= vl_n
        vl_acc = vl_ok / vl_n

        eig_lo, eig_hi = model.get_eigenvalue_range()
        dt = time.time() - t0

        # Log
        row = dict(epoch=epoch + 1, train_loss=f"{tr_loss:.6f}",
                   train_acc=f"{tr_acc:.4f}", val_loss=f"{vl_loss:.6f}",
                   val_acc=f"{vl_acc:.4f}", eig_min=f"{eig_lo:.4f}",
                   eig_max=f"{eig_hi:.4f}", time_s=f"{dt:.1f}")
        history.append(row)
        with open(csv_path, "a", newline="") as f:
            csv.DictWriter(f, csv_fields).writerow(row)

        print(f"Ep {epoch+1:3d} | Train {100*tr_acc:.1f}% ({tr_loss:.4f}) | "
              f"Val {100*vl_acc:.1f}% ({vl_loss:.4f}) | "
              f"Eig [{eig_lo:.2f}, {eig_hi:.2f}] | {dt:.1f}s")

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1

        # Checkpoint
        ckpt_data = dict(
            epoch=epoch, model=model.state_dict(),
            circuit_opt=circuit_opt.state_dict(),
            best_val_acc=best_val_acc, best_epoch=best_epoch,
            history=history)
        if H_opt:
            ckpt_data["H_opt"] = H_opt.state_dict()
        torch.save(ckpt_data, ckpt_path)

    # ── Final test ──
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    te_ok, te_n = 0, 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            te_ok += (logits.argmax(1) == yb).sum().item()
            te_n += len(yb)

    print(f"\nTest Accuracy: {100 * te_ok / te_n:.2f}% "
          f"(best val epoch: {best_epoch})")

    w_path = os.path.join(ckpt_dir, f"weights_mqfm_{args.job_id}.pt")
    torch.save(model.state_dict(), w_path)
    print(f"Saved to {w_path}")


if __name__ == "__main__":
    main()
