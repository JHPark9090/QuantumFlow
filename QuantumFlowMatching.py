"""
Quantum Flow Matching with Adaptive Observables
================================================

Combines three research innovations:
  1. SU(N) exponential flow encoding  (Wiersema et al., 2024)
  2. Learnable observables           (Chen et al., 2025)
  3. Adaptive non-local observables  (Lin et al., 2025)

Prototype on Make-Moons binary classification using PyTorch + PennyLane.

References:
  - Wiersema et al., "Here comes the SU(N)", Quantum 8, 1275 (2024)
  - Chen et al., "Learning to Measure Quantum Neural Networks", ICASSP 2025
  - Lin et al., "Adaptive Non-Local Observable on QNNs", IEEE QCE 2025
"""

import argparse
import os
import random
import copy
import time
import csv
from itertools import combinations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import scipy.constants  # noqa: F401 — pre-import for PennyLane/scipy compat
import pennylane as qml


# ---------------------------------------------------------------------------
# 1. Argparse
# ---------------------------------------------------------------------------
def get_args():
    parser = argparse.ArgumentParser(description="Quantum Flow Matching Prototype")
    # Quantum circuit
    parser.add_argument("--n-qubits", type=int, default=4)
    parser.add_argument("--n-blocks", type=int, default=2, help="Number of SU(4) brick-wall blocks")
    parser.add_argument("--encoding-type", type=str, default="sun", choices=["sun", "angle"],
                        help="'sun' = SU(4) flow encoding, 'angle' = RY angle embedding")
    # Observable
    parser.add_argument("--k-local", type=int, default=2,
                        help="Locality of ANO measurement. 0 = fixed PauliZ per qubit")
    parser.add_argument("--obs-scheme", type=str, default="sliding", choices=["sliding", "pairwise"],
                        help="Wire grouping scheme for k-local observables")
    # Optional variational layer
    parser.add_argument("--use-variational", action="store_true", default=False,
                        help="Add trainable RY + CNOT variational layer after encoding")
    parser.add_argument("--vqc-depth", type=int, default=1, help="Depth of optional variational block")
    # Training
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for circuit params")
    parser.add_argument("--lr-H", type=float, default=1e-1, help="Learning rate for observable params")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=2025)
    # Data
    parser.add_argument("--n-samples", type=int, default=500)
    parser.add_argument("--noise", type=float, default=0.15, help="Make-Moons noise")
    # I/O
    parser.add_argument("--job-id", type=str, default="qfm_001")
    parser.add_argument("--base-path", type=str, default=".")
    parser.add_argument("--resume", action="store_true", default=False)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# 2. Utility Functions
# ---------------------------------------------------------------------------
def set_all_seeds(seed: int = 42) -> None:
    """Seed every RNG we rely on (Python, NumPy, Torch, PennyLane, CUDNN)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    qml.numpy.random.seed(seed)


def create_Hermitian(N, A, B, D):
    """Build an N x N Hermitian matrix from learnable real parameters.

    Directly adapted from Lin et al. (2025) / ANO_sliding_klocal_MNIST_Github.py.
    Uses .clone() to preserve gradient flow.
    """
    h = torch.zeros((N, N), dtype=torch.complex128)
    count = 0
    for i in range(1, N):
        h[i - 1, i - 1] = D[i].clone()  # fill diagonal
        for j in range(i):
            h[i, j] = A[count + j].clone() + 1j * B[count + j].clone()  # fill off-diagonal
        count += i
    H = h.clone() + h.clone().conj().T
    return H


def get_wire_groups(n_qubits, k_local, obs_scheme):
    """Compute wire groups for k-local observable measurement.

    Returns:
        list of lists: Each inner list contains the wire indices for one observable group.
    """
    if k_local <= 0:
        # Fallback: one wire per observable (PauliZ per qubit)
        return [[q] for q in range(n_qubits)]

    if obs_scheme == "sliding":
        # Sliding window: (0,1), (1,2), ..., (n-k, n-k+1, ...)
        groups = []
        for start in range(n_qubits - k_local + 1):
            groups.append(list(range(start, start + k_local)))
        return groups

    elif obs_scheme == "pairwise":
        # All pairs (for k_local=2)
        return [list(c) for c in combinations(range(n_qubits), k_local)]

    else:
        raise ValueError(f"Unknown obs_scheme: {obs_scheme}")


# ---------------------------------------------------------------------------
# 3. QuantumFlowMatching Model
# ---------------------------------------------------------------------------
class QuantumFlowMatching(nn.Module):
    """Quantum Flow Matching model combining SU(N) encoding + ANO measurement."""

    def __init__(self, input_dim, output_dim, n_qubits, n_blocks,
                 encoding_type, k_local, obs_scheme,
                 use_variational, vqc_depth):
        super().__init__()

        self.n_qubits = n_qubits
        self.n_blocks = n_blocks
        self.encoding_type = encoding_type
        self.k_local = k_local
        self.use_variational = use_variational
        self.vqc_depth = vqc_depth

        # --- Encoding projection ---
        if encoding_type == "sun":
            # SU(4) brick-wall: count 2-qubit gates per block
            n_even = n_qubits // 2           # even sublayer pairs: (0,1),(2,3),...
            n_odd = (n_qubits - 1) // 2      # odd sublayer pairs: (1,2),(3,4),...
            self.gates_per_block = n_even + n_odd
            self.su4_params_per_gate = 15     # dim of su(4) Lie algebra
            total_encoding_params = n_blocks * self.gates_per_block * self.su4_params_per_gate
        else:  # angle encoding
            total_encoding_params = n_qubits
        self.encoding_proj = nn.Linear(input_dim, total_encoding_params)

        # --- Optional variational parameters ---
        if use_variational:
            self.var_params = nn.Parameter(0.01 * torch.randn(vqc_depth, n_qubits))
        else:
            self.var_params = None

        # --- ANO parameters ---
        self.wire_groups = get_wire_groups(n_qubits, k_local, obs_scheme)
        self.n_obs = len(self.wire_groups)

        if k_local > 0:
            K = 2 ** k_local  # dimension of each Hermitian
            self.obs_dim = K
            n_off = (K * (K - 1)) // 2
            self.A = nn.ParameterList([nn.Parameter(torch.empty(n_off)) for _ in range(self.n_obs)])
            self.B = nn.ParameterList([nn.Parameter(torch.empty(n_off)) for _ in range(self.n_obs)])
            self.D = nn.ParameterList([nn.Parameter(torch.empty(K)) for _ in range(self.n_obs)])
            for w in range(self.n_obs):
                nn.init.normal_(self.A[w], std=2.0)
                nn.init.normal_(self.B[w], std=2.0)
                nn.init.normal_(self.D[w], std=2.0)
        else:
            self.obs_dim = 0

        # --- Classification head ---
        self.head = nn.Linear(self.n_obs, output_dim)

        # --- Quantum device & QNode ---
        self.dev = qml.device("default.qubit", wires=n_qubits)

        # Capture wire_groups as a local for the closure
        _wire_groups = self.wire_groups

        @qml.qnode(self.dev, interface="torch", diff_method="best")
        def _circuit(encoded_params, var_params_flat, H_matrices):
            """Core quantum circuit (supports PennyLane parameter broadcasting).

            Args:
                encoded_params: Tensor of shape (batch, total_params) or (total_params,).
                var_params_flat: Flat tensor of variational params (or dummy if unused).
                H_matrices: List of Hermitian matrices for measurement (or empty).
            """
            if encoding_type == "sun":
                idx = 0
                for block in range(n_blocks):
                    # Even sublayer: pairs (0,1), (2,3), ...
                    for q in range(0, n_qubits - 1, 2):
                        params_slice = encoded_params[..., idx: idx + 15]
                        qml.SpecialUnitary(params_slice, wires=[q, q + 1])
                        idx += 15
                    # Odd sublayer: pairs (1,2), (3,4), ...
                    for q in range(1, n_qubits - 1, 2):
                        params_slice = encoded_params[..., idx: idx + 15]
                        qml.SpecialUnitary(params_slice, wires=[q, q + 1])
                        idx += 15
            else:  # angle encoding
                for q in range(n_qubits):
                    qml.RY(encoded_params[..., q], wires=q)

            # Optional variational layer
            if use_variational and var_params_flat is not None:
                var_2d = var_params_flat.reshape(vqc_depth, n_qubits)
                for layer in range(vqc_depth):
                    for q in range(n_qubits):
                        qml.RY(var_2d[layer, q], wires=q)
                    for q in range(0, n_qubits - 1, 2):
                        qml.CNOT(wires=[q, q + 1])
                    for q in range(1, n_qubits - 1, 2):
                        qml.CNOT(wires=[q, q + 1])

            # Measurement
            if k_local > 0:
                return [qml.expval(qml.Hermitian(H_matrices[w], wires=_wire_groups[w]))
                        for w in range(len(_wire_groups))]
            else:
                return [qml.expval(qml.PauliZ(q)) for q in range(n_qubits)]

        self._circuit = _circuit

        # Store structural info needed at forward time
        self._total_encoding_params = total_encoding_params

    def _build_H_matrices(self):
        """Construct Hermitian matrices from learnable A, B, D parameters."""
        if self.k_local <= 0:
            return []
        return [create_Hermitian(self.obs_dim, self.A[w], self.B[w], self.D[w])
                for w in range(self.n_obs)]

    def forward(self, x):
        """Forward pass: project → encode → measure → classify.

        Uses PennyLane parameter broadcasting to process the entire batch
        at once (no per-sample loop), matching the QCNN pattern.

        Args:
            x: Input tensor of shape (batch_size, input_dim).
        Returns:
            logits: Tensor of shape (batch_size, output_dim).
        """
        encoded = self.encoding_proj(x)  # (batch, total_encoding_params)

        # Build Hermitian matrices once for entire batch
        H_matrices = self._build_H_matrices()

        # Variational params (flatten for QNode)
        if self.use_variational and self.var_params is not None:
            var_flat = self.var_params.flatten()
        else:
            var_flat = torch.zeros(1)  # dummy

        # Batch QNode evaluation via PennyLane parameter broadcasting
        quantum_out = self._circuit(encoded, var_flat, H_matrices)
        # quantum_out is a list of tensors, each of shape (batch,)
        q_out = torch.stack(quantum_out, dim=1).float()  # (batch, n_obs)

        logits = self.head(q_out)  # (batch, output_dim)
        return logits

    def get_eigenvalue_range(self):
        """Diagnostic: compute eigenvalue range of learned observables.

        Returns:
            (lambda_min, lambda_max) across all observable groups,
            or (0, 0) if using fixed PauliZ.
        """
        if self.k_local <= 0:
            return 0.0, 0.0

        H_matrices = self._build_H_matrices()
        all_min, all_max = float("inf"), float("-inf")
        for H in H_matrices:
            eigs = torch.linalg.eigvalsh(H.detach().cpu().to(torch.complex128))
            eigs_real = eigs.real
            all_min = min(all_min, eigs_real.min().item())
            all_max = max(all_max, eigs_real.max().item())
        return all_min, all_max


# ---------------------------------------------------------------------------
# 4. Data Loading — Make-Moons
# ---------------------------------------------------------------------------
def load_moons(n_samples, noise, seed):
    """Create Make-Moons dataset with train/val/test splits.

    Returns:
        train_loader, val_loader, test_loader, input_dim
    """
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=seed)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # 70/15/15 split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=seed, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=seed, stratify=y_temp
    )

    def to_loader(X_np, y_np, batch_size, shuffle):
        ds = TensorDataset(
            torch.tensor(X_np, dtype=torch.float32),
            torch.tensor(y_np, dtype=torch.long),
        )
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    return X_train, y_train, X_val, y_val, X_test, y_test


# ---------------------------------------------------------------------------
# 5. Training Loop
# ---------------------------------------------------------------------------
def main():
    args = get_args()
    set_all_seeds(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"PennyLane version: {qml.__version__}")
    print(f"PyTorch version:   {torch.__version__}")
    print(f"Device:            {device}")
    print(f"Config:            encoding={args.encoding_type}, k_local={args.k_local}, "
          f"obs_scheme={args.obs_scheme}, variational={args.use_variational}, "
          f"n_qubits={args.n_qubits}, n_blocks={args.n_blocks}")

    # --- Data ---
    X_train, y_train, X_val, y_val, X_test, y_test = load_moons(
        args.n_samples, args.noise, args.seed
    )
    input_dim = X_train.shape[1]  # 2 for Make-Moons

    def make_loader(X_np, y_np, shuffle):
        ds = TensorDataset(
            torch.tensor(X_np, dtype=torch.float32),
            torch.tensor(y_np, dtype=torch.long),
        )
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle)

    train_loader = make_loader(X_train, y_train, shuffle=True)
    val_loader = make_loader(X_val, y_val, shuffle=False)
    test_loader = make_loader(X_test, y_test, shuffle=False)

    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # --- Model ---
    model = QuantumFlowMatching(
        input_dim=input_dim,
        output_dim=2,  # binary classification
        n_qubits=args.n_qubits,
        n_blocks=args.n_blocks,
        encoding_type=args.encoding_type,
        k_local=args.k_local,
        obs_scheme=args.obs_scheme,
        use_variational=args.use_variational,
        vqc_depth=args.vqc_depth,
    ).to(device)

    # --- Separate optimizers (Chen 2025: 100x ratio) ---
    H_params = []
    circuit_params = []
    for name, param in model.named_parameters():
        if name.startswith("A.") or name.startswith("B.") or name.startswith("D."):
            H_params.append(param)
        else:
            circuit_params.append(param)

    circuit_optimizer = torch.optim.Adam(circuit_params, lr=args.lr)
    if H_params:
        H_optimizer = torch.optim.Adam(H_params, lr=args.lr_H)
    else:
        H_optimizer = None

    criterion = nn.CrossEntropyLoss()

    # --- Checkpoint loading ---
    os.makedirs(args.base_path, exist_ok=True)
    checkpoint_path = os.path.join(args.base_path, f"checkpoint_qfm_{args.job_id}.pt")
    start_epoch = 0
    best_val_acc = 0.0
    best_state = None
    history = []

    if args.resume and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        circuit_optimizer.load_state_dict(ckpt["circuit_optimizer_state_dict"])
        if H_optimizer and "H_optimizer_state_dict" in ckpt:
            H_optimizer.load_state_dict(ckpt["H_optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val_acc = ckpt.get("best_val_acc", 0.0)
        history = ckpt.get("history", [])
        print(f"Resuming from epoch {start_epoch}")

    # --- CSV logger ---
    csv_path = os.path.join(args.base_path, f"log_qfm_{args.job_id}.csv")
    csv_fields = ["epoch", "train_loss", "train_acc", "val_loss", "val_acc",
                  "eig_min", "eig_max", "epoch_time_s"]

    # Write CSV header if starting fresh
    if start_epoch == 0:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=csv_fields)
            writer.writeheader()

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    h_total = sum(p.numel() for p in H_params)
    c_total = sum(p.numel() for p in circuit_params)
    print(f"Trainable params: {total_params} (circuit: {c_total}, observable: {h_total})")

    # --- Training ---
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        # Train
        model.train()
        total_loss, total_correct, n_samples = 0.0, 0, 0

        for X_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}",
                                     leave=False):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            circuit_optimizer.zero_grad()
            if H_optimizer:
                H_optimizer.zero_grad()

            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()

            circuit_optimizer.step()
            if H_optimizer:
                H_optimizer.step()

            total_loss += loss.item() * len(y_batch)
            total_correct += (logits.argmax(dim=1) == y_batch).sum().item()
            n_samples += len(y_batch)

        train_loss = total_loss / n_samples
        train_acc = total_correct / n_samples

        # Validate
        model.eval()
        val_loss, val_correct, val_n = 0.0, 0, 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
                val_loss += loss.item() * len(y_batch)
                val_correct += (logits.argmax(dim=1) == y_batch).sum().item()
                val_n += len(y_batch)

        val_loss = val_loss / val_n
        val_acc = val_correct / val_n

        # Eigenvalue range
        eig_min, eig_max = model.get_eigenvalue_range()

        epoch_time = time.time() - t0

        # Log
        row = {
            "epoch": epoch + 1,
            "train_loss": f"{train_loss:.6f}",
            "train_acc": f"{train_acc:.4f}",
            "val_loss": f"{val_loss:.6f}",
            "val_acc": f"{val_acc:.4f}",
            "eig_min": f"{eig_min:.4f}",
            "eig_max": f"{eig_max:.4f}",
            "epoch_time_s": f"{epoch_time:.1f}",
        }
        history.append(row)
        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=csv_fields)
            writer.writerow(row)

        print(f"Epoch {epoch+1:3d} | "
              f"Train Loss: {train_loss:.4f}, Acc: {100*train_acc:.1f}% | "
              f"Val Loss: {val_loss:.4f}, Acc: {100*val_acc:.1f}% | "
              f"Eig: [{eig_min:.3f}, {eig_max:.3f}] | "
              f"Time: {epoch_time:.1f}s")

        # Best model tracking
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1

        # Checkpoint
        ckpt_data = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "circuit_optimizer_state_dict": circuit_optimizer.state_dict(),
            "best_val_acc": best_val_acc,
            "history": history,
        }
        if H_optimizer:
            ckpt_data["H_optimizer_state_dict"] = H_optimizer.state_dict()
        torch.save(ckpt_data, checkpoint_path)

    # --- Final test evaluation ---
    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    test_correct, test_n = 0, 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            logits = model(X_batch)
            test_correct += (logits.argmax(dim=1) == y_batch).sum().item()
            test_n += len(y_batch)

    test_acc = test_correct / test_n
    print(f"\nFinal Test Accuracy: {100*test_acc:.2f}% (best val epoch: {best_epoch})")

    # Save final model
    weights_path = os.path.join(args.base_path, f"weights_qfm_{args.job_id}.pt")
    torch.save(model.state_dict(), weights_path)
    print(f"Model saved to {weights_path}")

    # --- Gradient check (first epoch diagnostic) ---
    print("\n--- Gradient Sanity Check ---")
    model.train()
    X_sample = torch.tensor(X_train[:2], dtype=torch.float32).to(device)
    y_sample = torch.tensor(y_train[:2], dtype=torch.long).to(device)
    circuit_optimizer.zero_grad()
    if H_optimizer:
        H_optimizer.zero_grad()
    logits = model(X_sample)
    loss = criterion(logits, y_sample)
    loss.backward()
    for name, param in model.named_parameters():
        if param.requires_grad:
            grad_status = "OK" if param.grad is not None and param.grad.abs().sum() > 0 else "NONE/ZERO"
            print(f"  {name:30s} | grad: {grad_status}")


if __name__ == "__main__":
    main()
