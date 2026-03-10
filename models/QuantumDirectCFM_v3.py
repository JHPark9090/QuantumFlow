"""
Quantum Direct CFM v3 — Multi-Observable Global ANO
====================================================

Flow matching directly in pixel space WITHOUT a VAE, using a single
SU(2^n) exponential mapping gate on ALL n qubits (same as v2), but with
GLOBAL (k=n) Adaptive Non-Local Observables instead of pairwise (k=2).

Key innovation: m independent 2^n × 2^n learnable Hermitian matrices,
each measuring the FULL n-qubit quantum state. This captures all
correlations (1-body through n-body) that the SU(2^n) encoding produces,
unlike pairwise ANO which only sees 2-body correlations.

Differences from QuantumDirectCFM_v2.py (v2):
  - Global ANO: k=n_qubits instead of k=2
  - Multiple observables: --n-observables independent 2^n × 2^n Hermitians
    on the same wire group (all qubits)
  - Each Hermitian has 4^n learnable parameters (vs 16 for pairwise k=2)
  - Default 15 observables: 15 × 4096 = 61,440 ANO params (vs 240 in v2)
  - No --k-local or --obs-scheme args (always global, always all qubits)

v1 vs v2 vs v3:
  v1: Brick-layer SU(2^k) + QViT + pairwise ANO (k=2)
  v2: Single SU(2^n)      + none + pairwise ANO (k=2)
  v3: Single SU(2^n)      + none + global ANO   (k=n, multi-observable)

Architecture:
  Image x_t (3, H, W) + time t
    -> ConvEncoder (auto Conv layers) -> flatten -> d_flat (4096)
    -> time_mlp projects t to d_flat dims, added to features -> d_flat
    -> enc_proj: Linear -> SiLU -> Linear -> 4^n - 1
    -> Single SU(2^n) gate on all n qubits (exponential mapping encoding)
    -> Optional QViT ansatz (default: none)
    -> m independent global Hermitian measurements -> m observables
    -> vel_head: Linear -> SiLU -> Linear -> d_flat
    -> ConvDecoder (auto Conv layers) -> velocity (3, H, W)

Usage:
  # Default: 6 qubits, SU(64), 15 global observables
  python QuantumDirectCFM_v3.py --dataset=cifar10 --velocity-field=quantum

  # More observables for richer output
  python QuantumDirectCFM_v3.py --n-observables=20

  # Fewer observables for faster training
  python QuantumDirectCFM_v3.py --n-observables=8

  # Classical baseline
  python QuantumDirectCFM_v3.py --velocity-field=classical

References:
  - Lipman et al. (2023). Flow Matching for Generative Modeling. ICLR 2023.
  - Esser et al. (2024). Scaling Rectified Flow Transformers. ICML 2024.
  - Wiersema et al. (2024). Here comes the SU(N). Quantum, 8, 1275.
  - Chen et al. (2025). Learning to Measure Quantum Neural Networks. ICASSP 2025 Workshop.
  - Lin et al. (2025). Adaptive Non-local Observable on QNNs. IEEE QCE 2025.
  - Cherrat et al. (2024). Quantum Vision Transformers. Quantum, 8, 1265.
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
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split
from tqdm import tqdm
import scipy.constants  # noqa: F401 -- pre-import for PennyLane/scipy compat
import pennylane as qml

DATA_ROOT = "/pscratch/sd/j/junghoon/data"


# ---------------------------------------------------------------------------
# 1. Argparse
# ---------------------------------------------------------------------------
def get_args():
    p = argparse.ArgumentParser(
        description="Quantum Direct CFM v3 — Multi-Observable Global ANO")

    # Quantum circuit
    p.add_argument("--n-circuits", type=int, default=1)
    p.add_argument("--n-qubits", type=int, default=6)
    p.add_argument("--n-observables", type=int, default=15,
                   help="Number of independent global Hermitian observables. "
                        "Each is a 2^n x 2^n learnable matrix on all qubits.")
    p.add_argument("--vqc-type", type=str, default="none",
                   choices=["qvit", "hardware_efficient", "none"])
    p.add_argument("--vqc-depth", type=int, default=2)
    p.add_argument("--qvit-circuit", type=str, default="butterfly",
                   choices=["butterfly", "pyramid", "x"])

    # Encoder/Decoder
    p.add_argument("--enc-channels", type=str, default="auto",
                   help="Channel progression for Conv encoder. "
                        "'auto' computes channels to achieve d_flat~=4^n "
                        "for ~1:1 enc_proj ratio with SU(2^n).")
    p.add_argument("--time-embed-dim", type=int, default=256)

    # Training
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lr-H", type=float, default=1e-1)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--seed", type=int, default=2025)

    # v9-style improvements
    p.add_argument("--logit-normal-std", type=float, default=1.0,
                   help="Std of logit-normal timestep sampling "
                        "(0 = uniform, >0 = logit-normal)")
    p.add_argument("--ode-solver", type=str, default="midpoint",
                   choices=["euler", "midpoint"])
    p.add_argument("--vf-ema-decay", type=float, default=0.999,
                   help="EMA decay for velocity field (0 = disable)")

    # Data
    p.add_argument("--dataset", type=str, default="cifar10",
                   choices=["cifar10", "coco", "mnist", "fashion"])
    p.add_argument("--n-train", type=int, default=10000)
    p.add_argument("--n-valtest", type=int, default=2000)
    p.add_argument("--img-size", type=int, default=32)

    # ODE sampling
    p.add_argument("--ode-steps", type=int, default=50)
    p.add_argument("--n-samples", type=int, default=64)

    # I/O
    p.add_argument("--job-id", type=str, default="qdcfm3_001")
    p.add_argument("--base-path", type=str, default=".")
    p.add_argument("--resume", action="store_true")

    # Velocity field type
    p.add_argument("--velocity-field", type=str, default="quantum",
                   choices=["quantum", "classical"])
    p.add_argument("--mlp-hidden-dims", type=str, default="256,256,256")

    # Evaluation
    p.add_argument("--compute-metrics", action="store_true")
    p.add_argument("--n-eval-samples", type=int, default=1024)

    return p.parse_args()


# ---------------------------------------------------------------------------
# 2. Utilities
# ---------------------------------------------------------------------------
def set_all_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    qml.numpy.random.seed(seed)


def auto_enc_channels(img_size, n_qubits):
    """Compute Conv encoder channels for ~1:1 enc_proj ratio with SU(2^n).

    Targets d_flat = 4^n (approx enc_size = 4^n - 1) by choosing the number
    of stride-2 Conv layers and final channel count based on image resolution.

    Strategy:
      - img_size < 128:  target final spatial = 8  (fewer layers)
      - img_size >= 128: target final spatial = 4  (more layers, richer encoder)

    Examples (n_qubits=6, target_d_flat=4096):
      32x32:   2 layers -> [32, 64]         -> 64x8x8  = 4096
      64x64:   3 layers -> [32, 64, 64]     -> 64x8x8  = 4096
      128x128: 5 layers -> [32,64,128,256,256] -> 256x4x4 = 4096
      256x256: 6 layers -> [32,64,128,256,256,256] -> 256x4x4 = 4096
    """
    target_d_flat = 4 ** n_qubits  # e.g., 4096 for 6 qubits

    # Choose target spatial resolution
    if img_size >= 128:
        target_spatial = 4
    else:
        target_spatial = 8

    # Number of stride-2 layers needed
    n_layers = int(round(math.log2(img_size / target_spatial)))
    n_layers = max(n_layers, 2)  # minimum 2 layers

    # Verify spatial
    spatial = img_size // (2 ** n_layers)
    if spatial < 1:
        spatial = 1
        n_layers = int(round(math.log2(img_size)))

    # Last channel to achieve target d_flat
    last_ch = target_d_flat // (spatial * spatial)
    last_ch = max(last_ch, 32)

    # Build progressive channel list: 32, 64, 128, ..., capped at last_ch
    channels = []
    ch = 32
    for i in range(n_layers):
        channels.append(min(ch, last_ch))
        ch = min(ch * 2, last_ch)
    channels[-1] = last_ch  # ensure last channel matches target

    actual_d_flat = last_ch * spatial * spatial
    print(f"  [auto_enc_channels] img_size={img_size}, n_qubits={n_qubits}")
    print(f"    target_d_flat={target_d_flat}, actual_d_flat={actual_d_flat}")
    print(f"    n_layers={n_layers}, spatial={spatial}, last_ch={last_ch}")
    print(f"    enc_channels={channels}")

    return tuple(channels)


def create_Hermitian(N, A, B, D):
    """Build NxN Hermitian from learnable real params (Lin et al., 2025)."""
    h = torch.zeros((N, N), dtype=torch.complex128)
    count = 0
    for i in range(1, N):
        h[i - 1, i - 1] = D[i].clone()
        for j in range(i):
            h[i, j] = A[count + j].clone() + 1j * B[count + j].clone()
        count += i
    return h.clone() + h.clone().conj().T


def _qvit_n_params(n_qubits, circuit_type):
    if circuit_type == "butterfly":
        count = 0
        n_layers = int(math.ceil(math.log2(n_qubits)))
        for layer in range(n_layers):
            stride = 2 ** layer
            for start in range(0, n_qubits - stride, 2 * stride):
                for offset in range(stride):
                    idx1 = start + offset
                    idx2 = start + offset + stride
                    if idx1 < n_qubits and idx2 < n_qubits:
                        count += 1
        return count
    elif circuit_type == "pyramid":
        return sum(n_qubits - layer - 1 for layer in range(n_qubits - 1))
    elif circuit_type == "x":
        return n_qubits // 2 + max(0, n_qubits // 2 - 1)
    raise ValueError(f"Unknown circuit_type: {circuit_type}")


def sinusoidal_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=t.device) / half)
    args = t.unsqueeze(-1) * freqs
    return torch.cat([args.cos(), args.sin()], dim=-1)


# ---------------------------------------------------------------------------
# 2b. Custom SU(2^k) gate in pure PyTorch
# ---------------------------------------------------------------------------
class TorchSUGate:
    """Custom SU(2^k) gate implementation in pure PyTorch.

    PennyLane's SpecialUnitary fails for k >= 6 due to numpy/torch
    incompatibility. This class precomputes Pauli string generators and
    uses torch.linalg.matrix_exp for the unitary computation.

    Memory usage: O(n_gen * dim^2) for precomputed generators.
      SU(64):  4095 * 64^2 * 16 bytes ~ 268 MB
      SU(128): 16383 * 128^2 * 16 bytes ~ 4.3 GB (uses streaming)
      SU(256): 65535 * 256^2 * 16 bytes ~ 68 GB (uses streaming)
    """

    _paulis = None

    @staticmethod
    def _get_paulis():
        if TorchSUGate._paulis is None:
            TorchSUGate._paulis = [
                torch.eye(2, dtype=torch.complex128),
                torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128),
                torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex128),
                torch.tensor([[1, 0], [0, -1]], dtype=torch.complex128),
            ]
        return TorchSUGate._paulis

    def __init__(self, n_qubits, precompute=True):
        from itertools import product as iter_product
        self.n_qubits = n_qubits
        self.dim = 2 ** n_qubits
        self.n_gen = self.dim ** 2 - 1
        self.precompute = precompute and n_qubits <= 6

        if self.precompute:
            paulis = self._get_paulis()
            generators = []
            for indices in iter_product(range(4), repeat=n_qubits):
                if all(i == 0 for i in indices):
                    continue
                mat = paulis[indices[0]]
                for k in range(1, n_qubits):
                    mat = torch.kron(mat, paulis[indices[k]])
                generators.append(mat)
            self.generators = torch.stack(generators, dim=0)
        else:
            self.generators = None

    def matrix(self, theta):
        """Compute U = exp(i * sum(theta_k * G_k)).

        Args:
            theta: Parameters, shape (..., n_gen). May require grad.

        Returns:
            Unitary matrix, shape (..., dim, dim).
        """
        if self.precompute:
            gens = self.generators.to(theta.device)
            A = torch.einsum("...g,gij->...ij", theta.to(torch.complex128),
                             gens)
        else:
            from itertools import product as iter_product
            paulis = [p.to(theta.device) for p in self._get_paulis()]
            batch_shape = theta.shape[:-1]
            A = torch.zeros(*batch_shape, self.dim, self.dim,
                            dtype=torch.complex128, device=theta.device)
            idx = 0
            for indices in iter_product(range(4), repeat=self.n_qubits):
                if all(i == 0 for i in indices):
                    continue
                mat = paulis[indices[0]]
                for k in range(1, self.n_qubits):
                    mat = torch.kron(mat, paulis[indices[k]])
                t = theta[..., idx].to(torch.complex128)
                A = A + t.unsqueeze(-1).unsqueeze(-1) * mat
                idx += 1
        return torch.linalg.matrix_exp(1j * A)


# ---------------------------------------------------------------------------
# 2c. Single-gate SU(2^n) encoding utilities
# ---------------------------------------------------------------------------
def compute_single_gate_encoding(n_qubits):
    """Compute encoding size for a single SU(2^n) gate on all n qubits.

    Returns:
        wires: list of all qubit indices [0, 1, ..., n-1]
        n_generators: 4^n - 1
        enc_size: same as n_generators (single gate)
    """
    dim = 2 ** n_qubits
    n_generators = dim ** 2 - 1
    wires = list(range(n_qubits))
    return wires, n_generators, n_generators


class EMAModel:
    """Exponential moving average of model weights."""

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.clone().detach()
                       for k, v in model.state_dict().items()}

    def update(self, model):
        with torch.no_grad():
            for k, v in model.state_dict().items():
                self.shadow[k].mul_(self.decay).add_(v, alpha=1.0 - self.decay)

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = {k: v.clone().detach() for k, v in state_dict.items()}

    def apply(self, model):
        model.load_state_dict(self.shadow)


# ---------------------------------------------------------------------------
# 3. Data Loaders
# ---------------------------------------------------------------------------
def load_cifar_2d(seed, n_train, n_valtest, batch_size, img_size=32):
    from torchvision.datasets import CIFAR10
    torch.manual_seed(seed)
    data_train = CIFAR10(root=DATA_ROOT, train=True, download=True)
    data_test = CIFAR10(root=DATA_ROOT, train=False, download=True)
    X_tr = torch.tensor(data_train.data).float().permute(0, 3, 1, 2) / 255.0
    X_te = torch.tensor(data_test.data).float().permute(0, 3, 1, 2) / 255.0
    if img_size != 32:
        X_tr = F.interpolate(X_tr, size=(img_size, img_size), mode="bilinear",
                             align_corners=False)
        X_te = F.interpolate(X_te, size=(img_size, img_size), mode="bilinear",
                             align_corners=False)
    X_tr = X_tr[torch.randperm(len(X_tr))[:n_train]]
    X_te = X_te[torch.randperm(len(X_te))[:n_valtest]]
    return _make_gen_loaders(X_tr, X_te, batch_size)


def load_mnist_2d(seed, n_train, n_valtest, batch_size, img_size=32):
    from torchvision.datasets import MNIST
    torch.manual_seed(seed)
    data_train = MNIST(root=DATA_ROOT, train=True, download=True)
    data_test = MNIST(root=DATA_ROOT, train=False, download=True)
    X_tr = data_train.data[:n_train].float().unsqueeze(1) / 255.0
    X_te = data_test.data[:n_valtest].float().unsqueeze(1) / 255.0
    X_tr = F.interpolate(X_tr, size=(img_size, img_size), mode="bilinear",
                         align_corners=False).repeat(1, 3, 1, 1)
    X_te = F.interpolate(X_te, size=(img_size, img_size), mode="bilinear",
                         align_corners=False).repeat(1, 3, 1, 1)
    X_tr = X_tr[torch.randperm(len(X_tr))]
    X_te = X_te[torch.randperm(len(X_te))]
    return _make_gen_loaders(X_tr, X_te, batch_size)


def load_fashion_2d(seed, n_train, n_valtest, batch_size, img_size=32):
    from torchvision.datasets import FashionMNIST
    torch.manual_seed(seed)
    data_train = FashionMNIST(root=DATA_ROOT, train=True, download=True)
    data_test = FashionMNIST(root=DATA_ROOT, train=False, download=True)
    X_tr = data_train.data[:n_train].float().unsqueeze(1) / 255.0
    X_te = data_test.data[:n_valtest].float().unsqueeze(1) / 255.0
    X_tr = F.interpolate(X_tr, size=(img_size, img_size), mode="bilinear",
                         align_corners=False).repeat(1, 3, 1, 1)
    X_te = F.interpolate(X_te, size=(img_size, img_size), mode="bilinear",
                         align_corners=False).repeat(1, 3, 1, 1)
    X_tr = X_tr[torch.randperm(len(X_tr))]
    X_te = X_te[torch.randperm(len(X_te))]
    return _make_gen_loaders(X_tr, X_te, batch_size)


class LazyCOCODataset(Dataset):
    """Loads COCO images on demand to avoid OOM with large datasets."""

    def __init__(self, img_paths, img_size=128):
        from torchvision import transforms
        self.img_paths = img_paths
        self.tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        from PIL import Image
        img = Image.open(self.img_paths[idx]).convert("RGB")
        x = self.tf(img)
        return (x,)


def load_coco_2d(seed, n_train, n_valtest, batch_size, img_size=32):
    torch.manual_seed(seed)
    np.random.seed(seed)
    data_dir = os.path.join(DATA_ROOT, "coco")
    ann_file = os.path.join(data_dir, "annotations/instances_train2017.json")
    img_dir = os.path.join(data_dir, "train2017")
    try:
        from pycocotools.coco import COCO
        coco = COCO(ann_file)
        img_ids = sorted(coco.getImgIds())
        rng = np.random.RandomState(seed)
        rng.shuffle(img_ids)
        img_ids = img_ids[:n_train + n_valtest]
        paths = []
        for iid in img_ids:
            info = coco.loadImgs(iid)[0]
            paths.append(os.path.join(img_dir, info["file_name"]))
        train_ds = LazyCOCODataset(paths[:n_train], img_size)
        valtest_ds = LazyCOCODataset(paths[n_train:n_train + n_valtest],
                                     img_size)
        val_sz = len(valtest_ds) // 2
        test_sz = len(valtest_ds) - val_sz
        val_ds, test_ds = random_split(valtest_ds, [val_sz, test_sz])
        train_loader = DataLoader(train_ds, batch_size=batch_size,
                                  shuffle=True, drop_last=True,
                                  num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                                num_workers=2, pin_memory=True)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                                 num_workers=2, pin_memory=True)
        return train_loader, val_loader, test_loader
    except Exception as e:
        print(f"  [WARN] COCO loading failed ({e}), using synthetic data")
        X_tr = torch.rand(n_train, 3, img_size, img_size)
        X_te = torch.rand(n_valtest, 3, img_size, img_size)
        return _make_gen_loaders(X_tr, X_te, batch_size)


def _make_gen_loaders(X_train, X_valtest, batch_size):
    train_ds = TensorDataset(X_train)
    valtest_ds = TensorDataset(X_valtest)
    val_sz = len(valtest_ds) // 2
    test_sz = len(valtest_ds) - val_sz
    val_ds, test_ds = random_split(valtest_ds, [val_sz, test_sz])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


# ---------------------------------------------------------------------------
# 4. Classical Encoder / Decoder
# ---------------------------------------------------------------------------
class ConvEncoder(nn.Module):
    """Lightweight Conv2d encoder: (3, H, W) -> flat feature vector.

    Default 2 layers for 32x32:
      (3,32,32) -> (32,16,16) -> (64,8,8) -> flatten -> 4096
    """

    def __init__(self, in_channels=3, channels=(32, 64), img_size=32):
        super().__init__()
        layers = []
        ch_in = in_channels
        for ch_out in channels:
            layers.append(nn.Conv2d(ch_in, ch_out, 4, 2, 1))
            layers.append(nn.SiLU())
            ch_in = ch_out
        self.net = nn.Sequential(*layers)

        n_down = len(channels)
        spatial = img_size // (2 ** n_down)
        self.flat_dim = channels[-1] * spatial * spatial
        self.out_channels = channels[-1]
        self.out_spatial = spatial

    def forward(self, x):
        h = self.net(x)
        return h.view(h.size(0), -1)


class ConvDecoder(nn.Module):
    """Lightweight ConvTranspose2d decoder: flat vector -> (3, H, W).

    Default 2 layers from spatial 8:
      (64,8,8) -> (32,16,16) -> (3,32,32)
    """

    def __init__(self, out_channels=3, channels=(64, 32),
                 start_spatial=8):
        super().__init__()
        self.start_channels = channels[0]
        self.start_spatial = start_spatial
        self.flat_dim = channels[0] * start_spatial * start_spatial

        layers = []
        for i in range(len(channels) - 1):
            layers.append(nn.ConvTranspose2d(channels[i], channels[i + 1],
                                             4, 2, 1))
            layers.append(nn.SiLU())
        # Final layer: no activation (velocity can be any value)
        layers.append(nn.ConvTranspose2d(channels[-1], out_channels, 4, 2, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x_flat):
        h = x_flat.view(-1, self.start_channels, self.start_spatial,
                        self.start_spatial)
        return self.net(h)


# ---------------------------------------------------------------------------
# 5. Quantum Circuit — Single SU(2^n) Gate + Multi-Observable Global ANO
# ---------------------------------------------------------------------------
class GlobalANOQuantumCircuit(nn.Module):
    """Quantum circuit with single SU(2^n) encoding and global ANO.

    Unlike v2's pairwise ANO (k=2, C(n,2) wire groups), this uses
    m independent 2^n x 2^n learnable Hermitian matrices, each measuring
    the FULL n-qubit quantum state. All observables share the same wire
    group (all qubits), but each has independent learnable parameters.

    For n=6, m=15:
      - Each Hermitian: 64x64 matrix, 4096 real params
      - Total ANO params: 15 * 4096 = 61,440
      - Output: 15 scalar expectation values
    """

    def __init__(self, input_dim, n_qubits, n_observables, vqc_type,
                 vqc_depth, qvit_circuit="butterfly", circuit_id=0):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_observables = n_observables
        self.vqc_type = vqc_type
        self.vqc_depth = vqc_depth
        self.circuit_id = circuit_id

        # Single-gate encoding: all qubits, one SU(2^n) gate
        wires, n_gen, enc_size = compute_single_gate_encoding(n_qubits)
        self.enc_wires = wires
        self.n_generators = n_gen
        self.enc_per_block = enc_size

        # Projection: input_dim -> enc_size
        proj_hidden = max(256, enc_size // 2)
        self.enc_proj = nn.Sequential(
            nn.Linear(input_dim, proj_hidden),
            nn.SiLU(),
            nn.Linear(proj_hidden, enc_size),
        )

        # VQC (optional -- default none for single-gate encoding)
        if vqc_type == "qvit":
            n_rbs = _qvit_n_params(n_qubits, qvit_circuit)
            self.qvit_params = nn.Parameter(
                0.01 * torch.randn(vqc_depth, n_rbs, 12))
        elif vqc_type == "hardware_efficient":
            self.var_params = nn.Parameter(
                0.01 * torch.randn(vqc_depth, n_qubits))

        # Global ANO: m independent Hermitians on ALL qubits
        K = 2 ** n_qubits  # Hermitian matrix size (e.g., 64 for 6 qubits)
        self.obs_dim = K
        self.n_obs = n_observables
        n_off = (K * (K - 1)) // 2
        self.A = nn.ParameterList(
            [nn.Parameter(torch.empty(n_off)) for _ in range(n_observables)])
        self.B = nn.ParameterList(
            [nn.Parameter(torch.empty(n_off)) for _ in range(n_observables)])
        self.D = nn.ParameterList(
            [nn.Parameter(torch.empty(K)) for _ in range(n_observables)])
        for w in range(n_observables):
            nn.init.normal_(self.A[w], std=2.0)
            nn.init.normal_(self.B[w], std=2.0)
            nn.init.normal_(self.D[w], std=2.0)

        ano_params = n_observables * (2 * n_off + K)
        print(f"  [Circuit {circuit_id}] Global ANO: {n_observables} "
              f"independent {K}x{K} Hermitians, "
              f"{ano_params:,} total ANO params")

        # SU gate: use PennyLane for n<=5, custom TorchSUGate for n>=6
        self._use_custom_su = (n_qubits >= 6)
        if self._use_custom_su:
            precompute = n_qubits <= 6
            self._su_gate = TorchSUGate(n_qubits, precompute)
            print(f"  [Circuit {circuit_id}] Using custom TorchSUGate for "
                  f"SU({2**n_qubits}) (n={n_qubits}, "
                  f"{n_gen} generators, "
                  f"{'precomputed' if precompute else 'streaming'})")
        else:
            self._su_gate = None
            print(f"  [Circuit {circuit_id}] Using PennyLane SpecialUnitary "
                  f"for SU({2**n_qubits}) (n={n_qubits}, "
                  f"{n_gen} generators)")

        # Build QNode
        dev = qml.device("default.qubit")
        _nq = n_qubits
        _vt = vqc_type
        _vd = vqc_depth
        _no = n_observables
        _qc = qvit_circuit
        _ew = wires
        _ng = n_gen
        _custom_su = self._use_custom_su

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def _circuit(enc, vqc_params, H_mats, *su_unitaries):
            def _qvit_gate(p, w1, w2):
                qml.U3(p[0], p[1], p[2], wires=w1)
                qml.IsingXX(p[3], wires=[w1, w2])
                qml.IsingYY(p[4], wires=[w1, w2])
                qml.IsingZZ(p[5], wires=[w1, w2])
                qml.U3(p[6], p[7], p[8], wires=w1)
                qml.IsingXX(p[9], wires=[w1, w2])
                qml.IsingYY(p[10], wires=[w1, w2])
                qml.IsingZZ(p[11], wires=[w1, w2])

            def _qvit_layer(params_ly):
                pidx = 0
                if _qc == "butterfly":
                    n_layers = int(math.ceil(math.log2(_nq)))
                    for layer in range(n_layers):
                        stride = 2 ** layer
                        for start in range(0, _nq - stride, 2 * stride):
                            for offset in range(stride):
                                w1 = start + offset
                                w2 = start + offset + stride
                                if w1 < _nq and w2 < _nq:
                                    _qvit_gate(params_ly[pidx], w1, w2)
                                    pidx += 1
                elif _qc == "pyramid":
                    for layer in range(_nq - 1):
                        for i in range(_nq - layer - 1):
                            _qvit_gate(params_ly[pidx], i, i + 1)
                            pidx += 1
                elif _qc == "x":
                    for i in range(_nq // 2):
                        w1, w2 = i, _nq - 1 - i
                        _qvit_gate(params_ly[pidx], w1, w2)
                        pidx += 1
                    for i in range(_nq // 2 - 1):
                        _qvit_gate(params_ly[pidx], i, i + 1)
                        pidx += 1

            def _hwe_layer(params_ly):
                for q in range(_nq):
                    qml.RY(params_ly[q], wires=q)
                for q in range(0, _nq - 1, 2):
                    qml.CNOT(wires=[q, q + 1])
                for q in range(1, _nq - 1, 2):
                    qml.CNOT(wires=[q, q + 1])

            # -- Encoding: single SU(2^n) gate on all qubits --
            if _custom_su:
                qml.QubitUnitary(su_unitaries[0], wires=_ew)
            else:
                qml.SpecialUnitary(enc[..., :_ng], wires=_ew)

            # -- VQC layers (optional) --
            if _vt == "qvit":
                for ly in range(_vd):
                    _qvit_layer(vqc_params[ly])
            elif _vt == "hardware_efficient":
                for ly in range(_vd):
                    _hwe_layer(vqc_params[ly])

            # -- Measurement: m independent global Hermitians --
            return [qml.expval(qml.Hermitian(H_mats[w], wires=_ew))
                    for w in range(_no)]

        self._circuit = _circuit

    def _build_H_matrices(self):
        return [create_Hermitian(self.obs_dim, self.A[w], self.B[w], self.D[w])
                for w in range(self.n_obs)]

    def forward(self, x):
        enc = self.enc_proj(x)
        H_mats = self._build_H_matrices()
        if self.vqc_type == "qvit":
            vqc_p = self.qvit_params
        elif self.vqc_type == "hardware_efficient":
            vqc_p = self.var_params
        else:
            vqc_p = torch.zeros(1)

        if self._use_custom_su:
            U = self._su_gate.matrix(enc)
            q_out = self._circuit(enc, vqc_p, H_mats, U)
        else:
            q_out = self._circuit(enc, vqc_p, H_mats)

        return torch.stack(q_out, dim=1).float()

    def get_eigenvalue_range(self):
        H_mats = self._build_H_matrices()
        lo, hi = float("inf"), float("-inf")
        for H in H_mats:
            eigs = torch.linalg.eigvalsh(
                H.detach().cpu().to(torch.complex128)).real
            lo = min(lo, eigs.min().item())
            hi = max(hi, eigs.max().item())
        return lo, hi


# ---------------------------------------------------------------------------
# 5b. Pixel-Space Quantum Velocity Field
# ---------------------------------------------------------------------------
class PixelQuantumVelocityField(nn.Module):
    """Quantum velocity field with single-gate SU(2^n) + global ANO.

    Architecture:
      ConvEncoder(x_t) -> flat features (d_flat)
      time_mlp projects t -> d_flat, added to flat features
      single SU(2^n) encoding -> optional QViT -> global ANO -> m observables
      vel_head -> flat features -> ConvDecoder -> pixel velocity
    """

    def __init__(self, img_channels, img_size, enc_channels, n_circuits,
                 n_qubits, n_observables, vqc_type, vqc_depth,
                 qvit_circuit="butterfly", time_embed_dim=256):
        super().__init__()
        self.img_channels = img_channels
        self.img_size = img_size
        self.time_embed_dim = time_embed_dim
        self.n_circuits = n_circuits
        self.n_qubits = n_qubits

        # Classical encoder
        self.encoder = ConvEncoder(in_channels=img_channels,
                                   channels=enc_channels,
                                   img_size=img_size)
        d_flat = self.encoder.flat_dim

        # Time MLP: sinusoidal(t) -> project to d_flat for additive
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, d_flat),
            nn.SiLU(),
            nn.Linear(d_flat, d_flat),
        )

        # Quantum circuits -- input_dim = d_flat (time added, not concatenated)
        self.circuits = nn.ModuleList()
        for k in range(n_circuits):
            self.circuits.append(GlobalANOQuantumCircuit(
                input_dim=d_flat, n_qubits=n_qubits,
                n_observables=n_observables,
                vqc_type=vqc_type, vqc_depth=vqc_depth,
                qvit_circuit=qvit_circuit, circuit_id=k))

        self.n_obs_per_circuit = self.circuits[0].n_obs
        self.total_obs = sum(c.n_obs for c in self.circuits)

        # Velocity head: observables -> flat features
        _vh = max(256, self.total_obs)
        self.vel_head = nn.Sequential(
            nn.Linear(self.total_obs, _vh),
            nn.SiLU(),
            nn.Linear(_vh, d_flat),
        )

        # Classical decoder
        dec_channels = list(reversed(enc_channels))
        self.decoder = ConvDecoder(
            out_channels=img_channels, channels=dec_channels,
            start_spatial=self.encoder.out_spatial)

    def _time_embedding(self, t):
        half = self.time_embed_dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device) / half)
        args = t.unsqueeze(-1) * freqs
        return torch.cat([args.cos(), args.sin()], dim=-1)

    def forward(self, x_t, t):
        # Encode image to flat features
        h = self.encoder(x_t)

        # Time embedding -- additive conditioning
        t_emb = self.time_mlp(self._time_embedding(t))
        z_combined = h + t_emb

        # Quantum circuits
        q_outputs = []
        for k in range(self.n_circuits):
            q_outputs.append(self.circuits[k](z_combined))
        q_all = torch.cat(q_outputs, dim=1)

        # Velocity head -> flat -> decode to pixel velocity
        v_flat = self.vel_head(q_all)
        v_img = self.decoder(v_flat)
        return v_img

    def get_eigenvalue_range(self):
        lo, hi = float("inf"), float("-inf")
        for circ in self.circuits:
            c_lo, c_hi = circ.get_eigenvalue_range()
            lo = min(lo, c_lo)
            hi = max(hi, c_hi)
        return lo, hi


# ---------------------------------------------------------------------------
# 5c. Pixel-Space Classical Velocity Field
# ---------------------------------------------------------------------------
class PixelClassicalVelocityField(nn.Module):
    """Classical MLP velocity field in pixel space (baseline)."""

    def __init__(self, img_channels, img_size, enc_channels,
                 hidden_dims=(256, 256, 256), time_embed_dim=256):
        super().__init__()
        self.img_channels = img_channels
        self.img_size = img_size
        self.time_embed_dim = time_embed_dim

        self.encoder = ConvEncoder(in_channels=img_channels,
                                   channels=enc_channels,
                                   img_size=img_size)
        d_flat = self.encoder.flat_dim

        # Time MLP: sinusoidal(t) -> project to d_flat for additive
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, d_flat),
            nn.SiLU(),
            nn.Linear(d_flat, d_flat),
        )

        # MLP core -- input is d_flat (time added, not concatenated)
        dims = [d_flat] + list(hidden_dims)
        layers = []
        for i in range(len(dims) - 1):
            layers += [nn.Linear(dims[i], dims[i + 1]), nn.SiLU()]
        layers.append(nn.Linear(dims[-1], d_flat))
        self.net = nn.Sequential(*layers)

        dec_channels = list(reversed(enc_channels))
        self.decoder = ConvDecoder(
            out_channels=img_channels, channels=dec_channels,
            start_spatial=self.encoder.out_spatial)

    def _time_embedding(self, t):
        half = self.time_embed_dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device) / half)
        args = t.unsqueeze(-1) * freqs
        return torch.cat([args.cos(), args.sin()], dim=-1)

    def forward(self, x_t, t):
        h = self.encoder(x_t)
        t_emb = self.time_mlp(self._time_embedding(t))
        v_flat = self.net(h + t_emb)
        v_img = self.decoder(v_flat)
        return v_img

    def get_eigenvalue_range(self):
        return 0.0, 0.0


# ---------------------------------------------------------------------------
# 5d. Build velocity field
# ---------------------------------------------------------------------------
def resolve_enc_channels(args):
    """Resolve enc_channels: 'auto' computes based on img_size and n_qubits."""
    if args.enc_channels == "auto":
        return auto_enc_channels(args.img_size, args.n_qubits)
    return tuple(int(c) for c in args.enc_channels.split(","))


def build_velocity_field(args, device):
    enc_channels = resolve_enc_channels(args)

    if args.velocity_field == "classical":
        hidden = [int(d) for d in args.mlp_hidden_dims.split(",")]
        vf = PixelClassicalVelocityField(
            img_channels=3, img_size=args.img_size,
            enc_channels=enc_channels, hidden_dims=hidden,
            time_embed_dim=args.time_embed_dim).to(device)
    else:
        vf = PixelQuantumVelocityField(
            img_channels=3, img_size=args.img_size,
            enc_channels=enc_channels, n_circuits=args.n_circuits,
            n_qubits=args.n_qubits,
            n_observables=args.n_observables,
            vqc_type=args.vqc_type, vqc_depth=args.vqc_depth,
            qvit_circuit=args.qvit_circuit,
            time_embed_dim=args.time_embed_dim).to(device)
    return vf


# ---------------------------------------------------------------------------
# 6. Training -- Single-phase CFM (no VAE pretraining)
# ---------------------------------------------------------------------------
def train_cfm(args, train_loader, val_loader, device):
    """Train the velocity field directly in pixel space."""

    vf = build_velocity_field(args, device)

    # Dual optimizer for quantum (Chen 2025)
    H_params, other_params = [], []
    for name, param in vf.named_parameters():
        if any(tag in name for tag in (".A.", ".B.", ".D.")):
            H_params.append(param)
        else:
            other_params.append(param)

    main_opt = torch.optim.Adam(other_params, lr=args.lr)
    H_opt = torch.optim.Adam(H_params, lr=args.lr_H) if H_params else None

    from torch.optim.lr_scheduler import CosineAnnealingLR
    main_sched = CosineAnnealingLR(main_opt, T_max=args.epochs)
    H_sched = CosineAnnealingLR(H_opt, T_max=args.epochs) if H_opt else None

    # EMA
    vf_ema = None
    if args.vf_ema_decay > 0:
        vf_ema = EMAModel(vf, decay=args.vf_ema_decay)

    # Parameter breakdown
    total_p = sum(p.numel() for p in vf.parameters() if p.requires_grad)
    h_p = sum(p.numel() for p in H_params)
    other_p = sum(p.numel() for p in other_params)
    print(f"[Train] VF params: total={total_p:,}  "
          f"main={other_p:,}  observable={h_p:,}")

    if args.velocity_field == "quantum":
        sun_n = 2 ** args.n_qubits
        enc_per = vf.circuits[0].enc_per_block
        d_flat = vf.encoder.flat_dim

        # Quantum-specific parameter breakdown
        enc_proj_params = sum(p.numel() for n, p in vf.named_parameters()
                              if "enc_proj" in n and "circuits" in n)
        vqc_params = sum(p.numel() for n, p in vf.named_parameters()
                         if "qvit_params" in n or "var_params" in n)
        q_total = enc_proj_params + vqc_params + h_p
        enc_dec_params = sum(p.numel() for n, p in vf.named_parameters()
                             if "encoder" in n or "decoder" in n)
        print(f"  Encoder flat_dim: {d_flat}")
        print(f"  Conv encoder/decoder params: {enc_dec_params:,}")
        print(f"  Quantum params (enc_proj+VQC+ANO): {q_total:,}")
        print(f"  Architecture: K={args.n_circuits} circuit(s), "
              f"{args.n_qubits}q each")
        print(f"  SU encoding: single SU({sun_n}) gate on all "
              f"{args.n_qubits} qubits, {enc_per} generators")
        print(f"  enc_proj ratio: {d_flat} -> {enc_per} "
              f"({d_flat/enc_per:.1f}:1)")
        print(f"  VQC: {args.vqc_type}"
              + (f" ({args.qvit_circuit}), depth={args.vqc_depth}"
                 if args.vqc_type != "none" else ""))
        print(f"  Global ANO: {args.n_observables} independent "
              f"{2**args.n_qubits}x{2**args.n_qubits} Hermitians, "
              f"total_obs={vf.total_obs}")
        print(f"  ANO observable params: {h_p:,}")
    else:
        d_flat = vf.encoder.flat_dim
        print(f"  Encoder flat_dim: {d_flat}")
        print(f"  Classical MLP: hidden_dims={args.mlp_hidden_dims}")

    if args.logit_normal_std > 0:
        print(f"  Logit-normal timestep sampling "
              f"(std={args.logit_normal_std})")
    print(f"  ODE solver: {args.ode_solver} ({args.ode_steps} steps)")
    if vf_ema:
        print(f"  VF EMA enabled (decay={args.vf_ema_decay})")

    # Checkpointing
    ckpt_dir = os.path.join(args.base_path, "checkpoints")
    results_dir = os.path.join(args.base_path, "results")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f"ckpt_dcfm3_{args.job_id}.pt")
    csv_path = os.path.join(results_dir, f"log_dcfm3_{args.job_id}.csv")
    fields = ["epoch", "train_loss", "val_loss", "val_loss_ema",
              "eig_min", "eig_max", "time_s"]

    start_epoch = 0
    best_val, best_state = float("inf"), None

    if args.resume and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, weights_only=False)
        vf.load_state_dict(ckpt["model"])
        main_opt.load_state_dict(ckpt["main_opt"])
        if H_opt and "H_opt" in ckpt:
            H_opt.load_state_dict(ckpt["H_opt"])
        if "main_sched" in ckpt:
            main_sched.load_state_dict(ckpt["main_sched"])
        if H_sched and "H_sched" in ckpt:
            H_sched.load_state_dict(ckpt["H_sched"])
        if vf_ema and "vf_ema" in ckpt:
            vf_ema.load_state_dict(ckpt["vf_ema"])
        start_epoch = ckpt["epoch"] + 1
        best_val = ckpt.get("best_val", float("inf"))
        print(f"  Resumed from epoch {start_epoch}")

    if start_epoch == 0:
        with open(csv_path, "w", newline="") as f:
            csv.DictWriter(f, fields).writeheader()

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        # -- Train --
        vf.train()
        tr_loss, tr_n = 0.0, 0
        for (xb,) in tqdm(train_loader,
                          desc=f"Ep {epoch+1}/{args.epochs}", leave=False):
            xb = xb.to(device)

            x_0 = torch.randn_like(xb)
            x_1 = xb

            if args.logit_normal_std > 0:
                t = torch.sigmoid(
                    torch.randn(xb.size(0), device=device)
                    * args.logit_normal_std)
            else:
                t = torch.rand(xb.size(0), device=device)

            t_view = t[:, None, None, None]
            x_t = (1.0 - t_view) * x_0 + t_view * x_1
            target = x_1 - x_0

            v_pred = vf(x_t, t)
            loss = F.mse_loss(v_pred, target)

            main_opt.zero_grad()
            if H_opt:
                H_opt.zero_grad()
            loss.backward()
            main_opt.step()
            if H_opt:
                H_opt.step()

            if vf_ema:
                vf_ema.update(vf)

            bs = xb.size(0)
            tr_loss += loss.item() * bs
            tr_n += bs

        tr_loss /= tr_n

        # -- Val --
        vf.eval()
        vl_loss, vl_n = 0.0, 0
        with torch.no_grad():
            for (xb,) in val_loader:
                xb = xb.to(device)
                x_0 = torch.randn_like(xb)
                x_1 = xb
                t = torch.rand(xb.size(0), device=device)
                t_view = t[:, None, None, None]
                x_t = (1.0 - t_view) * x_0 + t_view * x_1
                target = x_1 - x_0
                v_pred = vf(x_t, t)
                loss = F.mse_loss(v_pred, target)
                vl_loss += loss.item() * xb.size(0)
                vl_n += xb.size(0)
        vl_loss /= vl_n

        # EMA validation
        vl_loss_ema = vl_loss
        orig_state = None
        if vf_ema:
            orig_state = copy.deepcopy(vf.state_dict())
            vf_ema.apply(vf)
            vf.eval()
            vl_ema, vl_ema_n = 0.0, 0
            with torch.no_grad():
                for (xb,) in val_loader:
                    xb = xb.to(device)
                    x_0 = torch.randn_like(xb)
                    x_1 = xb
                    t = torch.rand(xb.size(0), device=device)
                    t_view = t[:, None, None, None]
                    x_t = (1.0 - t_view) * x_0 + t_view * x_1
                    target = x_1 - x_0
                    v_pred = vf(x_t, t)
                    loss = F.mse_loss(v_pred, target)
                    vl_ema += loss.item() * xb.size(0)
                    vl_ema_n += xb.size(0)
            vl_loss_ema = vl_ema / vl_ema_n
            vf.load_state_dict(orig_state)

        eig_lo, eig_hi = vf.get_eigenvalue_range()
        dt = time.time() - t0

        row = dict(epoch=epoch + 1, train_loss=f"{tr_loss:.6f}",
                   val_loss=f"{vl_loss:.6f}",
                   val_loss_ema=f"{vl_loss_ema:.6f}",
                   eig_min=f"{eig_lo:.4f}",
                   eig_max=f"{eig_hi:.4f}", time_s=f"{dt:.1f}")
        with open(csv_path, "a", newline="") as f:
            csv.DictWriter(f, fields).writerow(row)

        ema_str = f" EMA={vl_loss_ema:.6f}" if vf_ema else ""
        print(f"  Ep {epoch+1:3d} | Train {tr_loss:.6f} | Val {vl_loss:.6f}"
              f"{ema_str} | Eig [{eig_lo:.2f}, {eig_hi:.2f}] | {dt:.1f}s")

        # Track best
        track_val = vl_loss_ema if vf_ema else vl_loss
        if track_val < best_val:
            best_val = track_val
            if vf_ema:
                vf_ema.apply(vf)
                best_state = copy.deepcopy(vf.state_dict())
                vf.load_state_dict(orig_state)
            else:
                best_state = copy.deepcopy(vf.state_dict())

        # LR step
        main_sched.step()
        if H_sched:
            H_sched.step()

        # Checkpoint
        ckpt_data = dict(epoch=epoch, model=vf.state_dict(),
                         main_opt=main_opt.state_dict(),
                         main_sched=main_sched.state_dict(),
                         best_val=best_val)
        if H_opt:
            ckpt_data["H_opt"] = H_opt.state_dict()
        if H_sched:
            ckpt_data["H_sched"] = H_sched.state_dict()
        if vf_ema:
            ckpt_data["vf_ema"] = vf_ema.state_dict()
        torch.save(ckpt_data, ckpt_path)

    # Load best weights
    if best_state is not None:
        vf.load_state_dict(best_state)

    w_path = os.path.join(ckpt_dir, f"weights_dcfm3_{args.job_id}.pt")
    torch.save(vf.state_dict(), w_path)
    print(f"  VF saved to {w_path}")
    return vf


# ---------------------------------------------------------------------------
# 7. Generation (ODE integration in pixel space)
# ---------------------------------------------------------------------------
@torch.no_grad()
def generate_samples(vf, n_samples, img_channels, img_size, ode_steps,
                     device, save_path=None, ode_solver="midpoint"):
    """Generate images via ODE integration directly in pixel space."""
    vf.eval()

    x = torch.randn(n_samples, img_channels, img_size, img_size, device=device)
    dt = 1.0 / ode_steps

    for step in range(ode_steps):
        t_val = step * dt
        t = torch.full((n_samples,), t_val, device=device)

        if ode_solver == "midpoint":
            k1 = vf(x, t)
            x_mid = x + 0.5 * dt * k1
            t_mid = torch.full((n_samples,), t_val + 0.5 * dt, device=device)
            k2 = vf(x_mid, t_mid)
            x = x + dt * k2
        else:
            v = vf(x, t)
            x = x + dt * v

    images = x.clamp(0, 1)

    if save_path:
        from torchvision.utils import save_image
        save_image(images, save_path, nrow=8, padding=2)
        print(f"  Generated {n_samples} samples -> {save_path}")

    return images


# ---------------------------------------------------------------------------
# 7b. FID & IS Evaluation
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate_fid_is(vf, real_loader, n_samples, img_channels, img_size,
                    ode_steps, device, batch_size=64, ode_solver="midpoint"):
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image.inception import InceptionScore

    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    inception_score = InceptionScore(normalize=True).to(device)

    n_real = 0
    for (xb,) in real_loader:
        xb = xb.to(device)
        fid.update(xb, real=True)
        n_real += xb.size(0)
        if n_real >= n_samples:
            break

    n_gen = 0
    while n_gen < n_samples:
        bs = min(batch_size, n_samples - n_gen)
        imgs = generate_samples(vf, bs, img_channels, img_size, ode_steps,
                                device, ode_solver=ode_solver)
        fid.update(imgs, real=False)
        inception_score.update(imgs)
        n_gen += bs

    fid_val = fid.compute().item()
    is_mean, is_std = inception_score.compute()

    print(f"  FID = {fid_val:.2f}")
    print(f"  IS  = {is_mean.item():.2f} +/- {is_std.item():.2f}")
    return {"fid": fid_val, "is_mean": is_mean.item(), "is_std": is_std.item()}


# ---------------------------------------------------------------------------
# 8. Main
# ---------------------------------------------------------------------------
def main():
    args = get_args()
    set_all_seeds(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"PennyLane {qml.__version__}  |  PyTorch {torch.__version__}  |  "
          f"{device}")

    # Resolve auto enc_channels early for display
    enc_channels = resolve_enc_channels(args)
    # Store resolved channels back so build_velocity_field uses them
    args.enc_channels = ",".join(str(c) for c in enc_channels)

    if args.velocity_field == "quantum":
        sun_n = 2 ** args.n_qubits
        label = (f"Quantum {args.n_circuits}x{args.n_qubits}q "
                 f"SU({sun_n})")
    else:
        label = "Classical"
    print(f"=== Quantum Direct CFM v3 (Single-Gate SU + Global ANO) ===")
    print(f"Dataset: {args.dataset}  |  img_size: {args.img_size}  |  "
          f"VF: {label}")
    print(f"Encoder channels: {list(enc_channels)}")

    if args.velocity_field == "quantum":
        sun_n = 2 ** args.n_qubits
        wires, n_gen, enc_size = compute_single_gate_encoding(args.n_qubits)
        K = 2 ** args.n_qubits
        n_off = K * (K - 1) // 2
        ano_per_obs = 2 * n_off + K
        ano_total = args.n_observables * ano_per_obs
        print(f"Single SU({sun_n}) encoding on {args.n_qubits} qubits:")
        print(f"  Gate acts on all wires: {wires}")
        print(f"  Generators: {n_gen}")
        print(f"  Encoding params: {enc_size}")
        print(f"Global ANO: {args.n_observables} x ({K}x{K}) Hermitians")
        print(f"  Params per observable: {ano_per_obs:,}")
        print(f"  Total ANO params: {ano_total:,}")

    # -- Load data --
    loader_map = {
        "cifar10": load_cifar_2d,
        "coco": load_coco_2d,
        "mnist": load_mnist_2d,
        "fashion": load_fashion_2d,
    }
    train_loader, val_loader, test_loader = loader_map[args.dataset](
        seed=args.seed, n_train=args.n_train, n_valtest=args.n_valtest,
        batch_size=args.batch_size, img_size=args.img_size)

    print(f"Data: train={args.n_train}  valtest={args.n_valtest}")

    # -- Train --
    print(f"\n=== Training {label} Direct CFM v3 ===")
    vf = train_cfm(args, train_loader, val_loader, device)

    # -- Generate sample grid --
    img_path = os.path.join(args.base_path, "results",
                            f"samples_{args.job_id}.png")
    generate_samples(vf, min(args.n_samples, 64), 3, args.img_size,
                     args.ode_steps, device, save_path=img_path,
                     ode_solver=args.ode_solver)

    # -- Metrics --
    if args.compute_metrics:
        print("\n=== Computing FID & IS ===")
        metrics = evaluate_fid_is(
            vf, train_loader, args.n_eval_samples, 3, args.img_size,
            args.ode_steps, device, ode_solver=args.ode_solver)
        import json
        metrics_path = os.path.join(args.base_path, "results",
                                    f"metrics_{args.job_id}.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"  Metrics saved to {metrics_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
