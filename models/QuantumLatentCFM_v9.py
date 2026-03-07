"""
Quantum Latent Conditional Flow Matching v9 -- CFM Training Improvements
=========================================================================

Based on QuantumLatentCFM_v6.py with three targeted improvements to the
CFM training and inference pipeline:

  1. LOGIT-NORMAL TIMESTEP SAMPLING (Esser et al., 2024 / SD3):
     t ~ sigmoid(Normal(0, 1)) instead of t ~ Uniform(0, 1).
     Concentrates training on mid-range timesteps (t ~ 0.3-0.7) where the
     velocity field is hardest to predict, making better use of the quantum
     circuit's limited capacity.

  2. MIDPOINT ODE SOLVER (2nd-order, replaces Euler):
     k1 = v(z, t)
     k2 = v(z + 0.5*dt*k1, t + 0.5*dt)
     z  = z + dt * k2
     Same cost as 50 Euler steps (2 evaluations per step with 50 steps)
     but 2nd-order accuracy. Default: 50 midpoint steps = 100 VF evals.

  3. EMA FOR VELOCITY FIELD:
     Maintains exponential moving average (decay=0.999) of VF weights during
     Phase 2 training. Uses EMA weights for validation, generation, and
     final checkpoint. Stabilizes noisy quantum gradients.

Everything else is IDENTICAL to v6: same VAE architectures, same quantum
circuit (SingleQuantumCircuit), same Classical VF, same dual optimizer,
same data loaders, same evaluation metrics.

Velocity field variants (same as v6):
  1. Classical-A (--velocity-field=classical): MLP baseline
  2. v9 (--velocity-field=quantum --n-circuits=1): single 8q circuit
  3. v9-multi (--velocity-field=quantum --n-circuits=8): 8x8q shared input

Usage:
  # v9 quantum (single circuit)
  python QuantumLatentCFM_v9.py --phase=both --dataset=cifar10 \\
      --velocity-field=quantum --n-circuits=1 --n-qubits=8 --epochs=200

  # v9 classical baseline
  python QuantumLatentCFM_v9.py --phase=both --dataset=cifar10 \\
      --velocity-field=classical --epochs=200

  # Load external VAE v5 weights
  python QuantumLatentCFM_v9.py --phase=2 --dataset=cifar10 \\
      --velocity-field=quantum --vae-ckpt=checkpoints/weights_vae_v5_ema_cifar10_XXX.pt \\
      --vae-arch=v5 --latent-dim=64 --c-z=4 --epochs=200

References:
  - Lipman et al. (2023). Flow Matching for Generative Modeling. ICLR 2023.
  - Esser et al. (2024). Scaling Rectified Flow Transformers. ICML 2024. (logit-normal)
  - Wiersema et al. (2024). Here comes the SU(N). Quantum, 8, 1275.
  - Chen et al. (2025). Learning to Measure QNNs. ICASSP 2025 Workshop.
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
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm
import scipy.constants  # noqa: F401 -- pre-import for PennyLane/scipy compat
import pennylane as qml

DATA_ROOT = "/pscratch/sd/j/junghoon/data"


# ---------------------------------------------------------------------------
# 1. Argparse
# ---------------------------------------------------------------------------
def get_args():
    p = argparse.ArgumentParser(
        description="Quantum Latent CFM v9 -- CFM Training Improvements")

    # Phase
    p.add_argument("--phase", type=str, default="1",
                   choices=["1", "2", "generate", "both"],
                   help="1=VAE pretrain, 2=CFM train, generate=sample, both=1+2")

    # VAE
    p.add_argument("--latent-dim", type=int, default=32)
    p.add_argument("--beta", type=float, default=0.5)
    p.add_argument("--beta-warmup-epochs", type=int, default=20)
    p.add_argument("--lambda-perc", type=float, default=0.1)
    p.add_argument("--vae-arch", type=str, default="resconv",
                   choices=["resconv", "legacy", "v5", "v6"],
                   help="VAE architecture: resconv, legacy, v5, or v6 (gradual reduction)")
    p.add_argument("--c-z", type=int, default=4,
                   help="Channel count for VAE v5 bottleneck (only used with --vae-arch=v5)")

    # Quantum circuit
    p.add_argument("--n-circuits", type=int, default=1)
    p.add_argument("--n-qubits", type=int, default=8)
    p.add_argument("--encoding-type", type=str, default="sun",
                   choices=["sun", "angle"])
    p.add_argument("--vqc-type", type=str, default="qvit",
                   choices=["qvit", "hardware_efficient", "none"])
    p.add_argument("--vqc-depth", type=int, default=2)
    p.add_argument("--qvit-circuit", type=str, default="butterfly",
                   choices=["butterfly", "pyramid", "x"])
    p.add_argument("--k-local", type=int, default=2)
    p.add_argument("--obs-scheme", type=str, default="pairwise",
                   choices=["sliding", "pairwise"])

    # Time embedding
    p.add_argument("--time-embed-dim", type=int, default=32)

    # Training
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lr-H", type=float, default=1e-1)
    p.add_argument("--lr-vae", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--seed", type=int, default=2025)

    # v9 improvements
    p.add_argument("--logit-normal-std", type=float, default=1.0,
                   help="Std of logit-normal timestep sampling "
                        "(0 = uniform, >0 = logit-normal)")
    p.add_argument("--ode-solver", type=str, default="midpoint",
                   choices=["euler", "midpoint"],
                   help="ODE solver for generation")
    p.add_argument("--vf-ema-decay", type=float, default=0.999,
                   help="EMA decay for velocity field (0 = disable)")

    # Data
    p.add_argument("--dataset", type=str, default="cifar10",
                   choices=["cifar10", "coco", "mnist", "fashion"])
    p.add_argument("--n-train", type=int, default=10000)
    p.add_argument("--n-valtest", type=int, default=2000)
    p.add_argument("--img-size", type=int, default=32)

    # ODE sampling
    p.add_argument("--ode-steps", type=int, default=50,
                   help="ODE steps (default 50 for midpoint = 100 VF evals)")
    p.add_argument("--n-samples", type=int, default=64)

    # I/O
    p.add_argument("--job-id", type=str, default="qlcfm_v9_001")
    p.add_argument("--base-path", type=str, default=".")
    p.add_argument("--vae-ckpt", type=str, default="")
    p.add_argument("--cfm-ckpt", type=str, default="")
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


def get_wire_groups(n_qubits, k_local, obs_scheme):
    if k_local <= 0:
        return [[q] for q in range(n_qubits)]
    if obs_scheme == "sliding":
        return [list(range(s, s + k_local))
                for s in range(n_qubits - k_local + 1)]
    elif obs_scheme == "pairwise":
        return [list(c) for c in combinations(range(n_qubits), k_local)]
    raise ValueError(f"Unknown obs_scheme: {obs_scheme}")


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
        return n_qubits * (n_qubits - 1) // 2
    elif circuit_type == "x":
        return n_qubits // 2 + max(0, n_qubits // 2 - 1)
    raise ValueError(f"Unknown qvit circuit_type: {circuit_type}")


def sinusoidal_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=t.device) / half)
    args = t[:, None].float() * freqs[None, :]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


def _compute_enc_per_block(n_qubits, encoding_type):
    if encoding_type == "sun":
        n_even = n_qubits // 2
        n_odd = (n_qubits - 1) // 2
        return (n_even + n_odd) * 15
    else:
        return n_qubits


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
# 3. Data Loaders (identical to v6)
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


def load_coco_2d(seed, n_train, n_valtest, batch_size, img_size=32):
    torch.manual_seed(seed)
    np.random.seed(seed)
    data_dir = os.path.join(DATA_ROOT, "coco")
    ann_file = os.path.join(data_dir, "annotations/instances_train2017.json")
    img_dir = os.path.join(data_dir, "train2017")
    try:
        from pycocotools.coco import COCO
        from PIL import Image
        from torchvision import transforms
        coco = COCO(ann_file)
        img_ids = sorted(coco.getImgIds())[:n_train + n_valtest]
        tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])
        imgs = []
        for iid in tqdm(img_ids, desc="Loading COCO", leave=False):
            info = coco.loadImgs(iid)[0]
            path = os.path.join(img_dir, info["file_name"])
            img = Image.open(path).convert("RGB")
            imgs.append(tf(img))
        X = torch.stack(imgs)
        X = X * 2.0 - 1.0  # normalize to [-1, 1]
        X_tr = X[:n_train]
        X_te = X[n_train:n_train + n_valtest]
    except Exception as e:
        print(f"  [WARN] COCO loading failed ({e}), using synthetic data")
        X_tr = torch.rand(n_train, 3, img_size, img_size) * 2.0 - 1.0
        X_te = torch.rand(n_valtest, 3, img_size, img_size) * 2.0 - 1.0
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
# 4. VAE Architectures (identical to v6 + v5 option)
# ---------------------------------------------------------------------------
class ConvVAE(nn.Module):
    """Legacy convolutional VAE for 32x32x3 images."""

    def __init__(self, latent_dim=32, in_channels=3):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(),
        )
        self.fc_mu = nn.Linear(128 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(128 * 4 * 4, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, 128 * 4 * 4)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.ConvTranspose2d(32, in_channels, 4, 2, 1), nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.encoder(x).view(x.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)

    def decode(self, z):
        return self.decoder(F.relu(self.fc_dec(z)).view(-1, 128, 4, 4))

    def forward(self, x):
        mu, logvar = self.encode(x)
        return self.decode(self.reparameterize(mu, logvar)), mu, logvar


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(channels), nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels), nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
        )

    def forward(self, x):
        return x + self.block(x)


class ResConvVAE(nn.Module):
    """Deep residual VAE for 32x32x3 images (~2.1M params)."""

    def __init__(self, latent_dim=32, in_channels=3):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, 1, 1, bias=False),
            ResidualBlock(32), ResidualBlock(32),
            nn.Conv2d(32, 64, 4, 2, 1, bias=False),
            ResidualBlock(64), ResidualBlock(64),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            ResidualBlock(128), ResidualBlock(128),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            ResidualBlock(256), ResidualBlock(256),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 4, 2, 1, bias=False),
        )
        flat_dim = 256 * 2 * 2
        self.fc_mu = nn.Linear(flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(flat_dim, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, flat_dim)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 256, 4, 2, 1, bias=False),
            ResidualBlock(256), ResidualBlock(256),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            ResidualBlock(128), ResidualBlock(128),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            ResidualBlock(64), ResidualBlock(64),
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            ResidualBlock(32), ResidualBlock(32),
            nn.Conv2d(32, in_channels, 3, 1, 1), nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.encoder(x).view(x.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)

    def decode(self, z):
        return self.decoder(F.relu(self.fc_dec(z)).view(-1, 256, 2, 2))

    def forward(self, x):
        mu, logvar = self.encode(x)
        return self.decode(self.reparameterize(mu, logvar)), mu, logvar


class ResBlock_v3(nn.Module):
    """GroupNorm(32) -> SiLU -> Conv3x3 -> GroupNorm(32) -> SiLU -> Conv3x3 + skip."""

    def __init__(self, in_channels, out_channels=None):
        super().__init__()
        out_channels = out_channels or in_channels
        self.block = nn.Sequential(
            nn.GroupNorm(32, in_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.GroupNorm(32, out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
        )
        self.skip = (nn.Conv2d(in_channels, out_channels, 1, bias=False)
                     if in_channels != out_channels else nn.Identity())

    def forward(self, x):
        return self.skip(x) + self.block(x)


class SelfAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels)
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        self.scale = channels ** -0.5

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        q = self.q(h).view(B, C, H * W)
        k = self.k(h).view(B, C, H * W)
        v = self.v(h).view(B, C, H * W)
        attn = torch.bmm(q.transpose(1, 2), k) * self.scale
        attn = F.softmax(attn, dim=-1)
        out = torch.bmm(v, attn.transpose(1, 2)).view(B, C, H, W)
        return x + self.proj(out)


class ResConvVAE_v5(nn.Module):
    """VAE v5: Stops at 4x4, uses 1x1 conv for channel reduction.

    Output in [0, 1] via Sigmoid (matching v6 data pipeline).
    For VAE v5 weights trained with Tanh [-1,1], rescale at load time.
    """

    def __init__(self, latent_dim=64, in_channels=3, c_z=4):
        super().__init__()
        self.latent_dim = latent_dim
        self.c_z = c_z
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, 1, 1, bias=False),
            ResBlock_v3(64), ResBlock_v3(64),
            nn.Conv2d(64, 64, 3, 2, 1, bias=False),
            ResBlock_v3(64, 128), ResBlock_v3(128),
            nn.Conv2d(128, 128, 3, 2, 1, bias=False),
            ResBlock_v3(128, 256), ResBlock_v3(256),
            SelfAttention(256),
            nn.Conv2d(256, 256, 3, 2, 1, bias=False),
            ResBlock_v3(256), ResBlock_v3(256),
            SelfAttention(256),
            nn.GroupNorm(32, 256), nn.SiLU(inplace=True),
            nn.Conv2d(256, c_z, 1, bias=False),
        )
        flat_dim = c_z * 4 * 4
        self.fc_mu = nn.Linear(flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(flat_dim, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, flat_dim)
        self.decoder = nn.Sequential(
            nn.Conv2d(c_z, 256, 1, bias=False),
            ResBlock_v3(256), ResBlock_v3(256),
            SelfAttention(256),
            nn.ConvTranspose2d(256, 256, 4, 2, 1, bias=False),
            ResBlock_v3(256, 128), ResBlock_v3(128),
            SelfAttention(128),
            nn.ConvTranspose2d(128, 128, 4, 2, 1, bias=False),
            ResBlock_v3(128, 64), ResBlock_v3(64),
            nn.ConvTranspose2d(64, 64, 4, 2, 1, bias=False),
            ResBlock_v3(64), ResBlock_v3(64),
            nn.GroupNorm(32, 64), nn.SiLU(inplace=True),
            nn.Conv2d(64, in_channels, 3, 1, 1),
            nn.Tanh(),
        )

    def encode(self, x):
        h = self.encoder(x).view(x.size(0), -1)
        mu = self.fc_mu(h)
        logvar = torch.clamp(self.fc_logvar(h), -20, 2)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)

    def decode(self, z):
        h = F.silu(self.fc_dec(z)).view(-1, self.c_z, 4, 4)
        return self.decoder(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        return self.decode(self.reparameterize(mu, logvar)), mu, logvar


class ResConvVAE_v6(nn.Module):
    """VAE v6: Gradual channel reduction with spatial mixing at 4x4.

    Resolution-adaptive: supports img_size=32 (3 downsamples) and
    img_size=128 (5 downsamples). Both reach 4x4 spatial bottleneck.
    Output in [-1, 1] via Tanh.
    """

    def __init__(self, latent_dim=64, in_channels=3, c_z=4, img_size=32):
        super().__init__()
        self.latent_dim = latent_dim
        self.c_z = c_z

        enc_layers = []
        if img_size == 128:
            enc_layers += [
                nn.Conv2d(in_channels, 32, 3, 1, 1, bias=False),
                ResBlock_v3(32), ResBlock_v3(32),
                nn.Conv2d(32, 32, 3, 2, 1, bias=False),
                ResBlock_v3(32, 64), ResBlock_v3(64),
                nn.Conv2d(64, 64, 3, 2, 1, bias=False),
            ]
            first_ch = 64
        elif img_size == 32:
            enc_layers += [nn.Conv2d(in_channels, 64, 3, 1, 1, bias=False)]
            first_ch = 64
        else:
            raise ValueError(f"img_size={img_size} not supported (use 32 or 128)")

        enc_layers += [
            ResBlock_v3(first_ch, 64), ResBlock_v3(64),
            nn.Conv2d(64, 64, 3, 2, 1, bias=False),
            ResBlock_v3(64, 128), ResBlock_v3(128),
            nn.Conv2d(128, 128, 3, 2, 1, bias=False),
            ResBlock_v3(128, 256), ResBlock_v3(256),
            SelfAttention(256),
            nn.Conv2d(256, 256, 3, 2, 1, bias=False),
            ResBlock_v3(256), ResBlock_v3(256),
            SelfAttention(256),
            ResBlock_v3(256, 64),
            nn.GroupNorm(32, 64), nn.SiLU(inplace=True),
            nn.Conv2d(64, c_z, 3, 1, 1, bias=False),
        ]
        self.encoder = nn.Sequential(*enc_layers)

        flat_dim = c_z * 4 * 4
        self.fc_mu = nn.Linear(flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(flat_dim, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, flat_dim)

        dec_layers = [
            nn.Conv2d(c_z, 64, 3, 1, 1, bias=False),
            ResBlock_v3(64, 256),
            ResBlock_v3(256), ResBlock_v3(256),
            SelfAttention(256),
            nn.ConvTranspose2d(256, 256, 4, 2, 1, bias=False),
            ResBlock_v3(256, 128), ResBlock_v3(128),
            SelfAttention(128),
            nn.ConvTranspose2d(128, 128, 4, 2, 1, bias=False),
            ResBlock_v3(128, 64), ResBlock_v3(64),
            nn.ConvTranspose2d(64, 64, 4, 2, 1, bias=False),
        ]
        if img_size == 128:
            dec_layers += [
                ResBlock_v3(64), ResBlock_v3(64),
                nn.ConvTranspose2d(64, 64, 4, 2, 1, bias=False),
                ResBlock_v3(64, 32), ResBlock_v3(32),
                nn.ConvTranspose2d(32, 32, 4, 2, 1, bias=False),
                ResBlock_v3(32), ResBlock_v3(32),
                nn.GroupNorm(32, 32), nn.SiLU(inplace=True),
                nn.Conv2d(32, in_channels, 3, 1, 1),
                nn.Tanh(),
            ]
        else:
            dec_layers += [
                ResBlock_v3(64), ResBlock_v3(64),
                nn.GroupNorm(32, 64), nn.SiLU(inplace=True),
                nn.Conv2d(64, in_channels, 3, 1, 1),
                nn.Tanh(),
            ]
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x):
        h = self.encoder(x).view(x.size(0), -1)
        mu = self.fc_mu(h)
        logvar = torch.clamp(self.fc_logvar(h), -20, 2)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)

    def decode(self, z):
        h = F.silu(self.fc_dec(z)).view(-1, self.c_z, 4, 4)
        return self.decoder(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        return self.decode(self.reparameterize(mu, logvar)), mu, logvar


class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        from torchvision.models import vgg16, VGG16_Weights
        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features
        slices = [4, 9, 16, 23]
        self.blocks = nn.ModuleList()
        prev = 0
        for s in slices:
            self.blocks.append(nn.Sequential(*list(vgg.children())[prev:s]))
            prev = s
        for p in self.parameters():
            p.requires_grad = False
        self.register_buffer("mean",
                             torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std",
                             torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x, y):
        x = (x - self.mean) / self.std
        y = (y - self.mean) / self.std
        loss = 0.0
        for block in self.blocks:
            x = block(x)
            y = block(y)
            loss += F.l1_loss(x, y)
        return loss / len(self.blocks)


def build_vae(args, latent_dim=None, in_channels=3):
    ldim = latent_dim or args.latent_dim
    arch = getattr(args, "vae_arch", "resconv")
    if arch == "v6":
        return ResConvVAE_v6(latent_dim=ldim, in_channels=in_channels,
                             c_z=args.c_z,
                             img_size=getattr(args, "img_size", 32))
    elif arch == "v5":
        return ResConvVAE_v5(latent_dim=ldim, in_channels=in_channels,
                             c_z=args.c_z)
    elif arch == "resconv":
        return ResConvVAE(latent_dim=ldim, in_channels=in_channels)
    return ConvVAE(latent_dim=ldim, in_channels=in_channels)


# ---------------------------------------------------------------------------
# 5a. Classical Velocity Field (identical to v6)
# ---------------------------------------------------------------------------
class ClassicalVelocityField(nn.Module):
    def __init__(self, latent_dim=32, hidden_dims=(256, 256, 256),
                 time_embed_dim=64):
        super().__init__()
        self.latent_dim = latent_dim
        self.time_embed_dim = time_embed_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        dims = [latent_dim + time_embed_dim] + list(hidden_dims)
        layers = []
        for i in range(len(dims) - 1):
            layers += [nn.Linear(dims[i], dims[i + 1]), nn.SiLU()]
        layers.append(nn.Linear(dims[-1], latent_dim))
        self.net = nn.Sequential(*layers)

    def _time_embedding(self, t):
        half = self.time_embed_dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device) / half)
        args = t.unsqueeze(-1) * freqs
        return torch.cat([args.cos(), args.sin()], dim=-1)

    def forward(self, z_t, t):
        t_emb = self.time_mlp(self._time_embedding(t))
        return self.net(torch.cat([z_t, t_emb], dim=-1))

    def get_eigenvalue_range(self):
        return 0.0, 0.0


# ---------------------------------------------------------------------------
# 5b. Single Quantum Circuit Module (identical to v6)
# ---------------------------------------------------------------------------
class SingleQuantumCircuit(nn.Module):
    def __init__(self, input_dim, n_qubits, encoding_type, vqc_type,
                 vqc_depth, k_local, obs_scheme, qvit_circuit="butterfly",
                 circuit_id=0):
        super().__init__()
        self.n_qubits = n_qubits
        self.encoding_type = encoding_type
        self.vqc_type = vqc_type
        self.vqc_depth = vqc_depth
        self.k_local = k_local
        self.circuit_id = circuit_id
        self.enc_per_block = _compute_enc_per_block(n_qubits, encoding_type)

        self.enc_proj = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.SiLU(),
            nn.Linear(256, self.enc_per_block),
        )

        if vqc_type == "qvit":
            n_rbs = _qvit_n_params(n_qubits, qvit_circuit)
            self.qvit_params = nn.Parameter(
                0.01 * torch.randn(vqc_depth, n_rbs, 12))
        elif vqc_type == "hardware_efficient":
            self.var_params = nn.Parameter(
                0.01 * torch.randn(vqc_depth, n_qubits))

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

        dev = qml.device("default.qubit")
        _wg = self.wire_groups
        _nq = n_qubits
        _et = encoding_type
        _vt = vqc_type
        _vd = vqc_depth
        _kl = k_local
        _no = self.n_obs
        _qc = qvit_circuit

        @qml.qnode(dev, interface="torch", diff_method="best")
        def _circuit(enc, vqc_params, H_mats):
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

            if _et == "sun":
                idx = 0
                for q in range(0, _nq - 1, 2):
                    qml.SpecialUnitary(enc[..., idx:idx + 15],
                                       wires=[q, q + 1])
                    idx += 15
                for q in range(1, _nq - 1, 2):
                    qml.SpecialUnitary(enc[..., idx:idx + 15],
                                       wires=[q, q + 1])
                    idx += 15
            else:
                for q in range(_nq):
                    qml.RY(enc[..., q], wires=q)
                for q in range(0, _nq - 1, 2):
                    qml.CNOT(wires=[q, q + 1])
                for q in range(1, _nq - 1, 2):
                    qml.CNOT(wires=[q, q + 1])

            if _vt == "qvit":
                for ly in range(_vd):
                    _qvit_layer(vqc_params[ly])
            elif _vt == "hardware_efficient":
                for ly in range(_vd):
                    _hwe_layer(vqc_params[ly])

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
        enc = self.enc_proj(x)
        H_mats = self._build_H_matrices()
        if self.vqc_type == "qvit":
            vqc_p = self.qvit_params
        elif self.vqc_type == "hardware_efficient":
            vqc_p = self.var_params
        else:
            vqc_p = torch.zeros(1)
        q_out = self._circuit(enc, vqc_p, H_mats)
        return torch.stack(q_out, dim=1).float()

    def get_eigenvalue_range(self):
        if self.k_local <= 0:
            return float("inf"), float("-inf")
        H_mats = self._build_H_matrices()
        lo, hi = float("inf"), float("-inf")
        for H in H_mats:
            eigs = torch.linalg.eigvalsh(
                H.detach().cpu().to(torch.complex128)).real
            lo = min(lo, eigs.min().item())
            hi = max(hi, eigs.max().item())
        return lo, hi


# ---------------------------------------------------------------------------
# 5c. Quantum Velocity Field (identical to v6)
# ---------------------------------------------------------------------------
class QuantumVelocityField(nn.Module):
    def __init__(self, latent_dim, n_circuits, n_qubits, encoding_type,
                 vqc_type, vqc_depth, k_local, obs_scheme,
                 qvit_circuit="butterfly", time_embed_dim=64):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_circuits = n_circuits
        self.n_qubits = n_qubits
        self.time_embed_dim = time_embed_dim

        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        input_dim = latent_dim + time_embed_dim
        self.circuits = nn.ModuleList()
        for k in range(n_circuits):
            self.circuits.append(SingleQuantumCircuit(
                input_dim=input_dim, n_qubits=n_qubits,
                encoding_type=encoding_type, vqc_type=vqc_type,
                vqc_depth=vqc_depth, k_local=k_local,
                obs_scheme=obs_scheme, qvit_circuit=qvit_circuit,
                circuit_id=k))

        self.n_obs_per_circuit = self.circuits[0].n_obs
        self.total_obs = sum(c.n_obs for c in self.circuits)

        _vh = max(256, self.total_obs)
        self.vel_head = nn.Sequential(
            nn.Linear(self.total_obs, _vh),
            nn.SiLU(),
            nn.Linear(_vh, latent_dim),
        )

    def _time_embedding(self, t):
        half = self.time_embed_dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device) / half)
        args = t.unsqueeze(-1) * freqs
        return torch.cat([args.cos(), args.sin()], dim=-1)

    def forward(self, z_t, t):
        t_emb = self.time_mlp(self._time_embedding(t))
        z_combined = torch.cat([z_t, t_emb], dim=-1)
        q_outputs = []
        for k in range(self.n_circuits):
            q_outputs.append(self.circuits[k](z_combined))
        q_all = torch.cat(q_outputs, dim=1)
        return self.vel_head(q_all)

    def get_eigenvalue_range(self):
        lo, hi = float("inf"), float("-inf")
        for circ in self.circuits:
            c_lo, c_hi = circ.get_eigenvalue_range()
            lo = min(lo, c_lo)
            hi = max(hi, c_hi)
        return lo, hi


def build_velocity_field(args, device):
    if args.velocity_field == "classical":
        hidden = [int(d) for d in args.mlp_hidden_dims.split(",")]
        vf = ClassicalVelocityField(
            latent_dim=args.latent_dim, hidden_dims=hidden,
            time_embed_dim=args.time_embed_dim).to(device)
    else:
        vf = QuantumVelocityField(
            latent_dim=args.latent_dim, n_circuits=args.n_circuits,
            n_qubits=args.n_qubits, encoding_type=args.encoding_type,
            vqc_type=args.vqc_type, vqc_depth=args.vqc_depth,
            k_local=args.k_local, obs_scheme=args.obs_scheme,
            qvit_circuit=args.qvit_circuit,
            time_embed_dim=args.time_embed_dim).to(device)
    return vf


# ---------------------------------------------------------------------------
# 6. Phase 1 -- VAE Pretraining (identical to v6)
# ---------------------------------------------------------------------------
def train_vae(args, train_loader, val_loader, device):
    vae = build_vae(args).to(device)
    optimizer = torch.optim.Adam(vae.parameters(), lr=args.lr_vae)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6)

    perc_fn = None
    if args.lambda_perc > 0:
        perc_fn = VGGPerceptualLoss().to(device)
        print(f"[Phase 1] VGG perceptual loss enabled (lambda={args.lambda_perc})")

    total_p = sum(p.numel() for p in vae.parameters() if p.requires_grad)
    print(f"[Phase 1] VAE arch={args.vae_arch}  params: {total_p:,}")

    ckpt_dir = os.path.join(args.base_path, "checkpoints")
    results_dir = os.path.join(args.base_path, "results")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f"ckpt_vae_{args.job_id}.pt")
    csv_path = os.path.join(results_dir, f"log_vae_{args.job_id}.csv")
    fields = ["epoch", "train_loss", "train_recon", "train_kl", "train_perc",
              "val_loss", "val_recon", "val_kl", "val_perc", "time_s"]

    start_epoch = 0
    best_val, best_state = float("inf"), None

    if args.resume and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, weights_only=False)
        vae.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_val = ckpt.get("best_val", float("inf"))
        print(f"  Resumed from epoch {start_epoch}")

    if start_epoch == 0:
        with open(csv_path, "w", newline="") as f:
            csv.DictWriter(f, fields).writeheader()

    warmup = max(args.beta_warmup_epochs, 1)

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        beta_eff = args.beta * min(1.0, (epoch + 1) / warmup)

        vae.train()
        tr_loss, tr_rec, tr_kl, tr_perc, tr_n = 0., 0., 0., 0., 0
        for (xb,) in tqdm(train_loader, desc=f"VAE Ep {epoch+1}/{args.epochs}",
                          leave=False):
            xb = xb.to(device)
            x_hat, mu, logvar = vae(xb)
            recon = F.mse_loss(x_hat, xb)
            kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon + beta_eff * kl
            perc_val = 0.0
            if perc_fn is not None:
                perc = perc_fn(x_hat, xb)
                loss = loss + args.lambda_perc * perc
                perc_val = perc.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            bs = xb.size(0)
            tr_loss += loss.item() * bs
            tr_rec += recon.item() * bs
            tr_kl += kl.item() * bs
            tr_perc += perc_val * bs
            tr_n += bs

        tr_loss /= tr_n; tr_rec /= tr_n; tr_kl /= tr_n; tr_perc /= tr_n

        vae.eval()
        vl_loss, vl_rec, vl_kl, vl_perc, vl_n = 0., 0., 0., 0., 0
        with torch.no_grad():
            for (xb,) in val_loader:
                xb = xb.to(device)
                x_hat, mu, logvar = vae(xb)
                recon = F.mse_loss(x_hat, xb)
                kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                loss = recon + beta_eff * kl
                perc_val = 0.0
                if perc_fn is not None:
                    perc = perc_fn(x_hat, xb)
                    loss = loss + args.lambda_perc * perc
                    perc_val = perc.item()
                bs = xb.size(0)
                vl_loss += loss.item() * bs
                vl_rec += recon.item() * bs
                vl_kl += kl.item() * bs
                vl_perc += perc_val * bs
                vl_n += bs

        vl_loss /= vl_n; vl_rec /= vl_n; vl_kl /= vl_n; vl_perc /= vl_n
        dt = time.time() - t0

        row = dict(epoch=epoch + 1, train_loss=f"{tr_loss:.6f}",
                   train_recon=f"{tr_rec:.6f}", train_kl=f"{tr_kl:.6f}",
                   train_perc=f"{tr_perc:.6f}",
                   val_loss=f"{vl_loss:.6f}", val_recon=f"{vl_rec:.6f}",
                   val_kl=f"{vl_kl:.6f}", val_perc=f"{vl_perc:.6f}",
                   time_s=f"{dt:.1f}")
        with open(csv_path, "a", newline="") as f:
            csv.DictWriter(f, fields).writerow(row)

        print(f"  Ep {epoch+1:3d} | Train {tr_loss:.4f} "
              f"(rec={tr_rec:.4f} kl={tr_kl:.4f} perc={tr_perc:.4f}) | "
              f"Val {vl_loss:.4f} | beta={beta_eff:.3f} | {dt:.1f}s")

        if vl_loss < best_val:
            best_val = vl_loss
            best_state = copy.deepcopy(vae.state_dict())

        torch.save(dict(epoch=epoch, model=vae.state_dict(),
                        optimizer=optimizer.state_dict(),
                        scheduler=scheduler.state_dict(),
                        vae_arch=args.vae_arch,
                        best_val=best_val), ckpt_path)
        scheduler.step()

    if best_state is not None:
        vae.load_state_dict(best_state)

    w_path = os.path.join(ckpt_dir, f"weights_vae_{args.job_id}.pt")
    torch.save(vae.state_dict(), w_path)
    print(f"  VAE saved to {w_path}")
    return vae


# ---------------------------------------------------------------------------
# 7. Phase 2 -- CFM Training (v9 improvements: logit-normal + VF EMA)
# ---------------------------------------------------------------------------
def train_cfm(args, vae, train_loader, val_loader, device):
    """Train the velocity field with v9 improvements."""

    # Freeze VAE
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False

    vf = build_velocity_field(args, device)

    # Dual optimizer (Chen 2025)
    H_params, circuit_params = [], []
    for name, param in vf.named_parameters():
        if any(tag in name for tag in (".A.", ".B.", ".D.")):
            H_params.append(param)
        else:
            circuit_params.append(param)

    circ_opt = torch.optim.Adam(circuit_params, lr=args.lr)
    H_opt = torch.optim.Adam(H_params, lr=args.lr_H) if H_params else None

    from torch.optim.lr_scheduler import CosineAnnealingLR
    circ_sched = CosineAnnealingLR(circ_opt, T_max=args.epochs)
    H_sched = CosineAnnealingLR(H_opt, T_max=args.epochs) if H_opt else None

    # [v9] EMA for velocity field
    vf_ema = None
    if args.vf_ema_decay > 0:
        vf_ema = EMAModel(vf, decay=args.vf_ema_decay)

    total_p = sum(p.numel() for p in vf.parameters() if p.requires_grad)
    h_p = sum(p.numel() for p in H_params)
    c_p = sum(p.numel() for p in circuit_params)
    print(f"[Phase 2] Velocity field params: total={total_p}  "
          f"circuit={c_p}  observable={h_p}")

    # Print v9 improvements
    if args.logit_normal_std > 0:
        print(f"  [v9] Logit-normal timestep sampling "
              f"(std={args.logit_normal_std})")
    else:
        print(f"  [v9] Uniform timestep sampling (logit-normal disabled)")
    print(f"  [v9] ODE solver: {args.ode_solver} ({args.ode_steps} steps)")
    if vf_ema:
        print(f"  [v9] VF EMA enabled (decay={args.vf_ema_decay})")

    if args.velocity_field == "quantum":
        enc_per = vf.circuits[0].enc_per_block
        input_dim = args.latent_dim + args.time_embed_dim
        print(f"  Architecture: K={args.n_circuits} circuit(s), "
              f"{args.n_qubits}q each")
        print(f"  Input: concat(z_t[{args.latent_dim}], "
              f"t_emb[{args.time_embed_dim}]) = {input_dim}")
        print(f"  Encoding: {args.encoding_type}, {enc_per} params/circuit")
        print(f"  VQC: {args.vqc_type} ({args.qvit_circuit}), "
              f"depth={args.vqc_depth}")
        print(f"  ANO: k_local={args.k_local}, scheme={args.obs_scheme}, "
              f"total_obs={vf.total_obs}")
        print(f"  Ratio: {vf.total_obs}/{args.latent_dim} = "
              f"{vf.total_obs / args.latent_dim:.2f}")
    else:
        print(f"  Classical MLP: hidden_dims={args.mlp_hidden_dims}")

    ckpt_dir = os.path.join(args.base_path, "checkpoints")
    results_dir = os.path.join(args.base_path, "results")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f"ckpt_cfm_{args.job_id}.pt")
    csv_path = os.path.join(results_dir, f"log_cfm_{args.job_id}.csv")
    fields = ["epoch", "train_loss", "val_loss", "val_loss_ema",
              "eig_min", "eig_max", "time_s"]

    start_epoch = 0
    best_val, best_state = float("inf"), None

    if args.resume and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, weights_only=False)
        vf.load_state_dict(ckpt["model"])
        circ_opt.load_state_dict(ckpt["circ_opt"])
        if H_opt and "H_opt" in ckpt:
            H_opt.load_state_dict(ckpt["H_opt"])
        if "circ_sched" in ckpt:
            circ_sched.load_state_dict(ckpt["circ_sched"])
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
                          desc=f"CFM Ep {epoch+1}/{args.epochs}", leave=False):
            xb = xb.to(device)

            with torch.no_grad():
                z_1, _ = vae.encode(xb)

            z_0 = torch.randn_like(z_1)

            # [v9] Logit-normal timestep sampling
            if args.logit_normal_std > 0:
                t = torch.sigmoid(
                    torch.randn(z_1.size(0), device=device)
                    * args.logit_normal_std)
            else:
                t = torch.rand(z_1.size(0), device=device)

            # OT interpolation
            t_col = t[:, None]
            z_t = (1.0 - t_col) * z_0 + t_col * z_1
            target = z_1 - z_0

            v_pred = vf(z_t, t)
            loss = F.mse_loss(v_pred, target)

            circ_opt.zero_grad()
            if H_opt:
                H_opt.zero_grad()
            loss.backward()
            circ_opt.step()
            if H_opt:
                H_opt.step()

            # [v9] Update VF EMA
            if vf_ema:
                vf_ema.update(vf)

            bs = xb.size(0)
            tr_loss += loss.item() * bs
            tr_n += bs

        tr_loss /= tr_n

        # -- Val (evaluate both raw and EMA) --
        vf.eval()
        vl_loss, vl_n = 0.0, 0
        # Use uniform t for validation (consistent across epochs)
        with torch.no_grad():
            for (xb,) in val_loader:
                xb = xb.to(device)
                z_1, _ = vae.encode(xb)
                z_0 = torch.randn_like(z_1)
                t = torch.rand(z_1.size(0), device=device)
                t_col = t[:, None]
                z_t = (1.0 - t_col) * z_0 + t_col * z_1
                target = z_1 - z_0
                v_pred = vf(z_t, t)
                loss = F.mse_loss(v_pred, target)
                vl_loss += loss.item() * xb.size(0)
                vl_n += xb.size(0)
        vl_loss /= vl_n

        # [v9] EMA validation
        vl_loss_ema = vl_loss  # fallback if no EMA
        if vf_ema:
            orig_state = copy.deepcopy(vf.state_dict())
            vf_ema.apply(vf)
            vf.eval()
            vl_ema, vl_ema_n = 0.0, 0
            with torch.no_grad():
                for (xb,) in val_loader:
                    xb = xb.to(device)
                    z_1, _ = vae.encode(xb)
                    z_0 = torch.randn_like(z_1)
                    t = torch.rand(z_1.size(0), device=device)
                    t_col = t[:, None]
                    z_t = (1.0 - t_col) * z_0 + t_col * z_1
                    target = z_1 - z_0
                    v_pred = vf(z_t, t)
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

        # Track best by EMA val loss if available, else raw
        track_val = vl_loss_ema if vf_ema else vl_loss
        if track_val < best_val:
            best_val = track_val
            if vf_ema:
                # Save EMA weights as best
                vf_ema.apply(vf)
                best_state = copy.deepcopy(vf.state_dict())
                vf.load_state_dict(orig_state)
            else:
                best_state = copy.deepcopy(vf.state_dict())

        # LR step
        circ_sched.step()
        if H_sched:
            H_sched.step()

        # Checkpoint
        ckpt_data = dict(epoch=epoch, model=vf.state_dict(),
                         circ_opt=circ_opt.state_dict(),
                         circ_sched=circ_sched.state_dict(),
                         best_val=best_val)
        if H_opt:
            ckpt_data["H_opt"] = H_opt.state_dict()
        if H_sched:
            ckpt_data["H_sched"] = H_sched.state_dict()
        if vf_ema:
            ckpt_data["vf_ema"] = vf_ema.state_dict()
        torch.save(ckpt_data, ckpt_path)

    # Load best (EMA) weights
    if best_state is not None:
        vf.load_state_dict(best_state)

    w_path = os.path.join(ckpt_dir, f"weights_cfm_{args.job_id}.pt")
    torch.save(vf.state_dict(), w_path)
    print(f"  CFM velocity field saved to {w_path}")
    return vf


# ---------------------------------------------------------------------------
# 8. Generation (v9: midpoint ODE solver)
# ---------------------------------------------------------------------------
@torch.no_grad()
def generate_samples(vae, vf, n_samples, ode_steps, latent_dim, device,
                     save_path=None, ode_solver="midpoint"):
    """Generate images via ODE integration in latent space.

    Args:
        ode_solver: "euler" or "midpoint" (2nd-order, default).
    """
    vae.eval()
    vf.eval()

    z = torch.randn(n_samples, latent_dim, device=device)
    dt = 1.0 / ode_steps

    for step in range(ode_steps):
        t_val = step * dt
        t = torch.full((n_samples,), t_val, device=device)

        if ode_solver == "midpoint":
            # Midpoint method (2nd-order): k1 at t, k2 at t + dt/2
            k1 = vf(z, t)
            z_mid = z + 0.5 * dt * k1
            t_mid = torch.full((n_samples,), t_val + 0.5 * dt, device=device)
            k2 = vf(z_mid, t_mid)
            z = z + dt * k2
        else:
            # Euler (1st-order)
            v = vf(z, t)
            z = z + dt * v

    images = vae.decode(z)
    # Handle both [0,1] (Sigmoid) and [-1,1] (Tanh) outputs
    if images.min() < -0.5:
        images = (images.clamp(-1, 1) + 1) / 2  # Tanh -> [0, 1]
    else:
        images = images.clamp(0, 1)

    if save_path:
        from torchvision.utils import save_image
        save_image(images, save_path, nrow=8, padding=2)
        print(f"  Generated {n_samples} samples -> {save_path}")

    return images


# ---------------------------------------------------------------------------
# 8b. FID & IS Evaluation (v9: midpoint ODE)
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate_fid_is(vae, vf, real_loader, n_samples, ode_steps,
                    latent_dim, device, batch_size=64,
                    ode_solver="midpoint"):
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image.inception import InceptionScore

    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    inception_score = InceptionScore(normalize=True).to(device)

    n_real = 0
    for (xb,) in real_loader:
        xb = xb.to(device)
        # Ensure [0,1] for metrics
        if xb.min() < -0.5:
            xb = (xb.clamp(-1, 1) + 1) / 2
        fid.update(xb, real=True)
        n_real += xb.size(0)
        if n_real >= n_samples:
            break

    vae.eval()
    vf.eval()
    n_gen = 0
    while n_gen < n_samples:
        bs = min(batch_size, n_samples - n_gen)
        imgs = generate_samples(vae, vf, bs, ode_steps, latent_dim, device,
                                ode_solver=ode_solver)
        fid.update(imgs, real=False)
        inception_score.update(imgs)
        n_gen += bs

    fid_val = fid.compute().item()
    is_mean, is_std = inception_score.compute()

    print(f"  FID = {fid_val:.2f}")
    print(f"  IS  = {is_mean.item():.2f} +/- {is_std.item():.2f}")
    return {"fid": fid_val, "is_mean": is_mean.item(), "is_std": is_std.item()}


# ---------------------------------------------------------------------------
# 9. Main
# ---------------------------------------------------------------------------
def main():
    args = get_args()
    set_all_seeds(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"PennyLane {qml.__version__}  |  PyTorch {torch.__version__}  |  "
          f"{device}")

    if args.velocity_field == "quantum":
        label = f"Quantum {args.n_circuits}x{args.n_qubits}q"
    else:
        label = "Classical"
    print(f"Phase: {args.phase}  |  Dataset: {args.dataset}  |  "
          f"Latent dim: {args.latent_dim}  |  VF: {label}")
    print(f"v9 improvements: logit-normal(std={args.logit_normal_std}), "
          f"ODE={args.ode_solver}({args.ode_steps}), "
          f"VF-EMA(decay={args.vf_ema_decay})")

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

    print(f"Data: train={args.n_train}  valtest={args.n_valtest}  "
          f"img_size={args.img_size}")

    # -- Phase 1: VAE --
    if args.phase in ("1", "both"):
        print("\n=== Phase 1: VAE Pretraining ===")
        vae = train_vae(args, train_loader, val_loader, device)
    else:
        vae = None

    # -- Phase 2: CFM --
    if args.phase in ("2", "both"):
        print(f"\n=== Phase 2: {label} CFM Training (v9) ===")

        if vae is None:
            vae_path = args.vae_ckpt
            if not vae_path:
                vae_path = os.path.join(args.base_path, "checkpoints",
                                        f"weights_vae_{args.job_id}.pt")
            if not os.path.exists(vae_path):
                print(f"ERROR: VAE weights not found at {vae_path}")
                print("  Run --phase=1 first or provide --vae-ckpt=<path>")
                return
            vae = build_vae(args).to(device)
            vae.load_state_dict(
                torch.load(vae_path, weights_only=True, map_location=device))
            print(f"  Loaded VAE from {vae_path}")

        vf = train_cfm(args, vae, train_loader, val_loader, device)

        # Generate sample grid
        img_path = os.path.join(args.base_path, "results",
                                f"samples_{args.job_id}.png")
        generate_samples(vae, vf, min(args.n_samples, 64), args.ode_steps,
                         args.latent_dim, device, save_path=img_path,
                         ode_solver=args.ode_solver)

        if args.compute_metrics:
            print("\n=== Computing FID & IS ===")
            metrics = evaluate_fid_is(
                vae, vf, train_loader, args.n_eval_samples,
                args.ode_steps, args.latent_dim, device,
                ode_solver=args.ode_solver)
            import json
            metrics_path = os.path.join(args.base_path, "results",
                                        f"metrics_{args.job_id}.json")
            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=2)
            print(f"  Metrics saved to {metrics_path}")

    # -- Generate only --
    if args.phase == "generate":
        print("\n=== Generation ===")
        vae_path = args.vae_ckpt or os.path.join(
            args.base_path, "checkpoints", f"weights_vae_{args.job_id}.pt")
        cfm_path = args.cfm_ckpt or os.path.join(
            args.base_path, "checkpoints", f"weights_cfm_{args.job_id}.pt")

        if not os.path.exists(vae_path) or not os.path.exists(cfm_path):
            print(f"ERROR: Need both VAE ({vae_path}) and "
                  f"CFM ({cfm_path}) weights")
            return

        vae = build_vae(args).to(device)
        vae.load_state_dict(
            torch.load(vae_path, weights_only=True, map_location=device))

        vf = build_velocity_field(args, device)
        vf.load_state_dict(
            torch.load(cfm_path, weights_only=True, map_location=device))

        img_path = os.path.join(args.base_path, "results",
                                f"generated_{args.job_id}.png")
        generate_samples(vae, vf, args.n_samples, args.ode_steps,
                         args.latent_dim, device, save_path=img_path,
                         ode_solver=args.ode_solver)

        if args.compute_metrics:
            print("\n=== Computing FID & IS ===")
            metrics = evaluate_fid_is(
                vae, vf, train_loader, args.n_eval_samples,
                args.ode_steps, args.latent_dim, device,
                ode_solver=args.ode_solver)
            import json
            metrics_path = os.path.join(args.base_path, "results",
                                        f"metrics_{args.job_id}.json")
            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=2)
            print(f"  Metrics saved to {metrics_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
