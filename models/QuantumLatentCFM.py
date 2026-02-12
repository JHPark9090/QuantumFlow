"""
Quantum Latent Conditional Flow Matching (QLCFM)
=================================================

A generative model combining:
  1. Classical VAE for encoding 32x32 images to compact latent space
  2. Quantum Velocity Field (SU(4) + QCNN + ANO) for learning latent flows
  3. Conditional Flow Matching (OT-CFM) training objective

Architecture:
  Phase 1 -- VAE pretraining:
    Image (3,32,32) -> Encoder -> (mu, logvar) -> z -> Decoder -> Image
    Loss = MSE(x, x_hat) + beta * KL(q(z|x) || N(0,I))

  Phase 2 -- Quantum CFM:
    z_0 ~ N(0,I), z_1 = Encoder(x).mu (frozen VAE)
    z_t = (1-t)*z_0 + t*z_1
    v_theta(z_t, t) = QuantumVelocityField(z_t, t)
    Loss = MSE(v_theta, z_1 - z_0)

  Generation:
    z_0 ~ N(0,I) -> Euler ODE (t: 0->1) -> z_1 -> Decoder -> Image

References:
  - Lipman et al. (2023). Flow Matching for Generative Modeling. ICLR 2023.
  - Wiersema et al. (2024). Here comes the SU(N). Quantum, 8, 1275.
  - Chen et al. (2025). Learning to Measure QNNs. ICASSP 2025.
  - Lin et al. (2025). Adaptive Non-local Observable on QNNs. IEEE QCE 2025.

Usage:
  # Phase 1: Pretrain VAE
  python QuantumLatentCFM.py --phase=1 --dataset=cifar10 --epochs=200

  # Phase 2: Train quantum CFM (loads pretrained VAE)
  python QuantumLatentCFM.py --phase=2 --dataset=cifar10 --n-qubits=8 \
      --vqc-type=qcnn --k-local=2 --epochs=300 \
      --vae-ckpt=ckpt_vae_qlcfm_001.pt

  # Generate samples
  python QuantumLatentCFM.py --phase=generate --n-samples=64 \
      --vae-ckpt=ckpt_vae_qlcfm_001.pt --cfm-ckpt=ckpt_cfm_qlcfm_001.pt
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
    p = argparse.ArgumentParser(description="Quantum Latent CFM")

    # Phase
    p.add_argument("--phase", type=str, default="1",
                   choices=["1", "2", "generate", "both"],
                   help="1=VAE pretrain, 2=CFM train, generate=sample, both=1+2")

    # VAE
    p.add_argument("--latent-dim", type=int, default=256)
    p.add_argument("--beta", type=float, default=0.5,
                   help="KL weight in VAE loss")
    p.add_argument("--beta-warmup-epochs", type=int, default=20,
                   help="Linear ramp from 0 to beta over this many epochs")
    p.add_argument("--lambda-perc", type=float, default=0.1,
                   help="VGG perceptual loss weight (0 to disable)")
    p.add_argument("--vae-arch", type=str, default="resconv",
                   choices=["resconv", "legacy"],
                   help="VAE architecture: resconv (deep residual) or legacy")

    # Quantum circuit
    p.add_argument("--n-qubits", type=int, default=8)
    p.add_argument("--n-blocks", type=int, default=2,
                   help="SU(4) encoding blocks")
    p.add_argument("--encoding-type", type=str, default="sun",
                   choices=["sun", "angle"])
    p.add_argument("--vqc-type", type=str, default="qcnn",
                   choices=["qcnn", "hardware_efficient", "qvit", "none"])
    p.add_argument("--vqc-depth", type=int, default=2)
    p.add_argument("--qvit-circuit", type=str, default="butterfly",
                   choices=["butterfly", "pyramid", "x"])
    p.add_argument("--k-local", type=int, default=2)
    p.add_argument("--obs-scheme", type=str, default="sliding",
                   choices=["sliding", "pairwise"])

    # Training
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lr-H", type=float, default=1e-1,
                   help="LR for ANO params (Chen 2025: 100x)")
    p.add_argument("--lr-vae", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--seed", type=int, default=2025)

    # Data
    p.add_argument("--dataset", type=str, default="cifar10",
                   choices=["cifar10", "coco", "mnist", "fashion"])
    p.add_argument("--n-train", type=int, default=10000)
    p.add_argument("--n-valtest", type=int, default=2000)
    p.add_argument("--img-size", type=int, default=32)

    # ODE sampling
    p.add_argument("--ode-steps", type=int, default=100)
    p.add_argument("--n-samples", type=int, default=64)

    # I/O
    p.add_argument("--job-id", type=str, default="qlcfm_001")
    p.add_argument("--base-path", type=str, default=".")
    p.add_argument("--vae-ckpt", type=str, default="")
    p.add_argument("--cfm-ckpt", type=str, default="")
    p.add_argument("--resume", action="store_true")

    # Velocity field type
    p.add_argument("--velocity-field", type=str, default="quantum",
                   choices=["quantum", "classical"],
                   help="quantum=QuantumVelocityField, classical=MLP baseline")
    p.add_argument("--mlp-hidden-dims", type=str, default="256,256,256",
                   help="Hidden layer dims for classical MLP velocity field")

    # Evaluation metrics
    p.add_argument("--compute-metrics", action="store_true",
                   help="Compute FID and IS after Phase 2")
    p.add_argument("--n-eval-samples", type=int, default=1024,
                   help="Number of samples for FID/IS computation")

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
    """Wire groups for k-local observable measurement."""
    if k_local <= 0:
        return [[q] for q in range(n_qubits)]
    if obs_scheme == "sliding":
        return [list(range(s, s + k_local))
                for s in range(n_qubits - k_local + 1)]
    elif obs_scheme == "pairwise":
        return [list(c) for c in combinations(range(n_qubits), k_local)]
    raise ValueError(f"Unknown obs_scheme: {obs_scheme}")


def _qvit_n_params(n_qubits, circuit_type):
    """Count RBS parameters per depth layer for a QViT topology.

    Simulates the gate placement loop and counts only valid qubit pairs.

    Args:
        n_qubits: Number of qubits.
        circuit_type: One of 'butterfly', 'pyramid', 'x'.
    Returns:
        int: Number of RBS gates (= trainable angles) per layer.
    """
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
    """Sinusoidal time embedding (Vaswani et al., 2017)."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=t.device) / half)
    args = t[:, None].float() * freqs[None, :]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


# ---------------------------------------------------------------------------
# 3. Data Loaders  (2D images for generative models, no labels)
# ---------------------------------------------------------------------------
def load_cifar_2d(seed, n_train, n_valtest, batch_size, img_size=32):
    """CIFAR-10 as (B, 3, 32, 32), values in [0, 1]."""
    from torchvision.datasets import CIFAR10

    torch.manual_seed(seed)
    data_train = CIFAR10(root=DATA_ROOT, train=True, download=True)
    data_test = CIFAR10(root=DATA_ROOT, train=False, download=True)

    # (N, 32, 32, 3) uint8 -> (N, 3, 32, 32) float [0,1]
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
    """MNIST resized to (B, 3, img_size, img_size)."""
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
    """Fashion-MNIST resized to (B, 3, img_size, img_size)."""
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
    """COCO as (B, 3, img_size, img_size). Falls back to synthetic."""
    import json
    torch.manual_seed(seed)
    np.random.seed(seed)

    data_dir = os.path.join(DATA_ROOT, "coco")
    ann_file = os.path.join(data_dir, "annotations/instances_train2017.json")
    img_dir = os.path.join(data_dir, "train2017")

    try:
        if not (os.path.exists(ann_file) and os.path.exists(img_dir)):
            raise FileNotFoundError("COCO files not found")

        from torchvision.datasets import CocoDetection
        from torchvision.transforms import Compose, ToTensor, Resize
        transform = Compose([Resize((img_size, img_size)), ToTensor()])
        coco = CocoDetection(img_dir, ann_file, transform=transform)

        images = []
        for idx in range(min(n_train + n_valtest, len(coco))):
            img, _ = coco[idx]
            if img.shape[0] == 3:
                images.append(img)
            if len(images) >= n_train + n_valtest:
                break
        X = torch.stack(images)
    except Exception as e:
        print(f"COCO not available ({e}), using synthetic data")
        X = torch.rand(n_train + n_valtest, 3, img_size, img_size)

    X = X[torch.randperm(len(X))]
    X_tr, X_te = X[:n_train], X[n_train:n_train + n_valtest]
    return _make_gen_loaders(X_tr, X_te, batch_size)


def _make_gen_loaders(X_train, X_valtest, batch_size):
    """Create train/val/test DataLoaders for generative (no-label) training."""
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
# 4. Convolutional VAE
# ---------------------------------------------------------------------------
class ConvVAE(nn.Module):
    """Convolutional VAE for 32x32x3 images."""

    def __init__(self, latent_dim=128, in_channels=3):
        super().__init__()
        self.latent_dim = latent_dim

        # Encoder: (3,32,32) -> (128,4,4) -> flat -> mu, logvar
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(),
        )
        self.fc_mu = nn.Linear(128 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(128 * 4 * 4, latent_dim)

        # Decoder: latent -> (128,4,4) -> (3,32,32)
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
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def decode(self, z):
        h = F.relu(self.fc_dec(z)).view(-1, 128, 4, 4)
        return self.decoder(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


class ResidualBlock(nn.Module):
    """Pre-activation residual block: BN -> ReLU -> Conv -> BN -> ReLU -> Conv + skip."""

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

    def __init__(self, latent_dim=256, in_channels=3):
        super().__init__()
        self.latent_dim = latent_dim

        # Encoder: (3,32,32) -> ... -> (256,2,2) -> flat -> mu, logvar
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, 1, 1, bias=False),
            ResidualBlock(32), ResidualBlock(32),
            nn.Conv2d(32, 64, 4, 2, 1, bias=False),          # -> (64,16,16)
            ResidualBlock(64), ResidualBlock(64),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),         # -> (128,8,8)
            ResidualBlock(128), ResidualBlock(128),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),        # -> (256,4,4)
            ResidualBlock(256), ResidualBlock(256),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 4, 2, 1, bias=False),        # -> (256,2,2)
        )
        flat_dim = 256 * 2 * 2  # 1024
        self.fc_mu = nn.Linear(flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(flat_dim, latent_dim)

        # Decoder: latent -> (256,2,2) -> mirror of encoder -> (3,32,32)
        self.fc_dec = nn.Linear(latent_dim, flat_dim)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 256, 4, 2, 1, bias=False),  # -> (256,4,4)
            ResidualBlock(256), ResidualBlock(256),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),  # -> (128,8,8)
            ResidualBlock(128), ResidualBlock(128),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),   # -> (64,16,16)
            ResidualBlock(64), ResidualBlock(64),
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),    # -> (32,32,32)
            ResidualBlock(32), ResidualBlock(32),
            nn.Conv2d(32, in_channels, 3, 1, 1), nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.encoder(x).view(x.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def decode(self, z):
        h = F.relu(self.fc_dec(z)).view(-1, 256, 2, 2)
        return self.decoder(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def build_vae(args, latent_dim=None, in_channels=3):
    """Factory: return ResConvVAE or legacy ConvVAE based on args."""
    ldim = latent_dim or args.latent_dim
    arch = getattr(args, "vae_arch", "resconv")
    if arch == "resconv":
        return ResConvVAE(latent_dim=ldim, in_channels=in_channels)
    return ConvVAE(latent_dim=ldim, in_channels=in_channels)


class VGGPerceptualLoss(nn.Module):
    """Perceptual loss using frozen VGG16 features (relu1_2, 2_2, 3_3, 4_3)."""

    def __init__(self):
        super().__init__()
        from torchvision.models import vgg16, VGG16_Weights
        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features
        slices = [4, 9, 16, 23]  # relu1_2, relu2_2, relu3_3, relu4_3
        self.blocks = nn.ModuleList()
        prev = 0
        for s in slices:
            self.blocks.append(nn.Sequential(*list(vgg.children())[prev:s]))
            prev = s
        for p in self.parameters():
            p.requires_grad = False
        # ImageNet normalization
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


# ---------------------------------------------------------------------------
# 5a. Classical Velocity Field (MLP Baseline)
# ---------------------------------------------------------------------------
class ClassicalVelocityField(nn.Module):
    """Classical MLP velocity field baseline (same interface as quantum)."""

    def __init__(self, latent_dim=32, hidden_dims=(256, 256, 256),
                 time_embed_dim=64):
        super().__init__()
        self.latent_dim = latent_dim
        self.time_embed_dim = time_embed_dim

        # Sinusoidal time embedding (same as diffusion models)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        # MLP: (latent_dim + time_embed_dim) -> hidden -> latent_dim
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
        return 0.0, 0.0  # no ANO observable -- keep CSV compat


# ---------------------------------------------------------------------------
# 5b. Quantum Velocity Field
# ---------------------------------------------------------------------------
class QuantumVelocityField(nn.Module):
    """Quantum velocity field for latent CFM.

    SU(4) encoding + QCNN/QViT/HWE + ANO measurement, with classical
    pre/post heads.

    Forward (no skip — all info flows through quantum circuit):
      (z_t, t) -> [time_embed + concat] -> [FC -> enc_params]
                -> [SU(4) encoding -> QCNN/QViT/HWE -> ANO measurement]
                -> [FC(q_out) -> velocity]
    """

    def __init__(self, latent_dim, n_qubits, n_blocks, encoding_type,
                 vqc_type, vqc_depth, k_local, obs_scheme,
                 qvit_circuit="butterfly"):
        super().__init__()

        self.latent_dim = latent_dim
        self.n_qubits = n_qubits
        self.encoding_type = encoding_type
        self.vqc_type = vqc_type
        self.vqc_depth = vqc_depth
        self.k_local = k_local
        self.n_blocks = n_blocks
        self.qvit_circuit = qvit_circuit

        time_dim = latent_dim
        input_dim = latent_dim + time_dim  # z_t + time_emb

        # -- Encoding parameter count --
        if encoding_type == "sun":
            n_even = n_qubits // 2
            n_odd = (n_qubits - 1) // 2
            gates_per_block = n_even + n_odd
            self.total_enc = n_blocks * gates_per_block * 15
        else:
            self.total_enc = n_blocks * n_qubits

        # Classical pre-processing: z_combined -> encoding params
        self.enc_proj = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.SiLU(),
            nn.Linear(256, self.total_enc),
        )

        # -- VQC parameters --
        if vqc_type == "qcnn":
            self.conv_params = nn.Parameter(
                0.01 * torch.randn(vqc_depth, n_qubits, 15))
            self.pool_params = nn.Parameter(
                0.01 * torch.randn(vqc_depth, n_qubits // 2, 3))
        elif vqc_type == "hardware_efficient":
            self.var_params = nn.Parameter(
                0.01 * torch.randn(vqc_depth, n_qubits))
        elif vqc_type == "qvit":
            n_rbs = _qvit_n_params(n_qubits, qvit_circuit)
            # 12 params per gate: 2×U3(3) + 2×(IsingXX+IsingYY+IsingZZ)(3)
            self.qvit_params = nn.Parameter(
                0.01 * torch.randn(vqc_depth, n_rbs, 12))

        # -- ANO parameters --
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

        # Classical post-processing: q_out -> velocity (no skip)
        # All information must flow through the quantum circuit.
        self.vel_head = nn.Sequential(
            nn.Linear(self.n_obs, 256),
            nn.SiLU(),
            nn.Linear(256, latent_dim),
        )

        # -- Build QNode --
        dev = qml.device("default.qubit")

        _wg = self.wire_groups
        _nq = n_qubits
        _nb = n_blocks
        _et = encoding_type
        _vt = vqc_type
        _vd = vqc_depth
        _kl = k_local
        _no = self.n_obs
        _qc = qvit_circuit

        @qml.qnode(dev, interface="torch", diff_method="best")
        def _circuit(enc, vqc_p1, vqc_p2, H_mats):
            # Stage 1: Encoding
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
            else:  # angle
                for layer in range(_nb):
                    for q in range(_nq):
                        qml.RY(enc[..., layer * _nq + q], wires=q)
                    if layer < _nb - 1:
                        for q in range(0, _nq - 1, 2):
                            qml.CNOT(wires=[q, q + 1])
                        for q in range(1, _nq - 1, 2):
                            qml.CNOT(wires=[q, q + 1])

            # Stage 2: VQC
            if _vt == "qcnn":
                wires = list(range(_nq))
                for ly in range(_vd):
                    nw = len(wires)
                    if nw < 2:
                        break
                    # Convolution
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
                    # Pooling
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

            elif _vt == "qvit":
                # Enhanced QViT layers: U3 + IsingXX/YY/ZZ
                # Per gate pair: U3(w1) IsingXX/YY/ZZ(w1,w2) U3(w1) IsingXX/YY/ZZ(w1,w2)
                # 12 params per gate: [U3(3), Ising(3), U3(3), Ising(3)]
                def _qvit_gate(p, w1, w2):
                    qml.U3(p[0], p[1], p[2], wires=w1)
                    qml.IsingXX(p[3], wires=[w1, w2])
                    qml.IsingYY(p[4], wires=[w1, w2])
                    qml.IsingZZ(p[5], wires=[w1, w2])
                    qml.U3(p[6], p[7], p[8], wires=w1)
                    qml.IsingXX(p[9], wires=[w1, w2])
                    qml.IsingYY(p[10], wires=[w1, w2])
                    qml.IsingZZ(p[11], wires=[w1, w2])

                for ly in range(_vd):
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
                                        _qvit_gate(vqc_p1[ly, pidx], w1, w2)
                                        pidx += 1
                    elif _qc == "pyramid":
                        for layer in range(_nq - 1):
                            for i in range(_nq - layer - 1):
                                _qvit_gate(vqc_p1[ly, pidx], i, i + 1)
                                pidx += 1
                    elif _qc == "x":
                        for i in range(_nq // 2):
                            w1, w2 = i, _nq - 1 - i
                            _qvit_gate(vqc_p1[ly, pidx], w1, w2)
                            pidx += 1
                        for i in range(_nq // 2 - 1):
                            _qvit_gate(vqc_p1[ly, pidx], i, i + 1)
                            pidx += 1

            # Stage 3: Measurement
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

    def forward(self, z_t, t):
        """
        Args:
            z_t: (batch, latent_dim) -- noisy latent at time t
            t:   (batch,) -- time in [0, 1]
        Returns:
            v:   (batch, latent_dim) -- predicted velocity
        """
        t_emb = sinusoidal_embedding(t, self.latent_dim)
        z_combined = torch.cat([z_t, t_emb], dim=-1)

        enc = self.enc_proj(z_combined)
        H_mats = self._build_H_matrices()

        if self.vqc_type == "qcnn":
            p1, p2 = self.conv_params, self.pool_params
        elif self.vqc_type == "hardware_efficient":
            p1, p2 = self.var_params, torch.zeros(1)
        elif self.vqc_type == "qvit":
            p1, p2 = self.qvit_params, torch.zeros(1)
        else:
            p1, p2 = torch.zeros(1), torch.zeros(1)

        q_out = self._circuit(enc, p1, p2, H_mats)
        q_out = torch.stack(q_out, dim=1).float()

        return self.vel_head(q_out)

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
# 5c. Velocity Field Factory
# ---------------------------------------------------------------------------
def build_velocity_field(args, device):
    """Factory: build quantum or classical velocity field."""
    if args.velocity_field == "classical":
        hidden = [int(d) for d in args.mlp_hidden_dims.split(",")]
        vf = ClassicalVelocityField(
            latent_dim=args.latent_dim, hidden_dims=hidden).to(device)
    else:
        vf = QuantumVelocityField(
            latent_dim=args.latent_dim, n_qubits=args.n_qubits,
            n_blocks=args.n_blocks, encoding_type=args.encoding_type,
            vqc_type=args.vqc_type, vqc_depth=args.vqc_depth,
            k_local=args.k_local, obs_scheme=args.obs_scheme,
            qvit_circuit=args.qvit_circuit,
        ).to(device)
    return vf


# ---------------------------------------------------------------------------
# 6. Phase 1 -- VAE Pretraining
# ---------------------------------------------------------------------------
def train_vae(args, train_loader, val_loader, device):
    """Train the convolutional VAE."""

    vae = build_vae(args).to(device)
    optimizer = torch.optim.Adam(vae.parameters(), lr=args.lr_vae)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6)

    # Optional perceptual loss
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
              "val_loss", "val_recon", "val_kl", "val_perc",
              "beta_eff", "lr", "time_s"]

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
        cur_lr = optimizer.param_groups[0]["lr"]

        # -- Train --
        vae.train()
        tr_loss, tr_rec, tr_kl, tr_perc, tr_n = 0.0, 0.0, 0.0, 0.0, 0
        for (xb,) in tqdm(train_loader, desc=f"VAE Ep {epoch+1}/{args.epochs}",
                          leave=False):
            xb = xb.to(device)
            x_hat, mu, logvar = vae(xb)

            recon = F.mse_loss(x_hat, xb, reduction="mean")
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

        tr_loss /= tr_n
        tr_rec /= tr_n
        tr_kl /= tr_n
        tr_perc /= tr_n

        # -- Val --
        vae.eval()
        vl_loss, vl_rec, vl_kl, vl_perc, vl_n = 0.0, 0.0, 0.0, 0.0, 0
        with torch.no_grad():
            for (xb,) in val_loader:
                xb = xb.to(device)
                x_hat, mu, logvar = vae(xb)
                recon = F.mse_loss(x_hat, xb, reduction="mean")
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

        vl_loss /= vl_n
        vl_rec /= vl_n
        vl_kl /= vl_n
        vl_perc /= vl_n
        dt = time.time() - t0

        row = dict(epoch=epoch + 1, train_loss=f"{tr_loss:.6f}",
                   train_recon=f"{tr_rec:.6f}", train_kl=f"{tr_kl:.6f}",
                   train_perc=f"{tr_perc:.6f}",
                   val_loss=f"{vl_loss:.6f}", val_recon=f"{vl_rec:.6f}",
                   val_kl=f"{vl_kl:.6f}", val_perc=f"{vl_perc:.6f}",
                   beta_eff=f"{beta_eff:.4f}", lr=f"{cur_lr:.2e}",
                   time_s=f"{dt:.1f}")
        with open(csv_path, "a", newline="") as f:
            csv.DictWriter(f, fields).writerow(row)

        print(f"  Ep {epoch+1:3d} | Train {tr_loss:.4f} "
              f"(rec={tr_rec:.4f} kl={tr_kl:.4f} perc={tr_perc:.4f}) | "
              f"Val {vl_loss:.4f} | beta={beta_eff:.3f} lr={cur_lr:.2e} | "
              f"{dt:.1f}s")

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
# 7. Phase 2 -- Quantum CFM Training
# ---------------------------------------------------------------------------
def train_cfm(args, vae, train_loader, val_loader, device):
    """Train the velocity field (quantum or classical) with CFM loss."""

    # Freeze VAE
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False

    vf = build_velocity_field(args, device)

    # Dual optimizer (Chen 2025)
    H_params, circuit_params = [], []
    for name, param in vf.named_parameters():
        if name.startswith(("A.", "B.", "D.")):
            H_params.append(param)
        else:
            circuit_params.append(param)

    circ_opt = torch.optim.Adam(circuit_params, lr=args.lr)
    H_opt = torch.optim.Adam(H_params, lr=args.lr_H) if H_params else None

    # Cosine annealing LR schedule
    from torch.optim.lr_scheduler import CosineAnnealingLR
    circ_sched = CosineAnnealingLR(circ_opt, T_max=args.epochs)
    H_sched = CosineAnnealingLR(H_opt, T_max=args.epochs) if H_opt else None

    total_p = sum(p.numel() for p in vf.parameters() if p.requires_grad)
    h_p = sum(p.numel() for p in H_params)
    c_p = sum(p.numel() for p in circuit_params)
    print(f"[Phase 2] Velocity field params: total={total_p}  "
          f"circuit={c_p}  observable={h_p}")
    if args.velocity_field == "quantum":
        print(f"  Encoding: {args.encoding_type}, {args.n_blocks} blocks, "
              f"{vf.total_enc} params")
        print(f"  VQC: {args.vqc_type}, depth={args.vqc_depth}")
        print(f"  ANO: k_local={args.k_local}, scheme={args.obs_scheme}, "
              f"n_obs={vf.n_obs}")
    else:
        print(f"  Classical MLP: hidden_dims={args.mlp_hidden_dims}")

    ckpt_dir = os.path.join(args.base_path, "checkpoints")
    results_dir = os.path.join(args.base_path, "results")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f"ckpt_cfm_{args.job_id}.pt")
    csv_path = os.path.join(results_dir, f"log_cfm_{args.job_id}.csv")
    fields = ["epoch", "train_loss", "val_loss", "eig_min", "eig_max", "time_s"]

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
                z_1, _ = vae.encode(xb)  # mu only (deterministic)

            z_0 = torch.randn_like(z_1)
            t = torch.rand(z_1.size(0), device=device)

            # OT interpolation: z_t = (1-t)*z_0 + t*z_1
            t_col = t[:, None]
            z_t = (1.0 - t_col) * z_0 + t_col * z_1
            target = z_1 - z_0  # constant velocity field

            v_pred = vf(z_t, t)
            loss = F.mse_loss(v_pred, target)

            circ_opt.zero_grad()
            if H_opt:
                H_opt.zero_grad()
            loss.backward()
            circ_opt.step()
            if H_opt:
                H_opt.step()

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
        eig_lo, eig_hi = vf.get_eigenvalue_range()
        dt = time.time() - t0

        row = dict(epoch=epoch + 1, train_loss=f"{tr_loss:.6f}",
                   val_loss=f"{vl_loss:.6f}", eig_min=f"{eig_lo:.4f}",
                   eig_max=f"{eig_hi:.4f}", time_s=f"{dt:.1f}")
        with open(csv_path, "a", newline="") as f:
            csv.DictWriter(f, fields).writerow(row)

        print(f"  Ep {epoch+1:3d} | Train {tr_loss:.6f} | Val {vl_loss:.6f} | "
              f"Eig [{eig_lo:.2f}, {eig_hi:.2f}] | {dt:.1f}s")

        if vl_loss < best_val:
            best_val = vl_loss
            best_state = copy.deepcopy(vf.state_dict())

        # Step LR schedulers
        circ_sched.step()
        if H_sched:
            H_sched.step()

        ckpt_data = dict(epoch=epoch, model=vf.state_dict(),
                         circ_opt=circ_opt.state_dict(),
                         circ_sched=circ_sched.state_dict(),
                         best_val=best_val)
        if H_opt:
            ckpt_data["H_opt"] = H_opt.state_dict()
        if H_sched:
            ckpt_data["H_sched"] = H_sched.state_dict()
        torch.save(ckpt_data, ckpt_path)

    if best_state is not None:
        vf.load_state_dict(best_state)

    w_path = os.path.join(ckpt_dir, f"weights_cfm_{args.job_id}.pt")
    torch.save(vf.state_dict(), w_path)
    print(f"  CFM velocity field saved to {w_path}")
    return vf


# ---------------------------------------------------------------------------
# 8. Generation via Euler ODE
# ---------------------------------------------------------------------------
@torch.no_grad()
def generate_samples(vae, vf, n_samples, ode_steps, latent_dim, device,
                     save_path=None):
    """Generate images by Euler ODE integration in latent space."""
    vae.eval()
    vf.eval()

    # Start from noise
    z = torch.randn(n_samples, latent_dim, device=device)
    dt = 1.0 / ode_steps

    for step in range(ode_steps):
        t_val = step * dt
        t = torch.full((n_samples,), t_val, device=device)
        v = vf(z, t)
        z = z + dt * v

    # Decode
    images = vae.decode(z)
    images = images.clamp(0, 1)

    if save_path:
        from torchvision.utils import save_image
        save_image(images, save_path, nrow=8, padding=2)
        print(f"  Generated {n_samples} samples -> {save_path}")

    return images


# ---------------------------------------------------------------------------
# 8b. FID & IS Evaluation
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate_fid_is(vae, vf, real_loader, n_samples, ode_steps,
                    latent_dim, device, batch_size=64):
    """Compute FID and IS for generated vs real images."""
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image.inception import InceptionScore

    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    inception_score = InceptionScore(normalize=True).to(device)

    # Add real images
    n_real = 0
    for (xb,) in real_loader:
        xb = xb.to(device)
        fid.update(xb, real=True)
        n_real += xb.size(0)
        if n_real >= n_samples:
            break

    # Generate and add fake images
    vae.eval()
    vf.eval()
    n_gen = 0
    while n_gen < n_samples:
        bs = min(batch_size, n_samples - n_gen)
        z = torch.randn(bs, latent_dim, device=device)
        dt = 1.0 / ode_steps
        for step in range(ode_steps):
            t_val = step * dt
            t = torch.full((bs,), t_val, device=device)
            v = vf(z, t)
            z = z + dt * v
        imgs = vae.decode(z).clamp(0, 1)
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
    print(f"Phase: {args.phase}  |  Dataset: {args.dataset}  |  "
          f"Latent dim: {args.latent_dim}")

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
        vf_label = args.velocity_field.capitalize()
        print(f"\n=== Phase 2: {vf_label} CFM Training ===")

        # Load VAE if not already trained in this run
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
            vae.load_state_dict(torch.load(vae_path, weights_only=True))
            print(f"  Loaded VAE from {vae_path}")

        vf = train_cfm(args, vae, train_loader, val_loader, device)

        # Generate a sample grid
        img_path = os.path.join(args.base_path, "results",
                                f"samples_{args.job_id}.png")
        generate_samples(vae, vf, min(args.n_samples, 64), args.ode_steps,
                         args.latent_dim, device, save_path=img_path)

        if args.compute_metrics:
            print("\n=== Computing FID & IS ===")
            metrics = evaluate_fid_is(
                vae, vf, train_loader, args.n_eval_samples,
                args.ode_steps, args.latent_dim, device)
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
        vae.load_state_dict(torch.load(vae_path, weights_only=True))

        vf = build_velocity_field(args, device)
        vf.load_state_dict(torch.load(cfm_path, weights_only=True))

        img_path = os.path.join(args.base_path, "results",
                                f"generated_{args.job_id}.png")
        generate_samples(vae, vf, args.n_samples, args.ode_steps,
                         args.latent_dim, device, save_path=img_path)

        if args.compute_metrics:
            print("\n=== Computing FID & IS ===")
            metrics = evaluate_fid_is(
                vae, vf, train_loader, args.n_eval_samples,
                args.ode_steps, args.latent_dim, device)
            import json
            metrics_path = os.path.join(args.base_path, "results",
                                        f"metrics_{args.job_id}.json")
            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=2)
            print(f"  Metrics saved to {metrics_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
