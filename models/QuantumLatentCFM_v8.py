"""
Quantum Latent Conditional Flow Matching v8 — SU(16) Encoding Experiment
=========================================================================

Extends v6 with:
  - Generalized SU(N) encoding: --sun-group-size controls qubit grouping
    - SU(4):  group_size=2 (15 params/pair,  7 groups = 105 total on 8q)
    - SU(16): group_size=4 (255 params/group, 3 groups = 765 total on 8q)
  - Configurable enc_proj hidden dim: --enc-proj-hidden (256 for v6, 1024 for v8)
  - Classical sandwich velocity field: --velocity-field=classical_sandwich
    Shares enc_proj + vel_head with quantum, replaces circuit with classical MLP

Velocity field variants:
  1. Classical MLP (--velocity-field=classical):
     time_mlp(sin_embed(t,32)) -> concat(z_t[32], t_emb[32]) = 64
     -> Linear(64,256) -> SiLU -> Linear(256,256) -> SiLU
     -> Linear(256,256) -> SiLU -> Linear(256,32)

  2. v8-Quantum (--velocity-field=quantum --sun-group-size=4 --enc-proj-hidden=1024):
     Same time embedding -> concat = 64
     -> enc_proj: Linear(64,1024) -> SiLU -> Linear(1024,765)
     -> SU(16) encoding (8q, 3 groups of 4) -> QViT butterfly (depth=2)
     -> ANO pairwise k=2 -> 28 obs
     -> vel_head: Linear(28,256) -> SiLU -> Linear(256,32)

  3. Classical-D (--velocity-field=classical_sandwich --sun-group-size=4
                  --enc-proj-hidden=1024):
     Same time embedding -> concat = 64
     -> enc_proj: Linear(64,1024) -> SiLU -> Linear(1024,765)   [SAME as quantum]
     -> core: Linear(765,64) -> SiLU -> Linear(64,28)            [REPLACES quantum]
     -> vel_head: Linear(28,256) -> SiLU -> Linear(256,32)       [SAME as quantum]

References:
  - Lipman et al. (2023). Flow Matching for Generative Modeling. ICLR 2023.
  - Wiersema et al. (2024). Here comes the SU(N). Quantum, 8, 1275.
  - Chen et al. (2025). Learning to Measure QNNs. ICASSP 2025 Workshop.
  - Lin et al. (2025). Adaptive Non-local Observable on QNNs. IEEE QCE 2025.
  - Cherrat et al. (2024). Quantum Vision Transformers. Quantum, 8, 1265.

Usage:
  # v8-Quantum: SU(16) encoding, single circuit, 8 qubits
  python QuantumLatentCFM_v8.py --phase=2 --dataset=cifar10 \\
      --velocity-field=quantum --n-circuits=1 --n-qubits=8 \\
      --sun-group-size=4 --enc-proj-hidden=1024 \\
      --vae-ckpt=<path_to_vae.pt> --epochs=200

  # Classical-D: same enc_proj + vel_head, classical core
  python QuantumLatentCFM_v8.py --phase=2 --dataset=cifar10 \\
      --velocity-field=classical_sandwich --n-qubits=8 \\
      --sun-group-size=4 --enc-proj-hidden=1024 \\
      --vae-ckpt=<path_to_vae.pt> --epochs=200

  # Backward compatible: v6 config (SU(4), enc_proj_hidden=256)
  python QuantumLatentCFM_v8.py --phase=both --dataset=cifar10 \\
      --velocity-field=quantum --n-circuits=1 --n-qubits=8 \\
      --sun-group-size=2 --enc-proj-hidden=256 --epochs=200
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
        description="Quantum Latent CFM v8 — SU(16) Encoding Experiment")

    # Phase
    p.add_argument("--phase", type=str, default="1",
                   choices=["1", "2", "generate", "both"],
                   help="1=VAE pretrain, 2=CFM train, generate=sample, both=1+2")

    # VAE (matched to Classical-A defaults)
    p.add_argument("--latent-dim", type=int, default=32,
                   help="VAE latent dimension (default 32)")
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
    p.add_argument("--n-circuits", type=int, default=1,
                   help="Number of parallel quantum circuits (1=single, K=multi)")
    p.add_argument("--n-qubits", type=int, default=8,
                   help="Qubits per circuit")
    p.add_argument("--encoding-type", type=str, default="sun",
                   choices=["sun", "angle"])
    p.add_argument("--sun-group-size", type=int, default=2,
                   help="Qubits per SU(N) group: 2=SU(4), 4=SU(16) (default 2)")
    p.add_argument("--vqc-type", type=str, default="qvit",
                   choices=["qvit", "hardware_efficient", "none"])
    p.add_argument("--vqc-depth", type=int, default=2)
    p.add_argument("--qvit-circuit", type=str, default="butterfly",
                   choices=["butterfly", "pyramid", "x"])
    p.add_argument("--k-local", type=int, default=2)
    p.add_argument("--obs-scheme", type=str, default="pairwise",
                   choices=["sliding", "pairwise"])

    # Projection layers
    p.add_argument("--enc-proj-hidden", type=int, default=256,
                   help="Hidden dim in enc_proj: Linear(input,H)->SiLU->Linear(H,enc)")
    p.add_argument("--core-hidden", type=int, default=64,
                   help="Hidden dim in classical sandwich core (classical_sandwich only)")

    # Time embedding
    p.add_argument("--time-embed-dim", type=int, default=32,
                   help="Time embedding dimension (default 32 = latent_dim)")

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
    p.add_argument("--job-id", type=str, default="qlcfm_v8_001")
    p.add_argument("--base-path", type=str, default=".")
    p.add_argument("--vae-ckpt", type=str, default="")
    p.add_argument("--cfm-ckpt", type=str, default="")
    p.add_argument("--resume", action="store_true")

    # Velocity field type
    p.add_argument("--velocity-field", type=str, default="quantum",
                   choices=["quantum", "classical", "classical_sandwich"],
                   help="quantum | classical (MLP) | classical_sandwich (same "
                        "enc_proj+vel_head as quantum, classical core)")
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
    """Count RBS parameters per depth layer for a QViT topology."""
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


def _compute_enc_per_block(n_qubits, encoding_type, sun_group_size=2):
    """Compute encoding parameters needed per block for given config.

    For SU(N) encoding with brick-layer pattern:
      - Even layer: groups at [0, gs, 2*gs, ...]
      - Odd layer:  groups at [gs//2, gs//2 + gs, ...] (shifted)
      - Each group of `sun_group_size` qubits uses SU(2^sun_group_size)
        with N^2 - 1 real parameters.

    Examples:
      SU(4) on 8q:  group_size=2 -> 7 pairs  * 15  = 105
      SU(16) on 8q: group_size=4 -> 3 groups * 255 = 765
    """
    if encoding_type == "sun":
        gs = sun_group_size
        N = 2 ** gs
        params_per_group = N * N - 1
        n_even = n_qubits // gs
        shift = gs // 2
        n_odd = (n_qubits - shift) // gs
        return (n_even + n_odd) * params_per_group
    else:
        return n_qubits


# ---------------------------------------------------------------------------
# 3. Data Loaders (2D images for generative models, no labels)
# ---------------------------------------------------------------------------
def load_cifar_2d(seed, n_train, n_valtest, batch_size, img_size=32):
    """CIFAR-10 as (B, 3, 32, 32), values in [0, 1]."""
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
    """Pre-activation residual block."""

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
    """Perceptual loss using frozen VGG16 features."""

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


# ---------------------------------------------------------------------------
# 5a. Classical Velocity Field (simple MLP, same as v6)
# ---------------------------------------------------------------------------
class ClassicalVelocityField(nn.Module):
    """Classical MLP velocity field.

    Architecture:
      sin_embed(t, D) -> time_mlp(D->D->D) -> concat(z_t, t_emb)
      -> Linear -> SiLU -> ... -> Linear(out)
    """

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
# 5b. Single Quantum Circuit Module (generalized SU(N) encoding)
# ---------------------------------------------------------------------------
class SingleQuantumCircuit(nn.Module):
    """One quantum circuit with nonlinear enc_proj and generalized SU(N).

    Architecture per circuit:
      input -> Linear(input_dim, enc_proj_hidden) -> SiLU
      -> Linear(enc_proj_hidden, enc_per_block)
      -> SU(N) encoding (brick-layer) -> QViT/HWE VQC -> ANO measurement
      -> n_obs observables
    """

    def __init__(self, input_dim, n_qubits, encoding_type, sun_group_size,
                 enc_proj_hidden, vqc_type, vqc_depth, k_local, obs_scheme,
                 qvit_circuit="butterfly", circuit_id=0):
        super().__init__()

        self.n_qubits = n_qubits
        self.encoding_type = encoding_type
        self.sun_group_size = sun_group_size
        self.vqc_type = vqc_type
        self.vqc_depth = vqc_depth
        self.k_local = k_local
        self.circuit_id = circuit_id

        # Encoding parameter count (1 block per circuit)
        self.enc_per_block = _compute_enc_per_block(
            n_qubits, encoding_type, sun_group_size)

        # Encoding projection WITH nonlinearity
        self.enc_proj = nn.Sequential(
            nn.Linear(input_dim, enc_proj_hidden),
            nn.SiLU(),
            nn.Linear(enc_proj_hidden, self.enc_per_block),
        )

        # VQC parameters
        if vqc_type == "qvit":
            n_rbs = _qvit_n_params(n_qubits, qvit_circuit)
            self.qvit_params = nn.Parameter(
                0.01 * torch.randn(vqc_depth, n_rbs, 12))
        elif vqc_type == "hardware_efficient":
            self.var_params = nn.Parameter(
                0.01 * torch.randn(vqc_depth, n_qubits))

        # ANO parameters
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
                nn.init.normal_(self.A[w], 0, 0.01)
                nn.init.normal_(self.B[w], 0, 0.01)
                nn.init.uniform_(self.D[w], -1, 1)

        # Build PennyLane circuit (closures capture local variables)
        # Use default.qubit + backprop: supports SpecialUnitary natively
        # (lightning.qubit requires decomposition which fails for batched SU(N))
        dev = qml.device("default.qubit", wires=n_qubits)

        _nq = n_qubits
        _et = encoding_type
        _gs = sun_group_size
        _wg = self.wire_groups
        _vt = vqc_type
        _vd = vqc_depth
        _kl = k_local
        _no = self.n_obs
        _qc = qvit_circuit

        @qml.qnode(dev, interface="torch", diff_method="backprop")
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

            # Stage 1: Generalized SU(N) / angle encoding
            if _et == "sun":
                _N = 2 ** _gs
                _ppg = _N * _N - 1
                idx = 0
                # Even layer: groups at [0, gs, 2*gs, ...]
                for s in range(0, _nq, _gs):
                    if s + _gs <= _nq:
                        wires = list(range(s, s + _gs))
                        qml.SpecialUnitary(enc[..., idx:idx + _ppg],
                                           wires=wires)
                        idx += _ppg
                # Odd layer: groups shifted by gs//2
                _shift = _gs // 2
                for s in range(_shift, _nq, _gs):
                    if s + _gs <= _nq:
                        wires = list(range(s, s + _gs))
                        qml.SpecialUnitary(enc[..., idx:idx + _ppg],
                                           wires=wires)
                        idx += _ppg
            else:  # angle
                for q in range(_nq):
                    qml.RY(enc[..., q], wires=q)
                for q in range(0, _nq - 1, 2):
                    qml.CNOT(wires=[q, q + 1])
                for q in range(1, _nq - 1, 2):
                    qml.CNOT(wires=[q, q + 1])

            # Stage 2: VQC
            if _vt == "qvit":
                for ly in range(_vd):
                    _qvit_layer(vqc_params[ly])
            elif _vt == "hardware_efficient":
                for ly in range(_vd):
                    _hwe_layer(vqc_params[ly])

            # Stage 3: ANO measurement
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
# 5c. Quantum Velocity Field (unified single/multi-circuit)
# ---------------------------------------------------------------------------
class QuantumVelocityField(nn.Module):
    """Quantum velocity field with generalized SU(N) encoding.

    Architecture:
      sin_embed(t, D) -> time_mlp(D->D->D) -> concat(z_t, t_emb)
      -> [shared to all K circuits]
      -> Per circuit: enc_proj -> SU(N) encoding -> QViT -> ANO -> n_obs
      -> concat K*n_obs -> vel_head -> latent_dim
    """

    def __init__(self, latent_dim, n_circuits, n_qubits, encoding_type,
                 sun_group_size, enc_proj_hidden, vqc_type, vqc_depth,
                 k_local, obs_scheme, qvit_circuit="butterfly",
                 time_embed_dim=64):
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
                input_dim=input_dim,
                n_qubits=n_qubits,
                encoding_type=encoding_type,
                sun_group_size=sun_group_size,
                enc_proj_hidden=enc_proj_hidden,
                vqc_type=vqc_type,
                vqc_depth=vqc_depth,
                k_local=k_local,
                obs_scheme=obs_scheme,
                qvit_circuit=qvit_circuit,
                circuit_id=k,
            ))

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
            q_out_k = self.circuits[k](z_combined)
            q_outputs.append(q_out_k)

        q_all = torch.cat(q_outputs, dim=1)
        return self.vel_head(q_all)

    def get_eigenvalue_range(self):
        lo, hi = float("inf"), float("-inf")
        for circ in self.circuits:
            c_lo, c_hi = circ.get_eigenvalue_range()
            lo = min(lo, c_lo)
            hi = max(hi, c_hi)
        return lo, hi


# ---------------------------------------------------------------------------
# 5d. Classical Sandwich Velocity Field (Classical-D control)
# ---------------------------------------------------------------------------
class ClassicalSandwichVelocityField(nn.Module):
    """Classical control that mirrors quantum VF structure exactly.

    Shares enc_proj and vel_head with quantum, replaces the quantum
    circuit core with a classical MLP.

    Architecture:
      sin_embed(t, D) -> time_mlp(D->D->D) -> concat(z_t, t_emb) = input_dim
      -> enc_proj: Linear(input_dim, enc_proj_hidden) -> SiLU
                   -> Linear(enc_proj_hidden, enc_per_block)
      -> core: Linear(enc_per_block, core_hidden) -> SiLU
               -> Linear(core_hidden, n_obs)
      -> vel_head: Linear(n_obs, max(256,n_obs)) -> SiLU
                   -> Linear(max(256,n_obs), latent_dim)
    """

    def __init__(self, latent_dim, n_qubits, encoding_type, sun_group_size,
                 enc_proj_hidden, k_local, obs_scheme, core_hidden=64,
                 time_embed_dim=64):
        super().__init__()

        self.latent_dim = latent_dim
        self.time_embed_dim = time_embed_dim

        # Time embedding (identical to quantum VF)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        input_dim = latent_dim + time_embed_dim

        # Compute enc_per_block (same formula as quantum)
        self.enc_per_block = _compute_enc_per_block(
            n_qubits, encoding_type, sun_group_size)

        # Compute n_obs (same as quantum)
        self.wire_groups = get_wire_groups(n_qubits, k_local, obs_scheme)
        self.n_obs = len(self.wire_groups)
        self.total_obs = self.n_obs  # single path

        # enc_proj (IDENTICAL to quantum SingleQuantumCircuit.enc_proj)
        self.enc_proj = nn.Sequential(
            nn.Linear(input_dim, enc_proj_hidden),
            nn.SiLU(),
            nn.Linear(enc_proj_hidden, self.enc_per_block),
        )

        # Classical core (REPLACES quantum circuit: SU(N) + QViT + ANO)
        self.core = nn.Sequential(
            nn.Linear(self.enc_per_block, core_hidden),
            nn.SiLU(),
            nn.Linear(core_hidden, self.n_obs),
        )

        # vel_head (IDENTICAL to quantum QuantumVelocityField.vel_head)
        _vh = max(256, self.n_obs)
        self.vel_head = nn.Sequential(
            nn.Linear(self.n_obs, _vh),
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
        x = torch.cat([z_t, t_emb], dim=-1)
        enc = self.enc_proj(x)
        core_out = self.core(enc)
        return self.vel_head(core_out)

    def get_eigenvalue_range(self):
        return 0.0, 0.0  # no ANO -- CSV compat


# ---------------------------------------------------------------------------
# 5e. Velocity Field Factory
# ---------------------------------------------------------------------------
def build_velocity_field(args, device):
    """Factory: build quantum, classical, or classical_sandwich VF."""
    if args.velocity_field == "classical":
        hidden = [int(d) for d in args.mlp_hidden_dims.split(",")]
        vf = ClassicalVelocityField(
            latent_dim=args.latent_dim, hidden_dims=hidden,
            time_embed_dim=args.time_embed_dim).to(device)

    elif args.velocity_field == "classical_sandwich":
        vf = ClassicalSandwichVelocityField(
            latent_dim=args.latent_dim,
            n_qubits=args.n_qubits,
            encoding_type=args.encoding_type,
            sun_group_size=args.sun_group_size,
            enc_proj_hidden=args.enc_proj_hidden,
            k_local=args.k_local,
            obs_scheme=args.obs_scheme,
            core_hidden=args.core_hidden,
            time_embed_dim=args.time_embed_dim,
        ).to(device)

    else:  # quantum
        vf = QuantumVelocityField(
            latent_dim=args.latent_dim,
            n_circuits=args.n_circuits,
            n_qubits=args.n_qubits,
            encoding_type=args.encoding_type,
            sun_group_size=args.sun_group_size,
            enc_proj_hidden=args.enc_proj_hidden,
            vqc_type=args.vqc_type,
            vqc_depth=args.vqc_depth,
            k_local=args.k_local,
            obs_scheme=args.obs_scheme,
            qvit_circuit=args.qvit_circuit,
            time_embed_dim=args.time_embed_dim,
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

    perc_fn = None
    if args.lambda_perc > 0:
        perc_fn = VGGPerceptualLoss().to(device)
        print(f"[Phase 1] VGG perceptual loss enabled "
              f"(lambda={args.lambda_perc})")

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
              "val_psnr", "val_ssim", "val_lpips",
              "beta_eff", "lr", "time_s"]

    from torchmetrics.image import (StructuralSimilarityIndexMeasure,
                                    PeakSignalNoiseRatio)
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    lpips_metric = LearnedPerceptualImagePatchSimilarity(
        net_type="squeeze").to(device)

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
        psnr_metric.reset()
        ssim_metric.reset()
        lpips_metric.reset()
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

                x_hat_clamped = x_hat.clamp(0, 1)
                psnr_metric.update(x_hat_clamped, xb)
                ssim_metric.update(x_hat_clamped, xb)
                lpips_metric.update(x_hat_clamped, xb)

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
        vl_psnr = psnr_metric.compute().item()
        vl_ssim = ssim_metric.compute().item()
        vl_lpips = lpips_metric.compute().item()
        dt = time.time() - t0

        row = dict(epoch=epoch + 1, train_loss=f"{tr_loss:.6f}",
                   train_recon=f"{tr_rec:.6f}", train_kl=f"{tr_kl:.6f}",
                   train_perc=f"{tr_perc:.6f}",
                   val_loss=f"{vl_loss:.6f}", val_recon=f"{vl_rec:.6f}",
                   val_kl=f"{vl_kl:.6f}", val_perc=f"{vl_perc:.6f}",
                   val_psnr=f"{vl_psnr:.4f}", val_ssim=f"{vl_ssim:.4f}",
                   val_lpips=f"{vl_lpips:.4f}",
                   beta_eff=f"{beta_eff:.4f}", lr=f"{cur_lr:.2e}",
                   time_s=f"{dt:.1f}")
        with open(csv_path, "a", newline="") as f:
            csv.DictWriter(f, fields).writerow(row)

        print(f"  Ep {epoch+1:3d} | Train {tr_loss:.4f} "
              f"(rec={tr_rec:.4f} kl={tr_kl:.4f} perc={tr_perc:.4f}) | "
              f"Val {vl_loss:.4f} | "
              f"PSNR {vl_psnr:.2f} SSIM {vl_ssim:.4f} LPIPS {vl_lpips:.4f} | "
              f"beta={beta_eff:.3f} lr={cur_lr:.2e} | {dt:.1f}s")

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
# 7. Phase 2 -- CFM Training
# ---------------------------------------------------------------------------
def train_cfm(args, vae, train_loader, val_loader, device):
    """Train the velocity field (quantum, classical, or sandwich) with CFM."""

    # Freeze VAE
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False

    vf = build_velocity_field(args, device)

    # Dual optimizer (Chen 2025): ANO params at higher LR
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

    total_p = sum(p.numel() for p in vf.parameters() if p.requires_grad)
    h_p = sum(p.numel() for p in H_params)
    c_p = sum(p.numel() for p in circuit_params)
    print(f"[Phase 2] Velocity field params: total={total_p}  "
          f"circuit={c_p}  observable={h_p}")

    input_dim = args.latent_dim + args.time_embed_dim

    if args.velocity_field == "quantum":
        enc_per = vf.circuits[0].enc_per_block
        sun_N = 2 ** args.sun_group_size
        print(f"  Architecture: K={args.n_circuits} circuit(s), "
              f"{args.n_qubits}q each")
        print(f"  Time embed: dim={args.time_embed_dim} + time_mlp")
        print(f"  Input: concat(z_t[{args.latent_dim}], "
              f"t_emb[{args.time_embed_dim}]) = {input_dim} dims "
              f"(SHARED to all circuits)")
        print(f"  Encoding: SU({sun_N}) (group_size={args.sun_group_size}), "
              f"{enc_per} params/circuit")
        print(f"  enc_proj: Linear({input_dim},{args.enc_proj_hidden}) -> "
              f"SiLU -> Linear({args.enc_proj_hidden},{enc_per})")
        print(f"  VQC: {args.vqc_type} ({args.qvit_circuit}), "
              f"depth={args.vqc_depth}")
        print(f"  ANO: k_local={args.k_local}, scheme={args.obs_scheme}, "
              f"n_obs/circuit={vf.n_obs_per_circuit}, "
              f"total_obs={vf.total_obs}")
        print(f"  Ratio: {vf.total_obs}/{args.latent_dim} = "
              f"{vf.total_obs / args.latent_dim:.2f}")
        _vh = max(256, vf.total_obs)
        print(f"  vel_head: Linear({vf.total_obs},{_vh}) -> SiLU -> "
              f"Linear({_vh},{args.latent_dim})")

    elif args.velocity_field == "classical_sandwich":
        enc_per = vf.enc_per_block
        sun_N = 2 ** args.sun_group_size
        print(f"  Architecture: Classical Sandwich (Classical-D control)")
        print(f"  Time embed: dim={args.time_embed_dim} + time_mlp")
        print(f"  Input: concat(z_t[{args.latent_dim}], "
              f"t_emb[{args.time_embed_dim}]) = {input_dim} dims")
        print(f"  enc_proj: Linear({input_dim},{args.enc_proj_hidden}) -> "
              f"SiLU -> Linear({args.enc_proj_hidden},{enc_per})  "
              f"[SAME as SU({sun_N}) quantum]")
        print(f"  core: Linear({enc_per},{args.core_hidden}) -> SiLU -> "
              f"Linear({args.core_hidden},{vf.n_obs})  "
              f"[REPLACES quantum circuit]")
        _vh = max(256, vf.n_obs)
        print(f"  vel_head: Linear({vf.n_obs},{_vh}) -> SiLU -> "
              f"Linear({_vh},{args.latent_dim})  [SAME as quantum]")
        print(f"  n_obs={vf.n_obs}, ratio={vf.n_obs/args.latent_dim:.2f}")

    else:  # classical MLP
        print(f"  Classical MLP: hidden_dims={args.mlp_hidden_dims}")
        print(f"  Time embed: dim={args.time_embed_dim} + time_mlp")

    ckpt_dir = os.path.join(args.base_path, "checkpoints")
    results_dir = os.path.join(args.base_path, "results")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f"ckpt_cfm_{args.job_id}.pt")
    csv_path = os.path.join(results_dir, f"log_cfm_{args.job_id}.csv")
    fields = ["epoch", "train_loss", "val_loss", "eig_min", "eig_max",
              "time_s"]

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
                z_1, _ = vae.encode(xb)

            z_0 = torch.randn_like(z_1)
            t = torch.rand(z_1.size(0), device=device)

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

        print(f"  Ep {epoch+1:3d} | Train {tr_loss:.6f} | Val {vl_loss:.6f} "
              f"| Eig [{eig_lo:.2f}, {eig_hi:.2f}] | {dt:.1f}s")

        if vl_loss < best_val:
            best_val = vl_loss
            best_state = copy.deepcopy(vf.state_dict())

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

    z = torch.randn(n_samples, latent_dim, device=device)
    dt = 1.0 / ode_steps

    for step in range(ode_steps):
        t_val = step * dt
        t = torch.full((n_samples,), t_val, device=device)
        v = vf(z, t)
        z = z + dt * v

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

    n_real = 0
    for (xb,) in real_loader:
        xb = xb.to(device)
        fid.update(xb, real=True)
        n_real += xb.size(0)
        if n_real >= n_samples:
            break

    vae.eval()
    vf.eval()
    n_gen = 0
    while n_gen < n_samples:
        bs = min(batch_size, n_samples - n_gen)
        z = torch.randn(bs, latent_dim, device=device)
        dt_step = 1.0 / ode_steps
        for step in range(ode_steps):
            t_val = step * dt_step
            t = torch.full((bs,), t_val, device=device)
            v = vf(z, t)
            z = z + dt_step * v
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

    if args.velocity_field == "quantum":
        sun_N = 2 ** args.sun_group_size
        label = (f"Quantum {args.n_circuits}x{args.n_qubits}q "
                 f"SU({sun_N})")
    elif args.velocity_field == "classical_sandwich":
        sun_N = 2 ** args.sun_group_size
        label = f"Classical-D (sandwich, SU({sun_N}) dims)"
    else:
        label = "Classical MLP"

    print(f"Phase: {args.phase}  |  Dataset: {args.dataset}  |  "
          f"Latent dim: {args.latent_dim}  |  VF: {label}")

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
        print(f"\n=== Phase 2: {label} CFM Training ===")

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
