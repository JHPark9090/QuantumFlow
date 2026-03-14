"""
Quantum Latent CFM v14 — Custom VAE (1-downsample) + Multi-Chip Ensemble
=========================================================================

Custom VAE with minimal downsampling (32x32 → 16x16, 1 downsample only)
to preserve more spatial structure in the latent space, combined with
Multi-Chip Ensemble quantum velocity field (from v13).

VAE Architecture (Phase 1):
  Encoder: 32x32x3 → 1 downsample → 16x16 x c_z(=16) → flatten → 4096
  Decoder: 4096 → reshape 16x16 x c_z → 1 upsample → 32x32x3

Multi-Chip Ensemble (Phase 2):
  z_flat (4096) → split into 16 chunks of 256
  Each chip: chunk_i + time → SU(16) → ANO → vel_head_i → 256
  Concat all → velocity (4096)

Configurations:

  v14a: custom VAE (1-downsample) + 16 chips, concat time conditioning
        - Per chip: [chunk(256), t_emb(256)] = 512 → enc_proj → 255 (2.01:1)

  v14b: custom VAE (1-downsample) + 16 chips, additive time conditioning
        - Per chip: (chunk + t_emb)[:, :255] = 255 (1.00:1, no enc_proj)

All share:
  - Custom VAE: 1 downsample (32→16x16), c_z=16, latent_dim=4096
  - CIFAR-10 at native 32x32 resolution (no upscaling)
  - 16 chips, each: 4 qubits, SU(16) (255 generators)
  - Pairwise ANO k=2: C(4,2)=6 observables per chip
  - Per-chip vel_head: 6 → 256, concat → 4096
  - v9 training improvements (logit-normal, midpoint ODE, VF EMA)

Usage:
  # Phase 1: Train VAE
  python QuantumLatentCFM_v14.py --phase=1 --epochs=200 \\
      --job-id=qlcfm_v14_vae

  # Phase 2 v14a: concat
  python QuantumLatentCFM_v14.py --phase=2 --time-conditioning=concat \\
      --vae-ckpt=checkpoints/weights_vae_qlcfm_v14_vae.pt \\
      --job-id=qlcfm_v14a_001

  # Phase 2 v14b: additive (no enc_proj)
  python QuantumLatentCFM_v14.py --phase=2 --time-conditioning=additive \\
      --vae-ckpt=checkpoints/weights_vae_qlcfm_v14_vae.pt \\
      --job-id=qlcfm_v14b_001

References:
  - Lipman et al. (2023). Flow Matching for Generative Modeling. ICLR 2023.
  - Wiersema et al. (2024). Here comes the SU(N). Quantum, 8, 1275.
  - Chen et al. (2025). Learning to Measure QNNs. ICASSP 2025 Workshop.
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
        description="Quantum Latent CFM v14 — Custom VAE (1-downsample) + "
                    "Multi-Chip Ensemble")

    # Phase
    p.add_argument("--phase", type=str, default="1",
                   choices=["1", "2", "generate", "both"],
                   help="1=VAE pretrain, 2=CFM train, generate=sample, both=1+2")

    # VAE
    p.add_argument("--latent-dim", type=int, default=4096,
                   help="Latent dim (default 4096 = c_z(16) * 16 * 16)")
    p.add_argument("--c-z", type=int, default=16,
                   help="VAE bottleneck channels (c_z * 16 * 16 = latent_dim)")
    p.add_argument("--beta", type=float, default=0.5)
    p.add_argument("--beta-warmup-epochs", type=int, default=20)
    p.add_argument("--lambda-perc", type=float, default=0.1)
    p.add_argument("--lr-vae", type=float, default=1e-3)

    # Multi-Chip Ensemble
    p.add_argument("--n-chips", type=int, default=16,
                   help="Number of quantum chips")
    p.add_argument("--n-qubits", type=int, default=4,
                   help="Qubits per chip")
    p.add_argument("--k-local", type=int, default=2,
                   help="Locality of pairwise ANO")

    # Time conditioning
    p.add_argument("--time-conditioning", type=str, default="concat",
                   choices=["concat", "additive"])
    p.add_argument("--time-embed-dim", type=int, default=256)

    # Training
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lr-H", type=float, default=1e-1)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--seed", type=int, default=2025)

    # v9 improvements
    p.add_argument("--logit-normal-std", type=float, default=1.0)
    p.add_argument("--ode-solver", type=str, default="midpoint",
                   choices=["euler", "midpoint"])
    p.add_argument("--vf-ema-decay", type=float, default=0.999)

    # Data
    p.add_argument("--dataset", type=str, default="cifar10",
                   choices=["cifar10", "mnist", "fashion"])
    p.add_argument("--n-train", type=int, default=10000)
    p.add_argument("--n-valtest", type=int, default=2000)
    p.add_argument("--img-size", type=int, default=32)

    # ODE sampling
    p.add_argument("--ode-steps", type=int, default=50)
    p.add_argument("--n-samples", type=int, default=64)

    # I/O
    p.add_argument("--job-id", type=str, default="qlcfm_v14_001")
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


def get_wire_groups(n_qubits, k_local):
    return [list(c) for c in combinations(range(n_qubits), k_local)]


def compute_single_gate_encoding(n_qubits):
    dim = 2 ** n_qubits
    n_generators = dim ** 2 - 1
    wires = list(range(n_qubits))
    return wires, n_generators, n_generators


class EMAModel:
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
# 3. Data Loaders (native 32x32, [0,1] range)
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
    X_tr = X_tr * 2.0 - 1.0
    X_te = X_te * 2.0 - 1.0
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
    X_tr = X_tr * 2.0 - 1.0
    X_te = X_te * 2.0 - 1.0
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
    X_tr = X_tr * 2.0 - 1.0
    X_te = X_te * 2.0 - 1.0
    X_tr = X_tr[torch.randperm(len(X_tr))]
    X_te = X_te[torch.randperm(len(X_te))]
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
# 4. Custom VAE — 1 downsample (32x32 → 16x16)
# ---------------------------------------------------------------------------
class ResBlock(nn.Module):
    """GroupNorm(32) -> SiLU -> Conv3x3 -> GroupNorm(32) -> SiLU -> Conv3x3 + skip."""

    def __init__(self, in_channels, out_channels=None):
        super().__init__()
        out_channels = out_channels or in_channels
        n_groups = min(32, in_channels)
        n_groups_out = min(32, out_channels)
        self.block = nn.Sequential(
            nn.GroupNorm(n_groups, in_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.GroupNorm(n_groups_out, out_channels),
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
        n_groups = min(32, channels)
        self.norm = nn.GroupNorm(n_groups, channels)
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


class ResConvVAE_v14(nn.Module):
    """Custom VAE with 1 downsample only.

    32x32x3 → Conv → ResBlocks → 1 downsample → 16x16 × c_z
    → flatten → fc_mu/fc_logvar → latent_dim (4096)

    Preserves more spatial structure than standard 3-downsample VAEs.
    Output in [-1, 1] via Tanh.
    """

    def __init__(self, latent_dim=4096, in_channels=3, c_z=16):
        super().__init__()
        self.latent_dim = latent_dim
        self.c_z = c_z

        # Encoder: 32x32 → 16x16
        self.encoder = nn.Sequential(
            # 32x32x3 → 32x32x64
            nn.Conv2d(in_channels, 64, 3, 1, 1, bias=False),
            ResBlock(64), ResBlock(64),

            # 32x32x64 → 16x16x128 (the single downsample)
            nn.Conv2d(64, 128, 3, 2, 1, bias=False),
            ResBlock(128), ResBlock(128),
            SelfAttention(128),

            # 16x16x128 → 16x16x c_z (channel reduction)
            ResBlock(128, 64),
            ResBlock(64),
            nn.GroupNorm(min(32, 64), 64), nn.SiLU(inplace=True),
            nn.Conv2d(64, c_z, 3, 1, 1, bias=False),
        )

        flat_dim = c_z * 16 * 16  # 16 * 256 = 4096
        self.fc_mu = nn.Linear(flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(flat_dim, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, flat_dim)

        # Decoder: 16x16 → 32x32
        self.decoder = nn.Sequential(
            nn.Conv2d(c_z, 64, 3, 1, 1, bias=False),
            ResBlock(64), ResBlock(64),
            ResBlock(64, 128),
            SelfAttention(128),
            ResBlock(128), ResBlock(128),

            # 16x16x128 → 32x32x64 (the single upsample)
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            ResBlock(64), ResBlock(64),
            nn.GroupNorm(min(32, 64), 64), nn.SiLU(inplace=True),
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
        h = F.silu(self.fc_dec(z)).view(-1, self.c_z, 16, 16)
        return self.decoder(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        return self.decode(self.reparameterize(mu, logvar)), mu, logvar


def build_vae(args):
    return ResConvVAE_v14(latent_dim=args.latent_dim, in_channels=3,
                          c_z=args.c_z)


# ---------------------------------------------------------------------------
# 4b. VGG Perceptual Loss
# ---------------------------------------------------------------------------
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
        # Convert [-1, 1] -> [0, 1] before ImageNet normalization
        x = (x * 0.5 + 0.5 - self.mean) / self.std
        y = (y * 0.5 + 0.5 - self.mean) / self.std
        loss = 0.0
        for block in self.blocks:
            x = block(x)
            y = block(y)
            loss += F.l1_loss(x, y)
        return loss / len(self.blocks)


# ---------------------------------------------------------------------------
# 5a. Classical Velocity Field
# ---------------------------------------------------------------------------
class ClassicalVelocityField(nn.Module):
    def __init__(self, latent_dim=32, hidden_dims=(256, 256, 256),
                 time_embed_dim=64, time_conditioning="concat"):
        super().__init__()
        self.latent_dim = latent_dim
        self.time_embed_dim = time_embed_dim
        self.time_conditioning = time_conditioning

        if time_conditioning == "additive":
            self.time_mlp = nn.Sequential(
                nn.Linear(time_embed_dim, latent_dim),
                nn.SiLU(),
                nn.Linear(latent_dim, latent_dim),
            )
            input_dim = latent_dim
        else:
            self.time_mlp = nn.Sequential(
                nn.Linear(time_embed_dim, time_embed_dim),
                nn.SiLU(),
                nn.Linear(time_embed_dim, time_embed_dim),
            )
            input_dim = latent_dim + time_embed_dim

        dims = [input_dim] + list(hidden_dims)
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
        if self.time_conditioning == "additive":
            z_combined = z_t + t_emb
        else:
            z_combined = torch.cat([z_t, t_emb], dim=-1)
        return self.net(z_combined)

    def get_eigenvalue_range(self):
        return 0.0, 0.0


# ---------------------------------------------------------------------------
# 5b. Single Chip — SU(16) with Pairwise ANO
# ---------------------------------------------------------------------------
class ChipCircuit(nn.Module):
    """A single quantum chip: SU(2^n) encoding + pairwise ANO measurement.

    v14a (concat): enc_proj MLP maps (chunk_dim + t_emb_dim) -> n_gen.
    v14b (additive): no enc_proj; directly slices (chunk + t_emb) to n_gen.
    """

    def __init__(self, chunk_dim, n_qubits, k_local=2,
                 time_conditioning="concat", time_embed_dim=256,
                 chip_id=0):
        super().__init__()
        self.n_qubits = n_qubits
        self.chunk_dim = chunk_dim
        self.time_conditioning = time_conditioning
        self.chip_id = chip_id

        wires, n_gen, enc_size = compute_single_gate_encoding(n_qubits)
        self.enc_wires = wires
        self.n_generators = n_gen
        self.enc_size = enc_size

        if time_conditioning == "concat":
            input_dim = chunk_dim + time_embed_dim
            self.enc_proj = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.SiLU(),
                nn.Linear(256, enc_size),
            )
            self.use_enc_proj = True
            ratio = input_dim / enc_size
            if chip_id == 0:
                print(f"  [Chip {chip_id}] concat: enc_proj "
                      f"{input_dim} -> {enc_size} ({ratio:.2f}:1)")
        else:
            self.use_enc_proj = False
            if chip_id == 0:
                print(f"  [Chip {chip_id}] additive: slice "
                      f"{chunk_dim} -> {enc_size} (no enc_proj)")

        # Pairwise ANO
        K = 2 ** k_local
        self.obs_dim = K
        self.wire_groups = get_wire_groups(n_qubits, k_local)
        self.n_obs = len(self.wire_groups)
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

        ano_params = self.n_obs * (2 * n_off + K)
        if chip_id == 0:
            print(f"  [Chip {chip_id}] SU({2**n_qubits}) on {n_qubits}q, "
                  f"{n_gen} gen, pairwise ANO k={k_local}: "
                  f"{self.n_obs} obs, {ano_params} ANO params")

        # Per-chip velocity head: n_obs -> chunk_dim
        self.vel_head = nn.Sequential(
            nn.Linear(self.n_obs, max(256, self.n_obs)),
            nn.SiLU(),
            nn.Linear(max(256, self.n_obs), chunk_dim),
        )

        # Build QNode
        dev = qml.device("default.qubit")
        _wg = self.wire_groups
        _no = self.n_obs
        _ew = wires
        _ng = n_gen

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def _circuit(enc, H_mats):
            qml.SpecialUnitary(enc[..., :_ng], wires=_ew)
            return [qml.expval(qml.Hermitian(H_mats[w], wires=_wg[w]))
                    for w in range(_no)]

        self._circuit = _circuit

    def _build_H_matrices(self):
        return [create_Hermitian(self.obs_dim, self.A[w], self.B[w], self.D[w])
                for w in range(self.n_obs)]

    def forward(self, chunk_combined):
        if self.use_enc_proj:
            enc = self.enc_proj(chunk_combined)
        else:
            enc = chunk_combined[..., :self.n_generators]

        H_mats = self._build_H_matrices()
        q_out = self._circuit(enc, H_mats)
        q_stack = torch.stack(q_out, dim=1).float()
        return self.vel_head(q_stack)

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
# 5c. Multi-Chip Ensemble Quantum Velocity Field
# ---------------------------------------------------------------------------
class MultiChipQuantumVelocityField(nn.Module):
    """Multi-Chip Ensemble velocity field (from v13).

    Splits the full latent into N_chips chunks, each processed by an
    independent SU(16) quantum circuit with pairwise ANO.
    """

    def __init__(self, latent_dim, n_chips, n_qubits, k_local=2,
                 time_embed_dim=256, time_conditioning="concat"):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_chips = n_chips
        self.n_qubits = n_qubits
        self.time_embed_dim = time_embed_dim
        self.time_conditioning = time_conditioning

        assert latent_dim % n_chips == 0, \
            f"latent_dim ({latent_dim}) must be divisible by n_chips ({n_chips})"
        self.chunk_dim = latent_dim // n_chips

        print(f"\n  Multi-Chip Ensemble: {n_chips} chips x {n_qubits}q "
              f"SU({2**n_qubits})")
        print(f"  Latent {latent_dim} -> {n_chips} chunks of {self.chunk_dim}")
        print(f"  Time conditioning: {time_conditioning}")

        if time_conditioning == "additive":
            self.time_mlp = nn.Sequential(
                nn.Linear(time_embed_dim, self.chunk_dim),
                nn.SiLU(),
                nn.Linear(self.chunk_dim, self.chunk_dim),
            )
        else:
            self.time_mlp = nn.Sequential(
                nn.Linear(time_embed_dim, time_embed_dim),
                nn.SiLU(),
                nn.Linear(time_embed_dim, time_embed_dim),
            )

        self.chips = nn.ModuleList()
        for i in range(n_chips):
            self.chips.append(ChipCircuit(
                chunk_dim=self.chunk_dim,
                n_qubits=n_qubits,
                k_local=k_local,
                time_conditioning=time_conditioning,
                time_embed_dim=time_embed_dim,
                chip_id=i,
            ))

        self.n_obs_per_chip = self.chips[0].n_obs

    def _time_embedding(self, t):
        half = self.time_embed_dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device) / half)
        args = t.unsqueeze(-1) * freqs
        return torch.cat([args.cos(), args.sin()], dim=-1)

    def forward(self, z_t, t):
        t_emb = self.time_mlp(self._time_embedding(t))

        chunks = z_t.split(self.chunk_dim, dim=1)

        vel_chunks = []
        for i, chip in enumerate(self.chips):
            if self.time_conditioning == "additive":
                chip_input = chunks[i] + t_emb
            else:
                chip_input = torch.cat([chunks[i], t_emb], dim=-1)
            vel_chunks.append(chip(chip_input))

        return torch.cat(vel_chunks, dim=1)

    def get_eigenvalue_range(self):
        lo, hi = float("inf"), float("-inf")
        for chip in self.chips:
            c_lo, c_hi = chip.get_eigenvalue_range()
            lo = min(lo, c_lo)
            hi = max(hi, c_hi)
        return lo, hi


def build_velocity_field(args, device):
    if args.velocity_field == "classical":
        hidden = [int(d) for d in args.mlp_hidden_dims.split(",")]
        vf = ClassicalVelocityField(
            latent_dim=args.latent_dim, hidden_dims=hidden,
            time_embed_dim=args.time_embed_dim,
            time_conditioning=args.time_conditioning).to(device)
    else:
        vf = MultiChipQuantumVelocityField(
            latent_dim=args.latent_dim,
            n_chips=args.n_chips,
            n_qubits=args.n_qubits,
            k_local=args.k_local,
            time_embed_dim=args.time_embed_dim,
            time_conditioning=args.time_conditioning).to(device)
    return vf


# ---------------------------------------------------------------------------
# 6a. Phase 1 — VAE Pretraining
# ---------------------------------------------------------------------------
def train_vae(args, train_loader, val_loader, device):
    vae = build_vae(args).to(device)
    optimizer = torch.optim.Adam(vae.parameters(), lr=args.lr_vae)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6)

    perc_fn = None
    if args.lambda_perc > 0:
        perc_fn = VGGPerceptualLoss().to(device)
        print(f"[Phase 1] VGG perceptual loss enabled "
              f"(lambda={args.lambda_perc})")

    # Evaluation metrics (PSNR, SSIM, LPIPS)
    from torchmetrics.image import (StructuralSimilarityIndexMeasure,
                                    PeakSignalNoiseRatio)
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    psnr_metric = PeakSignalNoiseRatio(data_range=2.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=2.0).to(device)
    lpips_eval = LearnedPerceptualImagePatchSimilarity(
        net_type="alex", normalize=True).to(device)
    for p in lpips_eval.parameters():
        p.requires_grad = False

    total_p = sum(p.numel() for p in vae.parameters() if p.requires_grad)
    print(f"[Phase 1] VAE v14 (1-downsample)  params: {total_p:,}")
    print(f"  Architecture: 32x32 -> 1 downsample -> 16x16 x {args.c_z} "
          f"-> flatten({args.c_z * 16 * 16}) -> latent({args.latent_dim})")

    ckpt_dir = os.path.join(args.base_path, "checkpoints")
    results_dir = os.path.join(args.base_path, "results")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f"ckpt_vae_{args.job_id}.pt")
    csv_path = os.path.join(results_dir, f"log_vae_{args.job_id}.csv")
    fields = ["epoch", "train_loss", "train_recon", "train_kl", "train_perc",
              "val_loss", "val_recon", "val_kl", "val_perc",
              "val_psnr", "val_ssim", "val_lpips_eval",
              "active_dims", "time_s"]

    start_epoch = 0
    best_val, best_psnr, best_state = float("inf"), 0.0, None

    if args.resume and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, weights_only=False)
        vae.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_val = ckpt.get("best_val", float("inf"))
        best_psnr = ckpt.get("best_psnr", 0.0)
        print(f"  Resumed from epoch {start_epoch}, "
              f"best PSNR={best_psnr:.2f}")

    if start_epoch == 0:
        with open(csv_path, "w", newline="") as f:
            csv.DictWriter(f, fields).writeheader()

    warmup = max(args.beta_warmup_epochs, 1)

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        beta_eff = args.beta * min(1.0, (epoch + 1) / warmup)

        vae.train()
        tr_loss, tr_rec, tr_kl, tr_perc, tr_n = 0., 0., 0., 0., 0
        for (xb,) in tqdm(train_loader,
                          desc=f"VAE Ep {epoch+1}/{args.epochs}", leave=False):
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
        psnr_metric.reset(); ssim_metric.reset(); lpips_eval.reset()
        kl_per_dim_all = []
        with torch.no_grad():
            for (xb,) in val_loader:
                xb = xb.to(device)
                x_hat, mu, logvar = vae(xb)
                recon = F.mse_loss(x_hat, xb)
                kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
                kl_per_dim_all.append(kl_per_dim)
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
                x_hat_c = x_hat.clamp(-1, 1)
                psnr_metric.update(x_hat_c, xb)
                ssim_metric.update(x_hat_c, xb)
                # LPIPS expects [0,1] even with normalize=True
                lpips_eval.update((x_hat_c + 1) / 2, (xb + 1) / 2)

        vl_loss /= vl_n; vl_rec /= vl_n; vl_kl /= vl_n; vl_perc /= vl_n
        vl_psnr = psnr_metric.compute().item()
        vl_ssim = ssim_metric.compute().item()
        vl_lpips_e = lpips_eval.compute().item()
        kl_per_dim_combined = torch.cat(kl_per_dim_all, dim=0).mean(dim=0)
        active_dims = (kl_per_dim_combined > 0.01).sum().item()
        dt = time.time() - t0

        row = dict(epoch=epoch + 1, train_loss=f"{tr_loss:.6f}",
                   train_recon=f"{tr_rec:.6f}", train_kl=f"{tr_kl:.6f}",
                   train_perc=f"{tr_perc:.6f}",
                   val_loss=f"{vl_loss:.6f}", val_recon=f"{vl_rec:.6f}",
                   val_kl=f"{vl_kl:.6f}", val_perc=f"{vl_perc:.6f}",
                   val_psnr=f"{vl_psnr:.2f}", val_ssim=f"{vl_ssim:.4f}",
                   val_lpips_eval=f"{vl_lpips_e:.4f}",
                   active_dims=int(active_dims),
                   time_s=f"{dt:.1f}")
        with open(csv_path, "a", newline="") as f:
            csv.DictWriter(f, fields).writerow(row)

        print(f"  Ep {epoch+1:3d} | Train {tr_loss:.4f} "
              f"(rec={tr_rec:.4f} kl={tr_kl:.4f} perc={tr_perc:.4f}) | "
              f"Val {vl_loss:.4f} | "
              f"PSNR {vl_psnr:.2f} SSIM {vl_ssim:.4f} LPIPS {vl_lpips_e:.4f} | "
              f"active={int(active_dims)}/{args.latent_dim} | "
              f"beta={beta_eff:.3f} | {dt:.1f}s")

        if vl_psnr > best_psnr:
            best_psnr = vl_psnr
            best_val = vl_loss
            best_state = copy.deepcopy(vae.state_dict())
            print(f"    >> New best PSNR: {best_psnr:.2f} dB")

        torch.save(dict(epoch=epoch, model=vae.state_dict(),
                        optimizer=optimizer.state_dict(),
                        scheduler=scheduler.state_dict(),
                        best_val=best_val, best_psnr=best_psnr), ckpt_path)
        scheduler.step()

    if best_state is not None:
        vae.load_state_dict(best_state)
        print(f"\nLoaded best VAE (PSNR={best_psnr:.2f} dB)")

    # Final test evaluation
    vae.eval()
    psnr_metric.reset(); ssim_metric.reset(); lpips_eval.reset()
    kl_per_dim_all = []
    with torch.no_grad():
        for (xb,) in val_loader:
            xb = xb.to(device)
            x_hat, mu, logvar = vae(xb)
            kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
            kl_per_dim_all.append(kl_per_dim)
            x_hat_c = x_hat.clamp(-1, 1)
            psnr_metric.update(x_hat_c, xb)
            ssim_metric.update(x_hat_c, xb)
            # LPIPS expects [0,1] even with normalize=True
            lpips_eval.update((x_hat_c + 1) / 2, (xb + 1) / 2)

    test_psnr = psnr_metric.compute().item()
    test_ssim = ssim_metric.compute().item()
    test_lpips = lpips_eval.compute().item()
    kl_per_dim_combined = torch.cat(kl_per_dim_all, dim=0).mean(dim=0)
    active_dims = int((kl_per_dim_combined > 0.01).sum().item())

    print(f"\n=== Final VAE Metrics (best checkpoint) ===")
    print(f"  PSNR:  {test_psnr:.2f} dB")
    print(f"  SSIM:  {test_ssim:.4f}")
    print(f"  LPIPS: {test_lpips:.4f}")
    print(f"  Active dims: {active_dims}/{args.latent_dim}")

    w_path = os.path.join(ckpt_dir, f"weights_vae_{args.job_id}.pt")
    torch.save(vae.state_dict(), w_path)
    print(f"  VAE saved to {w_path}")

    # Save metrics summary
    metrics = dict(psnr=test_psnr, ssim=test_ssim, lpips=test_lpips,
                   active_dims=active_dims, latent_dim=args.latent_dim)
    metrics_path = os.path.join(results_dir,
                                f"metrics_vae_{args.job_id}.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Metrics saved to {metrics_path}")

    return vae


# ---------------------------------------------------------------------------
# 6b. Phase 2 — CFM Training (v9 improvements)
# ---------------------------------------------------------------------------
def train_cfm(args, vae, train_loader, val_loader, device):
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

    vf_ema = None
    if args.vf_ema_decay > 0:
        vf_ema = EMAModel(vf, decay=args.vf_ema_decay)

    total_p = sum(p.numel() for p in vf.parameters() if p.requires_grad)
    h_p = sum(p.numel() for p in H_params)
    c_p = sum(p.numel() for p in circuit_params)
    print(f"[Phase 2] Velocity field params: total={total_p}  "
          f"circuit={c_p}  observable={h_p}")

    if args.logit_normal_std > 0:
        print(f"  [v9] Logit-normal timestep sampling "
              f"(std={args.logit_normal_std})")
    print(f"  [v9] ODE solver: {args.ode_solver} ({args.ode_steps} steps)")
    if vf_ema:
        print(f"  [v9] VF EMA enabled (decay={args.vf_ema_decay})")

    if args.velocity_field == "quantum":
        n_gen = vf.chips[0].n_generators
        chunk_dim = vf.chunk_dim
        print(f"  Multi-Chip Ensemble: {args.n_chips} chips x "
              f"SU({2**args.n_qubits})")
        print(f"  Per chip: chunk_dim={chunk_dim}, "
              f"SU generators={n_gen}, ANO obs={vf.n_obs_per_chip}")

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

        vf.train()
        tr_loss, tr_n = 0.0, 0
        for (xb,) in tqdm(train_loader,
                          desc=f"CFM Ep {epoch+1}/{args.epochs}", leave=False):
            xb = xb.to(device)

            with torch.no_grad():
                z_1, _ = vae.encode(xb)

            z_0 = torch.randn_like(z_1)

            if args.logit_normal_std > 0:
                t = torch.sigmoid(
                    torch.randn(z_1.size(0), device=device)
                    * args.logit_normal_std)
            else:
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

        track_val = vl_loss_ema if vf_ema else vl_loss
        if track_val < best_val:
            best_val = track_val
            if vf_ema:
                vf_ema.apply(vf)
                best_state = copy.deepcopy(vf.state_dict())
                vf.load_state_dict(orig_state)
            else:
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
        if vf_ema:
            ckpt_data["vf_ema"] = vf_ema.state_dict()
        torch.save(ckpt_data, ckpt_path)

    if best_state is not None:
        vf.load_state_dict(best_state)

    w_path = os.path.join(ckpt_dir, f"weights_cfm_{args.job_id}.pt")
    torch.save(vf.state_dict(), w_path)
    print(f"  CFM velocity field saved to {w_path}")
    return vf


# ---------------------------------------------------------------------------
# 7. Generation (midpoint ODE)
# ---------------------------------------------------------------------------
@torch.no_grad()
def generate_samples(vae, vf, n_samples, ode_steps, latent_dim, device,
                     save_path=None, ode_solver="midpoint"):
    vae.eval()
    vf.eval()

    z = torch.randn(n_samples, latent_dim, device=device)
    dt = 1.0 / ode_steps

    for step in range(ode_steps):
        t_val = step * dt
        t = torch.full((n_samples,), t_val, device=device)

        if ode_solver == "midpoint":
            k1 = vf(z, t)
            z_mid = z + 0.5 * dt * k1
            t_mid = torch.full((n_samples,), t_val + 0.5 * dt, device=device)
            k2 = vf(z_mid, t_mid)
            z = z + dt * k2
        else:
            v = vf(z, t)
            z = z + dt * v

    images = vae.decode(z)
    images = (images.clamp(-1, 1) + 1) / 2  # Tanh [-1,1] -> [0,1]

    if save_path:
        from torchvision.utils import save_image
        save_image(images, save_path, nrow=8, padding=2)
        print(f"  Generated {n_samples} samples -> {save_path}")

    return images


# ---------------------------------------------------------------------------
# 7b. FID & IS Evaluation
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
        xb_01 = (xb.clamp(-1, 1) + 1) / 2  # [-1,1] -> [0,1]
        fid.update(xb_01, real=True)
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
# 8. Main
# ---------------------------------------------------------------------------
def main():
    args = get_args()
    set_all_seeds(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"PennyLane {qml.__version__}  |  PyTorch {torch.__version__}  |  "
          f"{device}")

    # Validate latent_dim = c_z * 16 * 16
    expected_latent = args.c_z * 16 * 16
    if args.latent_dim != expected_latent:
        print(f"WARNING: latent_dim ({args.latent_dim}) != "
              f"c_z({args.c_z}) * 16 * 16 = {expected_latent}")
        print(f"  Setting latent_dim = {expected_latent}")
        args.latent_dim = expected_latent

    chunk_dim = args.latent_dim // args.n_chips
    n_gen = (2 ** args.n_qubits) ** 2 - 1
    if args.velocity_field == "quantum":
        label = (f"Multi-Chip {args.n_chips}x{args.n_qubits}q "
                 f"SU({2**args.n_qubits})")
    else:
        label = "Classical"
    print(f"Phase: {args.phase}  |  Dataset: {args.dataset}  |  "
          f"Latent dim: {args.latent_dim}  |  VF: {label}")
    print(f"VAE: 1-downsample (32->16x16), c_z={args.c_z}")
    print(f"Multi-Chip: {args.n_chips} chips, chunk_dim={chunk_dim}, "
          f"SU gen={n_gen}")
    print(f"Time conditioning: {args.time_conditioning}  |  "
          f"time_embed_dim: {args.time_embed_dim}")

    # -- Load data --
    loader_map = {
        "cifar10": load_cifar_2d,
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
        print(f"\n=== Phase 1: VAE Pretraining (v14 — 1-downsample) ===")
        vae = train_vae(args, train_loader, val_loader, device)
    else:
        vae = None

    # -- Phase 2: CFM --
    if args.phase in ("2", "both"):
        print(f"\n=== Phase 2: {label} CFM Training "
              f"(v14 — Multi-Chip Ensemble) ===")

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
            metrics = evaluate_fid_is(
                vae, vf, train_loader, args.n_eval_samples,
                args.ode_steps, args.latent_dim, device,
                ode_solver=args.ode_solver)


if __name__ == "__main__":
    main()
