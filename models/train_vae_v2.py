"""
VAE v2 — Standalone training with fixed KL, free bits, and PSNR-based selection
================================================================================

Fixes over v6 VAE training:
  1. KL normalization: sum over latent dims, mean over batch (proper ELBO)
  2. Beta: 0.001 (was 0.5)
  3. Perceptual weight: lambda_perc=0.01 (was 0.1)
  4. Free bits: 0.25 nats/dim minimum KL (prevents posterior collapse)
  5. Best model selection: highest val PSNR (was lowest total loss)

Additional improvements:
  - logvar clamping [-20, 2] to prevent numerical instability
  - Extended CSV logging with per-dim KL stats, active_dims, mu_norm
  - Reconstruction grid saved every N epochs as PNG
  - Optional reconstruction FID at end of training
  - Beta warmup over 10 epochs (was 20)

Usage:
  # Quick sanity check
  python models/train_vae_v2.py --dataset=cifar10 --n-epochs=2 \\
      --n-train=1000 --n-valtest=200 --job-id=test

  # Full training
  python models/train_vae_v2.py --dataset=cifar10 --vae-arch=resconv \\
      --latent-dim=32 --beta=0.001 --lambda-perc=0.01 --free-bits=0.25 \\
      --n-epochs=200 --seed=2025 --compute-recon-fid \\
      --job-id=vae_v2_cifar_${SLURM_JOB_ID}
"""

import argparse
import os
import sys
import random
import copy
import json
import math
import time
import csv

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm

DATA_ROOT = "/pscratch/sd/j/junghoon/data"


# ---------------------------------------------------------------------------
# 1. Argparse
# ---------------------------------------------------------------------------
def get_args():
    p = argparse.ArgumentParser(description="VAE v2 — Fixed KL Training")

    p.add_argument("--dataset", type=str, default="cifar10",
                   choices=["cifar10", "mnist", "fashion"],
                   help="Dataset to train on")
    p.add_argument("--latent-dim", type=int, default=32,
                   help="VAE latent dimension")
    p.add_argument("--vae-arch", type=str, default="resconv",
                   choices=["resconv", "legacy"],
                   help="VAE architecture")

    # Loss weights (KEY FIXES)
    p.add_argument("--beta", type=float, default=0.001,
                   help="KL weight (v6 used 0.5, now 0.001)")
    p.add_argument("--beta-warmup-epochs", type=int, default=10,
                   help="Linear ramp from 0 to beta (v6 used 20)")
    p.add_argument("--lambda-perc", type=float, default=0.01,
                   help="VGG perceptual loss weight (v6 used 0.1)")
    p.add_argument("--free-bits", type=float, default=0.25,
                   help="Minimum KL per latent dimension in nats (NEW)")

    # Training
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--n-epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=128)

    # Data
    p.add_argument("--n-train", type=int, default=50000)
    p.add_argument("--n-valtest", type=int, default=10000)
    p.add_argument("--img-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=2025)

    # I/O
    p.add_argument("--job-id", type=str, default="vae_v2_001")
    p.add_argument("--base-path", type=str, default=".")
    p.add_argument("--resume", action="store_true")

    # Visualization / evaluation
    p.add_argument("--save-grid-every", type=int, default=10,
                   help="Save reconstruction grid every N epochs (0 to disable)")
    p.add_argument("--compute-recon-fid", action="store_true",
                   help="Compute reconstruction FID at end of training")

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


# ---------------------------------------------------------------------------
# 3. Data Loaders
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
# 4. Model Architecture (copied verbatim from QuantumLatentCFM_v6.py,
#    with logvar clamping added in ResConvVAE.encode)
# ---------------------------------------------------------------------------
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
    """Deep residual VAE for 32x32x3 images (~2.1M params).

    v2 change: logvar clamped to [-20, 2] in encode() for numerical stability.
    """

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
        mu = self.fc_mu(h)
        logvar = torch.clamp(self.fc_logvar(h), -20, 2)
        return mu, logvar

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
        mu = self.fc_mu(h)
        logvar = torch.clamp(self.fc_logvar(h), -20, 2)
        return mu, logvar

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


def build_vae(args, in_channels=3):
    if args.vae_arch == "resconv":
        return ResConvVAE(latent_dim=args.latent_dim, in_channels=in_channels)
    return ConvVAE(latent_dim=args.latent_dim, in_channels=in_channels)


class VGGPerceptualLoss(nn.Module):
    """Perceptual loss using frozen VGG16 features (relu1_2, 2_2, 3_3, 4_3)."""

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
# 5. KL Computation (CORE FIX)
# ---------------------------------------------------------------------------
def compute_kl(mu, logvar, free_bits=0.25):
    """Compute KL divergence with free bits, proper ELBO normalization.

    v6 bug: torch.mean over BOTH batch AND latent dims made KL ~32x too small.
    Fix: sum over latent dims (proper ELBO), mean over batch only.

    Args:
        mu: (batch, latent_dim)
        logvar: (batch, latent_dim)
        free_bits: minimum KL per dimension in nats

    Returns:
        kl: scalar — KL summed over latent dims, averaged over batch
        kl_per_dim_avg: (latent_dim,) — per-dim KL averaged over batch
    """
    # Per-sample, per-dimension KL: shape [batch, latent_dim]
    kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())

    # Average over batch to get per-dimension KL: [latent_dim]
    kl_per_dim_avg = kl_per_dim.mean(dim=0)

    # Free bits: clamp each dim to minimum (prevents posterior collapse)
    kl_per_dim_clamped = torch.clamp(kl_per_dim_avg, min=free_bits)

    # Sum over latent dims (proper ELBO)
    kl = kl_per_dim_clamped.sum()

    return kl, kl_per_dim_avg


# ---------------------------------------------------------------------------
# 6. Visualization
# ---------------------------------------------------------------------------
def save_recon_grid(vae, val_loader, device, save_path, n_images=8):
    """Save a side-by-side grid of originals and reconstructions."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    vae.eval()
    with torch.no_grad():
        for (xb,) in val_loader:
            xb = xb[:n_images].to(device)
            x_hat, _, _ = vae(xb)
            x_hat = x_hat.clamp(0, 1)
            break

    fig, axes = plt.subplots(2, n_images, figsize=(2 * n_images, 4))
    for i in range(n_images):
        orig = xb[i].cpu().permute(1, 2, 0).numpy()
        recon = x_hat[i].cpu().permute(1, 2, 0).numpy()
        axes[0, i].imshow(orig)
        axes[0, i].axis("off")
        if i == 0:
            axes[0, i].set_title("Original", fontsize=10)
        axes[1, i].imshow(recon)
        axes[1, i].axis("off")
        if i == 0:
            axes[1, i].set_title("Recon", fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved reconstruction grid to {save_path}")


# ---------------------------------------------------------------------------
# 7. Reconstruction FID
# ---------------------------------------------------------------------------
def compute_recon_fid(vae, test_loader, device):
    """Compute FID between original test images and their reconstructions."""
    try:
        from torchmetrics.image.fid import FrechetInceptionDistance
    except ImportError:
        print("  [WARN] torchmetrics FID not available, skipping recon FID")
        return float("nan")

    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)

    vae.eval()
    with torch.no_grad():
        for (xb,) in tqdm(test_loader, desc="Recon FID", leave=False):
            xb = xb.to(device)
            x_hat, _, _ = vae(xb)
            x_hat = x_hat.clamp(0, 1)
            fid.update(xb, real=True)
            fid.update(x_hat, real=False)

    return fid.compute().item()


# ---------------------------------------------------------------------------
# 8. Main Training Loop
# ---------------------------------------------------------------------------
def train_vae(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    set_all_seeds(args.seed)

    # Data
    loader_fn = {
        "cifar10": load_cifar_2d,
        "mnist": load_mnist_2d,
        "fashion": load_fashion_2d,
    }[args.dataset]
    train_loader, val_loader, test_loader = loader_fn(
        args.seed, args.n_train, args.n_valtest, args.batch_size, args.img_size)
    print(f"Data: {args.dataset} | train={args.n_train} val+test={args.n_valtest}")

    # Model
    vae = build_vae(args).to(device)
    total_p = sum(p.numel() for p in vae.parameters() if p.requires_grad)
    print(f"VAE arch={args.vae_arch} | params={total_p:,} | latent_dim={args.latent_dim}")

    optimizer = torch.optim.Adam(vae.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.n_epochs, eta_min=1e-6)

    perc_fn = None
    if args.lambda_perc > 0:
        perc_fn = VGGPerceptualLoss().to(device)
        print(f"VGG perceptual loss enabled (lambda={args.lambda_perc})")

    # Reconstruction quality metrics
    from torchmetrics.image import (StructuralSimilarityIndexMeasure,
                                    PeakSignalNoiseRatio)
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    lpips_metric = LearnedPerceptualImagePatchSimilarity(
        net_type="squeeze").to(device)

    # Directories
    ckpt_dir = os.path.join(args.base_path, "checkpoints")
    results_dir = os.path.join(args.base_path, "results")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    ckpt_path = os.path.join(ckpt_dir, f"ckpt_vae_v2_{args.dataset}_{args.job_id}.pt")
    weights_path = os.path.join(ckpt_dir, f"weights_vae_v2_{args.dataset}_{args.job_id}.pt")
    csv_path = os.path.join(results_dir, f"log_vae_v2_{args.dataset}_{args.job_id}.csv")
    metrics_path = os.path.join(results_dir, f"metrics_vae_v2_{args.dataset}_{args.job_id}.json")

    fields = [
        "epoch", "train_loss", "train_recon", "train_kl", "train_perc",
        "val_loss", "val_recon", "val_kl", "val_perc",
        "val_psnr", "val_ssim", "val_lpips",
        "kl_mean_per_dim", "kl_min_per_dim", "kl_max_per_dim",
        "active_dims", "mu_norm",
        "beta_eff", "lr", "time_s",
    ]

    # Print key hyperparameters
    print(f"\n--- VAE v2 Key Fixes ---")
    print(f"  beta={args.beta} (v6 was 0.5)")
    print(f"  lambda_perc={args.lambda_perc} (v6 was 0.1)")
    print(f"  free_bits={args.free_bits} nats/dim")
    print(f"  KL: sum over latent dims, mean over batch (v6 used mean over all)")
    print(f"  Best model by: val PSNR (v6 used lowest total loss)")
    print(f"  Beta warmup: {args.beta_warmup_epochs} epochs (v6 used 20)")
    print(f"  logvar clamped to [-20, 2]")
    print(f"------------------------\n")

    # Resume
    start_epoch = 0
    best_psnr = float("-inf")
    best_state = None

    if args.resume and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, weights_only=False, map_location=device)
        vae.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_psnr = ckpt.get("best_psnr", float("-inf"))
        print(f"Resumed from epoch {start_epoch}, best PSNR so far: {best_psnr:.2f}")

    if start_epoch == 0:
        with open(csv_path, "w", newline="") as f:
            csv.DictWriter(f, fields).writeheader()

    warmup = max(args.beta_warmup_epochs, 1)

    for epoch in range(start_epoch, args.n_epochs):
        t0 = time.time()
        beta_eff = args.beta * min(1.0, (epoch + 1) / warmup)
        cur_lr = optimizer.param_groups[0]["lr"]

        # -- Train --
        vae.train()
        tr_loss, tr_rec, tr_kl, tr_perc, tr_n = 0.0, 0.0, 0.0, 0.0, 0

        for (xb,) in tqdm(train_loader,
                          desc=f"VAE-v2 Ep {epoch+1}/{args.n_epochs}",
                          leave=False):
            xb = xb.to(device)
            x_hat, mu, logvar = vae(xb)

            recon = F.mse_loss(x_hat, xb, reduction="mean")
            kl, _ = compute_kl(mu, logvar, free_bits=args.free_bits)
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
        all_kl_per_dim = []
        all_mu_norms = []

        psnr_metric.reset()
        ssim_metric.reset()
        lpips_metric.reset()

        with torch.no_grad():
            for (xb,) in val_loader:
                xb = xb.to(device)
                x_hat, mu, logvar = vae(xb)

                recon = F.mse_loss(x_hat, xb, reduction="mean")
                kl, kl_per_dim_avg = compute_kl(
                    mu, logvar, free_bits=args.free_bits)
                loss = recon + beta_eff * kl

                perc_val = 0.0
                if perc_fn is not None:
                    perc = perc_fn(x_hat, xb)
                    loss = loss + args.lambda_perc * perc
                    perc_val = perc.item()

                # Per-dim KL stats
                all_kl_per_dim.append(kl_per_dim_avg.cpu())
                all_mu_norms.append(mu.norm(dim=1).mean().item())

                # Reconstruction quality metrics
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

        # Per-dim KL statistics
        kl_per_dim_combined = torch.stack(all_kl_per_dim).mean(dim=0)
        kl_mean_d = kl_per_dim_combined.mean().item()
        kl_min_d = kl_per_dim_combined.min().item()
        kl_max_d = kl_per_dim_combined.max().item()
        active_dims = (kl_per_dim_combined > args.free_bits).sum().item()
        mu_norm = np.mean(all_mu_norms)

        dt = time.time() - t0

        # Log
        row = dict(
            epoch=epoch + 1,
            train_loss=f"{tr_loss:.6f}",
            train_recon=f"{tr_rec:.6f}",
            train_kl=f"{tr_kl:.6f}",
            train_perc=f"{tr_perc:.6f}",
            val_loss=f"{vl_loss:.6f}",
            val_recon=f"{vl_rec:.6f}",
            val_kl=f"{vl_kl:.6f}",
            val_perc=f"{vl_perc:.6f}",
            val_psnr=f"{vl_psnr:.4f}",
            val_ssim=f"{vl_ssim:.4f}",
            val_lpips=f"{vl_lpips:.4f}",
            kl_mean_per_dim=f"{kl_mean_d:.6f}",
            kl_min_per_dim=f"{kl_min_d:.6f}",
            kl_max_per_dim=f"{kl_max_d:.6f}",
            active_dims=int(active_dims),
            mu_norm=f"{mu_norm:.4f}",
            beta_eff=f"{beta_eff:.6f}",
            lr=f"{cur_lr:.2e}",
            time_s=f"{dt:.1f}",
        )
        with open(csv_path, "a", newline="") as f:
            csv.DictWriter(f, fields).writerow(row)

        print(
            f"  Ep {epoch+1:3d} | "
            f"Train {tr_loss:.4f} (rec={tr_rec:.4f} kl={tr_kl:.4f} "
            f"perc={tr_perc:.4f}) | "
            f"Val {vl_loss:.4f} | "
            f"PSNR {vl_psnr:.2f} SSIM {vl_ssim:.4f} LPIPS {vl_lpips:.4f} | "
            f"KL/dim mean={kl_mean_d:.3f} min={kl_min_d:.3f} max={kl_max_d:.3f} "
            f"active={int(active_dims)}/{args.latent_dim} | "
            f"mu_norm={mu_norm:.2f} | "
            f"beta={beta_eff:.4f} lr={cur_lr:.2e} | {dt:.1f}s"
        )

        # Best model by PSNR (KEY FIX: was lowest total loss)
        if vl_psnr > best_psnr:
            best_psnr = vl_psnr
            best_state = copy.deepcopy(vae.state_dict())
            print(f"    >> New best PSNR: {best_psnr:.2f} dB")

        # Checkpoint (for resume)
        torch.save(dict(
            epoch=epoch,
            model=vae.state_dict(),
            optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(),
            vae_arch=args.vae_arch,
            best_psnr=best_psnr,
        ), ckpt_path)

        scheduler.step()

        # Save reconstruction grid
        if (args.save_grid_every > 0
                and (epoch + 1) % args.save_grid_every == 0):
            grid_path = os.path.join(
                results_dir,
                f"recon_grid_vae_v2_epoch{epoch+1}_{args.job_id}.png")
            save_recon_grid(vae, val_loader, device, grid_path)

    # -- Done: load best model --
    if best_state is not None:
        vae.load_state_dict(best_state)
        print(f"\nLoaded best model (PSNR={best_psnr:.2f} dB)")

    torch.save(vae.state_dict(), weights_path)
    print(f"Best VAE weights saved to {weights_path}")

    # Final reconstruction grid
    final_grid_path = os.path.join(
        results_dir, f"recon_grid_vae_v2_final_{args.job_id}.png")
    save_recon_grid(vae, val_loader, device, final_grid_path)

    # Reconstruction FID on test set
    recon_fid = float("nan")
    if args.compute_recon_fid:
        print("\nComputing reconstruction FID on test set...")
        recon_fid = compute_recon_fid(vae, test_loader, device)
        print(f"  Reconstruction FID: {recon_fid:.2f}")

    # Final test-set metrics
    psnr_metric.reset()
    ssim_metric.reset()
    lpips_metric.reset()
    vae.eval()
    with torch.no_grad():
        for (xb,) in test_loader:
            xb = xb.to(device)
            x_hat, _, _ = vae(xb)
            x_hat = x_hat.clamp(0, 1)
            psnr_metric.update(x_hat, xb)
            ssim_metric.update(x_hat, xb)
            lpips_metric.update(x_hat, xb)

    test_psnr = psnr_metric.compute().item()
    test_ssim = ssim_metric.compute().item()
    test_lpips = lpips_metric.compute().item()

    print(f"\n=== Final Test Metrics ===")
    print(f"  PSNR:  {test_psnr:.2f} dB")
    print(f"  SSIM:  {test_ssim:.4f}")
    print(f"  LPIPS: {test_lpips:.4f}")
    if not math.isnan(recon_fid):
        print(f"  Recon FID: {recon_fid:.2f}")

    # Save summary JSON
    summary = {
        "job_id": args.job_id,
        "dataset": args.dataset,
        "vae_arch": args.vae_arch,
        "latent_dim": args.latent_dim,
        "beta": args.beta,
        "lambda_perc": args.lambda_perc,
        "free_bits": args.free_bits,
        "beta_warmup_epochs": args.beta_warmup_epochs,
        "n_epochs": args.n_epochs,
        "seed": args.seed,
        "n_params": total_p,
        "best_val_psnr": best_psnr,
        "test_psnr": test_psnr,
        "test_ssim": test_ssim,
        "test_lpips": test_lpips,
        "recon_fid": recon_fid,
        "weights_path": weights_path,
    }
    with open(metrics_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Metrics saved to {metrics_path}")

    return vae


# ---------------------------------------------------------------------------
# 9. Entry Point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    args = get_args()
    train_vae(args)
