"""
VAE v5 -- Bottleneck Fix: Stop at 4x4, 1x1 Conv Channel Reduction
===================================================================

Same architecture as v3 (ResConvVAE with GroupNorm+SiLU+SelfAttention) but fixes
the two stacked bottlenecks that capped v3/v4 at PSNR ~19.5 dB:

  1. REMOVED: Final stride-2 downsample from 4x4 -> 2x2
  2. REPLACED: Flatten(1024) -> Linear(1024, 64) [16:1 compression]
     WITH:    Conv1x1(256, C_z) at 4x4 -> Flatten(C_z*16) -> Linear(C_z*16, 64)
     Default C_z=4: pre-flatten=64 dims -> Linear(64,64) = 1:1 (no compression)

Decoder mirrors: Linear(64, C_z*16) -> reshape(C_z,4,4) -> Conv1x1(C_z,256) -> ...

Everything else is identical to v3: GroupNorm(32)+SiLU ResBlocks, SelfAttention at
8x8 and 4x4, Tanh [-1,1] output, L1 recon, LPIPS, PatchGAN, EMA, free_bits KL.

Usage:
  # Quick test
  python models/train_vae_v5.py --dataset=cifar10 --n-epochs=5 \\
      --n-train=1000 --n-valtest=200 --job-id=test_v5

  # Full training (300 epochs)
  python models/train_vae_v5.py --dataset=cifar10 --latent-dim=64 --c-z=4 \\
      --n-epochs=300 --adversarial-start-epoch=51 \\
      --seed=2025 --compute-recon-fid --job-id=vae_v5_cifar_${SLURM_JOB_ID}

References:
  - VAE_BOTTLENECK_DIAGNOSIS.md for detailed analysis of the v3/v4 bottleneck
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
    p = argparse.ArgumentParser(description="VAE v5 -- Bottleneck Fix")

    p.add_argument("--dataset", type=str, default="cifar10",
                   choices=["cifar10", "mnist", "fashion"],
                   help="Dataset to train on")
    p.add_argument("--latent-dim", type=int, default=64,
                   help="VAE latent dimension")
    p.add_argument("--c-z", type=int, default=4,
                   help="Channel count after 1x1 conv at 4x4 "
                        "(pre-flatten = c_z * 16)")

    # Loss weights
    p.add_argument("--beta", type=float, default=0.001,
                   help="KL weight")
    p.add_argument("--beta-warmup-epochs", type=int, default=10,
                   help="Linear ramp from 0 to beta")
    p.add_argument("--lambda-lpips", type=float, default=1.0,
                   help="LPIPS perceptual loss weight")
    p.add_argument("--lambda-adv", type=float, default=0.1,
                   help="Adversarial loss weight (generator)")
    p.add_argument("--adaptive-adv-weight", action="store_true", default=True,
                   help="VQGAN-style adaptive adversarial weighting (default: on)")
    p.add_argument("--no-adaptive-adv-weight", action="store_false",
                   dest="adaptive_adv_weight",
                   help="Disable adaptive adversarial weighting")
    p.add_argument("--free-bits", type=float, default=0.25,
                   help="Minimum KL per latent dimension in nats")
    p.add_argument("--r1-gamma", type=float, default=10.0,
                   help="R1 gradient penalty weight (0 to disable)")
    p.add_argument("--r1-every", type=int, default=16,
                   help="Apply R1 penalty every N batches (lazy reg)")
    p.add_argument("--lpips-every", type=int, default=1,
                   help="Compute LPIPS loss every N batches (1=every batch)")

    # Adversarial training
    p.add_argument("--adversarial-start-epoch", type=int, default=51,
                   help="Epoch to start adversarial training (1-indexed)")
    p.add_argument("--disc-warmup-epochs", type=int, default=5,
                   help="Train D alone before adding adv loss to G")
    p.add_argument("--disc-ramp-epochs", type=int, default=20,
                   help="Linearly ramp adversarial weight over this many epochs")

    # Training
    p.add_argument("--lr", type=float, default=1e-4,
                   help="Generator (VAE) learning rate")
    p.add_argument("--lr-disc", type=float, default=2e-4,
                   help="Discriminator learning rate")
    p.add_argument("--n-epochs", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--ema-decay", type=float, default=0.999,
                   help="EMA decay rate for VAE weights")

    # Data
    p.add_argument("--n-train", type=int, default=50000)
    p.add_argument("--n-valtest", type=int, default=10000)
    p.add_argument("--img-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=2025)

    # I/O
    p.add_argument("--job-id", type=str, default="vae_v5_001")
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
# 3. Data Loaders (images scaled to [-1, 1] for Tanh output)
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
# 4. Model Architecture -- ResConvVAE_v5 (bottleneck fix)
# ---------------------------------------------------------------------------
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
    """Single-head self-attention with GroupNorm + residual."""

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
        out = torch.bmm(v, attn.transpose(1, 2))
        out = out.view(B, C, H, W)
        return x + self.proj(out)


class ResConvVAE_v5(nn.Module):
    """VAE v5: Stops at 4x4, uses 1x1 conv for channel reduction.

    Encoder: 32x32 -> 16x16 -> 8x8 -> 4x4 -> Conv1x1(256, c_z) -> flatten -> latent
    Decoder: latent -> reshape(c_z, 4, 4) -> Conv1x1(c_z, 256) -> 8x8 -> 16x16 -> 32x32

    With c_z=4 and latent_dim=64: pre-flatten = 4*4*4 = 64, so fc_mu is Linear(64,64)
    — essentially a learned rotation, NOT a destructive 16:1 compression like v3/v4.
    """

    def __init__(self, latent_dim=64, in_channels=3, c_z=4):
        super().__init__()
        self.latent_dim = latent_dim
        self.c_z = c_z

        # Encoder: 32x32 -> 16x16 -> 8x8 -> 4x4 (3 downsamples, NOT 4)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, 1, 1, bias=False),        # 32x32
            ResBlock_v3(64), ResBlock_v3(64),                        # 32x32
            nn.Conv2d(64, 64, 3, 2, 1, bias=False),                 # 16x16
            ResBlock_v3(64, 128), ResBlock_v3(128),                  # 16x16
            nn.Conv2d(128, 128, 3, 2, 1, bias=False),               # 8x8
            ResBlock_v3(128, 256), ResBlock_v3(256),                 # 8x8
            SelfAttention(256),                                      # 8x8
            nn.Conv2d(256, 256, 3, 2, 1, bias=False),               # 4x4
            ResBlock_v3(256), ResBlock_v3(256),                      # 4x4
            SelfAttention(256),                                      # 4x4
            nn.GroupNorm(32, 256), nn.SiLU(inplace=True),
            nn.Conv2d(256, c_z, 1, bias=False),                     # 4x4, c_z channels
        )

        flat_dim = c_z * 4 * 4  # e.g., 4 * 16 = 64
        self.fc_mu = nn.Linear(flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(flat_dim, latent_dim)

        # Decoder: latent -> (c_z, 4, 4) -> (256, 4x4) -> 8x8 -> 16x16 -> 32x32
        self.fc_dec = nn.Linear(latent_dim, flat_dim)
        self.decoder = nn.Sequential(
            nn.Conv2d(c_z, 256, 1, bias=False),                     # 4x4, expand channels
            ResBlock_v3(256), ResBlock_v3(256),
            SelfAttention(256),                                      # 4x4
            nn.ConvTranspose2d(256, 256, 4, 2, 1, bias=False),      # 8x8
            ResBlock_v3(256, 128), ResBlock_v3(128),
            SelfAttention(128),                                      # 8x8
            nn.ConvTranspose2d(128, 128, 4, 2, 1, bias=False),      # 16x16
            ResBlock_v3(128, 64), ResBlock_v3(64),                   # 16x16
            nn.ConvTranspose2d(64, 64, 4, 2, 1, bias=False),        # 32x32
            ResBlock_v3(64), ResBlock_v3(64),                        # 32x32
            nn.GroupNorm(32, 64), nn.SiLU(inplace=True),
            nn.Conv2d(64, in_channels, 3, 1, 1),
            nn.Tanh(),                                               # output [-1, 1]
        )

    def encode(self, x):
        h = self.encoder(x).view(x.size(0), -1)  # (B, c_z*4*4)
        mu = self.fc_mu(h)
        logvar = torch.clamp(self.fc_logvar(h), -20, 2)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def decode(self, z):
        h = F.silu(self.fc_dec(z)).view(-1, self.c_z, 4, 4)
        return self.decoder(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


# ---------------------------------------------------------------------------
# 5. PatchGAN Discriminator (v4 style: spectral norm, logistic loss)
# ---------------------------------------------------------------------------
class PatchGANDiscriminator_v2(nn.Module):
    """Enlarged PatchGAN with spectral normalization (~1.85M params)."""

    def __init__(self, in_channels=3, ndf=64):
        super().__init__()
        from torch.nn.utils import spectral_norm
        self.net = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels, ndf, 4, 2, 1)),      # 16x16
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(ndf, ndf * 2, 4, 2, 1)),          # 8x8
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1)),      # 4x4
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(ndf * 4, ndf * 8, 4, 1, 1)),      # 3x3
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(ndf * 8, 1, 4, 1, 1)),            # 2x2
        )

    def forward(self, x):
        return self.net(x)


def disc_loss_logistic(real_pred, fake_pred):
    """Non-saturating logistic loss for discriminator (no dead zone)."""
    return F.softplus(-real_pred).mean() + F.softplus(fake_pred).mean()


def gen_loss_logistic(fake_pred):
    """Non-saturating logistic loss for generator."""
    return F.softplus(-fake_pred).mean()


def r1_gradient_penalty(disc, real_images):
    """R1 gradient penalty (Mescheder et al., 2018 / StyleGAN2)."""
    real_images = real_images.detach().requires_grad_(True)
    real_pred = disc(real_images)
    grad_real = torch.autograd.grad(
        outputs=real_pred.sum(), inputs=real_images,
        create_graph=True, retain_graph=True)[0]
    return grad_real.pow(2).reshape(grad_real.size(0), -1).sum(1).mean()


def compute_adaptive_adv_weight(recon_loss, adv_loss, last_layer_weight):
    """VQGAN-style adaptive adversarial weight (Esser et al., 2021)."""
    recon_grads = torch.autograd.grad(
        recon_loss, last_layer_weight, retain_graph=True)[0]
    adv_grads = torch.autograd.grad(
        adv_loss, last_layer_weight, retain_graph=True)[0]
    adv_weight = torch.norm(recon_grads) / (torch.norm(adv_grads) + 1e-6)
    return adv_weight.clamp(0.0, 1e4).detach()


# ---------------------------------------------------------------------------
# 6. EMA Helper
# ---------------------------------------------------------------------------
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
        """Load EMA weights into model (for evaluation)."""
        model.load_state_dict(self.shadow)


# ---------------------------------------------------------------------------
# 7. KL Computation
# ---------------------------------------------------------------------------
def compute_kl(mu, logvar, free_bits=0.25):
    """KL with free bits: sum over latent dims, mean over batch."""
    kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    kl_per_dim_avg = kl_per_dim.mean(dim=0)
    kl_per_dim_clamped = torch.clamp(kl_per_dim_avg, min=free_bits)
    kl = kl_per_dim_clamped.sum()
    return kl, kl_per_dim_avg


# ---------------------------------------------------------------------------
# 8. Visualization
# ---------------------------------------------------------------------------
def save_recon_grid(vae, val_loader, device, save_path, n_images=8):
    """Save side-by-side grid of originals and reconstructions."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    vae.eval()
    with torch.no_grad():
        for (xb,) in val_loader:
            xb = xb[:n_images].to(device)
            x_hat, _, _ = vae(xb)
            break

    xb_disp = (xb.clamp(-1, 1) + 1) / 2
    xh_disp = (x_hat.clamp(-1, 1) + 1) / 2

    fig, axes = plt.subplots(2, n_images, figsize=(2 * n_images, 4))
    for i in range(n_images):
        orig = xb_disp[i].cpu().permute(1, 2, 0).numpy()
        recon = xh_disp[i].cpu().permute(1, 2, 0).numpy()
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
# 9. Reconstruction FID
# ---------------------------------------------------------------------------
def compute_recon_fid(vae, test_loader, device):
    try:
        from torchmetrics.image.fid import FrechetInceptionDistance
    except ImportError:
        print("  [WARN] torchmetrics FID not available, skipping")
        return float("nan")

    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    vae.eval()
    with torch.no_grad():
        for (xb,) in tqdm(test_loader, desc="Recon FID", leave=False):
            xb = xb.to(device)
            x_hat, _, _ = vae(xb)
            xb_01 = (xb.clamp(-1, 1) + 1) / 2
            xh_01 = (x_hat.clamp(-1, 1) + 1) / 2
            fid.update(xb_01, real=True)
            fid.update(xh_01, real=False)
    return fid.compute().item()


# ---------------------------------------------------------------------------
# 10. Main Training Loop
# ---------------------------------------------------------------------------
def train_vae(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    set_all_seeds(args.seed)

    # Data (scaled to [-1, 1])
    loader_fn = {
        "cifar10": load_cifar_2d,
        "mnist": load_mnist_2d,
        "fashion": load_fashion_2d,
    }[args.dataset]
    train_loader, val_loader, test_loader = loader_fn(
        args.seed, args.n_train, args.n_valtest, args.batch_size, args.img_size)
    print(f"Data: {args.dataset} | train={args.n_train} val+test={args.n_valtest}")
    print(f"Data range: [-1, 1] (Tanh output)")

    # VAE v5
    vae = ResConvVAE_v5(latent_dim=args.latent_dim, c_z=args.c_z).to(device)
    total_p = sum(p.numel() for p in vae.parameters() if p.requires_grad)
    flat_dim = args.c_z * 4 * 4
    print(f"VAE v5 | params={total_p:,} | latent_dim={args.latent_dim} | "
          f"c_z={args.c_z} | pre-flatten={flat_dim} | "
          f"FC ratio={flat_dim}:{args.latent_dim} = "
          f"{flat_dim / args.latent_dim:.1f}:1")

    # Discriminator (v4 style: spectral norm + logistic loss)
    disc = PatchGANDiscriminator_v2().to(device)
    disc_p = sum(p.numel() for p in disc.parameters() if p.requires_grad)
    print(f"Discriminator v2 | params={disc_p:,} (spectral norm, logistic loss)")

    # Optimizers
    opt_g = torch.optim.Adam(vae.parameters(), lr=args.lr, betas=(0.5, 0.9))
    opt_d = torch.optim.Adam(disc.parameters(), lr=args.lr_disc,
                             betas=(0.0, 0.99))
    sched_g = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt_g, T_max=args.n_epochs, eta_min=1e-6)
    # Constant LR for discriminator (no cosine decay)

    # LPIPS perceptual loss
    lpips_fn = None
    if args.lambda_lpips > 0:
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
        lpips_fn = LearnedPerceptualImagePatchSimilarity(
            net_type="vgg", normalize=False).to(device)
        for p in lpips_fn.parameters():
            p.requires_grad = False
        print(f"LPIPS loss enabled (lambda={args.lambda_lpips})")

    # EMA
    ema = EMAModel(vae, decay=args.ema_decay)

    # Metrics
    from torchmetrics.image import (StructuralSimilarityIndexMeasure,
                                    PeakSignalNoiseRatio)
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    psnr_metric = PeakSignalNoiseRatio(data_range=2.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=2.0).to(device)
    lpips_eval = LearnedPerceptualImagePatchSimilarity(
        net_type="squeeze", normalize=False).to(device)

    # Directories
    ckpt_dir = os.path.join(args.base_path, "checkpoints")
    results_dir = os.path.join(args.base_path, "results")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    ckpt_path = os.path.join(ckpt_dir,
                             f"ckpt_vae_v5_{args.dataset}_{args.job_id}.pt")
    weights_path = os.path.join(ckpt_dir,
                                f"weights_vae_v5_{args.dataset}_{args.job_id}.pt")
    ema_weights_path = os.path.join(
        ckpt_dir, f"weights_vae_v5_ema_{args.dataset}_{args.job_id}.pt")
    csv_path = os.path.join(results_dir,
                            f"log_vae_v5_{args.dataset}_{args.job_id}.csv")
    metrics_path = os.path.join(results_dir,
                                f"metrics_vae_v5_{args.dataset}_{args.job_id}.json")

    fields = [
        "epoch", "train_loss_g", "train_recon", "train_kl", "train_lpips",
        "train_adv_g", "train_adv_weight", "train_loss_d", "train_r1",
        "d_real_mean", "d_fake_mean", "d_real_std", "d_fake_std", "disc_factor",
        "val_loss", "val_recon", "val_kl", "val_lpips",
        "val_psnr", "val_ssim", "val_lpips_eval",
        "kl_mean_per_dim", "kl_min_per_dim", "kl_max_per_dim",
        "active_dims", "mu_norm",
        "beta_eff", "lr_g", "lr_d", "adv_active", "time_s",
    ]

    # Get decoder's last conv layer weight for adaptive adversarial weight
    dec_last_layer = vae.decoder[-2].weight  # Conv2d(64, 3, 3x3) before Tanh

    print(f"\n--- VAE v5 Config ---")
    print(f"  BOTTLENECK FIX: encoder stops at 4x4, "
          f"Conv1x1(256->{args.c_z}) -> flatten({flat_dim}) -> "
          f"Linear({flat_dim},{args.latent_dim})")
    print(f"  beta={args.beta}, free_bits={args.free_bits}")
    print(f"  lambda_lpips={args.lambda_lpips}")
    print(f"  lambda_adv={args.lambda_adv} (from epoch {args.adversarial_start_epoch})")
    print(f"  adaptive_adv_weight={args.adaptive_adv_weight}")
    print(f"  D warmup={args.disc_warmup_epochs} epochs, "
          f"ramp={args.disc_ramp_epochs} epochs")
    print(f"  D: spectral norm, logistic loss, "
          f"lr={args.lr_disc}, betas=(0.0, 0.99), constant LR")
    print(f"  R1 penalty: gamma={args.r1_gamma}, every {args.r1_every} batches")
    print(f"  G: lr={args.lr}, betas=(0.5, 0.9), cosine LR")
    print(f"  EMA decay={args.ema_decay}")
    print(f"---------------------\n")

    # Resume
    start_epoch = 0
    best_psnr = float("-inf")
    best_state = None

    if args.resume and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, weights_only=False, map_location=device)
        vae.load_state_dict(ckpt["model"])
        disc.load_state_dict(ckpt["disc"])
        opt_g.load_state_dict(ckpt["opt_g"])
        opt_d.load_state_dict(ckpt["opt_d"])
        if "sched_g" in ckpt:
            sched_g.load_state_dict(ckpt["sched_g"])
        if "ema" in ckpt:
            ema.load_state_dict(ckpt["ema"])
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
        adv_active = (epoch + 1) >= args.adversarial_start_epoch
        lr_g = opt_g.param_groups[0]["lr"]
        lr_d = opt_d.param_groups[0]["lr"]

        # Discriminator ramp factor
        disc_factor = 0.0
        if adv_active:
            adv_epoch = (epoch + 1) - args.adversarial_start_epoch
            if adv_epoch < args.disc_warmup_epochs:
                disc_factor = 0.0  # D trains, but G doesn't see adv loss
            elif adv_epoch < args.disc_warmup_epochs + args.disc_ramp_epochs:
                ramp_ep = adv_epoch - args.disc_warmup_epochs
                disc_factor = ramp_ep / args.disc_ramp_epochs
            else:
                disc_factor = 1.0

        # -- Train --
        vae.train()
        disc.train()
        tr_lg, tr_rec, tr_kl, tr_lp = 0., 0., 0., 0.
        tr_adv, tr_adv_w, tr_ld, tr_r1 = 0., 0., 0., 0.
        tr_d_real_m, tr_d_fake_m = 0., 0.
        tr_d_real_s, tr_d_fake_s = 0., 0.
        tr_n = 0
        batch_idx = 0

        for (xb,) in tqdm(train_loader,
                          desc=f"VAE-v5 Ep {epoch+1}/{args.n_epochs}",
                          leave=False):
            xb = xb.to(device)
            bs = xb.size(0)
            batch_idx += 1

            # --- Generator step ---
            x_hat, mu, logvar = vae(xb)

            recon = F.l1_loss(x_hat, xb, reduction="mean")
            kl, _ = compute_kl(mu, logvar, free_bits=args.free_bits)
            nll_loss = recon + beta_eff * kl

            lpips_val = 0.0
            if lpips_fn is not None and batch_idx % args.lpips_every == 0:
                lp = lpips_fn(x_hat.clamp(-1, 1), xb)
                nll_loss = nll_loss + args.lambda_lpips * lp
                lpips_val = lp.item()

            # Adversarial (generator)
            adv_g_val = 0.0
            adv_w_val = args.lambda_adv
            if adv_active and disc_factor > 0:
                fake_pred = disc(x_hat)
                adv_g = gen_loss_logistic(fake_pred)

                if args.adaptive_adv_weight:
                    adv_w = compute_adaptive_adv_weight(
                        nll_loss, adv_g, dec_last_layer)
                    adv_w_val = adv_w.item()
                    loss_g = nll_loss + disc_factor * adv_w * adv_g
                else:
                    loss_g = nll_loss + disc_factor * args.lambda_adv * adv_g
                adv_g_val = adv_g.item()
            else:
                loss_g = nll_loss

            opt_g.zero_grad()
            loss_g.backward()
            opt_g.step()

            # --- Discriminator step ---
            loss_d_val = 0.0
            r1_val = 0.0
            d_real_m, d_fake_m, d_real_s, d_fake_s = 0., 0., 0., 0.
            if adv_active:
                x_hat_det = x_hat.detach()
                real_pred = disc(xb)
                fake_pred = disc(x_hat_det)
                loss_d = disc_loss_logistic(real_pred, fake_pred)

                # Diagnostics
                d_real_m = real_pred.mean().item()
                d_fake_m = fake_pred.mean().item()
                d_real_s = real_pred.std().item()
                d_fake_s = fake_pred.std().item()

                if args.r1_gamma > 0 and batch_idx % args.r1_every == 0:
                    r1 = r1_gradient_penalty(disc, xb)
                    loss_d = loss_d + (args.r1_gamma / 2) * r1 * args.r1_every
                    r1_val = r1.item()

                opt_d.zero_grad()
                loss_d.backward()
                opt_d.step()
                loss_d_val = loss_d.item()

            # EMA update
            ema.update(vae)

            tr_lg += loss_g.item() * bs
            tr_rec += recon.item() * bs
            tr_kl += kl.item() * bs
            tr_lp += lpips_val * bs
            tr_adv += adv_g_val * bs
            tr_adv_w += adv_w_val * bs
            tr_ld += loss_d_val * bs
            tr_r1 += r1_val * bs
            tr_d_real_m += d_real_m * bs
            tr_d_fake_m += d_fake_m * bs
            tr_d_real_s += d_real_s * bs
            tr_d_fake_s += d_fake_s * bs
            tr_n += bs

        tr_lg /= tr_n
        tr_rec /= tr_n
        tr_kl /= tr_n
        tr_lp /= tr_n
        tr_adv /= tr_n
        tr_adv_w /= tr_n
        tr_ld /= tr_n
        tr_r1 /= tr_n
        tr_d_real_m /= tr_n
        tr_d_fake_m /= tr_n
        tr_d_real_s /= tr_n
        tr_d_fake_s /= tr_n

        # -- Val (using EMA weights) --
        orig_state = copy.deepcopy(vae.state_dict())
        ema.apply(vae)
        vae.eval()

        vl_loss, vl_rec, vl_kl, vl_lp, vl_n = 0., 0., 0., 0., 0
        all_kl_per_dim = []
        all_mu_norms = []
        psnr_metric.reset()
        ssim_metric.reset()
        lpips_eval.reset()

        with torch.no_grad():
            for (xb,) in val_loader:
                xb = xb.to(device)
                x_hat, mu, logvar = vae(xb)

                recon = F.l1_loss(x_hat, xb, reduction="mean")
                kl, kl_per_dim_avg = compute_kl(mu, logvar,
                                                free_bits=args.free_bits)
                loss = recon + beta_eff * kl

                lpips_val = 0.0
                if lpips_fn is not None:
                    lp = lpips_fn(x_hat.clamp(-1, 1), xb)
                    loss = loss + args.lambda_lpips * lp
                    lpips_val = lp.item()

                all_kl_per_dim.append(kl_per_dim_avg.cpu())
                all_mu_norms.append(mu.norm(dim=1).mean().item())

                x_hat_c = x_hat.clamp(-1, 1)
                psnr_metric.update(x_hat_c, xb)
                ssim_metric.update(x_hat_c, xb)
                lpips_eval.update(x_hat_c, xb)

                bs = xb.size(0)
                vl_loss += loss.item() * bs
                vl_rec += recon.item() * bs
                vl_kl += kl.item() * bs
                vl_lp += lpips_val * bs
                vl_n += bs

        vl_loss /= vl_n
        vl_rec /= vl_n
        vl_kl /= vl_n
        vl_lp /= vl_n

        vl_psnr = psnr_metric.compute().item()
        vl_ssim = ssim_metric.compute().item()
        vl_lpips_e = lpips_eval.compute().item()

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
            train_loss_g=f"{tr_lg:.6f}",
            train_recon=f"{tr_rec:.6f}",
            train_kl=f"{tr_kl:.6f}",
            train_lpips=f"{tr_lp:.6f}",
            train_adv_g=f"{tr_adv:.6f}",
            train_adv_weight=f"{tr_adv_w:.6f}",
            train_loss_d=f"{tr_ld:.6f}",
            train_r1=f"{tr_r1:.6f}",
            d_real_mean=f"{tr_d_real_m:.6f}",
            d_fake_mean=f"{tr_d_fake_m:.6f}",
            d_real_std=f"{tr_d_real_s:.6f}",
            d_fake_std=f"{tr_d_fake_s:.6f}",
            disc_factor=f"{disc_factor:.4f}",
            val_loss=f"{vl_loss:.6f}",
            val_recon=f"{vl_rec:.6f}",
            val_kl=f"{vl_kl:.6f}",
            val_lpips=f"{vl_lp:.6f}",
            val_psnr=f"{vl_psnr:.4f}",
            val_ssim=f"{vl_ssim:.4f}",
            val_lpips_eval=f"{vl_lpips_e:.4f}",
            kl_mean_per_dim=f"{kl_mean_d:.6f}",
            kl_min_per_dim=f"{kl_min_d:.6f}",
            kl_max_per_dim=f"{kl_max_d:.6f}",
            active_dims=int(active_dims),
            mu_norm=f"{mu_norm:.4f}",
            beta_eff=f"{beta_eff:.6f}",
            lr_g=f"{lr_g:.2e}",
            lr_d=f"{lr_d:.2e}",
            adv_active=int(adv_active),
            time_s=f"{dt:.1f}",
        )
        with open(csv_path, "a", newline="") as f:
            csv.DictWriter(f, fields).writerow(row)

        adv_str = ""
        if adv_active:
            adv_str = (f" adv_g={tr_adv:.4f} D={tr_ld:.4f} w={tr_adv_w:.3f}"
                       f" df={disc_factor:.2f}"
                       f" D(r)={tr_d_real_m:.3f} D(f)={tr_d_fake_m:.3f}")
        print(
            f"  Ep {epoch+1:3d} | "
            f"Train {tr_lg:.4f} (L1={tr_rec:.4f} kl={tr_kl:.4f} "
            f"lpips={tr_lp:.4f}{adv_str}) | "
            f"Val {vl_loss:.4f} | "
            f"PSNR {vl_psnr:.2f} SSIM {vl_ssim:.4f} LPIPS {vl_lpips_e:.4f} | "
            f"active={int(active_dims)}/{args.latent_dim} | "
            f"{dt:.1f}s"
        )

        # Best model by PSNR (using EMA weights)
        if vl_psnr > best_psnr:
            best_psnr = vl_psnr
            best_state = copy.deepcopy(vae.state_dict())
            print(f"    >> New best PSNR: {best_psnr:.2f} dB (EMA)")

        # Restore non-EMA weights for continued training
        vae.load_state_dict(orig_state)

        # Checkpoint
        torch.save(dict(
            epoch=epoch,
            model=vae.state_dict(),
            disc=disc.state_dict(),
            opt_g=opt_g.state_dict(),
            opt_d=opt_d.state_dict(),
            sched_g=sched_g.state_dict(),
            ema=ema.state_dict(),
            best_psnr=best_psnr,
            c_z=args.c_z,
            latent_dim=args.latent_dim,
        ), ckpt_path)

        sched_g.step()

        # Save reconstruction grid (using EMA)
        if (args.save_grid_every > 0
                and (epoch + 1) % args.save_grid_every == 0):
            ema.apply(vae)
            grid_path = os.path.join(
                results_dir,
                f"recon_grid_vae_v5_epoch{epoch+1}_{args.job_id}.png")
            save_recon_grid(vae, val_loader, device, grid_path)
            vae.load_state_dict(orig_state)

    # -- Done: save best EMA model --
    if best_state is not None:
        vae.load_state_dict(best_state)
        print(f"\nLoaded best EMA model (PSNR={best_psnr:.2f} dB)")

    torch.save(vae.state_dict(), ema_weights_path)
    print(f"Best EMA VAE weights saved to {ema_weights_path}")

    torch.save(vae.state_dict(), weights_path)
    print(f"VAE weights saved to {weights_path}")

    # Final reconstruction grid
    final_grid_path = os.path.join(
        results_dir, f"recon_grid_vae_v5_final_{args.job_id}.png")
    save_recon_grid(vae, val_loader, device, final_grid_path)

    # Reconstruction FID
    recon_fid = float("nan")
    if args.compute_recon_fid:
        print("\nComputing reconstruction FID on test set...")
        recon_fid = compute_recon_fid(vae, test_loader, device)
        print(f"  Reconstruction FID: {recon_fid:.2f}")

    # Final test-set metrics
    psnr_metric.reset()
    ssim_metric.reset()
    lpips_eval.reset()
    vae.eval()
    with torch.no_grad():
        for (xb,) in test_loader:
            xb = xb.to(device)
            x_hat, _, _ = vae(xb)
            x_hat_c = x_hat.clamp(-1, 1)
            psnr_metric.update(x_hat_c, xb)
            ssim_metric.update(x_hat_c, xb)
            lpips_eval.update(x_hat_c, xb)

    test_psnr = psnr_metric.compute().item()
    test_ssim = ssim_metric.compute().item()
    test_lpips = lpips_eval.compute().item()

    print(f"\n=== Final Test Metrics ===")
    print(f"  PSNR:  {test_psnr:.2f} dB")
    print(f"  SSIM:  {test_ssim:.4f}")
    print(f"  LPIPS: {test_lpips:.4f}")
    if not math.isnan(recon_fid):
        print(f"  Recon FID: {recon_fid:.2f}")

    summary = {
        "job_id": args.job_id,
        "dataset": args.dataset,
        "vae_arch": "resconv_v5",
        "disc_arch": "patchgan_v2_specnorm",
        "latent_dim": args.latent_dim,
        "c_z": args.c_z,
        "pre_flatten_dim": flat_dim,
        "fc_compression_ratio": f"{flat_dim}:{args.latent_dim}",
        "beta": args.beta,
        "lambda_lpips": args.lambda_lpips,
        "lambda_adv": args.lambda_adv,
        "adaptive_adv_weight": args.adaptive_adv_weight,
        "disc_warmup_epochs": args.disc_warmup_epochs,
        "disc_ramp_epochs": args.disc_ramp_epochs,
        "adversarial_start_epoch": args.adversarial_start_epoch,
        "loss_type": "logistic (softplus)",
        "free_bits": args.free_bits,
        "n_epochs": args.n_epochs,
        "seed": args.seed,
        "n_params_vae": total_p,
        "n_params_disc": disc_p,
        "best_val_psnr": best_psnr,
        "test_psnr": test_psnr,
        "test_ssim": test_ssim,
        "test_lpips": test_lpips,
        "recon_fid": recon_fid,
        "weights_path": ema_weights_path,
        "data_range": "[-1, 1]",
    }
    with open(metrics_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Metrics saved to {metrics_path}")

    return vae


if __name__ == "__main__":
    args = get_args()
    train_vae(args)
