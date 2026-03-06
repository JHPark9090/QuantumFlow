"""
VAE v4 — Discriminator & Adversarial Training Fixes
=====================================================

Same VAE architecture as v3 (ResConvVAE_v3, ~10M params). Fixes 6 root causes
of discriminator collapse (D_loss stuck at 2.0) observed in v3 training.

Fixes over v3:
  1. Enlarged PatchGAN discriminator with spectral norm (64->128->256->512->1, ~2.8M)
  2. Non-saturating logistic loss (softplus) instead of hinge (ReLU clips gradients)
  3. Adaptive adversarial weight ON by default (--no-adaptive-adv-weight to disable)
  4. Discriminator warmup + gradual ramp (D trains alone before G sees adv loss)
  5. D optimizer: betas=(0.0, 0.99), constant LR, lr=2e-4
  6. Diagnostic logging: D(real/fake) mean/std per epoch

Unchanged from v3:
  ResBlock_v3, SelfAttention, ResConvVAE_v3, EMAModel, compute_kl,
  data loaders, save_recon_grid, compute_recon_fid, r1_gradient_penalty,
  compute_adaptive_adv_weight, checkpoint structure.

Usage:
  # Quick test
  python models/train_vae_v4.py --dataset=cifar10 --n-epochs=5 \\
      --n-train=1000 --n-valtest=200 --adversarial-start-epoch=2 \\
      --disc-warmup-epochs=1 --disc-ramp-epochs=1 --job-id=test_v4

  # Full training (300 epochs, adversarial from epoch 51)
  python models/train_vae_v4.py --dataset=cifar10 --latent-dim=64 \\
      --n-epochs=300 --adversarial-start-epoch=51 \\
      --disc-warmup-epochs=5 --disc-ramp-epochs=20 \\
      --seed=2025 --compute-recon-fid --job-id=vae_v4_cifar_${SLURM_JOB_ID}
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
    p = argparse.ArgumentParser(description="VAE v4 — Discriminator & Adversarial Fixes")

    p.add_argument("--dataset", type=str, default="cifar10",
                   choices=["cifar10", "mnist", "fashion"],
                   help="Dataset to train on")
    p.add_argument("--latent-dim", type=int, default=64,
                   help="VAE latent dimension")

    # Loss weights
    p.add_argument("--beta", type=float, default=0.001,
                   help="KL weight")
    p.add_argument("--beta-warmup-epochs", type=int, default=10,
                   help="Linear ramp from 0 to beta")
    p.add_argument("--lambda-lpips", type=float, default=1.0,
                   help="LPIPS perceptual loss weight")
    p.add_argument("--lambda-adv", type=float, default=0.1,
                   help="Adversarial loss weight (generator)")
    # Fix 3: adaptive weight ON by default
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
    # Fix 4: discriminator warmup + ramp
    p.add_argument("--disc-warmup-epochs", type=int, default=5,
                   help="Train D alone for N epochs before adding adv loss to G")
    p.add_argument("--disc-ramp-epochs", type=int, default=20,
                   help="Linearly ramp disc_factor from 0->1 over N epochs after warmup")

    # Training
    p.add_argument("--lr", type=float, default=1e-4,
                   help="Generator (VAE) learning rate")
    # Fix 5: lower default D learning rate
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
    p.add_argument("--job-id", type=str, default="vae_v4_001")
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
    # Scale to [-1, 1]
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
# 4. Model Architecture — ResConvVAE_v3 (UNCHANGED from v3)
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
        attn = torch.bmm(q.transpose(1, 2), k) * self.scale  # (B, HW, HW)
        attn = F.softmax(attn, dim=-1)
        out = torch.bmm(v, attn.transpose(1, 2))  # (B, C, HW)
        out = out.view(B, C, H, W)
        return x + self.proj(out)


class ResConvVAE_v3(nn.Module):
    """SOTA VAE: GroupNorm+SiLU, 64->128->256->256 channels, self-attention, Tanh output.

    ~10M params. Encoder: 32x32 -> 2x2 -> flat -> latent.
    Decoder: latent -> 2x2 -> 32x32. Output range [-1, 1].
    """

    def __init__(self, latent_dim=64, in_channels=3):
        super().__init__()
        self.latent_dim = latent_dim

        # Encoder: 32x32 -> 16x16 -> 8x8 -> 4x4 -> 2x2
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
            nn.Conv2d(256, 256, 3, 2, 1, bias=False),               # 2x2
        )
        flat_dim = 256 * 2 * 2  # 1024
        self.fc_mu = nn.Linear(flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(flat_dim, latent_dim)

        # Decoder: 2x2 -> 4x4 -> 8x8 -> 16x16 -> 32x32
        self.fc_dec = nn.Linear(latent_dim, flat_dim)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 256, 4, 2, 1, bias=False),      # 4x4
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
        h = self.encoder(x).view(x.size(0), -1)
        mu = self.fc_mu(h)
        logvar = torch.clamp(self.fc_logvar(h), -20, 2)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def decode(self, z):
        h = F.silu(self.fc_dec(z)).view(-1, 256, 2, 2)
        return self.decoder(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


# ---------------------------------------------------------------------------
# 5. PatchGAN Discriminator v2 (Fix 1: enlarged + spectral norm)
# ---------------------------------------------------------------------------
class PatchGANDiscriminator_v2(nn.Module):
    """Enlarged PatchGAN with spectral normalization (~2.8M params).

    Channels: 64->128->256->512->1 (v3 was 64->128->256->1).
    Spectral norm on all Conv2d layers replaces GroupNorm for weight regularization.
    Input: (B, 3, 32, 32) -> output: (B, 1, 2, 2) patch predictions.
    """

    def __init__(self, in_channels=3):
        super().__init__()
        sn = torch.nn.utils.spectral_norm
        self.net = nn.Sequential(
            sn(nn.Conv2d(in_channels, 64, 4, 2, 1)),                # 16x16
            nn.LeakyReLU(0.2, inplace=True),
            sn(nn.Conv2d(64, 128, 4, 2, 1)),                        # 8x8
            nn.LeakyReLU(0.2, inplace=True),
            sn(nn.Conv2d(128, 256, 4, 2, 1)),                       # 4x4
            nn.LeakyReLU(0.2, inplace=True),
            sn(nn.Conv2d(256, 512, 3, 1, 1)),                       # 4x4
            nn.LeakyReLU(0.2, inplace=True),
            sn(nn.Conv2d(512, 1, 4, 2, 1)),                         # 2x2
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# 5b. Loss functions (Fix 2: logistic instead of hinge)
# ---------------------------------------------------------------------------
def disc_loss_logistic(real_pred, fake_pred):
    """Non-saturating logistic discriminator loss (softplus, never clips gradients)."""
    return F.softplus(-real_pred).mean() + F.softplus(fake_pred).mean()


def gen_loss_logistic(fake_pred):
    """Non-saturating logistic generator loss."""
    return F.softplus(-fake_pred).mean()


def r1_gradient_penalty(disc, real_images):
    """R1 gradient penalty (Mescheder et al., 2018 / StyleGAN2).

    Penalizes ||grad D(x_real)||^2 to prevent discriminator from
    becoming overconfident. Returns scalar penalty (before gamma scaling).
    """
    real_images = real_images.detach().requires_grad_(True)
    real_pred = disc(real_images)
    grad_real = torch.autograd.grad(
        outputs=real_pred.sum(), inputs=real_images,
        create_graph=True, retain_graph=True)[0]
    return grad_real.pow(2).reshape(grad_real.size(0), -1).sum(1).mean()


def compute_adaptive_adv_weight(recon_loss, adv_loss, last_layer_weight):
    """VQGAN-style adaptive adversarial weight (Esser et al., 2021).

    Balances gradient magnitudes: lambda = ||grad_recon|| / ||grad_adv||
    so neither loss dominates.
    """
    recon_grads = torch.autograd.grad(
        recon_loss, last_layer_weight, retain_graph=True)[0]
    adv_grads = torch.autograd.grad(
        adv_loss, last_layer_weight, retain_graph=True)[0]
    adv_weight = torch.norm(recon_grads) / (torch.norm(adv_grads) + 1e-6)
    return adv_weight.clamp(0.0, 1e4).detach()


# ---------------------------------------------------------------------------
# 6. EMA Helper (UNCHANGED from v3)
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
# 7. KL Computation (UNCHANGED from v3)
# ---------------------------------------------------------------------------
def compute_kl(mu, logvar, free_bits=0.25):
    """KL with free bits: sum over latent dims, mean over batch."""
    kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    kl_per_dim_avg = kl_per_dim.mean(dim=0)
    kl_per_dim_clamped = torch.clamp(kl_per_dim_avg, min=free_bits)
    kl = kl_per_dim_clamped.sum()
    return kl, kl_per_dim_avg


# ---------------------------------------------------------------------------
# 8. Visualization (UNCHANGED from v3)
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

    # Convert from [-1,1] to [0,1] for display
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
# 9. Reconstruction FID (UNCHANGED from v3)
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
            # FID expects [0,1]
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

    # VAE (unchanged from v3)
    vae = ResConvVAE_v3(latent_dim=args.latent_dim).to(device)
    total_p = sum(p.numel() for p in vae.parameters() if p.requires_grad)
    print(f"VAE v3 arch | params={total_p:,} | latent_dim={args.latent_dim}")

    # Fix 1: Enlarged discriminator with spectral norm
    disc = PatchGANDiscriminator_v2().to(device)
    disc_p = sum(p.numel() for p in disc.parameters() if p.requires_grad)
    print(f"Discriminator v2 (spectral norm) | params={disc_p:,}")
    print(f"  G:D param ratio = {total_p / disc_p:.1f}:1")

    # Fix 5: D optimizer with betas=(0.0, 0.99), G keeps (0.5, 0.9)
    opt_g = torch.optim.Adam(vae.parameters(), lr=args.lr,
                             betas=(0.5, 0.9))
    opt_d = torch.optim.Adam(disc.parameters(), lr=args.lr_disc,
                             betas=(0.0, 0.99))
    sched_g = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt_g, T_max=args.n_epochs, eta_min=1e-6)
    # Fix 5: No cosine annealing for D — constant LR
    # (We still create a dummy identity scheduler for checkpoint compat)

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
    psnr_metric = PeakSignalNoiseRatio(data_range=2.0).to(device)  # [-1,1] range=2
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=2.0).to(device)
    lpips_eval = LearnedPerceptualImagePatchSimilarity(
        net_type="squeeze", normalize=False).to(device)

    # Directories
    ckpt_dir = os.path.join(args.base_path, "checkpoints")
    results_dir = os.path.join(args.base_path, "results")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    ckpt_path = os.path.join(ckpt_dir, f"ckpt_vae_v4_{args.dataset}_{args.job_id}.pt")
    weights_path = os.path.join(ckpt_dir, f"weights_vae_v4_{args.dataset}_{args.job_id}.pt")
    ema_weights_path = os.path.join(ckpt_dir, f"weights_vae_v4_ema_{args.dataset}_{args.job_id}.pt")
    csv_path = os.path.join(results_dir, f"log_vae_v4_{args.dataset}_{args.job_id}.csv")
    metrics_path = os.path.join(results_dir, f"metrics_vae_v4_{args.dataset}_{args.job_id}.json")

    # Fix 6: 4 new diagnostic columns
    fields = [
        "epoch", "train_loss_g", "train_recon", "train_kl", "train_lpips",
        "train_adv_g", "train_adv_weight", "train_loss_d", "train_r1",
        "d_real_mean", "d_fake_mean", "d_real_std", "d_fake_std",
        "disc_factor",
        "val_loss", "val_recon", "val_kl", "val_lpips",
        "val_psnr", "val_ssim", "val_lpips_eval",
        "kl_mean_per_dim", "kl_min_per_dim", "kl_max_per_dim",
        "active_dims", "mu_norm",
        "beta_eff", "lr_g", "lr_d", "adv_active", "time_s",
    ]

    # Get decoder's last conv layer weight for adaptive adversarial weight
    dec_last_layer = vae.decoder[-2].weight  # Conv2d(64, 3, 3x3) before Tanh

    print(f"\n--- VAE v4 Config ---")
    print(f"  beta={args.beta}, free_bits={args.free_bits}")
    print(f"  lambda_lpips={args.lambda_lpips} (every {args.lpips_every} batches)")
    print(f"  lambda_adv={args.lambda_adv} (from epoch {args.adversarial_start_epoch})")
    print(f"  adaptive_adv_weight={args.adaptive_adv_weight}")
    print(f"  disc_warmup={args.disc_warmup_epochs}, disc_ramp={args.disc_ramp_epochs}")
    print(f"  R1 penalty: gamma={args.r1_gamma}, every {args.r1_every} batches")
    print(f"  lr_g={args.lr} betas=(0.5,0.9), lr_d={args.lr_disc} betas=(0.0,0.99)")
    print(f"  D scheduler: constant LR (no cosine annealing)")
    print(f"  Loss: logistic (softplus, non-saturating)")
    print(f"  EMA decay={args.ema_decay}")
    print(f"  Output: Tanh [-1, 1], recon loss: L1")
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
        # Note: no sched_d to restore (constant LR)
        if "ema" in ckpt:
            ema.load_state_dict(ckpt["ema"])
        start_epoch = ckpt["epoch"] + 1
        best_psnr = ckpt.get("best_psnr", float("-inf"))
        print(f"Resumed from epoch {start_epoch}, best PSNR so far: {best_psnr:.2f}")

    if start_epoch == 0:
        with open(csv_path, "w", newline="") as f:
            csv.DictWriter(f, fields).writeheader()
    elif os.path.exists(csv_path):
        # Migrate CSV if header changed (e.g., new columns added)
        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            old_header = next(reader, None)
        if old_header and old_header != fields:
            print(f"  CSV header mismatch — migrating to new schema")
            with open(csv_path, "r") as f:
                old_rows = list(csv.DictReader(f))
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fields, restval="")
                writer.writeheader()
                for r in old_rows:
                    writer.writerow(r)

    warmup = max(args.beta_warmup_epochs, 1)
    adv_start = args.adversarial_start_epoch  # 1-indexed epoch when adv begins
    d_warmup = args.disc_warmup_epochs
    d_ramp = args.disc_ramp_epochs

    for epoch in range(start_epoch, args.n_epochs):
        t0 = time.time()
        beta_eff = args.beta * min(1.0, (epoch + 1) / warmup)
        adv_active = (epoch + 1) >= adv_start
        lr_g = opt_g.param_groups[0]["lr"]
        lr_d = opt_d.param_groups[0]["lr"]

        # Fix 4: Compute disc_factor (warmup + ramp)
        if not adv_active:
            disc_factor = 0.0
        else:
            adv_epoch = (epoch + 1) - adv_start  # 0-indexed epoch within adv phase
            if adv_epoch < d_warmup:
                disc_factor = 0.0  # D trains, but no adv_g in G loss
            elif d_ramp > 0 and adv_epoch < d_warmup + d_ramp:
                disc_factor = (adv_epoch - d_warmup) / d_ramp
            else:
                disc_factor = 1.0

        # -- Train --
        vae.train()
        disc.train()
        tr_lg, tr_rec, tr_kl, tr_lp = 0., 0., 0., 0.
        tr_adv, tr_adv_w, tr_ld, tr_r1 = 0., 0., 0., 0.
        # Fix 6: accumulators for D diagnostics
        tr_d_real_sum, tr_d_fake_sum = 0., 0.
        tr_d_real_sq_sum, tr_d_fake_sq_sum = 0., 0.
        tr_d_n_batches = 0
        tr_n = 0
        batch_idx = 0

        for (xb,) in tqdm(train_loader,
                          desc=f"VAE-v4 Ep {epoch+1}/{args.n_epochs}",
                          leave=False):
            xb = xb.to(device)
            bs = xb.size(0)
            batch_idx += 1

            # --- Generator step ---
            x_hat, mu, logvar = vae(xb)

            recon = F.l1_loss(x_hat, xb, reduction="mean")
            kl, _ = compute_kl(mu, logvar, free_bits=args.free_bits)
            nll_loss = recon + beta_eff * kl

            # LPIPS (expects [-1, 1] input) — compute every N batches
            lpips_val = 0.0
            if lpips_fn is not None and batch_idx % args.lpips_every == 0:
                lp = lpips_fn(x_hat.clamp(-1, 1), xb)
                nll_loss = nll_loss + args.lambda_lpips * lp
                lpips_val = lp.item()

            # Adversarial (generator) with adaptive weight + disc_factor
            adv_g_val = 0.0
            adv_w_val = args.lambda_adv
            if adv_active and disc_factor > 0:
                fake_pred_g = disc(x_hat)
                adv_g = gen_loss_logistic(fake_pred_g)  # Fix 2

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
            if adv_active:
                x_hat_det = x_hat.detach()
                real_pred = disc(xb)
                fake_pred = disc(x_hat_det)
                loss_d = disc_loss_logistic(real_pred, fake_pred)  # Fix 2

                # Fix 6: collect D diagnostics
                with torch.no_grad():
                    d_real_mean_batch = real_pred.mean().item()
                    d_fake_mean_batch = fake_pred.mean().item()
                    tr_d_real_sum += d_real_mean_batch
                    tr_d_fake_sum += d_fake_mean_batch
                    tr_d_real_sq_sum += real_pred.std().item()
                    tr_d_fake_sq_sum += fake_pred.std().item()
                    tr_d_n_batches += 1

                # R1 gradient penalty (lazy regularization)
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
            tr_n += bs

        tr_lg /= tr_n
        tr_rec /= tr_n
        tr_kl /= tr_n
        tr_lp /= tr_n
        tr_adv /= tr_n
        tr_adv_w /= tr_n
        tr_ld /= tr_n
        tr_r1 /= tr_n

        # Fix 6: compute epoch-level D diagnostics (mean of per-batch means/stds)
        if tr_d_n_batches > 0:
            d_real_mean = tr_d_real_sum / tr_d_n_batches
            d_fake_mean = tr_d_fake_sum / tr_d_n_batches
            d_real_std = tr_d_real_sq_sum / tr_d_n_batches
            d_fake_std = tr_d_fake_sq_sum / tr_d_n_batches
        else:
            d_real_mean = d_fake_mean = d_real_std = d_fake_std = 0.0

        # -- Val (using EMA weights) --
        # Save current weights, load EMA for eval
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

                # Metrics (data range [-1, 1])
                x_hat_c = x_hat.clamp(-1, 1)
                psnr_metric.update(x_hat_c, xb)
                ssim_metric.update(x_hat_c, xb)
                # LPIPS eval expects [-1, 1] unnormalized
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
            # Fix 6: diagnostic columns
            d_real_mean=f"{d_real_mean:.6f}",
            d_fake_mean=f"{d_fake_mean:.6f}",
            d_real_std=f"{d_real_std:.6f}",
            d_fake_std=f"{d_fake_std:.6f}",
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
                       f" df={disc_factor:.2f}")
            if args.r1_gamma > 0:
                adv_str += f" r1={tr_r1:.4f}"
            # Fix 6: print D diagnostics
            adv_str += (f" | D(real)={d_real_mean:+.3f}+-{d_real_std:.3f}"
                        f" D(fake)={d_fake_mean:+.3f}+-{d_fake_std:.3f}")
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
            best_state = copy.deepcopy(vae.state_dict())  # EMA weights
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
        ), ckpt_path)

        sched_g.step()
        # Fix 5: no sched_d.step() — constant LR for D

        # Save reconstruction grid (using EMA)
        if (args.save_grid_every > 0
                and (epoch + 1) % args.save_grid_every == 0):
            # Temporarily use EMA for grid
            ema.apply(vae)
            grid_path = os.path.join(
                results_dir,
                f"recon_grid_vae_v4_epoch{epoch+1}_{args.job_id}.png")
            save_recon_grid(vae, val_loader, device, grid_path)
            vae.load_state_dict(orig_state)

    # -- Done: save best EMA model --
    if best_state is not None:
        vae.load_state_dict(best_state)
        print(f"\nLoaded best EMA model (PSNR={best_psnr:.2f} dB)")

    torch.save(vae.state_dict(), ema_weights_path)
    print(f"Best EMA VAE weights saved to {ema_weights_path}")

    # Also save raw (non-EMA) best as secondary
    torch.save(vae.state_dict(), weights_path)
    print(f"VAE weights saved to {weights_path}")

    # Final reconstruction grid
    final_grid_path = os.path.join(
        results_dir, f"recon_grid_vae_v4_final_{args.job_id}.png")
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
        "vae_arch": "resconv_v3",
        "disc_arch": "patchgan_v2_specnorm",
        "latent_dim": args.latent_dim,
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
