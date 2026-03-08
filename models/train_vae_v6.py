"""
VAE v6 -- Gradual Channel Reduction + No Adversarial Training
==============================================================

Fixes two problems found in v5:

  Problem 1 (Discriminator collapse):
    v4 and v5 both suffer from discriminator collapse — D outputs converge to
    zero, adaptive weight explodes (up to 1300+), and PSNR drops 1-1.3 dB.
    v4's reported PSNR 19.51 was actually the pre-collapse EMA checkpoint.
    FIX: Remove adversarial training entirely. It contributed 0 dB in v4.

  Problem 2 (1x1 conv bottleneck):
    v5's Conv2d(256, 4, 1) compresses 256 channels to 4 INDEPENDENTLY at each
    pixel (no spatial mixing). This is a 64:1 channel compression with zero
    spatial awareness, worse than v4's stride-2 conv which mixes 3x3 neighborhoods.
    v5 stage-1 PSNR was 18.84 vs v4's 19.38 (-0.54 dB).
    FIX: Gradual reduction via ResBlock: 256->64 (ResBlock with spatial mixing)
    then Conv2d(64, c_z, 3, 1, 1) (3x3 conv preserves spatial context).

Architecture (ResConvVAE_v6):
  Encoder: 32x32 -> 16x16 -> 8x8 -> 4x4 -> ResBlock(256,64) -> Conv3x3(64,c_z) -> flatten -> latent
  Decoder: latent -> reshape(c_z,4,4) -> Conv3x3(c_z,64) -> ResBlock(64,256) -> 8x8 -> 16x16 -> 32x32

  With c_z=4, latent_dim=64: flatten(4*4*4=64), fc_mu=Linear(64,64)=1:1
  Channel reduction: 256->64 (ResBlock, 4:1) then 64->4 (Conv3x3, 16:1)
  vs v5: 256->4 (Conv1x1, 64:1 with no spatial mixing)

Loss: L1 + beta*KL + lambda_lpips*LPIPS (no adversarial)
EMA decay=0.999 on VAE weights throughout.

Usage:
  python models/train_vae_v6.py --dataset=cifar10 --n-epochs=300 \\
      --seed=2025 --compute-recon-fid --job-id=vae_v6_cifar_${SLURM_JOB_ID}
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
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split
from tqdm import tqdm

DATA_ROOT = "/pscratch/sd/j/junghoon/data"


# ---------------------------------------------------------------------------
# 1. Argparse
# ---------------------------------------------------------------------------
def get_args():
    p = argparse.ArgumentParser(description="VAE v6 -- Gradual Reduction, No Adversarial")

    p.add_argument("--dataset", type=str, default="cifar10",
                   choices=["cifar10", "coco", "mnist", "fashion"])
    p.add_argument("--latent-dim", type=int, default=64)
    p.add_argument("--c-z", type=int, default=4,
                   help="Channel count after 3x3 conv at 4x4 "
                        "(pre-flatten = c_z * 16)")

    # Loss weights
    p.add_argument("--beta", type=float, default=0.001, help="KL weight")
    p.add_argument("--beta-warmup-epochs", type=int, default=10,
                   help="Linear ramp from 0 to beta")
    p.add_argument("--lambda-lpips", type=float, default=1.0,
                   help="LPIPS perceptual loss weight")
    p.add_argument("--free-bits", type=float, default=0.25,
                   help="Minimum KL per latent dimension in nats")
    p.add_argument("--lpips-every", type=int, default=1,
                   help="Compute LPIPS loss every N batches (1=every batch)")

    # Training
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--n-epochs", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--ema-decay", type=float, default=0.999)

    # Data
    p.add_argument("--n-train", type=int, default=50000)
    p.add_argument("--n-valtest", type=int, default=10000)
    p.add_argument("--img-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=2025)

    # I/O
    p.add_argument("--job-id", type=str, default="vae_v6_001")
    p.add_argument("--base-path", type=str, default=".")
    p.add_argument("--resume", action="store_true")

    # Visualization / evaluation
    p.add_argument("--save-grid-every", type=int, default=10,
                   help="Save reconstruction grid every N epochs (0 to disable)")
    p.add_argument("--compute-recon-fid", action="store_true")

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
# 3. Data Loaders (images scaled to [-1, 1])
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
        x = x * 2.0 - 1.0  # normalize to [-1, 1]
        return (x,)


def load_coco_2d(seed, n_train, n_valtest, batch_size, img_size=128):
    from pycocotools.coco import COCO as COCOApi
    torch.manual_seed(seed)
    np.random.seed(seed)
    data_dir = os.path.join(DATA_ROOT, "coco")
    ann_file = os.path.join(data_dir, "annotations/instances_train2017.json")
    img_dir = os.path.join(data_dir, "train2017")
    coco = COCOApi(ann_file)
    img_ids = sorted(coco.getImgIds())
    rng = np.random.RandomState(seed)
    rng.shuffle(img_ids)
    img_ids = img_ids[:n_train + n_valtest]
    paths = []
    for iid in img_ids:
        info = coco.loadImgs(iid)[0]
        paths.append(os.path.join(img_dir, info["file_name"]))
    train_ds = LazyCOCODataset(paths[:n_train], img_size)
    valtest_ds = LazyCOCODataset(paths[n_train:n_train + n_valtest], img_size)
    val_sz = len(valtest_ds) // 2
    test_sz = len(valtest_ds) - val_sz
    val_ds, test_ds = random_split(valtest_ds, [val_sz, test_sz])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              drop_last=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=2, pin_memory=True)
    return train_loader, val_loader, test_loader


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
# 4. Model Architecture -- ResConvVAE_v6 (gradual channel reduction)
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


class ResConvVAE_v6(nn.Module):
    """VAE v6: Gradual channel reduction with spatial mixing at 4x4.

    Resolution-adaptive: supports img_size=32 (3 downsamples) and
    img_size=128 (5 downsamples). Both reach 4x4 spatial bottleneck.

    Bottleneck: ResBlock(256->64) + Conv3x3(64->c_z) at 4x4, then
    flatten(c_z*16) -> fc_mu(latent_dim). Set c_z = latent_dim/16
    for 1:1 FC ratio.

    Channel progression:
      32x32:  64 -> 128 -> 256 -> 256(4x4)
      128x128: 32 -> 64 -> 64 -> 128 -> 256 -> 256(4x4)
    """

    def __init__(self, latent_dim=64, in_channels=3, c_z=4, img_size=32):
        super().__init__()
        self.latent_dim = latent_dim
        self.c_z = c_z

        # Build encoder dynamically based on img_size
        enc_layers = []
        if img_size == 128:
            # 128x128 -> 64x64 -> 32x32 (2 extra downsamples)
            enc_layers += [
                nn.Conv2d(in_channels, 32, 3, 1, 1, bias=False),    # 128x128
                ResBlock_v3(32), ResBlock_v3(32),                    # 128x128
                nn.Conv2d(32, 32, 3, 2, 1, bias=False),             # 64x64
                ResBlock_v3(32, 64), ResBlock_v3(64),                # 64x64
                nn.Conv2d(64, 64, 3, 2, 1, bias=False),             # 32x32
            ]
            first_ch = 64  # entering the shared 32x32 block
        elif img_size == 32:
            enc_layers += [
                nn.Conv2d(in_channels, 64, 3, 1, 1, bias=False),    # 32x32
            ]
            first_ch = 64
        else:
            raise ValueError(f"img_size={img_size} not supported (use 32 or 128)")

        # Shared 32x32 -> 16x16 -> 8x8 -> 4x4 (3 downsamples)
        enc_layers += [
            ResBlock_v3(first_ch, 64), ResBlock_v3(64),              # 32x32
            nn.Conv2d(64, 64, 3, 2, 1, bias=False),                 # 16x16
            ResBlock_v3(64, 128), ResBlock_v3(128),                  # 16x16
            nn.Conv2d(128, 128, 3, 2, 1, bias=False),               # 8x8
            ResBlock_v3(128, 256), ResBlock_v3(256),                 # 8x8
            SelfAttention(256),                                      # 8x8
            nn.Conv2d(256, 256, 3, 2, 1, bias=False),               # 4x4
            ResBlock_v3(256), ResBlock_v3(256),                      # 4x4
            SelfAttention(256),                                      # 4x4
            # --- Gradual channel reduction (v6 fix) ---
            ResBlock_v3(256, 64),                                    # 4x4, 256->64
            nn.GroupNorm(32, 64), nn.SiLU(inplace=True),
            nn.Conv2d(64, c_z, 3, 1, 1, bias=False),                # 4x4, 64->c_z
        ]
        self.encoder = nn.Sequential(*enc_layers)

        flat_dim = c_z * 4 * 4
        self.fc_mu = nn.Linear(flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(flat_dim, latent_dim)

        # Build decoder dynamically (mirrors encoder)
        self.fc_dec = nn.Linear(latent_dim, flat_dim)
        dec_layers = [
            # --- Gradual channel expansion ---
            nn.Conv2d(c_z, 64, 3, 1, 1, bias=False),               # 4x4, c_z->64
            ResBlock_v3(64, 256),                                    # 4x4, 64->256
            ResBlock_v3(256), ResBlock_v3(256),
            SelfAttention(256),                                      # 4x4
            nn.ConvTranspose2d(256, 256, 4, 2, 1, bias=False),      # 8x8
            ResBlock_v3(256, 128), ResBlock_v3(128),
            SelfAttention(128),                                      # 8x8
            nn.ConvTranspose2d(128, 128, 4, 2, 1, bias=False),      # 16x16
            ResBlock_v3(128, 64), ResBlock_v3(64),                   # 16x16
            nn.ConvTranspose2d(64, 64, 4, 2, 1, bias=False),        # 32x32
        ]
        if img_size == 128:
            dec_layers += [
                ResBlock_v3(64), ResBlock_v3(64),                    # 32x32
                nn.ConvTranspose2d(64, 64, 4, 2, 1, bias=False),    # 64x64
                ResBlock_v3(64, 32), ResBlock_v3(32),                # 64x64
                nn.ConvTranspose2d(32, 32, 4, 2, 1, bias=False),    # 128x128
                ResBlock_v3(32), ResBlock_v3(32),                    # 128x128
                nn.GroupNorm(32, 32), nn.SiLU(inplace=True),
                nn.Conv2d(32, in_channels, 3, 1, 1),
                nn.Tanh(),
            ]
        else:
            dec_layers += [
                ResBlock_v3(64), ResBlock_v3(64),                    # 32x32
                nn.GroupNorm(32, 64), nn.SiLU(inplace=True),
                nn.Conv2d(64, in_channels, 3, 1, 1),
                nn.Tanh(),
            ]
        self.decoder = nn.Sequential(*dec_layers)

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
# 5. EMA Helper
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
        model.load_state_dict(self.shadow)


# ---------------------------------------------------------------------------
# 6. KL Computation
# ---------------------------------------------------------------------------
def compute_kl(mu, logvar, free_bits=0.25):
    """KL with free bits: sum over latent dims, mean over batch."""
    kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    kl_per_dim_avg = kl_per_dim.mean(dim=0)
    kl_per_dim_clamped = torch.clamp(kl_per_dim_avg, min=free_bits)
    kl = kl_per_dim_clamped.sum()
    return kl, kl_per_dim_avg


# ---------------------------------------------------------------------------
# 7. Visualization
# ---------------------------------------------------------------------------
def save_recon_grid(vae, val_loader, device, save_path, n_images=8):
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
# 8. Reconstruction FID
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
# 9. Main Training Loop (NO adversarial training)
# ---------------------------------------------------------------------------
def train_vae(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    set_all_seeds(args.seed)

    # Data
    loader_fn = {
        "cifar10": load_cifar_2d,
        "coco": load_coco_2d,
        "mnist": load_mnist_2d,
        "fashion": load_fashion_2d,
    }[args.dataset]
    train_loader, val_loader, test_loader = loader_fn(
        args.seed, args.n_train, args.n_valtest, args.batch_size, args.img_size)
    print(f"Data: {args.dataset} | train={args.n_train} val+test={args.n_valtest}")
    print(f"Data range: [-1, 1] (Tanh output)")

    # VAE v6
    vae = ResConvVAE_v6(latent_dim=args.latent_dim, c_z=args.c_z,
                        img_size=args.img_size).to(device)
    total_p = sum(p.numel() for p in vae.parameters() if p.requires_grad)
    flat_dim = args.c_z * 4 * 4
    print(f"VAE v6 | params={total_p:,} | latent_dim={args.latent_dim} | "
          f"c_z={args.c_z} | pre-flatten={flat_dim} | "
          f"FC ratio={flat_dim}:{args.latent_dim} = "
          f"{flat_dim / args.latent_dim:.1f}:1")
    print(f"  Encoder bottleneck: ResBlock(256->64) + Conv3x3(64->{args.c_z}) at 4x4")
    print(f"  Channel reduction: 256 -> 64 (4:1, ResBlock) -> {args.c_z} (16:1, Conv3x3)")

    # Optimizer (single optimizer, no discriminator)
    optimizer = torch.optim.Adam(vae.parameters(), lr=args.lr, betas=(0.5, 0.9))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.n_epochs, eta_min=1e-6)

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
                             f"ckpt_vae_v6_{args.dataset}_{args.job_id}.pt")
    ema_weights_path = os.path.join(
        ckpt_dir, f"weights_vae_v6_ema_{args.dataset}_{args.job_id}.pt")
    csv_path = os.path.join(results_dir,
                            f"log_vae_v6_{args.dataset}_{args.job_id}.csv")
    metrics_path = os.path.join(results_dir,
                                f"metrics_vae_v6_{args.dataset}_{args.job_id}.json")

    fields = [
        "epoch", "train_loss", "train_recon", "train_kl", "train_lpips",
        "val_loss", "val_recon", "val_kl", "val_lpips",
        "val_psnr", "val_ssim", "val_lpips_eval",
        "kl_mean_per_dim", "kl_min_per_dim", "kl_max_per_dim",
        "active_dims", "mu_norm",
        "beta_eff", "lr", "time_s",
    ]

    print(f"\n--- VAE v6 Config ---")
    print(f"  NO ADVERSARIAL TRAINING (discriminator removed)")
    print(f"  Gradual reduction: ResBlock(256->64) + Conv3x3(64->{args.c_z})")
    print(f"  beta={args.beta}, free_bits={args.free_bits}")
    print(f"  lambda_lpips={args.lambda_lpips}")
    print(f"  lr={args.lr}, betas=(0.5, 0.9), cosine LR")
    print(f"  EMA decay={args.ema_decay}")
    print(f"---------------------\n")

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
        lr_now = optimizer.param_groups[0]["lr"]

        # -- Train --
        vae.train()
        tr_loss, tr_rec, tr_kl, tr_lp = 0., 0., 0., 0.
        tr_n = 0
        batch_idx = 0

        for (xb,) in tqdm(train_loader,
                          desc=f"VAE-v6 Ep {epoch+1}/{args.n_epochs}",
                          leave=False):
            xb = xb.to(device)
            bs = xb.size(0)
            batch_idx += 1

            x_hat, mu, logvar = vae(xb)

            recon = F.l1_loss(x_hat, xb, reduction="mean")
            kl, _ = compute_kl(mu, logvar, free_bits=args.free_bits)
            loss = recon + beta_eff * kl

            lpips_val = 0.0
            if lpips_fn is not None and batch_idx % args.lpips_every == 0:
                lp = lpips_fn(x_hat.clamp(-1, 1), xb)
                loss = loss + args.lambda_lpips * lp
                lpips_val = lp.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # EMA update
            ema.update(vae)

            tr_loss += loss.item() * bs
            tr_rec += recon.item() * bs
            tr_kl += kl.item() * bs
            tr_lp += lpips_val * bs
            tr_n += bs

        tr_loss /= tr_n
        tr_rec /= tr_n
        tr_kl /= tr_n
        tr_lp /= tr_n

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
            train_loss=f"{tr_loss:.6f}",
            train_recon=f"{tr_rec:.6f}",
            train_kl=f"{tr_kl:.6f}",
            train_lpips=f"{tr_lp:.6f}",
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
            lr=f"{lr_now:.2e}",
            time_s=f"{dt:.1f}",
        )
        with open(csv_path, "a", newline="") as f:
            csv.DictWriter(f, fields).writerow(row)

        print(
            f"  Ep {epoch+1:3d} | "
            f"Train {tr_loss:.4f} (L1={tr_rec:.4f} kl={tr_kl:.4f} "
            f"lpips={tr_lp:.4f}) | "
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
            optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(),
            ema=ema.state_dict(),
            best_psnr=best_psnr,
            c_z=args.c_z,
            latent_dim=args.latent_dim,
        ), ckpt_path)

        scheduler.step()

        # Save reconstruction grid (using EMA)
        if (args.save_grid_every > 0
                and (epoch + 1) % args.save_grid_every == 0):
            ema.apply(vae)
            grid_path = os.path.join(
                results_dir,
                f"recon_grid_vae_v6_epoch{epoch+1}_{args.job_id}.png")
            save_recon_grid(vae, val_loader, device, grid_path)
            vae.load_state_dict(orig_state)

    # -- Done: save best EMA model --
    if best_state is not None:
        vae.load_state_dict(best_state)
        print(f"\nLoaded best EMA model (PSNR={best_psnr:.2f} dB)")

    torch.save(vae.state_dict(), ema_weights_path)
    print(f"Best EMA VAE weights saved to {ema_weights_path}")

    # Final reconstruction grid
    final_grid_path = os.path.join(
        results_dir, f"recon_grid_vae_v6_final_{args.job_id}.png")
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
        "vae_arch": "resconv_v6",
        "adversarial": False,
        "latent_dim": args.latent_dim,
        "c_z": args.c_z,
        "pre_flatten_dim": flat_dim,
        "fc_compression_ratio": f"{flat_dim}:{args.latent_dim}",
        "encoder_reduction": f"ResBlock(256->64) + Conv3x3(64->{args.c_z})",
        "beta": args.beta,
        "lambda_lpips": args.lambda_lpips,
        "free_bits": args.free_bits,
        "n_epochs": args.n_epochs,
        "seed": args.seed,
        "n_params_vae": total_p,
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
