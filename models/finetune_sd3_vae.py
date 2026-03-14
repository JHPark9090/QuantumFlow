"""
Fine-tune SD3.5 VAE on CIFAR-10 32x32
=======================================

The pretrained SD3.5 AutoencoderKL was trained on high-resolution natural images.
This script fine-tunes it on CIFAR-10 32x32 to improve reconstruction quality
for small images.

At 32x32, the VAE does 8x spatial downscale: 32x32 -> 4x4 x 16ch = 256-d latent.

Strategy:
  - Full fine-tuning (all encoder + decoder parameters)
  - Low learning rate (1e-5) with cosine annealing
  - MSE + VGG perceptual loss
  - Optional KL regularization (low beta)
  - EMA weights for stability
  - Full CIFAR-10 (50k train) to reduce overfitting risk
  - Early stopping on validation loss

Usage:
  python models/finetune_sd3_vae.py --n-epochs=100 --seed=2025 \\
      --job-id=sd3_finetune_${SLURM_JOB_ID}

  # Resume:
  PREV_JOB_ID=XXXXX python models/finetune_sd3_vae.py --resume \\
      --job-id=sd3_finetune_${PREV_JOB_ID}
"""

import argparse
import os
import sys
import random
import copy
import json
import time
import csv
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

DATA_ROOT = "/pscratch/sd/j/junghoon/data"


# ---------------------------------------------------------------------------
# 1. Argparse
# ---------------------------------------------------------------------------
def get_args():
    p = argparse.ArgumentParser(description="Fine-tune SD3.5 VAE on CIFAR-10 32x32")

    p.add_argument("--sd3-model-id", type=str,
                   default="stabilityai/stable-diffusion-3.5-large")
    p.add_argument("--lr", type=float, default=1e-5,
                   help="Learning rate (low for fine-tuning)")
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--n-epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=64)

    # Loss weights
    p.add_argument("--lambda-perc", type=float, default=0.1,
                   help="VGG perceptual loss weight")
    p.add_argument("--beta", type=float, default=0.0001,
                   help="KL divergence weight (low to preserve reconstruction)")
    p.add_argument("--beta-warmup-epochs", type=int, default=10,
                   help="Linear ramp from 0 to beta")

    # EMA
    p.add_argument("--ema-decay", type=float, default=0.999)

    # Data
    p.add_argument("--n-train", type=int, default=50000,
                   help="Use full CIFAR-10 training set")
    p.add_argument("--n-valtest", type=int, default=10000)
    p.add_argument("--seed", type=int, default=2025)

    # Fine-tuning mode
    p.add_argument("--freeze-encoder", action="store_true",
                   help="Freeze encoder, only fine-tune decoder")

    # I/O
    p.add_argument("--job-id", type=str, default="sd3_finetune_001")
    p.add_argument("--base-path", type=str, default=".")
    p.add_argument("--resume", action="store_true")
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
# 3. VGG Perceptual Loss
# ---------------------------------------------------------------------------
class VGGPerceptualLoss(nn.Module):
    """Multi-scale feature matching loss using VGG16."""

    def __init__(self, device="cpu"):
        super().__init__()
        from torchvision.models import vgg16, VGG16_Weights
        vgg = vgg16(weights=VGG16_Weights.DEFAULT).features.to(device).eval()
        for p in vgg.parameters():
            p.requires_grad = False

        # Extract features at relu1_2, relu2_2, relu3_3
        self.blocks = nn.ModuleList([
            vgg[:4],    # relu1_2
            vgg[4:9],   # relu2_2
            vgg[9:16],  # relu3_3
        ])
        self.register_buffer("mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x, y):
        # x, y in [-1, 1] -> normalize to ImageNet range
        x = (x * 0.5 + 0.5 - self.mean) / self.std
        y = (y * 0.5 + 0.5 - self.mean) / self.std
        loss = 0.0
        for block in self.blocks:
            x = block(x)
            y = block(y)
            loss += F.mse_loss(x, y)
        return loss


# ---------------------------------------------------------------------------
# 4. Data Loader (images in [-1, 1])
# ---------------------------------------------------------------------------
def load_cifar_data(seed, n_train, n_valtest, batch_size):
    from torchvision.datasets import CIFAR10
    torch.manual_seed(seed)

    data_train = CIFAR10(root=DATA_ROOT, train=True, download=True)
    data_test = CIFAR10(root=DATA_ROOT, train=False, download=True)

    X_tr = torch.tensor(data_train.data).float().permute(0, 3, 1, 2) / 255.0
    X_te = torch.tensor(data_test.data).float().permute(0, 3, 1, 2) / 255.0

    # Scale to [-1, 1] (SD3.5 VAE convention)
    X_tr = X_tr * 2.0 - 1.0
    X_te = X_te * 2.0 - 1.0

    X_tr = X_tr[torch.randperm(len(X_tr))[:n_train]]
    X_te = X_te[torch.randperm(len(X_te))[:n_valtest]]

    n_val = min(len(X_te) // 2, 5000)
    X_val = X_te[:n_val]
    X_te = X_te[n_val:]

    train_loader = DataLoader(TensorDataset(X_tr), batch_size=batch_size,
                              shuffle=True, drop_last=True)
    val_loader = DataLoader(TensorDataset(X_val), batch_size=batch_size,
                            shuffle=False)
    test_loader = DataLoader(TensorDataset(X_te), batch_size=batch_size,
                             shuffle=False)

    print(f"  Data: train={len(X_tr)}, val={len(X_val)}, test={len(X_te)}")
    return train_loader, val_loader, test_loader


# ---------------------------------------------------------------------------
# 5. EMA helper
# ---------------------------------------------------------------------------
def update_ema(ema_model, model, decay):
    with torch.no_grad():
        for ema_p, p in zip(ema_model.parameters(), model.parameters()):
            ema_p.mul_(decay).add_(p.data, alpha=1.0 - decay)


# ---------------------------------------------------------------------------
# 6. Save reconstruction grid
# ---------------------------------------------------------------------------
@torch.no_grad()
def save_grid(vae, val_loader, epoch, job_id, device, base_path="."):
    from torchvision.utils import save_image
    x = next(iter(val_loader))[0][:16].to(device)

    z = vae.encode(x).latent_dist.mean
    x_recon = vae.decode(z).sample.clamp(-1, 1)

    grid = torch.cat([x, x_recon], dim=0)
    grid = grid * 0.5 + 0.5  # [-1,1] -> [0,1]

    save_dir = os.path.join(base_path, "checkpoints", "grids")
    os.makedirs(save_dir, exist_ok=True)
    save_image(grid, os.path.join(save_dir, f"recon_{job_id}_ep{epoch:04d}.png"),
               nrow=16)


# ---------------------------------------------------------------------------
# 7. Compute reconstruction FID
# ---------------------------------------------------------------------------
@torch.no_grad()
def compute_recon_fid(vae, test_loader, device):
    try:
        from torchmetrics.image.fid import FrechetInceptionDistance
    except ImportError:
        print("  torchmetrics not available, skipping FID")
        return None

    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)

    for (x,) in test_loader:
        x = x.to(device)
        z = vae.encode(x).latent_dist.mean
        x_recon = vae.decode(z).sample.clamp(-1, 1)

        # FID expects [0, 1]
        x_01 = x * 0.5 + 0.5
        r_01 = x_recon * 0.5 + 0.5

        fid.update(x_01, real=True)
        fid.update(r_01, real=False)

    return fid.compute().item()


# ---------------------------------------------------------------------------
# 8. Training loop
# ---------------------------------------------------------------------------
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_all_seeds(args.seed)
    print(f"Device: {device}")

    # Load data
    train_loader, val_loader, test_loader = load_cifar_data(
        args.seed, args.n_train, args.n_valtest, args.batch_size)

    # Load SD3.5 VAE
    from diffusers import AutoencoderKL
    print(f"Loading SD3.5 VAE from {args.sd3_model_id} ...")
    vae = AutoencoderKL.from_pretrained(
        args.sd3_model_id, subfolder="vae", torch_dtype=torch.float32)
    vae = vae.to(device)

    # Unfreeze parameters
    n_total = 0
    n_trainable = 0
    for name, p in vae.named_parameters():
        n_total += p.numel()
        if args.freeze_encoder and ("encoder" in name or "quant_conv" in name):
            p.requires_grad = False
        else:
            p.requires_grad = True
            n_trainable += p.numel()

    mode = "decoder-only" if args.freeze_encoder else "full"
    print(f"Fine-tuning mode: {mode}")
    print(f"  Total params: {n_total:,}")
    print(f"  Trainable params: {n_trainable:,}")

    # Check latent shape
    with torch.no_grad():
        dummy = torch.randn(1, 3, 32, 32, device=device)
        z_test = vae.encode(dummy).latent_dist.mean
        print(f"  Latent shape at 32x32: {z_test.shape} "
              f"(flat = {z_test.numel()})")

    # Perceptual loss
    vgg_loss = VGGPerceptualLoss(device=device).to(device)

    # EMA
    vae_ema = copy.deepcopy(vae)
    for p in vae_ema.parameters():
        p.requires_grad = False
    vae_ema.eval()

    # Optimizer + scheduler
    trainable_params = [p for p in vae.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.n_epochs, eta_min=args.lr * 0.01)

    # Checkpointing
    ckpt_dir = os.path.join(args.base_path, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    log_file = os.path.join(ckpt_dir, f"log_{args.job_id}.csv")

    start_epoch = 0
    best_val_loss = float("inf")

    # Resume
    ckpt_path = os.path.join(ckpt_dir, f"ckpt_{args.job_id}.pt")
    if args.resume and os.path.exists(ckpt_path):
        print(f"Resuming from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        vae.load_state_dict(ckpt["vae_state_dict"])
        vae_ema.load_state_dict(ckpt["vae_ema_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        print(f"  Resumed at epoch {start_epoch}, best_val_loss={best_val_loss:.4f}")

    # CSV logger
    if start_epoch == 0:
        with open(log_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "train_mse", "train_perc",
                             "train_kl", "val_loss", "val_mse", "lr"])

    print(f"\n{'='*60}")
    print(f"Fine-tuning SD3.5 VAE on CIFAR-10 32x32")
    print(f"  Epochs: {args.n_epochs}, LR: {args.lr}, Beta: {args.beta}")
    print(f"  Lambda_perc: {args.lambda_perc}, Weight decay: {args.weight_decay}")
    print(f"  EMA decay: {args.ema_decay}")
    print(f"{'='*60}\n")

    for epoch in range(start_epoch, args.n_epochs):
        vae.train()
        epoch_loss = 0.0
        epoch_mse = 0.0
        epoch_perc = 0.0
        epoch_kl = 0.0
        n_batches = 0

        # Beta warmup
        if epoch < args.beta_warmup_epochs:
            beta_t = args.beta * (epoch / max(args.beta_warmup_epochs, 1))
        else:
            beta_t = args.beta

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.n_epochs}",
                    leave=False)
        for (x,) in pbar:
            x = x.to(device)

            # Forward through VAE
            posterior = vae.encode(x).latent_dist
            z = posterior.sample()  # use sample (not mean) for KL
            x_recon = vae.decode(z).sample

            # MSE reconstruction loss
            mse_loss = F.mse_loss(x_recon, x)

            # Perceptual loss
            perc_loss = vgg_loss(x_recon.clamp(-1, 1), x)

            # KL divergence
            kl_loss = posterior.kl().mean()

            # Total loss
            loss = mse_loss + args.lambda_perc * perc_loss + beta_t * kl_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()

            # EMA update
            update_ema(vae_ema, vae, args.ema_decay)

            epoch_loss += loss.item()
            epoch_mse += mse_loss.item()
            epoch_perc += perc_loss.item()
            epoch_kl += kl_loss.item()
            n_batches += 1

            pbar.set_postfix(loss=f"{loss.item():.4f}",
                             mse=f"{mse_loss.item():.4f}",
                             perc=f"{perc_loss.item():.4f}")

        scheduler.step()

        avg_loss = epoch_loss / n_batches
        avg_mse = epoch_mse / n_batches
        avg_perc = epoch_perc / n_batches
        avg_kl = epoch_kl / n_batches

        # Validation (using EMA model)
        vae_ema.eval()
        val_loss = 0.0
        val_mse = 0.0
        val_n = 0
        with torch.no_grad():
            for (x,) in val_loader:
                x = x.to(device)
                z = vae_ema.encode(x).latent_dist.mean
                x_recon = vae_ema.decode(z).sample
                mse = F.mse_loss(x_recon, x)
                perc = vgg_loss(x_recon.clamp(-1, 1), x)
                val_loss += (mse + args.lambda_perc * perc).item()
                val_mse += mse.item()
                val_n += 1

        val_loss /= val_n
        val_mse /= val_n

        cur_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch+1:3d} | train_loss={avg_loss:.4f} "
              f"mse={avg_mse:.4f} perc={avg_perc:.4f} kl={avg_kl:.4f} | "
              f"val_loss={val_loss:.4f} val_mse={val_mse:.4f} | "
              f"lr={cur_lr:.2e} beta={beta_t:.5f}")

        # Log
        with open(log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch+1, f"{avg_loss:.6f}", f"{avg_mse:.6f}",
                             f"{avg_perc:.6f}", f"{avg_kl:.6f}",
                             f"{val_loss:.6f}", f"{val_mse:.6f}",
                             f"{cur_lr:.2e}"])

        # Save checkpoint
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss

        torch.save({
            "epoch": epoch,
            "vae_state_dict": vae.state_dict(),
            "vae_ema_state_dict": vae_ema.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_loss": best_val_loss,
            "args": vars(args),
        }, ckpt_path)

        # Save best EMA weights separately (for downstream use)
        if is_best:
            best_path = os.path.join(
                ckpt_dir, f"weights_sd3vae_finetuned_{args.job_id}.pt")
            torch.save(vae_ema.state_dict(), best_path)
            print(f"  ** New best val_loss={val_loss:.4f}, saved EMA weights")

        # Save grid
        if (args.save_grid_every > 0 and
                (epoch + 1) % args.save_grid_every == 0):
            save_grid(vae_ema, val_loader, epoch + 1, args.job_id,
                      device, args.base_path)

    # Final evaluation
    print(f"\n{'='*60}")
    print("Training complete. Final evaluation with EMA model:")

    if args.compute_recon_fid:
        print("Computing reconstruction FID ...")
        fid = compute_recon_fid(vae_ema, test_loader, device)
        if fid is not None:
            print(f"  Reconstruction FID: {fid:.2f}")

    # Save final weights
    final_path = os.path.join(
        ckpt_dir, f"weights_sd3vae_finetuned_final_{args.job_id}.pt")
    torch.save(vae_ema.state_dict(), final_path)
    print(f"  Final EMA weights saved to: {final_path}")

    # Compute final PSNR
    vae_ema.eval()
    psnr_sum = 0.0
    n = 0
    with torch.no_grad():
        for (x,) in test_loader:
            x = x.to(device)
            z = vae_ema.encode(x).latent_dist.mean
            x_recon = vae_ema.decode(z).sample.clamp(-1, 1)
            mse = F.mse_loss(x_recon, x, reduction="none").mean(dim=(1, 2, 3))
            # PSNR on [-1,1] range: max pixel value is 1, so range is 2
            psnr = 10 * torch.log10(4.0 / (mse + 1e-10))
            psnr_sum += psnr.sum().item()
            n += x.size(0)

    avg_psnr = psnr_sum / n
    print(f"  Test PSNR: {avg_psnr:.2f} dB")

    # Save summary
    summary = {
        "job_id": args.job_id,
        "model": "SD3.5 VAE fine-tuned",
        "mode": mode,
        "n_epochs": args.n_epochs,
        "lr": args.lr,
        "beta": args.beta,
        "lambda_perc": args.lambda_perc,
        "best_val_loss": best_val_loss,
        "test_psnr": avg_psnr,
        "n_train": args.n_train,
        "trainable_params": n_trainable,
    }
    if args.compute_recon_fid and fid is not None:
        summary["recon_fid"] = fid

    summary_path = os.path.join(ckpt_dir, f"summary_{args.job_id}.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary saved to: {summary_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    args = get_args()
    train(args)
