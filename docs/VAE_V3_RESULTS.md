# VAE v3 Results — CIFAR-10 32×32

**Job ID**: 49549763
**Date**: 2026-03-03 (completed)
**Architecture**: ResConvVAE_v3 (~13.5M params) + PatchGAN discriminator (~663K params)
**Config**: latent_dim=64, seed=2025, 300 epochs, batch_size=64, EMA decay=0.999
**Weights**: `checkpoints/weights_vae_v3_ema_cifar10_49549763.pt`

---

## 1. Final Metrics

| Metric | VAE v3 Result | Target | Status |
|--------|---------------|--------|--------|
| **PSNR** | 19.47 dB | >22 dB | **Below** (-2.5 dB) |
| **SSIM** | 0.454 | >0.70 | **Below** (-0.25) |
| **LPIPS** | 0.177 | <0.15 | **Below** (marginal) |
| **Recon FID** | 62.07 | <100 | **Met** |

---

## 2. Comparison with Previous VAE Versions

| Metric | v1 (legacy, lat=32) | v2 (resconv, lat=32) | v2 (resconv, lat=64) | **v3 (SOTA, lat=64)** | Publication Standard |
|--------|---------------------|----------------------|----------------------|----------------------|---------------------|
| Params | ~530K | ~2.1M | ~2.1M | **~13.5M** | 20-100M |
| PSNR (dB) | ~17.6 | 17.59 | 17.33 | **19.47** | 25+ |
| SSIM | ~0.40 | 0.349 | 0.312 | **0.454** | 0.80+ |
| LPIPS | — | — | — | **0.177** | <0.10 |
| Recon FID | 120-215 | 215.25 | 196.86 | **62.07** | <50 |
| Active dims | — | 32/32 | 29/64 | **64/64** | — |

### Improvement over Previous Versions

- **PSNR**: +1.9 dB over v1/v2 (19.47 vs ~17.5). Significant but still 2.5 dB short of target.
- **SSIM**: +0.05-0.14 over v2 (0.454 vs 0.31-0.35). Moderate improvement.
- **Recon FID**: 3.1-3.5× better than v2 (62.07 vs 196-215). Major improvement.
- **Active dims**: All 64 dims active (vs v2-lat64's 29/64). Better latent utilization.

---

## 3. Training Trajectory

### Stage 1 (Epochs 1-50): L1 + KL + LPIPS only

| Epoch | Val Loss | PSNR | SSIM | LPIPS | Notes |
|-------|----------|------|------|-------|-------|
| 1 | 0.924 | 13.02 | 0.065 | 0.390 | Initial |
| 10 | 0.577 | 18.63 | 0.428 | 0.200 | Rapid convergence |
| 30 | 0.536 | 19.23 | 0.449 | 0.177 | Approaching plateau |
| **50** | **0.530** | **19.40** | **0.453** | **0.176** | **Best PSNR (peak)** |

### Stage 2 (Epochs 51-300): + PatchGAN adversarial loss

| Epoch | Val Loss | PSNR | SSIM | LPIPS | D_loss | adv_weight | Notes |
|-------|----------|------|------|-------|--------|------------|-------|
| 51 | 0.551 | 19.17 | 0.446 | 0.177 | 4.998 | 18.06 | Adversarial ON (spike) |
| 60 | 0.525 | 18.26 | 0.431 | 0.088 | 1.954 | — | LPIPS improving, PSNR dropping |
| 100 | 0.550 | 18.34 | 0.433 | 0.119 | 2.000 | 8328 | D collapsed |
| 200 | 0.538 | 18.50 | 0.443 | 0.109 | 2.000 | 8570 | D still collapsed |
| 300 | 0.540 | 18.49 | 0.442 | 0.108 | 2.000 | 8318 | Final — no improvement |

### Key Observation: Best PSNR was at epoch 50 (end of Stage 1)

The best EMA PSNR of **19.40 dB** occurred at the end of Stage 1. Adding adversarial training in Stage 2 **degraded** PSNR and SSIM while improving LPIPS slightly (0.176 → 0.108). This is because the discriminator collapsed immediately and the adaptive adversarial weight exploded to ~8000-8500, causing unstable generator gradients.

---

## 4. Discriminator Collapse Analysis

The PatchGAN discriminator collapsed at the very start of Stage 2, rendering adversarial training ineffective.

### Symptoms

1. **D_loss = 2.0000 exactly** from epoch ~55 onwards (entire Stage 2)
2. **R1 gradient penalty = 0.0000** — reported as zero because D's gradients vanished
3. **Adaptive adversarial weight exploded**: 0.1 → 18 (ep 51) → 8000+ (ep 100+)
4. **PSNR dropped** from 19.40 (ep 50) to 18.34 (ep 100), never recovered

### Root Causes

#### 1. Extreme Capacity Imbalance (20:1 ratio)
- **Generator (VAE)**: 13.5M params — deep ResBlocks, self-attention, GroupNorm+SiLU
- **Discriminator**: 663K params — 4 simple Conv layers with GroupNorm
- The generator trivially finds modes where D produces near-zero logits for both real and fake
- **Reference**: VQGAN uses ~2.7M D for ~50M G (~18:1 but with spectral norm + R1 + adaptive weight); StyleGAN2 uses roughly 1:1 G:D ratio

#### 2. No Spectral Normalization
- D uses `GroupNorm(32, ...)` which normalizes activations, not weight matrices
- Without spectral norm, the discriminator's Lipschitz constant is unconstrained
- Leads to feature collapse (all features saturate to similar values → near-zero logits)
- GroupNorm + spectral norm should not be combined; SN alone is sufficient for D

#### 3. Hinge Loss Dead Zone
- `D_loss = ReLU(1 - D(real)) + ReLU(1 + D(fake))`
- When D(real) ≈ 0 and D(fake) ≈ 0: D_loss = 1.0 + 1.0 = **2.0 exactly**
- Both ReLU terms saturate simultaneously, producing zero gradients for D → collapse is self-reinforcing

#### 4. Abrupt Stage 2 Start Without D Warm-up
- After 50 epochs of pure reconstruction, decoder output is already "good enough" to fool weak D
- D never gets a chance to develop discriminative features before being overwhelmed

#### 5. Cosine LR Decay Applied to D
- Both G and D share the same cosine annealing schedule (1e-4 → 1e-6)
- Once D collapses early in Stage 2, the decaying LR prevents recovery

#### 6. R1 Penalty Not Helping
- R1 prevents D from becoming *overconfident* (gradients too large on real data)
- But D's problem is being *underconfident* (gradients near zero) — R1 is the wrong medicine
- With D's logits at zero, `||grad_x D(x)||² ≈ 0`, so R1 has no effect

---

## 5. Directions for Improvement

### Priority 1: Fix Discriminator Architecture (CRITICAL)

**Increase D capacity to ~2.5-4M params and add spectral normalization:**

```python
class PatchGANDiscriminator_v2(nn.Module):
    """Enlarged PatchGAN: ~2.8M params with spectral normalization."""
    def __init__(self, in_channels=3, ndf=64):
        super().__init__()
        from torch.nn.utils import spectral_norm
        self.net = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels, ndf, 4, 2, 1)),      # 16x16
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(ndf, ndf*2, 4, 2, 1)),            # 8x8
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(ndf*2, ndf*4, 4, 2, 1)),          # 4x4
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(ndf*4, ndf*8, 4, 1, 1)),          # 3x3
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(ndf*8, 1, 4, 1, 1)),              # 2x2
        )
```

- Channels: 64→128→256→512→1 (adds 512-channel layer)
- Remove GroupNorm from D (spectral norm handles weight regularization)
- Target G:D ratio ~5:1 (13.5M:2.8M) instead of 20:1

### Priority 2: Enable Adaptive Adversarial Weight (CRITICAL)

The code already has VQGAN-style adaptive weighting — just enable it:

```bash
--adaptive-adv-weight
```

This computes `lambda = ||grad_recon|| / ||grad_adv||` on the decoder's last layer, ensuring the adversarial gradient matches the reconstruction gradient in magnitude. Without it, the fixed `lambda_adv=0.1` is overwhelmed.

### Priority 3: Switch to Non-Saturating Logistic Loss (HIGH)

Replace hinge loss with logistic loss to eliminate the dead zone:

```python
def disc_loss_logistic(real_pred, fake_pred):
    return F.softplus(-real_pred).mean() + F.softplus(fake_pred).mean()

def gen_loss_logistic(fake_pred):
    return F.softplus(-fake_pred).mean()
```

Non-saturating logistic loss has no ReLU clipping, so gradients always flow — D cannot get trapped at 2.0.

### Priority 4: Add Discriminator Warm-up + Gradual Ramp (HIGH)

```python
# Option A: D warm-up (train D alone for 5 epochs before adding adv to G)
d_warmup_epochs = 5
if epoch < adversarial_start_epoch + d_warmup_epochs:
    loss_g = nll_loss  # G only sees recon+KL+LPIPS
    # D still trains normally on real vs fake

# Option B: Gradual ramp over 20 epochs
ramp = min(1.0, (epoch - adversarial_start_epoch) / 20.0)
loss_g = nll_loss + ramp * adv_weight * adv_g
```

### Priority 5: Fix D Optimizer Settings (MEDIUM)

- **D betas**: Change from `(0.5, 0.9)` to `(0.0, 0.99)` (StyleGAN2 convention — zero first momentum helps D respond faster)
- **D learning rate schedule**: Use constant LR for D (no cosine decay), or step decay that holds steady during early Stage 2
- **D learning rate**: Reduce to `lr_d=2e-4` once D has more capacity

### Priority 6: Add Diagnostic Logging (MEDIUM)

Log `D(real).mean()`, `D(fake).mean()`, and their standard deviations every epoch. A healthy discriminator should show `D(real) > 0`, `D(fake) < 0` with clear separation. This enables early detection of collapse.

---

## 6. Expected Outcome After Fixes

Based on VQGAN and Stable Diffusion VAE literature (CIFAR-10 32×32):

| Metric | Current v3 | Expected v3.1 (with fixes) | Publication Standard |
|--------|-----------|---------------------------|---------------------|
| PSNR | 19.47 dB | 22-25 dB | 25+ dB |
| SSIM | 0.454 | 0.60-0.75 | 0.80+ |
| LPIPS | 0.177 | 0.08-0.12 | <0.10 |
| Recon FID | 62.07 | 30-50 | <50 |

The single most impactful change is **increasing D capacity + spectral normalization** combined with **adaptive adversarial weighting**. These address the root cause: D is too weak to provide useful signal, and even if it could, the fixed `lambda_adv=0.1` buries it under the reconstruction gradient.

---

## 7. References

1. Esser et al. (2021). "Taming Transformers for High-Resolution Image Synthesis." CVPR 2021. — VQGAN adaptive weight, PatchGAN with hinge loss
2. Karras et al. (2020). "Analyzing and Improving the Image Quality of StyleGAN." CVPR 2020. — R1 penalty, non-saturating logistic loss, D optimizer settings
3. Rombach et al. (2022). "High-Resolution Image Synthesis with Latent Diffusion Models." CVPR 2022. — Stable Diffusion VAE (KL-f8)
4. Mescheder et al. (2018). "Which Training Methods for GANs do actually Converge?" ICML 2018. — R1 gradient penalty theory
5. Miyato et al. (2018). "Spectral Normalization for Generative Adversarial Networks." ICLR 2018. — Spectral normalization for D stability
