# VAE Bottleneck Diagnosis & Proposed Fix for Quantum VF

**Date**: 2026-03-05
**Context**: VAE v3 (PSNR 19.47) and VAE v4 (PSNR 19.51) both plateau well below publication targets (PSNR 25+). Discriminator fixes in v4 were ineffective because the reconstruction ceiling is set by information loss in the encoder, not perceptual sharpening.

---

## 1. Current Architecture (VAE v3 / v4)

```
ENCODER
  Input: (B, 3, 32, 32)                              3,072 dims
  Conv2d(3, 64, 3x3)                                 32x32
  ResBlock(64) x2                                     32x32
  Conv2d(64, 64, stride=2)                            16x16   downsample 1
  ResBlock(64->128), ResBlock(128)                    16x16
  Conv2d(128, 128, stride=2)                           8x8   downsample 2
  ResBlock(128->256), ResBlock(256)                    8x8
  SelfAttention(256)                                   8x8
  Conv2d(256, 256, stride=2)                           4x4   downsample 3
  ResBlock(256) x2                                     4x4
  SelfAttention(256)                                   4x4
  GroupNorm(256) -> SiLU                               4x4
  Conv2d(256, 256, stride=2)          <=== PROBLEM     2x2   downsample 4
  -------------------------------------------------------
  Flatten: 256 x 2 x 2 = 1,024
  fc_mu:     Linear(1024, 64)         <=== PROBLEM     64 dims
  fc_logvar: Linear(1024, 64)                          64 dims

LATENT: z in R^64 (reparameterized, KL-regularized)

DECODER
  fc_dec: Linear(64, 1024) -> SiLU -> reshape(256, 2, 2)
  ConvTranspose2d(256, 256)                            4x4
  ResBlock(256) x2, SelfAttention(256)                 4x4
  ConvTranspose2d(256, 256)                            8x8
  ResBlock(256->128), ResBlock(128), SelfAttention     8x8
  ConvTranspose2d(128, 128)                           16x16
  ResBlock(128->64), ResBlock(64)                     16x16
  ConvTranspose2d(64, 64)                             32x32
  ResBlock(64) x2                                     32x32
  GroupNorm(64) -> SiLU -> Conv2d(64, 3) -> Tanh      output [-1, 1]
```

**Total params**: ~13.5M (VAE) + 663K-1.85M (discriminator)

---

## 2. The Two Stacked Bottlenecks

### Bottleneck 1: Aggressive spatial collapse (4x4 -> 2x2)

The encoder performs **four** stride-2 downsampling steps:

| Stage | Resolution | Feature dims | Info per position |
|-------|-----------|-------------|-------------------|
| Input | 32x32 | 3,072 | — |
| After downsample 1 | 16x16 | 16,384 | Low-level edges |
| After downsample 2 | 8x8 | 16,384 | Mid-level textures |
| After downsample 3 | 4x4 | 4,096 | High-level semantics |
| **After downsample 4** | **2x2** | **1,024** | **Severely compressed** |

The final downsample from 4x4 to 2x2 reduces spatial positions from 16 to 4 — a **4x reduction** in spatial resolution at the point where features are already highly abstract. Each of the 4 remaining positions must encode information about an 8x8 image patch (one quadrant of the image). Fine-grained spatial relationships between the 16 positions at 4x4 are collapsed into just 4 positions.

### Bottleneck 2: FC flatten to vector (1024 -> 64)

```
(256, 2, 2) -> flatten -> 1024 -> Linear -> 64
```

This is a **16:1 compression** through a single linear layer. Combined with KL regularization (beta=0.001, free_bits=0.25), the 64-dim latent is pushed toward a factorized Gaussian, further limiting information capacity.

### Combined effect

```
Input:  3,072 dims (32x32x3)
After spatial collapse:  1,024 dims (256x2x2)  — 3:1 lossy
After FC:                   64 dims             — 16:1 lossy
Total:                      48:1 compression through two destructive stages
```

For comparison, Stable Diffusion's KL-VAE (256x256 images):
- Encodes to (4, 32, 32) = 4,096 latent dims via spatial convolution
- Compression ratio: 196,608 -> 4,096 = 48:1
- But crucially: **no FC flatten** — all compression is via convolution, preserving spatial structure
- PSNR: ~27 dB

The compression ratio is similar, but the mechanism matters: convolutional downsampling preserves local structure at each step, while FC flattening destroys it in one shot.

---

## 3. Evidence: Discriminator Fixes Cannot Help

### VAE v3 results (job 49549763)

| Metric | Value | Target |
|--------|-------|--------|
| PSNR | 19.47 dB | >22 dB |
| SSIM | 0.454 | >0.70 |
| LPIPS | 0.177 | <0.15 |
| Recon FID | 62.07 | <100 |

Discriminator collapsed at epoch ~55 (D_loss = 2.0, adaptive weight exploded to ~8000).

**Critical observation**: Best PSNR (19.40 dB) occurred at epoch 50 — the **end of Stage 1, before adversarial training even started**. Adding the discriminator in Stage 2 degraded PSNR from 19.40 to 18.49.

### VAE v4 results (job 49633525)

| Metric | Value | Change from v3 |
|--------|-------|----------------|
| PSNR | 19.51 dB | +0.04 dB |
| SSIM | 0.458 | +0.004 |
| LPIPS | 0.166 | -0.011 (better) |
| Recon FID | 59.12 | -2.95 (better) |

v4 fixed all 6 discriminator collapse root causes (spectral norm, logistic loss, adaptive weight, D warmup, D optimizer, diagnostic logging). The discriminator is now healthy and functional. **Yet PSNR improved by only 0.04 dB.**

### Conclusion

The reconstruction quality ceiling (~19.5 dB) is set by **information loss at the encoder bottleneck**, not by the discriminator or loss function. No adversarial training can recover information destroyed by the `Linear(1024, 64)` compression. The discriminator can only rearrange pixel-level statistics — it cannot hallucinate spatial details that were never encoded.

---

## 4. Design Constraints from the Quantum Velocity Field

The fix must be evaluated not just for VAE quality, but for how it affects the **quantum velocity field (VF)** performance in the CFM framework.

### Current v6 Quantum VF architecture

```
concat(z_t[32], t_emb[32]) = 64 dims
  -> enc_proj: Linear(64, 256) -> SiLU -> Linear(256, 105)
  -> SU(4) encoding on 8 qubits
  -> QViT butterfly (depth=2)
  -> ANO pairwise k=2 -> 28 observables
  -> vel_head: Linear(28, 256) -> SiLU -> Linear(256, 32) -> velocity
```

### What helps the quantum VF

| Factor | Reason |
|--------|--------|
| **Small latent dim** | Fewer velocity components to predict through 28 observables |
| **Decorrelated latent dims** | Matches all-to-all entanglement structure (no locality bias) |
| **Smooth velocity field** | Shallow quantum circuits approximate smooth functions better |
| **High-quality targets** | Better encoder -> more meaningful velocity -> stronger gradients |

### What hurts the quantum VF

| Factor | Reason |
|--------|--------|
| **Spatial structure in latent** | Quantum circuit has no locality inductive bias; cannot exploit 2D correlations |
| **Large latent dim** | Worsens the 28-observable measurement bottleneck |
| **Complex velocity field** | Shallow circuits struggle with high-frequency velocity components |

### The tension

Better VAE reconstruction requires either spatial latents or larger flat latents. But:
- **Spatial latents** introduce correlations that a classical Conv-based VF could exploit but the quantum VF cannot, potentially widening the quantum-classical gap.
- **Larger flat latents** worsen the measurement bottleneck (28 obs predicting 128+ dims).

The ideal fix improves VAE quality **without changing what the quantum VF sees** — specifically, a flat, decorrelated, small-dimensional latent vector.

---

## 5. Proposed Fix: Stop at 4x4, Reduce Channels via 1x1 Conv

### Core idea

Remove the final stride-2 downsample (4x4 -> 2x2). Instead, reduce channels from 256 to a small number C_z via a 1x1 convolution at 4x4 resolution, then flatten.

This preserves 4x richer spatial information before the flatten-to-latent step, while keeping the flat latent interface identical for the quantum VF.

### Proposed encoder (changes marked with <--)

```
ENCODER (v5 proposal)
  Input: (B, 3, 32, 32)
  Conv2d(3, 64, 3x3)                                  32x32
  ResBlock(64) x2                                      32x32
  Conv2d(64, 64, stride=2)                             16x16
  ResBlock(64->128), ResBlock(128)                     16x16
  Conv2d(128, 128, stride=2)                            8x8
  ResBlock(128->256), ResBlock(256)                     8x8
  SelfAttention(256)                                    8x8
  Conv2d(256, 256, stride=2)                            4x4
  ResBlock(256) x2                                      4x4
  SelfAttention(256)                                    4x4
  GroupNorm(256) -> SiLU
  Conv2d(256, C_z, 1x1)              <-- channel reduction (NOT stride-2)
  -------------------------------------------------------
  Flatten: C_z x 4 x 4
  fc_mu:     Linear(C_z * 16, latent_dim)
  fc_logvar: Linear(C_z * 16, latent_dim)
```

### Proposed decoder (changes marked with <--)

```
DECODER (v5 proposal)
  fc_dec: Linear(latent_dim, C_z * 16) -> SiLU -> reshape(C_z, 4, 4)
  Conv2d(C_z, 256, 1x1)              <-- channel expansion
  ResBlock(256) x2, SelfAttention(256)                  4x4
  ConvTranspose2d(256, 256)                             8x8
  ResBlock(256->128), ResBlock(128), SelfAttention       8x8
  ConvTranspose2d(128, 128)                            16x16
  ResBlock(128->64), ResBlock(64)                      16x16
  ConvTranspose2d(64, 64)                              32x32
  ResBlock(64) x2                                      32x32
  GroupNorm(64) -> SiLU -> Conv2d(64, 3) -> Tanh
```

### Channel reduction parameter C_z

| C_z | Pre-flatten dims | With latent_dim=64 | FC compression ratio | Notes |
|-----|-----------------|-------------------|---------------------|-------|
| 4 | 64 (4x4x4) | 64 -> 64 = 1:1 | **No FC compression** | Best case: FC is identity-like |
| 8 | 128 (8x4x4) | 128 -> 64 = 2:1 | Gentle | Good balance |
| 16 | 256 (16x4x4) | 256 -> 64 = 4:1 | Moderate | Still 4x better than current 16:1 |

**Recommended: C_z = 4, latent_dim = 64**

With C_z = 4 and latent_dim = 64, the `fc_mu` layer is `Linear(64, 64)` — essentially a **learned rotation** rather than a destructive compression. The 1x1 conv handles channel reduction (256 -> 4) at each spatial position independently, preserving the spatial relationships between the 16 positions.

### Why this works for the quantum VF

1. **Same flat latent interface**: The quantum VF still receives a flat 64-dim vector. No code changes needed in `QuantumLatentCFM_v6.py` or any VF architecture.

2. **KL still decorrelates**: With beta=0.001 and free_bits=0.25, the KL regularization still pushes the 64 latent dims toward independent Gaussians. The quantum circuit's all-to-all entanglement remains well-matched to this structure.

3. **Richer information content**: The 1x1 conv at 4x4 is a far gentler compression than the FC at 2x2. Each of the 16 spatial positions contributes meaningfully to the latent, giving the encoder more capacity to represent the image. This means:
   - Better-defined `z_1 = encoder(x).mu`
   - More meaningful target velocity `v* = z_1 - z_0`
   - Stronger, more informative gradients for the quantum VF

4. **No spatial bias in the latent**: Although the pre-flatten features are spatial (4x4), the FC layer + KL regularization produces a flat, decorrelated output. The quantum VF does not need to "know" about spatial structure — it just sees a better-encoded 64-dim vector.

5. **Fair quantum-classical comparison preserved**: Both quantum and classical VF receive the same flat vector. Neither has a structural advantage from spatial correlations.

### What changes vs. current v3/v4

| Component | Current (v3/v4) | Proposed (v5) |
|-----------|-----------------|---------------|
| Encoder final spatial | 2x2 | 4x4 |
| Encoder final downsample | Conv2d stride=2 (256->256) | Conv2d 1x1 (256->C_z) |
| Pre-flatten dim | 1,024 (256x2x2) | 64 (4x4x4) with C_z=4 |
| FC compression | 16:1 (1024->64) | 1:1 (64->64) |
| Decoder start | Linear(64,1024) -> reshape(256,2,2) | Linear(64,64) -> reshape(4,4,4) -> Conv1x1(4,256) |
| Decoder first upsample | ConvTranspose2d 2x2->4x4 | Removed (already at 4x4) |
| VAE param count | ~13.5M | ~12.5M (slightly fewer, no stride-2 conv layer) |
| Latent dim | 64 (flat) | 64 (flat) |
| Quantum VF interface | No change | No change |

---

## 6. Expected Improvement

### Information-theoretic argument

Current path: 256 channels at 4x4 -> stride-2 conv -> 256 channels at 2x2 -> flatten(1024) -> FC(64)
- The stride-2 conv discards 75% of spatial positions (16 -> 4)
- The FC discards 94% of remaining dimensions (1024 -> 64)
- Total: 98.4% of encoder output information discarded

Proposed path: 256 channels at 4x4 -> 1x1 conv(4) -> 4 channels at 4x4 -> flatten(64) -> FC(64)
- The 1x1 conv reduces channels 256 -> 4 at each position independently (98.4% reduction per position, but preserves all 16 spatial positions)
- The FC is 64 -> 64 (lossless linear transform)
- Total: channel compression is **learned and optimized** per spatial position

The key difference: channel reduction via 1x1 conv is a **per-position linear projection** that the network can optimize independently at each spatial location. The current approach forces a single FC layer to simultaneously handle spatial mixing and dimensionality reduction.

### Estimated metrics (based on literature)

| Metric | v3/v4 (2x2 flatten) | v5 estimate (4x4 + 1x1 conv) | Basis |
|--------|---------------------|-------------------------------|-------|
| PSNR | 19.5 dB | 21-23 dB | +1.5-3.5 dB from 4x spatial info |
| SSIM | 0.46 | 0.55-0.65 | Proportional to PSNR gain |
| LPIPS | 0.17 | 0.10-0.14 | Better structure -> lower perceptual dist |
| Recon FID | 59 | 30-50 | Better reconstruction -> closer distribution |

These estimates are conservative. The improvement from eliminating the FC bottleneck should be substantial because the current bottleneck is so severe (16:1 compression through one linear layer).

---

## 7. Implementation Plan

### Files to create/modify

1. **`models/train_vae_v5.py`** (new) — Copy from `train_vae_v4.py`, modify `ResConvVAE_v3` class:
   - Remove final `Conv2d(256, 256, 3, 2, 1)` from encoder
   - Add `Conv2d(256, C_z, 1, 1, 0)` (1x1 channel reduction)
   - Update `flat_dim = C_z * 4 * 4`
   - Update decoder: start from `(C_z, 4, 4)`, add `Conv2d(C_z, 256, 1)` expansion
   - Remove first `ConvTranspose2d(256, 256)` from decoder (no longer needed)
   - Add `--c-z` CLI argument (default=4)

2. **`jobs/run_vae_v5.sh`** (new) — SLURM script, same config as v4

3. **`models/QuantumLatentCFM_vaev5.py`** (new or modify vaev3) — Update VAE class reference

### Minimal code diff (encoder)

```python
# REMOVE from encoder:
#   nn.Conv2d(256, 256, 3, 2, 1, bias=False),               # 2x2

# ADD to encoder (after SelfAttention at 4x4):
    nn.GroupNorm(32, 256), nn.SiLU(inplace=True),
    nn.Conv2d(256, c_z, 1, bias=False),                      # 4x4, channel reduce

# UPDATE:
    flat_dim = c_z * 4 * 4  # e.g., 4 * 16 = 64
```

### Minimal code diff (decoder)

```python
# UPDATE fc_dec and reshape:
    self.fc_dec = nn.Linear(latent_dim, c_z * 4 * 4)
    # In decode():
    h = F.silu(self.fc_dec(z)).view(-1, c_z, 4, 4)

# ADD at start of decoder:
    nn.Conv2d(c_z, 256, 1, bias=False),                      # 4x4, channel expand

# REMOVE from decoder:
#   nn.ConvTranspose2d(256, 256, 4, 2, 1, bias=False),      # was 2x2->4x4
```

---

## 8. Risks and Mitigations

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| C_z=4 too aggressive (underfitting) | Low | Try C_z=8 as fallback (128 pre-flatten dims) |
| KL collapse with easier bottleneck | Medium | Monitor active_dims; increase free_bits if needed |
| Overfitting (less compression = more capacity) | Low | EMA + existing regularization sufficient |
| No PSNR improvement (bottleneck was elsewhere) | Low | All evidence points to FC as the bottleneck |
| Quantum VF performs worse on new latent | Very low | Flat interface unchanged; richer targets should help |

---

## 9. References

1. Rombach et al. (2022). "High-Resolution Image Synthesis with Latent Diffusion Models." CVPR 2022. — Spatial latent VAE (KL-f8), no FC bottleneck, PSNR ~27 dB.
2. Esser et al. (2021). "Taming Transformers for High-Resolution Image Synthesis." CVPR 2021. — VQGAN spatial codebook at 16x16.
3. Razavi et al. (2019). "Generating Diverse High-Fidelity Images with VQ-VAE-2." NeurIPS 2019. — Hierarchical spatial latents.
4. Kingma & Welling (2014). "Auto-Encoding Variational Bayes." ICLR 2014. — Original VAE with FC bottleneck.
5. He et al. (2022). "Masked Autoencoders Are Scalable Vision Learners." CVPR 2022. — Demonstrates spatial structure preservation importance in autoencoders.
