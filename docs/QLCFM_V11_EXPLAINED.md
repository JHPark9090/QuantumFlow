# QLCFM v11: SD3.5 Pretrained VAE with Classical Bottleneck and Quantum Velocity Field

## Overview

QLCFM v11 replaces the custom-trained VAE (v6/v8 architectures used in v9/v10) with a **pretrained Stable Diffusion 3.5 VAE** (Esser et al., 2024), eliminating Phase 1 VAE training entirely. A **classical bottleneck adapter** compresses the high-dimensional SD3.5 latent space (4,096-d for 128x128 images) to 256 dimensions before the quantum velocity field, then projects back after.

The quantum velocity field is **identical to v10**: single SU(16) encoding gate on 4 qubits with Adaptive Non-Local Observables (ANO). Six configurations (v11a--v11f) mirror the v10a--v10f grid.

## Motivation: Why Replace the Custom VAE?

### Limitations of Custom VAE at Low Resolution

Our custom VAEs (v6/v7/v8) were trained from scratch on CIFAR-10 at 32x32 resolution:

| VAE | Latent Dim | PSNR (dB) | SSIM |
|-----|-----------|-----------|------|
| v6  | 64        | 19.47     | 0.454 |
| v7  | 128       | 18.86     | 0.440 |
| v8  | 256       | 18.64     | 0.430 |

These reconstruction scores are modest -- a PSNR of ~19 dB means noticeable blurring and loss of detail. The VAE is a bottleneck: no matter how good the quantum velocity field is, it cannot generate images better than what the VAE can reconstruct.

### SD3.5 VAE Reconstruction Quality

The SD3.5 VAE was pretrained on millions of high-resolution images. At higher resolutions, its reconstruction is dramatically better:

| Dataset | Resolution | PSNR (dB) | SSIM | Latent Dim |
|---------|-----------|-----------|------|-----------|
| CIFAR-10 | 32x32 | 16.08 | 0.80 | 256 |
| CIFAR-10 | 128x128 | 37.33 | 0.998 | 4,096 |
| COCO | 128x128 | 24.79 | 0.96 | 4,096 |

At 32x32, SD3.5 VAE actually performs *worse* than our custom VAEs because its 8x spatial downsampling is too aggressive (32/8 = 4x4 spatial, too small). But at **128x128**, where the spatial latent becomes 16x16, the pretrained VAE provides near-perfect reconstruction (PSNR > 37 dB, SSIM > 0.99 on CIFAR-10). This removes the VAE as a quality bottleneck and lets us evaluate the quantum velocity field in isolation.

### Key Benefits of v11

1. **No Phase 1 training**: SD3.5 VAE is frozen (no finetuning needed), saving 24-48 hours of GPU time per experiment
2. **Higher resolution**: 128x128 instead of 32x32, producing visually meaningful images
3. **Better reconstruction ceiling**: PSNR 37+ dB means the VAE is near-transparent
4. **Standardized comparison**: Using a widely-available pretrained VAE makes results more reproducible
5. **Isolates the quantum contribution**: With a near-perfect VAE, generation quality directly reflects the velocity field's capability

## v10 vs v11: Architectural Comparison

### What Changed

| Aspect | v10 | v11 |
|--------|-----|-----|
| VAE | Custom v6/v8 (trained from scratch) | SD3.5 pretrained (frozen) |
| Image resolution | 32x32 | 128x128 |
| VAE latent channels | 4 (v6) or 16 (v8) | 16 (SD3.5 standard) |
| Spatial downsampling | 4x (v6) or 8x (v8) | 8x (SD3.5 standard) |
| Flat latent dim | 64, 128, or 256 | 4,096 (= 16 x 16 x 16) |
| Phase 1 training | Required (100-300 epochs) | None |
| Bottleneck adapter | None | Linear(4096→256) + Linear(256→4096) |
| Quantum circuit | Identical | Identical |
| Training improvements | v9 (logit-normal, midpoint, EMA) | v9 (logit-normal, midpoint, EMA) |
| Model file | `QuantumLatentCFM_v10.py` | `QuantumLatentCFM_v11.py` |

### What Stayed the Same

- **Quantum velocity field**: Single SU(16) gate on 4 qubits, 255 generators, ANO measurement
- **Time conditioning**: Concat or additive, same architecture
- **ANO types**: Pairwise k=2, pairwise k=3, global k=4
- **Dual optimizer**: lr=0.001 for circuit, lr=0.1 for ANO Hermitians
- **ODE solver**: Midpoint with 50 steps
- **EMA**: Decay 0.999
- **Evaluation**: FID and IS on 1,024 samples

## The Classical Bottleneck Adapter

### Problem: 4,096-d Latent Is Too Large for 4 Qubits

The SD3.5 VAE produces a 4,096-dimensional latent vector (16 channels x 16x16 spatial for 128x128 images). But our quantum circuit's SU(16) gate has only 255 parameters. Feeding 4,096 dimensions directly into a 255-parameter encoding would require a 16:1 compression ratio in the `enc_proj` layer, which is far too aggressive (v10 showed best results near 1:1).

### Solution: Linear Bottleneck

A simple learned linear projection compresses and expands the latent space around the quantum circuit:

```
SD3.5 Latent (4096-d)
    │
    ▼
┌────────────────────┐
│  bottleneck_in     │  Linear(4096 → 256)
│  (learnable)       │
└────────────────────┘
    │
    ▼
Working Space (256-d)   ← Same scale as v10 lat256
    │
    ├── [+ or concat] time embedding (256-d)
    │
    ▼
┌────────────────────┐
│  enc_proj          │  Linear(256 or 512 → 255)
│  (learnable)       │
└────────────────────┘
    │
    ▼
┌────────────────────┐
│  SU(16) Gate       │  4 qubits, 255 generators
│  (quantum)         │
└────────────────────┘
    │
    ▼
┌────────────────────┐
│  ANO Measurement   │  Pairwise or Global Hermitians
│  (learnable)       │
└────────────────────┘
    │
    ▼
┌────────────────────┐
│  vel_head          │  Linear(n_obs → 256) + SiLU + Linear(→ 256)
│  (learnable)       │
└────────────────────┘
    │
    ▼
Working Space (256-d)
    │
    ▼
┌────────────────────┐
│  bottleneck_out    │  Linear(256 → 4096)
│  (learnable)       │
└────────────────────┘
    │
    ▼
Predicted Velocity (4096-d)
```

### Why Linear Bottleneck Works

The bottleneck adapter is conceptually similar to the low-rank projections used in LoRA (Hu et al., 2022): the velocity field learns to predict updates in a compressed subspace, then projects back to the full latent space. The key insight is that the **meaningful signal in the velocity field may be low-rank** -- not all 4,096 dimensions need independent velocity predictions. A 256-d bottleneck can capture the dominant modes of variation.

The bottleneck layers are **trainable** (not frozen), so they learn the optimal compression/expansion jointly with the quantum circuit during Phase 2 training.

### enc_proj Ratios with Bottleneck

After the bottleneck, the working dimension is 256 regardless of the SD3.5 latent size. This matches the v10 lat256 setup exactly:

| Config | Time Cond. | Input to enc_proj | SU(16) Params | Ratio |
|--------|-----------|-------------------|---------------|-------|
| v11a/e | concat    | 256 + 256 = 512   | 255           | 2.01:1 |
| v11b/f | additive  | 256               | 255           | 1.00:1 |
| v11c   | concat    | 256 + 256 = 512   | 255           | 2.01:1 |
| v11d   | additive  | 256               | 255           | 1.00:1 |

The additive variants (v11b, v11d, v11f) achieve the ideal 1:1 ratio, matching the sweet spot found in v10.

## SD3.5 VAE Details

### Architecture

The SD3.5 VAE (`AutoencoderKL` from diffusers) has:

- **Parameters**: ~84 million (frozen in v11)
- **Latent channels**: 16 (vs 4 in SD 1.5/2.x)
- **Spatial downsampling**: 8x (encoder) / 8x (decoder)
- **Input range**: [-1, 1] (standard normalization)

For a 128x128 input image:
```
Input:   (batch, 3, 128, 128)
Latent:  (batch, 16, 16, 16)   → flat: (batch, 4096)
Output:  (batch, 3, 128, 128)
```

### Loading and Freezing

```python
from diffusers import AutoencoderKL

vae = AutoencoderKL.from_pretrained(
    "stabilityai/stable-diffusion-3.5-large",
    subfolder="vae", torch_dtype=torch.float32)

# Freeze all parameters
for p in vae.parameters():
    p.requires_grad = False
vae.eval()
```

The `SD3VAEWrapper` class in v11 handles encoding (image → flat latent) and decoding (flat latent → image) with `torch.no_grad()` to prevent gradient computation through the frozen VAE.

### Why Not Finetune the VAE?

Finetuning 84M parameters would overwhelm the quantum velocity field's signal (which has ~500-2000 trainable parameters). The pretrained VAE is already near-optimal for reconstruction; the research question is whether the quantum velocity field can learn meaningful flow dynamics in the pretrained latent space.

## v11 Configurations (v11a--v11f)

The six v11 variants mirror the v10a--v10f grid exactly:

|                  | Pairwise ANO (k=2) | Triple ANO (k=3) | Global ANO (k=4) |
|------------------|---------------------|-------------------|-------------------|
| **Concat time**  | **v11a**            | **v11e**          | **v11c**          |
| **Additive time**| **v11b**            | **v11f**          | **v11d**          |

### Configuration Details

**v11a -- Concat + Pairwise ANO (k=2):**
- Time: z_combined = [z_bottleneck, t_emb], dim = 512
- ANO: 6 pairwise 4x4 Hermitians (96 params)
- enc_proj ratio: 2.01:1

**v11b -- Additive + Pairwise ANO (k=2):**
- Time: z_combined = z_bottleneck + time_mlp(t), dim = 256
- ANO: 6 pairwise 4x4 Hermitians (96 params)
- enc_proj ratio: 1.00:1

**v11c -- Concat + Global ANO (k=4):**
- Time: z_combined = [z_bottleneck, t_emb], dim = 512
- ANO: 6 global 16x16 Hermitians (1,536 params)
- enc_proj ratio: 2.01:1

**v11d -- Additive + Global ANO (k=4):**
- Time: z_combined = z_bottleneck + time_mlp(t), dim = 256
- ANO: 6 global 16x16 Hermitians (1,536 params)
- enc_proj ratio: 1.00:1

**v11e -- Concat + Triple ANO (k=3):**
- Time: z_combined = [z_bottleneck, t_emb], dim = 512
- ANO: 4 triple 8x8 Hermitians (256 params)
- enc_proj ratio: 2.01:1

**v11f -- Additive + Triple ANO (k=3):**
- Time: z_combined = z_bottleneck + time_mlp(t), dim = 256
- ANO: 4 triple 8x8 Hermitians (256 params)
- enc_proj ratio: 1.00:1

### Parameter Counts

| Component | Params | Notes |
|-----------|--------|-------|
| SD3.5 VAE | ~84M | Frozen (not trained) |
| Bottleneck in | 4096 x 256 + 256 = 1,048,832 | Linear(4096→256) |
| Bottleneck out | 256 x 4096 + 4096 = 1,052,672 | Linear(256→4096) |
| enc_proj | ~131K (concat) or ~66K (additive) | 2-layer MLP |
| SU(16) encoding | 0 (data-dependent) | Params come from enc_proj |
| ANO Hermitians | 96 / 256 / 1,536 | k=2 / k=3 / k=4 |
| vel_head | ~66K--131K | MLP(n_obs → 256 → 256) |
| Time MLP | ~131K | 2-layer MLP |
| **Total trainable** | **~2.4M--2.5M** | Bottleneck dominates |

Note: The vast majority of trainable parameters are in the bottleneck layers (~2.1M), not the quantum circuit. The quantum circuit's contribution is in the *type* of processing (quantum expectation values via SU(16) + ANO), not parameter count.

## Data Pipeline

### v10 Pipeline (32x32 CIFAR-10)
```
CIFAR-10 (32x32)
    │
    ▼
Custom VAE Encoder → Latent (64/128/256-d)
    │
    ▼
Quantum Velocity Field (no bottleneck)
    │
    ▼
Custom VAE Decoder → Generated Image (32x32)
```

### v11 Pipeline (128x128 CIFAR-10)
```
CIFAR-10 (32x32) ─── Bicubic Upscale ──→ CIFAR-10 (128x128)
    │
    ▼
SD3.5 VAE Encoder (frozen) → Latent (4096-d)
    │
    ▼
Bottleneck In: Linear(4096→256)
    │
    ▼
Quantum Velocity Field (working dim = 256)
    │
    ▼
Bottleneck Out: Linear(256→4096)
    │
    ▼
SD3.5 VAE Decoder (frozen) → Generated Image (128x128)
```

### Image Preprocessing

Images are normalized to [-1, 1] for SD3.5 VAE compatibility:
```python
X = X * 2.0 - 1.0  # [0,1] → [-1,1]
```

For CIFAR-10, images are bicubic-upscaled from 32x32 to 128x128 before encoding. While this doesn't add information content, it puts images in a resolution range where SD3.5 VAE's encoder/decoder architecture operates optimally.

## Training Details

All v11 models inherit training improvements from v9:

1. **Logit-Normal Timestep Sampling** (std=1.0): Concentrates training on mid-range timesteps
2. **Midpoint ODE Solver**: 2nd-order accuracy, 50 steps (100 VF evaluations)
3. **VF EMA** (decay=0.999): Smoothed weights for validation and generation
4. **Dual Optimizer**: Adam with lr=0.001 (circuit + bottleneck) and lr=0.1 (ANO Hermitians)
5. **Cosine Annealing LR Scheduler**: For both optimizer groups

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Epochs | 200 |
| Batch size | 64 |
| n_train | 10,000 |
| n_valtest | 2,000 |
| Seed | 2025 |
| ODE steps | 50 (midpoint) |
| Evaluation samples | 1,024 |

### Resumption

Training can be resumed from checkpoints:
```bash
PREV_JOB_ID=XXXXX sbatch jobs/run_qlcfm_v11a.sh
```

## Expected Outcomes and Hypothesis

### What We Expect to Learn

1. **Does a pretrained VAE improve FID?** v10's best FID was ~106 on CIFAR-10 32x32. With SD3.5 VAE at 128x128 (near-perfect reconstruction), the VAE is no longer a bottleneck. If FID improves significantly, it confirms the custom VAE was the limiting factor. If FID stays similar, the quantum velocity field itself is the bottleneck.

2. **Does higher resolution help?** 128x128 images have more structure than 32x32, which may be easier for the velocity field to learn (smoother distributions in latent space).

3. **Does the bottleneck lose information?** The 4096→256 compression is 16:1. If v11 results are significantly worse than v10 despite the better VAE, the bottleneck is too aggressive and we should try larger working dimensions (512 or 1024).

4. **Which ANO type benefits most?** Global ANO (k=4) has the most parameters and may benefit more from the higher-quality latent space, or it may overfit more easily.

## File Index

| File | Description |
|------|-------------|
| `models/QuantumLatentCFM_v11.py` | Main model: SD3.5 VAE + bottleneck + quantum VF |
| `jobs/run_qlcfm_v11a.sh` | SLURM: concat + pairwise k=2 |
| `jobs/run_qlcfm_v11b.sh` | SLURM: additive + pairwise k=2 |
| `jobs/run_qlcfm_v11c.sh` | SLURM: concat + global k=4 |
| `jobs/run_qlcfm_v11d.sh` | SLURM: additive + global k=4 |
| `jobs/run_qlcfm_v11e.sh` | SLURM: concat + pairwise k=3 |
| `jobs/run_qlcfm_v11f.sh` | SLURM: additive + pairwise k=3 |
| `test_sd3_vae_recon.py` | SD3.5 VAE reconstruction quality test |
| `docs/QLCFM_V10_EXPLAINED.md` | v10 architecture documentation |

## References

- Esser, P. et al. (2024). "Scaling Rectified Flow Transformers for High-Resolution Image Synthesis." *ICML 2024*. (SD3)
- Lipman, Y. et al. (2023). "Flow Matching for Generative Modeling." *ICLR 2023*.
- Wiersema, R. et al. (2024). "Here comes the SU(N): multivariate quantum gates and gradients." *Quantum*, 8, 1275.
- Lin, S. et al. (2025). "Adaptive Non-local Observable on Quantum Neural Networks." *IEEE QCE 2025*.
- Chen, Y. et al. (2025). "Learning to Measure Quantum Neural Networks." *ICASSP 2025 Workshop*.
- Cherrat, E.A. et al. (2024). "Quantum Vision Transformers." *Quantum*, 8, 1265.
- Hu, E.J. et al. (2022). "LoRA: Low-Rank Adaptation of Large Language Models." *ICLR 2022*.
