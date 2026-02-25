# QLCFM Experiment Results Summary

All experiments on CIFAR-10 (50K train, 10K val/test, 32x32x3 images).

---

## Table of Contents

1. [Results Overview](#1-results-overview)
2. [Classical Baselines](#2-classical-baselines)
3. [Quantum Models v1-v5 (Uncontrolled)](#3-quantum-models-v1-v5-uncontrolled)
4. [Quantum Models v6-v7 (Controlled)](#4-quantum-models-v6-v7-controlled)
5. [Phase 1 VAE Results](#5-phase-1-vae-results)
6. [Phase 2 CFM Results](#6-phase-2-cfm-results)
7. [Known Confounds in v1-v5](#7-known-confounds-in-v1-v5)
8. [Key Findings](#8-key-findings)

---

## 1. Results Overview

### Generation Quality (FID / IS)

Only models that completed all 200 CFM epochs and ran FID/IS evaluation are shown.

| Model | FID (lower=better) | IS (higher=better) | CFM Val MSE |
|-------|-------------------|--------------------|-------------|
| Classical-A | 421.70 | 1.35 +/- 0.02 | 0.044 |
| Classical-B | **244.22** | **2.33 +/- 0.13** | 0.315 |
| Classical-C | 297.42 | 2.17 +/- 0.14 | 0.436 |
| **v6 (1x8q)** | **287.43** | **2.34 +/- 0.19** | 0.485 |
| v7 (8x8q) | — | — | 0.475* |

\* v7 at epoch 18/200, still running. Will require resume to complete.

**Note**: FID/IS computed with 1,024 samples (insufficient for publication; 10K-50K needed).
CFM MSE is not comparable across different VAEs — see [Section 7](#7-known-confounds-in-v1-v5).

---

## 2. Classical Baselines

### Classical-A

| Property | Value |
|----------|-------|
| Job ID | 48763211 |
| Codebase | `QuantumLatentCFM.py` (v1-v4) |
| VAE | Legacy ConvVAE, 530K params |
| Latent dim | 32 |
| Perceptual loss | No |
| Time embedding | 64-dim sinusoidal + time_mlp |
| Velocity field | MLP 96→256→256→256→32, SiLU |
| VF params | 172,960 |
| CFM Val MSE | **0.044** |
| FID / IS | 421.70 / 1.35 |

### Classical-B

| Property | Value |
|----------|-------|
| Job ID | 48804337 |
| Codebase | `QuantumLatentCFM.py` (v1-v4) |
| VAE | ResConvVAE, 10.5M params |
| Latent dim | **256** |
| Perceptual loss | Yes (lambda=0.1) |
| Time embedding | 64-dim sinusoidal + time_mlp |
| Velocity field | MLP 320→256→256→256→256, SiLU |
| VF params | 287,872 |
| CFM Val MSE | 0.315 |
| FID / IS | **244.22 / 2.33** |

### Classical-C

| Property | Value |
|----------|-------|
| Job ID | 49235600 |
| Codebase | `QuantumLatentCFM_v6.py` |
| VAE | ResConvVAE, 9.85M params |
| Latent dim | 32 |
| Perceptual loss | Yes (lambda=0.1) |
| Time embedding | 32-dim sinusoidal + time_mlp |
| Velocity field | MLP 64→256→256→256→32, SiLU |
| VF params | 158,560 |
| CFM Val MSE | 0.436 |
| FID / IS | 297.42 / 2.17 |

---

## 3. Quantum Models v1-v5 (Uncontrolled)

These models were developed iteratively with different VAE architectures, latent dims,
and time embedding schemes. **Cross-version MSE comparisons are confounded** — see
[Section 7](#7-known-confounds-in-v1-v5).

### v1-QCNN (8q, sliding k=2)

| Property | Value |
|----------|-------|
| Job ID | 48639698 |
| Codebase | `QuantumLatentCFM.py` |
| VAE | Legacy ConvVAE, 1.12M params |
| Latent dim | 128 |
| Perceptual loss | No |
| Time embedding | Sinusoidal(128), **no time_mlp** |
| Qubits | 8 |
| Encoding | SU(4) on pairs, 2 blocks, 210 params |
| VQC | QCNN, depth=2 |
| Observables | Sliding k=2, 7 obs (ratio = 7/128 = 0.05) |
| VF params | 155,082 |
| CFM Val MSE | 0.961 |
| Epochs completed | 200 |

### v1-QViT (8q, sliding k=2)

| Property | Value |
|----------|-------|
| Job ID | 48639699 |
| Codebase | `QuantumLatentCFM.py` |
| VAE | Legacy ConvVAE, 1.12M params (shared with v1-QCNN) |
| Latent dim | 128 |
| Perceptual loss | No |
| Time embedding | Sinusoidal(128), **no time_mlp** |
| Qubits | 8 |
| Encoding | SU(4) on pairs, 2 blocks, 210 params |
| VQC | QViT butterfly, depth=2 |
| Observables | Sliding k=2, 7 obs (ratio = 7/128 = 0.05) |
| VF params | 154,842 |
| CFM Val MSE | 0.956 |
| Epochs completed | 200 |

**v1 analysis**: Both v1 variants performed poorly (MSE ~0.96, near random).
Root cause: 128-dim latent with only 7 observables creates a 128:7 bottleneck
(ratio 0.05). The quantum circuit contributes almost nothing.

### v2 (12q, sliding k=2)

| Property | Value |
|----------|-------|
| Job ID | 48722700 |
| Codebase | `QuantumLatentCFM.py` |
| VAE | Legacy ConvVAE, 530K params (shared with Classical-A) |
| Latent dim | 32 |
| Perceptual loss | No |
| Time embedding | Sinusoidal(32), **no time_mlp** |
| Qubits | 12 |
| Encoding | SU(4) on pairs, 2 blocks, 330 params |
| VQC | QViT butterfly, depth=2 |
| Observables | Sliding k=2, 11 obs (ratio = 11/32 = 0.34) |
| VF params | 113,402 |
| CFM Val MSE | **0.532** |
| Epochs completed | 200 |

**v2 analysis**: Best quantum model from uncontrolled experiments. Shares identical
VAE with Classical-A, making this the only fair quantum-vs-classical comparison in
v1-v5. Gap: 0.044 (Classical-A) vs 0.532 (v2) = 12x.

### v3-pw2 (12q, pairwise k=2)

| Property | Value |
|----------|-------|
| Job ID | 48806160 |
| Codebase | `QuantumLatentCFM.py` |
| VAE | Legacy ConvVAE, 530K params |
| Latent dim | 32 |
| Perceptual loss | Yes (lambda=0.1) |
| Time embedding | Sinusoidal(32), **no time_mlp** |
| Qubits | 12 |
| Encoding | SU(4) on pairs, 2 blocks, 330 params |
| VQC | QViT butterfly, depth=2 |
| Observables | Pairwise k=2, 66 obs (ratio = 66/32 = 2.06) |
| VF params | 128,362 |
| CFM Val MSE | 0.647 |
| Epochs completed | 102 (timed out) |

### v3-pw3 (12q, pairwise k=3)

| Property | Value |
|----------|-------|
| Job ID | 48807333 |
| Codebase | `QuantumLatentCFM.py` |
| VAE | Legacy ConvVAE, 530K params |
| Latent dim | 32 |
| Perceptual loss | Yes (lambda=0.1) |
| Time embedding | Sinusoidal(32), **no time_mlp** |
| Qubits | 12 |
| Encoding | SU(4) on pairs, 2 blocks, 330 params |
| VQC | QViT butterfly, depth=2 |
| Observables | Pairwise k=3, 220 obs (ratio = 220/32 = 6.88) |
| VF params | 180,810 |
| CFM Val MSE | 0.685 |
| Epochs completed | 27 (timed out) |

### v3-sl3 (12q, sliding k=3)

| Property | Value |
|----------|-------|
| Job ID | 48807335 |
| Codebase | `QuantumLatentCFM.py` |
| VAE | Legacy ConvVAE, 530K params |
| Latent dim | 32 |
| Perceptual loss | Yes (lambda=0.1) |
| Time embedding | Sinusoidal(32), **no time_mlp** |
| Qubits | 12 |
| Encoding | SU(4) on pairs, 2 blocks, 330 params |
| VQC | QViT butterfly, depth=2 |
| Observables | Sliding k=3, 10 obs (ratio = 10/32 = 0.31) |
| VF params | 113,610 |
| CFM Val MSE | 0.983 |
| Epochs completed | 115 (timed out) |

**v3 analysis**: Increasing observable count (pairwise) or locality (k=3) did not
improve over v2. pw2 (ratio 2.06) worse than v2 (ratio 0.34). Higher ratio
does not guarantee better performance — architecture quality matters more.

### v4-L2 (12q, pairwise k=2, data re-uploading x2)

| Property | Value |
|----------|-------|
| Job ID | 48872407 |
| Codebase | `QuantumLatentCFM.py` |
| VAE | Legacy ConvVAE, 530K params |
| Latent dim | 32 |
| Perceptual loss | Yes (lambda=0.1) |
| Time embedding | Sinusoidal(32), **no time_mlp** |
| Qubits | 12 |
| Encoding | SU(4) on pairs, 2 blocks, 330 params, **re-upload x2** |
| VQC | QViT butterfly, depth=2 |
| Observables | Pairwise k=2, 66 obs (ratio = 2.06) |
| VF params | 128,362 |
| CFM Val MSE | 0.619 |
| Epochs completed | 195 (timed out) |

### v4-L4 (12q, pairwise k=2, data re-uploading x4)

| Property | Value |
|----------|-------|
| Job ID | 48872408 |
| Codebase | `QuantumLatentCFM.py` |
| VAE | Legacy ConvVAE, 530K params |
| Latent dim | 32 |
| Perceptual loss | Yes (lambda=0.1) |
| Time embedding | Sinusoidal(32), **no time_mlp** |
| Qubits | 12 |
| Encoding | SU(4) on pairs, 2 blocks, 660 params, **re-upload x4** |
| VQC | QViT butterfly, depth=2 |
| Observables | Pairwise k=2, 66 obs (ratio = 2.06) |
| VF params | 213,652 |
| CFM Val MSE | 0.637 |
| Epochs completed | 121 (timed out) |

**v4 analysis**: Data re-uploading gave marginal improvement over v3-pw2 (0.619 vs
0.647 for L2), but L4 was worse than L2, suggesting diminishing returns.

### v5 (16x8q, split input)

| Property | Value |
|----------|-------|
| Job ID | 48987378 |
| Codebase | `QuantumLatentCFM_v5.py` |
| VAE | Legacy ConvVAE, 727K params |
| Latent dim | **64** |
| Perceptual loss | Yes (lambda=0.1) |
| Time embedding | Sinusoidal(64), **no time_mlp** |
| Qubits | 16 circuits x 8 qubits |
| Encoding | SU(4) on pairs, 1 block/circuit, 105 params/circuit |
| VQC | QViT butterfly, depth=2 |
| Observables | Pairwise k=2, 28 obs/circuit, 448 total (ratio = 448/64 = 7.00) |
| Input | **Split**: 128 dims split into 16 chunks of ~8 dims |
| VF params | 55,632 |
| CFM Val MSE | 0.835 |
| Epochs completed | 21 (timed out) |

**v5 analysis**: Worst quantum result despite highest obs ratio (7.0). Root cause:
input splitting fragmented z_t and t_emb across circuits, so no single circuit saw
the complete state or time information. Also used different latent_dim (64) and
legacy VAE, introducing multiple confounds.

---

## 4. Quantum Models v6-v7 (Controlled)

v6 and v7 are a controlled experiment: **all components except the velocity field core
are identical** to Classical-C. Same VAE (ResConvVAE 9.85M), same latent_dim (32),
same time_embed_dim (32) with time_mlp, same data pipeline, same training loop.

### v6 (1x8q, shared input)

| Property | Value |
|----------|-------|
| Job ID | 49235601 |
| Codebase | `QuantumLatentCFM_v6.py` |
| VAE | ResConvVAE, 9.85M params (shared with Classical-C) |
| Latent dim | 32 |
| Perceptual loss | Yes (lambda=0.1) |
| Time embedding | 32-dim sinusoidal + time_mlp (matching Classical-C) |
| Qubits | 1 circuit x 8 qubits |
| Encoding | SU(4) on pairs, 1 block, 105 params |
| enc_proj | Linear(64,256) → SiLU → Linear(256,105) |
| VQC | QViT butterfly, depth=2, 288 params |
| Observables | Pairwise k=2, 28 obs (ratio = 28/32 = 0.88) |
| vel_head | Linear(28,256) → SiLU → Linear(256,32) |
| Input | Shared: concat(z_t[32], t_emb[32]) = 64 dims |
| VF params | 62,121 (circuit=61,673, ANO=448) |
| CFM Val MSE | **0.485** |
| FID / IS | **287.43 / 2.34** |
| Epochs completed | 200 |
| Time/epoch | ~500s |

### v7 (8x8q, shared input)

| Property | Value |
|----------|-------|
| Job ID | 49235602 |
| Codebase | `QuantumLatentCFM_v6.py` |
| VAE | ResConvVAE, 9.85M params (shared with Classical-C) |
| Latent dim | 32 |
| Perceptual loss | Yes (lambda=0.1) |
| Time embedding | 32-dim sinusoidal + time_mlp (matching Classical-C) |
| Qubits | 8 circuits x 8 qubits |
| Encoding | SU(4) on pairs, 1 block/circuit, 105 params/circuit |
| enc_proj | 8x [Linear(64,256) → SiLU → Linear(256,105)] |
| VQC | QViT butterfly, depth=2, 288 params/circuit |
| Observables | Pairwise k=2, 28 obs/circuit, 224 total (ratio = 224/32 = 7.00) |
| vel_head | Linear(224,256) → SiLU → Linear(256,32) |
| Input | Shared: concat(z_t[32], t_emb[32]) = 64 dims to ALL circuits |
| VF params | 422,824 (circuit=419,240, ANO=3,584) |
| CFM Val MSE | **0.475** (at epoch 18, still running) |
| FID / IS | Pending |
| Epochs completed | 18/200 (running, will need resume) |
| Time/epoch | ~4,150s |

---

## 5. Phase 1 VAE Results

### Across all experiments (final epoch)

| Model | VAE Arch | VAE Params | Latent | Perc. Loss | Val Recon | Val KL | Val PSNR | Val SSIM | Val LPIPS |
|-------|----------|-----------|--------|------------|-----------|--------|----------|----------|-----------|
| Classical-A | Legacy | 530K | 32 | No | 0.0516 | 0.0131 | — | — | — |
| Classical-B | ResConv | 10.5M | 256 | Yes | 0.0283 | 0.0280 | — | — | — |
| v1 (QCNN/QViT) | Legacy | 1.12M | 128 | No | 0.0380 | 0.0176 | — | — | — |
| v2 | Legacy | 530K | 32 | No | 0.0516 | 0.0131 | — | — | — |
| v3-pw2/v4 | Legacy | 530K | 32 | Yes | 0.0519 | 0.0135 | — | — | — |
| v5 | Legacy | 727K | 64 | Yes | 0.0453 | 0.0165 | — | — | — |
| **Classical-C** | ResConv | 9.85M | 32 | Yes | 0.0519 | 0.0138 | **12.85** | **0.187** | **0.466** |
| **v6** | ResConv | 9.85M | 32 | Yes | 0.0519 | 0.0138 | **12.85** | **0.187** | **0.466** |
| **v7** | ResConv | 9.85M | 32 | Yes | 0.0519 | 0.0138 | **12.85** | **0.187** | **0.466** |

PSNR, SSIM, and LPIPS were only implemented in the v6 codebase.

**Key observation**: Classical-C, v6, and v7 share **identical** VAE results, confirming
the controlled experiment is set up correctly. Any Phase 2 difference is attributable
solely to the velocity field.

---

## 6. Phase 2 CFM Results

### All models ranked by Val MSE

| Rank | Model | Qubits | Latent | VF Params | Obs Ratio | Val MSE | Epochs | FID | IS |
|------|-------|--------|--------|-----------|-----------|---------|--------|-----|-----|
| 1 | Classical-A | — | 32 | 173K | — | **0.044** | 200 | 421.70 | 1.35 |
| 2 | Classical-B | — | 256 | 288K | — | 0.315 | 200 | **244.22** | **2.33** |
| 3 | Classical-C | — | 32 | 159K | — | 0.436 | 200 | 297.42 | 2.17 |
| 4 | v7 (8x8q) | 8x8 | 32 | 423K | 7.00 | 0.475* | 18* | — | — |
| 5 | v6 (1x8q) | 1x8 | 32 | 62K | 0.88 | 0.485 | 200 | **287.43** | **2.34** |
| 6 | v2 (12q) | 12 | 32 | 113K | 0.34 | 0.532 | 200 | — | — |
| 7 | v4-L2 | 12 | 32 | 128K | 2.06 | 0.619 | 195 | — | — |
| 8 | v4-L4 | 12 | 32 | 214K | 2.06 | 0.637 | 121 | — | — |
| 9 | v3-pw2 | 12 | 32 | 128K | 2.06 | 0.647 | 102 | — | — |
| 10 | v3-pw3 | 12 | 32 | 181K | 6.88 | 0.685 | 27 | — | — |
| 11 | v5 (16x8q) | 16x8 | 64 | 56K | 7.00 | 0.835 | 21 | — | — |
| 12 | v1-QViT | 8 | 128 | 155K | 0.05 | 0.956 | 200 | — | — |
| 13 | v1-QCNN | 8 | 128 | 155K | 0.05 | 0.961 | 200 | — | — |
| 14 | v3-sl3 | 12 | 32 | 114K | 0.31 | 0.983 | 115 | — | — |

\* v7 still running at epoch 18/200.

### Controlled comparison (v6 codebase, same VAE)

| Model | VF Params | Val MSE | FID | IS | s/epoch |
|-------|-----------|---------|-----|-----|---------|
| Classical-C | 158,560 | 0.436 | 297.42 | 2.17 | 5.9 |
| **v6 (1x8q)** | 62,121 | 0.485 | **287.43** | **2.34** | 500 |
| v7 (8x8q) | 422,824 | 0.475* | — | — | 4,150 |

---

## 7. Known Confounds in v1-v5

Comparing across experiment versions is unreliable due to multiple uncontrolled
variables. The table below shows which settings differ:

| Confound | Classical-A | Classical-B | v1 | v2 | v3/v4 | v5 | v6/v7 + Classical-C |
|----------|------------|------------|-----|-----|-------|-----|-------------------|
| Codebase | v1-v4 | v1-v4 | v1-v4 | v1-v4 | v1-v4 | v5 | **v6** |
| VAE arch | Legacy 530K | ResConv 10.5M | Legacy 1.12M | Legacy 530K | Legacy 530K | Legacy 727K | **ResConv 9.85M** |
| Latent dim | 32 | **256** | **128** | 32 | 32 | **64** | **32** |
| Perceptual loss | No | Yes | No | No | Yes | Yes | **Yes** |
| Time embedding | 64 + mlp | 64 + mlp | sin only | **sin only** | **sin only** | **sin only** | **32 + mlp** |
| Input sharing | — | — | full | full | full | **split** | **shared** |
| enc_proj nonlin. | — | — | No | No | No | No | **Yes (SiLU)** |
| vel_head nonlin. | — | — | No | No | No | No | **Yes (SiLU)** |

**Bold** indicates the value that differs from the v6/v7 controlled setup.

The only fair comparisons within the old experiments:
- **Classical-A vs v2**: Same legacy VAE, same latent_dim=32. Difference: time_mlp + MLP vs no time_mlp + quantum.
- **v3-pw2 vs v3-pw3 vs v3-sl3**: Same VAE, same everything except observable scheme.
- **v4-L2 vs v4-L4**: Same everything except re-upload depth.

---

## 8. Key Findings

### Finding 1: v6 quantum achieves better FID/IS than Classical-C despite higher MSE

| | Classical-C | v6 (1x8q) |
|--|------------|-----------|
| Val MSE | 0.436 (better) | 0.485 |
| FID | 297.42 | **287.43** (better) |
| IS | 2.17 | **2.34** (better) |

This is a controlled comparison (identical VAE, data, time embedding). The quantum
velocity field produces better image quality despite higher flow matching MSE.
This suggests the quantum circuit learns qualitatively different flow trajectories
that translate to better perceptual quality.

### Finding 2: CFM MSE is not comparable across different VAEs

Classical-A (MSE 0.044) appears far better than Classical-C (MSE 0.436), but:
- Classical-A FID = 421.70 (poor), Classical-C FID = 297.42 (better)
- The ResConvVAE creates a richer latent space that is harder to model but produces
  better images

**Lesson**: Always compare FID/IS (final image quality), not CFM MSE, when VAEs differ.

### Finding 3: obs_ratio alone does not predict performance

| Model | Obs Ratio | Val MSE |
|-------|-----------|---------|
| v1 | 0.05 | 0.96 (worst) |
| v3-sl3 | 0.31 | 0.98 (near worst) |
| v2 | 0.34 | 0.53 (best old quantum) |
| v6 | 0.88 | 0.49 (best quantum) |
| v3-pw2 | 2.06 | 0.65 |
| v3-pw3 | 6.88 | 0.69 |
| v5 | 7.00 | 0.84 |
| v7 | 7.00 | 0.48* |

v5 (ratio 7.0, split input) was worst; v7 (ratio 7.0, shared input) is best.
Architecture quality (nonlinear layers, shared input, time conditioning) dominates
over raw observable count.

### Finding 4: Controlled experiment fixes resolved most confounds

The v6 codebase eliminates 7 confounding variables from v1-v5:
1. Time embedding: all models now use sinusoidal + time_mlp
2. VAE architecture: all use ResConvVAE
3. Latent dimension: all use 32
4. Perceptual loss: all use VGG (lambda=0.1)
5. Input sharing: all circuits see full input (no split)
6. enc_proj: uses SiLU nonlinearity
7. vel_head: uses SiLU nonlinearity

### Finding 5: v7 converges faster per epoch than v6

At epoch 18, v7 (val MSE 0.475) already surpasses v6's final result (val MSE 0.485).
However, v7 is 8.3x slower per epoch (4,150s vs 500s), so wall-clock efficiency is
lower. v7 needs resume to complete full 200 epochs (estimated 8.8 days total).

---

## Appendix: File Locations

### Logs (SLURM stdout/stderr)
```
/pscratch/sd/j/junghoon/QuantumFlow/logs/
```

### Checkpoints
```
/pscratch/sd/j/junghoon/QuantumFlow/checkpoints/
  ckpt_vae_<job_id>.pt       — VAE training checkpoint (resume)
  ckpt_cfm_<job_id>.pt       — CFM training checkpoint (resume)
  weights_vae_<job_id>.pt    — Final VAE weights
  weights_cfm_<job_id>.pt    — Final CFM velocity field weights
```

### Results
```
/pscratch/sd/j/junghoon/QuantumFlow/results/
  log_vae_<job_id>.csv        — Per-epoch VAE metrics
  log_cfm_<job_id>.csv        — Per-epoch CFM metrics
  metrics_<job_id>.json       — FID and IS
  samples_<job_id>.png        — Generated image grid
```

### Job ID Reference

| Model | SLURM Job ID |
|-------|-------------|
| Classical-A | 48763211 |
| Classical-B | 48804337 |
| Classical-C | 49235600 |
| v1-QCNN | 48639698 |
| v1-QViT | 48639699 |
| v2 | 48722700 |
| v3-pw2 | 48806160 |
| v3-pw3 | 48807333 |
| v3-sl3 | 48807335 |
| v4-L2 | 48872407 |
| v4-L4 | 48872408 |
| v5 | 48987378 |
| v6 | 49235601 |
| v7 | 49235602 |
