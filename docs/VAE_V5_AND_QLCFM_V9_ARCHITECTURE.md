# VAE v5/v6 and QLCFM v9 Architecture Guide

**Date**: 2026-03-06 (updated)
**Files**: `models/train_vae_v5.py`, `models/train_vae_v6.py`, `models/QuantumLatentCFM_v9.py`

---

## Table of Contents

1. [Overview](#1-overview)
2. [VAE v5: Encoder Bottleneck Fix (Failed)](#2-vae-v5-encoder-bottleneck-fix-failed)
   - [2.1 Motivation](#21-motivation)
   - [2.2 Architecture](#22-architecture)
   - [2.3 Discriminator and Training](#23-discriminator-and-training)
   - [2.4 Loss Functions](#24-loss-functions)
   - [2.5 Results and Failure Analysis](#25-results-and-failure-analysis)
3. [VAE v6: Gradual Channel Reduction + No Adversarial](#3-vae-v6-gradual-channel-reduction--no-adversarial)
   - [3.1 Root Cause Analysis](#31-root-cause-analysis)
   - [3.2 Fix 1: Remove Adversarial Training](#32-fix-1-remove-adversarial-training)
   - [3.3 Fix 2: Gradual Channel Reduction](#33-fix-2-gradual-channel-reduction)
   - [3.4 Architecture Comparison (v4 vs v5 vs v6)](#34-architecture-comparison-v4-vs-v5-vs-v6)
   - [3.5 Loss Functions](#35-loss-functions)
4. [QLCFM v9: CFM Training Improvements](#4-qlcfm-v9-cfm-training-improvements)
   - [4.1 Motivation](#41-motivation)
   - [4.2 Overall Pipeline](#42-overall-pipeline)
   - [4.3 Improvement 1: Logit-Normal Timestep Sampling](#43-improvement-1-logit-normal-timestep-sampling)
   - [4.4 Improvement 2: Midpoint ODE Solver](#44-improvement-2-midpoint-ode-solver)
   - [4.5 Improvement 3: EMA for Velocity Field](#45-improvement-3-ema-for-velocity-field)
   - [4.6 Quantum Velocity Field Architecture](#46-quantum-velocity-field-architecture)
5. [Butterfly vs. Pyramid Circuit Topology](#5-butterfly-vs-pyramid-circuit-topology)
   - [5.1 Analysis](#51-analysis)
   - [5.2 Recommendation](#52-recommendation)
6. [Training Workflow](#6-training-workflow)
7. [Hyperparameters](#7-hyperparameters)
8. [References](#8-references)

---

## 1. Overview

The Quantum Latent Conditional Flow Matching (QLCFM) framework generates images through a two-stage pipeline:

1. **VAE** compresses images into a flat latent vector `z in R^d`.
2. **Quantum Velocity Field (VF)** learns a flow from Gaussian noise `z_0 ~ N(0,I)` to the data distribution `z_1 ~ q(z|x)` in this latent space.

This document covers the VAE evolution (v5 attempt, v6 fix) and QLCFM v9's CFM improvements:

| Component | Version | Problem Addressed | Key Change | Outcome |
|---|---|---|---|---|
| VAE | v3/v4 -> **v5** | Encoder bottleneck (PSNR ~19.5) | 1x1 conv channel reduction | **Failed**: PSNR 18.84 (-0.54 dB) |
| VAE | v5 -> **v6** | 1x1 conv bottleneck + D collapse | Gradual ResBlock reduction + no adversarial | Pending |
| CFM | v6 -> **v9** | Suboptimal training & ODE integration | Logit-normal sampling, midpoint solver, VF EMA | Pending |

---

## 2. VAE v5: Encoder Bottleneck Fix (Failed)

### 2.1 Motivation

VAE v3 and v4 both plateaued at PSNR ~19.5 dB on CIFAR-10, well below publication targets of 25+ dB. The v4 experiment isolated the cause: fixing the discriminator (spectral normalization, logistic loss, adaptive weighting, D warmup) yielded only +0.04 dB improvement (19.47 -> 19.51), confirming the ceiling is in the **encoder**, not the adversarial training.

The encoder in v3/v4 has two stacked bottlenecks:

1. **Spatial collapse**: An unnecessary 4th stride-2 downsample from 4x4 to 2x2, destroying 75% of spatial information at the most abstract feature level.
2. **FC compression**: `Flatten(256 x 2 x 2 = 1024)` followed by `Linear(1024, 64)`, a **16:1 compression ratio** that forces the network to discard most of the 1024-dimensional feature map.

These two bottlenecks cascade: the spatial collapse loses fine structure, then the FC layer discards most of what remains.

### 2.2 Architecture

VAE v5 (`ResConvVAE_v5`) fixes both bottlenecks while keeping the latent dimension at 64 (required by the quantum VF):

#### Encoder (v3/v4 vs. v5)

```
v3/v4 ENCODER (problematic):
  Input: (B, 3, 32, 32)
  Conv2d(3, 64, 3x3)                      32x32
  ResBlock(64) x2                          32x32
  Conv2d(64, 64, stride=2)                 16x16   downsample 1
  ResBlock(64->128), ResBlock(128)         16x16
  Conv2d(128, 128, stride=2)               8x8    downsample 2
  ResBlock(128->256), ResBlock(256)        8x8
  SelfAttention(256)                       8x8
  Conv2d(256, 256, stride=2)               4x4    downsample 3
  ResBlock(256) x2                         4x4
  SelfAttention(256)                       4x4
  Conv2d(256, 256, stride=2)  <== REMOVED  2x2    downsample 4
  Flatten: 256 x 2 x 2 = 1024
  Linear(1024, 64)            <== 16:1 compression
```

```
v5 ENCODER (fixed):
  Input: (B, 3, 32, 32)
  Conv2d(3, 64, 3x3)                      32x32
  ResBlock(64) x2                          32x32
  Conv2d(64, 64, stride=2)                 16x16   downsample 1
  ResBlock(64->128), ResBlock(128)         16x16
  Conv2d(128, 128, stride=2)              8x8     downsample 2
  ResBlock(128->256), ResBlock(256)        8x8
  SelfAttention(256)                       8x8
  Conv2d(256, 256, stride=2)               4x4    downsample 3
  ResBlock(256) x2                         4x4
  SelfAttention(256)                       4x4
  GroupNorm(32, 256), SiLU
  Conv2d(256, C_z, 1)        <== NEW: 1x1 conv channel reduction
  Flatten: C_z x 4 x 4 = C_z * 16
  Linear(C_z * 16, 64)       <== 1:1 ratio (with C_z=4)
```

With the default `C_z=4`:
- Pre-flatten dimension = 4 channels x 4 x 4 spatial = **64**
- `fc_mu` = `Linear(64, 64)` = **1:1 ratio** (a learned rotation, not compression)
- `fc_logvar` = `Linear(64, 64)`

The 1x1 convolution acts as a learned channel selector that smoothly reduces 256 channels to `C_z` channels while preserving full 4x4 spatial resolution. This is analogous to the channel reduction used in Stable Diffusion's VAE (Rombach et al., 2022), which uses 1x1 convolutions at the bottleneck.

#### Decoder (mirrors the encoder)

```
v5 DECODER:
  Linear(64, C_z * 16), SiLU
  Reshape: (B, C_z, 4, 4)
  Conv2d(C_z, 256, 1)         <== 1x1 conv channel expansion
  ResBlock(256) x2
  SelfAttention(256)                       4x4
  ConvTranspose2d(256, 256, 4, 2, 1)      8x8    upsample 1
  ResBlock(256->128), ResBlock(128)
  SelfAttention(128)                       8x8
  ConvTranspose2d(128, 128, 4, 2, 1)      16x16   upsample 2
  ResBlock(128->64), ResBlock(64)
  ConvTranspose2d(64, 64, 4, 2, 1)        32x32   upsample 3
  ResBlock(64) x2
  GroupNorm(32, 64), SiLU
  Conv2d(64, 3, 3x3)
  Tanh                                     output [-1, 1]
```

#### Why flat latent (not spatial)

A spatial latent (e.g., keeping a 4x4 feature map) would improve classical decoders, but the quantum VF operates on a **flat vector**. The quantum circuit has all-to-all entanglement through its gate structure and no locality inductive bias. A flat, decorrelated latent vector maximizes the quantum VF's advantage: every latent dimension can influence every other through entanglement, without needing to learn spatial adjacency patterns that a classical convolutional VF handles natively.

### 2.3 Discriminator and Training

VAE v5 carries forward all discriminator fixes from v4:

| Component | v3 (broken) | v5 (fixed, from v4) |
|---|---|---|
| Discriminator | PatchGAN 64->128->256->1 (663K) | PatchGAN 64->128->256->512->1 (~1.85M) |
| Normalization | None | **Spectral normalization** on all conv layers |
| Loss function | Hinge (ReLU clips gradients, dead zone at D_loss=2.0) | **Logistic (softplus)**, non-saturating |
| Adaptive weight | Off | **On** (VQGAN-style, Esser et al., 2021) |
| D warmup | None | **5 epochs** D-only before adding adv loss to G |
| D ramp | None | **20 epochs** linear ramp of adversarial weight |
| D optimizer | Adam betas=(0.5, 0.9), cosine LR | Adam **betas=(0.0, 0.99)**, **constant LR=2e-4** |
| R1 penalty | None | **R1 gradient penalty** (gamma=10.0, every 16 batches) |

#### Training stages

1. **Epochs 1-50**: VAE-only training (L1 reconstruction + KL + LPIPS). No discriminator.
2. **Epochs 51-55**: Discriminator warmup. D trains on real/fake pairs, but adversarial loss is not yet applied to the generator.
3. **Epochs 56-75**: Adversarial ramp. Generator adversarial weight linearly ramps from 0 to full.
4. **Epochs 76-300**: Full adversarial training with adaptive weighting.

### 2.4 Loss Functions

**Generator loss**:
```
L_G = L1(x_hat, x) + beta * KL(q(z|x) || p(z)) + lambda_lpips * LPIPS(x_hat, x)
      + disc_factor * adaptive_weight * L_adv_G(D(x_hat))
```

- **L1 reconstruction**: `F.l1_loss(x_hat, x)` — robust to outliers
- **KL divergence**: With free bits (0.25 nats/dim) to prevent posterior collapse (Kingma et al., 2016)
- **LPIPS perceptual loss**: VGG-based (Zhang et al., 2018)
- **Adversarial (generator)**: `softplus(-D(x_hat))` — non-saturating logistic (Goodfellow et al., 2014)
- **Adaptive weight**: Balances reconstruction and adversarial gradients at the decoder's last layer (Esser et al., 2021)

**Discriminator loss**:
```
L_D = softplus(-D(x)) + softplus(D(x_hat_detached))
      + (gamma/2) * R1(D, x) * r1_every   [every r1_every batches]
```

- **Logistic loss**: `softplus(-D(real)) + softplus(D(fake))` — no gradient dead zones
- **R1 gradient penalty**: Stabilizes discriminator (Mescheder et al., 2018)

**EMA**: Exponential moving average (decay=0.999) of VAE weights is maintained throughout training. Validation and best-model selection use EMA weights.

### 2.5 Results and Failure Analysis

VAE v5 (job 49678464) produced **worse** results than v4:

| Metric | v4 (best EMA) | v5 (pre-adv, ep 50) | v5 (post-collapse, ep 122, cancelled) |
|---|---|---|---|
| PSNR (dB) | 19.44 (ep 57) | **18.86** (EMA best) | **17.57** (-1.29 dB from peak) |
| D_loss | Converges to 1.3863 | N/A (D starts ep 51) | 1.3863 (fully collapsed) |
| Adaptive weight | 4.1 -> 1300+ | N/A | 7.8 -> **9,660+** |
| D(real) / D(fake) | ~0 / ~0 | N/A | 0.000 / 0.000 |

**Two independent problems were identified:**

1. **1x1 conv bottleneck (v5-specific regression, -0.54 dB)**:
   - `Conv2d(256, 4, 1)` compresses 256 channels to 4 **independently at each pixel** with zero spatial mixing.
   - Each of the 16 pixels (4x4) is compressed from 256 to 4 channels independently -- the conv kernel sees only a single pixel.
   - v4's `Conv2d(256, 256, 3, 2, 1)` stride-2 downsample at least mixes 3x3 spatial neighborhoods before the FC layer.
   - The 1:1 FC ratio in v5 (64->64) is irrelevant because the damage is done at the conv layer.

2. **Discriminator collapse (inherited from v3/v4)**:
   - D outputs converge to ~0 during warmup: D(real)=0.026, D(fake)=0.024 after 5 warmup epochs.
   - D_loss converges to `2*ln(2) = 1.3863` (uninformative equilibrium where D cannot distinguish real from fake).
   - Adaptive weight explodes: 7.8 -> 9,660+ (v5, ep 122), 4.1 -> 1,300+ (v4).
   - PSNR drops after adversarial training starts: v4 peaked at 19.44 (ep 57), dropped to 18.10 (ep 75). v4's reported "best" of 19.51 was the pre-collapse EMA checkpoint.
   - Root cause: D is too weak relative to 13M-param G after only 5 warmup epochs. The adaptive weight amplifies instability, and R1 penalty (effective weight 80) may over-regularize D.

---

## 3. VAE v6: Gradual Channel Reduction + No Adversarial

**File**: `models/train_vae_v6.py`

### 3.1 Root Cause Analysis

VAE v5 failed for two independent reasons, requiring two independent fixes:

| Problem | Root Cause | Evidence | Fix |
|---|---|---|---|
| PSNR regression (-0.54 dB vs v4) | 1x1 conv has no spatial mixing | Each pixel compressed independently 256->4 | **Gradual reduction** via ResBlock(256->64) + Conv3x3(64->c_z) |
| D collapse (shared with v3/v4) | D too weak, adaptive weight amplifies | D_loss->1.3863, PSNR drops post-adversarial | **Remove adversarial training** entirely |

### 3.2 Fix 1: Remove Adversarial Training

The discriminator contributed **0 dB** to v4's final PSNR. The v4 "best" of 19.51 dB was the EMA checkpoint captured *before* discriminator collapse at epoch 57. After collapse, PSNR dropped to 18.10 dB. The adversarial training actively hurt performance.

v6 removes all adversarial components:
- No PatchGAN discriminator
- No adversarial loss term
- No adaptive weight computation
- No R1 gradient penalty
- No D warmup/ramp schedule
- Single optimizer: `Adam(vae.parameters(), lr=1e-4, betas=(0.5, 0.9))` with CosineAnnealingLR

This simplifies the training pipeline and eliminates an entire class of instability.

### 3.3 Fix 2: Gradual Channel Reduction

Replace v5's 1x1 conv with a two-stage reduction that preserves spatial mixing at each step:

```
v5 bottleneck (broken):
  ... -> 4x4, 256ch
  Conv2d(256, 4, 1)            64:1 channel compression, NO spatial mixing
  Flatten(4*16=64)
  Linear(64, 64)               1:1 (irrelevant, damage already done)

v6 bottleneck (fixed):
  ... -> 4x4, 256ch
  ResBlock_v3(256, 64)          4:1 reduction WITH 3x3 spatial mixing + nonlinearity
  GroupNorm(32, 64), SiLU
  Conv2d(64, 4, 3, 1, 1)      16:1 reduction WITH 3x3 spatial mixing
  Flatten(4*16=64)
  Linear(64, 64)               1:1
```

The ResBlock provides:
- **3x3 convolutions**: Each output pixel sees a 3x3 neighborhood of the input (spatial mixing)
- **Nonlinearity**: GroupNorm + SiLU between the two convolutions within the ResBlock
- **Skip connection**: Residual path with 1x1 projection for channel change (256->64)
- **Two-stage reduction**: 256->64 (ResBlock) then 64->4 (Conv3x3), rather than a single 256->4 jump

The decoder mirrors this with `Conv2d(c_z, 64, 3, 1, 1)` + `ResBlock_v3(64, 256)`.

### 3.4 Architecture Comparison (v4 vs v5 vs v6)

```
                v4 (baseline)              v5 (failed)                v6 (fixed)
                ─────────────              ───────────                ──────────
Spatial:        4x4 -> 2x2 (stride-2)     Stop at 4x4               Stop at 4x4
Channel:        256 channels at 2x2        Conv1x1(256->4) at 4x4    ResBlock(256->64) + Conv3x3(64->4) at 4x4
Spatial mixing: 3x3 stride-2 conv          NONE (1x1 kernel)         3x3 conv in ResBlock + 3x3 final conv
Pre-flatten:    256*2*2 = 1024             4*4*4 = 64                4*4*4 = 64
FC ratio:       1024->64 = 16:1            64->64 = 1:1              64->64 = 1:1
Adversarial:    Yes (collapsed)            Yes (collapsed)            NO
Stage-1 PSNR:   19.38 dB                   18.84 dB (-0.54)          Pending
```

### 3.5 Loss Functions

v6 uses a simplified loss with no adversarial terms:

```
L = L1(x_hat, x) + beta(t) * KL(q(z|x) || p(z)) + lambda_lpips * LPIPS(x_hat, x)
```

- **L1 reconstruction**: `F.l1_loss(x_hat, x)`
- **KL divergence**: With free bits (0.25 nats/dim), linear warmup over 10 epochs
- **LPIPS perceptual loss**: VGG-based (Zhang et al., 2018), computed every epoch
- **EMA**: decay=0.999 on VAE weights throughout training

No discriminator, no adaptive weight, no R1 penalty.

---

## 4. QLCFM v9: CFM Training Improvements

### 4.1 Motivation

QLCFM v6 established a fair controlled comparison framework between quantum and classical velocity fields. v9 adds three orthogonal improvements to the CFM training and inference pipeline that benefit **both** quantum and classical VFs, but are especially valuable for the quantum VF due to its limited capacity:

1. The quantum circuit has far fewer parameters than a classical MLP, so wasting training budget on "easy" timesteps near t=0 and t=1 is costly.
2. The Euler ODE solver (1st-order) requires many steps for accurate integration; the quantum VF's shallow circuit makes each evaluation relatively expensive.
3. Quantum gradients are inherently noisy due to the parameter-shift rule and measurement statistics, making training unstable without smoothing.

### 4.2 Overall Pipeline

```
Phase 1: VAE Pretraining (or load external VAE v6 weights)
  Input images x -> Encoder -> z = mu (latent code)
  z -> Decoder -> x_hat (reconstruction)
  Loss: L1 + beta*KL + lambda_lpips*LPIPS

Phase 2: CFM Training (quantum velocity field)
  z_1 = VAE.encode(x)          [frozen VAE]
  z_0 ~ N(0, I)                [noise]
  t ~ sigmoid(N(0, std))       [logit-normal sampling, NEW in v9]
  z_t = (1-t)*z_0 + t*z_1      [OT interpolation]
  target = z_1 - z_0            [constant velocity, OT path]
  v_pred = VF(z_t, t)           [quantum or classical]
  Loss = MSE(v_pred, target)

Generation:
  z_0 ~ N(0, I)
  for step in range(ode_steps):
    k1 = VF(z, t)
    k2 = VF(z + 0.5*dt*k1, t + 0.5*dt)   [midpoint, NEW in v9]
    z = z + dt * k2
  images = VAE.decode(z)
```

### 4.3 Improvement 1: Logit-Normal Timestep Sampling

**Problem**: In v6, timesteps are sampled uniformly: `t ~ Uniform(0, 1)`. This allocates equal training budget to all timesteps. However, the velocity field near t=0 (pure noise) and t=1 (nearly data) is relatively easy to learn, while the mid-range (t ~ 0.3-0.7) requires modeling complex transitions between noise and data distributions.

**Solution**: Logit-normal sampling from Esser et al. (2024):

```python
t = torch.sigmoid(torch.randn(B, device=device) * std)  # default std=1.0
```

This produces a bell-shaped distribution over [0, 1] centered at t=0.5, concentrating training samples where the velocity field is hardest to approximate. The `std` parameter controls the concentration:
- `std=1.0` (default): moderate concentration, ~60% of samples fall in [0.25, 0.75]
- `std=0.5`: stronger concentration
- `std -> inf`: approaches uniform

**Why this matters for quantum VF**: The quantum circuit has ~288 trainable VQC parameters (butterfly, depth=2) compared to thousands in a classical MLP. Focusing training on the most informative timesteps makes more efficient use of this limited capacity.

**Why not cosine schedule or loss weighting**: A cosine noise schedule would make the target velocity time-dependent (i.e., the target would no longer be the constant `z_1 - z_0`), which is harder for a shallow quantum circuit to learn. Loss weighting by timestep is redundant with logit-normal sampling — both mechanisms focus training on specific timestep ranges, and using both would double-count.

### 4.4 Improvement 2: Midpoint ODE Solver

**Problem**: v6 uses the Euler method (1st-order) with 100 steps for ODE integration during generation:

```python
# Euler (v6):
v = VF(z, t)
z = z + dt * v      # 1st-order, error O(dt^2) per step, O(dt) global
```

**Solution**: The midpoint method (2nd-order Runge-Kutta):

```python
# Midpoint (v9):
k1 = VF(z, t)
z_mid = z + 0.5 * dt * k1
t_mid = t + 0.5 * dt
k2 = VF(z_mid, t_mid)
z = z + dt * k2      # 2nd-order, error O(dt^3) per step, O(dt^2) global
```

Each midpoint step requires **2 VF evaluations** (vs. 1 for Euler), but the 2nd-order accuracy means far fewer steps are needed. The default configuration uses **50 midpoint steps = 100 VF evaluations**, matching the compute budget of 100 Euler steps but with significantly better accuracy.

The midpoint method evaluates the velocity at the middle of each interval, providing a better estimate of the average velocity over the step. This is particularly important for the quantum VF, which may have less smooth velocity predictions than a classical MLP due to the oscillatory nature of quantum expectation values.

### 4.5 Improvement 3: EMA for Velocity Field

**Problem**: Quantum circuit gradients are inherently noisy. The parameter-shift rule computes gradients through finite differences of quantum circuit evaluations, and the small number of trainable parameters means each gradient step has high variance. This makes the loss landscape rugged and training unstable.

**Solution**: Maintain an exponential moving average of VF weights during Phase 2 training:

```python
# EMA update (every training step):
shadow[k] = decay * shadow[k] + (1 - decay) * model[k]   # decay=0.999
```

The EMA model is used for:
- **Validation loss computation** (`val_loss_ema` column in CSV)
- **Best model selection** (tracked by EMA validation loss)
- **Final saved checkpoint** (EMA weights are saved as the best model)
- **Sample generation** (EMA weights produce smoother, more stable outputs)

This technique is standard in diffusion model training (Ho et al., 2020; Nichol & Dhariwal, 2021) and is especially beneficial for quantum VFs where gradient noise is higher than classical networks.

### 4.6 Quantum Velocity Field Architecture

The quantum VF architecture is identical to v6. For completeness:

```
Input: concat(z_t[64], t_emb[32]) = 96 dimensions

TIME EMBEDDING:
  t (scalar) -> sinusoidal_embedding(t, 32) -> time_mlp(Linear(32,32) -> SiLU -> Linear(32,32))
  -> t_emb[32]

QUANTUM CIRCUIT (SingleQuantumCircuit):
  1. Classical pre-processing:
     enc_proj: Linear(96, 256) -> SiLU -> Linear(256, 105)
     Maps 96-dim input to 105 encoding parameters

  2. SU(4) Encoding:
     7 blocks of SpecialUnitary gates (pairs of qubits):
       Even pairs: SU(4) on (0,1), (2,3), (4,5), (6,7)     [4 gates, 60 params]
       Odd pairs:  SU(4) on (1,2), (3,4), (5,6)             [3 gates, 45 params]
     Total: 7 SU(4) gates consuming 105 parameters
     Each SU(4) gate is a general 2-qubit unitary (15 params)

  3. QViT Variational Circuit (depth=2):
     Parameterized 2-qubit gates applied in butterfly/pyramid/x pattern
     Each gate: U3 + IsingXX + IsingYY + IsingZZ + U3 + IsingXX + IsingYY + IsingZZ
     (12 trainable parameters per gate)

  4. Adaptive Nonlinear Observables (ANO):
     28 pairwise observables: Hermitian(H_w) on each qubit pair (i,j) for i<j
     Each H_w is a parameterized 4x4 Hermitian matrix
     H_w = D_w + sum_k (A_wk * sigma_wk + B_wk * sigma_wk^T)
     Parameters: A[w], B[w] (off-diagonal), D[w] (diagonal) per observable

  Output: 28 expectation values

POST-PROCESSING:
  vel_head: Linear(28, 256) -> SiLU -> Linear(256, 64)
  Maps 28 quantum observables to 64-dim velocity prediction

DUAL OPTIMIZER (Chen, 2025):
  Circuit parameters: Adam lr=1e-3
  ANO parameters:     Adam lr=1e-1 (100x higher)
```

---

## 5. Butterfly vs. Pyramid Circuit Topology

### 5.1 Analysis

The current default QViT circuit topology is **butterfly**, which follows an FFT-like connectivity pattern. For 8 qubits with depth=2, we analyzed its expressibility:

#### Butterfly (default)
- **Pattern**: 3 layers per depth (stride-1, stride-2, stride-4), following the FFT structure
- **Gate pairs per depth layer**: 12 (out of 28 possible)
- **Total QViT params (depth=2)**: 288

```
Layer 0 (stride=1): (0,1), (2,3), (4,5), (6,7)
Layer 1 (stride=2): (0,2), (1,3), (4,6), (5,7)
Layer 2 (stride=4): (0,4), (1,5), (2,6), (3,7)
```

#### Pyramid (all-pairs)
- **Pattern**: All pairs of qubits, nearest-neighbor sweeps
- **Gate pairs per depth layer**: 28 (complete coverage)
- **Total QViT params (depth=2)**: 672

#### Comparison

| Property | Butterfly | Pyramid |
|---|---|---|
| Unique qubit pairs with direct gates | 12/28 (42.9%) | 28/28 (100%) |
| Trainable VQC params (depth=2) | 288 | 672 |
| Two-qubit Ising gates | 144 | 336 |
| All-to-all reachability (depth=2) | Yes (after 3 gate layers) | Yes (after 1 gate layer) |

#### The expressibility problem

The QLCFM measures **28 pairwise observables** (one per qubit pair). With butterfly topology:
- **12 observables** (42.9%) have a **direct** entangling gate between their qubit pair
- **16 observables** (57.1%) must build correlations **indirectly** through intermediate qubits

While information can reach all qubits after one full butterfly pass (3 gate layers), the indirect observables are constrained to be functions of the intermediate qubit chain. This limits the independent variation each observable can produce and makes gradients for those 16 observables noisier (backpropagation through more gates).

### 5.2 Recommendation

**Switch to pyramid topology** (`--qvit-circuit=pyramid`) for v9 experiments. This gives every observed pair a direct entangling gate, at the cost of 2.3x more VQC parameters (672 vs 288). Since the total VF parameter count is dominated by `enc_proj` (~25K) and `vel_head` (~7.4K), this QViT parameter increase is modest in the overall model.

An alternative is increasing butterfly depth from 2 to 3, which gives each pair more indirect paths. However, pyramid at depth=2 is strictly more expressive because every observable has a direct gate.

---

## 6. Training Workflow

### Step 1: Train VAE v6

```bash
sbatch jobs/run_vae_v6.sh
```

This trains the VAE for 300 epochs on CIFAR-10 with no adversarial training. Expected output:
- Weights: `checkpoints/weights_vae_v6_ema_cifar10_<job_id>.pt`
- Metrics: `results/metrics_vae_v6_cifar10_<job_id>.json`
- Target: PSNR > 19.5 dB (matching or exceeding v4 pre-collapse peak)

Resume if wall time runs out:
```bash
PREV_JOB_ID=<job_id> sbatch jobs/run_vae_v6.sh
```

### Step 2: Train QLCFM v9 with pretrained VAE v6

```bash
VAE_CKPT=checkpoints/weights_vae_v6_ema_cifar10_<job_id>.pt \
  sbatch jobs/run_qlcfm_cifar_v9.sh
```

This runs Phase 2 only (CFM training with frozen VAE v6). The VF is trained for 200 epochs.

### Step 3 (recommended): Try pyramid topology

Modify the SLURM script or run directly:

```bash
python -u models/QuantumLatentCFM_v9.py \
    --phase=2 --dataset=cifar10 \
    --vae-arch=v6 --vae-ckpt=<path> \
    --velocity-field=quantum --n-circuits=1 --n-qubits=8 \
    --qvit-circuit=pyramid \
    --logit-normal-std=1.0 --ode-solver=midpoint --ode-steps=50 \
    --vf-ema-decay=0.999 \
    --epochs=200 --seed=2025 \
    --job-id=qlcfm_cifar_v9_pyramid_${SLURM_JOB_ID}
```

---

## 7. Hyperparameters

### VAE v6

| Parameter | Default | Description |
|---|---|---|
| `--latent-dim` | 64 | Latent vector dimension |
| `--c-z` | 4 | Channels after gradual reduction (pre-flatten = c_z * 16) |
| `--beta` | 0.001 | KL weight |
| `--beta-warmup-epochs` | 10 | Linear ramp from 0 to beta |
| `--free-bits` | 0.25 | Minimum KL per latent dim (nats) |
| `--lambda-lpips` | 1.0 | LPIPS perceptual loss weight |
| `--lr` | 1e-4 | Learning rate (cosine annealing) |
| `--ema-decay` | 0.999 | EMA decay for VAE weights |
| `--n-epochs` | 300 | Total training epochs |

Note: v6 has no discriminator parameters (`--lambda-adv`, `--adversarial-start-epoch`, `--disc-warmup-epochs`, `--disc-ramp-epochs`, `--lr-disc`, `--r1-gamma` are all removed).

### VAE v5 (deprecated)

| Parameter | Default | Description |
|---|---|---|
| `--latent-dim` | 64 | Latent vector dimension |
| `--c-z` | 4 | Channels after 1x1 conv (pre-flatten = c_z * 16) |
| `--beta` | 0.001 | KL weight |
| `--lambda-lpips` | 1.0 | LPIPS perceptual loss weight |
| `--lambda-adv` | 0.1 | Adversarial loss weight |
| `--adversarial-start-epoch` | 51 | When to start adversarial training |
| `--disc-warmup-epochs` | 5 | D-only warmup before adding adv to G |
| `--disc-ramp-epochs` | 20 | Linear ramp of adversarial weight |
| `--lr` | 1e-4 | Generator learning rate (cosine decay) |
| `--lr-disc` | 2e-4 | Discriminator learning rate (constant) |
| `--r1-gamma` | 10.0 | R1 gradient penalty weight |
| `--n-epochs` | 300 | Total training epochs |

### QLCFM v9

| Parameter | Default | Description |
|---|---|---|
| `--logit-normal-std` | 1.0 | Logit-normal sampling std (0 = uniform) |
| `--ode-solver` | midpoint | ODE solver (midpoint or euler) |
| `--ode-steps` | 50 | ODE integration steps (50 midpoint = 100 VF evals) |
| `--vf-ema-decay` | 0.999 | VF EMA decay (0 = disabled) |
| `--vae-arch` | resconv | VAE architecture (resconv, legacy, v5, v6) |
| `--vae-ckpt` | "" | Path to external pretrained VAE weights |
| `--c-z` | 4 | Channels in VAE v5/v6 bottleneck |
| `--n-circuits` | 1 | Number of quantum circuits (K) |
| `--n-qubits` | 8 | Qubits per circuit |
| `--encoding-type` | sun | Encoding type (sun = SU(4), angle) |
| `--vqc-type` | qvit | Variational circuit type (qvit, hardware_efficient) |
| `--qvit-circuit` | butterfly | QViT topology (butterfly, pyramid, x) |
| `--vqc-depth` | 2 | Circuit depth (number of VQC layer repetitions) |
| `--k-local` | 2 | Observable locality (2 = pairwise) |
| `--obs-scheme` | pairwise | Observable grouping (pairwise, sliding) |
| `--lr` | 1e-3 | Circuit parameter learning rate |
| `--lr-H` | 1e-1 | ANO parameter learning rate (100x circuit lr) |
| `--time-embed-dim` | 32 | Time embedding dimension |
| `--epochs` | 200 | CFM training epochs |

---

## 8. References

1. **Lipman, Y., Chen, R.T.Q., Ben-Hamu, H., Nickel, M., & Le, M.** (2023). Flow Matching for Generative Modeling. *ICLR 2023*.
   - Foundation of the Conditional Flow Matching framework. Defines the OT interpolation path `z_t = (1-t)*z_0 + t*z_1` and constant velocity target `v* = z_1 - z_0` used in all QLCFM versions.

2. **Esser, P., Kulal, S., Blattmann, A., Entezari, R., Muller, J., Saini, H., Levi, Y., Lorenz, D., Sauer, A., Boeber, F., Dockhorn, T., Karber, F., Cascio Granziera, F., Qiu, R., & Rombach, R.** (2024). Scaling Rectified Flow Transformers for High-Resolution Image Synthesis. *ICML 2024* (Best Paper Award; Stable Diffusion 3). arXiv:2403.03206.
   - Source of the logit-normal timestep sampling strategy used in v9. Shows that concentrating training on mid-range timesteps improves sample quality for rectified flow models.

3. **Rombach, R., Blattmann, A., Lorenz, D., Esser, P., & Ommer, B.** (2022). High-Resolution Image Synthesis with Latent Diffusion Models. *CVPR 2022*.
   - Latent diffusion model framework. Motivates the two-stage VAE + generative model pipeline. Uses 1x1 convolutions for channel reduction at the VAE bottleneck, similar to our v5 fix.

4. **Esser, P., Rombach, R., & Ommer, B.** (2021). Taming Transformers for High-Resolution Image Synthesis. *CVPR 2021* (VQGAN).
   - Adaptive adversarial weighting used in the VAE discriminator training. Balances reconstruction and adversarial gradients at the decoder's last layer.

5. **Ho, J., Jain, A., & Abbeel, P.** (2020). Denoising Diffusion Probabilistic Models. *NeurIPS 2020*.
   - Introduced EMA for generative model training, showing that exponential moving average weights produce better samples than the raw training weights.

6. **Nichol, A.Q. & Dhariwal, P.** (2021). Improved Denoising Diffusion Probabilistic Models. *ICML 2021*.
   - Further establishes EMA as standard practice in diffusion/flow model training.

7. **Cherrat, E.A., Kerenidis, I., Mathur, N., Landman, J., Strahm, M., & Li, Y.Y.** (2024). Quantum Vision Transformers. *Quantum*, 8, 1265. DOI: 10.22331/q-2024-02-22-1265.
   - Source of the QViT butterfly, pyramid, and x circuit topologies used in the quantum VF. Defines the parameterized beam splitter (RBS) gate structure: U3 + IsingXX/YY/ZZ + U3 + IsingXX/YY/ZZ.

8. **Mescheder, L., Geiger, A., & Nowozin, S.** (2018). Which Training Methods for GANs do actually Converge? *ICML 2018*.
   - R1 gradient penalty used in the discriminator training.

9. **Zhang, R., Isola, P., Efros, A.A., Shechtman, E., & Wang, O.** (2018). The Unreasonable Effectiveness of Deep Features as a Perceptual Metric. *CVPR 2018*.
   - LPIPS perceptual loss used in VAE reconstruction.

10. **Kingma, D.P., Salimans, T., Jozefowicz, R., Chen, X., Sutskever, I., & Welling, M.** (2016). Improved Variational Inference with Inverse Autoregressive Flow. *NeurIPS 2016*.
    - Free bits technique for preventing KL posterior collapse in VAEs.

11. **Miyato, T., Kataoka, T., Koyama, M., & Yoshida, Y.** (2018). Spectral Normalization for Generative Adversarial Networks. *ICLR 2018*.
    - Spectral normalization applied to the PatchGAN discriminator.

12. **Goodfellow, I.J., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., Courville, A., & Bengio, Y.** (2014). Generative Adversarial Nets. *NeurIPS 2014*.
    - Non-saturating logistic loss for GAN training.

13. **Karras, T., Laine, S., Aittala, M., Hellsten, J., Lehtinen, J., & Aila, T.** (2020). Analyzing and Improving the Image Quality of StyleGAN. *CVPR 2020* (StyleGAN2).
    - R1 gradient penalty and lazy regularization (applying penalty every N batches).

14. **Chen, S.Y.-C., Wei, T.-C., Zhang, C., Yu, H., & Yoo, S.** (2022). Quantum convolutional neural networks for high energy physics data analysis. *Physical Review Research*, 4, 013231.
    - Quantum convolutional neural network architecture for classification tasks.

15. **Chen, S.Y.-C., Tseng, H.-H., Lin, H.-Y., & Yoo, S.** (2025). Learning to Program Quantum Measurements for Machine Learning. arXiv:2505.13525.
    - Dual optimizer pattern: separate optimizers for circuit parameters and observable parameters, with observable params at higher learning rate for improved convergence stability.
