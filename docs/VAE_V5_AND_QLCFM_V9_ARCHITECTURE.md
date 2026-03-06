# VAE v5 and QLCFM v9 Architecture Guide

**Date**: 2026-03-05
**Files**: `models/train_vae_v5.py`, `models/QuantumLatentCFM_v9.py`

---

## Table of Contents

1. [Overview](#1-overview)
2. [VAE v5: Encoder Bottleneck Fix](#2-vae-v5-encoder-bottleneck-fix)
   - [2.1 Motivation](#21-motivation)
   - [2.2 Architecture](#22-architecture)
   - [2.3 Discriminator and Training](#23-discriminator-and-training)
   - [2.4 Loss Functions](#24-loss-functions)
3. [QLCFM v9: CFM Training Improvements](#3-qlcfm-v9-cfm-training-improvements)
   - [3.1 Motivation](#31-motivation)
   - [3.2 Overall Pipeline](#32-overall-pipeline)
   - [3.3 Improvement 1: Logit-Normal Timestep Sampling](#33-improvement-1-logit-normal-timestep-sampling)
   - [3.4 Improvement 2: Midpoint ODE Solver](#34-improvement-2-midpoint-ode-solver)
   - [3.5 Improvement 3: EMA for Velocity Field](#35-improvement-3-ema-for-velocity-field)
   - [3.6 Quantum Velocity Field Architecture](#36-quantum-velocity-field-architecture)
4. [Butterfly vs. Pyramid Circuit Topology](#4-butterfly-vs-pyramid-circuit-topology)
   - [4.1 Analysis](#41-analysis)
   - [4.2 Recommendation](#42-recommendation)
5. [Training Workflow](#5-training-workflow)
6. [Hyperparameters](#6-hyperparameters)
7. [References](#7-references)

---

## 1. Overview

The Quantum Latent Conditional Flow Matching (QLCFM) framework generates images through a two-stage pipeline:

1. **VAE** compresses images into a flat latent vector `z in R^d`.
2. **Quantum Velocity Field (VF)** learns a flow from Gaussian noise `z_0 ~ N(0,I)` to the data distribution `z_1 ~ q(z|x)` in this latent space.

VAE v5 and QLCFM v9 address independent bottlenecks in each stage:

| Component | Version | Problem Addressed | Key Change |
|---|---|---|---|
| VAE | v3/v4 -> **v5** | Encoder information bottleneck (PSNR capped at ~19.5 dB) | Remove stride-2 downsample at 4x4; use 1x1 conv channel reduction |
| CFM | v6 -> **v9** | Suboptimal training distribution and ODE integration | Logit-normal sampling, midpoint ODE solver, VF EMA |

---

## 2. VAE v5: Encoder Bottleneck Fix

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

---

## 3. QLCFM v9: CFM Training Improvements

### 3.1 Motivation

QLCFM v6 established a fair controlled comparison framework between quantum and classical velocity fields. v9 adds three orthogonal improvements to the CFM training and inference pipeline that benefit **both** quantum and classical VFs, but are especially valuable for the quantum VF due to its limited capacity:

1. The quantum circuit has far fewer parameters than a classical MLP, so wasting training budget on "easy" timesteps near t=0 and t=1 is costly.
2. The Euler ODE solver (1st-order) requires many steps for accurate integration; the quantum VF's shallow circuit makes each evaluation relatively expensive.
3. Quantum gradients are inherently noisy due to the parameter-shift rule and measurement statistics, making training unstable without smoothing.

### 3.2 Overall Pipeline

```
Phase 1: VAE Pretraining (or load external VAE v5 weights)
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

### 3.3 Improvement 1: Logit-Normal Timestep Sampling

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

### 3.4 Improvement 2: Midpoint ODE Solver

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

### 3.5 Improvement 3: EMA for Velocity Field

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

### 3.6 Quantum Velocity Field Architecture

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

## 4. Butterfly vs. Pyramid Circuit Topology

### 4.1 Analysis

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

### 4.2 Recommendation

**Switch to pyramid topology** (`--qvit-circuit=pyramid`) for v9 experiments. This gives every observed pair a direct entangling gate, at the cost of 2.3x more VQC parameters (672 vs 288). Since the total VF parameter count is dominated by `enc_proj` (~25K) and `vel_head` (~7.4K), this QViT parameter increase is modest in the overall model.

An alternative is increasing butterfly depth from 2 to 3, which gives each pair more indirect paths. However, pyramid at depth=2 is strictly more expressive because every observable has a direct gate.

---

## 5. Training Workflow

### Step 1: Train VAE v5

```bash
sbatch jobs/run_vae_v5.sh
```

This trains the VAE for 300 epochs on CIFAR-10. Expected output:
- Weights: `checkpoints/weights_vae_v5_ema_cifar10_<job_id>.pt`
- Metrics: `results/metrics_vae_v5_cifar10_<job_id>.json`
- Target: PSNR 21-23 dB (vs. 19.5 dB for v3/v4)

### Step 2: Train QLCFM v9 with pretrained VAE v5

```bash
VAE_CKPT=checkpoints/weights_vae_v5_ema_cifar10_<job_id>.pt \
  sbatch jobs/run_qlcfm_cifar_v9.sh
```

This runs Phase 2 only (CFM training with frozen VAE v5). The VF is trained for 200 epochs.

### Step 3 (recommended): Try pyramid topology

Modify the SLURM script or run directly:

```bash
python -u models/QuantumLatentCFM_v9.py \
    --phase=2 --dataset=cifar10 \
    --vae-arch=v5 --vae-ckpt=<path> \
    --velocity-field=quantum --n-circuits=1 --n-qubits=8 \
    --qvit-circuit=pyramid \
    --logit-normal-std=1.0 --ode-solver=midpoint --ode-steps=50 \
    --vf-ema-decay=0.999 \
    --epochs=200 --seed=2025 \
    --job-id=qlcfm_cifar_v9_pyramid_${SLURM_JOB_ID}
```

---

## 6. Hyperparameters

### VAE v5

| Parameter | Default | Description |
|---|---|---|
| `--latent-dim` | 64 | Latent vector dimension |
| `--c-z` | 4 | Channels after 1x1 conv (pre-flatten = c_z * 16) |
| `--beta` | 0.001 | KL weight |
| `--beta-warmup-epochs` | 10 | Linear ramp from 0 to beta |
| `--free-bits` | 0.25 | Minimum KL per latent dim (nats) |
| `--lambda-lpips` | 1.0 | LPIPS perceptual loss weight |
| `--lambda-adv` | 0.1 | Adversarial loss weight |
| `--adversarial-start-epoch` | 51 | When to start adversarial training |
| `--disc-warmup-epochs` | 5 | D-only warmup before adding adv to G |
| `--disc-ramp-epochs` | 20 | Linear ramp of adversarial weight |
| `--lr` | 1e-4 | Generator learning rate (cosine decay) |
| `--lr-disc` | 2e-4 | Discriminator learning rate (constant) |
| `--ema-decay` | 0.999 | EMA decay for VAE weights |
| `--r1-gamma` | 10.0 | R1 gradient penalty weight |
| `--n-epochs` | 300 | Total training epochs |

### QLCFM v9

| Parameter | Default | Description |
|---|---|---|
| `--logit-normal-std` | 1.0 | Logit-normal sampling std (0 = uniform) |
| `--ode-solver` | midpoint | ODE solver (midpoint or euler) |
| `--ode-steps` | 50 | ODE integration steps (50 midpoint = 100 VF evals) |
| `--vf-ema-decay` | 0.999 | VF EMA decay (0 = disabled) |
| `--vae-arch` | resconv | VAE architecture (resconv, legacy, v5) |
| `--vae-ckpt` | "" | Path to external pretrained VAE weights |
| `--c-z` | 4 | Channels in VAE v5 bottleneck |
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

## 7. References

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
