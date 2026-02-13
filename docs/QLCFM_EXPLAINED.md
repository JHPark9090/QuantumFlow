# Quantum Latent Conditional Flow Matching (QLCFM) -- Detailed Explanation

**File:** `QuantumLatentCFM.py`
**Repository:** [github.com/JHPark9090/QuantumFlow](https://github.com/JHPark9090/QuantumFlow)

---

## Table of Contents

1. [What Problem Does QLCFM Solve?](#1-what-problem-does-qlcfm-solve)
2. [Background: Classical Flow Matching](#2-background-classical-flow-matching)
3. [Why Latent Space?](#3-why-latent-space)
4. [Full Pipeline Overview](#4-full-pipeline-overview)
5. [Phase 1: Convolutional VAE](#5-phase-1-convolutional-vae)
   - [5a: Legacy ConvVAE](#5a-legacy-convvae)
   - [5b: ResConvVAE (Deep Residual)](#5b-resconvvae-deep-residual)
   - [5c: VGG Perceptual Loss](#5c-vgg-perceptual-loss)
   - [5d: Beta Warmup](#5d-beta-warmup)
6. [Phase 2: Quantum Velocity Field](#6-phase-2-quantum-velocity-field)
   - [Step 6.1: Time Embedding](#step-61-time-embedding)
   - [Step 6.2: Classical Pre-Processing (enc_proj)](#step-62-classical-pre-processing-enc_proj)
   - [Step 6.3: Quantum Encoding (SU(4) and Angle)](#step-63-quantum-encoding-su4-and-angle)
   - [Step 6.4: Variational Quantum Circuits (QCNN, QViT, HWE)](#step-64-variational-quantum-circuits-qcnn-qvit-hwe)
   - [Step 6.5: Adaptive Non-Local Observable (ANO) Measurement](#step-65-adaptive-non-local-observable-ano-measurement)
   - [Step 6.6: Classical Post-Processing (vel_head)](#step-66-classical-post-processing-vel_head)
7. [Data Re-Uploading](#7-data-re-uploading)
8. [Classical MLP Baseline](#8-classical-mlp-baseline)
9. [CFM Training Objective](#9-cfm-training-objective)
10. [Dual Optimizer Strategy](#10-dual-optimizer-strategy)
11. [Generation via Euler ODE](#11-generation-via-euler-ode)
12. [FID and Inception Score Evaluation](#12-fid-and-inception-score-evaluation)
13. [Parameter Breakdown](#13-parameter-breakdown)
14. [Design Decisions](#14-design-decisions)
15. [CLI Reference](#15-cli-reference)
16. [References](#16-references)

---

## 1. What Problem Does QLCFM Solve?

QLCFM is a **generative model** for image synthesis. Given a dataset of images (e.g., CIFAR-10), it learns to generate new, unseen images from the same distribution. Unlike classification models that map images to labels, QLCFM maps random noise to realistic images.

The key innovation is that the core generative dynamics -- the velocity field that transports noise into data -- is implemented as a **quantum circuit** using SU(4) encoding, variational quantum circuits (QCNN, QViT, or HWE), and learnable Hermitian observables. All velocity information must pass through this quantum bottleneck.

---

## 2. Background: Classical Flow Matching

Flow Matching (Lipman et al., 2023) is a framework for training continuous normalizing flows. The idea:

**Core concept:** Define a time-dependent vector field `v(x, t)` that smoothly transports samples from a simple noise distribution (t=0) to the data distribution (t=1).

**The ODE:** A sample evolves according to:

```
dx/dt = v_theta(x, t),    t in [0, 1]
```

- At `t = 0`: `x` is random Gaussian noise
- At `t = 1`: `x` is a realistic data sample

**Optimal Transport Conditional Flow Matching (OT-CFM):**

Given a data point `x_1` and a noise sample `x_0 ~ N(0, I)`, the optimal transport interpolation path is:

```
x_t = (1 - t) * x_0 + t * x_1
```

This is a straight line from noise to data. The target velocity along this path is constant:

```
u_t = x_1 - x_0
```

**Training objective:**

```
L_CFM = E_{t ~ U(0,1), x_0 ~ N(0,I), x_1 ~ p_data} [ || v_theta(x_t, t) - (x_1 - x_0) ||^2 ]
```

We train a neural network `v_theta` to predict the velocity `x_1 - x_0` at any point along the interpolation path.

**Generation:** Sample `x_0 ~ N(0, I)`, then numerically integrate the ODE from `t=0` to `t=1` using the learned `v_theta`. The result is a generated sample.

---

## 3. Why Latent Space?

Images like CIFAR-10 live in a 3072-dimensional space (3 x 32 x 32). Quantum circuits with 12 qubits produce a handful of measurement outputs (e.g., 66 expectation values with pairwise k=2). Directly learning a velocity field from these outputs to 3072 dimensions is infeasible.

**Solution: Latent Flow Matching**

This is the same paradigm used by Stable Diffusion 3 (Esser et al., 2024) and Latent Diffusion Models (Rombach et al., 2022):

1. **Compress** images to a compact latent space using a classical autoencoder (VAE)
2. **Learn** the flow matching velocity field in the latent space
3. **Decompress** generated latent vectors back to images

```
Image Space (3072-dim)  <--VAE-->  Latent Space (32-dim)  <--Flow Matching-->  Noise
     x ~ p_data                       z_1 = Enc(x)                           z_0 ~ N(0,I)
```

The VAE reduces the dimensionality from 3072 to 32 (configurable via `--latent-dim`), making the quantum velocity field tractable. The flow matching loss operates entirely in latent space.

---

## 4. Full Pipeline Overview

```
============================== QLCFM PIPELINE ==============================

PHASE 1 -- VAE Pretraining (purely classical):

    Image (3, 32, 32)
         |
         v
    [Conv Encoder]  --> (mu, logvar) each of dim D (e.g., 32)
         |
    [Reparameterize]  z = mu + std * eps
         |
         v
    [Conv Decoder]  --> Reconstructed Image (3, 32, 32)

    Loss = MSE(x, x_hat) + beta * KL(q(z|x) || N(0,I))
         + lambda_perc * VGG_perceptual(x, x_hat)


PHASE 2 -- Quantum CFM Training (quantum or classical velocity field):

    Image x                                z_0 ~ N(0, I)
         |                                      |
    [Frozen VAE Encoder]                        |
         |                                      |
         v                                      v
    z_1 = mu (deterministic)              z_0 (noise)
         |                                      |
         +----------> z_t = (1-t)*z_0 + t*z_1 <+
                              |
                              v
                   [Quantum Velocity Field]
                     or [Classical MLP]
                              |
                              v
                      v_pred (D-dim)
                              |
                              v
                   Loss = MSE(v_pred, z_1 - z_0)


GENERATION:

    z_0 ~ N(0, I)
         |
    [Euler ODE: z += dt * v_theta(z, t)]  x 100 steps (t: 0 -> 1)
         |
         v
    z_1 (generated latent)
         |
    [VAE Decoder]
         |
         v
    Generated Image (3, 32, 32)

    [Optional: compute FID & IS against real images]

========================================================================
```

---

## 5. Phase 1: Convolutional VAE

**Purpose:** Learn a compressed, Gaussian-structured latent space for images.

**Why VAE instead of a plain autoencoder?** The KL divergence term in the VAE loss regularizes the latent space to be approximately N(0, I). This is critical because flow matching starts from N(0, I) noise -- if the latent space is not Gaussian-like, the flow cannot bridge noise and data smoothly.

Two VAE architectures are available, selected via `--vae-arch`:

### 5a: Legacy ConvVAE

The original lightweight architecture (~530K parameters).

**Encoder:**

```
Input: (batch, 3, 32, 32)
   |
   Conv2d(3, 32, kernel=4, stride=2, pad=1)  --> (batch, 32, 16, 16)
   BatchNorm2d(32) + ReLU
   |
   Conv2d(32, 64, kernel=4, stride=2, pad=1) --> (batch, 64, 8, 8)
   BatchNorm2d(64) + ReLU
   |
   Conv2d(64, 128, kernel=4, stride=2, pad=1) --> (batch, 128, 4, 4)
   BatchNorm2d(128) + ReLU
   |
   Flatten --> (batch, 2048)
   |
   +-- Linear(2048, D) --> mu     (batch, D)
   |
   +-- Linear(2048, D) --> logvar (batch, D)
```

**Decoder:** Mirror of encoder using `ConvTranspose2d`, ending with Sigmoid.

### 5b: ResConvVAE (Deep Residual)

The default architecture (~2.1M parameters), using pre-activation residual blocks.

**Residual block:** `BN -> ReLU -> Conv3x3 -> BN -> ReLU -> Conv3x3 + skip`

**Encoder:**

```
Input: (batch, 3, 32, 32)
   |
   Conv2d(3, 32, 3, 1, 1)                     --> (batch, 32, 32, 32)
   ResBlock(32) x 2
   Conv2d(32, 64, 4, 2, 1)                    --> (batch, 64, 16, 16)
   ResBlock(64) x 2
   Conv2d(64, 128, 4, 2, 1)                   --> (batch, 128, 8, 8)
   ResBlock(128) x 2
   Conv2d(128, 256, 4, 2, 1)                  --> (batch, 256, 4, 4)
   ResBlock(256) x 2
   BN(256) + ReLU
   Conv2d(256, 256, 4, 2, 1)                  --> (batch, 256, 2, 2)
   |
   Flatten --> (batch, 1024)
   |
   +-- Linear(1024, D) --> mu
   +-- Linear(1024, D) --> logvar
```

**Decoder:** Mirror using `ConvTranspose2d` + residual blocks, ending with Sigmoid.

### 5c: VGG Perceptual Loss

When `--lambda-perc > 0` (default 0.1), a frozen VGG16 network provides perceptual loss alongside pixel-wise MSE. Features are extracted at four levels (relu1_2, relu2_2, relu3_3, relu4_3) and compared with L1 loss.

```
L_VAE = L_recon + beta_eff * L_KL + lambda_perc * L_perceptual
```

### 5d: Beta Warmup

The KL weight ramps linearly from 0 to `beta` over `--beta-warmup-epochs` (default 20). This prevents posterior collapse by allowing the encoder to learn useful representations before being regularized.

```
beta_eff(epoch) = beta * min(1.0, (epoch + 1) / warmup_epochs)
```

### VAE Training

- **Optimizer:** Adam, lr = `--lr-vae` (default 1e-3)
- **Scheduler:** Cosine annealing over `--epochs`
- **Epochs:** 200 (typical)
- **Output:** Pretrained VAE weights saved to `weights_vae_<job_id>.pt`

After Phase 1, the VAE is **frozen** -- its weights are never updated again.

---

## 6. Phase 2: Quantum Velocity Field

This is the core of QLCFM. The velocity field `v_theta(z_t, t)` predicts the velocity that transports noise towards data in latent space. **Every velocity prediction must pass through the quantum circuit** -- there is no classical skip path.

### Complete Forward Pass (Standard, n_reupload=1)

```
=== QUANTUM VELOCITY FIELD: v_theta(z_t, t) ===

Inputs: z_t (batch, D),  t (batch,)
                |                |
                |    [Sinusoidal Embedding]
                |                |
                v                v
            z_t (D)        t_emb (D)
                |                |
                +--- concat -----+
                         |
                         v
                z_combined (2D)                      <-- Step 6.1-6.2
                         |
                    [enc_proj]
                  FC(2D, 256) + SiLU
                  FC(256, total_enc)
                         |
                         v
                enc_params (total_enc)               <-- encoding parameters
                         |
           ==============|==============
           |    QUANTUM CIRCUIT (N qubits)    |
           |                                  |
           |  [Encoding]  (Step 6.3)          |
           |    SU(4) or Angle embedding      |
           |         |                        |
           |  [VQC]  (Step 6.4)               |
           |    QCNN / QViT / HWE             |
           |         |                        |
           |  [ANO Measurement]  (Step 6.5)   |
           |    Learnable Hermitian obs        |
           |    sliding or pairwise groups     |
           |         |                        |
           =========|=========================
                    |
                    v
               q_out (n_obs)                         <-- quantum output
                    |
               [vel_head]                            <-- Step 6.6
             FC(n_obs, hidden) + SiLU
             FC(hidden, D)
                    |
                    v
             velocity v (D)                          <-- output
```

### Step 6.1: Time Embedding

The scalar time `t in [0, 1]` is embedded into a D-dimensional vector using sinusoidal positional encoding (Vaswani et al., 2017):

```
t_emb[i] = cos(t * 10000^(-i/(D/2)))       for i = 0, ..., D/2-1
t_emb[i] = sin(t * 10000^(-(i-D/2)/(D/2))) for i = D/2, ..., D-1
```

**Why?** A scalar `t` carries very little information. The sinusoidal embedding maps it to a rich, high-dimensional representation where nearby times have similar embeddings but distant times are distinguishable. This allows the network to learn different behavior at different flow times.

### Step 6.2: Classical Pre-Processing (enc_proj)

The time embedding is concatenated with the noisy latent vector:

```
z_combined = concat(z_t, t_emb)     shape: (batch, 2D)
```

Then a two-layer MLP maps this to the quantum encoding parameters:

```
z_combined (2D) --> Linear(2D, 256) --> SiLU --> Linear(256, total_enc) --> enc_params
```

**How is `total_enc` computed?** It depends on encoding type and re-upload configuration:

| Encoding | Params per block | total_enc (n_reupload=1) | total_enc (n_reupload=L) |
|---|---|---|---|
| SU(4) (`sun`) | `(n_even + n_odd) * 15` | `n_blocks * enc_per_block` | `L * enc_per_block` |
| Angle (`angle`) | `n_qubits` | `n_blocks * n_qubits` | `L * n_qubits` |

**Example (12 qubits, SU(4)):**
- Even pairs: (0,1), (2,3), (4,5), (6,7), (8,9), (10,11) = 6 gates
- Odd pairs: (1,2), (3,4), (5,6), (7,8), (9,10) = 5 gates
- enc_per_block = 11 gates x 15 params = 165
- n_blocks=2: total_enc = 330 | n_reupload=4: total_enc = 660

### Step 6.3: Quantum Encoding (SU(4) and Angle)

Two encoding strategies are available, selected via `--encoding-type`:

#### SU(4) Exponential Map Encoding (`sun`)

Uses `qml.SpecialUnitary(params, wires=[q1, q2])` to implement general 2-qubit unitary gates parameterized by 15 real numbers (the generators of the SU(4) Lie algebra):

```
U(theta) = exp(i * sum_{k=1}^{15} theta_k * G_k)
```

where `G_k` are the 15 generalized Gell-Mann matrices (a basis for the su(4) Lie algebra).

**Brick-wall layout (per block):**

```
Even layer:  SU4[q0,q1]  SU4[q2,q3]  ...  SU4[q_{N-2},q_{N-1}]
Odd layer:   SU4[q1,q2]  SU4[q3,q4]  ...  SU4[q_{N-3},q_{N-2}]
```

**Why SU(4)?** Unlike simpler encodings, SU(4) gates can generate any 2-qubit unitary, enabling the circuit to explore the full Hilbert space. The exponential map traces **geodesics** on the SU(N) manifold, providing a geometrically natural encoding (Wiersema et al., 2024).

#### Angle Encoding (`angle`)

Uses RY rotation gates with CNOT entangling layers:

```
Per block:
  RY(theta_0, q0)  RY(theta_1, q1)  ...  RY(theta_{N-1}, q_{N-1})
  CNOT(q0, q1)  CNOT(q2, q3)  ...    (even pairs)
  CNOT(q1, q2)  CNOT(q3, q4)  ...    (odd pairs, except last block)
```

Angle encoding uses fewer parameters per block (N vs 15*(N/2+...)) but has limited expressibility compared to SU(4).

**Critical property:** The encoding parameters are **data-dependent** -- they change for every input sample. This means the quantum state prepared is a function of `(z_t, t)`, allowing the circuit to represent different velocities for different inputs.

### Step 6.4: Variational Quantum Circuits (QCNN, QViT, HWE)

After encoding, the quantum state is processed by a variational quantum circuit with learnable (data-independent) parameters. Three VQC types are available, selected via `--vqc-type`:

#### QCNN (Quantum Convolutional Neural Network)

The QCNN alternates convolutional and pooling layers, progressively reducing the number of active qubits.

**Convolutional Layer:** Each layer applies staggered 2-qubit blocks across all pairs of adjacent qubits. Each block contains:

```
For qubit pair (w1, w2):
  U3(theta_1, theta_2, theta_3, wires=w1)
  U3(theta_4, theta_5, theta_6, wires=w2)
  IsingZZ(theta_7, wires=[w1, w2])
  IsingYY(theta_8, wires=[w1, w2])
  IsingXX(theta_9, wires=[w1, w2])
  U3(theta_10, theta_11, theta_12, wires=w1)
  U3(theta_13, theta_14, theta_15, wires=w2)
```

**15 parameters per 2-qubit block.** Applied in two passes (even parity, then odd parity).

**Pooling Layer:** Reduces active qubits by half. For each pair:
1. **Measure** the odd-indexed qubit via mid-circuit measurement
2. **Conditionally apply** a U3 rotation on the even-indexed qubit

**Note:** QCNN is excluded from data re-uploading (`n_reupload > 1`) due to memory constraints. If attempted, it falls back to `n_reupload=1` with a warning.

#### QViT (Quantum Vision Transformer)

QViT uses Reconfigurable Beam Splitter (RBS) inspired gates with three topology options selected via `--qvit-circuit`:

**Per gate pair (12 parameters):**

```
For qubit pair (w1, w2):
  U3(p[0:3], wires=w1)                    -- 3 params
  IsingXX(p[3]), IsingYY(p[4]), IsingZZ(p[5])  -- 3 params
  U3(p[6:9], wires=w1)                    -- 3 params
  IsingXX(p[9]), IsingYY(p[10]), IsingZZ(p[11]) -- 3 params
```

**Topologies:**

| Topology | Description | Gates per layer (12 qubits) |
|---|---|---|
| `butterfly` | Log-depth butterfly network (FFT-like) | 24 gates |
| `pyramid` | All-pairs triangle (full connectivity) | 66 gates |
| `x` | X-shaped: outer pairs + inner chain | 11 gates |

**Butterfly topology (default):** Inspired by the butterfly network in FFT, it provides O(N log N) connectivity in O(log N) depth. For 12 qubits, each layer has `ceil(log2(12))` = 4 sub-layers with increasing stride:

```
Stride 1:  (0,1) (2,3) (4,5) (6,7) (8,9) (10,11)
Stride 2:  (0,2) (1,3) (4,6) (5,7) (8,10) (9,11)
Stride 4:  (0,4) (1,5) (2,6) (3,7) (8,12)...  [filtered to valid qubits]
Stride 8:  (0,8) (1,9) (2,10) (3,11)
```

**QViT is the recommended VQC for re-uploading experiments** due to its balance of expressivity and efficiency.

#### Hardware-Efficient Ansatz (HWE)

The simplest VQC: alternating RY rotation and CNOT entangling layers.

```
Per layer:
  RY(theta_q, wires=q) for all q
  CNOT(q0,q1) CNOT(q2,q3) ...  (even pairs)
  CNOT(q1,q2) CNOT(q3,q4) ...  (odd pairs)
```

**N parameters per layer** (one RY angle per qubit). Simple but lacks the structured connectivity of QViT.

### Step 6.5: Adaptive Non-Local Observable (ANO) Measurement

Instead of measuring fixed Pauli operators (like PauliZ on each qubit), QLCFM uses **learnable Hermitian observables** that adapt during training (Chen et al., 2025; Lin et al., 2025).

#### Wire Groups and Observable Schemes

The observable measurement is controlled by two parameters:
- `--k-local`: Size of qubit groups (typically 2 or 3)
- `--obs-scheme`: How groups are formed (`sliding` or `pairwise`)

**Sliding scheme** (`obs_scheme="sliding"`): Consecutive windows of k qubits.

```
k=2, 12 qubits:
  [q0,q1], [q1,q2], [q2,q3], ..., [q10,q11]
  --> 11 groups (n_qubits - k + 1)

k=3, 12 qubits:
  [q0,q1,q2], [q1,q2,q3], ..., [q9,q10,q11]
  --> 10 groups
```

**Pairwise scheme** (`obs_scheme="pairwise"`): All C(n, k) combinations.

```
k=2, 12 qubits:
  [q0,q1], [q0,q2], ..., [q0,q11], [q1,q2], ..., [q10,q11]
  --> C(12,2) = 66 groups

k=3, 12 qubits:
  [q0,q1,q2], [q0,q1,q3], ..., [q9,q10,q11]
  --> C(12,3) = 220 groups
```

**The key metric is the observation ratio:** `n_obs / latent_dim`. A higher ratio means the quantum circuit produces more information per forward pass.

| Configuration | n_obs | ratio (D=32) |
|---|---|---|
| k=2, sliding, 12q | 11 | 0.34 |
| k=2, pairwise, 12q | 66 | 2.06 |
| k=3, pairwise, 12q | 220 | 6.88 |
| k=2, pairwise, 8q | 28 | 0.88 |

**Pairwise k=2** on 12 qubits gives a ratio of 2.06, meaning the quantum circuit produces ~2x the information needed per latent dimension. **Pairwise k=3** pushes this to 6.88, approaching the classical baseline's effective ratio of ~8.0.

#### Learnable Hermitian Matrix

For each wire group, we construct a K x K Hermitian matrix (K = 2^k) from learnable real parameters:

```
H = h + h^dagger

where h is a lower-triangular matrix built from:
  - D: K real values (diagonal entries)
  - A: K*(K-1)/2 real values (real part of off-diagonal)
  - B: K*(K-1)/2 real values (imaginary part of off-diagonal)
```

| k_local | K = 2^k | Params per group (A + B + D) | Example: 12q pairwise |
|---|---|---|---|
| 2 | 4 | 6 + 6 + 4 = 16 | 66 x 16 = 1,056 |
| 3 | 8 | 28 + 28 + 8 = 64 | 220 x 64 = 14,080 |

The constructed Hermitian matrix `H` has **learnable eigenvalues**. Unlike fixed PauliZ (eigenvalues {-1, +1}), the ANO eigenvalues can expand to arbitrary range during training (typically [-10, +10] or larger). This expandable spectral range is crucial for the expressiveness of the measurement (Chen et al., 2025).

#### Measurement

For each wire group `w`, we measure:

```
<psi| H_w |psi>
```

where `|psi>` is the quantum state after encoding + VQC, and `H_w` is the learnable Hermitian on that wire group.

This produces **n_obs real-valued expectation values**, which form the quantum output vector `q_out`.

### Step 6.6: Classical Post-Processing (vel_head)

The n_obs-dimensional quantum output is mapped to the D-dimensional velocity through a two-layer MLP:

```
q_out (n_obs) --> Linear(n_obs, hidden) --> SiLU --> Linear(hidden, D) --> velocity (D)
```

The hidden dimension is configurable via `--vel-head-hidden`. When set to 0 (default), it auto-selects `max(256, n_obs)` to ensure the head is at least as wide as the quantum output.

**No skip connection.** The classical input `z_combined` is NOT fed into `vel_head`. This is a deliberate design choice (see Section 14).

---

## 7. Data Re-Uploading

**Motivation:** In standard mode (`n_reupload=1`), the quantum circuit encodes input data once, then applies VQC layers. The accessible Fourier frequency spectrum of the circuit output is limited by a single encoding round. With pairwise k=2 at ratio=2.06 (or k=3 at ratio=6.88), the bottleneck shifts from measurement count to **per-channel expressivity** -- each ANO expectation value is too simple a function of the input.

Data re-uploading (Perez-Salinas et al., 2020; Schuld et al., 2021) interleaves encoding and variational layers, multiplying the accessible Fourier frequency spectrum by L (the number of re-upload rounds). This directly addresses the expressivity bottleneck.

### Circuit Structure

**Standard (n_reupload=1):**

```
|0> --> [Encode_1] --> [Encode_2] --> ... --> [Encode_B] --> [VQC_1] --> [VQC_2] --> ... --> [VQC_D] --> [Measure]
        |<---------- n_blocks ----------->|  |<---------- vqc_depth ------------>|
```

All encoding happens first, then all VQC layers. The circuit can represent Fourier frequencies up to the encoding depth.

**Re-uploading (n_reupload=L > 1):**

```
|0> --> [Encode(x)] --> [VQC(theta_1)] --> [Encode(x)] --> [VQC(theta_2)] --> ... --> [Encode(x)] --> [VQC(theta_L)] --> [Measure]
        |<---- round 1 ---->|              |<---- round 2 ---->|                     |<---- round L ---->|
```

Each round uses the **same input data** projected to different encoding parameters (via `enc_proj`), with **independent VQC parameters** per round. This is the standard re-uploading pattern from the literature.

### Parameter Changes with Re-Uploading

When `n_reupload > 1`:

| Parameter | Standard (n_reupload=1) | Re-uploading (n_reupload=L) |
|---|---|---|
| Encoding params | `n_blocks * enc_per_block` | `L * enc_per_block` |
| VQC params shape | `(vqc_depth, ...)` | `(L, ...)` |
| n_blocks, vqc_depth | Used | Ignored |
| Circuit structure | Sequential encode then VQC | Interleaved encode + VQC |

### Fourier Expressivity

A quantum circuit with L re-upload rounds of angle encoding can represent functions with Fourier frequencies up to L (Schuld et al., 2021). With SU(4) encoding, the expressivity gains are even greater since each encoding round contributes 15 parameters per 2-qubit gate instead of 1.

| n_reupload | Fourier frequencies | VQC param layers | total_enc (12q, SU(4)) |
|---|---|---|---|
| 1 (standard) | ~1 | vqc_depth | 330 (2 blocks) |
| 2 | ~2 | 2 | 330 |
| 4 | ~4 | 4 | 660 |

### Usage

```bash
# Standard mode (backward compatible)
python models/QuantumLatentCFM.py --n-reupload=1 --n-blocks=2 --vqc-depth=2

# Re-uploading with 4 rounds
python models/QuantumLatentCFM.py --n-reupload=4 --vqc-type=qvit --qvit-circuit=butterfly

# Re-uploading with angle encoding
python models/QuantumLatentCFM.py --n-reupload=3 --encoding-type=angle --vqc-type=qvit
```

### Compatibility Notes

- `--n-reupload=1` (default) preserves the exact existing code path
- When `n_reupload > 1`, `--n-blocks` and `--vqc-depth` are ignored
- QCNN is excluded from re-uploading (prints warning, falls back to n_reupload=1)
- QViT and HWE are fully supported
- All existing checkpoints and scripts continue to work

---

## 8. Classical MLP Baseline

A classical MLP velocity field is available via `--velocity-field=classical` for comparing quantum vs classical performance. It follows the same interface as the quantum velocity field.

### Architecture

```
t (scalar) --> [Sinusoidal embed (64-dim)] --> [time_mlp: FC(64)->SiLU->FC(64)] --> t_emb (64)

concat(z_t, t_emb) (D+64) --> FC(D+64, 256) --> SiLU
                            --> FC(256, 256)   --> SiLU
                            --> FC(256, 256)   --> SiLU
                            --> FC(256, D)     --> velocity (D)
```

Hidden dimensions are configurable via `--mlp-hidden-dims` (default: `256,256,256`).

The classical baseline uses ~200K parameters for D=32, providing a strong reference point. If the quantum velocity field cannot approach the classical baseline's loss, the quantum circuit's expressivity is the bottleneck.

---

## 9. CFM Training Objective

For each training batch of images `x`:

1. **Encode** images to latent space: `z_1 = VAE.encode(x).mu` (deterministic, no sampling)
2. **Sample noise:** `z_0 ~ N(0, I)` of same shape as `z_1`
3. **Sample time:** `t ~ Uniform(0, 1)` independently per sample
4. **Interpolate:** `z_t = (1 - t) * z_0 + t * z_1` (OT path)
5. **Target velocity:** `u = z_1 - z_0` (constant along the straight-line path)
6. **Predict velocity:** `v_pred = VelocityField(z_t, t)`
7. **Loss:** `L = MSE(v_pred, u) = (1/B) * sum || v_pred_i - u_i ||^2`

```
L_CFM = E_{t, z_0, z_1} [ || v_theta(z_t, t) - (z_1 - z_0) ||^2 ]
```

This is the standard OT-CFM loss (Lipman et al., 2023) applied in the VAE's latent space.

---

## 10. Dual Optimizer Strategy

Following Chen et al. (2025), QLCFM uses two separate Adam optimizers with different learning rates and cosine annealing schedules:

| Parameter Group | Examples | Learning Rate | Rationale |
|---|---|---|---|
| **Circuit parameters** | enc_proj, VQC params, vel_head | lr = 1e-3 | Standard rate for circuit/classical params |
| **Observable parameters** | A.*, B.*, D.* (ANO Hermitians) | lr_H = 1e-1 (100x) | ANO eigenvalues must expand quickly for circuit params to learn |

**Why 100x for observables?** The circuit parameters control the quantum state, but the gradient signal depends on the measurement operator's eigenvalue range. With fixed PauliZ (eigenvalues {-1, +1}), the output range is [-1, +1], limiting gradient magnitude. By using learnable observables with a high learning rate, the eigenvalue range expands rapidly (e.g., to [-10, +10]), amplifying gradients for the circuit parameters.

Both optimizers use cosine annealing to `eta_min=0` over the total number of epochs.

---

## 11. Generation via Euler ODE

After training, we generate new images by numerically integrating the learned velocity field from noise to data:

```python
z = z_0 ~ N(0, I)          # shape: (n_samples, D)
dt = 1.0 / ode_steps       # e.g., 0.01 for 100 steps

for step in range(ode_steps):
    t = step * dt           # t goes from 0.0 to 0.99
    v = v_theta(z, t)       # velocity field (quantum or classical)
    z = z + dt * v          # Euler step

images = VAE.decode(z)      # decode latent to image
images = images.clamp(0, 1) # ensure valid pixel range
```

This is the **forward Euler method** for ODE integration. More sophisticated integrators (RK4, adaptive step) could improve sample quality but Euler is standard for flow matching.

**Step count:** 100 steps is typical (`--ode-steps=100`). More steps = better quality but slower generation.

---

## 12. FID and Inception Score Evaluation

When `--compute-metrics` is passed, QLCFM computes two standard generative model metrics after Phase 2 training:

### Frechet Inception Distance (FID)

Measures the distance between the distribution of generated images and real images in InceptionV3 feature space. Lower is better.

```
FID = ||mu_real - mu_gen||^2 + Tr(Sigma_real + Sigma_gen - 2*(Sigma_real @ Sigma_gen)^{1/2})
```

### Inception Score (IS)

Measures both quality (how confident the classifier is) and diversity (how spread the label distribution is). Higher is better.

```
IS = exp(E_x [KL(p(y|x) || p(y))])
```

### Usage

```bash
python models/QuantumLatentCFM.py --phase=2 --compute-metrics --n-eval-samples=1024
```

Generates `--n-eval-samples` images via Euler ODE, computes FID against the training set, and saves results to `results/metrics_<job_id>.json`.

---

## 13. Parameter Breakdown

### Example 1: 12 qubits, QViT butterfly, pairwise k=2, n_reupload=1

| Component | Shape | Parameters | Type |
|---|---|---|---|
| enc_proj (layer 1) | Linear(64, 256) | 16,640 | Classical |
| enc_proj (layer 2) | Linear(256, 330) | 84,810 | Classical |
| qvit_params | (2, 24, 12) | 576 | Quantum (QViT) |
| ANO: A (66 groups) | 66 x (6,) | 396 | Observable |
| ANO: B (66 groups) | 66 x (6,) | 396 | Observable |
| ANO: D (66 groups) | 66 x (4,) | 264 | Observable |
| vel_head (layer 1) | Linear(66, 256) | 17,152 | Classical |
| vel_head (layer 2) | Linear(256, 32) | 8,224 | Classical |
| **Total** | | **~128K** | |
| Quantum params only | | **576** | QViT |
| Observable params only | | **1,056** | ANO |

### Example 2: 12 qubits, QViT butterfly, pairwise k=2, n_reupload=4

| Component | Shape | Parameters | Type |
|---|---|---|---|
| enc_proj (layer 1) | Linear(64, 256) | 16,640 | Classical |
| enc_proj (layer 2) | Linear(256, 660) | 169,620 | Classical |
| qvit_params | (4, 24, 12) | 1,152 | Quantum (QViT) |
| ANO (66 groups) | | 1,056 | Observable |
| vel_head | FC(66, 256, 32) | 25,376 | Classical |
| **Total** | | **~214K** | |
| Quantum params only | | **1,152** | QViT (2x standard) |

### Example 3: 12 qubits, QViT butterfly, pairwise k=3, n_reupload=1

| Component | Shape | Parameters | Type |
|---|---|---|---|
| enc_proj | FC(64, 256, 330) | 101,450 | Classical |
| qvit_params | (2, 24, 12) | 576 | Quantum |
| ANO (220 groups, k=3) | 220 x 64 | 14,080 | Observable |
| vel_head | FC(220, 256, 32) | 64,544 | Classical |
| **Total** | | **~181K** | |
| Observable params only | | **14,080** | ANO (dominant!) |

**Observation:** With pairwise k=3, the ANO parameters dominate the parameter budget. The 220 learnable 8x8 Hermitian matrices account for ~8% of total params but produce 220 measurement channels.

### Classical MLP Baseline (D=32)

| Component | Parameters |
|---|---|
| time_mlp | 8,320 |
| net (3 hidden layers) | 198,944 |
| **Total** | **~207K** |

---

## 14. Design Decisions

### Why No Classical Skip Path?

The velocity field was originally implemented with a skip connection:

```python
# OLD (with skip):
combined = concat(q_out, z_combined)    # n_obs + 2D dims
velocity = vel_head(combined)
```

In this design, the classical skip path could dominate, potentially learning to ignore the quantum features entirely.

The current design removes the skip:

```python
# CURRENT (no skip):
velocity = vel_head(q_out)              # FC(n_obs -> hidden -> D)
```

**All D dimensions** of the velocity must be predicted from just n_obs quantum expectation values. This forces the quantum circuit to be genuinely responsible for the velocity prediction.

**Empirical evidence:** Removing the skip increased quantum parameter gradients by ~15x.

### Why SU(4) Instead of Angle Encoding?

| Encoding | Params/gate | Expressibility | Circuit depth |
|---|---|---|---|
| Angle (RY) | 1 | Limited (product states only without entangling) | O(n) |
| SU(4) | 15 | Maximum (any 2-qubit unitary) | O(n) |

SU(4) encoding can generate arbitrary 2-qubit unitaries, exploring the full Hilbert space. The exponential map parameterization traces geodesics on SU(N) (Wiersema et al., 2024).

### Why QViT Over QCNN for Re-Uploading?

QCNN uses mid-circuit measurement and pooling, which:
- Reduces qubits each layer (irreversible)
- Requires large memory for backpropagation
- Cannot be interleaved with re-encoding (pooled qubits are gone)

QViT preserves all qubits throughout, making it naturally compatible with re-uploading. The butterfly topology provides O(N log N) all-to-all connectivity in O(log N) depth.

### Why Pairwise Over Sliding Observable Scheme?

| Scheme | Formula | n_obs (12q, k=2) | Ratio (D=32) |
|---|---|---|---|
| Sliding | `n - k + 1` | 11 | 0.34 |
| Pairwise | `C(n, k)` | 66 | 2.06 |

Sliding produces `O(n)` measurements while pairwise produces `O(n^k)`. The pairwise scheme captures correlations between **all** qubit pairs, not just adjacent ones. This is critical for extracting enough information from the quantum state to predict a 32-dimensional velocity vector.

**Experimental evidence:** Pairwise k=2 (ratio 2.06) achieves ~3x lower loss than sliding k=2 (ratio 0.34) on CIFAR-10.

### Why Data Re-Uploading?

With pairwise k=2 at ratio=2.06, the measurement count is sufficient but each expectation value `<H_w>` is a relatively simple function of the input. Re-uploading multiplies the Fourier spectrum by L rounds, making each channel a richer function of the input data.

This is analogous to increasing the depth of a classical neural network -- more layers allow more complex function compositions.

### Why Learnable Observables (ANO) Instead of Fixed Pauli?

Fixed PauliZ measurement restricts output to [-1, +1]. Learnable Hermitian observables have an **expandable eigenvalue range** that grows during training (we observed ranges like [-10, +10]). This larger range:
- Amplifies gradient signal for circuit parameters
- Increases the model's representational capacity
- Is essential for the dual optimizer strategy to work (Chen et al., 2025)

---

## 15. CLI Reference

### Phase Control

| Flag | Values | Description |
|---|---|---|
| `--phase` | `1`, `2`, `generate`, `both` | Training phase |
| `--velocity-field` | `quantum`, `classical` | Velocity field type |

### VAE Configuration

| Flag | Default | Description |
|---|---|---|
| `--vae-arch` | `resconv` | `resconv` (2.1M) or `legacy` (530K) |
| `--latent-dim` | `256` | Latent space dimensionality |
| `--beta` | `0.5` | KL divergence weight |
| `--beta-warmup-epochs` | `20` | Linear beta warmup epochs |
| `--lambda-perc` | `0.1` | VGG perceptual loss weight (0=disable) |

### Quantum Circuit

| Flag | Default | Description |
|---|---|---|
| `--n-qubits` | `8` | Number of qubits |
| `--n-blocks` | `2` | SU(4) encoding blocks (ignored when n_reupload>1) |
| `--encoding-type` | `sun` | `sun` (SU(4)) or `angle` (RY) |
| `--vqc-type` | `qcnn` | `qcnn`, `qvit`, `hardware_efficient`, `none` |
| `--vqc-depth` | `2` | VQC layers (ignored when n_reupload>1) |
| `--qvit-circuit` | `butterfly` | QViT topology: `butterfly`, `pyramid`, `x` |
| `--k-local` | `2` | Observable locality (2 or 3) |
| `--obs-scheme` | `sliding` | `sliding` or `pairwise` |
| `--vel-head-hidden` | `0` | vel_head hidden dim (0=auto: max(256, n_obs)) |
| `--n-reupload` | `1` | Data re-upload rounds (1=standard, >1=interleaved) |

### Training

| Flag | Default | Description |
|---|---|---|
| `--lr` | `1e-3` | Learning rate (circuit + classical params) |
| `--lr-H` | `1e-1` | Learning rate (ANO observable params, 100x) |
| `--lr-vae` | `1e-3` | VAE learning rate |
| `--batch-size` | `64` | Batch size |
| `--epochs` | `200` | Training epochs |
| `--seed` | `2025` | Random seed |

### Data

| Flag | Default | Description |
|---|---|---|
| `--dataset` | `cifar10` | `cifar10`, `coco`, `mnist`, `fashion` |
| `--n-train` | `10000` | Training samples |
| `--n-valtest` | `2000` | Validation + test samples |

### Evaluation & I/O

| Flag | Default | Description |
|---|---|---|
| `--compute-metrics` | `false` | Compute FID & IS after training |
| `--n-eval-samples` | `1024` | Samples for FID/IS computation |
| `--ode-steps` | `100` | Euler ODE integration steps |
| `--job-id` | `qlcfm_001` | Unique experiment identifier |
| `--resume` | `false` | Resume from checkpoint |

---

## 16. References

1. **Lipman, Y.** et al. (2023). Flow Matching for Generative Modeling. *ICLR 2023*. arXiv:2210.02747
2. **Wiersema, R.** et al. (2024). Here comes the SU(N): multivariate quantum gates and gradients. *Quantum*, 8, 1275.
3. **Chen, S. Y.-C.** et al. (2025). Learning to Measure: Adaptive Informationally Complete Generalized Measurements for Quantum Neural Networks. *ICASSP 2025 Workshop*. arXiv:2501.05663
4. **Lin, H.-Y.** et al. (2025). Adaptive Non-local Observable on Quantum Neural Networks. *IEEE QCE 2025*. arXiv:2504.13414
5. **Rombach, R.** et al. (2022). High-Resolution Image Synthesis with Latent Diffusion Models. *CVPR 2022*. arXiv:2112.10752
6. **Esser, P.** et al. (2024). Scaling Rectified Flow Transformers for High-Resolution Image Synthesis. *ICML 2024*. (Stable Diffusion 3)
7. **Vaswani, A.** et al. (2017). Attention Is All You Need. *NeurIPS 2017*. (sinusoidal embeddings)
8. **Sim, S.** et al. (2019). Expressibility and Entangling Capability of Parameterized Quantum Circuits. *Adv. Quantum Technol.*, 2, 1900070.
9. **Perez-Salinas, A.** et al. (2020). Data re-uploading for a universal quantum classifier. *Quantum*, 4, 226. arXiv:1907.02085
10. **Schuld, M.** et al. (2021). Effect of data encoding on the expressive power of variational quantum-machine-learning models. *Phys. Rev. A*, 103, 032430. arXiv:2008.08605
