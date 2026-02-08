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
6. [Phase 2: Quantum Velocity Field](#6-phase-2-quantum-velocity-field)
   - [Step 6.1: Time Embedding](#step-61-time-embedding)
   - [Step 6.2: Classical Pre-Processing (enc_proj)](#step-62-classical-pre-processing-enc_proj)
   - [Step 6.3: SU(4) Quantum Encoding](#step-63-su4-quantum-encoding)
   - [Step 6.4: Quantum Convolutional Neural Network (QCNN)](#step-64-quantum-convolutional-neural-network-qcnn)
   - [Step 6.5: Adaptive Non-Local Observable (ANO) Measurement](#step-65-adaptive-non-local-observable-ano-measurement)
   - [Step 6.6: Classical Post-Processing (vel_head)](#step-66-classical-post-processing-vel_head)
7. [CFM Training Objective](#7-cfm-training-objective)
8. [Dual Optimizer Strategy](#8-dual-optimizer-strategy)
9. [Generation via Euler ODE](#9-generation-via-euler-ode)
10. [Parameter Breakdown](#10-parameter-breakdown)
11. [Design Decisions](#11-design-decisions)
12. [References](#12-references)

---

## 1. What Problem Does QLCFM Solve?

QLCFM is a **generative model** for image synthesis. Given a dataset of images (e.g., CIFAR-10), it learns to generate new, unseen images from the same distribution. Unlike classification models that map images to labels, QLCFM maps random noise to realistic images.

The key innovation is that the core generative dynamics -- the velocity field that transports noise into data -- is implemented as a **quantum circuit** using SU(4) encoding, Quantum CNN, and learnable Hermitian observables. All velocity information must pass through this quantum bottleneck.

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

Images like CIFAR-10 live in a 3072-dimensional space (3 x 32 x 32). Quantum circuits with 8 qubits operate in a 256-dimensional Hilbert space but produce only a handful of measurement outputs (e.g., 7 expectation values). Directly learning a velocity field from 7 outputs to 3072 dimensions is infeasible.

**Solution: Latent Flow Matching**

This is the same paradigm used by Stable Diffusion 3 (Esser et al., 2024) and Latent Diffusion Models (Rombach et al., 2022):

1. **Compress** images to a compact latent space using a classical autoencoder (VAE)
2. **Learn** the flow matching velocity field in the latent space
3. **Decompress** generated latent vectors back to images

```
Image Space (3072-dim)  <--VAE-->  Latent Space (128-dim)  <--Flow Matching-->  Noise
     x ~ p_data                         z_1 = Enc(x)                           z_0 ~ N(0,I)
```

The VAE reduces the dimensionality from 3072 to 128, making the quantum velocity field tractable. The flow matching loss operates entirely in latent space.

---

## 4. Full Pipeline Overview

```
============================== QLCFM PIPELINE ==============================

PHASE 1 -- VAE Pretraining (purely classical):

    Image (3, 32, 32)
         |
         v
    [Conv Encoder]  --> (mu, logvar) each of dim 128
         |
    [Reparameterize]  z = mu + std * eps
         |
         v
    [Conv Decoder]  --> Reconstructed Image (3, 32, 32)

    Loss = MSE(x, x_hat) + beta * KL(q(z|x) || N(0,I))


PHASE 2 -- Quantum CFM Training (quantum velocity field):

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
                              |
                              v
                      v_pred (128-dim)
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

========================================================================
```

---

## 5. Phase 1: Convolutional VAE

**Purpose:** Learn a compressed, Gaussian-structured latent space for images.

**Why VAE instead of a plain autoencoder?** The KL divergence term in the VAE loss regularizes the latent space to be approximately N(0, I). This is critical because flow matching starts from N(0, I) noise -- if the latent space is not Gaussian-like, the flow cannot bridge noise and data smoothly.

### Encoder Architecture

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
   +-- Linear(2048, 128) --> mu     (batch, 128)
   |
   +-- Linear(2048, 128) --> logvar (batch, 128)
```

### Reparameterization Trick

```
std = exp(0.5 * logvar)
eps ~ N(0, I)
z = mu + std * eps
```

This allows gradients to flow through the sampling step during backpropagation.

### Decoder Architecture

```
Input: z (batch, 128)
   |
   Linear(128, 2048) + ReLU --> (batch, 2048)
   |
   Reshape --> (batch, 128, 4, 4)
   |
   ConvTranspose2d(128, 64, kernel=4, stride=2, pad=1) --> (batch, 64, 8, 8)
   BatchNorm2d(64) + ReLU
   |
   ConvTranspose2d(64, 32, kernel=4, stride=2, pad=1)  --> (batch, 32, 16, 16)
   BatchNorm2d(32) + ReLU
   |
   ConvTranspose2d(32, 3, kernel=4, stride=2, pad=1)   --> (batch, 3, 32, 32)
   Sigmoid  (output in [0, 1])
```

### VAE Loss Function

```
L_VAE = L_recon + beta * L_KL

L_recon = MSE(x_hat, x)                                     (pixel-level reconstruction)
L_KL    = -0.5 * mean(1 + logvar - mu^2 - exp(logvar))      (KL divergence to N(0,I))
```

- **L_recon** ensures the decoder can reconstruct images from latent codes
- **L_KL** ensures the encoder maps data to a distribution close to N(0, I)
- **beta = 0.5** balances reconstruction quality vs latent space regularity

### Phase 1 Training

- **Optimizer:** Adam, lr = 1e-3
- **Epochs:** 200 (typical)
- **Output:** Pretrained VAE weights saved to `weights_vae_<job_id>.pt`

After Phase 1, the VAE is **frozen** -- its weights are never updated again.

---

## 6. Phase 2: Quantum Velocity Field

This is the core of QLCFM. The velocity field `v_theta(z_t, t)` predicts the velocity that transports noise towards data in latent space. **Every velocity prediction must pass through the quantum circuit** -- there is no classical skip path.

### Complete Forward Pass

```
=== QUANTUM VELOCITY FIELD: v_theta(z_t, t) ===

Inputs: z_t (batch, 128),  t (batch,)
                |                |
                |    [Sinusoidal Embedding]
                |                |
                v                v
            z_t (128)      t_emb (128)
                |                |
                +--- concat -----+
                         |
                         v
                z_combined (256)                     <-- Step 6.1-6.2
                         |
                    [enc_proj]
                  FC(256, 256) + SiLU
                  FC(256, 210)
                         |
                         v
                enc_params (210)                     <-- encoding parameters
                         |
           ==============|==============
           |    QUANTUM CIRCUIT (8 qubits)    |
           |                                  |
           |  [SU(4) Encoding]  (Step 6.3)    |
           |    2 blocks x 7 gates x 15 params|
           |         |                        |
           |  [QCNN]  (Step 6.4)              |
           |    Layer 1: conv(8 qubits)       |
           |             pool -> 4 qubits     |
           |    Layer 2: conv(4 qubits)       |
           |             pool -> 2 qubits     |
           |         |                        |
           |  [ANO Measurement]  (Step 6.5)   |
           |    7 learnable Hermitian obs      |
           |    on sliding 2-local groups     |
           |         |                        |
           =========|=========================
                    |
                    v
               q_out (7)                             <-- quantum output
                    |
               [vel_head]                            <-- Step 6.6
             FC(7, 256) + SiLU
             FC(256, 128)
                    |
                    v
             velocity v (128)                        <-- output
```

### Step 6.1: Time Embedding

The scalar time `t in [0, 1]` is embedded into a 128-dimensional vector using sinusoidal positional encoding (Vaswani et al., 2017):

```
t_emb[i] = cos(t * 10000^(-i/64))    for i = 0, ..., 63
t_emb[i] = sin(t * 10000^(-(i-64)/64))  for i = 64, ..., 127
```

**Why?** A scalar `t` carries very little information. The sinusoidal embedding maps it to a rich, high-dimensional representation where nearby times have similar embeddings but distant times are distinguishable. This allows the network to learn different behavior at different flow times.

**Code reference:** `sinusoidal_embedding()` (line 163)

### Step 6.2: Classical Pre-Processing (enc_proj)

The time embedding is concatenated with the noisy latent vector:

```
z_combined = concat(z_t, t_emb)     shape: (batch, 256)
```

Then a two-layer MLP maps this 256-dimensional input to the quantum encoding parameters:

```
z_combined (256) --> Linear(256, 256) --> SiLU --> Linear(256, 210) --> enc_params (210)
```

**Why 210?** With 8 qubits and 2 SU(4) encoding blocks:
- Even pairs: (0,1), (2,3), (4,5), (6,7) = 4 gates
- Odd pairs: (1,2), (3,4), (5,6) = 3 gates
- Total gates per block: 7
- Parameters per SU(4) gate: 15 (dimension of su(4) Lie algebra)
- Total: 2 blocks x 7 gates x 15 params = **210 parameters**

The `enc_proj` learns to transform the latent+time information into parameters that control the quantum state preparation. This is the **data-encoding** step -- it determines what quantum state the circuit prepares.

**Code reference:** `self.enc_proj` (line 375)

### Step 6.3: SU(4) Quantum Encoding

The 210 encoding parameters from `enc_proj` are used to prepare a quantum state on 8 qubits via **SU(4) exponential map encoding** (Wiersema et al., 2024).

**SU(4) gate:** `qml.SpecialUnitary(params, wires=[q1, q2])` implements a general 2-qubit unitary gate parameterized by 15 real numbers (the generators of the SU(4) Lie algebra):

```
U(theta) = exp(i * sum_{k=1}^{15} theta_k * G_k)
```

where `G_k` are the 15 generalized Gell-Mann matrices (a basis for the su(4) Lie algebra).

**Brick-wall layout (2 blocks):**

```
Block 1:
  Even layer:  SU4[q0,q1]  SU4[q2,q3]  SU4[q4,q5]  SU4[q6,q7]   (4 gates, 60 params)
  Odd layer:   SU4[q1,q2]  SU4[q3,q4]  SU4[q5,q6]               (3 gates, 45 params)

Block 2:
  Even layer:  SU4[q0,q1]  SU4[q2,q3]  SU4[q4,q5]  SU4[q6,q7]   (4 gates, 60 params)
  Odd layer:   SU4[q1,q2]  SU4[q3,q4]  SU4[q5,q6]               (3 gates, 45 params)

Total: 14 SU(4) gates, 210 parameters
```

**Why SU(4)?** Unlike simpler encodings (e.g., angle embedding with RY gates), SU(4) gates can generate any 2-qubit unitary, enabling the circuit to explore the full Hilbert space. The exponential map traces **geodesics** on the SU(N) manifold, providing a geometrically natural encoding.

**Critical property:** The encoding parameters are **data-dependent** -- they change for every input sample. This means the quantum state prepared is a function of `(z_t, t)`, allowing the circuit to represent different velocities for different inputs.

**Code reference:** Lines 434-445

### Step 6.4: Quantum Convolutional Neural Network (QCNN)

After encoding, the quantum state is processed by a QCNN with learnable (data-independent) parameters. The QCNN alternates convolutional and pooling layers, progressively reducing the number of active qubits.

#### Convolutional Layer

Each convolutional layer applies staggered 2-qubit blocks across all pairs of adjacent qubits. Each block contains:

```
For qubit pair (w1, w2):
  U3(theta_1, theta_2, theta_3, wires=w1)           -- local rotation on w1
  U3(theta_4, theta_5, theta_6, wires=w2)           -- local rotation on w2
  IsingZZ(theta_7, wires=[w1, w2])                   -- ZZ entangling gate
  IsingYY(theta_8, wires=[w1, w2])                   -- YY entangling gate
  IsingXX(theta_9, wires=[w1, w2])                   -- XX entangling gate
  U3(theta_10, theta_11, theta_12, wires=w1)         -- local rotation on w1
  U3(theta_13, theta_14, theta_15, wires=w2)         -- local rotation on w2
```

**15 parameters per 2-qubit block.** This is a highly expressive parameterization that can represent a wide variety of 2-qubit operations.

The convolution is applied in two passes (even parity, then odd parity) to ensure all adjacent pairs interact:

```
Pass 1 (even):  [q0,q1]  [q2,q3]  [q4,q5]  [q6,q7]
Pass 2 (odd):   [q1,q2]  [q3,q4]  [q5,q6]
```

#### Pooling Layer

Pooling reduces the number of active qubits by half. For each pair of adjacent qubits:

1. **Measure** the odd-indexed qubit via mid-circuit measurement
2. **Conditionally apply** a U3 rotation on the even-indexed qubit based on the measurement outcome

```
For qubit pair (q_{i-1}, q_i) where i is odd:
  m = measure(q_i)
  if m == 1:
    U3(phi_1, phi_2, phi_3, wires=q_{i-1})
```

After pooling, only the even-indexed qubits remain active.

#### QCNN with 2 Layers on 8 Qubits

```
Input: 8 active qubits [q0, q1, q2, q3, q4, q5, q6, q7]

Layer 1:
  Conv: 7 two-qubit blocks x 15 params = 105 params
  Pool: measure q1, q3, q5, q7 --> 4 active qubits [q0, q2, q4, q6]
        4 conditional U3 gates x 3 params = 12 params

Layer 2:
  Conv: 3 two-qubit blocks x 15 params = 45 params
  Pool: measure q2, q6 --> 2 active qubits [q0, q4]
        2 conditional U3 gates x 3 params = 6 params

Total QCNN params: 105 + 12 + 45 + 6 = 168 trainable parameters
```

**Important:** QCNN parameters are **data-independent** (shared across all samples). They are trainable weights of the model, analogous to convolutional filters in a classical CNN. Only the encoding parameters change per sample.

**Code reference:** Lines 457-485

### Step 6.5: Adaptive Non-Local Observable (ANO) Measurement

Instead of measuring fixed Pauli operators (like PauliZ on each qubit), QLCFM uses **learnable Hermitian observables** that adapt during training (Chen et al., 2025; Lin et al., 2025).

#### Wire Groups (k-local, sliding)

With `k_local = 2` and `obs_scheme = "sliding"` on 8 qubits:

```
Group 0: [q0, q1]
Group 1: [q1, q2]
Group 2: [q2, q3]
Group 3: [q3, q4]
Group 4: [q4, q5]
Group 5: [q5, q6]
Group 6: [q6, q7]
```

**7 groups**, each measuring a 2-qubit subsystem.

#### Learnable Hermitian Matrix

For each wire group, we construct a 4x4 Hermitian matrix (K = 2^k = 4) from learnable real parameters:

```
H = h + h^dagger

where h is a lower-triangular matrix built from:
  - D: K = 4 real values (diagonal entries)
  - A: K*(K-1)/2 = 6 real values (real part of off-diagonal)
  - B: K*(K-1)/2 = 6 real values (imaginary part of off-diagonal)

Total per group: 4 + 6 + 6 = 16 learnable parameters
Total for 7 groups: 7 x 16 = 112 ANO parameters
```

The constructed Hermitian matrix `H` has **learnable eigenvalues**. Unlike fixed PauliZ (eigenvalues {-1, +1}), the ANO eigenvalues can expand to arbitrary range during training. This expandable spectral range is crucial for the expressiveness of the measurement (Chen et al., 2025).

#### Measurement

For each wire group `w`, we measure:

```
<psi| H_w |psi>
```

where `|psi>` is the quantum state after encoding + QCNN, and `H_w` is the learnable Hermitian on that wire group.

This produces **7 real-valued expectation values**, which form the quantum output vector `q_out`.

**Code reference:** Lines 391-409, 496-501

### Step 6.6: Classical Post-Processing (vel_head)

The 7-dimensional quantum output is mapped to the 128-dimensional velocity through a small MLP:

```
q_out (7) --> Linear(7, 256) --> SiLU --> Linear(256, 128) --> velocity (128)
```

**No skip connection.** The classical input `z_combined` is NOT fed into `vel_head`. This is a deliberate design choice (see Section 11).

**Code reference:** Lines 412-418, 535

---

## 7. CFM Training Objective

For each training batch of images `x`:

1. **Encode** images to latent space: `z_1 = VAE.encode(x).mu` (deterministic, no sampling)
2. **Sample noise:** `z_0 ~ N(0, I)` of same shape as `z_1`
3. **Sample time:** `t ~ Uniform(0, 1)` independently per sample
4. **Interpolate:** `z_t = (1 - t) * z_0 + t * z_1` (OT path)
5. **Target velocity:** `u = z_1 - z_0` (constant along the straight-line path)
6. **Predict velocity:** `v_pred = QuantumVelocityField(z_t, t)`
7. **Loss:** `L = MSE(v_pred, u) = (1/B) * sum || v_pred_i - u_i ||^2`

```
L_CFM = E_{t, z_0, z_1} [ || v_theta(z_t, t) - (z_1 - z_0) ||^2 ]
```

This is the standard OT-CFM loss (Lipman et al., 2023) applied in the VAE's latent space.

**Code reference:** Lines 734-746

---

## 8. Dual Optimizer Strategy

Following Chen et al. (2025), QLCFM uses two separate Adam optimizers with different learning rates:

| Parameter Group | Examples | Learning Rate | Rationale |
|---|---|---|---|
| **Circuit parameters** | enc_proj, conv_params, pool_params, vel_head | lr = 1e-3 | Standard rate for circuit/classical params |
| **Observable parameters** | A.*, B.*, D.* (ANO Hermitians) | lr_H = 1e-1 (100x) | ANO eigenvalues must expand quickly for circuit params to learn |

**Why 100x for observables?** The circuit parameters control the quantum state, but the gradient signal depends on the measurement operator's eigenvalue range. With fixed PauliZ (eigenvalues {-1, +1}), the output range is [-1, +1], limiting gradient magnitude. By using learnable observables with a high learning rate, the eigenvalue range expands rapidly (e.g., to [-10, +10]), amplifying gradients for the circuit parameters.

**Code reference:** Lines 680-689

---

## 9. Generation via Euler ODE

After training, we generate new images by numerically integrating the learned velocity field from noise to data:

```python
z = z_0 ~ N(0, I)          # shape: (n_samples, 128)
dt = 1.0 / ode_steps       # e.g., 0.01 for 100 steps

for step in range(ode_steps):
    t = step * dt           # t goes from 0.0 to 0.99
    v = v_theta(z, t)       # quantum velocity field
    z = z + dt * v          # Euler step

images = VAE.decode(z)      # decode latent to image
```

This is the **forward Euler method** for ODE integration. More sophisticated integrators (RK4, adaptive step) could improve sample quality but Euler is standard for flow matching.

**Step count:** 100 steps is typical. More steps = better quality but slower generation.

**Code reference:** Lines 816-842

---

## 10. Parameter Breakdown

### Default Configuration (8 qubits, 2 SU(4) blocks, QCNN depth 2, k_local=2)

#### Phase 1: ConvVAE (~530K parameters)

| Component | Shape | Parameters |
|---|---|---|
| Encoder Conv layers | 3->32->64->128 | ~100K |
| Encoder BatchNorm | 3 layers | ~448 |
| fc_mu | Linear(2048, 128) | 262,272 |
| fc_logvar | Linear(2048, 128) | 262,272 |
| fc_dec | Linear(128, 2048) | 264,192 |
| Decoder ConvTranspose | 128->64->32->3 | ~100K |
| Decoder BatchNorm | 2 layers | ~192 |
| **Total** | | **~530,435** |

#### Phase 2: Quantum Velocity Field

| Component | Shape | Parameters | Type |
|---|---|---|---|
| enc_proj (layer 1) | Linear(256, 256) | 65,792 | Classical |
| enc_proj (layer 2) | Linear(256, 210) | 53,970 | Classical |
| conv_params | (2, 8, 15) | 240 | Quantum (QCNN) |
| pool_params | (2, 4, 3) | 24 | Quantum (QCNN) |
| ANO: A (7 groups) | 7 x (6,) | 42 | Observable |
| ANO: B (7 groups) | 7 x (6,) | 42 | Observable |
| ANO: D (7 groups) | 7 x (4,) | 28 | Observable |
| vel_head (layer 1) | Linear(7, 256) | 2,048 | Classical |
| vel_head (layer 2) | Linear(256, 128) | 32,896 | Classical |
| **Total** | | **~155K** | |
| Quantum params only | | **376** | 240 + 24 + 112 |
| Observable params only | | **112** | 42 + 42 + 28 |

---

## 11. Design Decisions

### Why No Classical Skip Path?

The velocity field was originally implemented with a skip connection:

```python
# OLD (with skip):
combined = concat(q_out, z_combined)    # 7 + 256 = 263 dims
velocity = vel_head(combined)            # FC(263 -> 128)
```

In this design, only 2.7% (7/263) of the `vel_head` input came from the quantum circuit. The classical skip path (256 dims) could dominate, potentially learning to ignore the quantum features entirely.

The current design removes the skip:

```python
# CURRENT (no skip):
velocity = vel_head(q_out)              # FC(7 -> 256 -> 128)
```

**All 128 dimensions** of the velocity must be predicted from just 7 quantum expectation values. This forces the quantum circuit to be genuinely responsible for the velocity prediction.

**Empirical evidence:** Removing the skip increased quantum parameter gradients by ~15x, confirming that gradient signal now flows more strongly through the quantum circuit.

### Why SU(4) Instead of Angle Encoding?

| Encoding | Params/gate | Expressibility | Circuit depth |
|---|---|---|---|
| Angle (RY) | 1 | Limited (product states only without entangling) | O(n) |
| SU(4) | 15 | Maximum (any 2-qubit unitary) | O(n) |

SU(4) encoding can generate arbitrary 2-qubit unitaries, exploring the full Hilbert space. The exponential map parameterization traces geodesics on SU(N), providing geometrically natural data encoding (Wiersema et al., 2024).

### Why QCNN Instead of Hardware-Efficient Ansatz?

QCNN provides **hierarchical feature extraction** through its conv+pool structure:
- Convolution captures local correlations between adjacent qubits
- Pooling implements coarse-graining, similar to classical CNN max-pooling
- The multi-scale structure is well-suited for processing structured data

Hardware-efficient ansatz is simpler (just RY + CNOT layers) but lacks the hierarchical structure.

### Why Learnable Observables (ANO) Instead of Fixed Pauli?

Fixed PauliZ measurement restricts output to [-1, +1]. Learnable Hermitian observables have an **expandable eigenvalue range** that grows during training (we observed ranges like [-10, +10]). This larger range:
- Amplifies gradient signal for circuit parameters
- Increases the model's representational capacity
- Is essential for the dual optimizer strategy to work (Chen et al., 2025)

---

## 12. References

1. **Lipman, Y.** et al. (2023). Flow Matching for Generative Modeling. *ICLR 2023*. arXiv:2210.02747
2. **Wiersema, R.** et al. (2024). Here comes the SU(N): multivariate quantum gates and gradients. *Quantum*, 8, 1275.
3. **Chen, S. Y.-C.** et al. (2025). Learning to Measure: Adaptive Informationally Complete Generalized Measurements for Quantum Neural Networks. *ICASSP 2025 Workshop*. arXiv:2501.05663
4. **Lin, H.-Y.** et al. (2025). Adaptive Non-local Observable on Quantum Neural Networks. *IEEE QCE 2025*. arXiv:2504.13414
5. **Rombach, R.** et al. (2022). High-Resolution Image Synthesis with Latent Diffusion Models. *CVPR 2022*. arXiv:2112.10752
6. **Esser, P.** et al. (2024). Scaling Rectified Flow Transformers for High-Resolution Image Synthesis. *ICML 2024*. (Stable Diffusion 3)
7. **Vaswani, A.** et al. (2017). Attention Is All You Need. *NeurIPS 2017*. (sinusoidal embeddings)
8. **Sim, S.** et al. (2019). Expressibility and Entangling Capability of Parameterized Quantum Circuits. *Adv. Quantum Technol.*, 2, 1900070.
