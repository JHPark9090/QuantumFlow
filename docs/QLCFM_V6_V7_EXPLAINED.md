# Quantum Latent Conditional Flow Matching: v6 and v7 Explained

This document provides a step-by-step explanation of the v6 (single-circuit quantum)
and v7 (multi-circuit quantum) models implemented in `models/QuantumLatentCFM_v6.py`.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Phase 1: VAE Training (Shared)](#2-phase-1-vae-training-shared)
3. [Phase 2: Conditional Flow Matching](#3-phase-2-conditional-flow-matching)
4. [Time Embedding (Shared Across All Models)](#4-time-embedding-shared-across-all-models)
5. [v6: Single-Circuit Quantum Velocity Field](#5-v6-single-circuit-quantum-velocity-field)
6. [v7: Multi-Circuit Quantum Velocity Field](#6-v7-multi-circuit-quantum-velocity-field)
7. [Quantum Circuit Internals](#7-quantum-circuit-internals)
8. [Generation via Euler ODE](#8-generation-via-euler-ode)
9. [Metrics](#9-metrics)
10. [Parameter Counts](#10-parameter-counts)
11. [Classical-C Baseline](#11-classical-c-baseline)
12. [Key Design Decisions](#12-key-design-decisions)

---

## 1. Overview

The Quantum Latent Conditional Flow Matching (QLCFM) framework generates images
through a two-phase pipeline:

```
Phase 1 (VAE):   Image (32x32x3) ──encoder──> Latent z (32-dim) ──decoder──> Reconstructed Image

Phase 2 (CFM):   Noise z_0 ~ N(0,I) ──velocity field──> Latent z_1 ──decoder──> Generated Image
```

The velocity field in Phase 2 is the component being compared:
- **Classical-C**: A 3-layer MLP
- **v6**: A single 8-qubit quantum circuit
- **v7**: Eight 8-qubit quantum circuits in parallel

Everything else (VAE, data pipeline, time embedding, training loop) is identical
across all three models, making this a controlled experiment.

---

## 2. Phase 1: VAE Training (Shared)

All models share the same Variational Autoencoder. This phase is independent of the
velocity field choice.

### Step-by-step

1. **Input**: A batch of CIFAR-10 images, shape `(B, 3, 32, 32)`, pixel values in `[0, 1]`.

2. **Encoder** (ResConvVAE, ~9.85M params):
   ```
   Image (3, 32, 32)
     -> Conv2d(3, 32, 3x3) + 2x ResidualBlock(32)
     -> Conv2d(32, 64, 4x4, stride=2) + 2x ResidualBlock(64)     # -> (64, 16, 16)
     -> Conv2d(64, 128, 4x4, stride=2) + 2x ResidualBlock(128)   # -> (128, 8, 8)
     -> Conv2d(128, 256, 4x4, stride=2) + 2x ResidualBlock(256)  # -> (256, 4, 4)
     -> BN + ReLU + Conv2d(256, 256, 4x4, stride=2)              # -> (256, 2, 2)
     -> Flatten                                                    # -> 1024
     -> fc_mu: Linear(1024, 32)     ->  mu     (32-dim)
     -> fc_logvar: Linear(1024, 32) ->  logvar (32-dim)
   ```

3. **Reparameterization trick**: `z = mu + exp(0.5 * logvar) * epsilon`, where
   `epsilon ~ N(0, I)`.

4. **Decoder** (mirrors encoder):
   ```
   z (32-dim)
     -> Linear(32, 1024) + ReLU -> reshape to (256, 2, 2)
     -> ConvTranspose2d + 2x ResidualBlock (x4 stages)
     -> Conv2d(32, 3, 3x3) + Sigmoid
     -> Reconstructed image (3, 32, 32)
   ```

5. **Loss function**:
   ```
   L_VAE = MSE(x_hat, x) + beta * KL(q(z|x) || N(0,I)) + lambda * L_perceptual
   ```
   - `MSE`: pixel-wise mean squared error
   - `KL = -0.5 * mean(1 + logvar - mu^2 - exp(logvar))`: Gaussian KL divergence
   - `L_perceptual`: L1 distance in VGG16 feature space (layers relu1_2, 2_2, 3_3, 4_3)
   - `beta` ramps linearly from 0 to 0.5 over 20 warmup epochs
   - `lambda = 0.1`

6. **Optimizer**: Adam with lr=1e-3, cosine annealing to 1e-6 over 200 epochs.

7. **Metrics computed per epoch on validation set**:
   - PSNR (Peak Signal-to-Noise Ratio) — higher is better
   - SSIM (Structural Similarity Index) — higher is better
   - LPIPS (Learned Perceptual Image Patch Similarity) — lower is better

8. **Output**: Trained VAE weights are saved. The encoder's `mu` output provides the
   deterministic latent representations `z_1` used in Phase 2.

---

## 3. Phase 2: Conditional Flow Matching

In Phase 2, the VAE is frozen. A velocity field learns to transport Gaussian noise
`z_0 ~ N(0, I)` to the data latent distribution `z_1 = encoder(x)` via Optimal
Transport Conditional Flow Matching (OT-CFM).

### Training step-by-step (one batch)

1. **Encode real images**: `z_1 = VAE.encode(x)` (mu only, deterministic). Shape: `(B, 32)`.

2. **Sample noise**: `z_0 ~ N(0, I)`. Shape: `(B, 32)`.

3. **Sample time**: `t ~ Uniform(0, 1)`. Shape: `(B,)`.

4. **OT interpolation**: Construct the noisy intermediate:
   ```
   z_t = (1 - t) * z_0 + t * z_1
   ```
   At `t=0`, `z_t = z_0` (pure noise). At `t=1`, `z_t = z_1` (data latent).

5. **Target velocity**: The constant-velocity OT target:
   ```
   target = z_1 - z_0
   ```
   This is the straight-line direction from noise to data.

6. **Predict velocity**: `v_pred = velocity_field(z_t, t)`. Shape: `(B, 32)`.

7. **Loss**:
   ```
   L_CFM = MSE(v_pred, target)
   ```

8. **Dual optimizer** (for quantum models):
   - Circuit parameters (encoding projection, VQC, velocity head): Adam lr=1e-3
   - ANO observable parameters (A, B, D): Adam lr=1e-1 (100x higher)
   - Both use cosine annealing over 200 epochs

This is where v6, v7, and Classical-C differ — in how they compute `v_pred` from
`(z_t, t)`.

---

## 4. Time Embedding (Shared Across All Models)

All velocity fields use the **identical** time conditioning pipeline:

```
t (scalar per sample)
  -> sinusoidal_embedding(t, dim=32)          # 16 cos + 16 sin frequencies
  -> time_mlp: Linear(32, 32) -> SiLU -> Linear(32, 32)
  -> t_emb (32-dim)

z_combined = concat(z_t, t_emb) = (B, 64)    # 32 latent + 32 time = 64
```

The sinusoidal embedding uses log-spaced frequencies (Vaswani et al., 2017):
```
freq_k = exp(-ln(10000) * k / 16)    for k = 0, 1, ..., 15
embedding = [cos(t * freq_0), ..., cos(t * freq_15), sin(t * freq_0), ..., sin(t * freq_15)]
```

The `time_mlp` is a learnable refinement that transforms the fixed sinusoidal features
into task-adapted time representations. This is critical — without it, the velocity
field has a much weaker sense of "where in the flow" it is.

---

## 5. v6: Single-Circuit Quantum Velocity Field

v6 uses **one** 8-qubit quantum circuit as the core of the velocity field.

### Forward pass step-by-step

```
Input: z_t (B, 32), t (B,)

Step 1: Time embedding
  t -> sinusoidal(t, 32) -> time_mlp -> t_emb (B, 32)

Step 2: Concatenate
  z_combined = concat(z_t, t_emb) = (B, 64)

Step 3: Encoding projection (classical, with nonlinearity)
  z_combined (B, 64) -> Linear(64, 256) -> SiLU -> Linear(256, 105) -> enc (B, 105)

  Why 105? SU(4) encoding on 8 qubits requires:
    - Even pairs: (0,1), (2,3), (4,5), (6,7) = 4 pairs x 15 params = 60
    - Odd pairs:  (1,2), (3,4), (5,6)        = 3 pairs x 15 params = 45
    - Total: 105 parameters per encoding block

Step 4: Quantum circuit execution
  enc (B, 105) -> [SU(4) encoding] -> [QViT butterfly VQC, depth=2] -> [ANO measurement]
  -> q_out (B, 28)

  Why 28? Pairwise observables on 8 qubits: C(8, 2) = 28

Step 5: Velocity head (classical, with nonlinearity)
  q_out (B, 28) -> Linear(28, 256) -> SiLU -> Linear(256, 32) -> v_pred (B, 32)

Output: v_pred (B, 32) — predicted velocity in latent space
```

### Data flow diagram

```
z_t (32) ─────────────┐
                       ├─ concat ─> (64) ─> enc_proj ─> (105) ─> [8-qubit circuit] ─> (28) ─> vel_head ─> v (32)
t ─> sin_embed ─> time_mlp ─> t_emb (32) ─┘
```

### Observation ratio

The quantum circuit produces 28 measurements to predict 32 velocity dimensions:
```
ratio = n_obs / latent_dim = 28 / 32 = 0.875
```

This means the velocity head must expand 28 values into 32 — a slight information
bottleneck that the nonlinear `vel_head` helps bridge.

---

## 6. v7: Multi-Circuit Quantum Velocity Field

v7 uses **eight** independent 8-qubit quantum circuits. Each circuit receives the
**full shared input** (not a split).

### Forward pass step-by-step

```
Input: z_t (B, 32), t (B,)

Step 1: Time embedding (identical to v6)
  t -> sinusoidal(t, 32) -> time_mlp -> t_emb (B, 32)

Step 2: Concatenate (identical to v6)
  z_combined = concat(z_t, t_emb) = (B, 64)

Step 3: Run 8 circuits in parallel, each on the FULL shared input
  For circuit k = 0, 1, ..., 7:
    z_combined (B, 64) -> enc_proj_k: Linear(64, 256) -> SiLU -> Linear(256, 105) -> enc_k (B, 105)
    enc_k -> [8-qubit circuit k] -> q_out_k (B, 28)

  Each circuit has its OWN:
    - enc_proj (different learned Linear layers)
    - VQC parameters (different QViT weights)
    - ANO parameters (different A, B, D Hermitian matrices)

Step 4: Concatenate all circuit outputs
  q_all = concat(q_out_0, q_out_1, ..., q_out_7) = (B, 224)

  Why 224? 8 circuits x 28 observables each = 224

Step 5: Velocity head (classical, with nonlinearity)
  q_all (B, 224) -> Linear(224, 256) -> SiLU -> Linear(256, 32) -> v_pred (B, 32)

Output: v_pred (B, 32) — predicted velocity in latent space
```

### Data flow diagram

```
z_t (32) ────────────────┐
                         ├─ concat ─> z_combined (64) ─┬─> enc_proj_0 ─> [circuit 0] ─> (28) ─┐
t ─> sin_embed ─> time_mlp ─> t_emb (32) ─┘           ├─> enc_proj_1 ─> [circuit 1] ─> (28) ─┤
                                                        ├─> enc_proj_2 ─> [circuit 2] ─> (28) ─┤
                                                        ├─> enc_proj_3 ─> [circuit 3] ─> (28) ─┤
                                                        ├─> enc_proj_4 ─> [circuit 4] ─> (28) ─┼─ concat ─> (224) ─> vel_head ─> v (32)
                                                        ├─> enc_proj_5 ─> [circuit 5] ─> (28) ─┤
                                                        ├─> enc_proj_6 ─> [circuit 6] ─> (28) ─┤
                                                        └─> enc_proj_7 ─> [circuit 7] ─> (28) ─┘
```

### Observation ratio

```
ratio = total_obs / latent_dim = 224 / 32 = 7.0
```

With 7x more observables than output dimensions, the `vel_head` has rich quantum
features to work with. Each circuit can specialize in different aspects of the velocity
field.

### Key difference from v5 (previous multi-circuit design)

In v5, the 64-dim input was **split** into 16 chunks of ~4 dims, and each circuit
saw only its chunk. This fragmented both `z_t` and `t_emb`, so no single circuit
had a complete picture of the state or time.

In v7, all 8 circuits see the **full** 64-dim input. Each circuit learns its own
`enc_proj` to extract different 105-dim features from the same input. This is
analogous to multi-head attention — each "head" (circuit) attends to the full
input but learns different projections.

---

## 7. Quantum Circuit Internals

Each quantum circuit (used in both v6 and v7) has three stages:

### Stage 1: SU(4) Encoding

The 105 encoding parameters are loaded into the quantum state using `SpecialUnitary`
gates on pairs of qubits (Wiersema et al., 2024):

```
Qubit pairs (even):  (0,1), (2,3), (4,5), (6,7)   — 4 pairs
Qubit pairs (odd):   (1,2), (3,4), (5,6)           — 3 pairs

Each pair gets a SpecialUnitary gate with 15 parameters (generators of SU(4)).
Total: 7 pairs x 15 = 105 parameters
```

SU(4) gates are the most general 2-qubit unitary operations. They can represent
any quantum transformation on a pair of qubits, making them maximally expressive
for data encoding.

### Stage 2: QViT Butterfly Variational Circuit (depth=2)

The Quantum Vision Transformer butterfly topology (Cherrat et al., 2024) applies
parameterized 2-qubit gates in a butterfly pattern across 2 depth layers:

```
Butterfly pattern on 8 qubits (per depth layer):

Layer 0 (stride=1):   (0,1) (2,3) (4,5) (6,7)         — 4 gates
Layer 1 (stride=2):   (0,2) (1,3) (4,6) (5,7)         — 4 gates
Layer 2 (stride=4):   (0,4) (1,5) (2,6) (3,7)         — 4 gates
                                                Total: 12 gates per depth

With depth=2: 2 x 12 = 24 gates total
Each gate has 12 parameters: 24 x 12 = 288 VQC parameters
```

Each 2-qubit gate applies:
```
U3(p[0], p[1], p[2], wire1)       # Single-qubit rotation
IsingXX(p[3], [wire1, wire2])     # XX entangling
IsingYY(p[4], [wire1, wire2])     # YY entangling
IsingZZ(p[5], [wire1, wire2])     # ZZ entangling
U3(p[6], p[7], p[8], wire1)       # Single-qubit rotation
IsingXX(p[9], [wire1, wire2])     # XX entangling
IsingYY(p[10], [wire1, wire2])    # YY entangling
IsingZZ(p[11], [wire1, wire2])    # ZZ entangling
```

The butterfly pattern ensures all qubits are connected within `log2(8) = 3`
sub-layers per depth, providing efficient long-range entanglement.

### Stage 3: Adaptive Non-Local Observable (ANO) Measurement

Instead of measuring each qubit independently (PauliZ), ANO uses learnable Hermitian
operators on pairs of qubits (Lin et al., 2025; Chen et al., 2025):

```
For each pair (i, j) from C(8, 2) = 28 pairs:

  Construct a 4x4 Hermitian matrix H_ij:
    H = U * diag(D) * U^dagger

  where U = exp(i * skew_hermitian(A, B)) is a unitary matrix
  parameterized by:
    - A: 6 real parameters (upper-triangle real parts)
    - B: 6 real parameters (upper-triangle imaginary parts)
    - D: 4 real parameters (eigenvalues / diagonal)

  Measure: <psi| H_ij |psi> = scalar expectation value

28 pairs -> 28 scalar observables
```

The ANO parameters (A, B, D) are trained at 100x the learning rate of circuit
parameters (lr=0.1 vs lr=0.001). This is because the measurement basis adapts faster
than the quantum state preparation, and a higher learning rate prevents the
observables from being a bottleneck.

The eigenvalue range of the Hermitian matrices (logged as `eig_min` and `eig_max`
during training) indicates the dynamic range of the measurements. Larger eigenvalue
spread means the observables can distinguish finer differences in quantum states.

---

## 8. Generation via Euler ODE

After training, new images are generated by solving the learned flow ODE:

```
Step 1: Sample noise
  z_0 ~ N(0, I), shape (N, 32)

Step 2: Euler integration from t=0 to t=1 in 100 steps
  dt = 1/100 = 0.01

  For step = 0, 1, ..., 99:
    t = step * dt
    v = velocity_field(z, t)     # Predicted velocity at current (z, t)
    z = z + dt * v               # Euler step

Step 3: Decode
  images = VAE.decode(z)
  images = clamp(images, 0, 1)
```

The Euler method traces the learned trajectory from noise (`t=0`) to data (`t=1`).
Each step queries the velocity field for the instantaneous direction, then takes a
small step in that direction.

---

## 9. Metrics

### Phase 1 (VAE Reconstruction) — computed per epoch on validation set

| Metric | Equation | Direction |
|--------|----------|-----------|
| Recon MSE | `mean((x_hat - x)^2)` | Lower is better |
| KL Divergence | `-0.5 * mean(1 + logvar - mu^2 - exp(logvar))` | Lower is better |
| VGG Perceptual | `L1(VGG_features(x_hat), VGG_features(x))` | Lower is better |
| PSNR | `10 * log10(1.0 / MSE)` | Higher is better |
| SSIM | Structural similarity (Wang et al., 2004) | Higher is better |
| LPIPS | Learned perceptual distance (Zhang et al., 2018) | Lower is better |

### Phase 2 (Flow Matching) — computed per epoch

| Metric | Equation | Direction |
|--------|----------|-----------|
| CFM MSE | `mean((v_pred - (z_1 - z_0))^2)` | Lower is better |
| Eig range | Min/max eigenvalues of ANO Hermitian matrices | Informational |

### Phase 2 (Generation Quality) — computed once after training

| Metric | Description | Direction |
|--------|-------------|-----------|
| FID | Frechet Inception Distance (feature=2048, Inception-v3) | Lower is better |
| IS | Inception Score | Higher is better |

**Important**: CFM MSE is only comparable across models that share the same VAE.
Different VAEs produce different latent spaces, so the MSE is in different "units".
FID and IS measure final image quality and are always comparable.

---

## 10. Parameter Counts

### Shared across all models
- ResConvVAE: **9,849,603** parameters (frozen during Phase 2)

### Velocity field parameters (Phase 2)

| Component | Classical-C | v6 (1x8q) | v7 (8x8q) |
|-----------|------------|-----------|-----------|
| time_mlp | 2,112 | 2,112 | 2,112 |
| enc_proj | — | 17,065 | 8 x 17,065 = 136,520 |
| VQC (QViT) | — | 288 | 8 x 288 = 2,304 |
| ANO (A, B, D) | — | 28 x 16 = 448 | 8 x 448 = 3,584 |
| MLP / vel_head | 156,448 | 41,248 | 57,888 |
| Quantum (enc+VQC) | — | 808 | 6,416 |
| **Total** | **158,560** | **62,121** | **422,824** |

For v6/v7, the "quantum" parameters are the VQC weights and encoding parameters
actually executed on the quantum circuit. The enc_proj (classical) and vel_head
(classical) are pre/post-processing layers.

---

## 11. Classical-C Baseline

The classical control uses an MLP in place of the quantum circuit:

```
z_t (32) ─────────────┐
                       ├─ concat ─> (64) ─> Linear(64, 256) ─> SiLU
t ─> sin_embed ─> time_mlp ─> t_emb (32) ─┘                ─> Linear(256, 256) ─> SiLU
                                                             ─> Linear(256, 256) ─> SiLU
                                                             ─> Linear(256, 32) ─> v (32)
```

Everything except the core transformation (MLP vs quantum circuit) is identical to
v6/v7: same time embedding, same input dimension, same output dimension.

---

## 12. Key Design Decisions

### Why these designs matter (lessons from v1-v5)

Previous versions had multiple confounding variables that made quantum-vs-classical
comparison unfair. v6/v7 fix all of them:

| Confound | v1-v5 Problem | v6/v7 Fix |
|----------|--------------|-----------|
| Time embedding | Quantum used raw sinusoidal; Classical used sinusoidal + time_mlp | All use time_mlp |
| Input sharing | v5 split input across circuits; each saw ~8 dims | All circuits see full 64 dims |
| Encoding projection | v5 used linear-only enc_proj | enc_proj uses SiLU nonlinearity |
| Velocity head | v5 used linear-only vel_head | vel_head uses SiLU nonlinearity |
| VAE architecture | Mixed legacy/resconv across runs | All use ResConvVAE |
| Latent dimension | v5 used latent=64 | All use latent=32 |

### Why shared input (not split)?

In v5, splitting a 128-dim input across 16 circuits gave each circuit ~8 dims. This
meant no single circuit saw both `z_t` and `t_emb` together — the time information
was fragmented. With shared input, each circuit gets the complete `(z_t, t_emb)`
vector and can learn its own optimal projection through its individual `enc_proj`.

### Why SiLU nonlinearities in enc_proj and vel_head?

The Classical-C baseline uses SiLU activations throughout its MLP. Without matching
nonlinearities in the quantum model's pre/post-processing, any performance difference
could be attributed to the missing nonlinearities rather than the quantum circuit
itself.

### Why dual learning rate for ANO?

The ANO observable parameters (A, B, D) define *what* to measure, while circuit
parameters define *what state to prepare*. Chen et al. (2025) found that observables
benefit from faster adaptation (lr=0.1) while circuit parameters train stably at
lower rates (lr=0.001). This prevents the measurement basis from being a training
bottleneck.

---

## References

- Lipman et al. (2023). *Flow Matching for Generative Modeling.* ICLR 2023.
- Wiersema et al. (2024). *Here comes the SU(N).* Quantum, 8, 1275.
- Cherrat et al. (2024). *Quantum Vision Transformers.* Quantum, 8, 1265.
- Chen et al. (2025). *Learning to Measure Quantum Neural Networks.* ICASSP 2025 Workshop.
- Lin et al. (2025). *Adaptive Non-local Observable on Quantum Neural Networks.* IEEE QCE 2025.
- Wang et al. (2004). *Image Quality Assessment: From Error Visibility to Structural Similarity.* IEEE TIP.
- Zhang et al. (2018). *The Unreasonable Effectiveness of Deep Features as a Perceptual Metric.* CVPR 2018.
