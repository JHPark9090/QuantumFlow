# Quantum Direct CFM: VAE-Free Quantum Flow Matching

## Table of Contents

1. [Overview](#1-overview)
2. [Architecture: v1 (Brick-Layer)](#2-architecture-v1-brick-layer)
3. [Architecture: v2 (Single-Gate, Pairwise ANO)](#3-architecture-v2-single-gate-pairwise-ano)
4. [Architecture: v3 (Single-Gate, Global ANO)](#4-architecture-v3-single-gate-global-ano)
5. [Data Flow: Input to Output](#5-data-flow-input-to-output)
6. [Generalized SU(2^k) Encoding (v1)](#6-generalized-su2k-encoding-v1)
7. [Single-Gate SU(2^n) Encoding (v2/v3)](#7-single-gate-su2n-encoding-v2v3)
8. [Training Pipeline](#8-training-pipeline)
9. [Generation via ODE Integration](#9-generation-via-ode-integration)
10. [v1 vs v2 vs v3: Key Differences](#10-v1-vs-v2-vs-v3-key-differences)
11. [Four-Way Comparison: QLCFM v9 vs Direct CFM v1 vs v2 vs v3](#11-four-way-comparison-qlcfm-v9-vs-direct-cfm-v1-vs-v2-vs-v3)
12. [Parameter Analysis](#12-parameter-analysis)
13. [Practical Considerations](#13-practical-considerations)
14. [Usage Examples](#14-usage-examples)

---

## 1. Overview

**Quantum Direct CFM** performs conditional flow matching (Lipman et al., 2023) **directly in pixel space** without a Variational Autoencoder (VAE). A lightweight classical Conv2d encoder compresses the image into a compact feature vector, the quantum circuit processes it via exponential mapping encoding (Wiersema et al., 2024) and Adaptive Non-Local Observables (ANO; Lin et al., 2025; Chen et al., 2025), and a classical ConvTranspose2d decoder expands the quantum output back to a pixel-space velocity field.

Three versions exist:

- **v1** (`models/QuantumDirectCFM.py`): Brick-layer SU(2^k) encoding + QViT ansatz (Cherrat et al., 2024) + pairwise ANO measurement. Uses 8 qubits with configurable SU group sizes from SU(4) through SU(256).
- **v2** (`models/QuantumDirectCFM_v2.py`): Single SU(2^n) gate on ALL n qubits (no brick-layer) + pairwise ANO measurement (no QViT ansatz). Uses 6 qubits with auto-configured Conv layers for a ~1:1 enc_proj ratio.
- **v3** (`models/QuantumDirectCFM_v3.py`): Single SU(2^n) gate on ALL n qubits + **global ANO measurement** (k=n, multi-observable). Uses 6 qubits. Each observable is a 2^n × 2^n learnable Hermitian acting on all qubits simultaneously, capturing all 1-through-n-body correlations.

**Key design goal:** Eliminate the VAE entirely and test whether a quantum velocity field can learn flow dynamics directly from pixel-level representations, bypassing the two-phase training pipeline of QLCFM v9.

**v1 focus:** Configurable SU(2^k) encoding gates from SU(4) through SU(256), allowing the quantum circuit to absorb significantly more input information than the standard SU(4) brick-layer encoding.

**v2 focus:** Maximally expressive encoding via a single SU(2^n) gate that creates full entanglement across all qubits, making the QViT ansatz redundant and yielding a cleaner quantum architecture (Encoding + ANO only).

**v3 focus:** Theoretically optimal measurement — each global Hermitian observable reads correlations across **all** qubits simultaneously, unlike pairwise ANO which only captures 2-body correlations. This is the most expressive quantum pipeline: full encoding (SU(2^n)) + full measurement (global ANO).

---

## 2. Architecture: v1 (Brick-Layer)

### 2.1. High-Level Diagram (v1)

```
Input:  x_t (noisy image, [B, 3, H, W])    +    time t (scalar)
         |                                        |
         v                                        v
  +--------------+                        +------------------+
  |  ConvEncoder |                        |  Time Embedding  |
  | 3x32x32      |                        |  sinusoidal      |
  | -> 32x16x16  |                        |  -> time_mlp     |
  | -> 64x8x8    |                        |  -> (2048,)      |
  | -> 128x4x4   |                        +------------------+
  | -> flatten    |                                |
  | -> (2048,)   |                                |
  +--------------+                                |
         |                                        |
         +------ addition -----------------------+
                    |
                    v
             (2048,)  [d_flat]
                    |
                    v
            +--------------+
            |   enc_proj   |
            | FC -> SiLU   |
            | FC -> (enc)  |
            +--------------+
                    |
                    v
   +================================+
   |     QUANTUM CIRCUIT (8q)       |
   |                                |
   |  [SU(2^k) Encoding]           |
   |   Brick-layer exponential map  |
   |                                |
   |  [QViT Ansatz]                 |
   |   Butterfly / Pyramid / X      |
   |   depth = 2 layers             |
   |                                |
   |  [ANO Measurement]             |
   |   Pairwise k=2 Hermitian       |
   |   -> 28 observables            |
   +================================+
                    |
                    v
              (28,)  [n_obs]
                    |
                    v
            +--------------+
            |   vel_head   |
            | FC -> SiLU   |
            | FC -> (2048) |
            +--------------+
                    |
                    v
            +--------------+
            |  ConvDecoder |
            | reshape      |
            | -> 128x4x4   |
            | -> 64x8x8    |
            | -> 32x16x16  |
            | -> 3x32x32   |
            +--------------+
                    |
                    v
Output:  velocity v(x_t, t) in pixel space [B, 3, H, W]
```

### 2.2. ConvEncoder

Lightweight stride-2 Conv2d downsampler. Compresses images to a flat feature vector.

```
Input: (B, 3, 32, 32)
  -> Conv2d(3, 32, kernel=4, stride=2, pad=1) + SiLU  -> (B, 32, 16, 16)
  -> Conv2d(32, 64, kernel=4, stride=2, pad=1) + SiLU -> (B, 64, 8, 8)
  -> Conv2d(64, 128, kernel=4, stride=2, pad=1) + SiLU -> (B, 128, 4, 4)
  -> Flatten -> (B, 2048)
```

**Flat dimension formula:** `d_flat = channels[-1] * (img_size / 2^n_layers)^2`

For default `--enc-channels=32,64,128` on 32x32: `d_flat = 128 * 4 * 4 = 2048`

**No batch norm, no residual blocks** -- intentionally lightweight so the quantum circuit is the primary learnable component.

### 2.3. ConvDecoder

Mirror of encoder using ConvTranspose2d. Expands flat features back to pixel-space velocity.

```
Input: (B, 2048)
  -> Reshape -> (B, 128, 4, 4)
  -> ConvTranspose2d(128, 64, 4, 2, 1) + SiLU -> (B, 64, 8, 8)
  -> ConvTranspose2d(64, 32, 4, 2, 1) + SiLU  -> (B, 32, 16, 16)
  -> ConvTranspose2d(32, 3, 4, 2, 1)           -> (B, 3, 32, 32)
```

**No final activation:** The velocity field output is unbounded (can be any real value), unlike image reconstruction which is bounded to [0, 1].

### 2.4. Time Embedding

Sinusoidal positional encoding (Vaswani et al., 2017) followed by a 2-layer MLP that projects to d_flat dimensions for additive conditioning:

```python
# Sinusoidal encoding: t -> (time_embed_dim,)  (default 256)
freqs = exp(-log(10000) * i / (dim/2))   for i in [0, dim/2)
t_emb = [cos(t * freqs), sin(t * freqs)]

# Time MLP: project to d_flat for additive conditioning
time_mlp = Linear(256, d_flat) -> SiLU -> Linear(d_flat, d_flat)
# e.g., for 32x32 CIFAR-10: Linear(256, 2048) -> SiLU -> Linear(2048, 2048)
```

The time embedding is **added** to the image features (not concatenated), giving time equal influence over the feature representation without increasing the input dimension to the quantum circuit.

---

## 3. Architecture: v2 (Single-Gate)

### 3.1. High-Level Diagram (v2)

```
Input:  x_t (noisy image, [B, 3, H, W])    +    time t (scalar)
         |                                        |
         v                                        v
  +--------------+                        +------------------+
  |  ConvEncoder |                        |  Time Embedding  |
  | (auto-config |                        |  sinusoidal      |
  |  for ~1:1    |                        |  -> time_mlp     |
  |  enc_proj)   |                        |  -> (d_flat,)    |
  | -> (d_flat,) |                        +------------------+
  +--------------+                                |
         |                                        |
         +------ addition -----------------------+
                    |
                    v
             (d_flat,)  [≈ 4^n]
                    |
                    v
            +--------------+
            |   enc_proj   |
            | FC -> SiLU   |
            | FC -> (4^n-1)|
            +--------------+
                    |
                    v
   +================================+
   |   QUANTUM CIRCUIT (6q default) |
   |                                |
   |  [Single SU(2^n) Encoding]    |
   |   One gate on ALL n qubits    |
   |   Full entanglement: 100%     |
   |                                |
   |  [No QViT Ansatz]             |
   |   (optional, default=none)    |
   |                                |
   |  [ANO Measurement]            |
   |   Pairwise k=2 Hermitian      |
   |   -> 15 observables           |
   +================================+
                    |
                    v
              (15,)  [C(6,2)]
                    |
                    v
            +--------------+
            |   vel_head   |
            | FC -> SiLU   |
            | FC -> (d_flat)|
            +--------------+
                    |
                    v
            +--------------+
            |  ConvDecoder |
            | (mirrors     |
            |  ConvEncoder)|
            | -> 3xHxW     |
            +--------------+
                    |
                    v
Output:  velocity v(x_t, t) in pixel space [B, 3, H, W]
```

### 3.2. ConvEncoder (v2 — Auto-Configured)

v2 uses `auto_enc_channels(img_size, n_qubits)` to automatically compute the Conv layer progression such that `d_flat ≈ 4^n` (matching the SU(2^n) encoding size for a ~1:1 ratio).

| Image Size | n_qubits | Conv Channels | Spatial | d_flat | enc_size (4^n - 1) | Ratio |
|------------|----------|---------------|---------|--------|--------------------|-------|
| 32×32 | 6 | [32, 64] | 8×8 | 4096 | 4095 | 1.00:1 |
| 64×64 | 6 | [32, 64, 64] | 8×8 | 4096 | 4095 | 1.00:1 |
| 128×128 | 6 | [32, 64, 128, 256, 256] | 4×4 | 4096 | 4095 | 1.00:1 |
| 256×256 | 6 | [32, 64, 128, 256, 256, 256] | 4×4 | 4096 | 4095 | 1.00:1 |

**Strategy:**
- `img_size < 128`: target final spatial = 8 (fewer Conv layers)
- `img_size >= 128`: target final spatial = 4 (more Conv layers, richer encoder)
- Final channel count = `4^n / spatial^2`

### 3.3. Why No QViT Ansatz?

The single SU(2^n) gate acts on ALL n qubits simultaneously, creating **full entanglement** across every qubit pair:

- v1 (brick-layer SU(4), 8 qubits): 7/28 pairs directly entangled (25%)
- v2 (single SU(64), 6 qubits): **15/15 pairs directly entangled (100%)**

In v1, the QViT ansatz is essential — it creates the missing entanglement patterns for the 21 unentangled qubit pairs. In v2, every pair is already entangled by the encoding, so QViT adds no structural benefit.

Furthermore, in the Heisenberg picture, the QViT rotation W(θ) applied before measurement H(ϕ) can be absorbed:

```
⟨ψ| W†(θ) H(ϕ) W(θ) |ψ⟩  ≡  ⟨ψ| H'(ϕ') |ψ⟩
```

Since ANO's learnable Hermitian H(ϕ) has free eigenvalues (unlike fixed-eigenvalue Pauli-Z), it can represent any observable that W†HW could produce. This makes QViT mathematically redundant when the state is already fully entangled.

### 3.4. ~1:1 enc_proj Ratio

The `enc_proj` layer projects from `d_flat` to `enc_size = 4^n - 1`:

| Model | d_flat | enc_size | enc_proj ratio | Compression |
|-------|--------|----------|---------------|-------------|
| v1 SU(4), 8q | 2048 | 105 | 19.5:1 | Heavy compression |
| v1 SU(16), 8q | 2048 | 765 | 2.7:1 | Moderate compression |
| **v2 SU(64), 6q** | 4096 | 4095 | **1.00:1** | **Near-identity** |

The ~1:1 ratio in v2 means `enc_proj` is essentially a learned reparameterization rather than a lossy compression. The quantum circuit receives nearly all the information from the classical encoder.

---

## 4. Architecture: v3 (Single-Gate, Global ANO)

### 4.1. High-Level Diagram (v3)

```
Input:  x_t (noisy image, [B, 3, H, W])    +    time t (scalar)
         |                                        |
         v                                        v
  +--------------+                        +------------------+
  |  ConvEncoder |                        |  Time Embedding  |
  | (auto-config |                        |  sinusoidal      |
  |  for ~1:1    |                        |  -> time_mlp     |
  |  enc_proj)   |                        |  -> (d_flat,)    |
  | -> (d_flat,) |                        +------------------+
  +--------------+                                |
         |                                        |
         +------ addition -----------------------+
                    |
                    v
             (d_flat,)  [≈ 4^n]
                    |
                    v
            +--------------+
            |   enc_proj   |
            | FC -> SiLU   |
            | FC -> (4^n-1)|
            +--------------+
                    |
                    v
   +================================+
   |   QUANTUM CIRCUIT (6q default) |
   |                                |
   |  [Single SU(2^n) Encoding]    |
   |   One gate on ALL n qubits    |
   |   Full entanglement: 100%     |
   |                                |
   |  [No QViT Ansatz]             |
   |   (not needed)                |
   |                                |
   |  [Global ANO Measurement]     |
   |   k=n (all qubits)           |
   |   m independent 2^n x 2^n    |
   |   Hermitian observables       |
   |   -> m observables            |
   +================================+
                    |
                    v
              (m,)  [n_observables]
                    |
                    v
            +--------------+
            |   vel_head   |
            | FC -> SiLU   |
            | FC -> (d_flat)|
            +--------------+
                    |
                    v
            +--------------+
            |  ConvDecoder |
            | (mirrors     |
            |  ConvEncoder)|
            | -> 3xHxW     |
            +--------------+
                    |
                    v
Output:  velocity v(x_t, t) in pixel space [B, 3, H, W]
```

### 4.2. Global ANO: Why and How

**The limitation of pairwise ANO (k=2):**

In v1 and v2, each ANO observable is a 4×4 learnable Hermitian acting on 2 qubits. This captures **2-body correlations** only — the expectation value `⟨ψ|H_{ij}|ψ⟩` depends on the reduced density matrix of qubits i and j. Higher-order correlations (3-body, 4-body, etc.) involving more than 2 qubits simultaneously are invisible to pairwise measurement.

**Why not simply use k=n with a single observable?**

With k=n (all qubits), `C(n, n) = 1` — there is only one wire group: all n qubits. A single global Hermitian gives just one scalar output, which is insufficient information for the velocity head.

**v3's solution: multiple independent global Hermitians.**

v3 creates `m` independent 2^n × 2^n Hermitian matrices, **all acting on the same wire group** [0, 1, ..., n-1]. Each observable reads the full quantum state but with different learned eigenvalue structures:

```python
# m = n_observables (default: 15)
# K = 2^n_qubits (e.g., 64 for 6 qubits)

for w in range(m):
    H_w = construct_hermitian(A[w], B[w], D[w])  # K × K Hermitian
    obs_w = qml.expval(qml.Hermitian(H_w, wires=[0, ..., n-1]))
```

Each Hermitian H_w is parameterized by:
- `A[w]`: K(K-1)/2 off-diagonal real parts
- `B[w]`: K(K-1)/2 off-diagonal imaginary parts
- `D[w]`: K diagonal elements (eigenvalue-related)

### 4.3. Correlations Captured

| ANO type | Matrix size | Correlations | Example (6 qubits) |
|----------|------------|-------------|---------------------|
| Pairwise k=2 | 4×4 | 2-body only | `⟨Z₁Z₂⟩`, `⟨X₃Y₅⟩` |
| Global k=n | 2^n × 2^n | **All (1 through n-body)** | `⟨Z₁Z₂Z₃⟩`, `⟨X₁Y₂Z₃X₄Y₅Z₆⟩` |

A 64×64 Hermitian on 6 qubits can represent **any** observable expressible as a linear combination of all 4^6 = 4096 Pauli strings (including the identity). This is the maximally expressive measurement possible on 6 qubits.

### 4.4. Parameter Comparison: Pairwise vs Global ANO

| | v2 (Pairwise k=2) | v3 (Global k=6) |
|---|---|---|
| **Wire groups** | C(6,2) = 15 pairs | 1 group (all 6 qubits), repeated m times |
| **Hermitian size** | 4×4 | 64×64 |
| **Params per Hermitian** | 16 (4² real + imaginary + diagonal) | 4,096 (64² real + imaginary + diagonal) |
| **Number of observables** | 15 | m (default: 15) |
| **Total ANO params** | 240 | 61,440 |
| **Correlations** | 2-body | All (1 through 6-body) |

### 4.5. ConvEncoder (v3 — Same as v2)

v3 uses the same `auto_enc_channels` system as v2 for ~1:1 enc_proj ratio. The encoding pipeline is identical to v2; only the measurement differs.

---

## 5. Data Flow: Input to Output

### 5.1. Forward Pass (Training, v1)

Given a batch of real images `x_1` (shape `[B, 3, 32, 32]`, values in [0, 1]):

```
1. Sample noise:     x_0 ~ N(0, I)                    shape: [B, 3, 32, 32]
2. Sample time:      t ~ logit-normal or uniform       shape: [B]
3. Interpolate:      x_t = (1 - t) * x_0 + t * x_1    shape: [B, 3, 32, 32]
4. Target velocity:  u = x_1 - x_0                     shape: [B, 3, 32, 32]

5. Encode image:     h = ConvEncoder(x_t)               shape: [B, 2048]
6. Time embed:       t_emb = time_mlp(sinusoidal(t))    shape: [B, 2048]
7. Add:              z = h + t_emb                       shape: [B, 2048]

8. Encode for circuit:  enc = enc_proj(z)               shape: [B, enc_per_block]
9. Quantum circuit:     q_out = circuit(enc, vqc, H)    shape: [B, 28]
10. Velocity head:      v_flat = vel_head(q_out)        shape: [B, 2048]
11. Decode to pixels:   v_pred = ConvDecoder(v_flat)     shape: [B, 3, 32, 32]

12. Loss:  L = MSE(v_pred, u)
```

### 5.2. Key Difference from QLCFM v9

In QLCFM v9, steps 5 and 11 are replaced by VAE operations:
- Step 5 becomes: `z_1 = VAE.encode(x).mu` (frozen, 32-dim latent)
- Step 11 becomes: `images = VAE.decode(z)` (only at generation time)

In Quantum Direct CFM, the encoder and decoder are **part of the velocity field** and trained end-to-end with the quantum circuit.

---

## 6. Generalized SU(2^k) Encoding (v1)

### 6.1. Exponential Mapping

The SU(2^k) encoding maps classical parameters to a unitary gate via:

```
U(theta) = exp(i * sum_{j=1}^{N^2-1} theta_j * G_j)
```

where `G_j` are the generators of the Lie algebra su(N), constructed as tensor products of Pauli matrices `{I, X, Y, Z}^{otimes k}` (excluding the all-identity string).

### 6.2. Brick-Layer Pattern

Gates are applied in a brick-layer (even/odd) pattern on groups of k qubits:

**Even layer:** Non-overlapping groups starting at qubit 0
**Odd layer:** Shifted groups starting at the smallest offset that yields at least one group

### 6.3. SU Group Table (8 qubits)

| SU(N) | k | Generators/gate | Even groups | Odd groups | Total gates | Encoding size |
|-------|---|----------------|-------------|------------|-------------|---------------|
| SU(4) | 2 | 15 | [0,1],[2,3],[4,5],[6,7] | [1,2],[3,4],[5,6] | 7 | **105** |
| SU(8) | 3 | 63 | [0,1,2],[3,4,5] | [1,2,3],[4,5,6] | 4 | **252** |
| SU(16) | 4 | 255 | [0,1,2,3],[4,5,6,7] | [1,2,3,4] | 3 | **765** |
| SU(32) | 5 | 1,023 | [0,1,2,3,4] | [1,2,3,4,5] | 2 | **2,046** |
| SU(64) | 6 | 4,095 | [0,1,2,3,4,5] | [1,2,3,4,5,6] | 2 | **8,190** |
| SU(128) | 7 | 16,383 | [0,1,2,3,4,5,6] | [1,2,3,4,5,6,7] | 2 | **32,766** |
| SU(256) | 8 | 65,535 | [0,1,2,3,4,5,6,7] | (none) | 1 | **65,535** |

**Pattern:** Generator count grows as `4^k - 1`. Larger k means fewer gates but exponentially more parameters per gate.

### 6.4. Implementation Strategy

| SU(N) | Implementation | Notes |
|-------|---------------|-------|
| SU(4) -- SU(32) | PennyLane `qml.SpecialUnitary` | Native, efficient |
| SU(64) | Custom `TorchSUGate` (precomputed) | Precomputes all 4,095 generators (~268 MB) |
| SU(128) | Custom `TorchSUGate` (streaming) | Computes generators on-the-fly (~4.3 GB avoided) |
| SU(256) | Custom `TorchSUGate` (streaming) | Very slow, requires GPU with large memory |

**TorchSUGate** builds the Hermitian matrix `A = sum(theta_j * G_j)` using Pauli tensor products, then computes `U = torch.linalg.matrix_exp(1j * A)`, and applies it via `qml.QubitUnitary`.

### 6.5. What Larger SU Gates Mean

As the SU group size increases:

1. **More input capacity:** SU(4) absorbs 105 values, SU(64) absorbs 8,190 values from the classical encoder
2. **Richer quantum encoding:** Each gate parameterizes a larger portion of the unitary group, accessing more of the Hilbert space
3. **Classical projection cost:** The `enc_proj` layer must project from `d_flat` to `enc_per_block`, which grows with SU size. This makes the classical layers larger.
4. **Quantum parameters unchanged:** The VQC (QViT) and ANO measurement parameters remain fixed at 736 regardless of SU size. Only the encoding changes.

---

## 7. Single-Gate SU(2^n) Encoding (v2/v3)

### 7.1. Single-Gate Exponential Mapping

v2 uses the same exponential mapping formula as v1:

```
U(theta) = exp(i * sum_{j=1}^{N^2-1} theta_j * G_j)
```

The critical difference is that **one gate acts on all n qubits simultaneously**, producing a single `2^n × 2^n` unitary matrix. There is no brick-layer pattern — no even/odd layers, no qubit grouping.

### 7.2. Single-Gate Parameter Table

| n_qubits | SU(2^n) | Generators (4^n - 1) | Unitary Size | Entangled Pairs | Target d_flat |
|----------|---------|---------------------|--------------|-----------------|---------------|
| 4 | SU(16) | 255 | 16×16 | 6/6 (100%) | 256 |
| 5 | SU(32) | 1,023 | 32×32 | 10/10 (100%) | 1,024 |
| **6** | **SU(64)** | **4,095** | **64×64** | **15/15 (100%)** | **4,096** |
| 7 | SU(128) | 16,383 | 128×128 | 21/21 (100%) | 16,384 |
| 8 | SU(256) | 65,535 | 256×256 | 28/28 (100%) | 65,536 |

**Default: 6 qubits.** SU(64) with 4,095 generators provides 100% entanglement coverage while remaining computationally feasible (64×64 matrix exponential).

### 7.3. Entanglement Coverage: Brick-Layer vs Single-Gate

For 8 qubits with SU(4) brick-layer (v1):
```
Even layer:  (0,1)  (2,3)  (4,5)  (6,7)     → 4 pairs
Odd layer:   (1,2)  (3,4)  (5,6)             → 3 pairs
Total:       7 / C(8,2) = 7/28 = 25%
```

For 6 qubits with single SU(64) (v2):
```
One gate on qubits [0,1,2,3,4,5]
All C(6,2) = 15 pairs entangled = 100%
```

The remaining 21 pairs in v1 are only entangled indirectly through the QViT ansatz. In v2, all pairs are entangled directly by the encoding gate.

### 7.4. Implementation (TorchSUGate)

For `n_qubits >= 6`, PennyLane's `qml.SpecialUnitary` cannot handle the computation. v2 uses a custom `TorchSUGate` class:

1. **Precompute generators** (n ≤ 6): All 4^n - 1 Pauli tensor products stored in memory (~268 MB for SU(64))
2. **Streaming generators** (n ≥ 7): Computed on-the-fly to avoid multi-GB memory allocation
3. **Matrix exponential**: `U = torch.linalg.matrix_exp(1j * sum(theta_j * G_j))`
4. **Applied via**: `qml.QubitUnitary(U, wires=[0, ..., n-1])`

Note: v2 and v3 use `default.qubit` (not `lightning.qubit`) because `SpecialUnitary` and batched `QubitUnitary` are not supported by the lightning backend.

---

## 8. Training Pipeline

### 8.1. Single-Phase Training

Unlike QLCFM v9 which requires VAE pretraining (Phase 1) before CFM training (Phase 2), Quantum Direct CFM has **one training phase**:

```
For each epoch:
    For each batch of real images x_1:
        1. x_0 = randn_like(x_1)                    # Gaussian noise in pixel space
        2. t = sample_timestep()                      # logit-normal or uniform
        3. x_t = (1 - t) * x_0 + t * x_1             # OT interpolation
        4. target = x_1 - x_0                         # velocity target
        5. v_pred = velocity_field(x_t, t)            # full forward pass
        6. loss = MSE(v_pred, target)
        7. loss.backward()                            # end-to-end gradients
        8. optimizer.step()
```

All components (ConvEncoder, enc_proj, quantum circuit, vel_head, ConvDecoder) are trained jointly.

### 8.2. Dual Optimizer

Following Chen et al. (2025), two separate optimizers with different learning rates:

| Group | Parameters | Learning Rate | Rationale |
|-------|-----------|---------------|-----------|
| Main | ConvEncoder, enc_proj, VQC, vel_head, ConvDecoder | 1e-3 | Standard gradient descent |
| Observable | ANO Hermitian params (A, B, D) | 1e-1 (100x higher) | Eigenvalue range must expand rapidly |

Both use CosineAnnealingLR schedule.

### 8.3. Timestep Sampling

Two options (Esser et al., 2024):

- **Uniform:** `t ~ U(0, 1)` -- standard, equal weight to all timesteps
- **Logit-normal:** `t = sigmoid(N(0, std^2))` -- concentrates training on mid-range timesteps (t ~ 0.3-0.7) where the velocity field is hardest to predict

### 8.4. EMA (Exponential Moving Average)

Optional velocity field EMA (decay=0.999) maintains a smoothed copy of all model weights. During validation, both raw and EMA weights are evaluated. Best checkpoint is selected based on EMA validation loss.

---

## 9. Generation via ODE Integration

### 9.1. Sampling Process

Starting from pure Gaussian noise in pixel space:

```python
x = randn(n_samples, 3, H, W)     # start at t=0 (noise)
dt = 1.0 / ode_steps               # e.g., 0.02 for 50 steps

for step in range(ode_steps):
    t = step * dt                   # t: 0 -> 1

    # Euler:
    v = velocity_field(x, t)
    x = x + dt * v

    # -- OR Midpoint (2nd-order): --
    k1 = velocity_field(x, t)
    x_mid = x + 0.5 * dt * k1
    k2 = velocity_field(x_mid, t + 0.5 * dt)
    x = x + dt * k2

images = x.clamp(0, 1)             # final generated images
```

### 9.2. No VAE Decoder Needed

In QLCFM v9, the ODE integration produces latent vectors that must be decoded via `VAE.decode(z)`. In Quantum Direct CFM, the ODE integration produces pixel-space images directly -- **no decoding step required**.

This means generation quality is not bounded by VAE reconstruction quality, which was the dominant bottleneck in QLCFM v9 (PSNR 18-19 dB for CIFAR-10).

---

## 10. v1 vs v2 vs v3: Key Differences

### 10.1. Architecture Comparison

| Aspect | v1 (`QuantumDirectCFM.py`) | v2 (`QuantumDirectCFM_v2.py`) | v3 (`QuantumDirectCFM_v3.py`) |
|--------|---------------------------|-------------------------------|-------------------------------|
| **Encoding** | Brick-layer SU(2^k) on groups of k qubits | Single SU(2^n) on ALL n qubits | Single SU(2^n) on ALL n qubits |
| **Entanglement from encoding** | Partial (e.g., 7/28 = 25% for SU(4), 8q) | **Full (100%)** | **Full (100%)** |
| **QViT ansatz** | Required (fills entanglement gaps) | Not needed (default: none) | Not needed |
| **ANO locality** | Pairwise k=2 (4×4 Hermitians) | Pairwise k=2 (4×4 Hermitians) | **Global k=n (2^n × 2^n Hermitians)** |
| **Correlations measured** | 2-body | 2-body | **All (1 through n-body)** |
| **Quantum architecture** | Encoding + QViT + ANO | Encoding + ANO | **Encoding + ANO (global)** |
| **Default qubits** | 8 | 6 | 6 |
| **Default enc_channels** | `32, 64, 128` (fixed) | `auto` (adapts to image size) | `auto` (adapts to image size) |
| **Default d_flat** | 2048 | 4096 | 4096 |
| **enc_proj ratio** | 19.5:1 for SU(4), varies with group size | **~1:1 by design** | **~1:1 by design** |
| **ANO params** | 448 (28 × 16) | 240 (15 × 16) | **61,440 (15 × 4,096)** |
| **SU group config** | `--sun-group-size` (2 to 8) | Always n_qubits (single gate) | Always n_qubits (single gate) |

### 10.2. Why These Differences Matter

**1. Stronger encoding eliminates the need for QViT (v1 → v2/v3):**

In v1, the brick-layer SU(4) encoding only entangles neighboring qubit groups. The QViT ansatz is essential to create entanglement across the remaining pairs. In v2/v3, the single SU(2^n) gate entangles all pairs simultaneously, making QViT redundant.

This is not merely an engineering choice — it follows from the Heisenberg picture: when the quantum state is fully entangled, any rotation W(θ) before measurement H(ϕ) can be absorbed into a richer observable H'(ϕ'), which ANO already provides.

**2. ~1:1 enc_proj ratio preserves information (v1 → v2/v3):**

In v1, `enc_proj` compresses 2048 features into 105 encoding parameters (SU(4)) — a 19.5:1 compression that discards most information before it reaches the quantum circuit. In v2/v3, `enc_proj` maps 4096 features to 4095 encoding parameters — essentially a learned reparameterization with minimal information loss.

**3. Global ANO captures all correlations (v2 → v3):**

Pairwise ANO (k=2) measures only 2-body correlations: how pairs of qubits are correlated. For a 6-qubit system with full entanglement from SU(64), significant information is encoded in 3-body, 4-body, and higher-order correlations that pairwise measurement cannot access.

Global ANO (k=n) uses 2^n × 2^n Hermitians that can represent any observable on the full Hilbert space, including all multi-body correlations. This is the theoretically optimal measurement for maximally entangled states.

**4. Parameter cost of global ANO:**

The trade-off is parameter count: v3's 61,440 ANO params vs v2's 240. However, these are genuine quantum parameters (they define the measurement basis), not classical overhead. They constitute 0.16% of v3's total params — still a small fraction.

### 10.3. What Stays the Same

All three versions share:
- Pixel-space flow matching (no VAE)
- ConvEncoder → quantum circuit → ConvDecoder architecture
- Additive time conditioning (sinusoidal embedding, dim=256)
- Dual optimizer (100x LR for observable params)
- Logit-normal timestep sampling
- Midpoint ODE solver
- EMA velocity field

v2 and v3 additionally share:
- Single SU(2^n) encoding (identical quantum state preparation)
- Auto-configured Conv layers for ~1:1 enc_proj ratio
- No QViT ansatz

---

## 11. Four-Way Comparison: QLCFM v9 vs Direct CFM v1 vs v2 vs v3

### 11.1. Architecture Summary

| Aspect | QLCFM v9 | Direct CFM v1 | Direct CFM v2 | Direct CFM v3 |
|--------|----------|---------------|---------------|---------------|
| **File** | `QuantumLatentCFM_v9.py` | `QuantumDirectCFM.py` | `QuantumDirectCFM_v2.py` | `QuantumDirectCFM_v3.py` |
| **Flow space** | VAE latent space | Pixel space | Pixel space | Pixel space |
| **VAE required** | Yes (pretrained, frozen) | No | No | No |
| **Training phases** | 2 (VAE + CFM) | 1 | 1 | 1 |
| **Encoding** | Brick-layer SU(4), 8q | Brick-layer SU(2^k), 8q | Single SU(2^n), 6q | Single SU(2^n), 6q |
| **Ansatz** | QViT butterfly | QViT pyramid | None (default) | None |
| **Measurement** | ANO pairwise k=2 | ANO pairwise k=2 | ANO pairwise k=2 | **ANO global k=n** |
| **Quantum pipeline** | Enc + QViT + ANO | Enc + QViT + ANO | Enc + ANO | **Enc + ANO (global)** |
| **Entanglement** | 7/28 (25%) + QViT | 7/28 (25%) + QViT | 15/15 (100%) | 15/15 (100%) |
| **enc_proj ratio** | Varies (latent-based) | 19.5:1 (SU(4)) | 1.00:1 | 1.00:1 |
| **n_obs** | 28 | 28 | 15 | 15 |
| **ANO params** | 448 | 448 | 240 | **61,440** |
| **Correlations** | 2-body | 2-body | 2-body | **All (1-6 body)** |

### 11.2. Data Flow Comparison

**QLCFM v9** (latent-space):
```
Training:
  Image → [Frozen VAE Encoder] → z_1 (e.g., 128-dim latent)
  z_0 ~ N(0, I)
  z_t = interpolate(z_0, z_1)
  v = QuantumVelocityField(z_t, t)   [8q: SU(4) + QViT + ANO]
  Loss = MSE(v, z_1 - z_0)

Generation:
  z_0 ~ N(0, I) → ODE → z_1 → [Frozen VAE Decoder] → Image
```

**Direct CFM v1** (pixel-space, brick-layer):
```
Training:
  x_0 ~ N(0, I)  (pixel noise)
  x_t = interpolate(x_0, x_1)
  v = ConvEnc → quantum [8q: SU(4) + Pyramid QViT + ANO(k=2)] → ConvDec
  Loss = MSE(v, x_1 - x_0)

Generation:
  x_0 ~ N(0, I) → ODE → x_1.clamp(0, 1)  (direct pixel output)
```

**Direct CFM v2** (pixel-space, single-gate, pairwise ANO):
```
Training:
  x_0 ~ N(0, I)  (pixel noise)
  x_t = interpolate(x_0, x_1)
  v = ConvEnc → quantum [6q: SU(64) + ANO(k=2), no QViT] → ConvDec
  Loss = MSE(v, x_1 - x_0)

Generation:
  x_0 ~ N(0, I) → ODE → x_1.clamp(0, 1)  (direct pixel output)
```

**Direct CFM v3** (pixel-space, single-gate, global ANO):
```
Training:
  x_0 ~ N(0, I)  (pixel noise)
  x_t = interpolate(x_0, x_1)
  v = ConvEnc → quantum [6q: SU(64) + ANO(k=6, 15 global obs), no QViT] → ConvDec
  Loss = MSE(v, x_1 - x_0)

Generation:
  x_0 ~ N(0, I) → ODE → x_1.clamp(0, 1)  (direct pixel output)
```

### 11.3. Bottleneck Analysis

| Model | Bottleneck | Description |
|-------|-----------|-------------|
| QLCFM v9 | VAE reconstruction quality | Generation cannot exceed VAE decode quality |
| Direct CFM v1 | Quantum circuit capacity + enc_proj compression | 28 observables from 2048-dim input (73:1) |
| Direct CFM v2 | Measurement expressiveness | 15 pairwise observables capture only 2-body correlations from a fully entangled state |
| Direct CFM v3 | Quantum circuit capacity | 15 global observables read all correlations; bottleneck shifts to vel_head (15 → d_flat) |

### 11.4. Expected Performance

**Image generation quality:** QLCFM v9 > Direct CFM v3 ≥ v2 ≥ v1

- QLCFM v9 benefits from the VAE's structured latent space — the quantum velocity field operates on a smooth, compressed manifold rather than raw pixel features
- v3 should outperform v2 because global ANO extracts more information from the quantum state
- v2 should match or slightly outperform v1 due to full entanglement and ~1:1 encoding ratio
- This mirrors the classical literature: latent diffusion (Rombach et al., 2022) outperforms pixel-space diffusion

**Quantum expressiveness:** Direct CFM v3 > v2 > v1 ≈ QLCFM v9

- v3 has the most expressive quantum pipeline: SU(64) encoding (100% entanglement) + global ANO (all correlations)
- v2 has full encoding but limited measurement (2-body only)
- v1/v9's brick-layer SU(4) only accesses 4-dim subspaces with 25% entanglement

**Training stability:** QLCFM v9 > Direct CFM v1 > v2 ≥ v3

- v9: VAE pretrained separately; flow operates on well-structured latent space
- v1: End-to-end pixel-space training with moderate encoding complexity
- v2/v3: SU(64) has 4095 generators → complex gradient landscape
- v3: 61K ANO params need careful optimization (mitigated by 100x higher LR)

**Scientific contribution:** Direct CFM v3 > v2 > v1 > QLCFM v9

- v3 has the theoretically optimal quantum pipeline: full encoding + full measurement. It cleanly tests whether a quantum circuit with maximally expressive encoding AND measurement can drive image generation
- v2 has clean encoding but limited measurement — tests whether 2-body correlations suffice
- v2 vs v3 comparison isolates the measurement expressiveness question
- v1 mixes SU encoding with classical QViT ansatz — harder to attribute performance to quantum effects
- v9 has the most classical components — the quantum circuit is a small part of a mostly classical pipeline

### 11.5. Recommended Strategy

For a publication, the four models form a systematic ablation:

| Role | Model | Rationale |
|------|-------|-----------|
| **Best image quality** | QLCFM v9 | Latent-space advantage; competitive FID scores |
| **Strongest quantum argument** | Direct CFM v3 | Theoretically optimal quantum pipeline (full encoding + full measurement) |
| **Ablation: measurement** | Direct CFM v2 | Same encoding as v3 but pairwise ANO → isolates measurement effect |
| **Ablation: encoding** | Direct CFM v1 | Brick-layer encoding + QViT → isolates encoding effect |

- Present v3 as the **theoretically optimal quantum architecture** with maximal encoding (SU(2^n)) and maximal measurement (global ANO)
- Present v2 vs v3 as the **measurement expressiveness ablation**: does global ANO improve over pairwise ANO when the quantum state is already fully entangled?
- Present v1 vs v2 as the **encoding ablation**: does single-gate SU(2^n) improve over brick-layer SU(2^k) + QViT?
- Present v9 as the **practical quantum-hybrid** that achieves competitive image quality

### 11.6. Can ANO Replace QViT? (Per-Model Analysis)

| Question | QLCFM v9 | Direct CFM v1 | Direct CFM v2 | Direct CFM v3 |
|----------|----------|---------------|---------------|---------------|
| Replace encoding with ANO? | No (constant) | No (constant) | No (constant) | No (constant) |
| Replace QViT with ANO? | No (25% ent.) | No (25% ent.) | **Already done** | **Already done** |
| Current architecture | Enc + QViT + ANO | Enc + QViT + ANO | **Enc + ANO(k=2)** | **Enc + ANO(k=n)** |

ANO can absorb the QViT ansatz (via Heisenberg picture equivalence) **only when the encoding already provides full entanglement coverage**. v2 and v3 satisfy this condition; v1 and v9 do not.

### 11.7. Quantum Pipeline Evolution

```
v1:  Brick-layer SU(2^k) encoding  →  QViT ansatz  →  Pairwise ANO (k=2)
     [partial entanglement]            [fills gaps]     [2-body correlations]

v2:  Single SU(2^n) encoding       →  (none)        →  Pairwise ANO (k=2)
     [full entanglement]               [redundant]      [2-body correlations]

v3:  Single SU(2^n) encoding       →  (none)        →  Global ANO (k=n)
     [full entanglement]               [redundant]      [ALL correlations]
```

Each step removes a limitation: v2 achieves full entanglement (removing QViT), v3 achieves full measurement (removing the 2-body restriction). v3 represents the theoretically complete quantum pipeline.

---

## 12. Parameter Analysis

### 12.1. Component Breakdown (CIFAR-10, 32x32)

| Component | QLCFM v9 | v1 SU(4), 8q | **v2 SU(64), 6q** | **v3 SU(64), 6q** |
|-----------|----------|--------------|-------------------|-------------------|
| VAE (frozen) | ~3,000,000 | -- | -- | -- |
| ConvEncoder + Decoder | -- | 331,075 | 68,739 | 68,739 |
| time_mlp | 2,144 | 4,722,688 | 17,833,984 | 17,833,984 |
| enc_proj | 43,497 | 551,529 | 16,773,119 | 16,773,119 |
| VQC (QViT) | 144 | 672 | **0** (none) | **0** (none) |
| ANO (A+B+D) | 448 | 448 | 240 | **61,440** |
| vel_head | 8,480 | 533,760 | 1,056,768 | 1,056,768 |
| **Total** | **~3,054,713** | **6,140,172** | **35,732,850** | **35,794,050** |
| **Quantum only (VQC+ANO)** | **592** | **1,120** | **240** | **61,440** |
| **Quantum %** | 0.02% | 0.018% | 0.001% | **0.172%** |

Notes:
- v2 has 15 ANO observables (C(6,2)=15 pairs, 16 params each) vs v1's 28 (C(8,2)=28 pairs)
- v3 has 15 ANO observables (15 × 4,096 params each = 61,440 total) — same count as v2 but 256x more params per observable
- v2/v3's large enc_proj comes from the ~1:1 ratio: Linear(4096, 4096) → SiLU → Linear(4096, 4095)
- v3 has the highest quantum parameter fraction (0.172%) of any model variant

### 12.2. Scaling with SU Group Size (v1)

All on 8 qubits, QViT butterfly, pairwise k=2, single circuit:

| SU(N) | Encoding size | enc_proj params | Total params | Quantum % | Status |
|-------|-------------|----------------|-------------|-----------|--------|
| SU(4) | 105 | 560K | **1.4M** | 0.05% | OK |
| SU(8) | 252 | 597K | **1.5M** | 0.05% | OK |
| SU(16) | 765 | 919K | **2.0M** | 0.04% | OK |
| SU(32) | 2,046 | 3.0M | **5.1M** | 0.01% | OK |
| SU(64) | 8,190 | 37M | **42.9M** | 0.002% | OK (custom) |
| SU(128) | 32,766 | 539M | **571.8M** | 0.0001% | Slow |
| SU(256) | 65,535 | 2.2B | **2.2B** | ~0% | GPU only |

**Key insight:** The quantum parameter count (736) is **fixed** regardless of SU encoding size. All the parameter growth occurs in the classical `enc_proj` layer that projects features into encoding parameters. Larger SU gates absorb more information but at the cost of a proportionally larger classical projection.

---

## 13. Practical Considerations

### 13.1. When to Use Each Architecture

| Scenario | Recommended | Reason |
|----------|------------|--------|
| Best image quality | QLCFM v9 | Latent-space advantage; VAE handles reconstruction |
| Strongest quantum argument | Direct CFM v3 | Theoretically optimal (full encoding + full measurement) |
| Ablation: measurement expressiveness | v2 vs v3 | Same encoding, different ANO locality |
| Ablation: encoding strategy | v1 vs v2 | Same ANO, different encoding (brick-layer vs single-gate) |
| Testing SU(N) scaling | Direct CFM v1 | Supports SU(4)--SU(256) via `--sun-group-size` |
| Quick prototyping | Direct CFM v2 | Single-phase, no VAE, cleanest setup |
| Large images (128x128+) | QLCFM v9 | Pixel-space flow matching struggles at high resolution |
| Publication: primary results | QLCFM v9 + Direct CFM v3 | Quality (v9) + quantum contribution (v3) |
| Publication: full ablation | v1 + v2 + v3 | Encoding effect (v1→v2) + measurement effect (v2→v3) |

### 13.2. Memory and Compute Requirements

| Config | Memory (batch=64) | Time/epoch (est.) |
|--------|------------------|-------------------|
| v1 SU(4), 8q | ~2 GB | ~500s |
| v1 SU(16), 8q | ~3 GB | ~800s |
| v1 SU(64), 8q | ~10 GB | ~3000s |
| **v2 SU(64), 6q** | ~10 GB | ~3000s |
| **v3 SU(64), 6q** | ~12 GB | ~3500s |
| v1 SU(128), 8q | ~40 GB+ | ~10000s+ |
| QLCFM v9 SU(4), 8q | ~4 GB (with VAE) | ~500s |

Note: v3 requires slightly more memory and compute than v2 due to the 15 global 64×64 Hermitian matrices (vs 15 pairwise 4×4 Hermitians).

### 13.3. Expected Behavior

- **Direct CFM (v1, v2, v3) will likely produce worse images** than QLCFM v9, because pixel-space flow matching is fundamentally harder than latent-space flow matching
- **v3 is the most scientifically valuable** because it tests the theoretically optimal quantum pipeline (maximal encoding + maximal measurement) and demonstrates whether capturing all n-body correlations improves generation
- **v2 vs v3 comparison** directly answers: do higher-order correlations (3-body through 6-body) matter for image generation when the quantum state is fully entangled?
- **v1 vs v2 comparison** isolates the effect of brick-layer vs single-gate encoding in pixel space
- **v1 SU scaling experiment** will show whether larger encoding capacity compensates for the lack of a VAE

---

## 14. Usage Examples

### 14.1. Training (v1)

```bash
# Quantum SU(4) on CIFAR-10 (v1 default)
python models/QuantumDirectCFM.py \
    --dataset=cifar10 --velocity-field=quantum \
    --n-qubits=8 --sun-group-size=2 \
    --epochs=200 --batch-size=64 \
    --job-id=dcfm_su4_cifar

# Quantum SU(16) on CIFAR-10 (v1)
python models/QuantumDirectCFM.py \
    --dataset=cifar10 --velocity-field=quantum \
    --n-qubits=8 --sun-group-size=4 \
    --epochs=200 --batch-size=64 \
    --job-id=dcfm_su16_cifar
```

### 14.2. Training (v2)

```bash
# Quantum SU(64), 6 qubits on CIFAR-10 (v2 default, auto enc_channels)
python models/QuantumDirectCFM_v2.py \
    --dataset=cifar10 --velocity-field=quantum \
    --epochs=200 --batch-size=64 \
    --job-id=dcfm2_su64_cifar

# Quantum SU(64), 6 qubits on COCO 128x128 (auto-configured Conv layers)
python models/QuantumDirectCFM_v2.py \
    --dataset=coco --img-size=128 --velocity-field=quantum \
    --epochs=200 --batch-size=32 \
    --job-id=dcfm2_su64_coco128

# With optional QViT ansatz (for ablation)
python models/QuantumDirectCFM_v2.py \
    --dataset=cifar10 --velocity-field=quantum \
    --vqc-type=qvit --vqc-depth=1 \
    --epochs=200 --batch-size=64 \
    --job-id=dcfm2_su64_qvit_cifar

# Classical baseline
python models/QuantumDirectCFM_v2.py \
    --dataset=cifar10 --velocity-field=classical \
    --epochs=200 --batch-size=64 \
    --job-id=dcfm2_classical_cifar

# Resume from checkpoint
python models/QuantumDirectCFM_v2.py \
    --dataset=cifar10 --velocity-field=quantum \
    --resume --job-id=dcfm2_su64_cifar
```

### 14.3. Training (v3)

```bash
# Quantum SU(64), 6 qubits, global ANO on CIFAR-10 (v3 default)
python models/QuantumDirectCFM_v3.py \
    --dataset=cifar10 --velocity-field=quantum \
    --n-qubits=6 --n-observables=15 \
    --epochs=200 --batch-size=64 \
    --job-id=dcfm3_su64_cifar

# Quantum SU(64), 6 qubits, global ANO on COCO 128x128
python models/QuantumDirectCFM_v3.py \
    --dataset=coco --img-size=128 --velocity-field=quantum \
    --n-qubits=6 --n-observables=15 \
    --epochs=200 --batch-size=32 \
    --job-id=dcfm3_su64_coco128

# Classical baseline
python models/QuantumDirectCFM_v3.py \
    --dataset=cifar10 --velocity-field=classical \
    --epochs=200 --batch-size=64 \
    --job-id=dcfm3_classical_cifar

# Resume from checkpoint
python models/QuantumDirectCFM_v3.py \
    --dataset=cifar10 --velocity-field=quantum \
    --resume --job-id=dcfm3_su64_cifar
```

### 14.4. SLURM Batch Script Template (v1)

```bash
#!/bin/bash
#SBATCH -A m4807_g
#SBATCH -C gpu&hbm80g
#SBATCH -q shared
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH -t 48:00:00
#SBATCH -J dcfm_su4
#SBATCH -o logs/dcfm_su4_%j.out
#SBATCH -e logs/dcfm_su4_%j.err

export PYTHONNOUSERSITE=1
eval "$(conda shell.bash hook 2>/dev/null)"
conda activate /pscratch/sd/j/junghoon/conda-envs/qml_eeg

cd /pscratch/sd/j/junghoon/QuantumFlow

python -u models/QuantumDirectCFM.py \
    --dataset=cifar10 \
    --velocity-field=quantum \
    --n-qubits=8 \
    --sun-group-size=2 \
    --vqc-depth=2 \
    --k-local=2 \
    --obs-scheme=pairwise \
    --qvit-circuit=butterfly \
    --epochs=200 \
    --batch-size=64 \
    --n-train=10000 \
    --n-valtest=2000 \
    --seed=2025 \
    --compute-metrics \
    --job-id=dcfm_su4_${SLURM_JOB_ID}
```

### 14.5. SLURM Batch Script Template (v2)

```bash
#!/bin/bash
#SBATCH -A m4807_g
#SBATCH -C gpu&hbm80g
#SBATCH -q shared
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH -t 48:00:00
#SBATCH -J dcfm2_su64
#SBATCH -o logs/dcfm2_su64_%j.out
#SBATCH -e logs/dcfm2_su64_%j.err

export PYTHONNOUSERSITE=1
eval "$(conda shell.bash hook 2>/dev/null)"
conda activate /pscratch/sd/j/junghoon/conda-envs/qml_eeg

cd /pscratch/sd/j/junghoon/QuantumFlow

python -u models/QuantumDirectCFM_v2.py \
    --dataset=cifar10 \
    --velocity-field=quantum \
    --n-qubits=6 \
    --k-local=2 \
    --obs-scheme=pairwise \
    --epochs=200 \
    --batch-size=64 \
    --n-train=10000 \
    --n-valtest=2000 \
    --seed=2025 \
    --compute-metrics \
    --job-id=dcfm2_su64_${SLURM_JOB_ID}
```

### 14.6. SLURM Batch Script Template (v3)

```bash
#!/bin/bash
#SBATCH -A m4807_g
#SBATCH -C gpu&hbm80g
#SBATCH -q shared
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH -t 48:00:00
#SBATCH -J dcfm3_su64
#SBATCH -o logs/dcfm3_su64_%j.out
#SBATCH -e logs/dcfm3_su64_%j.err

export PYTHONNOUSERSITE=1
eval "$(conda shell.bash hook 2>/dev/null)"
conda activate /pscratch/sd/j/junghoon/conda-envs/qml_eeg

cd /pscratch/sd/j/junghoon/QuantumFlow

python -u models/QuantumDirectCFM_v3.py \
    --dataset=cifar10 \
    --velocity-field=quantum \
    --n-qubits=6 \
    --n-observables=15 \
    --vqc-type=none \
    --enc-channels=auto \
    --time-embed-dim=256 \
    --lr=1e-3 \
    --lr-H=1e-1 \
    --epochs=200 \
    --batch-size=64 \
    --n-train=10000 \
    --n-valtest=2000 \
    --seed=2025 \
    --logit-normal-std=1.0 \
    --ode-solver=midpoint \
    --ode-steps=50 \
    --vf-ema-decay=0.999 \
    --compute-metrics \
    --job-id=dcfm3_su64_${SLURM_JOB_ID}
```

### 14.7. CLI Arguments

#### v1 (`QuantumDirectCFM.py`)

| Argument | Default | Description |
|----------|---------|-------------|
| `--velocity-field` | quantum | `quantum` or `classical` |
| `--n-qubits` | 8 | Number of qubits |
| `--sun-group-size` | 2 | Qubits per SU gate (2=SU(4), ..., 8=SU(256)) |
| `--encoding-type` | sun | `sun` (exponential map) or `angle` |
| `--vqc-type` | qvit | `qvit`, `hardware_efficient`, or `none` |
| `--vqc-depth` | 2 | Number of VQC layers |
| `--qvit-circuit` | butterfly | QViT topology: `butterfly`, `pyramid`, `x` |
| `--k-local` | 2 | Locality of ANO measurements |
| `--obs-scheme` | pairwise | `pairwise` or `sliding` |
| `--enc-channels` | 32,64,128 | Channel progression for Conv encoder |
| `--n-circuits` | 1 | Number of parallel quantum circuits |
| `--logit-normal-std` | 1.0 | Logit-normal timestep std (0=uniform) |
| `--ode-solver` | midpoint | `euler` or `midpoint` |
| `--vf-ema-decay` | 0.999 | EMA decay (0=disable) |
| `--dataset` | cifar10 | `cifar10`, `coco`, `mnist`, `fashion` |
| `--img-size` | 32 | Image resolution |
| `--epochs` | 200 | Training epochs |
| `--batch-size` | 64 | Batch size |
| `--compute-metrics` | False | Compute FID and IS after training |

#### v2 (`QuantumDirectCFM_v2.py`)

| Argument | Default | Description |
|----------|---------|-------------|
| `--velocity-field` | quantum | `quantum` or `classical` |
| `--n-qubits` | **6** | Number of qubits (single SU(2^n) gate) |
| `--enc-channels` | **auto** | `auto` or comma-separated (e.g., `32,64`) |
| `--vqc-type` | **none** | `qvit`, `hardware_efficient`, or `none` |
| `--vqc-depth` | 2 | Number of VQC layers (if vqc-type != none) |
| `--qvit-circuit` | butterfly | QViT topology (if vqc-type = qvit) |
| `--k-local` | 2 | Locality of ANO measurements |
| `--obs-scheme` | pairwise | `pairwise` or `sliding` |
| `--time-embed-dim` | 256 | Sinusoidal time embedding dimension |
| `--n-circuits` | 1 | Number of parallel quantum circuits |
| `--logit-normal-std` | 1.0 | Logit-normal timestep std (0=uniform) |
| `--ode-solver` | midpoint | `euler` or `midpoint` |
| `--vf-ema-decay` | 0.999 | EMA decay (0=disable) |
| `--dataset` | cifar10 | `cifar10`, `coco`, `mnist`, `fashion` |
| `--img-size` | 32 | Image resolution |
| `--epochs` | 200 | Training epochs |
| `--batch-size` | 64 | Batch size |
| `--compute-metrics` | False | Compute FID and IS after training |

Key differences from v1:
- No `--sun-group-size` or `--encoding-type` (always single SU(2^n) on all qubits)
- `--enc-channels=auto` automatically computes Conv layers for ~1:1 enc_proj ratio
- `--vqc-type=none` by default (no QViT ansatz)
- `--n-qubits=6` by default (vs v1's 8)

#### v3 (`QuantumDirectCFM_v3.py`)

| Argument | Default | Description |
|----------|---------|-------------|
| `--velocity-field` | quantum | `quantum` or `classical` |
| `--n-qubits` | **6** | Number of qubits (single SU(2^n) gate) |
| `--n-observables` | **15** | Number of independent global Hermitian observables |
| `--enc-channels` | **auto** | `auto` or comma-separated (e.g., `32,64`) |
| `--vqc-type` | **none** | `qvit`, `hardware_efficient`, or `none` |
| `--vqc-depth` | 2 | Number of VQC layers (if vqc-type != none) |
| `--time-embed-dim` | 256 | Sinusoidal time embedding dimension |
| `--n-circuits` | 1 | Number of parallel quantum circuits |
| `--logit-normal-std` | 1.0 | Logit-normal timestep std (0=uniform) |
| `--ode-solver` | midpoint | `euler` or `midpoint` |
| `--vf-ema-decay` | 0.999 | EMA decay (0=disable) |
| `--dataset` | cifar10 | `cifar10`, `coco`, `mnist`, `fashion` |
| `--img-size` | 32 | Image resolution |
| `--epochs` | 200 | Training epochs |
| `--batch-size` | 64 | Batch size |
| `--compute-metrics` | False | Compute FID and IS after training |

Key differences from v2:
- No `--k-local` or `--obs-scheme` (always global ANO on all qubits)
- `--n-observables` controls how many independent 2^n × 2^n Hermitians to create
- Each observable acts on ALL qubits (k=n_qubits), capturing all 1-through-n-body correlations
- ANO params = n_observables × (2^n)² (e.g., 15 × 4096 = 61,440 for 6 qubits)

---

## References

1. Lipman et al. (2023). Flow Matching for Generative Modeling. ICLR 2023.
2. Esser et al. (2024). Scaling Rectified Flow Transformers for High-Resolution Image Synthesis. ICML 2024.
3. Wiersema et al. (2024). Here comes the SU(N): multivariate quantum gates and gradients. Quantum, 8, 1275.
4. Chen et al. (2025). Learning to Measure Quantum Neural Networks. ICASSP 2025 Workshop.
5. Lin et al. (2025). Adaptive Non-local Observable on Quantum Neural Networks. IEEE QCE 2025.
6. Cherrat et al. (2024). Quantum Vision Transformers. Quantum, 8, 1265.
7. Vaswani et al. (2017). Attention Is All You Need. NeurIPS 2017.
8. Rombach et al. (2022). High-Resolution Image Synthesis with Latent Diffusion Models. CVPR 2022.
