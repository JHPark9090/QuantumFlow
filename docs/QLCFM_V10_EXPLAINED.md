# QLCFM v10: Single-Gate SU(16) with Adaptive Non-Local Observables

## Overview

QLCFM v10 represents a fundamental shift in the quantum velocity field architecture. Instead of using brick-layer SU(4) encoding with a QViT ansatz circuit (v9), v10 uses a **single SU(16) gate on all 4 qubits** with **Adaptive Non-Local Observables (ANO)** for measurement. This eliminates the need for a post-encoding variational ansatz entirely.

Four configurations (v10a--v10d) explore two independent design axes:
- **Time conditioning**: how the diffusion timestep is injected into the circuit
- **ANO type**: how quantum observables are structured for measurement

## v9 vs v10: Architectural Comparison

### v9: Brick-Layer SU(4) + QViT Ansatz

In v9, the quantum velocity field operates on **8 qubits** using a two-stage pipeline:

**Stage 1 -- Brick-Layer SU(4) Encoding:**

The classical input vector (latent code concatenated with time embedding) is encoded into the quantum state via pairs of SU(4) gates arranged in a brick-layer pattern:

```
q0 ─[SU(4)]─         ─[SU(4)]─
q1 ─[SU(4)]─ ─[SU(4)]─[SU(4)]─
q2 ─[SU(4)]─ ─[SU(4)]─[SU(4)]─
q3 ─[SU(4)]─ ─[SU(4)]─[SU(4)]─
q4 ─[SU(4)]─ ─[SU(4)]─[SU(4)]─
q5 ─[SU(4)]─ ─[SU(4)]─[SU(4)]─
q6 ─[SU(4)]─ ─[SU(4)]─
q7 ─[SU(4)]─
     Even        Odd
```

Each SU(4) gate acts on **2 qubits** and has **15 free parameters** (4^2 - 1 = 15 generators of SU(4)). For 8 qubits:
- Even layer: pairs (0,1), (2,3), (4,5), (6,7) = 4 gates x 15 = 60 params
- Odd layer: pairs (1,2), (3,4), (5,6) = 3 gates x 15 = 45 params
- **Total encoding parameters: 105**

A linear projection maps the input vector to these 105 parameters. This is the `enc_proj` layer.

**Entanglement limitation:** Each SU(4) gate entangles only its 2-qubit pair. While the odd layer creates some cross-pair correlations, qubits that are far apart (e.g., q0 and q7) have **no direct entanglement** from encoding alone. This is why v9 needs a QViT ansatz to build long-range entanglement.

**Stage 2 -- QViT Ansatz (Pyramid):**

After encoding, a parametrized variational circuit (QViT) is applied to create entanglement across all qubits. The Pyramid ansatz applies U3 + Ising gates in a pattern inspired by Vision Transformers:

```
Depth 1:  U3(q0)──Ising──U3(q1)   U3(q2)──Ising──U3(q3)   ...
Depth 2:  U3(q0)──Ising──U3(q2)   U3(q1)──Ising──U3(q3)   ...
```

The QViT ansatz has its own trainable parameters (separate from the encoding). These are optimized alongside the encoding projection during training.

**Measurement:**

Pairwise ANO with k=2: C(8,2) = 28 wire pairs, each measured with a learnable 4x4 Hermitian matrix. Total: 28 x 16 = 448 ANO parameters.

### v10: Single SU(16) Gate, No Ansatz

In v10, the quantum velocity field operates on **4 qubits** using a radically simpler architecture:

**Single-Gate SU(2^n) Encoding:**

Instead of applying SU(4) gates to pairs of qubits, v10 applies a **single SU(16) gate to all 4 qubits simultaneously**:

```
q0 ─┐
q1 ─┤ SU(16)
q2 ─┤
q3 ─┘
```

This single gate has **255 free parameters** (16^2 - 1 = 255 generators of SU(16)). A linear projection maps the input vector to these 255 parameters.

**Why this works better with fewer qubits:**

1. **100% entanglement coverage**: The SU(16) gate acts on the full 16-dimensional Hilbert space of 4 qubits. Every qubit is directly entangled with every other qubit in a single operation. There are no "distant" qubits that lack direct entanglement.

2. **QViT becomes redundant**: The purpose of the QViT ansatz in v9 was to build entanglement that the brick-layer encoding couldn't provide. With a single SU(16) gate, the encoding itself already explores the full Hilbert space. Adding a QViT ansatz on top would be redundant -- in the Heisenberg picture, applying a unitary W after encoding U is equivalent to measuring a rotated observable W^dag H W, which the learnable ANO Hermitians already cover.

3. **Near-perfect enc_proj ratio**: With 255 encoding parameters, an input vector of dimension ~256 achieves a ~1:1 encoding ratio. For comparison, v9 with 8 qubits only has 105 encoding parameters, requiring significant compression of larger inputs.

**Why only 4 qubits?**

The number of SU(2^n) generators grows as (2^n)^2 - 1, which is exponential in n:

| Qubits (n) | Hilbert dim (2^n) | SU generators ((2^n)^2 - 1) |
|-------------|-------------------|------------------------------|
| 2           | 4                 | 15                           |
| 3           | 8                 | 63                           |
| **4**       | **16**            | **255**                      |
| 5           | 32                | 1,023                        |
| 6           | 64                | 4,095                        |

At 4 qubits, the 255 generators provide a good match for typical latent dimensions (128, 256, 512). Adding more qubits would exponentially increase the encoding dimension, requiring either massive latent spaces or wasteful parameter redundancy. Furthermore, PennyLane's `SpecialUnitary` gate becomes computationally expensive beyond 5 qubits, as the generator matrices grow as 2^n x 2^n.

### Side-by-Side Summary

| Aspect | v9 | v10 |
|--------|-----|-----|
| Qubits | 8 | 4 |
| Encoding | Brick-layer SU(4) pairs | Single SU(16) on all qubits |
| Encoding params | 105 (7 gates x 15) | 255 (1 gate x 255) |
| Entanglement from encoding | Partial (nearest-neighbor pairs) | Full (all qubits) |
| QViT ansatz | Required (Pyramid/Butterfly/X) | Not needed (redundant) |
| ANO types | Pairwise only | Pairwise or Global |
| Time conditioning | Concat only | Concat or Additive |
| Hilbert space utilization | Partial | Full |
| Model file | `QuantumLatentCFM_v9.py` | `QuantumLatentCFM_v10.py` |

## Why QViT Was Replaced by ANO

### The Role of QViT in v9

In v9, the brick-layer SU(4) encoding only entangles neighboring qubit pairs. The QViT ansatz serves as a **post-encoding entangling circuit** that builds correlations across all qubits. Without it, distant qubits would remain unentangled, limiting the expressivity of the quantum state.

### Why QViT Is Unnecessary in v10

The single SU(16) gate in v10 already generates **arbitrary 4-qubit entangled states** from the encoding step alone. In the Heisenberg picture, the measurement process can be understood as:

```
Expectation value = <0| U^dag(x) H U(x) |0>
```

where U(x) is the encoding unitary parameterized by input x, and H is the observable. If we were to add a variational ansatz W after encoding:

```
Expectation value = <0| U^dag(x) W^dag H W U(x) |0>
```

This is equivalent to measuring a **rotated observable** H' = W^dag H W. Since our ANO Hermitians are already fully learnable, they can absorb any rotation that W would provide. The ansatz W is therefore redundant.

This insight -- that learnable observables can replace post-encoding circuits -- comes from the **Adaptive Non-Local Observable (ANO)** framework (Lin et al., 2025; Chen et al., 2025). Instead of adding circuit depth to build expressivity, we make the measurement operators themselves learnable.

### ANO: Learnable Hermitian Observables

Traditional quantum circuits use fixed Pauli observables (X, Y, Z) for measurement. ANO replaces these with **learnable Hermitian matrices** that are optimized during training alongside the circuit parameters.

Each ANO Hermitian H is parametrized as:

```
H = A^dag A + diag(D)
```

where:
- A is a complex matrix (upper triangular, with learnable real and imaginary parts)
- D is a real diagonal vector
- This ensures H is Hermitian (H = H^dag) and positive semi-definite

The ANO parameters are trained with a **separate optimizer** at a higher learning rate (lr=0.1 for ANO vs lr=0.001 for circuit parameters), following the dual-optimizer strategy from Chen et al. (2025).

## Pairwise ANO vs Global ANO

The two ANO types differ in **which qubits are measured together** and **what correlations they can capture**.

### Pairwise ANO (k=2)

Pairwise ANO measures **2-qubit subsystems**. For 4 qubits, there are C(4,2) = 6 possible pairs:

```
Observable 1: qubits (0, 1)  →  4x4 Hermitian
Observable 2: qubits (0, 2)  →  4x4 Hermitian
Observable 3: qubits (0, 3)  →  4x4 Hermitian
Observable 4: qubits (1, 2)  →  4x4 Hermitian
Observable 5: qubits (1, 3)  →  4x4 Hermitian
Observable 6: qubits (2, 3)  →  4x4 Hermitian
```

Each 4x4 Hermitian has 16 free parameters (6 off-diagonal real + 6 off-diagonal imaginary + 4 diagonal).

- **Total ANO parameters**: 6 observables x 16 params = **96 parameters**
- **Correlation order**: Captures up to **2-body correlations** -- how pairs of qubits are correlated
- **Cannot capture**: 3-body or 4-body correlations (e.g., simultaneous entanglement patterns across 3+ qubits)

### Global ANO (k=4)

Global ANO measures the **full 4-qubit system** with each observable. All 6 observables act on all 4 qubits simultaneously:

```
Observable 1: qubits (0, 1, 2, 3)  →  16x16 Hermitian
Observable 2: qubits (0, 1, 2, 3)  →  16x16 Hermitian
Observable 3: qubits (0, 1, 2, 3)  →  16x16 Hermitian
Observable 4: qubits (0, 1, 2, 3)  →  16x16 Hermitian
Observable 5: qubits (0, 1, 2, 3)  →  16x16 Hermitian
Observable 6: qubits (0, 1, 2, 3)  →  16x16 Hermitian
```

Each 16x16 Hermitian has 256 free parameters (120 off-diagonal real + 120 off-diagonal imaginary + 16 diagonal).

- **Total ANO parameters**: 6 observables x 256 params = **1,536 parameters**
- **Correlation order**: Captures **all 1-through-4-body correlations** -- every possible entanglement pattern in the 4-qubit system
- **Strictly more expressive**: Any pairwise observable can be embedded in a global observable (by zeroing out higher-order terms), but not vice versa

### Comparison

| Aspect | Pairwise (k=2) | Global (k=4) |
|--------|----------------|--------------|
| Wire groups | C(4,2) = 6 pairs | 6 copies of all 4 qubits |
| Hermitian size | 4x4 | 16x16 |
| Params per observable | 16 | 256 |
| Total ANO params | 96 | 1,536 |
| Max correlation order | 2-body | 4-body (all) |
| Expressivity | Limited to pairwise | Full Hilbert space |
| Training cost | Lower (fewer params) | Higher (16x more params) |

## v10 Configurations (v10a--v10d)

The four v10 variants explore two independent design axes in a 2x2 grid:

|                  | Pairwise ANO (k=2)  | Global ANO (k=4)    |
|------------------|----------------------|----------------------|
| **Concat time**  | **v10a**             | **v10c**             |
| **Additive time**| **v10b**             | **v10d**             |

### Time Conditioning: Concat vs Additive

**Concat** (v10a, v10c): The noisy latent code z_t and time embedding t_emb are concatenated before encoding:

```
z_combined = [z_t, t_emb]    # dim = latent_dim + time_embed_dim
```

For latent_dim=256 with time_embed_dim=256: input_dim = 512, projected to 255 SU(16) params (2.01:1 compression).

For latent_dim=128 with time_embed_dim=128: input_dim = 256, projected to 255 SU(16) params (1.00:1 ratio).

**Additive** (v10b, v10d): The time embedding is projected to the same dimension as z_t and added:

```
t_proj = time_mlp(t_emb)     # dim = latent_dim
z_combined = z_t + t_proj    # dim = latent_dim
```

For latent_dim=256: input_dim = 256, projected to 255 SU(16) params (1.00:1 ratio).

For latent_dim=128: input_dim = 128, projected to 255 SU(16) params (0.50:1 expansion).

### Encoding Ratio (enc_proj) Across Configurations

The enc_proj ratio measures how much information compression or expansion occurs between the input vector and the SU(16) encoding parameters. A 1:1 ratio is ideal.

| Config | Latent | Time Cond. | Input Dim | SU(16) Params | Ratio |
|--------|--------|------------|-----------|---------------|-------|
| v10a   | 128    | concat     | 256       | 255           | 1.00:1 |
| v10a   | 256    | concat     | 512       | 255           | 2.01:1 |
| v10a   | 512    | concat     | 1024      | 255           | 4.02:1 |
| v10b   | 128    | additive   | 128       | 255           | 0.50:1 |
| v10b   | 256    | additive   | 256       | 255           | 1.00:1 |
| v10b   | 512    | additive   | 512       | 255           | 2.01:1 |

The same ratios apply to v10c/v10d (which share time conditioning with v10a/v10b respectively).

### Configuration Details

**v10a -- Concat + Pairwise ANO:**
- Time: z_combined = [z_t, t_emb]
- ANO: 6 pairwise 4x4 Hermitians (96 params)
- Best enc_proj ratio at latent_dim=128 (1.00:1)

**v10b -- Additive + Pairwise ANO:**
- Time: z_combined = z_t + time_mlp(t)
- ANO: 6 pairwise 4x4 Hermitians (96 params)
- Best enc_proj ratio at latent_dim=256 (1.00:1)

**v10c -- Concat + Global ANO:**
- Time: z_combined = [z_t, t_emb]
- ANO: 6 global 16x16 Hermitians (1,536 params)
- Best enc_proj ratio at latent_dim=128 (1.00:1)

**v10d -- Additive + Global ANO:**
- Time: z_combined = z_t + time_mlp(t)
- ANO: 6 global 16x16 Hermitians (1,536 params)
- Best enc_proj ratio at latent_dim=256 (1.00:1)

## Training Details

All v10 models inherit the training improvements from v9:

1. **Logit-Normal Timestep Sampling** (Esser et al., 2024): t ~ sigmoid(Normal(0, sigma)) instead of Uniform(0,1). Concentrates training on mid-range timesteps where the velocity field is hardest.

2. **Midpoint ODE Solver** (2nd-order): Replaces Euler for generation. 50 midpoint steps = 100 velocity field evaluations, with 2nd-order accuracy.

3. **EMA for Velocity Field**: Exponential moving average (decay=0.999) of velocity field weights. Uses EMA weights for validation and generation.

4. **Dual Optimizer** (Chen et al., 2025): Separate Adam optimizers for ANO Hermitian parameters (lr=0.1) and all other parameters (lr=0.001).

## References

- Wiersema et al. (2024). "Here comes the SU(N): multivariate quantum gates and gradients." *Quantum*, 8, 1275.
- Lin et al. (2025). "Adaptive Non-local Observable on Quantum Neural Networks." *IEEE QCE 2025*.
- Chen et al. (2025). "Learning to Measure Quantum Neural Networks." *ICASSP 2025 Workshop*.
- Cherrat et al. (2024). "Quantum Vision Transformers." *Quantum*, 8, 1265.
- Esser et al. (2024). "Scaling Rectified Flow Transformers for High-Resolution Image Synthesis." *ICML 2024*.
- Lipman et al. (2023). "Flow Matching for Generative Modeling." *ICLR 2023*.
