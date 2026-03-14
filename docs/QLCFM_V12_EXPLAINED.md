# QLCFM v12: SD3.5 Pretrained VAE with v9 Quantum Circuit (Brick-Layer SU(4) + QViT)

## Overview

QLCFM v12 combines the **SD3.5 pretrained VAE with classical bottleneck** (from v11) with the **v9 quantum circuit architecture** (8 qubits, brick-layer SU(4) encoding, QViT pyramid ansatz, pairwise ANO). This provides a direct comparison between the two quantum circuit designs under identical VAE conditions:

- **v11**: SD3.5 VAE + bottleneck + **v10 circuit** (4q, single SU(16), ANO-only)
- **v12**: SD3.5 VAE + bottleneck + **v9 circuit** (8q, brick-layer SU(4), QViT+ANO)

## Why v12?

The v9 and v10 quantum circuits represent fundamentally different design philosophies:

| Aspect | v9 (used in v12) | v10 (used in v11) |
|--------|------------------|-------------------|
| Qubits | 8 | 4 |
| Encoding | Brick-layer SU(4) pairs | Single SU(16) on all qubits |
| Encoding params | 105 (7 gates x 15) | 255 (1 gate x 255) |
| Entanglement from encoding | Partial (nearest-neighbor) | Full (all qubits) |
| Post-encoding ansatz | QViT pyramid (trainable) | None (redundant) |
| ANO | Pairwise k=2, 28 obs | Pairwise/Triple/Global |
| Hilbert space dim | 2^8 = 256 | 2^4 = 16 |

In v9/v10 comparisons on CIFAR-10 32x32 with custom VAE, we couldn't isolate whether performance differences were due to the circuit or the VAE. By running both circuits with the **identical SD3.5 VAE**, v12 enables a clean apples-to-apples comparison.

### Key Question v12 Answers

**Does a larger Hilbert space (8 qubits, dim=256) with partial entanglement + QViT ansatz outperform a smaller but fully-entangled space (4 qubits, dim=16) with learnable observables?**

## Architecture

### Full Data Flow

```
CIFAR-10 (32x32) ─── Bicubic Upscale ──→ CIFAR-10 (128x128)
    │
    ▼
SD3.5 VAE Encoder (frozen, ~84M params) → Latent (4096-d)
    │
    ▼
┌────────────────────────┐
│  bottleneck_in         │  Linear(4096 → 256)
│  (learnable)           │
└────────────────────────┘
    │
    ├── [concat] time embedding (256-d)
    │               ↓
    │   Input vector (512-d)
    ▼
┌────────────────────────┐
│  enc_proj              │  Linear(512 → 256 → 105)
│  (learnable MLP)       │
└────────────────────────┘
    │
    ▼
┌────────────────────────┐
│  Brick-Layer SU(4)     │  8 qubits, even+odd layers
│  Encoding              │  7 gates x 15 params = 105
│                        │
│  q0─┐SU(4)             │
│  q1─┘    ┌─SU(4)       │
│  q2─┐SU(4)─┘           │
│  q3─┘    ┌─SU(4)       │
│  q4─┐SU(4)─┘           │
│  q5─┘    ┌─SU(4)       │
│  q6─┐SU(4)─┘           │
│  q7─┘                  │
│    Even    Odd          │
└────────────────────────┘
    │
    ▼
┌────────────────────────┐
│  QViT Pyramid Ansatz   │  depth=2, trainable params
│  (post-encoding)       │  Builds cross-pair entanglement
│                        │
│  U3──Ising──U3         │
│    ──Ising──           │
│  (pyramid pattern)     │
└────────────────────────┘
    │
    ▼
┌────────────────────────┐
│  Pairwise ANO k=2      │  C(8,2)=28 wire pairs
│  (learnable Hermitians)│  28 x 4x4 = 448 ANO params
└────────────────────────┘
    │
    ▼
28 expectation values
    │
    ▼
┌────────────────────────┐
│  vel_head              │  Linear(28 → 256 → 256)
│  (learnable MLP)       │
└────────────────────────┘
    │
    ▼
┌────────────────────────┐
│  bottleneck_out        │  Linear(256 → 4096)
│  (learnable)           │
└────────────────────────┘
    │
    ▼
Predicted Velocity (4096-d)
    │
    ▼
SD3.5 VAE Decoder (frozen) → Generated Image (128x128)
```

### v12 vs v11 Circuit Comparison

```
v11 (v10 circuit):                    v12 (v9 circuit):

Input (512-d or 256-d)                Input (512-d)
    │                                     │
    ▼                                     ▼
enc_proj → 255 params                enc_proj → 105 params
    │                                     │
    ▼                                     ▼
┌──────────┐                          ┌──────────────────┐
│ SU(16)   │ ← single gate,          │ SU(4) brick-layer│ ← 7 gates,
│ 4 qubits │   full entanglement     │ 8 qubits         │   partial entanglement
└──────────┘                          └──────────────────┘
    │                                     │
    │ (no ansatz needed)                  ▼
    │                                 ┌──────────────────┐
    │                                 │ QViT pyramid     │ ← trainable,
    │                                 │ depth=2          │   builds entanglement
    │                                 └──────────────────┘
    ▼                                     │
ANO measurement                           ▼
(6 or 4 obs)                          ANO measurement
                                      (28 obs)
    │                                     │
    ▼                                     ▼
vel_head → 256-d                      vel_head → 256-d
```

### enc_proj Ratio

The v9 circuit has only 105 encoding parameters (vs 255 for v10's SU(16)), leading to a higher compression ratio:

| Model | Input Dim | Encoding Params | Ratio |
|-------|-----------|----------------|-------|
| v11a (v10, concat) | 512 | 255 | 2.01:1 |
| v11b (v10, additive) | 256 | 255 | 1.00:1 |
| **v12a (v9, concat)** | **512** | **105** | **4.88:1** |

The 4.88:1 ratio is higher than ideal (1:1), meaning significant information is compressed before entering the quantum circuit. However, this is inherent to the v9 architecture — the QViT ansatz compensates by building additional expressivity after encoding.

### Parameter Comparison

| Component | v11a (v10 circuit) | v12a (v9 circuit) |
|-----------|-------------------|-------------------|
| SD3.5 VAE | ~84M (frozen) | ~84M (frozen) |
| Bottleneck in | 1,048,832 | 1,048,832 |
| Bottleneck out | 1,052,672 | 1,052,672 |
| enc_proj | ~131K | ~27K |
| QViT ansatz | 0 | 6,720 (28 RBS x 12 x 2 depths) |
| ANO Hermitians | 96 (k=2, 6 obs) | 448 (k=2, 28 obs) |
| vel_head | ~66K | ~7.4K (28→256→256) |
| Time MLP | ~131K | ~131K |
| **Total trainable** | **~2.4M** | **~2.3M** |

Both have similar total parameter counts (~2.3-2.4M), dominated by the bottleneck layers. The quantum circuit itself has more parameters in v12 (QViT params + 28 ANO obs vs 6 ANO obs), but the bottleneck dominates.

## Configuration: v12a

v12 has one primary configuration (v9 uses concat-only time conditioning):

**v12a -- SD3.5 VAE + Bottleneck + v9 Circuit:**
- VAE: SD3.5 pretrained, frozen
- Image: CIFAR-10, 128x128 (bicubic upscaled from 32x32)
- Bottleneck: Linear(4096→256) + Linear(256→4096)
- Circuit: 8 qubits, brick-layer SU(4), 105 encoding params
- Ansatz: QViT pyramid, depth=2
- ANO: Pairwise k=2, C(8,2)=28 observables, 4x4 Hermitians
- Time: concat [z_bottleneck(256), t_emb(256)] = 512 input
- enc_proj ratio: 4.88:1

### Training Configuration

All inherited from v9:

| Parameter | Value |
|-----------|-------|
| Epochs | 200 |
| Batch size | 64 |
| lr (circuit + bottleneck) | 1e-3 |
| lr (ANO Hermitians) | 1e-1 |
| Logit-normal std | 1.0 |
| ODE solver | midpoint (50 steps) |
| VF EMA decay | 0.999 |
| n_train | 10,000 |
| n_valtest | 2,000 |
| Seed | 2025 |

## Expected Outcomes

### Hypothesis 1: v10 Circuit (v11) Outperforms v9 Circuit (v12)

The v10 circuit's single SU(16) gate provides full entanglement in one operation, with a near-ideal enc_proj ratio for additive conditioning. v12's 4.88:1 compression and partial entanglement may limit performance despite the larger Hilbert space and QViT ansatz.

**If confirmed**: Validates the v10 design philosophy — full entanglement from encoding makes post-encoding ansatz unnecessary.

### Hypothesis 2: v9 Circuit (v12) Outperforms v10 Circuit (v11)

The larger Hilbert space (dim=256 vs dim=16) and 28 pairwise observables (vs 6) may capture richer features. The QViT ansatz adds expressivity that the enc_proj ratio cannot.

**If confirmed**: Suggests that Hilbert space dimension matters more than enc_proj ratio, and the QViT ansatz provides value beyond what ANO alone can achieve.

### Hypothesis 3: Similar Performance

Both circuits may hit the same performance ceiling, suggesting the bottleneck adapter is the limiting factor, not the quantum circuit architecture.

**If confirmed**: Focus future work on improving the bottleneck (larger working dim, nonlinear adapter, etc.) rather than circuit design.

## File Index

| File | Description |
|------|-------------|
| `models/QuantumLatentCFM_v12.py` | Main model: SD3.5 VAE + bottleneck + v9 circuit |
| `jobs/run_qlcfm_v12a.sh` | SLURM: 8q brick-layer SU(4) + QViT pyramid |
| `models/QuantumLatentCFM_v11.py` | Comparison: SD3.5 VAE + bottleneck + v10 circuit |
| `models/QuantumLatentCFM_v9.py` | Original v9 with custom VAE |
| `models/QuantumLatentCFM_v10.py` | Original v10 with custom VAE |
| `docs/QLCFM_V11_EXPLAINED.md` | v11 architecture documentation |
| `docs/QLCFM_V10_EXPLAINED.md` | v10 architecture documentation |

## References

- Lipman, Y. et al. (2023). "Flow Matching for Generative Modeling." *ICLR 2023*.
- Esser, P. et al. (2024). "Scaling Rectified Flow Transformers for High-Resolution Image Synthesis." *ICML 2024*. (SD3)
- Wiersema, R. et al. (2024). "Here comes the SU(N): multivariate quantum gates and gradients." *Quantum*, 8, 1275.
- Cherrat, E.A. et al. (2024). "Quantum Vision Transformers." *Quantum*, 8, 1265.
- Lin, S. et al. (2025). "Adaptive Non-local Observable on Quantum Neural Networks." *IEEE QCE 2025*.
- Chen, Y. et al. (2025). "Learning to Measure Quantum Neural Networks." *ICASSP 2025 Workshop*.
