# Quantum Direct CFM — Experiment Configurations

Submitted: 2026-03-09

## Job IDs

| Job | SLURM ID | Script |
|-----|----------|--------|
| v1 CIFAR-10 | 49828854 | `jobs/run_dcfm_v1_cifar.sh` |
| v1 COCO | 49828855 | `jobs/run_dcfm_v1_coco.sh` |
| v2 CIFAR-10 | 49828856 | `jobs/run_dcfm_v2_cifar.sh` |
| v2 COCO | 49828857 | `jobs/run_dcfm_v2_coco.sh` |
| v3 CIFAR-10 | 49828858 | `jobs/run_dcfm_v3_cifar.sh` |
| v3 COCO | 49828859 | `jobs/run_dcfm_v3_coco.sh` |

---

## Model Files

| Version | File | Key Feature |
|---------|------|-------------|
| v1 | `models/QuantumDirectCFM.py` | Brick-layer SU(2^k) + QViT + pairwise ANO |
| v2 | `models/QuantumDirectCFM_v2.py` | Single SU(2^n) + pairwise ANO (no QViT) |
| v3 | `models/QuantumDirectCFM_v3.py` | Single SU(2^n) + global ANO (no QViT) |

---

## 1. Dataset Configuration

| | CIFAR-10 | COCO |
|---|---|---|
| **Image size** | 32 x 32 x 3 | 128 x 128 x 3 |
| **Raw pixel dims** | 3,072 | 49,152 |
| **Training samples** | 10,000 | 80,000 |
| **Val + Test samples** | 2,000 | 10,000 |
| **Batch size** | 64 | 32 |
| **Epochs** | 200 | 200 |

---

## 2. Quantum Circuit Configuration

### 2.1. Encoding

| | v1 | v2 | v3 |
|---|---|---|---|
| **Encoding type** | Brick-layer SU(4) | Single SU(64) | Single SU(64) |
| **Qubits** | 8 | 6 | 6 |
| **SU group size** | k=2 (SU(4), 15 generators/gate) | k=6 (SU(64), 4095 generators) | k=6 (SU(64), 4095 generators) |
| **Number of SU gates** | 7 (4 even + 3 odd) | 1 (single gate) | 1 (single gate) |
| **Total encoding params** | 105 | 4,095 | 4,095 |
| **Entanglement coverage** | 7/28 pairs (25%) | 15/15 pairs (100%) | 15/15 pairs (100%) |

### 2.2. Ansatz (VQC)

| | v1 | v2 | v3 |
|---|---|---|---|
| **VQC type** | QViT | None | None |
| **QViT circuit** | Pyramid | -- | -- |
| **VQC depth** | 2 | -- | -- |
| **QViT gate pairs** | 28 (all C(8,2) pairs) | -- | -- |
| **Params per gate** | 12 (U3 + IsingXXYYZZ + U3 + IsingXXYYZZ) | -- | -- |
| **Total VQC params** | 672 (2 depths x 28 pairs x 12) | 0 | 0 |
| **Purpose** | Fills missing entanglement from brick-layer | Not needed (100% from encoding) | Not needed (100% from encoding) |

### 2.3. Measurement (ANO)

| | v1 | v2 | v3 |
|---|---|---|---|
| **ANO locality** | k=2 (pairwise) | k=2 (pairwise) | **k=6 (global)** |
| **Observable scheme** | All C(8,2) pairs | All C(6,2) pairs | 15 independent on all qubits |
| **Number of observables** | 28 | 15 | 15 |
| **Hermitian matrix size** | 4 x 4 | 4 x 4 | **64 x 64** |
| **Params per observable** | 16 | 16 | **4,096** |
| **Total ANO params** | 448 | 240 | **61,440** |
| **Correlations captured** | 2-body only | 2-body only | **All (1 through 6-body)** |

### 2.4. Quantum Pipeline Summary

| | v1 | v2 | v3 |
|---|---|---|---|
| **Pipeline** | Encoding + QViT + ANO | Encoding + ANO | Encoding + ANO |
| **Total quantum params** | 1,120 (105 enc + 672 VQC + 448 ANO) | 240 (ANO only) | **61,440** (ANO only) |

Note: enc_proj parameters (classical Linear layers that project features into encoding params) are not counted as "quantum params" in this table. They are listed separately below.

---

## 3. Classical Components

### 3.1. ConvEncoder

| | v1 CIFAR | v1 COCO | v2 CIFAR | v2 COCO | v3 CIFAR | v3 COCO |
|---|---|---|---|---|---|---|
| **enc_channels** | [32,64,128] | [32,64,128,256,256] | [32,64] (auto) | [32,64,128,256,256] (auto) | [32,64] (auto) | [32,64,128,256,256] (auto) |
| **Conv layers** | 3 | 5 | 2 | 5 | 2 | 5 |
| **Final spatial** | 4 x 4 | 4 x 4 | 8 x 8 | 4 x 4 | 8 x 8 | 4 x 4 |
| **d_flat** | 2,048 | 4,096 | 4,096 | 4,096 | 4,096 | 4,096 |

### 3.2. enc_proj (Classical projection into encoding parameters)

| | v1 CIFAR | v1 COCO | v2 CIFAR | v2 COCO | v3 CIFAR | v3 COCO |
|---|---|---|---|---|---|---|
| **Input dim** | 2,048 | 4,096 | 4,096 | 4,096 | 4,096 | 4,096 |
| **Output dim** | 105 | 105 | 4,095 | 4,095 | 4,095 | 4,095 |
| **Ratio** | 19.5:1 | 39:1 | **1.00:1** | **1.00:1** | **1.00:1** | **1.00:1** |
| **enc_proj params** | 551,529 | 1,075,817 | 16,773,119 | 16,773,119 | 16,773,119 | 16,773,119 |

### 3.3. Time Embedding

| | All jobs |
|---|---|
| **Sinusoidal dim** | 256 |
| **Conditioning** | Additive (time_mlp output added to image features) |
| **time_mlp** | Linear(256, d_flat) -> SiLU -> Linear(d_flat, d_flat) |

| | v1 CIFAR | v1 COCO | v2/v3 CIFAR | v2/v3 COCO |
|---|---|---|---|---|
| **time_mlp params** | 4,722,688 | 17,833,984 | 17,833,984 | 17,833,984 |

### 3.4. vel_head (Observable output to flat features)

| | v1 CIFAR | v1 COCO | v2/v3 CIFAR | v2/v3 COCO |
|---|---|---|---|---|
| **Input dim** | 28 | 28 | 15 | 15 |
| **Output dim** | 2,048 | 4,096 | 4,096 | 4,096 |
| **vel_head params** | 533,760 | 1,060,096 | 1,056,768 | 1,056,768 |

---

## 4. Total Parameter Counts

| Component | v1 CIFAR | v1 COCO | v2 CIFAR | v2 COCO | v3 CIFAR | v3 COCO |
|-----------|----------|---------|----------|---------|----------|---------|
| ConvEncoder + ConvDecoder | 331,075 | 3,477,699 | 68,739 | 3,477,699 | 68,739 | 3,477,699 |
| time_mlp | 4,722,688 | 17,833,984 | 17,833,984 | 17,833,984 | 17,833,984 | 17,833,984 |
| enc_proj | 551,529 | 1,075,817 | 16,773,119 | 16,773,119 | 16,773,119 | 16,773,119 |
| VQC (QViT) | 672 | 672 | 0 | 0 | 0 | 0 |
| ANO (A+B+D) | 448 | 448 | 240 | 240 | 61,440 | 61,440 |
| vel_head | 533,760 | 1,060,096 | 1,056,768 | 1,056,768 | 1,056,768 | 1,056,768 |
| **Total** | **6,140,172** | **23,448,716** | **35,732,850** | **39,141,810** | **35,794,050** | **39,203,010** |
| **Quantum only (VQC + ANO)** | **1,120** | **1,120** | **240** | **240** | **61,440** | **61,440** |
| **Quantum %** | 0.018% | 0.005% | 0.001% | 0.001% | 0.172% | 0.157% |

---

## 5. Training Configuration

| Parameter | All jobs |
|-----------|---------|
| **Main optimizer** | Adam, lr=1e-3 |
| **ANO optimizer** | Adam, lr=1e-1 (100x main) |
| **LR schedule** | CosineAnnealingLR (T_max=epochs) |
| **Timestep sampling** | Logit-normal (std=1.0) |
| **ODE solver** | Midpoint (2nd-order) |
| **ODE steps** | 50 |
| **EMA decay** | 0.999 |
| **Seed** | 2025 |
| **Evaluation** | FID + IS (compute-metrics enabled) |

---

## 6. Architecture Comparison Diagram

### v1: Brick-Layer SU(4) + Pyramid QViT + Pairwise ANO

```
Image (3,H,W) + time t
  -> ConvEncoder -> d_flat
  -> + time_mlp(t)
  -> enc_proj -> 105 params                      [19.5:1 or 39:1 compression]
  -> 7x SU(4) gates (brick-layer)                [25% entanglement]
  -> Pyramid QViT (28 gate pairs, depth=2)       [fills remaining 75%]
  -> 28x pairwise Hermitian (4x4)                [2-body correlations]
  -> vel_head -> d_flat
  -> ConvDecoder -> velocity (3,H,W)
```

### v2: Single SU(64) + Pairwise ANO

```
Image (3,H,W) + time t
  -> ConvEncoder (auto) -> d_flat=4096
  -> + time_mlp(t)
  -> enc_proj -> 4095 params                     [1.00:1 ratio]
  -> 1x SU(64) gate (single, all qubits)         [100% entanglement]
  -> 15x pairwise Hermitian (4x4)                [2-body correlations]
  -> vel_head -> d_flat
  -> ConvDecoder -> velocity (3,H,W)
```

### v3: Single SU(64) + Global ANO

```
Image (3,H,W) + time t
  -> ConvEncoder (auto) -> d_flat=4096
  -> + time_mlp(t)
  -> enc_proj -> 4095 params                     [1.00:1 ratio]
  -> 1x SU(64) gate (single, all qubits)         [100% entanglement]
  -> 15x global Hermitian (64x64)                [ALL correlations]
  -> vel_head -> d_flat
  -> ConvDecoder -> velocity (3,H,W)
```

---

## 7. Key Differences Summary

| Aspect | v1 | v2 | v3 |
|--------|----|----|-----|
| Encoding | Brick-layer SU(4) | Single SU(64) | Single SU(64) |
| Entanglement | 25% (needs QViT) | 100% | 100% |
| Ansatz | Pyramid QViT | None | None |
| ANO locality | k=2 (pairwise) | k=2 (pairwise) | k=6 (global) |
| ANO params | 448 | 240 | 61,440 |
| enc_proj ratio | 19.5:1 / 39:1 | 1.00:1 | 1.00:1 |
| Quantum pipeline | Enc + QViT + ANO | Enc + ANO | Enc + ANO |
| Correlations measured | 2-body | 2-body | All (1 through 6-body) |
| Scientific role | Ablation (brick-layer baseline) | Principled quantum (clean architecture) | Theoretically optimal (full encoding + full measurement) |

---

## 8. What We Expect to Learn

1. **v1 vs v2**: Does single-gate SU(64) encoding (100% entanglement, 1:1 ratio) outperform brick-layer SU(4) + QViT (25% + ansatz, 19-39:1 compression)?

2. **v2 vs v3**: Does global ANO (capturing all n-body correlations) improve over pairwise ANO (only 2-body correlations), given that SU(64) encoding creates full entanglement?

3. **CIFAR vs COCO**: How do all three architectures scale from simple (32x32) to complex (128x128) images in pixel-space flow matching?

4. **Quantum parameter efficiency**: v3 has 61K quantum params (0.16%) vs v1's 1.1K (0.02%) — does the additional ANO expressiveness justify the extra parameters?
