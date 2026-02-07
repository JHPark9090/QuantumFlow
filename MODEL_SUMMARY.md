# ModularQFM and ModularQTS: Detailed Model Summary

## Table of Contents

1. [Overview](#1-overview)
2. [ModularQFM: Quantum Flow Matching for Image Classification](#2-modularqfm)
3. [ModularQTS: Quantum Time-Series with QSVT + ANO](#3-modularqts)
4. [Shared Components](#4-shared-components)
5. [Architecture Comparison](#5-architecture-comparison)
6. [Parameter Breakdown](#6-parameter-breakdown)
7. [Training Infrastructure](#7-training-infrastructure)
8. [CLI Reference](#8-cli-reference)

---

## 1. Overview

Both models live in `QuantumFlow/` and share a common measurement backend -- **Adaptive Non-Local Observables (ANO)** -- but differ in their quantum circuit core:

| Aspect | ModularQFM | ModularQTS |
|--------|-----------|-----------|
| **File** | `ModularQFM.py` (590 lines) | `ModularQTS.py` (668 lines) |
| **Domain** | Image classification | Spatio-temporal classification |
| **Input shape** | `(batch, flat_dim)` | `(batch, channels, timesteps)` |
| **Quantum core** | Encoding + swappable VQC | QSVT via LCU block encoding |
| **Measurement** | ANO (shared) | ANO (shared) |
| **Datasets** | MNIST, Fashion-MNIST, CIFAR-10 | PhysioNet EEG (extensible) |
| **Loss** | CrossEntropyLoss (multi-class) | BCEWithLogitsLoss (binary) / CrossEntropyLoss |
| **References** | Wiersema 2024, Lin 2025, Chen 2025 | Sim 2019, Chen 2025, Lin 2025 |

---

## 2. ModularQFM

**File:** `QuantumFlow/ModularQFM.py`

### 2.1 Purpose

Modular quantum image classifier with three independently swappable stages: encoding, variational processing, and measurement. Designed to allow ablation studies across different quantum encoding strategies and circuit ansatze.

### 2.2 Architecture

```
Flattened Image: (batch, input_dim)
    |
    v
[Stage 1: Encoding]
    |-- SU(4) exponential map (Wiersema 2024)
    |     direct:     pad/crop raw data to total_enc params
    |     projected:  Linear(input_dim, total_enc)
    |     gate:       qml.SpecialUnitary(15 params, wires=[q, q+1])
    |     layout:     brick-wall (even pairs, then odd pairs) x n_blocks
    |
    |-- Angle embedding (baseline)
    |     direct:     RY(x * pi) on each qubit, n_blocks re-upload layers
    |     projected:  Linear(input_dim, n_qubits) -> single RY layer
    |     entangling: CNOT brick-wall between re-upload layers
    |
    v
[Stage 2: VQC]
    |-- QCNN (default)
    |     conv:  staggered 2-qubit blocks (U3, IsingZZ/YY/XX, U3) -- 15 params each
    |     pool:  mid-circuit measure + conditional U3 on survivor qubit
    |     depth: vqc_depth layers, active wires halve each layer
    |
    |-- Hardware-efficient
    |     RY rotations + CNOT brick-wall, vqc_depth layers
    |
    |-- SU(4) re-upload
    |     re-encode data (same enc) + trainable SU(4) brick-wall per layer
    |
    |-- None (skip)
    |
    v
[Stage 3: ANO Measurement]
    |-- k_local > 0: qml.expval(qml.Hermitian(H_w, wires=group_w))
    |     H_w built from learnable A, B, D params
    |     n_obs = len(wire_groups)
    |
    |-- k_local = 0: qml.expval(qml.PauliZ(q)) for each qubit
    |     n_obs = n_qubits
    |
    v
Linear(n_obs, output_dim) -> class logits
```

### 2.3 Encoding Stage Details

**SU(4) exponential map** (`encoding_type="sun"`):
- Uses `qml.SpecialUnitary` -- a 2-qubit gate parameterized by 15 real numbers (generators of SU(4) Lie algebra)
- Brick-wall pattern: first all even pairs `(0,1), (2,3), ...`, then odd pairs `(1,2), (3,4), ...`
- Each block consumes `gates_per_block * 15` parameters
- `n_blocks` controls circuit depth (auto-computed from `input_dim` if 0)
- **Direct mode:** raw pixel data scaled by `encoding_scale` (default pi), zero-padded or truncated to fit
- **Projected mode:** `nn.Linear(input_dim, total_enc)` maps data to exact parameter count

**Angle embedding** (`encoding_type="angle"`):
- Single-qubit RY rotations, one angle per qubit per block
- CNOT entangling layers between re-upload blocks (even-odd brick-wall)
- Simpler baseline requiring fewer parameters per block (`n_qubits` vs `gates * 15`)

### 2.4 VQC Stage Details

**QCNN** (`vqc_type="qcnn"`):
- Convolution: staggered 2-qubit unitary blocks, each parameterized by 15 values
  - Gate sequence: `U3(q1) U3(q2) IsingZZ IsingYY IsingXX U3(q1) U3(q2)`
  - Applied in two passes: even-indexed pairs, then odd-indexed pairs
- Pooling: mid-circuit measurement on odd-indexed wires
  - Measured qubit is traced out; conditional `U3` applied to surviving neighbor
  - Active wire count halves each layer (`wires = wires[::2]`)
- Parameters: `conv_params` shape `(vqc_depth, n_qubits, 15)`, `pool_params` shape `(vqc_depth, n_qubits//2, 3)`
- Requires PennyLane's deferred measurement support (device created without explicit `wires=` argument)

**Hardware-efficient** (`vqc_type="hardware_efficient"`):
- Per layer: RY rotation on each qubit + CNOT brick-wall (even, then odd pairs)
- Parameters: `var_params` shape `(vqc_depth, n_qubits)`

**SU(4) re-upload** (`vqc_type="su4_reupload"`):
- Per layer: re-encode data using same SU(4) brick-wall as encoding, then apply trainable SU(4) brick-wall
- Data re-encoding uses the same `enc` parameters (no new data parameters)
- Trainable parameters: `reupload_params` shape `(vqc_depth, gates_per_block, 15)`

### 2.5 QNode Configuration

- **Device:** `qml.device("default.qubit")` -- no explicit wire count (supports mid-circuit measurement ancillas)
- **Differentiation:** `diff_method="best"` (PennyLane auto-selects backprop for simulation)
- **Interface:** PyTorch
- **Batching:** PennyLane parameter broadcasting via `enc[..., idx]` slicing

---

## 3. ModularQTS

**File:** `QuantumFlow/ModularQTS.py`

### 3.1 Purpose

Quantum time-series classifier that integrates QSVT (Quantum Singular Value Transformation) via LCU (Linear Combination of Unitaries) block encoding with ANO measurement. Designed for spatio-temporal signals where the input has explicit channel and temporal dimensions (e.g., EEG with 64 channels x T timesteps).

### 3.2 Architecture

```
Spatio-Temporal Input: (batch, n_channels, n_timesteps)
    |
    v  permute -> (batch, T, C)
    |
    v  Dropout(p)
    |
    v  Linear(n_channels, n_rots)   [feature_projection]
    |
    v  Sigmoid                       [rotation scaling to [0, 1]]
    |
    v  ts_params: (batch, T, n_rots)
    |
    v
[Single QNode: QSVT via LCU]
    |
    |  PCPhase(phi_0, dim=2^n_qubits, wires=anc+main)
    |
    |  for k in range(degree):
    |      PREPARE(prep_params)          -- learnable V on ancilla
    |      if k even:
    |          SELECT([U_0(x_0), ..., U_T(x_T)])
    |      else:
    |          SELECT_dag([U_0(x_0), ..., U_T(x_T)])
    |      PREPARE_dag                   -- V_dag on ancilla
    |      PCPhase(phi_{k+1})            -- signal processing angle
    |
    |  QFF: sim14(qff_params) on main register
    |
    v
[ANO Measurement on main register]
    |-- k_local > 0: qml.expval(qml.Hermitian(H_w, wires=group_w))
    |-- k_local = 0: qml.expval(qml.PauliZ(q))
    |
    v  (batch, n_obs) expectation values
    |
    v  Linear(n_obs, output_dim)
    |
    v  (batch, output_dim) class logits
```

### 3.3 Qubit Register Layout

The circuit uses two disjoint qubit registers:

| Register | Wires | Count | Purpose |
|----------|-------|-------|---------|
| **Main** | `[0, ..., n_qubits-1]` | `n_qubits` | Data processing, measurement |
| **Ancilla** | `[n_qubits, ..., n_qubits+n_ancilla-1]` | `ceil(log2(n_timesteps))` | SELECT control for LCU |

**Total wires** = `n_qubits + n_ancilla`

Example: 6 qubits, 26 timesteps -> n_ancilla = ceil(log2(26)) = 5 -> 11 total wires

### 3.4 QSVT via LCU Block Encoding

The circuit implements QSVT by alternating **signal processing rotations** (PCPhase) with **LCU block encodings** (PREPARE-SELECT-PREPARE-dag).

**PCPhase** (`qml.PCPhase`):
- Projects onto the ancilla=|0> subspace
- `dim = 2^n_qubits` selects the first 2^n_qubits computational basis states
- Applied to `anc_wires + main_wires` (ancilla first, then main)
- `degree + 1` learnable signal processing angles

**PREPARE** (learnable unitary V on ancilla):
- `n_prep_layers = n_ancilla` layers of:
  - RY(theta) and RZ(phi) on each ancilla qubit
  - CNOT chain connecting adjacent ancilla qubits
- Parameters: `prepare_params` shape `(n_prep_layers, n_ancilla, 2)`
- Encodes the LCU coefficients -- determines how the `n_timesteps` unitaries are combined

**SELECT** (`qml.Select`):
- Controlled dispatch: ancilla state |j> activates unitary U_j on main register
- Each U_j is a sim14 circuit parameterized by data from timestep j: `ts_params[..., j, :]`
- Padded to `2^n_ancilla` operators with Identity for unused indices
- On even iterations: SELECT applied directly
- On odd iterations: SELECT applied as adjoint (conjugate transpose)

**sim14 circuit** (Sim et al., 2019):
- Gate sequence per layer: `RY -> CRX(ring) -> RY -> CRX(counter-ring)`
- Ring: qubit i controls qubit (i+1) mod n, traversed in reverse order
- Counter-ring: qubit i controls qubit (i-1) mod n, special ordering
- Parameters per layer: `4 * n_qubits`
- Total parameters per timestep: `4 * n_qubits * n_ansatz_layers`

**QFF** (Quantum Feature Finalization):
- Single sim14 layer applied to main register after the QSVT loop
- Data-independent trainable parameters: `qff_params` shape `(4 * n_qubits,)`
- Adds expressibility to the final quantum state before measurement

### 3.5 Feature Projection

The classical pre-processing converts raw channel data into rotation angles:

```
(batch, C, T) -> permute -> (batch, T, C) -> Dropout -> Linear(C, n_rots) -> Sigmoid
```

- `n_rots = 4 * n_qubits * n_ansatz_layers` (matches sim14 parameter count per timestep)
- Sigmoid squashes to [0, 1], ensuring bounded rotation angles
- Each timestep gets its own set of rotation angles derived from its channel features

### 3.6 QNode Configuration

- **Device:** `qml.device("default.qubit", wires=total_wires)` -- explicit wire count
- **Differentiation:** `diff_method="backprop"` -- explicit backpropagation through statevector
- **Interface:** PyTorch
- **Batching:** PennyLane parameter broadcasting via `ts_params[..., t, param_idx]` slicing

---

## 4. Shared Components

### 4.1 Adaptive Non-Local Observables (ANO)

Both models share the same ANO implementation (Chen et al., 2025; Lin et al., 2025):

**Wire grouping** (`get_wire_groups`):
- `sliding` (default): overlapping windows of size `k_local`
  - Example (6 qubits, k=2): `[0,1], [1,2], [2,3], [3,4], [4,5]` -> 5 groups
- `pairwise`: all combinations of `k_local` qubits
  - Example (6 qubits, k=2): `[0,1], [0,2], ..., [4,5]` -> 15 groups
- `k_local=0`: falls back to single-qubit PauliZ on each qubit

**Hermitian construction** (`create_Hermitian`):
- Builds an `N x N` Hermitian matrix (where `N = 2^k_local`) from three learnable parameter vectors:
  - `A[w]`: real parts of lower-triangular off-diagonal entries, shape `(N*(N-1)/2,)`
  - `B[w]`: imaginary parts of lower-triangular off-diagonal entries, shape `(N*(N-1)/2,)`
  - `D[w]`: diagonal entries, shape `(N,)`
- Construction: `H = L + L^dagger` where L is the lower-triangular matrix with `L[i,j] = A[k] + i*B[k]`
- The resulting matrix is guaranteed Hermitian by construction
- Initialization: `Normal(0, 2.0)` for all three parameter types

**Parameter counts for ANO** (per group, k_local=2):
- K = 2^2 = 4 (matrix dimension)
- n_off = 4*3/2 = 6 off-diagonal elements
- Per group: 6 (A) + 6 (B) + 4 (D) = 16 parameters
- Total: 16 * n_obs

**Eigenvalue diagnostic** (`get_eigenvalue_range`):
- Computes `torch.linalg.eigvalsh` on each Hermitian matrix
- Returns `(min_eigenvalue, max_eigenvalue)` across all groups
- Logged per epoch for observable health monitoring

### 4.2 Dual Optimizer Pattern

Both models partition parameters into two groups with separate Adam optimizers:

| Group | Parameters | Learning Rate | Rationale |
|-------|-----------|---------------|-----------|
| **Circuit** | Encoding/projection, VQC/QSVT, QFF, head | `--lr` (default 1e-3) | Standard QML training rate |
| **Observable (ANO)** | A, B, D ParameterLists | `--lr-H` (default 1e-1) | 100x higher per Chen 2025 |

The 100x learning rate ratio for observable parameters is motivated by the observation that Hermitian observable parameters converge much slower than circuit parameters at equal learning rates. The higher rate allows the measurement basis to adapt quickly while circuit parameters evolve at a standard pace.

Parameter detection: `name.startswith(("A.", "B.", "D."))` identifies ANO parameters.

### 4.3 Reproducibility

```python
def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    qml.numpy.random.seed(seed)
```

Both models use `seed=2025` by default.

---

## 5. Architecture Comparison

### 5.1 Data Flow

```
ModularQFM:
  (batch, flat) -> [Encoding] -> [VQC] -> [ANO] -> Linear -> logits
  Single-shot: one encoding pass, one VQC pass, one measurement

ModularQTS:
  (batch, C, T) -> permute -> Linear+Sigmoid -> [QSVT loop x degree] -> [QFF] -> [ANO] -> Linear -> logits
  Iterative: degree iterations of LCU block encoding with signal processing
```

### 5.2 Quantum Circuit Depth

**ModularQFM** depth depends on configuration:
- SU(4) encoding: `n_blocks * 2` layers of SU(4) gates (even + odd)
- QCNN: `vqc_depth` convolution-pooling layers (each with 2 passes of 2-qubit blocks)
- Total: encoding depth + VQC depth

**ModularQTS** depth scales with:
- `degree` QSVT iterations, each containing:
  - PREPARE: `n_ancilla` layers of RY/RZ + CNOT chain
  - SELECT: `n_timesteps` sim14 circuits (each `n_ansatz_layers` deep)
  - PREPARE-dag
  - PCPhase
- Plus one final QFF sim14 layer
- Significantly deeper than ModularQFM for equivalent qubit counts

### 5.3 Temporal vs Spatial Processing

| Feature | ModularQFM | ModularQTS |
|---------|-----------|-----------|
| **Input structure** | Flattened (spatial info lost) | Preserves channel+time axes |
| **Temporal modeling** | None (static image) | LCU SELECT iterates over timesteps |
| **Channel mixing** | Handled by encoding | Linear projection per timestep |
| **Sequence length** | N/A | Determines ancilla count |

### 5.4 Measurement Stage

Both models can use the same ANO measurement, but the underlying quantum states differ:

- **ModularQFM:** ANO measures the state after encoding + VQC processing. The VQC (especially QCNN with pooling) may reduce the effective qubit count measured.
- **ModularQTS:** ANO measures the `main_wires` only (not ancilla). The QSVT loop has processed temporal information via the ancilla-controlled SELECT, and the QFF sim14 layer finalizes the state on main register before measurement.

---

## 6. Parameter Breakdown

### 6.1 ModularQFM (typical: 10 qubits, SU(4) direct, QCNN depth 2, k_local=2)

| Component | Shape | Count |
|-----------|-------|-------|
| SU(4) encoding (direct) | -- | 0 (data-driven) |
| conv_params | (2, 10, 15) | 300 |
| pool_params | (2, 5, 3) | 30 |
| A (9 groups) | 9 x (6,) | 54 |
| B (9 groups) | 9 x (6,) | 54 |
| D (9 groups) | 9 x (4,) | 36 |
| head | Linear(9, 10) | 100 |
| **Total** | | **574** |

### 6.2 ModularQTS (typical: 4 qubits, degree 1, 1 layer, k_local=2, 26 timesteps)

| Component | Shape | Count |
|-----------|-------|-------|
| feature_projection | Linear(64, 16) | 1040 |
| prepare_params | (5, 5, 2) | 50 |
| signal_angles | (2,) | 2 |
| qff_params | (16,) | 16 |
| A (3 groups) | 3 x (6,) | 18 |
| B (3 groups) | 3 x (6,) | 18 |
| D (3 groups) | 3 x (4,) | 12 |
| head | Linear(3, 1) | 4 |
| **Total** | | **1160** |

Note: The feature projection dominates ModularQTS's parameter count because it maps 64 EEG channels to 16 rotation angles per timestep.

---

## 7. Training Infrastructure

### 7.1 Loss Functions

| Model | Binary | Multi-class |
|-------|--------|------------|
| ModularQFM | -- | CrossEntropyLoss |
| ModularQTS | BCEWithLogitsLoss (`output_dim=1`) | CrossEntropyLoss (`output_dim=num_classes`) |

ModularQFM always uses CrossEntropyLoss (designed for multi-class image classification). ModularQTS auto-selects based on `--num-classes`: binary (<=2) uses BCEWithLogitsLoss with sigmoid predictions; multi-class uses CrossEntropyLoss with argmax predictions.

### 7.2 Metrics

| Metric | ModularQFM | ModularQTS |
|--------|-----------|-----------|
| Loss | Train, Val | Train, Val |
| Accuracy | Train, Val, Test | Train, Val, Test |
| AUC | -- | Train, Val, Test (binary only) |
| Eigenvalue range | Train (per epoch) | Train (per epoch) |
| Time per epoch | Yes | Yes |

### 7.3 Checkpointing

Both models save per-epoch checkpoints containing:
- Model state dict
- Circuit optimizer state dict
- Observable optimizer state dict (if ANO enabled)
- Best validation accuracy and epoch
- Full training history

| File | ModularQFM | ModularQTS |
|------|-----------|-----------|
| Checkpoint | `ckpt_mqfm_{job_id}.pt` | `ckpt_mqts_{job_id}.pt` |
| CSV log | `log_mqfm_{job_id}.csv` | `log_mqts_{job_id}.csv` |
| Best weights | `weights_mqfm_{job_id}.pt` | `weights_mqts_{job_id}.pt` |

Resume with `--resume` flag; requires matching `--job-id`.

### 7.4 Early Stopping

- **ModularQFM:** No early stopping (runs all `--epochs`)
- **ModularQTS:** Configurable early stopping via `--patience` (default 20). Stops when validation accuracy shows no improvement for `patience` consecutive epochs. Set `--patience=0` to disable.

### 7.5 Test Evaluation

Both models reload the best validation epoch weights before final test evaluation.

---

## 8. CLI Reference

### 8.1 ModularQFM

```
python ModularQFM.py [options]

Encoding:
  --n-qubits INT          Number of qubits (default: 10)
  --n-blocks INT          Encoding blocks, 0=auto (default: 0)
  --encoding-type STR     sun | angle (default: sun)
  --encoding-mode STR     direct | projected (default: direct)
  --encoding-scale FLOAT  Scale for direct mode (default: pi)

VQC:
  --vqc-type STR          qcnn | hardware_efficient | su4_reupload | none (default: qcnn)
  --vqc-depth INT         VQC layers (default: 2)

Observable:
  --k-local INT           ANO locality, 0=PauliZ (default: 2)
  --obs-scheme STR        sliding | pairwise (default: sliding)

Training:
  --lr FLOAT              Circuit LR (default: 1e-3)
  --lr-H FLOAT            Observable LR (default: 1e-1)
  --batch-size INT        Batch size (default: 32)
  --epochs INT            Training epochs (default: 30)
  --seed INT              Random seed (default: 2025)

Data:
  --dataset STR           mnist | fashion | cifar10 (default: mnist)
  --n-train INT           Training samples (default: 1000)
  --n-valtest INT         Val+test samples (default: 500)
  --num-classes INT       Output classes (default: 10)

I/O:
  --job-id STR            Job identifier (default: mqfm_001)
  --base-path STR         Output directory (default: .)
  --resume                Resume from checkpoint
```

### 8.2 ModularQTS

```
python ModularQTS.py [options]

Model:
  --n-qubits INT          Main register qubits (default: 6)
  --n-layers INT          sim14 layers per timestep (default: 2)
  --degree INT            QSVT polynomial degree (default: 2)
  --dropout FLOAT         Dropout rate (default: 0.1)

Observable:
  --k-local INT           ANO locality, 0=PauliZ (default: 2)
  --obs-scheme STR        sliding | pairwise (default: sliding)

Training:
  --n-epochs INT          Training epochs (default: 50)
  --batch-size INT        Batch size (default: 32)
  --lr FLOAT              Circuit LR (default: 1e-3)
  --lr-H FLOAT            Observable LR (default: 1e-1)
  --wd FLOAT              Weight decay (default: 0.0)
  --patience INT          Early stopping, 0=disabled (default: 20)

Data:
  --dataset STR           Dataset name (default: physionet)
  --num-classes INT       Number of classes (default: 2)
  --sampling-freq INT     PhysioNet sampling freq (default: 32)
  --sample-size INT       PhysioNet subject count (default: 50)

I/O:
  --seed INT              Random seed (default: 2025)
  --job-id STR            Job identifier (default: mqts_001)
  --base-path STR         Output directory (default: .)
  --resume                Resume from checkpoint
```

### 8.3 Example Commands

**ModularQFM -- MNIST with SU(4) encoding + QCNN:**
```bash
python ModularQFM.py --dataset=mnist --n-qubits=10 --encoding-type=sun \
    --vqc-type=qcnn --k-local=2 --epochs=30
```

**ModularQFM -- Fashion-MNIST with angle encoding + hardware-efficient VQC:**
```bash
python ModularQFM.py --dataset=fashion --n-qubits=8 --encoding-type=angle \
    --encoding-mode=projected --vqc-type=hardware_efficient --vqc-depth=3
```

**ModularQTS -- PhysioNet EEG (full):**
```bash
python ModularQTS.py --dataset=physionet --n-qubits=6 --degree=2 \
    --n-layers=2 --k-local=2 --n-epochs=50 --batch-size=32 \
    --sampling-freq=32 --sample-size=50
```

**ModularQTS -- Quick smoke test:**
```bash
python ModularQTS.py --dataset=physionet --n-qubits=4 --degree=1 \
    --n-layers=1 --k-local=2 --sampling-freq=8 --sample-size=10 \
    --n-epochs=3 --batch-size=16
```

---

## Appendix: Key References

- **Sim et al. (2019):** sim14 circuit ansatz -- expressibility and entangling capability benchmarks for parameterized quantum circuits
- **Wiersema et al. (2024):** SU(4) exponential map encoding -- data encoding via Lie algebra generators of SU(4)
- **Lin et al. (2025):** Hermitian matrix parameterization -- constructing learnable N x N Hermitians from real-valued parameters
- **Chen et al. (2025):** Adaptive Non-Local Observables (ANO) and dual-rate optimization -- 100x learning rate ratio for observable vs circuit parameters
