# QuantumFlow

A quantum machine learning framework combining **Quantum Singular Value Transformation (QSVT)**, **SU(N) flow encoding**, and **Adaptive Non-Local Observables (ANO)** for classification and generative modeling.

## Key Innovations

1. **QSVT via LCU** — Hardware-compatible block encoding using PREPARE/SELECT/PREPARE† with PCPhase signal processing. No statevector extraction or classical O(2^n) operations.
2. **SU(N) Flow Encoding** (Wiersema et al., 2024) — Geodesic data encoding on the unitary manifold via exponential mapping.
3. **Adaptive Non-Local Observables** (Lin et al., 2025; Chen et al., 2025) — Learnable k-local Hermitian measurements that shift circuit complexity to the measurement side.
4. **Dual Optimizer** (Chen et al., 2025) — Circuit parameters at lr, observable parameters at 100x lr for accelerated eigenvalue expansion.

## Models

### Discriminative Models

| Model | File | Task | Architecture |
|-------|------|------|--------------|
| **ModularQFM** | `models/ModularQFM.py` | Image classification | SU(4) encoding + QCNN/HWE + ANO |
| **ModularQTS** | `models/ModularQTS.py` | Spatio-temporal classification | QSVT via LCU + QFF + ANO |
| **ModularQTS_NLP** | `models/ModularQTS_NLP.py` | NLP (GLUE benchmarks) | Token embedding + QSVT via LCU + ANO |
| **QuantumVisionTransformer** | `models/QuantumVisionTransformer.py` | Image classification | RBS-based orthogonal QViT (Cherrat et al., 2024) |

### Generative Models

| Model | File | Task | Architecture |
|-------|------|------|--------------|
| **QuantumLatentCFM** | `models/QuantumLatentCFM.py` | Image generation | ConvVAE + Quantum Velocity Field (SU(4)+QCNN/QViT/HWE+ANO) |
| **QuantumLatentCFM_Text** | `models/QuantumLatentCFM_Text.py` | Text generation | Mamba TextVAE + QSVTVelocityField |

### QSVT Transformers

| Model | File | Description |
|-------|------|-------------|
| **QTSTransformer** | `models/QTSTransformer.py` | Original QSVT via LCU with fixed PauliX/Y/Z measurement |
| **QTSTransformer v1.5** | `models/QTSTransformer_v1_5.py` | + Positional Encoding, 2pi angle scaling, SELECT caching |
| **QTSTransformer v2** | `models/QTSTransformer_v2.py` | + All v1.5 changes + temporal chunking |

### Legacy

| Model | File | Description |
|-------|------|-------------|
| **QuantumFlowMatching** | `models/QuantumFlowMatching.py` | Original QFM prototype (superseded by ModularQFM) |

## Architecture Overview

### QSVT via LCU (ModularQTS / ModularQTS_NLP)

```
Input (B, C, T)
  -> permute -> + Sinusoidal PE
  -> Linear + Sigmoid * 2pi -> (B, T, n_rots)
  -> Single QNode:
      PCPhase(phi_0)
      select_ops = build_select_ops()       # cached once
      for k in range(degree):
          PREPARE (learnable V on ancilla)
          SELECT(select_ops)                 # data-dependent sim14 unitaries
          PREPARE†
          PCPhase(phi_{k+1})
      QFF sim14 on main register
      -> ANO: learnable k-local Hermitian observables
  -> Linear -> output
```

### Quantum Latent CFM (QuantumLatentCFM / QuantumLatentCFM_Text)

```
Phase 1 -- VAE pretraining:
  x -> Encoder -> (mu, logvar) -> z -> Decoder -> x_hat
  Loss = Recon + beta * KL

Phase 2 -- Quantum CFM:
  z_0 ~ N(0,I),  z_1 = Encoder(x).mu  (frozen)
  z_t = (1-t)*z_0 + t*z_1              (OT interpolation)
  v_theta(z_t, t) = QuantumVelocityField(z_t, t)
  Loss = MSE(v_theta, z_1 - z_0)

Generation:
  z_0 ~ N(0,I) -> Euler ODE (t: 0->1) -> z_1 -> Decoder -> output
```

## Installation

### Dependencies

- Python 3.11+
- PyTorch 2.0+
- PennyLane 0.42+
- NumPy, SciPy, Scikit-learn, Matplotlib, tqdm
- HuggingFace `datasets` and `transformers` (for NLP models)

### Conda Environment (Perlmutter)

```bash
conda activate /pscratch/sd/j/junghoon/conda-envs/qml_eeg
```

**Important:** Set `PYTHONNOUSERSITE=1` in SLURM scripts to avoid `~/.local` package conflicts.

## Quick Start

### Image Classification (ModularQFM)

```bash
python models/ModularQFM.py --dataset=mnist --n-qubits=10 \
    --encoding-type=sun --vqc-type=qcnn --k-local=2 --epochs=30
```

### Spatio-Temporal Classification (ModularQTS)

```bash
python models/ModularQTS.py --dataset=physionet --n-qubits=6 \
    --degree=2 --n-layers=2 --k-local=2 --n-epochs=50 --batch-size=32
```

### NLP Classification (ModularQTS_NLP)

```bash
python models/ModularQTS_NLP.py --task=sst2 --n-qubits=4 --degree=2 \
    --n-layers=2 --k-local=2 --embed-dim=32 --chunk-size=16 \
    --max-length=64 --n-epochs=30 --batch-size=8
```

Supported GLUE tasks: `cola`, `sst2`, `mrpc`, `qqp`, `stsb`, `mnli`, `qnli`, `rte`, `wnli`

### Image Generation (QuantumLatentCFM)

```bash
# Phase 1: Train ConvVAE
python models/QuantumLatentCFM.py --phase=1 --dataset=cifar10 --epochs=200

# Phase 2: Train quantum velocity field
python models/QuantumLatentCFM.py --phase=2 --dataset=cifar10 --n-qubits=8 \
    --vqc-type=qcnn --k-local=2 --epochs=300 --vae-ckpt=checkpoints/weights_vae_*.pt

# Generate samples
python models/QuantumLatentCFM.py --phase=generate --n-samples=64 \
    --vae-ckpt=... --cfm-ckpt=...
```

### Text Generation (QuantumLatentCFM_Text)

```bash
# Both phases + generation (text8, default)
python models/QuantumLatentCFM_Text.py --phase=both --dataset=text8 \
    --n-qubits=8 --degree=3 --n-layers=2 --k-local=2 \
    --latent-dim=128 --seq-len=256 --epochs=200 --batch-size=64

# Penn Treebank
python models/QuantumLatentCFM_Text.py --phase=both --dataset=ptb \
    --n-qubits=8 --degree=3 --n-layers=2 --k-local=2 \
    --latent-dim=128 --seq-len=256 --epochs=200 --batch-size=64

# WikiText-103
python models/QuantumLatentCFM_Text.py --phase=both --dataset=wikitext-103 \
    --n-qubits=8 --degree=3 --n-layers=2 --k-local=2 \
    --latent-dim=128 --seq-len=256 --epochs=200 --batch-size=64
```

Supported text datasets: `text8`, `wikitext-2`, `wikitext-103`, `ptb`

## Project Structure

```
QuantumFlow/
├── models/
│   ├── ModularQFM.py                 # Image classification: SU(4) + VQC + ANO
│   ├── ModularQTS.py                 # Spatio-temporal: QSVT-LCU + ANO
│   ├── ModularQTS_NLP.py             # NLP: QSVT-LCU + ANO for GLUE
│   ├── QuantumLatentCFM.py           # Image generation: ConvVAE + Quantum CFM
│   ├── QuantumLatentCFM_Text.py      # Text generation: Mamba VAE + QSVT CFM
│   ├── QuantumVisionTransformer.py   # RBS-based orthogonal QViT
│   ├── QTSTransformer.py             # Original QSVT transformer (v1)
│   ├── QTSTransformer_v1_5.py        # + PE, 2pi scaling, SELECT caching (v1.5)
│   ├── QTSTransformer_v2.py          # + temporal chunking (v2)
│   └── QuantumFlowMatching.py        # Original QFM prototype
├── data/
│   ├── Load_Image_Datasets.py        # MNIST, Fashion-MNIST, CIFAR-10 loaders
│   ├── Load_PhysioNet_EEG.py         # PhysioNet Motor Imagery EEG loader
│   ├── Load_GLUE.py                  # HuggingFace GLUE with BERT tokenizer
│   └── Load_Text_Datasets.py        # text8, WikiText-2/103, PTB (char-level)
├── jobs/                              # SLURM batch scripts
├── docs/                              # Paper summaries and design documents
├── reference/                         # Reference implementations
├── results/                           # Generated samples and logs
└── checkpoints/                       # Model weights
```

## QTSTransformer Version History

| Version | PE | Angle Scaling | SELECT Caching | Chunking | Measurement |
|---------|:--:|:-------------:|:--------------:|:--------:|:-----------:|
| v1      |    |               |                |          | Fixed PauliX/Y/Z |
| v1.5    | Yes | Sigmoid * 2pi | Yes           |          | Fixed PauliX/Y/Z |
| v2      | Yes | Sigmoid * 2pi | Yes           | Yes      | Fixed PauliX/Y/Z |
| ModularQTS | Yes | Sigmoid * 2pi | Yes        |          | Learnable ANO |

## Datasets

| Dataset | Type | Loader | Models |
|---------|------|--------|--------|
| MNIST / Fashion-MNIST / CIFAR-10 | Image | `Load_Image_Datasets.py` | ModularQFM, QuantumLatentCFM |
| COCO (2017) | Image | `Load_Image_Datasets.py` | QuantumLatentCFM |
| PhysioNet EEG | Spatio-temporal | `Load_PhysioNet_EEG.py` | ModularQTS |
| GLUE (SST-2, CoLA, MRPC, ...) | NLP | `Load_GLUE.py` | ModularQTS_NLP |
| text8 | Character-level text (100M chars) | `Load_Text_Datasets.py` | QuantumLatentCFM_Text |
| WikiText-2 | Character-level text (~13M chars) | `Load_Text_Datasets.py` | QuantumLatentCFM_Text |
| WikiText-103 | Character-level text (~512M chars) | `Load_Text_Datasets.py` | QuantumLatentCFM_Text |
| Penn Treebank (PTB) | Character-level text (~5.7M chars) | `Load_Text_Datasets.py` | QuantumLatentCFM_Text |

## Shared Components

### Adaptive Non-Local Observables (ANO)

Learnable k-local Hermitian matrices replacing fixed Pauli measurements:
- `create_Hermitian(N, A, B, D)`: Build N x N Hermitian from real parameters
- `get_wire_groups(n_qubits, k_local, scheme)`: `sliding` or `pairwise` grouping
- Per group (k=2): 4x4 Hermitian with 16 learnable parameters
- Eigenvalue range logged per epoch for diagnostic monitoring

### Dual Optimizer (Chen et al., 2025)

```python
optimizer_circuit = Adam(circuit_params, lr=1e-3)
optimizer_obs    = Adam(observable_params, lr=1e-1)   # 100x higher
```

Observable eigenvalues must expand quickly so circuit parameters can learn meaningful gradients.

## SLURM Usage

```bash
sbatch jobs/run_mqfm.sh          # ModularQFM on MNIST
sbatch jobs/run_qlcfm_phase1.sh  # VAE pretraining
sbatch jobs/run_qlcfm_phase2.sh  # Quantum CFM training
```

Typical configuration: 1 GPU (A100 80GB), 32 CPUs, 48h wall time. See `jobs/` for all scripts.

## Documentation

| Document | Description |
|----------|-------------|
| `docs/MODEL_SUMMARY.md` | Detailed comparison of ModularQFM vs ModularQTS |
| `docs/QLCFM_EXPLAINED.md` | Deep dive into Quantum Latent CFM architecture |
| `docs/QuantumFlowMatching.md` | Framework design document |
| `docs/Chen2025_LearnableObservables_Summary.md` | Paper summary |
| `docs/Lin2025_AdaptiveNonLocalObservable_Summary.md` | Paper summary |
| `docs/Wiersema2024_SUN_Summary.md` | Paper summary |

## References

1. Wiersema, R. et al. (2024). "Here comes the SU(N)." *Quantum*, 8, 1275.
2. Chen, S. Y.-C. et al. (2025). "Learning to Measure Quantum Neural Networks." *ICASSP 2025*.
3. Lin, H.-Y. et al. (2025). "Adaptive Non-Local Observable on QNNs." *IEEE QCE 2025*.
4. Sim, S. et al. (2019). "Expressibility and Entangling Capability of Parameterized Quantum Circuits." *Adv. Quantum Technol.*, 2, 1900070.
5. Lipman, Y. et al. (2023). "Flow Matching for Generative Modeling." *ICLR 2023*.
6. Cherrat, E. A. et al. (2024). "Quantum Vision Transformers." *Quantum*, 8, 1265.
7. Gu, A. & Dao, T. (2023). "Mamba: Linear-Time Sequence Modeling with Selective State Spaces."
8. Vaswani, A. et al. (2017). "Attention Is All You Need." *NeurIPS 2017*.

## License

This project is for research purposes.
