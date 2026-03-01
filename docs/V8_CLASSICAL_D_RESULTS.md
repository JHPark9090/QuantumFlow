# v8-Quantum vs Classical-D: Results and Analysis

**Date**: 2026-02-26
**Jobs**: v8-Quantum (49342732), Classical-D (49342733)
**VAE**: Reused v6 VAE weights (`weights_vae_qlcfm_cifar_v6_49235601.pt`, resconv, latent_dim=32)

## 1. Experiment Setup

v8 introduced SU(16) encoding — a more expressive quantum data encoding that maps input features into higher-dimensional special unitary groups. Classical-D is its fair classical control, sharing the identical `enc_proj` and `vel_head` layers. The ONLY difference is the core that maps 765 encoding dimensions to 28 outputs.

### Architecture Comparison

| Component | v8-Quantum | Classical-D |
|-----------|-----------|-------------|
| **time_mlp** | 32-dim | 32-dim |
| **Input** | concat(z_t[32], t_emb[32]) = 64 | same |
| **enc_proj** | Linear(64,1024) → SiLU → Linear(1024,765) | same |
| **Core** | SU(16) encoding → QViT butterfly (depth=2) → ANO pairwise k=2 | MLP: Linear(765,64) → SiLU → Linear(64,28) |
| **vel_head** | Linear(28,256) → SiLU → Linear(256,32) | same |
| **Total VF params** | 869,181 | 919,324 |
| **Core params** | 1,102 (circuit) + 448 (ANO) = 1,550 | 50,876 |

The SU(16) encoding uses `sun_group_size=4`, creating 3 groups of 4 qubits on the 8-qubit circuit. Each group is parameterized by a 15-dim special unitary, totaling 765 encoding parameters per circuit.

### Comparison with v6

| Component | v6-Quantum | v8-Quantum |
|-----------|-----------|-----------|
| **enc_proj** | Linear(64,256) → SiLU → Linear(256,N_enc) | Linear(64,1024) → SiLU → Linear(1024,765) |
| **Encoding** | SU(4) (~100 params) | SU(16) (765 params) |
| **enc_proj hidden** | 256 | 1024 |
| **Total VF params** | ~76,000 | 869,181 |

## 2. Results

### Final Metrics

| Metric | v8-Quantum (ep 63, cancelled) | Classical-D (ep 200) | v6-Quantum (ep 200) |
|--------|-------------------------------|----------------------|---------------------|
| **Val MSE** | 1.244 | **0.476** | **0.485** |
| **FID (1024)** | — (not computed) | **265.98** | 287.43 |
| **IS** | — | **2.37 ± 0.02** | 2.34 |
| **Wall time** | 7h 41m (cancelled) | 24 min | 27.6h |
| **Time/epoch** | ~440s | ~6.5s | ~498s |

### Training Curve Milestones

| Epoch | v8-Q val_loss | Classical-D val_loss | v6-Q val_loss | v8 Eig range |
|-------|---------------|----------------------|---------------|--------------|
| 1 | 1.042 | 0.583 | 0.596 | [-14, +23] |
| 10 | 1.064 | 0.515 | 0.548 | [-79, +75] |
| 20 | 1.241 | 0.510 | 0.528 | [-101, +104] |
| 40 | 1.237 | 0.497 | 0.526 | [-92, +127] |
| 62 | 1.244 | 0.491 | 0.517 | [-118, +137] |
| 100 | — | 0.480 | 0.505 | — |
| 200 | — | 0.476 | 0.485 | — |

### v8-Quantum Training Trajectory (Detailed)

The v8-Quantum training shows a distinctive failure pattern:

1. **Epochs 1–10**: Initial descent from 1.04 to 1.06 val MSE, with eigenvalues growing rapidly ([-14,+23] → [-79,+75]).
2. **Epoch 11**: Sudden jump to val MSE 1.24 — the model collapses.
3. **Epochs 11–63**: Complete plateau at val MSE ~1.24 with no recovery. Eigenvalues continue growing to [-118,+137].

## 3. Interpretation

### Why Classical-D Outperforms Classical-C

Classical-D (FID 265.98) is notably better than Classical-C (FID 276.89). This is because Classical-D shares v8's larger `enc_proj` (1024 hidden, 765 output) and matched `vel_head` architecture, giving it a richer intermediate representation. Classical-C had a different topology and bottleneck structure, making it an unfair comparison to any quantum model.

**Classical-D is the proper classical control for v8-Quantum** — they differ ONLY in the 765→28 core.

### Why v8-Quantum Failed

Despite sharing the same `enc_proj` that helps Classical-D, v8-Quantum performs 2.6x worse than both Classical-D and v6-Quantum. The failure has three interrelated causes:

#### (a) SU(16) Creates a Harder Optimization Landscape

SU(4) encoding (v6) maps ~100 parameters into 2-qubit unitary gates — a relatively smooth landscape where gradient-based optimization works. SU(16) encoding maps 765 parameters into 4-qubit unitary gates operating in a 16-dimensional Hilbert space. The optimization landscape of SU(16) is exponentially more complex:
- More local minima and barren plateau regions
- Gradient information becomes less informative as the unitary dimension grows
- The enc_proj must learn a precise 64→765 mapping where small perturbations in the 765-dim space produce meaningful changes in quantum state — a much harder task than the 64→100 mapping for SU(4)

#### (b) Eigenvalue Explosion Indicates Runaway Observable Optimization

The ANO (Adaptive Non-local Observable) eigenvalues in v8 grow to [-118, +137] by epoch 62, compared to v6's much more controlled [-47, +27] at the same epoch. This indicates that the observable parameters are being driven to extreme values in an attempt to extract signal from a quantum circuit that isn't producing useful states. The high `lr-H=0.1` learning rate for observables, which works well for v6, is too aggressive when paired with a circuit that can't keep up.

#### (c) The Input Encoding Bottleneck (Not Just Output)

Previous analysis focused on the **output bottleneck** — whether 28 observables are sufficient to represent a 32-dim latent velocity. The v8 result reveals an equally important **input bottleneck**: the quantum circuit's ability to meaningfully process its encoding parameters.

- Classical-D's MLP core processes 765 input dims effortlessly via standard backpropagation.
- v8-Quantum's circuit must convert 765 encoding parameters into quantum gates that produce states where 28 pairwise measurements yield useful velocity predictions. This encoding→measurement pipeline is far harder to optimize than a simple MLP.

### The Paradox: More Expressive Encoding ≠ Better Performance

| | Encoding dim | Val MSE | Eig range (ep 62) |
|---|---|---|---|
| **v6-Quantum** (SU(4)) | ~100 | **0.517** | [-47, +27] |
| **v8-Quantum** (SU(16)) | 765 | **1.244** | [-118, +137] |

v8 has 7.6x more encoding parameters but performs 2.4x worse. This is a clear case of **expressivity-trainability tradeoff**: increasing the encoding space's theoretical expressivity made it practically untrainable. The quantum circuit has enough parameters to represent the solution, but gradient-based optimization cannot find it.

## 4. Comparison Across All Quantum Models

| Model | Qubits | Encoding | Val MSE (ep 21) | Final Val MSE | Epochs |
|-------|--------|----------|-----------------|---------------|--------|
| **v7** | 8×8q | SU(4), shared | **0.471** | **0.462** (ep 40) | 40 (timeout) |
| **v6** | 1×8q | SU(4), shared | 0.526 | **0.485** (ep 200) | 200 |
| v4-L2 | 1×12q | Re-upload L=2 | 0.694 | 0.619 (ep 195) | 195 |
| v4-L4 | 1×12q | Re-upload L=4 | 0.697 | 0.637 (ep 121) | 121 (timeout) |
| v5 | 16×8q | SU(4), split | 0.835 | 0.835 (ep 21) | 21 (timeout) |
| **v8** | 1×8q | SU(16), shared | **1.249** | **1.244** (ep 63) | 63 (cancelled) |

**Ranking: v7 > v6 >> v4-L2 > v4-L4 >> v5 >> v8**

The best-performing quantum models (v6, v7) share three properties:
1. SU(4) encoding (trainable)
2. Shared full input (not split across circuits)
3. Moderate enc_proj size (256 hidden)

## 5. Key Takeaways

1. **SU(16) encoding is a negative result.** Despite 7.6x more encoding parameters and 11x more total VF parameters, v8-Quantum is the worst-performing quantum model tested. The expressivity-trainability tradeoff decisively favors smaller encoding groups.

2. **Classical controls benefit from larger enc_proj; quantum does not.** Classical-D improved over Classical-C with the same architectural changes that broke v8-Quantum. This asymmetry is fundamental: classical cores (MLPs) scale gracefully with input dimension, while quantum circuits face exponentially harder optimization landscapes.

3. **The quantum bottleneck is two-sided.** Prior focus was on the output channel (28 obs vs latent_dim). v8 shows the input encoding is equally critical — the circuit must be trainable, not just expressive.

4. **v8-Quantum should be cancelled** (done) — the flat plateau since epoch 11 with no recovery after 63 epochs indicates the model is stuck in a barren plateau and will not improve. Remaining wall time (~40h) would be wasted compute.

5. **For future work**: If richer encoding is desired, consider intermediate approaches — SU(8) (group_size=3, 63 params per group) or multiple SU(4) layers — rather than jumping to SU(16). Alternatively, investigate learning rate schedules that warm up the encoding parameters more gradually.

## 6. Files

| File | Description |
|------|-------------|
| `results/log_cfm_qlcfm_cifar_v8_49342732.csv` | v8-Quantum training log (63 epochs) |
| `results/log_cfm_classical_cfm_D_49342733.csv` | Classical-D training log (200 epochs) |
| `results/metrics_classical_cfm_D_49342733.json` | Classical-D FID/IS metrics |
| `checkpoints/weights_cfm_classical_cfm_D_49342733.pt` | Classical-D trained weights |
| `logs/qlcfm_cifar_v8_49342732.out` | v8-Quantum stdout log |
| `logs/classical_cfm_D_49342733.out` | Classical-D stdout log |
