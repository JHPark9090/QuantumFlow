# QLCFM Paper Assessment: Honest Critical Analysis

**Date:** 2026-03-08
**Purpose:** Candid evaluation of publishability at NeurIPS main track and recommendations for maximizing paper impact.

---

## 1. What NeurIPS Main Track Reviewers Would Want to See

### 1.1 Theoretical Contributions (Most Critical Gap)

NeurIPS main track accepts QML papers through one of two paths: strong empirical results or strong theory. Our empirical results (quantum VF comparable to but not better than classical MLP) are insufficient alone. A theoretical contribution is essential.

**What reviewers expect:**

**(a) Expressivity Results**
- A theorem showing that quantum velocity fields can represent certain flow families that require exponentially larger classical networks (or vice versa).
- Currently we have: "SU(4) encoding generates any 2-qubit unitary" -- this is a known fact about SU(4), not a theorem about velocity field expressivity.
- What we would need: A formal statement like "The class of velocity fields representable by an n-qubit QViT circuit with depth d and pairwise ANO has Fourier spectrum X, whereas a classical MLP with P parameters can only represent spectrum Y" with proof.
- Difficulty: High. This requires new mathematical analysis connecting quantum circuit Fourier spectra (Schuld et al., 2021) to the flow matching objective.

**(b) Sample Complexity / Generalization Bounds**
- A theorem showing that quantum VF generalizes better with fewer training samples than a classical MLP of comparable capacity.
- Could leverage the implicit regularization from the 28-dimensional observable bottleneck.
- Difficulty: Very high. No existing QML generalization theory covers this setting.

**(c) Geometric / Manifold Arguments**
- A formal connection between unitary group structure SU(2^n) and optimal transport on the data manifold.
- The exponential mapping U(theta) = exp(i sum_k theta_k G_k) traces geodesics on SU(N). If the data manifold has structure that aligns with this geometry, quantum VF could be provably better.
- Currently we have: qualitative description. What we need: a theorem with assumptions and proof.
- Difficulty: High but most promising direction.

**(d) Noise Robustness Guarantees**
- A theorem showing that the ANO measurement scheme degrades gracefully under depolarizing noise at rate p.
- Could prove that learnable observables with expanding eigenvalue range partially compensate for noise-induced contraction.
- Difficulty: Moderate. Could potentially adapt results from error mitigation literature.

**Assessment:** Without at least one theorem with proof, the paper will be rejected at NeurIPS main track. Reviewers will classify it as "interesting engineering" rather than "scientific contribution."

### 1.2 Empirical Results (Secondary but Necessary)

Even with theory, reviewers will evaluate empirical quality. Current state:

**Generation Quality (FID):**

| Model | FID | IS | Context |
|-------|-----|-----|---------|
| QLCFM v6 (quantum, lat=32) | 287.43 | 2.34 | Our best controlled result |
| Classical-D (fair control) | 265.98 | 2.37 | Classical baseline |
| QLCFM quantum (lat=128) | 192.73 | 2.65 | Best quantum FID |
| Classical MLP (lat=128) | 188.70 | 2.90 | Best classical FID |
| DDPM (Ho et al., 2020) | ~3.17 | — | SOTA diffusion |
| StyleGAN2-ADA | ~2.92 | — | SOTA GAN |

**The gap to SOTA is ~60-100x.** Reviewers will note this immediately.

**What we need to improve:**
1. Use a stronger pretrained VAE (e.g., from Stable Diffusion's VAE or a separately trained high-quality VAE) to bring FID below 50.
2. Run experiments on multiple datasets (CIFAR-10, COCO, CelebA) to show generalization.
3. Include reconstruction FID to separate VAE quality from VF quality.
4. Report with 50K evaluation samples (already implemented).

**Quantum vs Classical Comparison (Critical):**

| latent_dim | Quantum FID | Classical FID | Quantum wins? |
|------------|-------------|---------------|---------------|
| 32 | 236.39 | 230.30 | No |
| 64 | 238.78 | 233.72 | No |
| 128 | 192.73 | 188.70 | No |

Classical wins at every latent_dim. The gap is small (2-5%) but consistent. Reviewers will ask: "Why should anyone use the quantum version?"

**Possible reframing strategies:**
- Parameter efficiency: Quantum VF uses fewer total parameters (51K vs 173K at lat=32). But 98.6% of quantum VF params are classical wrappers.
- Show that with matched parameter counts, quantum is competitive -- design a classical MLP with exactly 51K params and compare.
- Focus on scaling behavior: as data complexity increases, quantum may scale differently.

### 1.3 Ablation Studies (Partially Done)

Reviewers will expect systematic ablations. We have some, need others:

| Ablation | Status | Finding |
|----------|--------|---------|
| obs/latent_dim ratio | Done | Ratio is NOT the bottleneck |
| Butterfly vs pyramid topology | Analyzed, not run | Butterfly covers 42.9% of pairs |
| QViT vs QCNN gates | Done (v1) | QViT superior |
| SU(4) vs angle encoding | Done (v8) | SU(16) collapses, SU(4) works |
| Number of circuits (1 vs 8) | Done (v6 vs v7) | 8x slower, 5% better MSE |
| Fixed Pauli vs ANO | Done (v1 vs v2) | ANO essential |
| Logit-normal vs uniform sampling | Not done | Need for v9 |
| Midpoint vs Euler ODE | Not done | Need for v9 |
| VF EMA vs no EMA | Not done | Need for v9 |
| Noise robustness sweep | Not done | Needed |

**Missing critical ablation:** A parameter-matched classical baseline. Our classical MLP has 173K params vs quantum's 51K. We need a 51K-param classical MLP to make a fair efficiency claim.

### 1.4 Writing and Presentation Standards

- Clear problem formulation with precise mathematical notation
- Algorithm box (pseudocode) for the full QLCFM pipeline
- Circuit diagrams (already have infrastructure for this)
- Convergence plots (loss curves, FID over epochs)
- Generated sample grids (qualitative evaluation)
- Computational cost comparison (wall time, FLOPS)
- Limitations section (honest about quantum-classical gap)

---

## 2. NeurIPS-Specific Concerns

### 2.1 QML Skepticism

NeurIPS/ICML reviewers have seen many "first quantum X" papers:
- "Quantum GANs" (multiple papers, no advantage shown)
- "Quantum Diffusion Models" (recent, inconclusive)
- "Quantum Transformers" (various, limited to small-scale)

Our paper risks being grouped with these. To differentiate:
- Acknowledge the quantum-classical gap honestly
- Focus on methodological contribution (how to integrate quantum circuits into flow matching) rather than claiming advantage
- Emphasize what we learned (VAE dominates, ratio doesn't matter, ANO is essential)

### 2.2 Reproducibility Concerns

NeurIPS requires reproducible results. Quantum simulation is inherently slow:
- Our experiments took ~25h per 200-epoch run on A100
- Reviewers may ask for additional experiments that take weeks
- Need to provide clear instructions and code

### 2.3 Relevance to ML Community

The core NeurIPS question: "Does this advance machine learning?"
- If the answer is "it's the same as classical but quantum" -- that's insufficient
- If the answer is "it reveals insights about velocity field design through the quantum lens" -- that's borderline
- If the answer is "quantum VF has provable properties that classical can't match" -- that's a contribution
- If the answer is "quantum VF achieves comparable results with fundamentally different computational primitives, suggesting future quantum hardware could accelerate flow matching" -- that's interesting but speculative

### 2.4 Scalability Questions

Reviewers will ask about scaling to practical sizes:
- 8 qubits = 256-dimensional Hilbert space (tiny)
- Current quantum hardware: 100-1000+ qubits
- How does performance change with qubit count? (We don't know)
- Can this work on real hardware? (Not yet demonstrated)

### 2.5 Comparison Fairness

Reviewers will scrutinize the quantum-classical comparison:
- Classical MLP has 173K params, quantum VF has 51K -- but only 736 are quantum
- The "sandwich" architecture (enc_proj -> quantum -> vel_head) means most computation is classical
- A fair comparison needs parameter-matched and FLOP-matched baselines

---

## 3. Noise and Hardware Feasibility Analyses

### 3.1 Depolarizing Noise Sweep

**What it is:** Apply depolarizing noise at various error rates after every gate in the quantum circuit. Depolarizing noise replaces the qubit state with the maximally mixed state with probability p.

**How to implement:**
```python
import pennylane as qml
from pennylane import noise as qml_noise

# Parameterized noise model
def make_noise_model(p_single=0.001, p_two=0.01):
    single_q = qml_noise.op_in([qml.RX, qml.RY, qml.RZ, qml.U3])
    two_q = qml_noise.op_in([qml.IsingXX, qml.IsingYY, qml.IsingZZ])

    def single_noise(op, **kwargs):
        qml.DepolarizingChannel(p_single, wires=op.wires)

    def two_noise(op, **kwargs):
        for w in op.wires:
            qml.DepolarizingChannel(p_two, wires=w)

    return qml_noise.NoiseModel({single_q: single_noise, two_q: two_noise})

# Sweep: p in [0, 1e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1]
# For each p: load trained weights, evaluate FID/IS on test set
```

**What it proves:**
- At what noise rate does generation quality degrade significantly?
- Compare to IBM Eagle/Heron error rates (~1e-3 single-qubit, ~1e-2 two-qubit)
- If FID is stable up to p=1e-2, the model is compatible with current hardware
- If FID degrades sharply at p=1e-3, real hardware deployment is infeasible without error mitigation

**Expected outcome:** With 8 qubits and depth ~10-20, depolarizing noise at p=1e-2 will likely degrade FID by 20-50%. At p=1e-3, degradation should be <5%.

**Cost:** Low. Use pre-trained weights and evaluate only (no retraining). ~1-2 hours per noise level.

### 3.2 Amplitude Damping Sweep

**What it is:** Models T1 relaxation (energy loss to environment). The excited state |1> decays to |0> with probability gamma per gate.

**How to implement:**
```python
def make_amp_damp_model(gamma=0.01):
    all_gates = qml_noise.op_in([qml.RX, qml.RY, qml.RZ, qml.U3,
                                  qml.IsingXX, qml.IsingYY, qml.IsingZZ])
    def amp_noise(op, **kwargs):
        for w in op.wires:
            qml.AmplitudeDamping(gamma, wires=w)
    return qml_noise.NoiseModel({all_gates: amp_noise})

# Sweep: gamma in [0, 1e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1]
```

**What it proves:**
- T1 relaxation is the dominant error on superconducting hardware
- IBM Eagle T1 ~ 100-300 us, gate time ~ 0.1-1 us, so gamma ~ 0.001-0.01 per gate
- Shows whether amplitude damping (asymmetric noise) affects ANO measurements differently than depolarizing (symmetric noise)

**Cost:** Same as depolarizing sweep. ~1-2 hours per noise level.

### 3.3 Combined Realistic Hardware Noise Model

**What it is:** Simulate a realistic hardware noise profile combining multiple error sources.

**How to implement:**
```python
def make_realistic_noise(p_single=0.001, p_two=0.01, gamma=0.005, p_readout=0.01):
    """
    Realistic noise model based on IBM Eagle/Heron specifications:
    - Single-qubit gate error: ~0.1% (depolarizing)
    - Two-qubit gate error: ~1% (depolarizing)
    - T1 amplitude damping: ~0.5% per gate
    - Readout error: ~1%
    """
    single_q = qml_noise.op_in([qml.RX, qml.RY, qml.RZ, qml.U3])
    two_q = qml_noise.op_in([qml.IsingXX, qml.IsingYY, qml.IsingZZ])

    def single_noise(op, **kwargs):
        qml.DepolarizingChannel(p_single, wires=op.wires)
        qml.AmplitudeDamping(gamma, wires=op.wires)

    def two_noise(op, **kwargs):
        for w in op.wires:
            qml.DepolarizingChannel(p_two, wires=w)
            qml.AmplitudeDamping(gamma, wires=w)

    return qml_noise.NoiseModel({single_q: single_noise, two_q: two_noise})
    # Note: readout error applied separately via qml.transforms or post-processing
```

**What it proves:**
- End-to-end feasibility on current quantum hardware
- Whether the combined effect of multiple noise sources catastrophically degrades performance
- Provides a single "would this work on real hardware?" answer

**Cost:** ~2-3 hours per hardware profile (IBM Eagle, IBM Heron, IonQ Aria).

### 3.4 ZNE Error Mitigation

**What it is:** Zero-Noise Extrapolation runs the circuit at multiple noise levels and extrapolates to zero noise. Already implemented in our codebase (MultiChip.py).

**How to implement:**
```python
# Already available via:
scale_factors = [1, 2, 3]
mitigated_qnode = qml.transforms.mitigate_with_zne(
    noisy_qnode,
    scale_factors=scale_factors,
    fold_method=qml.transforms.fold_global,
    extrapolate_method=qml.transforms.richardson_extrapolate
)
```

**What it proves:**
- Whether ZNE can recover near-ideal performance from noisy circuits
- The overhead (3x circuit evaluations for scale_factors=[1,2,3])
- Whether Richardson extrapolation is stable for our circuit depth
- Practical error mitigation applicability

**Expected outcome:** ZNE should recover 50-80% of the noise-induced FID degradation at moderate noise (p=1e-2). At high noise (p=0.1), extrapolation becomes unstable.

**Cost:** 3x the evaluation cost of a single noise level (~3-6 hours).

### 3.5 Circuit Depth and Gate Count Analysis

**What it is:** Count the exact number of native gates after transpilation to hardware gate sets.

**How to implement:**
```python
import pennylane as qml

# Get circuit specs
specs = qml.specs(circuit)(sample_input)
print(f"Depth: {specs['depth']}")
print(f"Gate counts: {specs['resources'].gate_counts}")
print(f"Num operations: {specs['resources'].num_gates}")

# Transpile to native gate set
# IBM: CX + U3 (or CZ + sqrt(X) + Rz for newer hardware)
# IonQ: MS + R (Molmer-Sorensen + single-qubit rotations)
```

**What it proves:**
- The raw circuit depth determines noise accumulation
- If depth < quantum volume of target hardware, the circuit is executable
- IBM Eagle quantum volume: 127 (log2 ~ 7 layers of CX gates on 7+ qubits)
- Our 8-qubit circuit with depth ~20 is likely within reach

**Cost:** Negligible (static analysis, no simulation needed).

### 3.6 Quantum Volume and Feasibility Assessment

**What it is:** Compare our circuit requirements against current hardware specifications.

**Analysis table to produce:**

| Metric | Our Circuit | IBM Eagle (127q) | IBM Heron (133q) | IonQ Aria (25q) |
|--------|-------------|-------------------|-------------------|-----------------|
| Qubits needed | 8 | 127 available | 133 available | 25 available |
| Circuit depth | ~20 | QV=127 (depth~7) | QV=TBD | QV=25 (depth~25) |
| Two-qubit gates | ~24 | CX error ~1% | CX error ~0.3% | MS error ~0.5% |
| Single-qubit gates | ~48 | U3 error ~0.1% | error ~0.03% | R error ~0.03% |
| Connectivity | All-to-all | Heavy-hex (limited) | Heavy-hex | All-to-all |
| Estimated fidelity | — | ~0.78 | ~0.93 | ~0.88 |

**What it proves:**
- Which current hardware platforms could run our circuit
- IonQ's all-to-all connectivity is ideal for our QViT butterfly topology
- IBM's limited connectivity requires SWAP gates, increasing depth
- Estimated circuit fidelity = product of all gate fidelities

**Cost:** Negligible (calculation from published hardware specs).

### 3.7 Noise-Aware Training (Advanced)

**What it is:** Train the quantum VF with noise injected during training, not just evaluation.

**How to implement:**
```python
# Replace ideal device with noisy device during training
dev = qml.device("default.mixed", wires=n_qubits)
# Apply noise model to QNode
noisy_qnode = qml.QNode(circuit, dev)
noisy_qnode = qml.transforms.add_noise(noisy_qnode, noise_model)
```

**What it proves:**
- Whether the model can learn to compensate for noise
- If noise-aware training produces better noisy-hardware performance than ideal training + post-hoc noise
- This is a stronger result than post-hoc noise evaluation

**Cost:** High. Full retraining required (~25h per noise level). But only need 2-3 noise levels (ideal, moderate p=1e-2, high p=5e-2).

**Priority:** Do this only if post-hoc noise evaluation shows promising robustness.

### 3.8 Recommended Analysis Priority

| Priority | Analysis | Cost | Impact | Do it? |
|----------|----------|------|--------|--------|
| 1 | Circuit depth / gate count (3.5) | Minutes | Medium | Yes -- trivial |
| 2 | Quantum volume assessment (3.6) | Minutes | Medium | Yes -- trivial |
| 3 | Depolarizing noise sweep (3.1) | ~10 GPU-hours | High | Yes |
| 4 | Amplitude damping sweep (3.2) | ~10 GPU-hours | Medium | Yes |
| 5 | Combined realistic noise (3.3) | ~5 GPU-hours | High | Yes |
| 6 | ZNE error mitigation (3.4) | ~15 GPU-hours | High | Yes |
| 7 | Noise-aware training (3.7) | ~75 GPU-hours | Very High | If 3-6 look good |

**Total estimated cost for priorities 1-6: ~40 GPU-hours (less than 2 days on a single A100).**

---

## 4. Honest Recommendations

### 4.1 Assessment Summary

| Criterion | Current State | Needed for NeurIPS | Gap |
|-----------|--------------|-------------------|-----|
| Novelty | First quantum flow matching | Novel method + insight | Partial (method novel, insight needed) |
| Theory | Architecture description only | At least one theorem | Large |
| Empirical | FID ~190, quantum ~= classical | Clear benefit or deep analysis | Medium |
| Ablations | Ratio experiment done | Parameter-matched baselines | Small |
| Hardware | None | Noise analysis | Medium |
| Writing | Docs exist | Polished 9-page paper | Medium |

### 4.2 Probability Estimates for NeurIPS Main Track

| Scenario | Estimated acceptance probability |
|----------|--------------------------------|
| Current results only | <5% |
| + Better VAE (FID < 50) | ~5-8% |
| + Theory sections (no theorems) | ~8-12% |
| + Noise analysis (Sections 3.1-3.6) | ~12-15% |
| + One formal theorem with proof | ~20-30% |
| + Parameter-matched classical baseline showing quantum competitive | ~25-35% |
| All of the above combined | ~30-40% |

Even with all improvements, acceptance is not guaranteed. The fundamental issue -- no demonstrated quantum advantage in generation quality -- cannot be engineered away with better experiments. It requires either a theoretical justification or a genuinely new finding.

### 4.3 What Would Genuinely Change the Story

**(a) Discovery of a setting where quantum VF wins.** If there exists a data distribution, latent structure, or noise regime where the quantum VF demonstrably outperforms the classical MLP at matched parameters, that would be a NeurIPS paper. We haven't found this yet, but possibilities include:
- Structured latent spaces with symmetries matching SU(N)
- Low-data regime (few-shot flow matching)
- Specific noise regimes where quantum circuits are naturally robust

**(b) A no-go theorem.** Proving that quantum VFs *cannot* outperform classical MLPs for flow matching would also be a NeurIPS contribution (negative results are valued when they're surprising and well-proven). This would require showing that the observable bottleneck fundamentally limits quantum VF expressivity.

**(c) Connection to quantum chemistry / physics.** If the flow matching framework could be applied to a domain where quantum circuits have a natural advantage (molecular dynamics, quantum state generation), the paper becomes much stronger. The method paper plus a compelling application would be competitive.

### 4.4 Recommended Venue Strategy

**Tier 1 Target: npj Quantum Information (or Physical Review Research)**
- Timeline: Submit within 2-3 months
- Content: Current method + noise analysis + improved VAE + theoretical framework (no formal theorems needed)
- Probability: ~60-70%
- Impact factor: ~6.7 (npj QI)
- Audience: Quantum computing community, where "first quantum flow matching" is genuinely novel

**Tier 2 Target: IEEE QCE 2026 or Quantum Science and Technology**
- Timeline: Submit within 1-2 months
- Content: Current method + basic noise analysis
- Probability: ~70-80%
- More applied/engineering-focused audience

**Ambitious Target: ICLR 2027 (October 2026 deadline)**
- Timeline: 6-7 months to develop theory
- Content: Everything above + at least one theorem + parameter-matched baselines
- Probability: ~20-30%
- Advantage over NeurIPS: ICLR has been slightly more receptive to QML papers

**NeurIPS 2026 (May 2026 deadline)**
- Timeline: ~2 months (very tight)
- Probability with all improvements: ~30-40%
- Risk: High chance of rejection even with significant effort

### 4.5 Recommended Action Plan

**Phase 1 (Immediate, 1-2 weeks):**
1. Complete VAE scaling experiment (64 to 2048 latent dims) -- already running
2. Run circuit depth / gate count analysis (trivial)
3. Run quantum volume assessment (trivial)
4. Design parameter-matched classical baseline (51K params MLP)

**Phase 2 (Weeks 2-4):**
5. Run noise analysis suite (depolarizing, amplitude damping, combined, ZNE)
6. Run QLCFM v9 with best VAE weights (butterfly and pyramid topologies)
7. Run parameter-matched classical baseline experiments
8. Begin writing theoretical framework section

**Phase 3 (Weeks 4-8):**
9. Attempt to prove at least one formal result about quantum VF expressivity
10. Write full paper draft
11. Generate all figures and tables
12. Internal review and revision

**Phase 4 (Week 8+):**
13. Submit to chosen venue
14. If targeting NeurIPS: submit by May deadline
15. If targeting npj QI: submit when ready (rolling submission)

### 4.6 Final Honest Assessment

This project has produced a **genuine methodological contribution**: the first integration of quantum circuits with conditional flow matching. The systematic experiments (10+ quantum configurations, 4 classical baselines, ratio bottleneck study, VAE scaling) demonstrate serious research effort and yield useful insights for the QML community.

However, the core finding is that **quantum velocity fields are slightly worse than classical MLPs** for image generation, and the quantum component constitutes only 1.4% of the total model parameters. No amount of engineering or noise analysis changes this fundamental result.

**The strongest version of this paper** frames it as:
1. "Here is how to do quantum flow matching" (methodological novelty)
2. "Here is what we learned about quantum velocity fields" (empirical insights)
3. "Here is when it could work on real hardware" (feasibility analysis)
4. "Here is why the quantum structure matters theoretically" (if we can prove something)

This is a solid paper for quantum computing venues. For top ML venues, it needs the theoretical component that we currently lack.
