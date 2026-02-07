# Chen et al. (2025) — "Learning to Measure Quantum Neural Networks"

**Citation:** Chen, S. Y.-C., Tseng, H.-H., Lin, H.-Y., & Yoo, S. (2025). Learning to Measure Quantum Neural Networks. *ICASSP 2025 Workshop: Quantum Machine Learning in Signal Processing and Artificial Intelligence*. https://arxiv.org/abs/2501.05663

---

## Core Contribution

The paper proposes making the **quantum observable (Hermitian measurement matrix) a learnable parameter**, trained end-to-end alongside standard VQC rotation angles via gradient-based optimization. This replaces fixed Pauli measurements (e.g., Z) with a parameterized Hermitian matrix whose spectral range adapts during training.

---

## 1. Motivation: The Fixed Observable Bottleneck

**The Problem:**
Standard VQCs measure fixed Pauli observables (X, Y, or Z), which have eigenvalues λ = ±1. This restricts the expectation value output to the range [-1, 1]:

$$\lambda_{\min} \leq \langle\psi| H |\psi\rangle \leq \lambda_{\max}$$

By the **Rayleigh quotient**, the output of a VQC is bounded by the eigenvalue spectrum of its observable. For complex ML tasks (regression with unbounded targets, multi-class classification), this [-1, 1] restriction severely limits model expressivity.

**The Insight:**
The observable contributes **multiplicatively** to model capacity. Optimizing only circuit parameters while keeping the measurement fixed leaves significant expressive power on the table.

---

## 2. Mathematical Formulation

### 2.1 Standard VQC Output

For a standard VQC with encoding unitary U(x) and variational unitary W(Θ):

$$f(\mathbf{x}; \Theta) = \langle 0 | W^\dagger(\Theta)\, U^\dagger(\mathbf{x})\, H\, U(\mathbf{x})\, W(\Theta) | 0 \rangle$$

where H is a **fixed** observable (e.g., Pauli-Z on one qubit).

### 2.2 Learnable Observable Formulation

Replace fixed H with a **parameterized Hermitian matrix** B(b):

$$f(\mathbf{x}; \Theta, \mathbf{b}) = \langle 0 | W^\dagger(\Theta)\, U^\dagger(\mathbf{x})\, B(\mathbf{b})\, U(\mathbf{x})\, W(\Theta) | 0 \rangle$$

### 2.3 Hermitian Parameterization

The learnable observable B(b) is parameterized as:

$$B(\mathbf{b}) = \sum_{i,j} b_{ij}\, E_{ij}$$

where:
- **E_{ij}** are indicator (basis) matrices with a 1 at position (i,j) and 0 elsewhere
- **Hermiticity constraint:** b_{ij} = b̄_{ji} (complex conjugate symmetry)
- For an N×N matrix, this requires **N² real parameters:**
  - N real diagonal elements
  - N(N-1)/2 off-diagonal pairs (each with real and imaginary parts)

### 2.4 Gradient with Respect to Observable Parameters

The expectation value is **linear** in the observable parameters:

$$\langle\psi| B(\mathbf{b}) |\psi\rangle = \sum_{i,j} b_{ij} \langle\psi| E_{ij} |\psi\rangle$$

Therefore, the partial derivative with respect to observable parameters:

$$\frac{\partial}{\partial b_{kl}} \langle\psi| B(\mathbf{b}) |\psi\rangle = \langle\psi| E_{kl} |\psi\rangle$$

This is simply the (k,l) element of the **density matrix** ρ = |ψ⟩⟨ψ|. This makes gradient computation for observable parameters essentially free — no additional circuit evaluations needed beyond what's already computed for the state.

---

## 3. Spectral Range Expansion Mechanism

**How it works:**
- During training, the eigenvalues λ₁, ..., λ_N of B(b) are free to grow
- The output range expands from [-1, 1] to [λ_min, λ_max]
- The model learns to **scale** its output dynamically to fit the target distribution

**Rayleigh quotient bound:**

$$\lambda_{\min}(B) \leq f(\mathbf{x}; \Theta, \mathbf{b}) \leq \lambda_{\max}(B)$$

The spectral range [λ_min, λ_max] grows during training as the optimizer adjusts b to improve the loss function.

---

## 4. Training Procedure

### 4.1 Separate Optimizer Strategy (Key Finding)

The paper discovers that **separate optimizers** for circuit and observable parameters are crucial:

| Parameter Set | Optimizer | Learning Rate |
|---|---|---|
| Circuit rotations θ | RMSProp | 0.001 – 0.01 |
| Observable coefficients b | Adam | 0.1 |

**Rationale:** Observable parameters need a **much larger learning rate** (10-100x) because:
- The eigenvalue spectrum must expand quickly to enable the circuit parameters to train effectively
- Observable parameters control the output *scale*, while circuit parameters control the output *shape*
- Without fast eigenvalue expansion, the circuit gradients are suppressed by the narrow output range

### 4.2 Training Loop

```
For each epoch:
    1. Forward pass: compute |ψ⟩ = U(x) W(Θ) |0⟩
    2. Measure: f = ⟨ψ| B(b) |ψ⟩
    3. Compute loss L(f, y_target)
    4. Backpropagate:
       - ∂L/∂Θ → update circuit params with RMSProp (lr=0.001)
       - ∂L/∂b → update observable params with Adam (lr=0.1)
```

---

## 5. Experimental Results

### 5.1 Make-Moons Classification

- **Setup:** 4 qubits, 2-layer VQC, noise levels 0.1/0.2/0.3
- **Data:** 200 train, 100 test samples
- **Result:** Learnable observables with separate optimizers consistently outperform fixed Pauli-Z across all noise levels
- Eigenvalue ranges visibly expand during training (confirmed via spectral monitoring)

### 5.2 Speaker Recognition (VCTK Dataset)

- **Setup:** 3 CNN layers → 10 features → 10-qubit VQC → 10-class classification
- **Data:** VCTK speech spectrograms (257×128)
- **30 epochs, 5 independent trials**

| Configuration | Test Accuracy |
|---|---|
| Fixed Pauli-Z | 70.59% |
| Learnable observable (joint optimizer) | 76.83% |
| Learnable observable (separate optimizers) | **96.33%** |

The **25.74 percentage point improvement** from fixed to learnable (separate) demonstrates the critical role of measurement optimization.

---

## 6. Key Takeaways

1. **Measurement is as important as the circuit:** The observable is not just a readout — it's a multiplicative factor in model capacity
2. **Eigenvalue expansion is the mechanism:** Models learn to widen the spectral range [λ_min, λ_max] to match task requirements
3. **Separate optimizers are essential:** Observable parameters need 10-100x larger learning rates than circuit parameters
4. **Gradient computation is efficient:** Observable gradients reduce to density matrix elements — no extra circuit evaluations
5. **N² parameter overhead:** For n qubits, a full observable on the Hilbert space adds 2^{2n} real parameters; for k-qubit local observables this reduces to 2^{2k}

---

## 7. Relevance to Quantum Flow Matching

| Concept in QFM Framework | Backing from this paper |
|---|---|
| Learnable observable B(b) in f(x) = ⟨0|U†(x) B(b) U(x)|0⟩ | Core contribution: parameterized Hermitian measurement |
| Expanding output range beyond [-1,1] | Spectral range expansion via learnable eigenvalues |
| Efficient gradient for observable params | Gradients = density matrix elements (free) |
| Separate optimizer for observable vs circuit | Critical finding: 10-100x larger LR for observables |
| Compatibility with shallow circuits | Observable compensates for limited circuit depth |

---

## Limitations

- **Scalability:** Full N×N Hermitian has N² = 2^{2n} parameters for n qubits (exponential scaling)
- **Hardware validation:** Only simulation results; no demonstration on noisy quantum hardware
- **Measurement cost:** Extracting full density matrix elements on hardware requires tomography, which scales poorly
- **Local observables** (as addressed by Lin et al. 2025) can mitigate the scalability issue
