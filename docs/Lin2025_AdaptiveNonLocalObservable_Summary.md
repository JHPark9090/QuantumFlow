# Lin et al. (2025) — "Adaptive Non-Local Observable on Quantum Neural Networks"

**Citation:** Lin, H.-Y., Tseng, H.-H., Chen, S. Y.-C., & Yoo, S. (2025). Adaptive Non-local Observable on Quantum Neural Networks. *2025 IEEE International Conference on Quantum Computing and Engineering (QCE)*. https://arxiv.org/abs/2504.13414

---

## Core Contribution

The paper introduces **Adaptive Non-Local Observables (ANO)** — learnable k-local Hermitian measurement matrices that act on subsystems of k qubits. Using the Heisenberg picture as a design principle, ANO shifts complexity from deep variational circuits to richer measurement observables, enabling **shallower circuits** while maintaining or improving expressivity.

---

## 1. Motivation: The Heisenberg Picture Argument

### 1.1 The Equivalence

In the **Schrödinger picture**, we evolve the state and measure a fixed observable:

$$\langle\psi(\theta)| H |\psi(\theta)\rangle, \quad \text{where } |\psi(\theta)\rangle = U(\theta)|\psi\rangle$$

In the **Heisenberg picture**, we keep the state fixed and measure an evolved observable:

$$\langle\psi| H(\theta) |\psi\rangle, \quad \text{where } H(\theta) = U^\dagger(\theta)\, H\, U(\theta)$$

These are **mathematically equivalent:**

$$\langle\psi| U^\dagger(\theta)\, H\, U(\theta) |\psi\rangle = \langle\psi(\theta)| H |\psi(\theta)\rangle$$

### 1.2 The Design Principle

This equivalence implies that instead of making the circuit deeper (adding more variational layers to U(θ) to "twist" the quantum state), we can keep the circuit **shallow** and instead make the **observable richer** — i.e., learn a more expressive measurement operator H(ϕ) that captures the same correlations.

**Key insight:** The variational rotation U(θ) maps one Hermitian to another:

$$H \mapsto U^\dagger(\theta)\, H\, U(\theta)$$

So a trainable H(ϕ) combined with a shallow U(θ) can achieve the same effect as a fixed H with a deep U(θ).

---

## 2. Mathematical Formulation: The k-Local ANO

### 2.1 Definition

For an n-qubit system with k ≤ n, let **K = 2^k** (dimension of k-qubit Hilbert space). An adaptive k-local observable is a K×K Hermitian matrix:

$$H(\phi) = \begin{pmatrix}
c_{11} & a_{12} + ib_{12} & a_{13} + ib_{13} & \cdots & a_{1K} + ib_{1K} \\
* & c_{22} & a_{23} + ib_{23} & \cdots & a_{2K} + ib_{2K} \\
* & * & c_{33} & \cdots & a_{3K} + ib_{3K} \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
* & * & * & \cdots & c_{KK}
\end{pmatrix}$$

where:
- **ϕ = (a_{ij}, b_{ij}, c_{ii})** for i,j = 1,...,K are **K² real parameters**
  - K diagonal elements c_{ii} (real)
  - K(K-1)/2 off-diagonal pairs, each with 2 real parameters (a_{ij}, b_{ij})
  - Total: K + 2·K(K-1)/2 = K² parameters
- Lower triangle entries are complex conjugates: h_{ji} = h̄_{ij} (Hermiticity)
- The * entries denote the conjugate transpose values

### 2.2 Parameter Scaling

| k (locality) | K = 2^k | K² (real params per observable) |
|---|---|---|
| 1 | 2 | 4 |
| 2 | 4 | 16 |
| 3 | 8 | 64 |
| 4 | 16 | 256 |
| 5 | 32 | 1024 |

The parameter count grows as **4^k** — exponential in k but independent of total qubit count n.

---

## 3. Two Measurement Schemes

### 3.1 Sliding k-Local Measurements

Measures overlapping contiguous groups of k qubits in a sliding window:

For n qubits with locality k, the windows are:
- (qubit 1, ..., qubit k)
- (qubit 2, ..., qubit k+1)
- ...
- (qubit n-k+1, ..., qubit n)

**Number of windows:** M = n - k + 1

Each window w has its own adaptive observable H_w(ϕ_w) — a K×K Hermitian with K² parameters.

**Total parameters for sliding scheme:** (n - k + 1) × K²

**Advantages:**
- Captures nearest-neighbor and short-range correlations
- Linear scaling in n (number of windows)
- Systematic coverage of the qubit register

### 3.2 Pairwise Combinatorial Measurements

Measures all C(n, k) combinations of k qubits:

For pairwise (k=2) on n qubits:
- Pairs: (1,2), (1,3), ..., (1,n), (2,3), ..., (n-1,n)
- **Number of pairs:** C(n,2) = n(n-1)/2

Each pair has its own adaptive observable — a 4×4 Hermitian with 16 parameters.

**Total parameters for pairwise k=2:** n(n-1)/2 × 16

**Advantages:**
- Captures **all** pairwise correlations (not just nearest-neighbor)
- Strong performance even **without** variational rotation gates
- Efficient at capturing long-range feature interactions

---

## 4. Experimental Results

### 4.1 Banknote Authentication Dataset

- **Features:** 4 input features → 4 qubits
- **Setup:** 10 independent trials per configuration
- **Tested:** k = 1, 2, 3 for sliding scheme; with and without rotation gates

**Key finding:** Performance improves with k; removing rotation gates has decreasing impact as k grows (the observable alone captures correlations).

### 4.2 MNIST Classification

**Sliding k-Local ANO Results (Table II(a)):**

Accuracy improvement as k increases shows **diminishing returns:**
- 1-local → 2-local: **+9%**
- 2-local → 3-local: **+8%**
- 3-local → 4-local: **+6%**
- 4-local → 5-local: **+4%**

**Key observations:**
1. Consistent accuracy improvement as k increases
2. Diminishing marginal gains — suggests a practical optimal k
3. Larger k captures richer qubit interactions but with less incremental benefit

**Pairwise Combinatorial Results (Table II(b)):**
- Strong performance with **fewer parameters** than sliding
- Works well **without rotation gates** — highlighting efficiency in capturing feature interactions
- Particularly effective because it captures all pairwise correlations, not just contiguous ones

### 4.3 The Critical Finding: Observable Locality vs. Circuit Depth

**As non-local size k increases, the benefit of variational rotation gates diminishes.**

This experimentally validates the Heisenberg picture argument:
- At small k: rotation gates (circuit depth) are essential for performance
- At large k: the observable alone captures sufficient correlations
- **Implication:** Rich observables can substitute for deep circuits

---

## 5. Key Theoretical Insights

### 5.1 Expressivity Hierarchy

$$\text{Fixed Pauli} \subset \text{1-local ANO} \subset \text{2-local ANO} \subset \cdots \subset \text{n-local ANO (full)}$$

Higher k-local observables span a larger function class, with the full n-local observable being equivalent to the full learnable Hermitian of Chen et al. (2025).

### 5.2 The Depth-Locality Tradeoff

The paper demonstrates a fundamental tradeoff:

| More circuit depth (larger L) | More observable locality (larger k) |
|---|---|
| More rotation gates | More measurement parameters |
| Higher gate noise | Higher measurement overhead |
| Deeper circuits needed | Shallower circuits sufficient |
| Standard approach | ANO approach |

**Sweet spot:** Moderate k (2-3) with moderate depth — captures most correlations while keeping both circuit noise and measurement overhead manageable.

### 5.3 Connection to Chen et al. (2025)

| Aspect | Chen et al. (2025) | Lin et al. (2025) |
|---|---|---|
| Observable scope | Full N×N Hermitian | k-local K×K Hermitian (K=2^k) |
| Parameter count | N² = 2^{2n} (exponential in n) | K² = 4^k per window (exponential in k, not n) |
| Scalability | Limited by exponential growth | Scalable via locality truncation |
| Measurement scheme | Full system | Sliding windows or pairwise combinatorial |
| Key insight | Eigenvalue expansion | Heisenberg picture depth reduction |
| Complementary? | Yes — Chen provides the "what" | Yes — Lin provides the "how" to scale it |

---

## 6. Relevance to Quantum Flow Matching

| Concept in QFM Framework | Backing from this paper |
|---|---|
| Shifting complexity from circuit to measurement | Core thesis: Heisenberg picture ANO framework |
| Shallow circuit feasibility on NISQ hardware | Validated: rich observables reduce depth requirements |
| Practical measurement schemes | Sliding k-local and pairwise combinatorial |
| Scalable observable parameterization | k-local ANO with K² = 4^k parameters (tractable for small k) |
| Observable captures data correlations | Pairwise ANO captures feature interactions without rotation gates |
| Diminishing returns guide design | Optimal k exists — balances expressivity vs. measurement cost |

---

## 7. Implementation Considerations

### 7.1 Measurement Overhead
- Each k-local observable decomposes into up to 4^k Pauli strings
- Pauli strings can be grouped for joint measurement (QWC grouping)
- Tools: PennyLane's `qml.pauli.group_observables` for efficient grouping

### 7.2 Gradient Computation
- Observable parameters ϕ: linear dependence → gradients are density matrix elements (efficient)
- Circuit parameters θ: standard parameter-shift rules apply
- Separate optimizers recommended (following Chen et al. 2025 finding)

### 7.3 Sample Complexity
- Scales as O(4^k / ε²) per observable for precision ε
- Sliding scheme: O(n) observables → linear total measurement scaling in n
- Pairwise scheme: O(n²) observables → quadratic total measurement scaling in n

### 7.4 Suggested Practical Defaults (from paper patterns)
- Start with k = 2 (sliding) for initial experiments
- Use pairwise combinatorial for tasks requiring long-range correlations
- Compare with/without rotation gates to assess observable sufficiency
- 10 independent trials for statistical significance
