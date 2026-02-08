import math

import torch
import pennylane as qml
from math import ceil, log2

# --------------------------------------------------------------------------------
# PennyLane Quantum Circuit — sim14 ansatz (Sim et al., 2019)
# --------------------------------------------------------------------------------
def sim14_circuit(params, wires, layers=1):
    """
    Implements the 'sim14' circuit from Sim et al. (2019) using PennyLane.
    This function is batch-aware and handles both 1D and 2D parameter tensors.

    Gate sequence per layer: RY → CRX(ring) → RY → CRX(counter-ring)
    Parameters per layer: 4 * wires
    """
    is_batched = params.ndim == 2
    param_idx = 0
    for _ in range(layers):
        for i in range(wires):
            angle = params[:, param_idx] if is_batched else params[param_idx]
            qml.RY(angle, wires=i)
            param_idx += 1
        for i in range(wires - 1, -1, -1):
            angle = params[:, param_idx] if is_batched else params[param_idx]
            qml.CRX(angle, wires=[i, (i + 1) % wires])
            param_idx += 1
        for i in range(wires):
            angle = params[:, param_idx] if is_batched else params[param_idx]
            qml.RY(angle, wires=i)
            param_idx += 1
        wire_order = [wires - 1] + list(range(wires - 1))
        for i in wire_order:
            angle = params[:, param_idx] if is_batched else params[param_idx]
            qml.CRX(angle, wires=[i, (i - 1) % wires])
            param_idx += 1


# --------------------------------------------------------------------------------
# Quantum Time-Series Transformer — Hardware-Compatible QSVT via LCU
#
# Architecture:
#   feature_proj(x_t) → sigmoid →
#     Single QNode:
#       PCPhase(φ₀)                              ← signal processing
#       for k in range(degree):
#           PREPARE (learnable V on ancilla)      ← encodes LCU coefficients
#           SELECT([U(x₀),...,U(x_T)])            ← data-dependent unitaries
#           PREPARE†
#           PCPhase(φ_{k+1})                      ← signal processing
#       QFF sim14 on main register
#       → measure PauliX/Y/Z on main register
#
# The LCU (PREPARE·SELECT·PREPARE†) IS the block encoding.
# QSVT alternates this block encoding with PCPhase signal processing.
# No statevector extraction, no classical O(2^n) operations, no StatePrep.
# --------------------------------------------------------------------------------

class QuantumTSTransformer(torch.nn.Module):
    def __init__(self,
                 n_qubits: int,
                 n_timesteps: int,
                 degree: int,
                 n_ansatz_layers: int,
                 feature_dim: int,
                 output_dim: int,
                 dropout: float,
                 device,
                 chunk_size: int = None):
        super().__init__()

        self.n_qubits = n_qubits
        self.n_timesteps = n_timesteps
        self.degree = degree
        self.n_ansatz_layers = n_ansatz_layers
        self.device = device

        # ── Temporal chunking ──
        # When chunk_size is set, the sequence is split into non-overlapping
        # windows, each processed by its own QSVT+LCU pass.  This keeps
        # n_ancilla = ceil(log2(chunk_size)) instead of ceil(log2(T)),
        # e.g. chunk_size=16 → 4 ancilla regardless of total sequence length.
        self.chunk_size = chunk_size or n_timesteps
        self.n_chunks = ceil(n_timesteps / self.chunk_size)

        # ── Qubit registers (sized by chunk, not full sequence) ──
        self.n_ancilla = ceil(log2(max(self.chunk_size, 2)))
        self.main_wires = list(range(n_qubits))
        self.anc_wires = list(range(n_qubits, n_qubits + self.n_ancilla))
        self.total_wires = n_qubits + self.n_ancilla
        self.n_select_ops = 2 ** self.n_ancilla

        # ── Parameter counts ──
        self.n_rots = 4 * n_qubits * n_ansatz_layers
        self.qff_n_rots = 4 * n_qubits * 1

        # ── Classical Layers (same interface as original) ──
        self.feature_projection = torch.nn.Linear(feature_dim, self.n_rots)
        self.dropout = torch.nn.Dropout(dropout)
        self.rot_sigm = torch.nn.Sigmoid()
        self.output_ff = torch.nn.Linear(3 * n_qubits, output_dim)

        # ── Sinusoidal Positional Encoding (Vaswani et al., 2017) ──
        pe = torch.zeros(n_timesteps, feature_dim)
        pos = torch.arange(n_timesteps).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, feature_dim, 2).float()
            * -(math.log(10000.0) / feature_dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[:feature_dim // 2])
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, T, feature_dim)

        # ── Trainable Quantum Parameters ──
        # PREPARE ansatz on ancilla (replaces mix_coeffs)
        self.n_prep_layers = self.n_ancilla
        self.prepare_params = torch.nn.Parameter(
            0.1 * torch.randn(self.n_prep_layers, self.n_ancilla, 2))

        # QSVT signal processing angles (replaces poly_coeffs)
        self.signal_angles = torch.nn.Parameter(
            0.1 * torch.randn(degree + 1))

        # QFF parameters (unchanged role)
        self.qff_params = torch.nn.Parameter(torch.rand(self.qff_n_rots))

        # ── PennyLane Device and QNode ──
        self.dev = qml.device("default.qubit", wires=self.total_wires)

        # Capture instance attributes as locals for the QNode closure
        _n_qubits = n_qubits
        _chunk_size = self.chunk_size
        _n_ancilla = self.n_ancilla
        _n_ansatz_layers = n_ansatz_layers
        _n_prep_layers = self.n_prep_layers
        _main_wires = self.main_wires
        _anc_wires = self.anc_wires
        _degree = degree
        _n_select_ops = self.n_select_ops
        _pcphase_wires = self.anc_wires + self.main_wires
        _pcphase_dim = 2 ** n_qubits

        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def _circuit(ts_params, prep_p, sig_ang, qff_p):
            """
            Single coherent quantum circuit implementing QSVT via LCU.

            Args:
                ts_params: (B, chunk_size, n_rots) — one chunk of rotation params
                prep_p:    (n_prep_layers, n_ancilla, 2) — PREPARE ansatz params
                sig_ang:   (degree+1,) — signal processing angles
                qff_p:     (qff_n_rots,) — QFF circuit params
            """

            def prepare():
                """Learnable PREPARE unitary V on ancilla register."""
                for ly in range(_n_prep_layers):
                    for qi, q in enumerate(_anc_wires):
                        qml.RY(prep_p[ly, qi, 0], wires=q)
                        qml.RZ(prep_p[ly, qi, 1], wires=q)
                    for i in range(_n_ancilla - 1):
                        qml.CNOT(wires=[_anc_wires[i], _anc_wires[i + 1]])

            def build_select_ops():
                """Build list of sim14 unitaries for qml.Select.

                Each timestep t in the chunk gets a sim14 operator
                parameterized by ts_params[..., t, :].
                Padded to 2^n_ancilla with Identity.
                """
                select_ops = []
                for t in range(_chunk_size):
                    gates = []
                    param_idx = 0
                    for _ in range(_n_ansatz_layers):
                        # RY layer
                        for i in range(_n_qubits):
                            gates.append(qml.RY(
                                ts_params[..., t, param_idx],
                                wires=_main_wires[i]))
                            param_idx += 1
                        # CRX ring (reverse order)
                        for i in range(_n_qubits - 1, -1, -1):
                            gates.append(qml.CRX(
                                ts_params[..., t, param_idx],
                                wires=[_main_wires[i],
                                       _main_wires[(i + 1) % _n_qubits]]))
                            param_idx += 1
                        # RY layer
                        for i in range(_n_qubits):
                            gates.append(qml.RY(
                                ts_params[..., t, param_idx],
                                wires=_main_wires[i]))
                            param_idx += 1
                        # CRX counter-ring
                        wire_order = [_n_qubits - 1] + list(range(_n_qubits - 1))
                        for i in wire_order:
                            gates.append(qml.CRX(
                                ts_params[..., t, param_idx],
                                wires=[_main_wires[i],
                                       _main_wires[(i - 1) % _n_qubits]]))
                            param_idx += 1
                    # qml.prod applies right-to-left; reversed gives correct order
                    select_ops.append(qml.prod(*reversed(gates)))
                # Pad to 2^n_ancilla with Identity
                while len(select_ops) < _n_select_ops:
                    select_ops.append(qml.Identity(wires=_main_wires[0]))
                return select_ops

            # ── Build SELECT operators once (reused across QSVT iterations) ──
            select_ops = build_select_ops()

            # ── QSVT: alternating signal processing and LCU block encoding ──
            qml.PCPhase(sig_ang[0], dim=_pcphase_dim, wires=_pcphase_wires)

            for k in range(_degree):
                # LCU block encoding: PREPARE · SELECT · PREPARE†
                prepare()
                if k % 2 == 0:
                    qml.Select(select_ops, control=_anc_wires)
                else:
                    qml.adjoint(qml.Select)(select_ops, control=_anc_wires)
                qml.adjoint(prepare)()
                qml.PCPhase(sig_ang[k + 1], dim=_pcphase_dim,
                            wires=_pcphase_wires)

            # ── QFF on main register ──
            sim14_circuit(qff_p, wires=_n_qubits, layers=1)

            # ── Measure PauliX/Y/Z on main register ──
            observables = (
                [qml.PauliX(i) for i in _main_wires] +
                [qml.PauliY(i) for i in _main_wires] +
                [qml.PauliZ(i) for i in _main_wires])
            return [qml.expval(op) for op in observables]

        self._circuit = _circuit

    def forward(self, x):
        # x: (batch, feature_dim, n_timesteps)
        B = x.size(0)
        x = x.permute(0, 2, 1)                       # (B, T, feature_dim)
        x = x + self.pe[:, :x.size(1)]                # sinusoidal PE
        x = self.feature_projection(self.dropout(x))
        ts_params = self.rot_sigm(x) * (2 * math.pi)  # (B, T, n_rots) in [0, 2π]

        # ── Process chunks ──
        chunk_results = []
        for start in range(0, self.n_timesteps, self.chunk_size):
            chunk = ts_params[:, start:start + self.chunk_size]
            # Pad last chunk to chunk_size with zeros (→ identity-like rotations)
            pad_len = self.chunk_size - chunk.size(1)
            if pad_len > 0:
                chunk = torch.nn.functional.pad(chunk, (0, 0, 0, pad_len))
            exps = self._circuit(
                chunk, self.prepare_params,
                self.signal_angles, self.qff_params)
            chunk_results.append(torch.stack(exps, dim=1).float())

        # Aggregate chunks via mean pooling → (B, 3*n_qubits)
        exps = torch.stack(chunk_results, dim=0).mean(dim=0)
        return self.output_ff(exps).squeeze(1)
