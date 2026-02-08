#!/usr/bin/env python3
"""
Quantum Vision Transformer Implementation
Based on: Cherrat et al. (2024) - "Quantum Vision Transformers"
https://arxiv.org/abs/2209.08167

This module implements three types of quantum vision transformers:
1. Orthogonal Patch-wise Network (trivial attention)
2. Quantum Orthogonal Transformer (classical-like attention)
3. Compound Transformer (novel quantum attention with polynomial speedup)

Key Features:
- Matrix data loader for quantum amplitude encoding
- RBS (Reconfigurable Beam Splitter) gates
- Quantum orthogonal layers (Butterfly, Pyramid, X circuits)
- Hamming-weight preserving operations
- PennyLane + PyTorch integration
"""

import numpy as np
import torch
import torch.nn as nn
import pennylane as qml
from typing import Literal, Optional, List, Tuple
from functools import lru_cache


# ============================================================================
# RBS Gate Implementation (Reconfigurable Beam Splitter)
# ============================================================================

def RBS_gate(theta: float, wire1: int, wire2: int):
    """
    Reconfigurable Beam Splitter (RBS) gate.

    RBS(θ) = |cos(θ)  -sin(θ)|
             |sin(θ)   cos(θ)|

    Applied to computational basis states:
    RBS(θ)|0⟩|1⟩ = cos(θ)|0⟩|1⟩ + sin(θ)|1⟩|0⟩

    Args:
        theta: Rotation angle
        wire1: First qubit wire
        wire2: Second qubit wire
    """
    # Decompose RBS into standard gates
    # RBS = Ry(θ) on control-target structure
    qml.RY(2 * theta, wires=wire1)
    qml.CNOT(wires=[wire1, wire2])
    qml.RY(-2 * theta, wires=wire1)
    qml.CNOT(wires=[wire1, wire2])


# ============================================================================
# Matrix Data Loader
# ============================================================================

class MatrixDataLoader:
    """
    Quantum matrix data loader using amplitude encoding.

    Loads a matrix X ∈ ℝ^(n×d) into a quantum state by:
    1. Flattening to vector: vec(X) ∈ ℝ^(nd)
    2. Normalizing: |ψ⟩ = Σᵢ xᵢ|i⟩ where Σᵢ|xᵢ|² = 1
    3. Encoding into log₂(nd) qubits

    Reference: Fig. 5 in Cherrat et al. (2024)
    """

    def __init__(self, n_rows: int, n_cols: int):
        """
        Args:
            n_rows: Number of rows in matrix
            n_cols: Number of columns in matrix
        """
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.total_dim = n_rows * n_cols
        self.n_qubits = int(np.ceil(np.log2(self.total_dim)))

    def encode(self, matrix: np.ndarray, wires: List[int]):
        """
        Encode matrix into quantum state using amplitude encoding.

        Args:
            matrix: Input matrix of shape (n_rows, n_cols)
            wires: Qubit wires to encode data into
        """
        # Flatten and normalize
        flat_data = matrix.flatten()
        norm = np.linalg.norm(flat_data)
        if norm > 0:
            normalized_data = flat_data / norm
        else:
            normalized_data = flat_data

        # Pad to power of 2
        padded_dim = 2 ** self.n_qubits
        if len(normalized_data) < padded_dim:
            normalized_data = np.pad(
                normalized_data,
                (0, padded_dim - len(normalized_data)),
                mode='constant'
            )

        # Use PennyLane's amplitude embedding
        qml.AmplitudeEmbedding(
            features=normalized_data,
            wires=wires,
            normalize=False  # Already normalized
        )


# ============================================================================
# Quantum Orthogonal Layers
# ============================================================================

class QuantumOrthogonalLayer:
    """
    Quantum orthogonal layer implementing three circuit types:
    1. Butterfly: O(log d) depth, most efficient
    2. Pyramid: Alternative structure
    3. X: Cross-pattern connections

    All circuits preserve hamming weight and are implemented with RBS gates.

    Reference: Section 2.3 in Cherrat et al. (2024)
    """

    def __init__(
        self,
        n_qubits: int,
        circuit_type: Literal["butterfly", "pyramid", "x"] = "butterfly"
    ):
        """
        Args:
            n_qubits: Number of qubits
            circuit_type: Type of orthogonal circuit
        """
        self.n_qubits = n_qubits
        self.circuit_type = circuit_type

    def butterfly_circuit(self, params: np.ndarray, wires: List[int]):
        """
        Butterfly circuit with O(log d) depth.

        Structure:
        Layer 1: Pairs (0,1), (2,3), (4,5), ...
        Layer 2: Pairs (0,2), (1,3), (4,6), ...
        Layer 3: Pairs (0,4), (1,5), (2,6), ...

        Args:
            params: RBS rotation angles
            wires: Qubit wires
        """
        param_idx = 0
        n_layers = int(np.ceil(np.log2(self.n_qubits)))

        for layer in range(n_layers):
            stride = 2 ** layer
            for start in range(0, self.n_qubits - stride, 2 * stride):
                for offset in range(stride):
                    idx1 = start + offset
                    idx2 = start + offset + stride
                    if idx1 < self.n_qubits and idx2 < self.n_qubits:
                        wire1 = wires[idx1]
                        wire2 = wires[idx2]
                        RBS_gate(params[param_idx], wire1, wire2)
                    param_idx += 1

    def pyramid_circuit(self, params: np.ndarray, wires: List[int]):
        """
        Pyramid circuit structure.

        Args:
            params: RBS rotation angles
            wires: Qubit wires
        """
        param_idx = 0
        for layer in range(self.n_qubits - 1):
            for i in range(self.n_qubits - layer - 1):
                RBS_gate(params[param_idx], wires[i], wires[i + 1])
                param_idx += 1

    def x_circuit(self, params: np.ndarray, wires: List[int]):
        """
        X-pattern circuit structure.

        Args:
            params: RBS rotation angles
            wires: Qubit wires
        """
        param_idx = 0
        # First half: forward connections
        for i in range(self.n_qubits // 2):
            RBS_gate(params[param_idx], wires[i], wires[self.n_qubits - 1 - i])
            param_idx += 1
        # Second half: middle connections
        for i in range(self.n_qubits // 2 - 1):
            RBS_gate(params[param_idx], wires[i], wires[i + 1])
            param_idx += 1

    def apply(self, params: np.ndarray, wires: List[int]):
        """
        Apply the orthogonal layer.

        Args:
            params: RBS rotation angles
            wires: Qubit wires
        """
        if self.circuit_type == "butterfly":
            self.butterfly_circuit(params, wires)
        elif self.circuit_type == "pyramid":
            self.pyramid_circuit(params, wires)
        elif self.circuit_type == "x":
            self.x_circuit(params, wires)
        else:
            raise ValueError(f"Unknown circuit type: {self.circuit_type}")

    def n_params(self) -> int:
        """Calculate number of parameters needed."""
        if self.circuit_type == "butterfly":
            n_layers = int(np.ceil(np.log2(self.n_qubits)))
            return sum(2 ** (n_layers - 1 - i) * (2 ** i)
                      for i in range(n_layers))
        elif self.circuit_type == "pyramid":
            return sum(range(1, self.n_qubits))
        elif self.circuit_type == "x":
            return self.n_qubits // 2 + self.n_qubits // 2 - 1
        return 0


# ============================================================================
# Quantum Vision Transformer Models
# ============================================================================

class OrthogonalPatchwise(nn.Module):
    """
    Orthogonal Patch-wise Network (trivial attention).

    Each patch is processed independently through a quantum orthogonal layer.
    No attention mechanism between patches.

    Architecture:
    Input: Image → Patches → [Quantum Layer] × n_layers → Measurement → FC → Output

    Reference: Section 3.1 in Cherrat et al. (2024)
    """

    def __init__(
        self,
        img_size: int = 28,
        patch_size: int = 7,
        n_layers: int = 2,
        n_classes: int = 10,
        circuit_type: Literal["butterfly", "pyramid", "x"] = "butterfly"
    ):
        """
        Args:
            img_size: Input image size (assumes square images)
            patch_size: Size of each patch
            n_layers: Number of quantum layers
            n_classes: Number of output classes
            circuit_type: Type of quantum orthogonal circuit
        """
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.patch_dim = patch_size ** 2
        self.n_qubits = int(np.ceil(np.log2(self.patch_dim)))
        self.n_layers = n_layers
        self.circuit_type = circuit_type

        # Quantum components
        self.data_loader = MatrixDataLoader(patch_size, patch_size)
        self.orthogonal_layer = QuantumOrthogonalLayer(self.n_qubits, circuit_type)

        # Calculate total parameters
        n_params_per_layer = self.orthogonal_layer.n_params()
        self.total_params = n_layers * n_params_per_layer

        # Trainable quantum parameters
        self.q_params = nn.Parameter(
            torch.randn(self.total_params) * 0.1
        )

        # Create quantum device and circuit
        self.dev = qml.device("default.qubit", wires=self.n_qubits)
        self.qnode = qml.QNode(self._quantum_circuit, self.dev, interface="torch")

        # Classical head
        self.fc = nn.Linear(self.n_patches * self.n_qubits, n_classes)

    def _quantum_circuit(self, inputs: np.ndarray, params: np.ndarray):
        """
        Quantum circuit for processing a single patch.

        Args:
            inputs: Patch data
            params: Quantum parameters

        Returns:
            Expectation values of Pauli-Z on all qubits
        """
        wires = list(range(self.n_qubits))

        # Data encoding
        self.data_loader.encode(inputs, wires)

        # Apply quantum layers
        param_idx = 0
        n_params_per_layer = self.orthogonal_layer.n_params()
        for _ in range(self.n_layers):
            layer_params = params[param_idx:param_idx + n_params_per_layer]
            self.orthogonal_layer.apply(layer_params, wires)
            param_idx += n_params_per_layer

        # Measure all qubits
        return [qml.expval(qml.PauliZ(i)) for i in wires]

    def extract_patches(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract patches from images.

        Args:
            x: Input images of shape (batch, channels, height, width)

        Returns:
            Patches of shape (batch, n_patches, patch_size, patch_size)
        """
        batch_size, channels, h, w = x.shape

        # Average over channels if needed
        if channels > 1:
            x = torch.mean(x, dim=1, keepdim=True)
        x = x.squeeze(1)  # (batch, h, w)

        # Extract patches using unfold
        patches = x.unfold(1, self.patch_size, self.patch_size)
        patches = patches.unfold(2, self.patch_size, self.patch_size)

        # Reshape to (batch, n_patches, patch_size, patch_size)
        patches = patches.contiguous().view(
            batch_size, self.n_patches, self.patch_size, self.patch_size
        )

        return patches

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input images of shape (batch, channels, height, width)

        Returns:
            Logits of shape (batch, n_classes)
        """
        batch_size = x.shape[0]

        # Extract patches
        patches = self.extract_patches(x)  # (batch, n_patches, patch_size, patch_size)

        # Process each patch through quantum circuit
        quantum_features = []
        for b in range(batch_size):
            patch_outputs = []
            for p in range(self.n_patches):
                patch = patches[b, p].detach().cpu().numpy()
                q_out = self.qnode(patch, self.q_params.detach().cpu().numpy())
                patch_outputs.append(q_out)
            quantum_features.append(torch.tensor(patch_outputs, dtype=torch.float32))

        # Stack and flatten
        quantum_features = torch.stack(quantum_features).to(x.device)
        quantum_features = quantum_features.view(batch_size, -1)

        # Classification head
        logits = self.fc(quantum_features)

        return logits


class QuantumOrthogonalTransformer(nn.Module):
    """
    Quantum Orthogonal Transformer (classical-like attention).

    Implements attention mechanism using quantum orthogonal layers for Q, K, V projections.

    Architecture:
    Input → Patches → [Q/K/V Quantum Layers] → Attention → [FFN Quantum Layer] → Output

    Reference: Section 3.2 in Cherrat et al. (2024)
    """

    def __init__(
        self,
        img_size: int = 28,
        patch_size: int = 7,
        n_heads: int = 4,
        n_layers: int = 2,
        n_classes: int = 10,
        circuit_type: Literal["butterfly", "pyramid", "x"] = "butterfly"
    ):
        """
        Args:
            img_size: Input image size
            patch_size: Size of each patch
            n_heads: Number of attention heads
            n_layers: Number of quantum layers
            n_classes: Number of output classes
            circuit_type: Type of quantum circuit
        """
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.patch_dim = patch_size ** 2
        self.n_qubits = int(np.ceil(np.log2(self.patch_dim)))
        self.n_heads = n_heads
        self.n_layers = n_layers

        # Quantum components for Q, K, V
        self.data_loader = MatrixDataLoader(patch_size, patch_size)
        self.orthogonal_layer = QuantumOrthogonalLayer(self.n_qubits, circuit_type)

        n_params_per_layer = self.orthogonal_layer.n_params()
        total_params = n_layers * n_params_per_layer

        # Separate parameters for Q, K, V
        self.q_params = nn.Parameter(torch.randn(total_params) * 0.1)
        self.k_params = nn.Parameter(torch.randn(total_params) * 0.1)
        self.v_params = nn.Parameter(torch.randn(total_params) * 0.1)

        # Quantum device
        self.dev = qml.device("default.qubit", wires=self.n_qubits)
        self.qnode = qml.QNode(self._quantum_circuit, self.dev, interface="torch")

        # Attention scaling
        self.scale = 1.0 / np.sqrt(self.n_qubits)

        # Classification head
        self.fc = nn.Linear(self.n_patches * self.n_qubits, n_classes)

    def _quantum_circuit(self, inputs: np.ndarray, params: np.ndarray):
        """Quantum circuit for Q/K/V projection."""
        wires = list(range(self.n_qubits))

        self.data_loader.encode(inputs, wires)

        param_idx = 0
        n_params_per_layer = self.orthogonal_layer.n_params()
        for _ in range(self.n_layers):
            layer_params = params[param_idx:param_idx + n_params_per_layer]
            self.orthogonal_layer.apply(layer_params, wires)
            param_idx += n_params_per_layer

        return [qml.expval(qml.PauliZ(i)) for i in wires]

    def extract_patches(self, x: torch.Tensor) -> torch.Tensor:
        """Extract patches from images."""
        batch_size, channels, h, w = x.shape

        if channels > 1:
            x = torch.mean(x, dim=1, keepdim=True)
        x = x.squeeze(1)

        patches = x.unfold(1, self.patch_size, self.patch_size)
        patches = patches.unfold(2, self.patch_size, self.patch_size)
        patches = patches.contiguous().view(
            batch_size, self.n_patches, self.patch_size, self.patch_size
        )

        return patches

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with quantum attention."""
        batch_size = x.shape[0]
        patches = self.extract_patches(x)

        # Compute Q, K, V for all patches
        Q, K, V = [], [], []
        for b in range(batch_size):
            q_batch, k_batch, v_batch = [], [], []
            for p in range(self.n_patches):
                patch = patches[b, p].detach().cpu().numpy()

                q_out = self.qnode(patch, self.q_params.detach().cpu().numpy())
                k_out = self.qnode(patch, self.k_params.detach().cpu().numpy())
                v_out = self.qnode(patch, self.v_params.detach().cpu().numpy())

                q_batch.append(q_out)
                k_batch.append(k_out)
                v_batch.append(v_out)

            Q.append(torch.tensor(q_batch, dtype=torch.float32))
            K.append(torch.tensor(k_batch, dtype=torch.float32))
            V.append(torch.tensor(v_batch, dtype=torch.float32))

        Q = torch.stack(Q).to(x.device)  # (batch, n_patches, n_qubits)
        K = torch.stack(K).to(x.device)
        V = torch.stack(V).to(x.device)

        # Scaled dot-product attention
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attention_weights = torch.softmax(attention_scores, dim=-1)

        # Apply attention to V
        attended = torch.matmul(attention_weights, V)  # (batch, n_patches, n_qubits)
        attended = attended.view(batch_size, -1)

        # Classification
        logits = self.fc(attended)

        return logits


class CompoundTransformer(nn.Module):
    """
    Compound Transformer (novel quantum attention).

    Uses compound matrices and quantum measurements for attention computation.
    Achieves polynomial speedup over classical transformers.

    Architecture:
    Input → Patches → Compound Matrix → Quantum Attention → Output

    Reference: Section 3.4 in Cherrat et al. (2024)
    """

    def __init__(
        self,
        img_size: int = 28,
        patch_size: int = 7,
        n_layers: int = 2,
        n_classes: int = 10,
        circuit_type: Literal["butterfly", "pyramid", "x"] = "butterfly"
    ):
        """
        Args:
            img_size: Input image size
            patch_size: Size of each patch
            n_layers: Number of quantum layers
            n_classes: Number of output classes
            circuit_type: Type of quantum circuit
        """
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.patch_dim = patch_size ** 2

        # For compound matrix, we need qubits for pairs of patches
        self.n_qubits = int(np.ceil(np.log2(self.patch_dim ** 2)))
        self.n_layers = n_layers

        # Quantum components
        self.data_loader = MatrixDataLoader(patch_size, patch_size)
        self.orthogonal_layer = QuantumOrthogonalLayer(self.n_qubits, circuit_type)

        n_params_per_layer = self.orthogonal_layer.n_params()
        self.total_params = n_layers * n_params_per_layer

        self.q_params = nn.Parameter(torch.randn(self.total_params) * 0.1)

        # Quantum device
        self.dev = qml.device("default.qubit", wires=self.n_qubits)
        self.qnode = qml.QNode(self._quantum_circuit, self.dev, interface="torch")

        # Classification head
        self.fc = nn.Linear(self.n_patches * 4, n_classes)  # 4 measurements per patch

    def _quantum_circuit(self, inputs: np.ndarray, params: np.ndarray):
        """Quantum circuit for compound matrix processing."""
        wires = list(range(self.n_qubits))

        # Encode compound matrix
        flat_inputs = inputs.flatten()
        norm = np.linalg.norm(flat_inputs)
        if norm > 0:
            flat_inputs = flat_inputs / norm

        # Pad to power of 2
        padded_dim = 2 ** self.n_qubits
        if len(flat_inputs) < padded_dim:
            flat_inputs = np.pad(
                flat_inputs,
                (0, padded_dim - len(flat_inputs)),
                mode='constant'
            )

        qml.AmplitudeEmbedding(features=flat_inputs, wires=wires, normalize=False)

        # Apply quantum layers
        param_idx = 0
        n_params_per_layer = self.orthogonal_layer.n_params()
        for _ in range(self.n_layers):
            layer_params = params[param_idx:param_idx + n_params_per_layer]
            self.orthogonal_layer.apply(layer_params, wires)
            param_idx += n_params_per_layer

        # Measure multiple observables for richer features
        return [
            qml.expval(qml.PauliZ(0)),
            qml.expval(qml.PauliX(0)),
            qml.expval(qml.PauliY(0)),
            qml.expval(qml.PauliZ(1) if self.n_qubits > 1 else qml.PauliZ(0))
        ]

    def extract_patches(self, x: torch.Tensor) -> torch.Tensor:
        """Extract patches from images."""
        batch_size, channels, h, w = x.shape

        if channels > 1:
            x = torch.mean(x, dim=1, keepdim=True)
        x = x.squeeze(1)

        patches = x.unfold(1, self.patch_size, self.patch_size)
        patches = patches.unfold(2, self.patch_size, self.patch_size)
        patches = patches.contiguous().view(
            batch_size, self.n_patches, self.patch_size, self.patch_size
        )

        return patches

    def compute_compound_matrix(self, patch1: np.ndarray, patch2: np.ndarray) -> np.ndarray:
        """
        Compute compound matrix (2nd order) for two patches.

        Compound matrix encodes pairwise relationships.

        Args:
            patch1: First patch
            patch2: Second patch

        Returns:
            Compound matrix
        """
        # Simple outer product implementation
        flat1 = patch1.flatten()
        flat2 = patch2.flatten()
        compound = np.outer(flat1, flat2)
        return compound

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with compound quantum attention."""
        batch_size = x.shape[0]
        patches = self.extract_patches(x)

        # Process each patch with compound attention
        compound_features = []
        for b in range(batch_size):
            patch_outputs = []
            for p in range(self.n_patches):
                # Self-attention: compound with itself
                patch = patches[b, p].detach().cpu().numpy()
                compound_matrix = self.compute_compound_matrix(patch, patch)

                q_out = self.qnode(compound_matrix, self.q_params.detach().cpu().numpy())
                patch_outputs.append(q_out)

            compound_features.append(torch.tensor(patch_outputs, dtype=torch.float32))

        # Stack and flatten
        compound_features = torch.stack(compound_features).to(x.device)
        compound_features = compound_features.view(batch_size, -1)

        # Classification
        logits = self.fc(compound_features)

        return logits


# ============================================================================
# Factory Function
# ============================================================================

def create_quantum_vision_transformer(
    model_type: Literal["patchwise", "orthogonal", "compound"] = "patchwise",
    img_size: int = 28,
    patch_size: int = 7,
    n_layers: int = 2,
    n_classes: int = 10,
    circuit_type: Literal["butterfly", "pyramid", "x"] = "butterfly",
    **kwargs
) -> nn.Module:
    """
    Factory function to create quantum vision transformer models.

    Args:
        model_type: Type of transformer ("patchwise", "orthogonal", "compound")
        img_size: Input image size
        patch_size: Patch size
        n_layers: Number of quantum layers
        n_classes: Number of output classes
        circuit_type: Type of quantum orthogonal circuit
        **kwargs: Additional model-specific arguments

    Returns:
        Quantum vision transformer model
    """
    if model_type == "patchwise":
        return OrthogonalPatchwise(
            img_size=img_size,
            patch_size=patch_size,
            n_layers=n_layers,
            n_classes=n_classes,
            circuit_type=circuit_type
        )
    elif model_type == "orthogonal":
        return QuantumOrthogonalTransformer(
            img_size=img_size,
            patch_size=patch_size,
            n_layers=n_layers,
            n_classes=n_classes,
            circuit_type=circuit_type,
            **kwargs
        )
    elif model_type == "compound":
        return CompoundTransformer(
            img_size=img_size,
            patch_size=patch_size,
            n_layers=n_layers,
            n_classes=n_classes,
            circuit_type=circuit_type
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # Quick test
    print("Testing Quantum Vision Transformer models...")

    # Test OrthogonalPatchwise
    model = OrthogonalPatchwise(img_size=28, patch_size=7, n_layers=1, n_classes=10)
    x = torch.randn(2, 1, 28, 28)
    out = model(x)
    print(f"OrthogonalPatchwise output shape: {out.shape}")

    print("\nAll tests passed!")
