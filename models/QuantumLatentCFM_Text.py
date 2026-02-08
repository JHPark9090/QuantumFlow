"""
Quantum Latent Conditional Flow Matching for Text (QLCFM-Text)
================================================================

Generative text model combining:
  1. MambaTextVAE (Embedding -> Mamba SSM blocks -> latent) for character-level
     encoding, based on Gu & Dao (2023) Selective State Space Models
  2. QSVTVelocityField (QSVT via LCU + QFF + ANO) for latent flow matching,
     using the QSVT block-encoding pipeline from ModularQTS
  3. Conditional Flow Matching (OT-CFM) training objective

Architecture:
  Phase 1 -- MambaTextVAE pretraining:
    text8 sequence: (B, seq_len) int tokens (0-26)
      -> nn.Embedding(27, embed_dim) -> (B, seq_len, embed_dim)
      -> MambaBlock layers (selective SSM + gating)
      -> mean pool -> Linear -> mu, logvar -> z (latent_dim)
      -> Decoder: z -> repeat -> MambaBlock layers -> Linear(d_model, 27)
    Loss = CrossEntropy(logits, tokens) + beta * KL

  Phase 2 -- Quantum CFM:
    z_1 = MambaTextVAE.encode(x).mu (frozen)
    z_0 ~ N(0, I)
    z_t = (1-t)*z_0 + t*z_1
    v_theta(z_t, t) = QSVTVelocityField(z_t, t)
    Loss = MSE(v_theta, z_1 - z_0)

  Generation:
    z_0 ~ N(0, I) -> Euler ODE (t: 0->1) -> z_1 -> MambaTextVAE.decode -> text

Target benchmark: text8 (character-level, 27-char vocab: space + a-z)

Usage:
  # Both phases:
  python QuantumLatentCFM_Text.py --phase=both --n-qubits=8 \
      --degree=2 --n-layers=2 --n-select=4 --k-local=2 \
      --latent-dim=128 --seq-len=256 --epochs=100

  # Generate:
  python QuantumLatentCFM_Text.py --phase=generate --n-qubits=8 \
      --degree=2 --n-layers=2 --n-select=4 --k-local=2 \
      --vae-ckpt=<path> --cfm-ckpt=<path>
"""

import argparse
import os
import sys
import random
import copy
import math
import time
import csv
import zipfile
import urllib.request
from math import ceil, log2
from itertools import combinations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm
import scipy.constants  # noqa: F401 -- pre-import for PennyLane/scipy compat
import pennylane as qml

DATA_ROOT = "/pscratch/sd/j/junghoon/data"

# Character mapping for text8: space + a-z = 27 chars
VOCAB_SIZE = 27
CHAR2IDX = {' ': 0}
CHAR2IDX.update({chr(ord('a') + i): i + 1 for i in range(26)})
IDX2CHAR = {v: k for k, v in CHAR2IDX.items()}


# ---------------------------------------------------------------------------
# 1. Argparse
# ---------------------------------------------------------------------------
def get_args():
    p = argparse.ArgumentParser(description="Quantum Latent CFM for Text")

    # Phase
    p.add_argument("--phase", type=str, default="both",
                   choices=["1", "2", "generate", "both"],
                   help="1=VAE pretrain, 2=CFM train, generate=sample, both=1+2")

    # MambaTextVAE
    p.add_argument("--latent-dim", type=int, default=128)
    p.add_argument("--embed-dim", type=int, default=64)
    p.add_argument("--hidden-dim", type=int, default=256,
                   help="Mamba d_model for VAE blocks")
    p.add_argument("--n-mamba-layers", type=int, default=2,
                   help="Number of Mamba blocks in encoder/decoder")
    p.add_argument("--d-state", type=int, default=16,
                   help="SSM state dimension")
    p.add_argument("--d-conv", type=int, default=4,
                   help="1D conv kernel size in Mamba blocks")
    p.add_argument("--expand", type=int, default=2,
                   help="Mamba inner dimension expansion factor")
    p.add_argument("--seq-len", type=int, default=256,
                   help="Character sequence length")
    p.add_argument("--beta", type=float, default=0.5,
                   help="KL weight in VAE loss")

    # QSVTVelocityField
    p.add_argument("--n-qubits", type=int, default=8)
    p.add_argument("--degree", type=int, default=2,
                   help="QSVT polynomial degree")
    p.add_argument("--n-layers", type=int, default=2,
                   help="sim14 ansatz layers per SELECT unitary")
    p.add_argument("--n-select", type=int, default=4,
                   help="Number of virtual timesteps (SELECT unitaries)")
    p.add_argument("--k-local", type=int, default=2)
    p.add_argument("--obs-scheme", type=str, default="sliding",
                   choices=["sliding", "pairwise"])

    # Training
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lr-H", type=float, default=1e-1,
                   help="LR for ANO params (Chen 2025: 100x)")
    p.add_argument("--lr-vae", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--seed", type=int, default=2025)

    # Data
    p.add_argument("--n-train", type=int, default=10000)
    p.add_argument("--n-valtest", type=int, default=2000)

    # Generation
    p.add_argument("--ode-steps", type=int, default=100)
    p.add_argument("--n-samples", type=int, default=16)
    p.add_argument("--temperature", type=float, default=1.0,
                   help="Generation sampling temperature")

    # I/O
    p.add_argument("--job-id", type=str, default="qlcfm_text_001")
    p.add_argument("--base-path", type=str, default=".")
    p.add_argument("--vae-ckpt", type=str, default="")
    p.add_argument("--cfm-ckpt", type=str, default="")
    p.add_argument("--resume", action="store_true")

    return p.parse_args()


# ---------------------------------------------------------------------------
# 2. Utilities
# ---------------------------------------------------------------------------
def set_all_seeds(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    qml.numpy.random.seed(seed)


def create_Hermitian(N, A, B, D):
    """Build N x N Hermitian from learnable real params (Lin et al., 2025)."""
    h = torch.zeros((N, N), dtype=torch.complex128)
    count = 0
    for i in range(1, N):
        h[i - 1, i - 1] = D[i].clone()
        for j in range(i):
            h[i, j] = A[count + j].clone() + 1j * B[count + j].clone()
        count += i
    return h.clone() + h.clone().conj().T


def get_wire_groups(n_qubits, k_local, obs_scheme):
    """Wire groups for k-local observable measurement."""
    if k_local <= 0:
        return [[q] for q in range(n_qubits)]
    if obs_scheme == "sliding":
        return [list(range(s, s + k_local))
                for s in range(n_qubits - k_local + 1)]
    elif obs_scheme == "pairwise":
        return [list(c) for c in combinations(range(n_qubits), k_local)]
    raise ValueError(f"Unknown obs_scheme: {obs_scheme}")


def sinusoidal_embedding(t, dim):
    """Sinusoidal time embedding for CFM.

    Args:
        t: (B,) time values in [0, 1]
        dim: embedding dimension
    Returns:
        (B, dim) sinusoidal embedding
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=t.device).float() / half
    )
    args = t[:, None].float() * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


def sim14_circuit(params, wires, layers=1):
    """sim14 ansatz: RY -> CRX(ring) -> RY -> CRX(counter-ring).
    Parameters per layer: 4 * wires.
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


# ---------------------------------------------------------------------------
# 3. text8 Data Loader
# ---------------------------------------------------------------------------
def load_text8(seq_len, n_train, n_valtest, batch_size):
    """Load text8 dataset and create DataLoaders."""
    cache_dir = os.path.join(DATA_ROOT, "text8")
    os.makedirs(cache_dir, exist_ok=True)
    text8_path = os.path.join(cache_dir, "text8")

    if not os.path.exists(text8_path):
        zip_path = os.path.join(cache_dir, "text8.zip")
        url = "http://mattmahoney.net/dc/text8.zip"
        print(f"Downloading text8 from {url}...")
        urllib.request.urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(cache_dir)
        os.remove(zip_path)
        print(f"Extracted to {text8_path}")

    with open(text8_path, 'r') as f:
        text = f.read().strip()

    text = ''.join(c for c in text if c in CHAR2IDX)
    print(f"text8: {len(text):,} characters")

    data = np.array([CHAR2IDX[c] for c in text], dtype=np.int64)

    total_needed = (n_train + n_valtest) * seq_len
    if total_needed > len(data):
        total_needed = len(data)
        total_chunks = total_needed // seq_len
        n_train = min(n_train, int(total_chunks * 0.8))
        n_valtest = total_chunks - n_train
        print(f"Adjusted: n_train={n_train}, n_valtest={n_valtest}")

    n_chunks = total_needed // seq_len
    data = data[:n_chunks * seq_len].reshape(n_chunks, seq_len)
    data = torch.from_numpy(data).long()

    perm = torch.randperm(n_chunks)
    data = data[perm]

    X_train = data[:n_train]
    X_valtest = data[n_train:n_train + n_valtest]

    train_ds = TensorDataset(X_train)
    valtest_ds = TensorDataset(X_valtest)
    val_sz = len(valtest_ds) // 2
    test_sz = len(valtest_ds) - val_sz
    val_ds, test_ds = random_split(valtest_ds, [val_sz, test_sz])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    print(f"text8 splits: train={len(X_train)}  val={val_sz}  test={test_sz}  "
          f"seq_len={seq_len}")

    return train_loader, val_loader, test_loader


# ---------------------------------------------------------------------------
# 4. Mamba Components (Gu & Dao, 2023)
# ---------------------------------------------------------------------------
class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return self.weight * (x / rms)


class SelectiveSSM(nn.Module):
    """Selective State Space Model core.

    Implements discrete-time SSM with input-dependent B, C, dt:
        x[t] = A_d * x[t-1] + B_d * u[t]
        y[t] = C * x[t] + D * u[t]

    Based on the official Mamba implementation (state-spaces/mamba).
    Uses sequential scan (pure PyTorch, no CUDA kernel).
    """

    def __init__(self, d_model, d_state=16, dt_rank='auto',
                 dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        if dt_rank == 'auto':
            dt_rank = math.ceil(d_model / 16)
        self.dt_rank = dt_rank

        # A: diagonal state matrix (log-parameterized for stability)
        A = torch.arange(1, d_state + 1, dtype=torch.float32
                         ).unsqueeze(0).expand(d_model, -1)
        self.A_log = nn.Parameter(torch.log(A))

        # D: skip connection
        self.D = nn.Parameter(torch.ones(d_model))

        # dt projection with softplus initialization (official Mamba pattern)
        self.dt_proj = nn.Linear(dt_rank, d_model, bias=True)
        dt_init_std = dt_rank ** -0.5
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)

        dt = torch.exp(
            torch.rand(d_model) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

    def forward(self, x, dt, B, C):
        """Sequential selective SSM scan.

        Args:
            x:  (B, L, d_model)
            dt: (B, L, d_model)
            B:  (B, L, d_state)
            C:  (B, L, d_state)
        Returns:
            y:  (B, L, d_model)
        """
        batch, seq_len, d_model = x.shape

        A = -torch.exp(self.A_log.float())  # (d_model, d_state)
        dt_exp = dt.unsqueeze(-1)  # (B, L, d_model, 1)

        # ZOH discretization: A_d = exp(A * dt)
        A_disc = torch.exp(A.unsqueeze(0).unsqueeze(0) * dt_exp)
        B_disc = dt_exp * B.unsqueeze(2)  # (B, L, d_model, d_state)

        state = torch.zeros(batch, d_model, self.d_state,
                            device=x.device, dtype=x.dtype)
        outputs = []

        for t in range(seq_len):
            state = A_disc[:, t] * state + x[:, t].unsqueeze(-1) * B_disc[:, t]
            y_t = torch.sum(C[:, t].unsqueeze(1) * state, dim=-1)
            outputs.append(y_t)

        y = torch.stack(outputs, dim=1)
        y = y + x * self.D.unsqueeze(0).unsqueeze(0)
        return y


class MambaBlock(nn.Module):
    """Mamba block: in_proj -> split(x,z) -> conv1d -> SiLU -> SSM -> gate -> out_proj.

    Following the official architecture from state-spaces/mamba:
    - in_proj: Linear(d_model, d_inner * 2) for x and z (gate) branches
    - conv1d: depthwise 1D convolution on x branch for local context
    - x_proj: Linear(d_inner, dt_rank + d_state * 2) for dt, B, C
    - dt_proj: inside SSM, Linear(dt_rank, d_inner) with softplus
    - Gating: y * SiLU(z)
    - out_proj: Linear(d_inner, d_model) + residual
    """

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dt_rank='auto'):
        super().__init__()
        self.d_model = d_model
        self.d_inner = int(expand * d_model)

        if dt_rank == 'auto':
            self.dt_rank = math.ceil(d_model / 16)
        else:
            self.dt_rank = dt_rank

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.x_proj = nn.Linear(self.d_inner,
                                self.dt_rank + 2 * d_state, bias=False)
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner, kernel_size=d_conv,
            groups=self.d_inner, padding=d_conv - 1, bias=True)

        self.ssm = SelectiveSSM(
            d_model=self.d_inner, d_state=d_state, dt_rank=self.dt_rank)

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.norm = RMSNorm(d_model)

    def forward(self, u):
        """Args: u (B, L, d_model). Returns: (B, L, d_model)."""
        residual = u
        u_norm = self.norm(u)

        xz = self.in_proj(u_norm)
        x, z = xz.chunk(2, dim=-1)  # (B, L, d_inner)

        # 1D conv + SiLU
        x_conv = self.conv1d(x.transpose(1, 2))[:, :, :u.size(1)]
        x = F.silu(x_conv.transpose(1, 2))

        # Project to dt, B, C
        x_proj = self.x_proj(x)
        dt, B, C = torch.split(
            x_proj,
            [self.dt_rank, self.ssm.d_state, self.ssm.d_state],
            dim=-1)

        dt = F.softplus(self.ssm.dt_proj(dt))

        # Selective SSM
        y = self.ssm(x, dt, B, C)

        # Gating
        y = y * F.silu(z)

        return self.out_proj(y) + residual


# ---------------------------------------------------------------------------
# 5. MambaTextVAE
# ---------------------------------------------------------------------------
class MambaTextVAE(nn.Module):
    """Mamba-based VAE for character-level text.

    Encoder: Embedding -> MambaBlocks -> mean pool -> mu, logvar
    Decoder: z -> Linear -> repeat to seq_len -> MambaBlocks -> Linear -> logits
    """

    def __init__(self, vocab_size=27, embed_dim=64, d_model=256,
                 latent_dim=128, n_layers=2, seq_len=256,
                 d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.latent_dim = latent_dim
        self.seq_len = seq_len

        # Encoder
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.enc_proj = nn.Linear(embed_dim, d_model)
        self.encoder_blocks = nn.ModuleList([
            MambaBlock(d_model, d_state=d_state, d_conv=d_conv, expand=expand)
            for _ in range(n_layers)
        ])
        self.enc_norm = RMSNorm(d_model)
        self.fc_mu = nn.Linear(d_model, latent_dim)
        self.fc_logvar = nn.Linear(d_model, latent_dim)

        # Decoder
        self.fc_dec = nn.Linear(latent_dim, d_model)
        self.decoder_blocks = nn.ModuleList([
            MambaBlock(d_model, d_state=d_state, d_conv=d_conv, expand=expand)
            for _ in range(n_layers)
        ])
        self.dec_norm = RMSNorm(d_model)
        self.output_proj = nn.Linear(d_model, vocab_size)

    def encode(self, x):
        """Encode token sequence to latent distribution.

        Args:
            x: (B, L) int token IDs
        Returns:
            mu: (B, latent_dim), logvar: (B, latent_dim)
        """
        h = self.enc_proj(self.embedding(x))  # (B, L, d_model)
        for block in self.encoder_blocks:
            h = block(h)
        h = self.enc_norm(h)
        h = h.mean(dim=1)  # (B, d_model) -- mean pool
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def decode(self, z):
        """Decode latent vector to token logits.

        Args:
            z: (B, latent_dim)
        Returns:
            logits: (B, seq_len, vocab_size)
        """
        h = F.silu(self.fc_dec(z))                         # (B, d_model)
        h = h.unsqueeze(1).expand(-1, self.seq_len, -1)    # (B, L, d_model)
        for block in self.decoder_blocks:
            h = block(h)
        h = self.dec_norm(h)
        return self.output_proj(h)                          # (B, L, vocab_size)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        logits = self.decode(z)
        return logits, mu, logvar


def decode_tokens(tokens):
    """Convert int tensor to string."""
    if isinstance(tokens, torch.Tensor):
        tokens = tokens.cpu().numpy()
    return ''.join(IDX2CHAR.get(int(t), '?') for t in tokens)


# ---------------------------------------------------------------------------
# 6. QSVTVelocityField
# ---------------------------------------------------------------------------
class QSVTVelocityField(nn.Module):
    """Velocity field for CFM using QSVT via LCU + QFF + ANO.

    Architecture:
      (z_t, t) -> time embed -> concat -> FC -> (B, n_select, n_rots) -> sigmoid
        -> QSVT circuit:
            PCPhase(phi_0)
            for k in range(degree):
                PREPARE (learnable V on ancilla)
                SELECT([U_0(x_0), ..., U_{n_select-1}(x_{n_select-1})])
                PREPARE_dag
                PCPhase(phi_{k+1})
            QFF sim14 on main register
            -> ANO measurement
        -> Linear(n_obs, latent_dim) -> velocity

    The n_select parameter controls the LCU decomposition complexity.
    Each SELECT unitary is a sim14 ansatz parameterized by the projected
    latent+time input ("virtual timesteps").
    """

    def __init__(self, latent_dim, n_qubits, n_select, degree,
                 n_ansatz_layers, k_local, obs_scheme, device):
        super().__init__()

        self.latent_dim = latent_dim
        self.n_qubits = n_qubits
        self.n_select = n_select
        self.degree = degree
        self.n_ansatz_layers = n_ansatz_layers
        self.k_local = k_local

        # Qubit registers
        self.n_ancilla = ceil(log2(max(n_select, 2)))
        self.main_wires = list(range(n_qubits))
        self.anc_wires = list(range(n_qubits, n_qubits + self.n_ancilla))
        self.total_wires = n_qubits + self.n_ancilla
        self.n_select_ops = 2 ** self.n_ancilla

        # Parameter counts
        self.n_rots = 4 * n_qubits * n_ansatz_layers
        self.qff_n_rots = 4 * n_qubits * 1

        # Classical projection: (z_t, t_emb) -> (B, n_select, n_rots)
        self.time_embed = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
        )
        self.input_proj = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim * 2),
            nn.SiLU(),
            nn.Linear(latent_dim * 2, n_select * self.n_rots),
        )
        self.rot_sigm = nn.Sigmoid()

        # PREPARE ansatz on ancilla
        self.n_prep_layers = self.n_ancilla
        self.prepare_params = nn.Parameter(
            0.1 * torch.randn(self.n_prep_layers, self.n_ancilla, 2))

        # QSVT signal processing angles
        self.signal_angles = nn.Parameter(0.1 * torch.randn(degree + 1))

        # QFF parameters (single sim14 layer on main register)
        self.qff_params = nn.Parameter(torch.rand(self.qff_n_rots))

        # ANO parameters (Chen et al., 2025; Lin et al., 2025)
        self.wire_groups = get_wire_groups(n_qubits, k_local, obs_scheme)
        self.n_obs = len(self.wire_groups)

        if k_local > 0:
            K = 2 ** k_local
            self.obs_dim = K
            n_off = (K * (K - 1)) // 2
            self.A = nn.ParameterList(
                [nn.Parameter(torch.empty(n_off)) for _ in range(self.n_obs)])
            self.B = nn.ParameterList(
                [nn.Parameter(torch.empty(n_off)) for _ in range(self.n_obs)])
            self.D_obs = nn.ParameterList(
                [nn.Parameter(torch.empty(K)) for _ in range(self.n_obs)])
            for w in range(self.n_obs):
                nn.init.normal_(self.A[w], std=2.0)
                nn.init.normal_(self.B[w], std=2.0)
                nn.init.normal_(self.D_obs[w], std=2.0)
        else:
            self.obs_dim = 0

        # Output head: n_obs -> latent_dim
        self.head = nn.Linear(self.n_obs, latent_dim)

        # PennyLane device and QNode
        self.dev = qml.device("default.qubit", wires=self.total_wires)

        # Capture instance attributes for QNode closure
        _n_qubits = n_qubits
        _n_select = n_select
        _n_ancilla = self.n_ancilla
        _n_ansatz_layers = n_ansatz_layers
        _n_prep_layers = self.n_prep_layers
        _main_wires = self.main_wires
        _anc_wires = self.anc_wires
        _degree = degree
        _n_select_ops = self.n_select_ops
        _pcphase_wires = self.anc_wires + self.main_wires
        _pcphase_dim = 2 ** n_qubits
        _kl = k_local
        _wg = self.wire_groups
        _no = self.n_obs

        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def _circuit(ts_params, prep_p, sig_ang, qff_p, H_mats):
            """QSVT via LCU + ANO measurement.

            Args:
                ts_params: (B, n_select, n_rots) circuit rotation params
                prep_p:    (n_prep_layers, n_ancilla, 2) PREPARE params
                sig_ang:   (degree+1,) signal processing angles
                qff_p:     (qff_n_rots,) QFF circuit params
                H_mats:    list of Hermitian matrices for ANO
            """

            def prepare():
                for ly in range(_n_prep_layers):
                    for qi, q in enumerate(_anc_wires):
                        qml.RY(prep_p[ly, qi, 0], wires=q)
                        qml.RZ(prep_p[ly, qi, 1], wires=q)
                    for i in range(_n_ancilla - 1):
                        qml.CNOT(wires=[_anc_wires[i], _anc_wires[i + 1]])

            def build_select_ops():
                select_ops = []
                for s in range(_n_select):
                    gates = []
                    param_idx = 0
                    for _ in range(_n_ansatz_layers):
                        # RY layer
                        for i in range(_n_qubits):
                            gates.append(qml.RY(
                                ts_params[..., s, param_idx],
                                wires=_main_wires[i]))
                            param_idx += 1
                        # CRX ring (reverse order)
                        for i in range(_n_qubits - 1, -1, -1):
                            gates.append(qml.CRX(
                                ts_params[..., s, param_idx],
                                wires=[_main_wires[i],
                                       _main_wires[(i + 1) % _n_qubits]]))
                            param_idx += 1
                        # RY layer
                        for i in range(_n_qubits):
                            gates.append(qml.RY(
                                ts_params[..., s, param_idx],
                                wires=_main_wires[i]))
                            param_idx += 1
                        # CRX counter-ring
                        wire_order = ([_n_qubits - 1]
                                      + list(range(_n_qubits - 1)))
                        for i in wire_order:
                            gates.append(qml.CRX(
                                ts_params[..., s, param_idx],
                                wires=[_main_wires[i],
                                       _main_wires[(i - 1) % _n_qubits]]))
                            param_idx += 1
                    select_ops.append(qml.prod(*reversed(gates)))
                # Pad to 2^n_ancilla with Identity
                while len(select_ops) < _n_select_ops:
                    select_ops.append(qml.Identity(wires=_main_wires[0]))
                return select_ops

            # QSVT: alternating signal processing and LCU block encoding
            select_ops = build_select_ops()  # build once, reuse

            qml.PCPhase(sig_ang[0], dim=_pcphase_dim,
                        wires=_pcphase_wires)

            for k in range(_degree):
                prepare()
                if k % 2 == 0:
                    qml.Select(select_ops, control=_anc_wires)
                else:
                    qml.adjoint(qml.Select)(select_ops, control=_anc_wires)
                qml.adjoint(prepare)()
                qml.PCPhase(sig_ang[k + 1], dim=_pcphase_dim,
                            wires=_pcphase_wires)

            # QFF on main register
            sim14_circuit(qff_p, wires=_n_qubits, layers=1)

            # Measurement: ANO or fixed PauliZ
            if _kl > 0:
                return [qml.expval(qml.Hermitian(H_mats[w], wires=_wg[w]))
                        for w in range(_no)]
            else:
                return [qml.expval(qml.PauliZ(q)) for q in _main_wires]

        self._circuit = _circuit

    def _build_H_matrices(self):
        if self.k_local <= 0:
            return []
        return [create_Hermitian(self.obs_dim, self.A[w], self.B[w],
                                 self.D_obs[w])
                for w in range(self.n_obs)]

    def forward(self, z_t, t):
        """Compute velocity v(z_t, t).

        Args:
            z_t: (B, latent_dim)
            t:   (B,) time in [0, 1]
        Returns:
            v:   (B, latent_dim)
        """
        t_emb = sinusoidal_embedding(t, self.latent_dim)
        t_emb = self.time_embed(t_emb)

        h = torch.cat([z_t, t_emb], dim=-1)        # (B, latent_dim*2)
        h = self.input_proj(h)                       # (B, n_select*n_rots)
        h = h.reshape(z_t.size(0), self.n_select,
                      self.n_rots)                   # (B, n_select, n_rots)
        ts_params = self.rot_sigm(h) * (2 * math.pi)  # [0, 2pi]

        H_mats = self._build_H_matrices()
        exps = self._circuit(
            ts_params, self.prepare_params,
            self.signal_angles, self.qff_params, H_mats)
        exps = torch.stack(exps, dim=1).float()      # (B, n_obs)
        return self.head(exps)                        # (B, latent_dim)

    def get_eigenvalue_range(self):
        if self.k_local <= 0:
            return 0.0, 0.0
        H_mats = self._build_H_matrices()
        lo, hi = float("inf"), float("-inf")
        for H in H_mats:
            eigs = torch.linalg.eigvalsh(
                H.detach().cpu().to(torch.complex128)).real
            lo = min(lo, eigs.min().item())
            hi = max(hi, eigs.max().item())
        return lo, hi


# ---------------------------------------------------------------------------
# 7. Phase 1 -- MambaTextVAE Pretraining
# ---------------------------------------------------------------------------
def train_text_vae(args, train_loader, val_loader, device):
    """Train the MambaTextVAE."""

    vae = MambaTextVAE(
        vocab_size=VOCAB_SIZE,
        embed_dim=args.embed_dim,
        d_model=args.hidden_dim,
        latent_dim=args.latent_dim,
        n_layers=args.n_mamba_layers,
        seq_len=args.seq_len,
        d_state=args.d_state,
        d_conv=args.d_conv,
        expand=args.expand,
    ).to(device)

    optimizer = torch.optim.Adam(vae.parameters(), lr=args.lr_vae)

    total_p = sum(p.numel() for p in vae.parameters() if p.requires_grad)
    print(f"[Phase 1] MambaTextVAE params: {total_p:,}")

    ckpt_dir = os.path.join(args.base_path, "checkpoints")
    results_dir = os.path.join(args.base_path, "results")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f"ckpt_text_vae_{args.job_id}.pt")
    csv_path = os.path.join(results_dir, f"log_text_vae_{args.job_id}.csv")
    fields = ["epoch", "train_loss", "train_ce", "train_kl", "train_bpc",
              "val_loss", "val_ce", "val_kl", "val_bpc", "time_s"]

    start_epoch = 0
    best_val, best_state = float("inf"), None

    if args.resume and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, weights_only=False)
        vae.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        best_val = ckpt.get("best_val", float("inf"))
        print(f"  Resumed from epoch {start_epoch}")

    if start_epoch == 0:
        with open(csv_path, "w", newline="") as f:
            csv.DictWriter(f, fields).writeheader()

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        # -- Train --
        vae.train()
        tr_loss, tr_ce, tr_kl, tr_n = 0.0, 0.0, 0.0, 0
        for (xb,) in tqdm(train_loader,
                          desc=f"MambaVAE Ep {epoch+1}/{args.epochs}",
                          leave=False):
            xb = xb.to(device)
            logits, mu, logvar = vae(xb)

            ce = F.cross_entropy(
                logits.reshape(-1, VOCAB_SIZE), xb.reshape(-1),
                reduction="mean")
            kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = ce + args.beta * kl

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            bs = xb.size(0)
            tr_loss += loss.item() * bs
            tr_ce += ce.item() * bs
            tr_kl += kl.item() * bs
            tr_n += bs

        tr_loss /= tr_n
        tr_ce /= tr_n
        tr_kl /= tr_n
        tr_bpc = tr_ce / math.log(2)

        # -- Val --
        vae.eval()
        vl_loss, vl_ce, vl_kl, vl_n = 0.0, 0.0, 0.0, 0
        with torch.no_grad():
            for (xb,) in val_loader:
                xb = xb.to(device)
                logits, mu, logvar = vae(xb)

                ce = F.cross_entropy(
                    logits.reshape(-1, VOCAB_SIZE), xb.reshape(-1),
                    reduction="mean")
                kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                loss = ce + args.beta * kl

                bs = xb.size(0)
                vl_loss += loss.item() * bs
                vl_ce += ce.item() * bs
                vl_kl += kl.item() * bs
                vl_n += bs

        vl_loss /= vl_n
        vl_ce /= vl_n
        vl_kl /= vl_n
        vl_bpc = vl_ce / math.log(2)
        dt = time.time() - t0

        row = dict(epoch=epoch + 1,
                   train_loss=f"{tr_loss:.6f}", train_ce=f"{tr_ce:.6f}",
                   train_kl=f"{tr_kl:.6f}", train_bpc=f"{tr_bpc:.4f}",
                   val_loss=f"{vl_loss:.6f}", val_ce=f"{vl_ce:.6f}",
                   val_kl=f"{vl_kl:.6f}", val_bpc=f"{vl_bpc:.4f}",
                   time_s=f"{dt:.1f}")
        with open(csv_path, "a", newline="") as f:
            csv.DictWriter(f, fields).writerow(row)

        print(f"  Ep {epoch+1:3d} | Train {tr_loss:.4f} "
              f"(ce={tr_ce:.4f} kl={tr_kl:.4f} bpc={tr_bpc:.3f}) | "
              f"Val {vl_loss:.4f} bpc={vl_bpc:.3f} | {dt:.1f}s")

        if vl_loss < best_val:
            best_val = vl_loss
            best_state = copy.deepcopy(vae.state_dict())

        torch.save(dict(epoch=epoch, model=vae.state_dict(),
                        optimizer=optimizer.state_dict(),
                        best_val=best_val), ckpt_path)

    if best_state is not None:
        vae.load_state_dict(best_state)

    w_path = os.path.join(ckpt_dir, f"weights_text_vae_{args.job_id}.pt")
    torch.save(vae.state_dict(), w_path)
    print(f"  MambaTextVAE saved to {w_path}")

    # Show a reconstruction sample
    vae.eval()
    with torch.no_grad():
        for (xb,) in val_loader:
            xb = xb[:1].to(device)
            logits, _, _ = vae(xb)
            recon = logits.argmax(dim=-1)
            print(f"  Original:  '{decode_tokens(xb[0][:80])}'...")
            print(f"  Recon:     '{decode_tokens(recon[0][:80])}'...")
            break

    return vae


# ---------------------------------------------------------------------------
# 8. Phase 2 -- Quantum CFM Training (QSVTVelocityField)
# ---------------------------------------------------------------------------
def train_text_cfm(args, vae, train_loader, val_loader, device):
    """Train the QSVTVelocityField with CFM loss on text latents."""

    # Freeze VAE
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False

    vf = QSVTVelocityField(
        latent_dim=args.latent_dim,
        n_qubits=args.n_qubits,
        n_select=args.n_select,
        degree=args.degree,
        n_ansatz_layers=args.n_layers,
        k_local=args.k_local,
        obs_scheme=args.obs_scheme,
        device=device,
    ).to(device)

    # Dual optimizer (Chen 2025)
    H_params, circuit_params = [], []
    for name, param in vf.named_parameters():
        if name.startswith(("A.", "B.", "D_obs.")):
            H_params.append(param)
        else:
            circuit_params.append(param)

    circ_opt = torch.optim.Adam(circuit_params, lr=args.lr)
    H_opt = torch.optim.Adam(H_params, lr=args.lr_H) if H_params else None

    total_p = sum(p.numel() for p in vf.parameters() if p.requires_grad)
    h_p = sum(p.numel() for p in H_params)
    c_p = sum(p.numel() for p in circuit_params)
    print(f"[Phase 2] QSVTVelocityField params: total={total_p}  "
          f"circuit={c_p}  observable={h_p}")
    print(f"  QSVT: degree={args.degree}  n_select={args.n_select}  "
          f"layers={args.n_layers}")
    print(f"  Qubits: main={args.n_qubits}  ancilla={vf.n_ancilla}  "
          f"total={vf.total_wires}")
    print(f"  ANO: k_local={args.k_local}, scheme={args.obs_scheme}, "
          f"n_obs={vf.n_obs}")

    ckpt_dir = os.path.join(args.base_path, "checkpoints")
    results_dir = os.path.join(args.base_path, "results")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f"ckpt_text_cfm_{args.job_id}.pt")
    csv_path = os.path.join(results_dir, f"log_text_cfm_{args.job_id}.csv")
    fields = ["epoch", "train_loss", "val_loss", "eig_min", "eig_max",
              "time_s"]

    start_epoch = 0
    best_val, best_state = float("inf"), None

    if args.resume and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, weights_only=False)
        vf.load_state_dict(ckpt["model"])
        circ_opt.load_state_dict(ckpt["circ_opt"])
        if H_opt and "H_opt" in ckpt:
            H_opt.load_state_dict(ckpt["H_opt"])
        start_epoch = ckpt["epoch"] + 1
        best_val = ckpt.get("best_val", float("inf"))
        print(f"  Resumed from epoch {start_epoch}")

    if start_epoch == 0:
        with open(csv_path, "w", newline="") as f:
            csv.DictWriter(f, fields).writeheader()

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        # -- Train --
        vf.train()
        tr_loss, tr_n = 0.0, 0
        for (xb,) in tqdm(train_loader,
                          desc=f"QSVT CFM Ep {epoch+1}/{args.epochs}",
                          leave=False):
            xb = xb.to(device)

            with torch.no_grad():
                z_1, _ = vae.encode(xb)  # mu only (deterministic)

            z_0 = torch.randn_like(z_1)
            t = torch.rand(z_1.size(0), device=device)

            # OT interpolation
            t_col = t[:, None]
            z_t = (1.0 - t_col) * z_0 + t_col * z_1
            target = z_1 - z_0  # constant velocity

            v_pred = vf(z_t, t)
            loss = F.mse_loss(v_pred, target)

            circ_opt.zero_grad()
            if H_opt:
                H_opt.zero_grad()
            loss.backward()
            circ_opt.step()
            if H_opt:
                H_opt.step()

            bs = xb.size(0)
            tr_loss += loss.item() * bs
            tr_n += bs

        tr_loss /= tr_n

        # -- Val --
        vf.eval()
        vl_loss, vl_n = 0.0, 0
        with torch.no_grad():
            for (xb,) in val_loader:
                xb = xb.to(device)
                z_1, _ = vae.encode(xb)
                z_0 = torch.randn_like(z_1)
                t = torch.rand(z_1.size(0), device=device)
                t_col = t[:, None]
                z_t = (1.0 - t_col) * z_0 + t_col * z_1
                target = z_1 - z_0

                v_pred = vf(z_t, t)
                loss = F.mse_loss(v_pred, target)
                vl_loss += loss.item() * xb.size(0)
                vl_n += xb.size(0)

        vl_loss /= vl_n
        eig_lo, eig_hi = vf.get_eigenvalue_range()
        dt = time.time() - t0

        row = dict(epoch=epoch + 1, train_loss=f"{tr_loss:.6f}",
                   val_loss=f"{vl_loss:.6f}", eig_min=f"{eig_lo:.4f}",
                   eig_max=f"{eig_hi:.4f}", time_s=f"{dt:.1f}")
        with open(csv_path, "a", newline="") as f:
            csv.DictWriter(f, fields).writerow(row)

        print(f"  Ep {epoch+1:3d} | Train {tr_loss:.6f} | Val {vl_loss:.6f} | "
              f"Eig [{eig_lo:.2f}, {eig_hi:.2f}] | {dt:.1f}s")

        if vl_loss < best_val:
            best_val = vl_loss
            best_state = copy.deepcopy(vf.state_dict())

        ckpt_data = dict(epoch=epoch, model=vf.state_dict(),
                         circ_opt=circ_opt.state_dict(),
                         best_val=best_val)
        if H_opt:
            ckpt_data["H_opt"] = H_opt.state_dict()
        torch.save(ckpt_data, ckpt_path)

    if best_state is not None:
        vf.load_state_dict(best_state)

    w_path = os.path.join(ckpt_dir, f"weights_text_cfm_{args.job_id}.pt")
    torch.save(vf.state_dict(), w_path)
    print(f"  QSVTVelocityField saved to {w_path}")
    return vf


# ---------------------------------------------------------------------------
# 9. Generation via Euler ODE
# ---------------------------------------------------------------------------
@torch.no_grad()
def generate_text(vae, vf, n_samples, ode_steps, latent_dim, device,
                  temperature=1.0, save_path=None):
    """Generate text by Euler ODE integration in latent space."""
    vae.eval()
    vf.eval()

    z = torch.randn(n_samples, latent_dim, device=device)
    dt = 1.0 / ode_steps

    for step in range(ode_steps):
        t_val = step * dt
        t = torch.full((n_samples,), t_val, device=device)
        v = vf(z, t)
        z = z + dt * v

    logits = vae.decode(z)

    if temperature > 0:
        probs = F.softmax(logits / temperature, dim=-1)
        tokens = torch.multinomial(
            probs.reshape(-1, VOCAB_SIZE), num_samples=1
        ).reshape(n_samples, -1)
    else:
        tokens = logits.argmax(dim=-1)

    texts = [decode_tokens(tokens[i]) for i in range(n_samples)]

    print(f"\n  Generated {n_samples} text samples:")
    for i, text in enumerate(texts[:8]):
        print(f"    [{i}] '{text[:80]}'...")

    if save_path:
        with open(save_path, "w") as f:
            for i, text in enumerate(texts):
                f.write(f"=== Sample {i} ===\n{text}\n\n")
        print(f"  Saved to {save_path}")

    return texts


# ---------------------------------------------------------------------------
# 10. Main
# ---------------------------------------------------------------------------
def main():
    args = get_args()
    set_all_seeds(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"PennyLane {qml.__version__}  |  PyTorch {torch.__version__}  |  "
          f"{device}")
    print(f"Phase: {args.phase}  |  Latent dim: {args.latent_dim}  |  "
          f"Seq len: {args.seq_len}")

    # -- Load text8 data --
    train_loader, val_loader, test_loader = load_text8(
        seq_len=args.seq_len, n_train=args.n_train,
        n_valtest=args.n_valtest, batch_size=args.batch_size)

    # Helper to create VAE with current args
    def make_vae():
        return MambaTextVAE(
            vocab_size=VOCAB_SIZE,
            embed_dim=args.embed_dim,
            d_model=args.hidden_dim,
            latent_dim=args.latent_dim,
            n_layers=args.n_mamba_layers,
            seq_len=args.seq_len,
            d_state=args.d_state,
            d_conv=args.d_conv,
            expand=args.expand,
        ).to(device)

    # Helper to create velocity field with current args
    def make_vf():
        return QSVTVelocityField(
            latent_dim=args.latent_dim,
            n_qubits=args.n_qubits,
            n_select=args.n_select,
            degree=args.degree,
            n_ansatz_layers=args.n_layers,
            k_local=args.k_local,
            obs_scheme=args.obs_scheme,
            device=device,
        ).to(device)

    # -- Phase 1: MambaTextVAE --
    if args.phase in ("1", "both"):
        print("\n=== Phase 1: MambaTextVAE Pretraining ===")
        vae = train_text_vae(args, train_loader, val_loader, device)
    else:
        vae = None

    # -- Phase 2: Quantum CFM --
    if args.phase in ("2", "both"):
        print("\n=== Phase 2: Quantum CFM Training ===")

        if vae is None:
            vae_path = args.vae_ckpt
            if not vae_path:
                vae_path = os.path.join(args.base_path, "checkpoints",
                                        f"weights_text_vae_{args.job_id}.pt")
            if not os.path.exists(vae_path):
                print(f"ERROR: MambaTextVAE weights not found at {vae_path}")
                print("  Run --phase=1 first or provide --vae-ckpt=<path>")
                return
            vae = make_vae()
            vae.load_state_dict(torch.load(vae_path, weights_only=True))
            print(f"  Loaded MambaTextVAE from {vae_path}")

        vf = train_text_cfm(args, vae, train_loader, val_loader, device)

        # Generate sample text
        results_dir = os.path.join(args.base_path, "results")
        os.makedirs(results_dir, exist_ok=True)
        txt_path = os.path.join(results_dir,
                                f"generated_text_{args.job_id}.txt")
        generate_text(vae, vf, min(args.n_samples, 16), args.ode_steps,
                      args.latent_dim, device,
                      temperature=args.temperature, save_path=txt_path)

    # -- Generate only --
    if args.phase == "generate":
        print("\n=== Generation ===")
        ckpt_dir = os.path.join(args.base_path, "checkpoints")

        vae_path = args.vae_ckpt or os.path.join(
            ckpt_dir, f"weights_text_vae_{args.job_id}.pt")
        cfm_path = args.cfm_ckpt or os.path.join(
            ckpt_dir, f"weights_text_cfm_{args.job_id}.pt")

        if not os.path.exists(vae_path) or not os.path.exists(cfm_path):
            print(f"ERROR: Need both MambaTextVAE ({vae_path}) and "
                  f"CFM ({cfm_path}) weights")
            return

        vae = make_vae()
        vae.load_state_dict(torch.load(vae_path, weights_only=True))

        vf = make_vf()
        vf.load_state_dict(torch.load(cfm_path, weights_only=True))

        results_dir = os.path.join(args.base_path, "results")
        os.makedirs(results_dir, exist_ok=True)
        txt_path = os.path.join(results_dir,
                                f"generated_text_{args.job_id}.txt")
        generate_text(vae, vf, args.n_samples, args.ode_steps,
                      args.latent_dim, device,
                      temperature=args.temperature, save_path=txt_path)

    print("\nDone.")


if __name__ == "__main__":
    main()
