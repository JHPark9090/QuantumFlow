"""
Quantum Latent CFM v12 — SD3.5 VAE + Classical Bottleneck + v9 Quantum VF
=========================================================================

Combines the pretrained Stable Diffusion 3.5 VAE (from v11) with the v9
quantum circuit architecture (8 qubits, brick-layer SU(4) encoding, QViT
pyramid ansatz, pairwise ANO k=2). A classical bottleneck adapter compresses
the high-dimensional SD3.5 latent (4096-d for 128x128 images) to a
manageable dimension (default 256) before the quantum velocity field.

This provides a direct comparison with v11 (which uses the v10 circuit):
  - v11: SD3.5 VAE + bottleneck + v10 circuit (4q, single SU(16), ANO-only)
  - v12: SD3.5 VAE + bottleneck + v9 circuit (8q, brick-layer SU(4), QViT+ANO)

The v9 circuit uses concat-only time conditioning (no additive option), so
v12 has fewer variants than v11.

Configurations:

  v12a: SD3.5 VAE + bottleneck=256 + 8q brick-layer SU(4) + QViT pyramid
        depth=2, pairwise ANO k=2, C(8,2)=28 obs, time_embed_dim=256

All share:
  - SD3.5 pretrained VAE (16-channel latent, 8x spatial downscale)
  - 8 qubits, brick-layer SU(4) encoding (105 parameters)
  - QViT pyramid ansatz (depth=2, trainable)
  - Pairwise ANO k=2 (28 wire groups, 4x4 Hermitians, 448 ANO params)
  - Classical bottleneck: 4096 -> 256 -> quantum -> 256 -> 4096
  - v9 training improvements (logit-normal, midpoint ODE, VF EMA)
  - Concat time conditioning: [z_bottleneck(256), t_emb(256)] = 512 input

Usage:
  # v12a: SD3.5 VAE, bottleneck=256, 8q SU(4) brick + QViT pyramid
  python QuantumLatentCFM_v12.py --phase=2 --dataset=cifar10 \\
      --img-size=128 --bottleneck-dim=256 \\
      --n-qubits=8 --encoding-type=sun --vqc-type=qvit \\
      --qvit-circuit=pyramid --vqc-depth=2 \\
      --time-embed-dim=256

References:
  - Lipman et al. (2023). Flow Matching for Generative Modeling. ICLR 2023.
  - Esser et al. (2024). Scaling Rectified Flow Transformers. ICML 2024. (SD3)
  - Wiersema et al. (2024). Here comes the SU(N). Quantum, 8, 1275.
  - Chen et al. (2025). Learning to Measure QNNs. ICASSP 2025 Workshop.
  - Lin et al. (2025). Adaptive Non-local Observable on QNNs. IEEE QCE 2025.
  - Cherrat et al. (2024). Quantum Vision Transformers. Quantum, 8, 1265.
"""

import argparse
import os
import sys
import random
import copy
import math
import time
import csv
from itertools import combinations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split
from tqdm import tqdm
import scipy.constants  # noqa: F401 -- pre-import for PennyLane/scipy compat
import pennylane as qml

DATA_ROOT = "/pscratch/sd/j/junghoon/data"


# ---------------------------------------------------------------------------
# 1. Argparse
# ---------------------------------------------------------------------------
def get_args():
    p = argparse.ArgumentParser(
        description="Quantum Latent CFM v12 — SD3.5 VAE + Bottleneck + "
                    "v9 Circuit (Brick-Layer SU(4), QViT, ANO)")

    # Phase (no phase 1 for SD3)
    p.add_argument("--phase", type=str, default="2",
                   choices=["2", "generate"],
                   help="2=CFM train, generate=sample")

    # VAE
    p.add_argument("--latent-dim", type=int, default=4096,
                   help="Full latent dim (auto-computed for SD3: "
                        "16*(img_size//8)^2)")
    p.add_argument("--vae-arch", type=str, default="sd3",
                   choices=["sd3"],
                   help="VAE architecture (sd3 = SD3.5 pretrained)")
    p.add_argument("--sd3-model-id", type=str,
                   default="stabilityai/stable-diffusion-3.5-large",
                   help="HuggingFace model ID for SD3.5 VAE")
    p.add_argument("--sd3-weights", type=str, default="",
                   help="Path to fine-tuned SD3.5 VAE state_dict "
                        "(empty = use pretrained frozen weights)")

    # Bottleneck
    p.add_argument("--bottleneck-dim", type=int, default=256,
                   help="Bottleneck dim for VF (0 = no bottleneck)")

    # Quantum circuit (v9 defaults: 8 qubits, brick-layer SU(4), QViT)
    p.add_argument("--n-circuits", type=int, default=1)
    p.add_argument("--n-qubits", type=int, default=8)
    p.add_argument("--encoding-type", type=str, default="sun",
                   choices=["sun", "angle"],
                   help="sun=brick-layer SU(4), angle=RY+CNOT")
    p.add_argument("--vqc-type", type=str, default="qvit",
                   choices=["qvit", "hardware_efficient", "none"])
    p.add_argument("--vqc-depth", type=int, default=2)
    p.add_argument("--qvit-circuit", type=str, default="pyramid",
                   choices=["butterfly", "pyramid", "x"])

    # ANO
    p.add_argument("--k-local", type=int, default=2,
                   help="Locality of pairwise ANO")
    p.add_argument("--obs-scheme", type=str, default="pairwise",
                   choices=["sliding", "pairwise"])

    # Time conditioning (v9 uses concat only)
    p.add_argument("--time-embed-dim", type=int, default=256)

    # Training
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lr-H", type=float, default=1e-1)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--seed", type=int, default=2025)

    # v9 improvements
    p.add_argument("--logit-normal-std", type=float, default=1.0)
    p.add_argument("--ode-solver", type=str, default="midpoint",
                   choices=["euler", "midpoint"])
    p.add_argument("--vf-ema-decay", type=float, default=0.999)

    # Data
    p.add_argument("--dataset", type=str, default="cifar10",
                   choices=["cifar10", "coco", "mnist", "fashion"])
    p.add_argument("--n-train", type=int, default=10000)
    p.add_argument("--n-valtest", type=int, default=2000)
    p.add_argument("--img-size", type=int, default=128)

    # ODE sampling
    p.add_argument("--ode-steps", type=int, default=50)
    p.add_argument("--n-samples", type=int, default=64)

    # I/O
    p.add_argument("--job-id", type=str, default="qlcfm_v12_001")
    p.add_argument("--base-path", type=str, default=".")
    p.add_argument("--cfm-ckpt", type=str, default="")
    p.add_argument("--resume", action="store_true")

    # Velocity field type
    p.add_argument("--velocity-field", type=str, default="quantum",
                   choices=["quantum", "classical"])
    p.add_argument("--mlp-hidden-dims", type=str, default="256,256,256")

    # Evaluation
    p.add_argument("--compute-metrics", action="store_true")
    p.add_argument("--n-eval-samples", type=int, default=1024)

    return p.parse_args()


# ---------------------------------------------------------------------------
# 2. Utilities
# ---------------------------------------------------------------------------
def set_all_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    qml.numpy.random.seed(seed)


def create_Hermitian(N, A, B, D):
    """Build NxN Hermitian from learnable real params (Lin et al., 2025)."""
    h = torch.zeros((N, N), dtype=torch.complex128)
    count = 0
    for i in range(1, N):
        h[i - 1, i - 1] = D[i].clone()
        for j in range(i):
            h[i, j] = A[count + j].clone() + 1j * B[count + j].clone()
        count += i
    return h.clone() + h.clone().conj().T


def get_wire_groups(n_qubits, k_local, obs_scheme):
    if k_local <= 0:
        return [[q] for q in range(n_qubits)]
    if obs_scheme == "sliding":
        return [list(range(s, s + k_local))
                for s in range(n_qubits - k_local + 1)]
    elif obs_scheme == "pairwise":
        return [list(c) for c in combinations(range(n_qubits), k_local)]
    raise ValueError(f"Unknown obs_scheme: {obs_scheme}")


def _qvit_n_params(n_qubits, circuit_type):
    if circuit_type == "butterfly":
        count = 0
        n_layers = int(math.ceil(math.log2(n_qubits)))
        for layer in range(n_layers):
            stride = 2 ** layer
            for start in range(0, n_qubits - stride, 2 * stride):
                for offset in range(stride):
                    idx1 = start + offset
                    idx2 = start + offset + stride
                    if idx1 < n_qubits and idx2 < n_qubits:
                        count += 1
        return count
    elif circuit_type == "pyramid":
        return sum(n_qubits - layer - 1 for layer in range(n_qubits - 1))
    elif circuit_type == "x":
        return n_qubits // 2 + max(0, n_qubits // 2 - 1)
    raise ValueError(f"Unknown circuit_type: {circuit_type}")


def _compute_enc_per_block(n_qubits, encoding_type):
    """Compute encoding parameters for brick-layer SU(4) or angle encoding."""
    if encoding_type == "sun":
        n_even = n_qubits // 2
        n_odd = (n_qubits - 1) // 2
        return (n_even + n_odd) * 15
    else:
        return n_qubits


class EMAModel:
    """Exponential moving average of model weights."""

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.clone().detach()
                       for k, v in model.state_dict().items()}

    def update(self, model):
        with torch.no_grad():
            for k, v in model.state_dict().items():
                self.shadow[k].mul_(self.decay).add_(v, alpha=1.0 - self.decay)

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = {k: v.clone().detach() for k, v in state_dict.items()}

    def apply(self, model):
        model.load_state_dict(self.shadow)


# ---------------------------------------------------------------------------
# 3. Data Loaders
# ---------------------------------------------------------------------------
def load_cifar_2d(seed, n_train, n_valtest, batch_size, img_size=128):
    from torchvision.datasets import CIFAR10
    torch.manual_seed(seed)
    data_train = CIFAR10(root=DATA_ROOT, train=True, download=True)
    data_test = CIFAR10(root=DATA_ROOT, train=False, download=True)
    X_tr = torch.tensor(data_train.data).float().permute(0, 3, 1, 2) / 255.0
    X_te = torch.tensor(data_test.data).float().permute(0, 3, 1, 2) / 255.0
    if img_size != 32:
        X_tr = F.interpolate(X_tr, size=(img_size, img_size), mode="bicubic",
                             align_corners=False)
        X_te = F.interpolate(X_te, size=(img_size, img_size), mode="bicubic",
                             align_corners=False)
    # Normalize to [-1, 1] for SD3.5 VAE
    X_tr = X_tr * 2.0 - 1.0
    X_te = X_te * 2.0 - 1.0
    X_tr = X_tr[torch.randperm(len(X_tr))[:n_train]]
    X_te = X_te[torch.randperm(len(X_te))[:n_valtest]]
    return _make_gen_loaders(X_tr, X_te, batch_size)


def load_mnist_2d(seed, n_train, n_valtest, batch_size, img_size=128):
    from torchvision.datasets import MNIST
    torch.manual_seed(seed)
    data_train = MNIST(root=DATA_ROOT, train=True, download=True)
    data_test = MNIST(root=DATA_ROOT, train=False, download=True)
    X_tr = data_train.data[:n_train].float().unsqueeze(1) / 255.0
    X_te = data_test.data[:n_valtest].float().unsqueeze(1) / 255.0
    X_tr = F.interpolate(X_tr, size=(img_size, img_size), mode="bicubic",
                         align_corners=False).repeat(1, 3, 1, 1)
    X_te = F.interpolate(X_te, size=(img_size, img_size), mode="bicubic",
                         align_corners=False).repeat(1, 3, 1, 1)
    X_tr = X_tr * 2.0 - 1.0
    X_te = X_te * 2.0 - 1.0
    X_tr = X_tr[torch.randperm(len(X_tr))]
    X_te = X_te[torch.randperm(len(X_te))]
    return _make_gen_loaders(X_tr, X_te, batch_size)


def load_fashion_2d(seed, n_train, n_valtest, batch_size, img_size=128):
    from torchvision.datasets import FashionMNIST
    torch.manual_seed(seed)
    data_train = FashionMNIST(root=DATA_ROOT, train=True, download=True)
    data_test = FashionMNIST(root=DATA_ROOT, train=False, download=True)
    X_tr = data_train.data[:n_train].float().unsqueeze(1) / 255.0
    X_te = data_test.data[:n_valtest].float().unsqueeze(1) / 255.0
    X_tr = F.interpolate(X_tr, size=(img_size, img_size), mode="bicubic",
                         align_corners=False).repeat(1, 3, 1, 1)
    X_te = F.interpolate(X_te, size=(img_size, img_size), mode="bicubic",
                         align_corners=False).repeat(1, 3, 1, 1)
    X_tr = X_tr * 2.0 - 1.0
    X_te = X_te * 2.0 - 1.0
    X_tr = X_tr[torch.randperm(len(X_tr))]
    X_te = X_te[torch.randperm(len(X_te))]
    return _make_gen_loaders(X_tr, X_te, batch_size)


class LazyCOCODataset(Dataset):
    """Loads COCO images on demand to avoid OOM with large datasets."""

    def __init__(self, img_paths, img_size=128):
        from torchvision import transforms
        self.img_paths = img_paths
        self.tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        from PIL import Image
        img = Image.open(self.img_paths[idx]).convert("RGB")
        x = self.tf(img)
        return (x,)


def load_coco_2d(seed, n_train, n_valtest, batch_size, img_size=128):
    torch.manual_seed(seed)
    np.random.seed(seed)
    data_dir = os.path.join(DATA_ROOT, "coco")
    ann_file = os.path.join(data_dir, "annotations/instances_train2017.json")
    img_dir = os.path.join(data_dir, "train2017")
    try:
        from pycocotools.coco import COCO
        coco = COCO(ann_file)
        img_ids = sorted(coco.getImgIds())
        rng = np.random.RandomState(seed)
        rng.shuffle(img_ids)
        img_ids = img_ids[:n_train + n_valtest]
        paths = []
        for iid in img_ids:
            info = coco.loadImgs(iid)[0]
            paths.append(os.path.join(img_dir, info["file_name"]))
        train_ds = LazyCOCODataset(paths[:n_train], img_size)
        valtest_ds = LazyCOCODataset(paths[n_train:n_train + n_valtest],
                                     img_size)
        val_sz = len(valtest_ds) // 2
        test_sz = len(valtest_ds) - val_sz
        val_ds, test_ds = random_split(valtest_ds, [val_sz, test_sz])
        train_loader = DataLoader(train_ds, batch_size=batch_size,
                                  shuffle=True, drop_last=True,
                                  num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                                num_workers=2, pin_memory=True)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                                 num_workers=2, pin_memory=True)
        return train_loader, val_loader, test_loader
    except Exception as e:
        print(f"  [WARN] COCO loading failed ({e}), using synthetic data")
        X_tr = torch.rand(n_train, 3, img_size, img_size) * 2.0 - 1.0
        X_te = torch.rand(n_valtest, 3, img_size, img_size) * 2.0 - 1.0
        return _make_gen_loaders(X_tr, X_te, batch_size)


def _make_gen_loaders(X_train, X_valtest, batch_size):
    train_ds = TensorDataset(X_train)
    valtest_ds = TensorDataset(X_valtest)
    val_sz = len(valtest_ds) // 2
    test_sz = len(valtest_ds) - val_sz
    val_ds, test_ds = random_split(valtest_ds, [val_sz, test_sz])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


# ---------------------------------------------------------------------------
# 4. SD3.5 VAE Wrapper
# ---------------------------------------------------------------------------
class SD3VAEWrapper(nn.Module):
    """Wraps the pretrained SD3.5 AutoencoderKL for use as a frozen encoder/decoder.

    The SD3.5 VAE uses 16-channel latents with 8x spatial downscale.
    For a 128x128 input image, the latent is (16, 16, 16) -> flat dim = 4096.

    encode(x) returns (z_flat, None) for compatibility with CFM training.
    decode(z_flat) reshapes and decodes back to image space.
    """

    def __init__(self, model_id="stabilityai/stable-diffusion-3.5-large",
                 img_size=128, weights_path=""):
        super().__init__()
        from diffusers import AutoencoderKL

        self.img_size = img_size
        spatial = img_size // 8
        self.latent_channels = 16
        self.latent_spatial = (spatial, spatial)
        self.latent_flat_dim = self.latent_channels * spatial * spatial

        print(f"  Loading SD3.5 VAE from {model_id} ...")
        self.vae = AutoencoderKL.from_pretrained(
            model_id, subfolder="vae", torch_dtype=torch.float32)

        if weights_path:
            print(f"  Loading fine-tuned weights from {weights_path} ...")
            state_dict = torch.load(weights_path, map_location="cpu")
            self.vae.load_state_dict(state_dict)
            print(f"  Fine-tuned SD3.5 VAE weights loaded.")

        # Freeze all VAE parameters
        for p in self.vae.parameters():
            p.requires_grad = False
        self.vae.eval()

        print(f"  SD3.5 VAE loaded: latent shape = ({self.latent_channels}, "
              f"{spatial}, {spatial}), flat dim = {self.latent_flat_dim}")

    def encode(self, x):
        """Encode images to flattened latent vectors.

        Args:
            x: (batch, 3, H, W) images in [-1, 1]

        Returns:
            z_flat: (batch, latent_flat_dim) flattened latent
            None: placeholder for compatibility (no logvar)
        """
        with torch.no_grad():
            z = self.vae.encode(x).latent_dist.mean  # deterministic
        z_flat = z.reshape(z.size(0), -1)
        return z_flat, None

    def decode(self, z_flat):
        """Decode flattened latent vectors to images.

        Args:
            z_flat: (batch, latent_flat_dim) flattened latent

        Returns:
            images: (batch, 3, H, W) in [-1, 1]
        """
        z = z_flat.view(-1, self.latent_channels,
                        self.latent_spatial[0], self.latent_spatial[1])
        with torch.no_grad():
            images = self.vae.decode(z).sample
        return images.clamp(-1, 1)

    def forward(self, x):
        z_flat, _ = self.encode(x)
        return self.decode(z_flat)


def build_vae(args):
    """Build VAE (SD3.5 only in v12)."""
    if args.vae_arch == "sd3":
        return SD3VAEWrapper(model_id=args.sd3_model_id,
                             img_size=args.img_size,
                             weights_path=args.sd3_weights)
    raise ValueError(f"v12 only supports vae_arch='sd3', got '{args.vae_arch}'")


# ---------------------------------------------------------------------------
# 5a. Classical Velocity Field
# ---------------------------------------------------------------------------
class ClassicalVelocityField(nn.Module):
    def __init__(self, latent_dim=32, hidden_dims=(256, 256, 256),
                 time_embed_dim=64):
        super().__init__()
        self.latent_dim = latent_dim
        self.time_embed_dim = time_embed_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        dims = [latent_dim + time_embed_dim] + list(hidden_dims)
        layers = []
        for i in range(len(dims) - 1):
            layers += [nn.Linear(dims[i], dims[i + 1]), nn.SiLU()]
        layers.append(nn.Linear(dims[-1], latent_dim))
        self.net = nn.Sequential(*layers)

    def _time_embedding(self, t):
        half = self.time_embed_dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device) / half)
        args = t.unsqueeze(-1) * freqs
        return torch.cat([args.cos(), args.sin()], dim=-1)

    def forward(self, z_t, t):
        t_emb = self.time_mlp(self._time_embedding(t))
        return self.net(torch.cat([z_t, t_emb], dim=-1))

    def get_eigenvalue_range(self):
        return 0.0, 0.0


# ---------------------------------------------------------------------------
# 5b. Brick-Layer SU(4) Quantum Circuit with QViT Ansatz + ANO (v9 style)
# ---------------------------------------------------------------------------
class BrickLayerQuantumCircuit(nn.Module):
    """Quantum circuit with brick-layer SU(4) encoding, QViT ansatz, and ANO.

    This is the v9 quantum circuit architecture:
      - Encoding: Brick-layer SU(4) on qubit pairs (even + odd layers)
        For 8 qubits: 4 even + 3 odd = 7 gates x 15 = 105 encoding params
      - QViT ansatz: Butterfly/Pyramid/X pattern with U3+Ising gates
      - ANO: Pairwise k-local Hermitian observables

    Uses PennyLane SpecialUnitary with default.qubit + backprop.
    """

    def __init__(self, input_dim, n_qubits, encoding_type, vqc_type,
                 vqc_depth, k_local, obs_scheme, qvit_circuit="pyramid",
                 circuit_id=0):
        super().__init__()
        self.n_qubits = n_qubits
        self.encoding_type = encoding_type
        self.vqc_type = vqc_type
        self.vqc_depth = vqc_depth
        self.k_local = k_local
        self.circuit_id = circuit_id
        self.enc_per_block = _compute_enc_per_block(n_qubits, encoding_type)

        self.enc_proj = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.SiLU(),
            nn.Linear(256, self.enc_per_block),
        )

        print(f"  [Circuit {circuit_id}] Brick-layer {encoding_type.upper()}"
              f"({n_qubits}q), enc_proj {input_dim}->{self.enc_per_block} "
              f"({input_dim/self.enc_per_block:.2f}:1)")

        if vqc_type == "qvit":
            n_rbs = _qvit_n_params(n_qubits, qvit_circuit)
            self.qvit_params = nn.Parameter(
                0.01 * torch.randn(vqc_depth, n_rbs, 12))
            print(f"  [Circuit {circuit_id}] QViT {qvit_circuit} ansatz: "
                  f"depth={vqc_depth}, {n_rbs} RBS gates/layer, "
                  f"{vqc_depth * n_rbs * 12} VQC params")
        elif vqc_type == "hardware_efficient":
            self.var_params = nn.Parameter(
                0.01 * torch.randn(vqc_depth, n_qubits))
            print(f"  [Circuit {circuit_id}] HWE ansatz: "
                  f"depth={vqc_depth}, {vqc_depth * n_qubits} params")

        # ANO observables
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
            self.D = nn.ParameterList(
                [nn.Parameter(torch.empty(K)) for _ in range(self.n_obs)])
            for w in range(self.n_obs):
                nn.init.normal_(self.A[w], std=2.0)
                nn.init.normal_(self.B[w], std=2.0)
                nn.init.normal_(self.D[w], std=2.0)
            ano_params = self.n_obs * (2 * n_off + K)
            print(f"  [Circuit {circuit_id}] Pairwise ANO (k={k_local}): "
                  f"{self.n_obs} wire groups, {K}x{K} Hermitians, "
                  f"{ano_params:,} total ANO params")
        else:
            self.obs_dim = 0

        # Build QNode
        dev = qml.device("default.qubit")
        _wg = self.wire_groups
        _nq = n_qubits
        _et = encoding_type
        _vt = vqc_type
        _vd = vqc_depth
        _kl = k_local
        _no = self.n_obs
        _qc = qvit_circuit

        @qml.qnode(dev, interface="torch", diff_method="best")
        def _circuit(enc, vqc_params, H_mats):
            def _qvit_gate(p, w1, w2):
                qml.U3(p[0], p[1], p[2], wires=w1)
                qml.IsingXX(p[3], wires=[w1, w2])
                qml.IsingYY(p[4], wires=[w1, w2])
                qml.IsingZZ(p[5], wires=[w1, w2])
                qml.U3(p[6], p[7], p[8], wires=w1)
                qml.IsingXX(p[9], wires=[w1, w2])
                qml.IsingYY(p[10], wires=[w1, w2])
                qml.IsingZZ(p[11], wires=[w1, w2])

            def _qvit_layer(params_ly):
                pidx = 0
                if _qc == "butterfly":
                    n_layers = int(math.ceil(math.log2(_nq)))
                    for layer in range(n_layers):
                        stride = 2 ** layer
                        for start in range(0, _nq - stride, 2 * stride):
                            for offset in range(stride):
                                w1 = start + offset
                                w2 = start + offset + stride
                                if w1 < _nq and w2 < _nq:
                                    _qvit_gate(params_ly[pidx], w1, w2)
                                    pidx += 1
                elif _qc == "pyramid":
                    for layer in range(_nq - 1):
                        for i in range(_nq - layer - 1):
                            _qvit_gate(params_ly[pidx], i, i + 1)
                            pidx += 1
                elif _qc == "x":
                    for i in range(_nq // 2):
                        w1, w2 = i, _nq - 1 - i
                        _qvit_gate(params_ly[pidx], w1, w2)
                        pidx += 1
                    for i in range(_nq // 2 - 1):
                        _qvit_gate(params_ly[pidx], i, i + 1)
                        pidx += 1

            def _hwe_layer(params_ly):
                for q in range(_nq):
                    qml.RY(params_ly[q], wires=q)
                for q in range(0, _nq - 1, 2):
                    qml.CNOT(wires=[q, q + 1])
                for q in range(1, _nq - 1, 2):
                    qml.CNOT(wires=[q, q + 1])

            # -- Encoding: brick-layer SU(4) --
            if _et == "sun":
                idx = 0
                # Even layer: pairs (0,1), (2,3), (4,5), (6,7)
                for q in range(0, _nq - 1, 2):
                    qml.SpecialUnitary(enc[..., idx:idx + 15],
                                       wires=[q, q + 1])
                    idx += 15
                # Odd layer: pairs (1,2), (3,4), (5,6)
                for q in range(1, _nq - 1, 2):
                    qml.SpecialUnitary(enc[..., idx:idx + 15],
                                       wires=[q, q + 1])
                    idx += 15
            else:
                # Angle encoding fallback
                for q in range(_nq):
                    qml.RY(enc[..., q], wires=q)
                for q in range(0, _nq - 1, 2):
                    qml.CNOT(wires=[q, q + 1])
                for q in range(1, _nq - 1, 2):
                    qml.CNOT(wires=[q, q + 1])

            # -- QViT ansatz (post-encoding entangling circuit) --
            if _vt == "qvit":
                for ly in range(_vd):
                    _qvit_layer(vqc_params[ly])
            elif _vt == "hardware_efficient":
                for ly in range(_vd):
                    _hwe_layer(vqc_params[ly])

            # -- ANO measurement --
            if _kl > 0:
                return [qml.expval(qml.Hermitian(H_mats[w], wires=_wg[w]))
                        for w in range(_no)]
            else:
                return [qml.expval(qml.PauliZ(q)) for q in range(_nq)]

        self._circuit = _circuit

    def _build_H_matrices(self):
        if self.k_local <= 0:
            return []
        return [create_Hermitian(self.obs_dim, self.A[w], self.B[w], self.D[w])
                for w in range(self.n_obs)]

    def forward(self, x):
        enc = self.enc_proj(x)
        H_mats = self._build_H_matrices()
        if self.vqc_type == "qvit":
            vqc_p = self.qvit_params
        elif self.vqc_type == "hardware_efficient":
            vqc_p = self.var_params
        else:
            vqc_p = torch.zeros(1)
        q_out = self._circuit(enc, vqc_p, H_mats)
        return torch.stack(q_out, dim=1).float()

    def get_eigenvalue_range(self):
        if self.k_local <= 0:
            return float("inf"), float("-inf")
        H_mats = self._build_H_matrices()
        lo, hi = float("inf"), float("-inf")
        for H in H_mats:
            eigs = torch.linalg.eigvalsh(
                H.detach().cpu().to(torch.complex128)).real
            lo = min(lo, eigs.min().item())
            hi = max(hi, eigs.max().item())
        return lo, hi


# ---------------------------------------------------------------------------
# 5c. Quantum Velocity Field (with bottleneck, v9 circuit)
# ---------------------------------------------------------------------------
class QuantumVelocityField(nn.Module):
    """Quantum velocity field using v9 brick-layer circuit with bottleneck.

    v9 uses concat-only time conditioning: [z_t, t_emb].
    With bottleneck: [z_bottleneck(256), t_emb(256)] = 512 input.
    """

    def __init__(self, latent_dim, n_circuits, n_qubits, encoding_type,
                 vqc_type, vqc_depth, k_local, obs_scheme,
                 qvit_circuit="pyramid", time_embed_dim=256,
                 bottleneck_dim=0):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_circuits = n_circuits
        self.n_qubits = n_qubits
        self.time_embed_dim = time_embed_dim
        self.bottleneck_dim = bottleneck_dim

        # Determine working dimension (bottleneck or full latent)
        use_bottleneck = (bottleneck_dim > 0 and bottleneck_dim != latent_dim)
        self.use_bottleneck = use_bottleneck
        working_dim = bottleneck_dim if use_bottleneck else latent_dim

        if use_bottleneck:
            self.bottleneck_in = nn.Linear(latent_dim, bottleneck_dim)
            print(f"  Bottleneck: {latent_dim} -> {bottleneck_dim} -> "
                  f"quantum -> {bottleneck_dim} -> {latent_dim}")

        # v9 uses concat only: [z_working, t_emb]
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        input_dim = working_dim + time_embed_dim

        self.circuits = nn.ModuleList()
        for k in range(n_circuits):
            self.circuits.append(BrickLayerQuantumCircuit(
                input_dim=input_dim, n_qubits=n_qubits,
                encoding_type=encoding_type, vqc_type=vqc_type,
                vqc_depth=vqc_depth, k_local=k_local,
                obs_scheme=obs_scheme, qvit_circuit=qvit_circuit,
                circuit_id=k))

        self.n_obs_per_circuit = self.circuits[0].n_obs
        self.total_obs = sum(c.n_obs for c in self.circuits)

        _vh = max(256, self.total_obs)
        self.vel_head = nn.Sequential(
            nn.Linear(self.total_obs, _vh),
            nn.SiLU(),
            nn.Linear(_vh, working_dim),
        )

        if use_bottleneck:
            self.bottleneck_out = nn.Linear(working_dim, latent_dim)

    def _time_embedding(self, t):
        half = self.time_embed_dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device) / half)
        args = t.unsqueeze(-1) * freqs
        return torch.cat([args.cos(), args.sin()], dim=-1)

    def forward(self, z_t, t):
        # Bottleneck compression
        if self.use_bottleneck:
            z_work = self.bottleneck_in(z_t)
        else:
            z_work = z_t

        # Concat time conditioning (v9 style)
        t_emb = self.time_mlp(self._time_embedding(t))
        z_combined = torch.cat([z_work, t_emb], dim=-1)

        # Quantum circuits
        q_outputs = []
        for k in range(self.n_circuits):
            q_outputs.append(self.circuits[k](z_combined))
        q_all = torch.cat(q_outputs, dim=1)

        # Velocity head
        v = self.vel_head(q_all)

        # Bottleneck expansion
        if self.use_bottleneck:
            v = self.bottleneck_out(v)

        return v

    def get_eigenvalue_range(self):
        lo, hi = float("inf"), float("-inf")
        for circ in self.circuits:
            c_lo, c_hi = circ.get_eigenvalue_range()
            lo = min(lo, c_lo)
            hi = max(hi, c_hi)
        return lo, hi


def build_velocity_field(args, device):
    if args.velocity_field == "classical":
        hidden = [int(d) for d in args.mlp_hidden_dims.split(",")]
        vf = ClassicalVelocityField(
            latent_dim=args.latent_dim, hidden_dims=hidden,
            time_embed_dim=args.time_embed_dim).to(device)
    else:
        vf = QuantumVelocityField(
            latent_dim=args.latent_dim, n_circuits=args.n_circuits,
            n_qubits=args.n_qubits, encoding_type=args.encoding_type,
            vqc_type=args.vqc_type, vqc_depth=args.vqc_depth,
            k_local=args.k_local, obs_scheme=args.obs_scheme,
            qvit_circuit=args.qvit_circuit,
            time_embed_dim=args.time_embed_dim,
            bottleneck_dim=args.bottleneck_dim).to(device)
    return vf


# ---------------------------------------------------------------------------
# 6. Phase 2 — CFM Training (v9 improvements)
# ---------------------------------------------------------------------------
def train_cfm(args, vae, train_loader, val_loader, device):
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False

    vf = build_velocity_field(args, device)

    # Dual optimizer (Chen 2025)
    H_params, circuit_params = [], []
    for name, param in vf.named_parameters():
        if any(tag in name for tag in (".A.", ".B.", ".D.")):
            H_params.append(param)
        else:
            circuit_params.append(param)

    circ_opt = torch.optim.Adam(circuit_params, lr=args.lr)
    H_opt = torch.optim.Adam(H_params, lr=args.lr_H) if H_params else None

    from torch.optim.lr_scheduler import CosineAnnealingLR
    circ_sched = CosineAnnealingLR(circ_opt, T_max=args.epochs)
    H_sched = CosineAnnealingLR(H_opt, T_max=args.epochs) if H_opt else None

    # EMA for velocity field
    vf_ema = None
    if args.vf_ema_decay > 0:
        vf_ema = EMAModel(vf, decay=args.vf_ema_decay)

    total_p = sum(p.numel() for p in vf.parameters() if p.requires_grad)
    h_p = sum(p.numel() for p in H_params)
    c_p = sum(p.numel() for p in circuit_params)
    print(f"[Phase 2] Velocity field params: total={total_p}  "
          f"circuit={c_p}  observable={h_p}")

    # Print architecture info
    if args.logit_normal_std > 0:
        print(f"  [v9] Logit-normal timestep sampling "
              f"(std={args.logit_normal_std})")
    else:
        print(f"  [v9] Uniform timestep sampling")
    print(f"  [v9] ODE solver: {args.ode_solver} ({args.ode_steps} steps)")
    if vf_ema:
        print(f"  [v9] VF EMA enabled (decay={args.vf_ema_decay})")

    if args.velocity_field == "quantum":
        enc_per = vf.circuits[0].enc_per_block
        working_dim = (args.bottleneck_dim
                       if vf.use_bottleneck else args.latent_dim)
        input_dim = working_dim + args.time_embed_dim
        print(f"  Architecture: K={args.n_circuits} circuit(s), "
              f"{args.n_qubits}q each")
        print(f"  Time conditioning: concat [z_work, t_emb]")
        print(f"  Input: concat(z_work[{working_dim}], "
              f"t_emb[{args.time_embed_dim}]) = {input_dim}")
        print(f"  Encoding: {args.encoding_type}, "
              f"{enc_per} params/circuit")
        print(f"  VQC: {args.vqc_type} ({args.qvit_circuit}), "
              f"depth={args.vqc_depth}")
        print(f"  ANO: k_local={args.k_local}, scheme={args.obs_scheme}, "
              f"total_obs={vf.total_obs}")
        print(f"  enc_proj ratio: {input_dim/enc_per:.2f}:1")
        if vf.use_bottleneck:
            print(f"  Bottleneck: {args.latent_dim} -> {args.bottleneck_dim} "
                  f"-> quantum -> {args.bottleneck_dim} -> {args.latent_dim}")
    else:
        print(f"  Classical MLP: hidden_dims={args.mlp_hidden_dims}")

    ckpt_dir = os.path.join(args.base_path, "checkpoints")
    results_dir = os.path.join(args.base_path, "results")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f"ckpt_cfm_{args.job_id}.pt")
    csv_path = os.path.join(results_dir, f"log_cfm_{args.job_id}.csv")
    fields = ["epoch", "train_loss", "val_loss", "val_loss_ema",
              "eig_min", "eig_max", "time_s"]

    start_epoch = 0
    best_val, best_state = float("inf"), None

    if args.resume and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, weights_only=False)
        vf.load_state_dict(ckpt["model"])
        circ_opt.load_state_dict(ckpt["circ_opt"])
        if H_opt and "H_opt" in ckpt:
            H_opt.load_state_dict(ckpt["H_opt"])
        if "circ_sched" in ckpt:
            circ_sched.load_state_dict(ckpt["circ_sched"])
        if H_sched and "H_sched" in ckpt:
            H_sched.load_state_dict(ckpt["H_sched"])
        if vf_ema and "vf_ema" in ckpt:
            vf_ema.load_state_dict(ckpt["vf_ema"])
        start_epoch = ckpt["epoch"] + 1
        best_val = ckpt.get("best_val", float("inf"))
        print(f"  Resumed from epoch {start_epoch}")

    if start_epoch == 0:
        with open(csv_path, "w", newline="") as f:
            csv.DictWriter(f, fields).writeheader()

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        vf.train()
        tr_loss, tr_n = 0.0, 0
        for (xb,) in tqdm(train_loader,
                          desc=f"CFM Ep {epoch+1}/{args.epochs}", leave=False):
            xb = xb.to(device)

            with torch.no_grad():
                z_1, _ = vae.encode(xb)

            z_0 = torch.randn_like(z_1)

            if args.logit_normal_std > 0:
                t = torch.sigmoid(
                    torch.randn(z_1.size(0), device=device)
                    * args.logit_normal_std)
            else:
                t = torch.rand(z_1.size(0), device=device)

            t_col = t[:, None]
            z_t = (1.0 - t_col) * z_0 + t_col * z_1
            target = z_1 - z_0

            v_pred = vf(z_t, t)
            loss = F.mse_loss(v_pred, target)

            circ_opt.zero_grad()
            if H_opt:
                H_opt.zero_grad()
            loss.backward()
            circ_opt.step()
            if H_opt:
                H_opt.step()

            if vf_ema:
                vf_ema.update(vf)

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

        # EMA validation
        vl_loss_ema = vl_loss
        orig_state = None
        if vf_ema:
            orig_state = copy.deepcopy(vf.state_dict())
            vf_ema.apply(vf)
            vf.eval()
            vl_ema, vl_ema_n = 0.0, 0
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
                    vl_ema += loss.item() * xb.size(0)
                    vl_ema_n += xb.size(0)
            vl_loss_ema = vl_ema / vl_ema_n
            vf.load_state_dict(orig_state)

        eig_lo, eig_hi = vf.get_eigenvalue_range()
        dt = time.time() - t0

        row = dict(epoch=epoch + 1, train_loss=f"{tr_loss:.6f}",
                   val_loss=f"{vl_loss:.6f}",
                   val_loss_ema=f"{vl_loss_ema:.6f}",
                   eig_min=f"{eig_lo:.4f}",
                   eig_max=f"{eig_hi:.4f}", time_s=f"{dt:.1f}")
        with open(csv_path, "a", newline="") as f:
            csv.DictWriter(f, fields).writerow(row)

        ema_str = f" EMA={vl_loss_ema:.6f}" if vf_ema else ""
        print(f"  Ep {epoch+1:3d} | Train {tr_loss:.6f} | Val {vl_loss:.6f}"
              f"{ema_str} | Eig [{eig_lo:.2f}, {eig_hi:.2f}] | {dt:.1f}s")

        track_val = vl_loss_ema if vf_ema else vl_loss
        if track_val < best_val:
            best_val = track_val
            if vf_ema:
                vf_ema.apply(vf)
                best_state = copy.deepcopy(vf.state_dict())
                vf.load_state_dict(orig_state)
            else:
                best_state = copy.deepcopy(vf.state_dict())

        circ_sched.step()
        if H_sched:
            H_sched.step()

        ckpt_data = dict(epoch=epoch, model=vf.state_dict(),
                         circ_opt=circ_opt.state_dict(),
                         circ_sched=circ_sched.state_dict(),
                         best_val=best_val)
        if H_opt:
            ckpt_data["H_opt"] = H_opt.state_dict()
        if H_sched:
            ckpt_data["H_sched"] = H_sched.state_dict()
        if vf_ema:
            ckpt_data["vf_ema"] = vf_ema.state_dict()
        torch.save(ckpt_data, ckpt_path)

    if best_state is not None:
        vf.load_state_dict(best_state)

    w_path = os.path.join(ckpt_dir, f"weights_cfm_{args.job_id}.pt")
    torch.save(vf.state_dict(), w_path)
    print(f"  CFM velocity field saved to {w_path}")
    return vf


# ---------------------------------------------------------------------------
# 7. Generation (midpoint ODE)
# ---------------------------------------------------------------------------
@torch.no_grad()
def generate_samples(vae, vf, n_samples, ode_steps, latent_dim, device,
                     save_path=None, ode_solver="midpoint"):
    vae.eval()
    vf.eval()

    z = torch.randn(n_samples, latent_dim, device=device)
    dt = 1.0 / ode_steps

    for step in range(ode_steps):
        t_val = step * dt
        t = torch.full((n_samples,), t_val, device=device)

        if ode_solver == "midpoint":
            k1 = vf(z, t)
            z_mid = z + 0.5 * dt * k1
            t_mid = torch.full((n_samples,), t_val + 0.5 * dt, device=device)
            k2 = vf(z_mid, t_mid)
            z = z + dt * k2
        else:
            v = vf(z, t)
            z = z + dt * v

    images = vae.decode(z)
    # SD3.5 VAE outputs in [-1, 1], convert to [0, 1] for saving
    images = (images.clamp(-1, 1) + 1) / 2

    if save_path:
        from torchvision.utils import save_image
        save_image(images, save_path, nrow=8, padding=2)
        print(f"  Generated {n_samples} samples -> {save_path}")

    return images


# ---------------------------------------------------------------------------
# 7b. FID & IS Evaluation
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate_fid_is(vae, vf, real_loader, n_samples, ode_steps,
                    latent_dim, device, batch_size=64,
                    ode_solver="midpoint"):
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image.inception import InceptionScore

    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    inception_score = InceptionScore(normalize=True).to(device)

    n_real = 0
    for (xb,) in real_loader:
        xb = xb.to(device)
        # Convert from [-1,1] to [0,1] for metrics
        xb = (xb.clamp(-1, 1) + 1) / 2
        fid.update(xb, real=True)
        n_real += xb.size(0)
        if n_real >= n_samples:
            break

    vae.eval()
    vf.eval()
    n_gen = 0
    while n_gen < n_samples:
        bs = min(batch_size, n_samples - n_gen)
        imgs = generate_samples(vae, vf, bs, ode_steps, latent_dim, device,
                                ode_solver=ode_solver)
        fid.update(imgs, real=False)
        inception_score.update(imgs)
        n_gen += bs

    fid_val = fid.compute().item()
    is_mean, is_std = inception_score.compute()

    print(f"  FID = {fid_val:.2f}")
    print(f"  IS  = {is_mean.item():.2f} +/- {is_std.item():.2f}")
    return {"fid": fid_val, "is_mean": is_mean.item(), "is_std": is_std.item()}


# ---------------------------------------------------------------------------
# 8. Main
# ---------------------------------------------------------------------------
def main():
    args = get_args()
    set_all_seeds(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"PennyLane {qml.__version__}  |  PyTorch {torch.__version__}  |  "
          f"{device}")

    # Compute latent_dim for SD3 VAE
    if args.vae_arch == "sd3":
        args.latent_dim = 16 * (args.img_size // 8) ** 2
        print(f"SD3.5 VAE latent_dim = 16 * ({args.img_size}//8)^2 = "
              f"{args.latent_dim}")

    if args.velocity_field == "quantum":
        label = (f"Quantum {args.n_circuits}x{args.n_qubits}q "
                 f"brick-layer {args.encoding_type.upper()}, "
                 f"QViT={args.vqc_type}")
    else:
        label = "Classical"
    print(f"Phase: {args.phase}  |  Dataset: {args.dataset}  |  "
          f"Latent dim: {args.latent_dim}  |  VF: {label}")
    if args.bottleneck_dim > 0 and args.bottleneck_dim != args.latent_dim:
        print(f"Bottleneck: {args.latent_dim} -> {args.bottleneck_dim} -> "
              f"quantum -> {args.bottleneck_dim} -> {args.latent_dim}")
    print(f"v9 improvements: logit-normal(std={args.logit_normal_std}), "
          f"ODE={args.ode_solver}({args.ode_steps}), "
          f"VF-EMA(decay={args.vf_ema_decay})")

    # -- Load data --
    loader_map = {
        "cifar10": load_cifar_2d,
        "coco": load_coco_2d,
        "mnist": load_mnist_2d,
        "fashion": load_fashion_2d,
    }
    train_loader, val_loader, test_loader = loader_map[args.dataset](
        seed=args.seed, n_train=args.n_train, n_valtest=args.n_valtest,
        batch_size=args.batch_size, img_size=args.img_size)

    print(f"Data: train={args.n_train}  valtest={args.n_valtest}  "
          f"img_size={args.img_size}")

    # -- Phase 2: CFM --
    if args.phase == "2":
        print(f"\n=== Phase 2: {label} CFM Training (v12 — SD3.5 VAE + "
              f"v9 Circuit) ===")

        vae = build_vae(args).to(device)
        vf = train_cfm(args, vae, train_loader, val_loader, device)

        img_path = os.path.join(args.base_path, "results",
                                f"samples_{args.job_id}.png")
        generate_samples(vae, vf, min(args.n_samples, 64), args.ode_steps,
                         args.latent_dim, device, save_path=img_path,
                         ode_solver=args.ode_solver)

        if args.compute_metrics:
            print("\n=== Computing FID & IS ===")
            metrics = evaluate_fid_is(
                vae, vf, train_loader, args.n_eval_samples,
                args.ode_steps, args.latent_dim, device,
                ode_solver=args.ode_solver)
            import json
            metrics_path = os.path.join(args.base_path, "results",
                                        f"metrics_{args.job_id}.json")
            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=2)
            print(f"  Metrics saved to {metrics_path}")

    # -- Generate only --
    if args.phase == "generate":
        print("\n=== Generation ===")
        cfm_path = args.cfm_ckpt or os.path.join(
            args.base_path, "checkpoints", f"weights_cfm_{args.job_id}.pt")

        if not os.path.exists(cfm_path):
            print(f"ERROR: CFM weights not found at {cfm_path}")
            return

        vae = build_vae(args).to(device)

        vf = build_velocity_field(args, device)
        vf.load_state_dict(
            torch.load(cfm_path, weights_only=True, map_location=device))

        img_path = os.path.join(args.base_path, "results",
                                f"generated_{args.job_id}.png")
        generate_samples(vae, vf, args.n_samples, args.ode_steps,
                         args.latent_dim, device, save_path=img_path,
                         ode_solver=args.ode_solver)

        if args.compute_metrics:
            metrics = evaluate_fid_is(
                vae, vf, train_loader, args.n_eval_samples,
                args.ode_steps, args.latent_dim, device,
                ode_solver=args.ode_solver)


if __name__ == "__main__":
    main()
