#!/bin/bash
#SBATCH -A m4807_g
#SBATCH -C gpu&hbm80g
#SBATCH -q shared
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH -t 48:00:00
#SBATCH -J v10b_cif
#SBATCH -o /pscratch/sd/j/junghoon/QuantumFlow/logs/qlcfm_v10b_cifar_%j.out
#SBATCH -e /pscratch/sd/j/junghoon/QuantumFlow/logs/qlcfm_v10b_cifar_%j.err

export PYTHONNOUSERSITE=1
eval "$(conda shell.bash hook 2>/dev/null)"
conda activate /pscratch/sd/j/junghoon/conda-envs/qml_eeg

cd /pscratch/sd/j/junghoon/QuantumFlow

# QLCFM v10b — CIFAR-10, lat256, additive conditioning, pairwise ANO
# VAE: v6 arch, lat256, c_z=16 (v8 CIFAR, 256/256 active, PSNR 18.64)
# Quantum: 4 qubits SU(16), 255 generators, no QViT
# Time: additive z_t(256) + time_mlp(t) → 256 input → 255 enc (1.00:1)
# ANO: pairwise k=2, C(4,2)=6 observables, 4×4 Hermitians, 96 ANO params
#
# NOTE: VAE v8 still training (ep ~206/300). Extract weights from checkpoint.
# Resume: PREV_JOB_ID=XXXXX sbatch jobs/run_qlcfm_v10b.sh

PREV_JOB_ID="${PREV_JOB_ID:-}"

RESUME_FLAG=""
JOB_SUFFIX="${SLURM_JOB_ID}"
if [ -n "$PREV_JOB_ID" ]; then
    RESUME_FLAG="--resume"
    JOB_SUFFIX="${PREV_JOB_ID}"
    echo "Resuming from checkpoint: v10b_cifar_${PREV_JOB_ID}"
fi

# Extract VAE weights from checkpoint if final weights don't exist yet
VAE_CKPT="checkpoints/weights_vae_v8_cifar_extracted.pt"
VAE_FULL_CKPT="checkpoints/ckpt_vae_v6_cifar10_vae_v8_cifar_49792489.pt"
if [ ! -f "$VAE_CKPT" ]; then
    echo "Extracting VAE weights from checkpoint..."
    python -c "
import torch
ckpt = torch.load('$VAE_FULL_CKPT', weights_only=False, map_location='cpu')
torch.save(ckpt['model'], '$VAE_CKPT')
print(f'  Extracted VAE weights (epoch {ckpt[\"epoch\"]}) -> $VAE_CKPT')
"
fi

if [ ! -f "$VAE_CKPT" ]; then
    echo "ERROR: VAE weights not found at $VAE_CKPT"
    exit 1
fi

python -u models/QuantumLatentCFM_v10.py \
    --phase=2 \
    --dataset=cifar10 \
    --img-size=32 \
    --latent-dim=256 \
    --vae-arch=v6 \
    --c-z=16 \
    --vae-ckpt="$VAE_CKPT" \
    --velocity-field=quantum \
    --n-qubits=4 \
    --n-circuits=1 \
    --vqc-type=none \
    --time-conditioning=additive \
    --time-embed-dim=128 \
    --ano-type=pairwise \
    --n-observables=6 \
    --lr=1e-3 \
    --lr-H=1e-1 \
    --epochs=200 \
    --batch-size=64 \
    --n-train=10000 \
    --n-valtest=2000 \
    --seed=2025 \
    --logit-normal-std=1.0 \
    --ode-solver=midpoint \
    --ode-steps=50 \
    --vf-ema-decay=0.999 \
    --compute-metrics \
    $RESUME_FLAG \
    --job-id=v10b_cifar_${JOB_SUFFIX}
