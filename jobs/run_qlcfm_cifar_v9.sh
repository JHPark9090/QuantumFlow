#!/bin/bash
#SBATCH -A m4807_g
#SBATCH -C gpu&hbm80g
#SBATCH -q shared
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH -t 48:00:00
#SBATCH -J qlcfm_v9
#SBATCH -o /pscratch/sd/j/junghoon/QuantumFlow/logs/qlcfm_cifar_v9_%j.out
#SBATCH -e /pscratch/sd/j/junghoon/QuantumFlow/logs/qlcfm_cifar_v9_%j.err

export PYTHONNOUSERSITE=1
eval "$(conda shell.bash hook 2>/dev/null)"
conda activate /pscratch/sd/j/junghoon/conda-envs/qml_eeg

cd /pscratch/sd/j/junghoon/QuantumFlow

# v9: Three CFM improvements over v6:
#   1. Logit-normal timestep sampling (std=1.0) -- focuses training on mid-range t
#   2. Midpoint ODE solver (50 steps = 100 VF evals at 2nd-order accuracy)
#   3. VF EMA (decay=0.999) -- stabilizes noisy quantum gradients
#
# Uses VAE v5 (bottleneck fix) with external pretrained weights.
# Quantum VF: 1x8q, SU(4) encoding, QViT butterfly depth=2, ANO pairwise k=2
#
# IMPORTANT: Set VAE_CKPT to the path of pretrained VAE v5 weights before submitting.
# Example:
#   VAE_CKPT=checkpoints/weights_vae_v5_ema_cifar10_JOBID.pt sbatch jobs/run_qlcfm_cifar_v9.sh

VAE_CKPT="${VAE_CKPT:-}"

if [ -z "$VAE_CKPT" ]; then
    echo "WARNING: No VAE_CKPT specified. Running Phase 1+2 with built-in VAE v5."
    PHASE="both"
    VAE_CKPT_FLAG=""
else
    echo "Using external VAE weights: $VAE_CKPT"
    PHASE="2"
    VAE_CKPT_FLAG="--vae-ckpt=${VAE_CKPT}"
fi

python -u models/QuantumLatentCFM_v9.py \
    --phase=${PHASE} \
    --dataset=cifar10 \
    --vae-arch=v5 \
    --latent-dim=64 \
    --c-z=4 \
    --beta=0.001 \
    --velocity-field=quantum \
    --n-circuits=1 \
    --n-qubits=8 \
    --encoding-type=sun \
    --vqc-type=qvit \
    --qvit-circuit=butterfly \
    --vqc-depth=2 \
    --k-local=2 \
    --obs-scheme=pairwise \
    --time-embed-dim=32 \
    --logit-normal-std=1.0 \
    --ode-solver=midpoint \
    --ode-steps=50 \
    --vf-ema-decay=0.999 \
    --lr=1e-3 \
    --lr-H=1e-1 \
    --lr-vae=1e-3 \
    --batch-size=32 \
    --epochs=200 \
    --n-train=50000 \
    --n-valtest=10000 \
    --seed=2025 \
    --n-samples=64 \
    --compute-metrics \
    ${VAE_CKPT_FLAG} \
    --job-id=qlcfm_cifar_v9_${SLURM_JOB_ID}
