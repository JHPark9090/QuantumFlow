#!/bin/bash
#SBATCH -A m4807_g
#SBATCH -C gpu&hbm80g
#SBATCH -q shared
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH -t 02:00:00
#SBATCH -J cfm_vaev3_c
#SBATCH -o /pscratch/sd/j/junghoon/QuantumFlow/logs/cfm_vaev3_classical_%j.out
#SBATCH -e /pscratch/sd/j/junghoon/QuantumFlow/logs/cfm_vaev3_classical_%j.err

export PYTHONNOUSERSITE=1
eval "$(conda shell.bash hook 2>/dev/null)"
conda activate /pscratch/sd/j/junghoon/conda-envs/qml_eeg

cd /pscratch/sd/j/junghoon/QuantumFlow

# Classical CFM with VAE v3 (control baseline)
# MLP velocity field (256x3), latent_dim=64
#
# IMPORTANT: Set VAE_CKPT to the path from run_vae_v3.sh output
# Example: checkpoints/weights_vae_v3_ema_cifar10_vae_v3_cifar_<JOB_ID>.pt

VAE_CKPT="${VAE_CKPT:-checkpoints/weights_vae_v3_ema_cifar10_vae_v3_cifar_PLACEHOLDER.pt}"

if [ ! -f "$VAE_CKPT" ]; then
    echo "ERROR: VAE checkpoint not found at $VAE_CKPT"
    echo "Set VAE_CKPT environment variable or edit this script"
    exit 1
fi

python -u models/QuantumLatentCFM_vaev3.py \
    --phase=2 \
    --dataset=cifar10 \
    --latent-dim=64 \
    --velocity-field=classical \
    --mlp-hidden-dims=256,256,256 \
    --time-embed-dim=32 \
    --lr=1e-3 \
    --batch-size=64 \
    --epochs=200 \
    --n-train=50000 \
    --n-valtest=10000 \
    --seed=2025 \
    --ode-steps=100 \
    --n-samples=64 \
    --compute-metrics \
    --n-eval-samples=50000 \
    --vae-ckpt="$VAE_CKPT" \
    --job-id=cfm_vaev3_classical_${SLURM_JOB_ID}
