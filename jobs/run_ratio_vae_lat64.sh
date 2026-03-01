#!/bin/bash
#SBATCH -A m4807_g
#SBATCH -C gpu&hbm80g
#SBATCH -q shared
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH -t 04:00:00
#SBATCH -J ratio_vae_lat64
#SBATCH -o /pscratch/sd/j/junghoon/QuantumFlow/logs/ratio_vae_lat64_%j.out
#SBATCH -e /pscratch/sd/j/junghoon/QuantumFlow/logs/ratio_vae_lat64_%j.err

export PYTHONNOUSERSITE=1
eval "$(conda shell.bash hook 2>/dev/null)"
conda activate /pscratch/sd/j/junghoon/conda-envs/qml_eeg

cd /pscratch/sd/j/junghoon/QuantumFlow

# Ratio bottleneck experiment: VAE with latent_dim=64
# Deterministic job-id (no SLURM_JOB_ID) so CFM scripts can reference weights directly
# Output: checkpoints/weights_vae_v2_cifar10_ratio_lat64.pt

python -u models/train_vae_v2.py \
    --dataset=cifar10 \
    --vae-arch=resconv \
    --latent-dim=64 \
    --beta=0.001 \
    --beta-warmup-epochs=10 \
    --lambda-perc=0.01 \
    --free-bits=0.25 \
    --lr=1e-3 \
    --n-epochs=200 \
    --batch-size=128 \
    --n-train=50000 \
    --n-valtest=10000 \
    --seed=2025 \
    --save-grid-every=50 \
    --compute-recon-fid \
    --job-id=ratio_lat64
