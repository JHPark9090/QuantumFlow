#!/bin/bash
#SBATCH -A m4807_g
#SBATCH -C gpu&hbm80g
#SBATCH -q shared
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH -t 04:00:00
#SBATCH -J vae_v2
#SBATCH -o /pscratch/sd/j/junghoon/QuantumFlow/logs/vae_v2_cifar_%j.out
#SBATCH -e /pscratch/sd/j/junghoon/QuantumFlow/logs/vae_v2_cifar_%j.err

export PYTHONNOUSERSITE=1
eval "$(conda shell.bash hook 2>/dev/null)"
conda activate /pscratch/sd/j/junghoon/conda-envs/qml_eeg

cd /pscratch/sd/j/junghoon/QuantumFlow

# VAE v2: Fixed KL normalization, free bits, PSNR-based model selection
# Key changes from v6:
#   beta:        0.5   -> 0.001
#   lambda_perc: 0.1   -> 0.01
#   free_bits:   none  -> 0.25 nats/dim
#   KL:          mean(all) -> sum(latent), mean(batch)
#   best model:  lowest loss -> highest PSNR
#   warmup:      20 epochs -> 10 epochs
#   logvar:      unclamped -> clamped [-20, 2]

python -u models/train_vae_v2.py \
    --dataset=cifar10 \
    --vae-arch=resconv \
    --latent-dim=32 \
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
    --save-grid-every=10 \
    --compute-recon-fid \
    --job-id=vae_v2_cifar_${SLURM_JOB_ID}
