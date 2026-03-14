#!/bin/bash
#SBATCH -A m4807_g
#SBATCH -C gpu&hbm80g
#SBATCH -q shared
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH -t 12:00:00
#SBATCH -J v14_vae
#SBATCH -o /pscratch/sd/j/junghoon/QuantumFlow/logs/qlcfm_v14_vae_%j.out
#SBATCH -e /pscratch/sd/j/junghoon/QuantumFlow/logs/qlcfm_v14_vae_%j.err

export PYTHONNOUSERSITE=1
eval "$(conda shell.bash hook 2>/dev/null)"
conda activate /pscratch/sd/j/junghoon/conda-envs/qml_eeg

cd /pscratch/sd/j/junghoon/QuantumFlow

# Phase 1: Train custom VAE with 1 downsample (32x32 → 16x16)
# c_z=16, latent_dim=16*16*16=4096, native CIFAR-10 32x32
#
# Resume: PREV_JOB_ID=XXXXX sbatch jobs/run_qlcfm_v14_vae.sh

PREV_JOB_ID="${PREV_JOB_ID:-}"
RESUME_FLAG=""
JOB_SUFFIX="${SLURM_JOB_ID}"
if [ -n "$PREV_JOB_ID" ]; then
    RESUME_FLAG="--resume"
    JOB_SUFFIX="${PREV_JOB_ID}"
    echo "Resuming from checkpoint: qlcfm_v14_vae_${PREV_JOB_ID}"
fi

python -u models/QuantumLatentCFM_v14.py \
    --phase=1 \
    --dataset=cifar10 \
    --img-size=32 \
    --c-z=16 \
    --latent-dim=4096 \
    --beta=0.5 \
    --beta-warmup-epochs=20 \
    --lambda-perc=0.1 \
    --lr-vae=1e-3 \
    --epochs=200 \
    --batch-size=64 \
    --n-train=10000 \
    --n-valtest=2000 \
    --seed=2025 \
    $RESUME_FLAG \
    --job-id=qlcfm_v14_vae_${JOB_SUFFIX}
