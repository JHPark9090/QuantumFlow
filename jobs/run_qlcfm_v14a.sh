#!/bin/bash
#SBATCH -A m4807_g
#SBATCH -C gpu&hbm80g
#SBATCH -q shared
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH -t 48:00:00
#SBATCH -J v14a
#SBATCH -o /pscratch/sd/j/junghoon/QuantumFlow/logs/qlcfm_v14a_%j.out
#SBATCH -e /pscratch/sd/j/junghoon/QuantumFlow/logs/qlcfm_v14a_%j.err

export PYTHONNOUSERSITE=1
eval "$(conda shell.bash hook 2>/dev/null)"
conda activate /pscratch/sd/j/junghoon/conda-envs/qml_eeg

cd /pscratch/sd/j/junghoon/QuantumFlow

# Phase 2: v14a — Custom VAE (1-downsample) + Multi-Chip Ensemble, concat
# 16 chips x 4q SU(16), pairwise ANO k=2, concat time conditioning
# Per chip: [chunk(256), t_emb(256)] = 512 → enc_proj → 255 (2.01:1)
# Native CIFAR-10 32x32
#
# IMPORTANT: Set VAE_CKPT to the trained VAE weights path
# VAE_CKPT=checkpoints/weights_vae_qlcfm_v14_vae_XXXXX.pt sbatch jobs/run_qlcfm_v14a.sh
#
# Resume: PREV_JOB_ID=XXXXX sbatch jobs/run_qlcfm_v14a.sh

VAE_CKPT="${VAE_CKPT:-}"
if [ -z "$VAE_CKPT" ]; then
    echo "ERROR: VAE_CKPT not set. Set it to the trained VAE weights path."
    echo "  VAE_CKPT=checkpoints/weights_vae_qlcfm_v14_vae_XXXXX.pt sbatch jobs/run_qlcfm_v14a.sh"
    exit 1
fi

PREV_JOB_ID="${PREV_JOB_ID:-}"
RESUME_FLAG=""
JOB_SUFFIX="${SLURM_JOB_ID}"
if [ -n "$PREV_JOB_ID" ]; then
    RESUME_FLAG="--resume"
    JOB_SUFFIX="${PREV_JOB_ID}"
    echo "Resuming from checkpoint: qlcfm_v14a_${PREV_JOB_ID}"
fi

python -u models/QuantumLatentCFM_v14.py \
    --phase=2 \
    --dataset=cifar10 \
    --img-size=32 \
    --c-z=16 \
    --latent-dim=4096 \
    --vae-ckpt="${VAE_CKPT}" \
    --n-chips=16 \
    --n-qubits=4 \
    --k-local=2 \
    --time-conditioning=concat \
    --time-embed-dim=256 \
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
    --job-id=qlcfm_v14a_${JOB_SUFFIX}
