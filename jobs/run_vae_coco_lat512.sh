#!/bin/bash
#SBATCH -A m4807_g
#SBATCH -C gpu&hbm80g
#SBATCH -q shared
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH -t 48:00:00
#SBATCH -J vae_co512
#SBATCH -o /pscratch/sd/j/junghoon/QuantumFlow/logs/vae_coco_lat512_%j.out
#SBATCH -e /pscratch/sd/j/junghoon/QuantumFlow/logs/vae_coco_lat512_%j.err

export PYTHONNOUSERSITE=1
eval "$(conda shell.bash hook 2>/dev/null)"
conda activate /pscratch/sd/j/junghoon/conda-envs/qml_eeg

cd /pscratch/sd/j/junghoon/QuantumFlow

# COCO 128x128, latent_dim=512, c_z=32
# Bottleneck: ResBlock(256->64) + Conv3x3(64->32) at 4x4
# flat_dim = 32 * 4 * 4 = 512, fc_mu = Linear(512, 512) = 1:1
#
# Resume: PREV_JOB_ID=XXXXX sbatch jobs/run_vae_coco_lat512.sh

PREV_JOB_ID="${PREV_JOB_ID:-}"

RESUME_FLAG=""
JOB_SUFFIX="${SLURM_JOB_ID}"
if [ -n "$PREV_JOB_ID" ]; then
    RESUME_FLAG="--resume"
    JOB_SUFFIX="vae_coco_lat512_${PREV_JOB_ID}"
    echo "Resuming from checkpoint: vae_coco_lat512_${PREV_JOB_ID}"
fi

python -u models/train_vae_v6.py \
    --dataset=coco \
    --img-size=128 \
    --latent-dim=512 \
    --c-z=32 \
    --beta=0.001 \
    --beta-warmup-epochs=10 \
    --lambda-lpips=1.0 \
    --free-bits=0.25 \
    --lpips-every=1 \
    --lr=1e-4 \
    --ema-decay=0.999 \
    --n-epochs=300 \
    --batch-size=32 \
    --n-train=80000 \
    --n-valtest=10000 \
    --seed=2025 \
    --save-grid-every=10 \
    --compute-recon-fid \
    $RESUME_FLAG \
    --job-id=vae_coco_lat512_${JOB_SUFFIX}
