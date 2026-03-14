#!/bin/bash
#SBATCH -A m4807_g
#SBATCH -C gpu&hbm80g
#SBATCH -q shared
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH -t 12:00:00
#SBATCH -J ft_sd3
#SBATCH -o /pscratch/sd/j/junghoon/QuantumFlow/logs/finetune_sd3_vae_%j.out
#SBATCH -e /pscratch/sd/j/junghoon/QuantumFlow/logs/finetune_sd3_vae_%j.err

export PYTHONNOUSERSITE=1
eval "$(conda shell.bash hook 2>/dev/null)"
conda activate /pscratch/sd/j/junghoon/conda-envs/qml_eeg

cd /pscratch/sd/j/junghoon/QuantumFlow

# Fine-tune SD3.5 VAE on CIFAR-10 32x32
# Full fine-tuning with low LR, VGG perceptual loss, cosine schedule
#
# Resume: PREV_JOB_ID=XXXXX sbatch jobs/run_finetune_sd3_vae.sh

PREV_JOB_ID="${PREV_JOB_ID:-}"
RESUME_FLAG=""
JOB_SUFFIX="${SLURM_JOB_ID}"
if [ -n "$PREV_JOB_ID" ]; then
    RESUME_FLAG="--resume"
    JOB_SUFFIX="${PREV_JOB_ID}"
    echo "Resuming from checkpoint: sd3_finetune_${PREV_JOB_ID}"
fi

python -u models/finetune_sd3_vae.py \
    --lr=1e-5 \
    --weight-decay=1e-4 \
    --n-epochs=100 \
    --batch-size=64 \
    --lambda-perc=0.1 \
    --beta=0.0001 \
    --beta-warmup-epochs=10 \
    --ema-decay=0.999 \
    --n-train=50000 \
    --n-valtest=10000 \
    --seed=2025 \
    --save-grid-every=10 \
    --compute-recon-fid \
    $RESUME_FLAG \
    --job-id=sd3_finetune_${JOB_SUFFIX}
