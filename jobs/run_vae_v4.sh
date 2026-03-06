#!/bin/bash
#SBATCH -A m4807_g
#SBATCH -C gpu&hbm80g
#SBATCH -q shared
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH -t 24:00:00
#SBATCH -J vae_v4
#SBATCH -o /pscratch/sd/j/junghoon/QuantumFlow/logs/vae_v4_cifar_%j.out
#SBATCH -e /pscratch/sd/j/junghoon/QuantumFlow/logs/vae_v4_cifar_%j.err

export PYTHONNOUSERSITE=1
eval "$(conda shell.bash hook 2>/dev/null)"
conda activate /pscratch/sd/j/junghoon/conda-envs/qml_eeg

cd /pscratch/sd/j/junghoon/QuantumFlow

# VAE v4: Discriminator & adversarial training fixes (same VAE arch as v3)
# Key changes from v3:
#   Discriminator:  64->128->256->1 (663K)    -> 64->128->256->512->1 (2.8M) + spectral norm
#   Loss:           hinge (ReLU clips grads)   -> logistic (softplus, non-saturating)
#   Adaptive weight: off by default            -> on by default
#   D warmup:       none                       -> 5 epochs D-only + 20 epoch ramp
#   D optimizer:    betas=(0.5,0.9) cosine     -> betas=(0.0,0.99) constant LR=2e-4
#   Diagnostics:    none                       -> D(real/fake) mean/std per epoch
#
# Resume: set PREV_JOB_ID to the checkpoint job ID
# Example: PREV_JOB_ID=49600000 sbatch jobs/run_vae_v4.sh

PREV_JOB_ID="${PREV_JOB_ID:-}"

RESUME_FLAG=""
JOB_SUFFIX="${SLURM_JOB_ID}"
if [ -n "$PREV_JOB_ID" ]; then
    RESUME_FLAG="--resume"
    JOB_SUFFIX="vae_v4_cifar_${PREV_JOB_ID}"
    echo "Resuming from checkpoint: vae_v4_cifar_${PREV_JOB_ID}"
fi

python -u models/train_vae_v4.py \
    --dataset=cifar10 \
    --latent-dim=64 \
    --beta=0.001 \
    --beta-warmup-epochs=10 \
    --lambda-lpips=1.0 \
    --lambda-adv=0.1 \
    --adversarial-start-epoch=51 \
    --disc-warmup-epochs=5 \
    --disc-ramp-epochs=20 \
    --free-bits=0.25 \
    --r1-gamma=10.0 \
    --r1-every=16 \
    --lpips-every=4 \
    --lr=1e-4 \
    --lr-disc=2e-4 \
    --ema-decay=0.999 \
    --n-epochs=300 \
    --batch-size=64 \
    --n-train=50000 \
    --n-valtest=10000 \
    --seed=2025 \
    --save-grid-every=10 \
    --compute-recon-fid \
    $RESUME_FLAG \
    --job-id=${JOB_SUFFIX}
