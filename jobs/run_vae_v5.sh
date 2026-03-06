#!/bin/bash
#SBATCH -A m4807_g
#SBATCH -C gpu&hbm80g
#SBATCH -q shared
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH -t 24:00:00
#SBATCH -J vae_v5
#SBATCH -o /pscratch/sd/j/junghoon/QuantumFlow/logs/vae_v5_cifar_%j.out
#SBATCH -e /pscratch/sd/j/junghoon/QuantumFlow/logs/vae_v5_cifar_%j.err

export PYTHONNOUSERSITE=1
eval "$(conda shell.bash hook 2>/dev/null)"
conda activate /pscratch/sd/j/junghoon/conda-envs/qml_eeg

cd /pscratch/sd/j/junghoon/QuantumFlow

# VAE v5: Bottleneck fix -- stop at 4x4, 1x1 Conv channel reduction
# Key changes from v3/v4:
#   Encoder:  Conv2d(256,256,3,2,1)->2x2->flatten(1024)->Linear(1024,64)  [16:1]
#         ->  Conv2d(256, c_z, 1) at 4x4 -> flatten(c_z*16) -> Linear(c_z*16, 64) [1:1 with c_z=4]
#   Decoder:  mirrors the encoder change
#   Discriminator & training: same as v4 (spectral norm, logistic loss, adaptive weight, D warmup)
#
# Resume: PREV_JOB_ID=XXXXX sbatch jobs/run_vae_v5.sh

PREV_JOB_ID="${PREV_JOB_ID:-}"

RESUME_FLAG=""
JOB_SUFFIX="${SLURM_JOB_ID}"
if [ -n "$PREV_JOB_ID" ]; then
    RESUME_FLAG="--resume"
    JOB_SUFFIX="vae_v5_cifar_${PREV_JOB_ID}"
    echo "Resuming from checkpoint: vae_v5_cifar_${PREV_JOB_ID}"
fi

python -u models/train_vae_v5.py \
    --dataset=cifar10 \
    --latent-dim=64 \
    --c-z=4 \
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
    --lpips-every=1 \
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
    --job-id=vae_v5_cifar_${JOB_SUFFIX}
