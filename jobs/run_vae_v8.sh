#!/bin/bash
#SBATCH -A m4807_g
#SBATCH -C gpu&hbm80g
#SBATCH -q shared
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH -t 48:00:00
#SBATCH -J vae_v8
#SBATCH -o /pscratch/sd/j/junghoon/QuantumFlow/logs/vae_v8_cifar_%j.out
#SBATCH -e /pscratch/sd/j/junghoon/QuantumFlow/logs/vae_v8_cifar_%j.err

export PYTHONNOUSERSITE=1
eval "$(conda shell.bash hook 2>/dev/null)"
conda activate /pscratch/sd/j/junghoon/conda-envs/qml_eeg

cd /pscratch/sd/j/junghoon/QuantumFlow

# VAE v8: Same architecture as v6/v7 (gradual channel reduction, no adversarial)
#          but with latent_dim=256, c_z=16
#
#   Bottleneck: ResBlock(256->64) + Conv3x3(64->16) at 4x4
#   flat_dim = 16 * 4 * 4 = 256, fc_mu = Linear(256, 256) = 1:1
#
#   v6: latent_dim=64,  c_z=4  -> PSNR 18.48 peak (overfitting after ep ~69)
#   v7: latent_dim=128, c_z=8  -> PSNR 18.67 peak (overfitting after ep ~65)
#   v8: latent_dim=256, c_z=16 -> test if more latent dims help CIFAR-10
#
# Resume: PREV_JOB_ID=XXXXX sbatch jobs/run_vae_v8.sh

PREV_JOB_ID="${PREV_JOB_ID:-}"

RESUME_FLAG=""
JOB_SUFFIX="${SLURM_JOB_ID}"
if [ -n "$PREV_JOB_ID" ]; then
    RESUME_FLAG="--resume"
    JOB_SUFFIX="vae_v8_cifar_${PREV_JOB_ID}"
    echo "Resuming from checkpoint: vae_v8_cifar_${PREV_JOB_ID}"
fi

python -u models/train_vae_v6.py \
    --dataset=cifar10 \
    --latent-dim=256 \
    --c-z=16 \
    --beta=0.001 \
    --beta-warmup-epochs=10 \
    --lambda-lpips=1.0 \
    --free-bits=0.25 \
    --lpips-every=1 \
    --lr=1e-4 \
    --ema-decay=0.999 \
    --n-epochs=300 \
    --batch-size=64 \
    --n-train=50000 \
    --n-valtest=10000 \
    --seed=2025 \
    --save-grid-every=10 \
    --compute-recon-fid \
    $RESUME_FLAG \
    --job-id=vae_v8_cifar_${JOB_SUFFIX}
