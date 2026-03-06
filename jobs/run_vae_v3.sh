#!/bin/bash
#SBATCH -A m4807_g
#SBATCH -C gpu&hbm80g
#SBATCH -q shared
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH -t 24:00:00
#SBATCH -J vae_v3
#SBATCH -o /pscratch/sd/j/junghoon/QuantumFlow/logs/vae_v3_cifar_%j.out
#SBATCH -e /pscratch/sd/j/junghoon/QuantumFlow/logs/vae_v3_cifar_%j.err

export PYTHONNOUSERSITE=1
eval "$(conda shell.bash hook 2>/dev/null)"
conda activate /pscratch/sd/j/junghoon/conda-envs/qml_eeg

cd /pscratch/sd/j/junghoon/QuantumFlow

# VAE v3: SOTA architecture with adversarial training fixes
# Key changes from v2:
#   Architecture:  32->64->128->256 BN+ReLU   -> 128->256->512->512 GN+SiLU
#   Attention:     none                        -> self-attention at 8x8 and 4x4
#   Output:        Sigmoid [0,1]               -> Tanh [-1,1]
#   Recon loss:    MSE                         -> L1 (sharper)
#   Perceptual:    VGG L1                      -> LPIPS (lambda=1.0)
#   Adversarial:   none                        -> PatchGAN hinge (from epoch 51)
#   EMA:           none                        -> decay=0.999
#   Params:        ~2.1M                       -> ~10M
#   Latent dim:    32                          -> 64
#
# Adversarial training fixes (v3.1):
#   R1 gradient penalty (gamma=10, lazy every 16 batches) — slows discriminator
#   VQGAN-style adaptive adversarial weight — auto-balances recon vs adv losses
#   LPIPS computed every 4 batches — reduces epoch time from 559s to ~160s
#
# Resume: set PREV_JOB_ID to the checkpoint job ID
# Example: PREV_JOB_ID=49524905 sbatch jobs/run_vae_v3.sh

PREV_JOB_ID="${PREV_JOB_ID:-}"

RESUME_FLAG=""
JOB_SUFFIX="${SLURM_JOB_ID}"
if [ -n "$PREV_JOB_ID" ]; then
    RESUME_FLAG="--resume"
    JOB_SUFFIX="vae_v3_cifar_${PREV_JOB_ID}"
    echo "Resuming from checkpoint: vae_v3_cifar_${PREV_JOB_ID}"
fi

python -u models/train_vae_v3.py \
    --dataset=cifar10 \
    --latent-dim=64 \
    --beta=0.001 \
    --beta-warmup-epochs=10 \
    --lambda-lpips=1.0 \
    --lambda-adv=0.1 \
    --adaptive-adv-weight \
    --adversarial-start-epoch=51 \
    --free-bits=0.25 \
    --r1-gamma=10.0 \
    --r1-every=16 \
    --lpips-every=4 \
    --lr=1e-4 \
    --lr-disc=4e-4 \
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
