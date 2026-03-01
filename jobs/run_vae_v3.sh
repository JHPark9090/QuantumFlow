#!/bin/bash
#SBATCH -A m4807_g
#SBATCH -C gpu&hbm80g
#SBATCH -q shared
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH -t 06:00:00
#SBATCH -J vae_v3
#SBATCH -o /pscratch/sd/j/junghoon/QuantumFlow/logs/vae_v3_cifar_%j.out
#SBATCH -e /pscratch/sd/j/junghoon/QuantumFlow/logs/vae_v3_cifar_%j.err

export PYTHONNOUSERSITE=1
eval "$(conda shell.bash hook 2>/dev/null)"
conda activate /pscratch/sd/j/junghoon/conda-envs/qml_eeg

cd /pscratch/sd/j/junghoon/QuantumFlow

# VAE v3: SOTA architecture upgrade
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

python -u models/train_vae_v3.py \
    --dataset=cifar10 \
    --latent-dim=64 \
    --beta=0.001 \
    --beta-warmup-epochs=10 \
    --lambda-lpips=1.0 \
    --lambda-adv=0.1 \
    --adversarial-start-epoch=51 \
    --free-bits=0.25 \
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
    --job-id=vae_v3_cifar_${SLURM_JOB_ID}
