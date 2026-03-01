#!/bin/bash
#SBATCH -A m4807_g
#SBATCH -C gpu&hbm80g
#SBATCH -q shared
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH -t 02:00:00
#SBATCH -J ratio_c_lat128
#SBATCH -o /pscratch/sd/j/junghoon/QuantumFlow/logs/ratio_c_lat128_%j.out
#SBATCH -e /pscratch/sd/j/junghoon/QuantumFlow/logs/ratio_c_lat128_%j.err

export PYTHONNOUSERSITE=1
eval "$(conda shell.bash hook 2>/dev/null)"
conda activate /pscratch/sd/j/junghoon/conda-envs/qml_eeg

cd /pscratch/sd/j/junghoon/QuantumFlow

# Ratio bottleneck experiment: Classical CFM, latent_dim=128
# MLP 256x3 velocity field
# VAE: ratio_lat128 (deterministic job-id)

python -u models/QuantumLatentCFM_v6.py \
    --phase=2 \
    --dataset=cifar10 \
    --vae-arch=resconv \
    --latent-dim=128 \
    --time-embed-dim=32 \
    --velocity-field=classical \
    --mlp-hidden-dims=256,256,256 \
    --lr=1e-3 \
    --batch-size=32 \
    --epochs=200 \
    --n-train=50000 \
    --n-valtest=10000 \
    --seed=2025 \
    --ode-steps=100 \
    --n-samples=64 \
    --compute-metrics \
    --vae-ckpt=checkpoints/weights_vae_v2_cifar10_ratio_lat128.pt \
    --job-id=ratio_c_lat128_${SLURM_JOB_ID}
