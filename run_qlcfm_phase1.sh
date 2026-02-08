#!/bin/bash
#SBATCH -A m4138_g
#SBATCH -C gpu&hbm80g
#SBATCH -q shared
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH -t 24:00:00
#SBATCH -J qlcfm_vae
#SBATCH -o qlcfm_vae_%j.out
#SBATCH -e qlcfm_vae_%j.err

export PYTHONNOUSERSITE=1
eval "$(conda shell.bash hook 2>/dev/null)"
conda activate /pscratch/sd/j/junghoon/conda-envs/qml_eeg

cd /pscratch/sd/j/junghoon/QuantumFlow

python QuantumLatentCFM.py \
    --phase=1 \
    --dataset=cifar10 \
    --latent-dim=128 \
    --beta=0.5 \
    --lr-vae=1e-3 \
    --batch-size=64 \
    --epochs=200 \
    --n-train=50000 \
    --n-valtest=10000 \
    --seed=2025 \
    --job-id=qlcfm_${SLURM_JOB_ID}
