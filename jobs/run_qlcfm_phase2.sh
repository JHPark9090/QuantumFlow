#!/bin/bash
#SBATCH -A m4807_g
#SBATCH -C gpu&hbm80g
#SBATCH -q shared
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH -t 48:00:00
#SBATCH -J qlcfm_cfm
#SBATCH -o logs/qlcfm_cfm_%j.out
#SBATCH -e logs/qlcfm_cfm_%j.err

export PYTHONNOUSERSITE=1
eval "$(conda shell.bash hook 2>/dev/null)"
conda activate /pscratch/sd/j/junghoon/conda-envs/qml_eeg

cd /pscratch/sd/j/junghoon/QuantumFlow

# Set VAE checkpoint path (update after Phase 1 completes)
VAE_CKPT="weights_vae_qlcfm_${VAE_JOB_ID}.pt"

python models/QuantumLatentCFM.py \
    --phase=2 \
    --dataset=cifar10 \
    --latent-dim=128 \
    --n-qubits=8 \
    --n-blocks=2 \
    --encoding-type=sun \
    --vqc-type=qcnn \
    --vqc-depth=2 \
    --k-local=2 \
    --obs-scheme=sliding \
    --lr=1e-3 \
    --lr-H=1e-1 \
    --batch-size=32 \
    --epochs=300 \
    --n-train=50000 \
    --n-valtest=10000 \
    --seed=2025 \
    --job-id=qlcfm_${SLURM_JOB_ID} \
    --vae-ckpt=${VAE_CKPT} \
    --ode-steps=100 \
    --n-samples=64
