#!/bin/bash
#SBATCH -A m4807_g
#SBATCH -C gpu&hbm80g
#SBATCH -q shared
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH -t 48:00:00
#SBATCH -J text_cfm_ptb
#SBATCH -o text_cfm_ptb_%j.out
#SBATCH -e text_cfm_ptb_%j.err

export PYTHONNOUSERSITE=1
eval "$(conda shell.bash hook 2>/dev/null)"
conda activate /pscratch/sd/j/junghoon/conda-envs/qml_eeg

cd /pscratch/sd/j/junghoon/QuantumFlow

python models/QuantumLatentCFM_Text.py \
    --phase=both \
    --dataset=ptb \
    --latent-dim=128 \
    --embed-dim=64 \
    --hidden-dim=256 \
    --n-mamba-layers=2 \
    --seq-len=256 \
    --n-qubits=8 \
    --degree=3 \
    --n-layers=2 \
    --n-select=4 \
    --k-local=2 \
    --obs-scheme=sliding \
    --lr=1e-3 \
    --lr-H=1e-1 \
    --lr-vae=1e-3 \
    --batch-size=64 \
    --epochs=200 \
    --n-train=10000 \
    --n-valtest=2000 \
    --seed=2025 \
    --kl-warmup-epochs=10 \
    --free-bits=0.1 \
    --job-id=text_cfm_ptb_${SLURM_JOB_ID} \
    --ode-steps=100 \
    --n-samples=32 \
    --temperature=1.0
