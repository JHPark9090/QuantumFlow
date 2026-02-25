#!/bin/bash
#SBATCH -A m4807_g
#SBATCH -C gpu&hbm80g
#SBATCH -q shared
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH -t 48:00:00
#SBATCH -J classical_cfm_C
#SBATCH -o /pscratch/sd/j/junghoon/QuantumFlow/logs/classical_cfm_C_%j.out
#SBATCH -e /pscratch/sd/j/junghoon/QuantumFlow/logs/classical_cfm_C_%j.err

export PYTHONNOUSERSITE=1
eval "$(conda shell.bash hook 2>/dev/null)"
conda activate /pscratch/sd/j/junghoon/conda-envs/qml_eeg

cd /pscratch/sd/j/junghoon/QuantumFlow

# Classical-C: Same architecture as Classical-A but with time_embed_dim=32
# Architecture: time_mlp(32) + MLP(64->256->256->256->32) with SiLU
# Serves as the classical control for v6/v7 (all share time_embed_dim=32)

python models/QuantumLatentCFM_v6.py \
    --phase=both \
    --dataset=cifar10 \
    --vae-arch=resconv \
    --latent-dim=32 \
    --beta=0.5 \
    --velocity-field=classical \
    --mlp-hidden-dims=256,256,256 \
    --time-embed-dim=32 \
    --lr=1e-3 \
    --lr-vae=1e-3 \
    --batch-size=32 \
    --epochs=200 \
    --n-train=50000 \
    --n-valtest=10000 \
    --seed=2025 \
    --ode-steps=100 \
    --n-samples=64 \
    --compute-metrics \
    --job-id=classical_cfm_C_${SLURM_JOB_ID}
