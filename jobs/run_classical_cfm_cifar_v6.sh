#!/bin/bash
#SBATCH -A m4807_g
#SBATCH -C gpu&hbm80g
#SBATCH -q shared
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH -t 48:00:00
#SBATCH -J classical_cfm_v6
#SBATCH -o /pscratch/sd/j/junghoon/QuantumFlow/logs/classical_cfm_v6_%j.out
#SBATCH -e /pscratch/sd/j/junghoon/QuantumFlow/logs/classical_cfm_v6_%j.err

export PYTHONNOUSERSITE=1
eval "$(conda shell.bash hook 2>/dev/null)"
conda activate /pscratch/sd/j/junghoon/conda-envs/qml_eeg

cd /pscratch/sd/j/junghoon/QuantumFlow

# Classical-A baseline re-run using v6 codebase (for fair comparison)
# Ensures identical VAE, data pipeline, and metrics as v6/v7
# Architecture: time_mlp(32) + MLP(64->256->256->256->32) with SiLU

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
    --job-id=classical_cfm_v6_${SLURM_JOB_ID}
