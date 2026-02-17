#!/bin/bash
#SBATCH -A m4807_g
#SBATCH -C gpu&hbm80g
#SBATCH -q shared
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH -t 48:00:00
#SBATCH -J classical_cfm_coco
#SBATCH -o logs/classical_cfm_coco_%j.out
#SBATCH -e logs/classical_cfm_coco_%j.err

export PYTHONNOUSERSITE=1
eval "$(conda shell.bash hook 2>/dev/null)"
conda activate /pscratch/sd/j/junghoon/conda-envs/qml_eeg

cd /pscratch/sd/j/junghoon/QuantumFlow

python models/QuantumLatentCFM.py \
    --phase=both \
    --dataset=coco \
    --latent-dim=32 \
    --beta=0.5 \
    --velocity-field=classical \
    --mlp-hidden-dims=256,256,256 \
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
    --n-eval-samples=1024 \
    --job-id=classical_cfm_coco_${SLURM_JOB_ID}
