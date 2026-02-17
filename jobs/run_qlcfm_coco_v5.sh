#!/bin/bash
#SBATCH -A m4807_g
#SBATCH -C gpu&hbm80g
#SBATCH -q shared
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH -t 48:00:00
#SBATCH -J qlcfm_coco_v5
#SBATCH -o logs/qlcfm_coco_v5_%j.out
#SBATCH -e logs/qlcfm_coco_v5_%j.err

export PYTHONNOUSERSITE=1
eval "$(conda shell.bash hook 2>/dev/null)"
conda activate /pscratch/sd/j/junghoon/conda-envs/qml_eeg

cd /pscratch/sd/j/junghoon/QuantumFlow

python models/QuantumLatentCFM_v5.py \
    --phase=both \
    --dataset=coco \
    --vae-arch=legacy \
    --latent-dim=64 \
    --beta=0.5 \
    --n-circuits=16 \
    --n-qubits=8 \
    --encoding-type=sun \
    --vqc-type=qvit \
    --qvit-circuit=butterfly \
    --vqc-depth=2 \
    --k-local=2 \
    --obs-scheme=pairwise \
    --lr=1e-3 \
    --lr-H=1e-1 \
    --lr-vae=1e-3 \
    --batch-size=32 \
    --epochs=200 \
    --n-train=50000 \
    --n-valtest=10000 \
    --seed=2025 \
    --ode-steps=100 \
    --n-samples=64 \
    --compute-metrics \
    --job-id=qlcfm_coco_v5_${SLURM_JOB_ID}
