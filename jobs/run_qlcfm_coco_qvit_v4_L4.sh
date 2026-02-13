#!/bin/bash
#SBATCH -A m4807_g
#SBATCH -C gpu&hbm80g
#SBATCH -q shared
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH -t 48:00:00
#SBATCH -J qlcfm_coco_qvit_v4_L4
#SBATCH -o logs/qlcfm_coco_qvit_v4_L4_%j.out
#SBATCH -e logs/qlcfm_coco_qvit_v4_L4_%j.err

export PYTHONNOUSERSITE=1
eval "$(conda shell.bash hook 2>/dev/null)"
conda activate /pscratch/sd/j/junghoon/conda-envs/qml_eeg

cd /pscratch/sd/j/junghoon/QuantumFlow

python models/QuantumLatentCFM.py \
    --phase=both \
    --dataset=coco \
    --vae-arch=legacy \
    --latent-dim=32 \
    --beta=0.5 \
    --n-qubits=12 \
    --encoding-type=sun \
    --vqc-type=qvit \
    --qvit-circuit=butterfly \
    --k-local=2 \
    --obs-scheme=pairwise \
    --n-reupload=4 \
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
    --job-id=qlcfm_coco_qvit_v4_L4_${SLURM_JOB_ID}
