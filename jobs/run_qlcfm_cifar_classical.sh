#!/bin/bash
#SBATCH -A m4807_g
#SBATCH -C gpu&hbm80g
#SBATCH -q shared
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH -t 48:00:00
#SBATCH -J qlcfm_cifar_classical
#SBATCH -o qlcfm_cifar_classical_%j.out
#SBATCH -e qlcfm_cifar_classical_%j.err

export PYTHONNOUSERSITE=1
eval "$(conda shell.bash hook 2>/dev/null)"
conda activate /pscratch/sd/j/junghoon/conda-envs/qml_eeg

cd /pscratch/sd/j/junghoon/QuantumFlow

python models/QuantumLatentCFM.py \
    --phase=both \
    --dataset=cifar10 \
    --vae-arch=resconv \
    --latent-dim=256 \
    --beta=0.5 \
    --beta-warmup-epochs=20 \
    --lambda-perc=0.1 \
    --velocity-field=classical \
    --mlp-hidden-dims=256,256,256 \
    --lr=1e-3 \
    --lr-vae=1e-3 \
    --batch-size=64 \
    --epochs=200 \
    --n-train=50000 \
    --n-valtest=10000 \
    --seed=2025 \
    --ode-steps=100 \
    --n-samples=64 \
    --compute-metrics \
    --n-eval-samples=1024 \
    --job-id=qlcfm_cifar_classical_${SLURM_JOB_ID}
