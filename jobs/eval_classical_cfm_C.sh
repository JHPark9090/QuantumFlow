#!/bin/bash
#SBATCH -A m4807_g
#SBATCH -C gpu&hbm80g
#SBATCH -q shared
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH -t 02:00:00
#SBATCH -J eval_cfm_C
#SBATCH -o /pscratch/sd/j/junghoon/QuantumFlow/logs/eval_cfm_C_%j.out
#SBATCH -e /pscratch/sd/j/junghoon/QuantumFlow/logs/eval_cfm_C_%j.err

export PYTHONNOUSERSITE=1
eval "$(conda shell.bash hook 2>/dev/null)"
conda activate /pscratch/sd/j/junghoon/conda-envs/qml_eeg

cd /pscratch/sd/j/junghoon/QuantumFlow

# Recompute FID/IS for Classical-C with 50k eval samples
python models/QuantumLatentCFM_v6.py \
    --phase=generate \
    --dataset=cifar10 \
    --vae-arch=resconv \
    --latent-dim=32 \
    --velocity-field=classical \
    --mlp-hidden-dims=256,256,256 \
    --time-embed-dim=32 \
    --batch-size=32 \
    --n-train=50000 \
    --n-valtest=10000 \
    --seed=2025 \
    --ode-steps=100 \
    --n-samples=64 \
    --compute-metrics \
    --n-eval-samples=50000 \
    --vae-ckpt=checkpoints/weights_vae_classical_cfm_C_49235600.pt \
    --cfm-ckpt=checkpoints/weights_cfm_classical_cfm_C_49235600.pt \
    --job-id=eval_classical_cfm_C_50k_${SLURM_JOB_ID}
