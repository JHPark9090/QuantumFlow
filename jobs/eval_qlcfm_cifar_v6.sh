#!/bin/bash
#SBATCH -A m4807
#SBATCH -C cpu
#SBATCH -q shared
#SBATCH -n 1
#SBATCH -c 64
#SBATCH -t 06:00:00
#SBATCH -J eval_v6_cpu
#SBATCH -o /pscratch/sd/j/junghoon/QuantumFlow/logs/eval_v6_%j.out
#SBATCH -e /pscratch/sd/j/junghoon/QuantumFlow/logs/eval_v6_%j.err

export PYTHONNOUSERSITE=1
eval "$(conda shell.bash hook 2>/dev/null)"
conda activate /pscratch/sd/j/junghoon/conda-envs/qml_eeg

cd /pscratch/sd/j/junghoon/QuantumFlow

# Recompute FID/IS for v6-Quantum with 50k eval samples
python -u models/QuantumLatentCFM_v6.py \
    --phase=generate \
    --dataset=cifar10 \
    --vae-arch=resconv \
    --latent-dim=32 \
    --velocity-field=quantum \
    --n-circuits=1 \
    --n-qubits=8 \
    --encoding-type=sun \
    --vqc-type=qvit \
    --qvit-circuit=butterfly \
    --vqc-depth=2 \
    --k-local=2 \
    --obs-scheme=pairwise \
    --time-embed-dim=32 \
    --batch-size=32 \
    --n-train=50000 \
    --n-valtest=10000 \
    --seed=2025 \
    --ode-steps=100 \
    --n-samples=64 \
    --compute-metrics \
    --n-eval-samples=50000 \
    --vae-ckpt=checkpoints/weights_vae_qlcfm_cifar_v6_49235601.pt \
    --cfm-ckpt=checkpoints/weights_cfm_qlcfm_cifar_v6_49235601.pt \
    --job-id=eval_qlcfm_v6_50k_${SLURM_JOB_ID}
