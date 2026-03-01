#!/bin/bash
#SBATCH -A m4807_g
#SBATCH -C gpu&hbm80g
#SBATCH -q shared
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH -t 36:00:00
#SBATCH -J ratio_q_lat32
#SBATCH -o /pscratch/sd/j/junghoon/QuantumFlow/logs/ratio_q_lat32_%j.out
#SBATCH -e /pscratch/sd/j/junghoon/QuantumFlow/logs/ratio_q_lat32_%j.err

export PYTHONNOUSERSITE=1
eval "$(conda shell.bash hook 2>/dev/null)"
conda activate /pscratch/sd/j/junghoon/conda-envs/qml_eeg

cd /pscratch/sd/j/junghoon/QuantumFlow

# Ratio bottleneck experiment: Quantum CFM, latent_dim=32
# 8q, pairwise k=2 -> 28 obs, ratio = 28/32 = 0.875
# Input to VF: concat(z_t[32], t_emb[32]) = 64 dims
# VAE: reuse existing lat=32 VAE from job 49387885

python -u models/QuantumLatentCFM_v6.py \
    --phase=2 \
    --dataset=cifar10 \
    --vae-arch=resconv \
    --latent-dim=32 \
    --time-embed-dim=32 \
    --velocity-field=quantum \
    --n-circuits=1 \
    --n-qubits=8 \
    --encoding-type=sun \
    --vqc-type=qvit \
    --qvit-circuit=butterfly \
    --vqc-depth=2 \
    --k-local=2 \
    --obs-scheme=pairwise \
    --lr=1e-3 \
    --lr-H=1e-1 \
    --batch-size=32 \
    --epochs=200 \
    --n-train=50000 \
    --n-valtest=10000 \
    --seed=2025 \
    --ode-steps=100 \
    --n-samples=64 \
    --compute-metrics \
    --vae-ckpt=checkpoints/weights_vae_v2_cifar10_vae_v2_cifar_49387885.pt \
    --job-id=ratio_q_lat32_${SLURM_JOB_ID}
