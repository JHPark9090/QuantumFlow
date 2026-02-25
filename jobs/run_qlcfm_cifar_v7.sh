#!/bin/bash
#SBATCH -A m4807_g
#SBATCH -C gpu&hbm80g
#SBATCH -q shared
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH -t 48:00:00
#SBATCH -J qlcfm_cifar_v7
#SBATCH -o /pscratch/sd/j/junghoon/QuantumFlow/logs/qlcfm_cifar_v7_%j.out
#SBATCH -e /pscratch/sd/j/junghoon/QuantumFlow/logs/qlcfm_cifar_v7_%j.err

export PYTHONNOUSERSITE=1
eval "$(conda shell.bash hook 2>/dev/null)"
conda activate /pscratch/sd/j/junghoon/conda-envs/qml_eeg

cd /pscratch/sd/j/junghoon/QuantumFlow

# v7: Multi-circuit quantum (8x8q), shared full input
# Identical classical parts to Classical-A:
#   - latent_dim=32, resconv VAE, time_embed_dim=32 + time_mlp
#   - enc_proj with SiLU, vel_head with SiLU
# Quantum: 8 circuits x 8q, SU(4) sun, QViT butterfly depth=2, ANO pairwise k=2
# Input: concat(z_t[32], t_emb[32]) = 64 dims (SHARED to all 8 circuits)
# Obs: 8*28=224, ratio = 224/32 = 7.0

python models/QuantumLatentCFM_v6.py \
    --phase=both \
    --dataset=cifar10 \
    --vae-arch=resconv \
    --latent-dim=32 \
    --beta=0.5 \
    --velocity-field=quantum \
    --n-circuits=8 \
    --n-qubits=8 \
    --encoding-type=sun \
    --vqc-type=qvit \
    --qvit-circuit=butterfly \
    --vqc-depth=2 \
    --k-local=2 \
    --obs-scheme=pairwise \
    --time-embed-dim=32 \
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
    --job-id=qlcfm_cifar_v7_${SLURM_JOB_ID}
