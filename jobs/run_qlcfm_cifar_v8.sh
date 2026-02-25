#!/bin/bash
#SBATCH -A m4807_g
#SBATCH -C gpu&hbm80g
#SBATCH -q shared
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH -t 48:00:00
#SBATCH -J qlcfm_cifar_v8
#SBATCH -o /pscratch/sd/j/junghoon/QuantumFlow/logs/qlcfm_cifar_v8_%j.out
#SBATCH -e /pscratch/sd/j/junghoon/QuantumFlow/logs/qlcfm_cifar_v8_%j.err

export PYTHONNOUSERSITE=1
eval "$(conda shell.bash hook 2>/dev/null)"
conda activate /pscratch/sd/j/junghoon/conda-envs/qml_eeg

cd /pscratch/sd/j/junghoon/QuantumFlow

# v8-Quantum: SU(16) encoding on 8 qubits, single circuit
# Key differences from v6:
#   - SU(16) encoding: sun_group_size=4, 3 groups of 4 qubits, 765 enc params
#   - Larger enc_proj hidden: 1024 (vs v6's 256)
#   - Same QViT butterfly depth=2, ANO pairwise k=2 -> 28 obs
# Reuses v6 VAE (resconv, latent_dim=32) — phase=2 only
#
# Architecture:
#   time_mlp(32) + concat(z_t[32], t_emb[32]) = 64
#   -> enc_proj: Linear(64,1024) -> SiLU -> Linear(1024,765)
#   -> SU(16) encoding (8q) -> QViT butterfly (depth=2) -> ANO pairwise k=2
#   -> 28 obs -> vel_head: Linear(28,256) -> SiLU -> Linear(256,32)

python models/QuantumLatentCFM_v8.py \
    --phase=2 \
    --dataset=cifar10 \
    --vae-arch=resconv \
    --latent-dim=32 \
    --velocity-field=quantum \
    --n-circuits=1 \
    --n-qubits=8 \
    --encoding-type=sun \
    --sun-group-size=4 \
    --enc-proj-hidden=1024 \
    --vqc-type=qvit \
    --qvit-circuit=butterfly \
    --vqc-depth=2 \
    --k-local=2 \
    --obs-scheme=pairwise \
    --time-embed-dim=32 \
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
    --n-eval-samples=50000 \
    --vae-ckpt=checkpoints/weights_vae_qlcfm_cifar_v6_49235601.pt \
    --job-id=qlcfm_cifar_v8_${SLURM_JOB_ID}
