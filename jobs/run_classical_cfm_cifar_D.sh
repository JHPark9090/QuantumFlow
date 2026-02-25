#!/bin/bash
#SBATCH -A m4807_g
#SBATCH -C gpu&hbm80g
#SBATCH -q shared
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH -t 48:00:00
#SBATCH -J classical_cfm_D
#SBATCH -o /pscratch/sd/j/junghoon/QuantumFlow/logs/classical_cfm_D_%j.out
#SBATCH -e /pscratch/sd/j/junghoon/QuantumFlow/logs/classical_cfm_D_%j.err

export PYTHONNOUSERSITE=1
eval "$(conda shell.bash hook 2>/dev/null)"
conda activate /pscratch/sd/j/junghoon/conda-envs/qml_eeg

cd /pscratch/sd/j/junghoon/QuantumFlow

# Classical-D: Fair classical control for v8-Quantum
# Same enc_proj + vel_head as v8-quantum, classical MLP core replaces quantum
# Reuses v6 VAE (resconv, latent_dim=32) — phase=2 only
#
# Architecture comparison with v8-quantum:
#   enc_proj: Linear(64,1024) -> SiLU -> Linear(1024,765)   [IDENTICAL]
#   core:     Linear(765,64) -> SiLU -> Linear(64,28)        [REPLACES quantum]
#   vel_head: Linear(28,256) -> SiLU -> Linear(256,32)       [IDENTICAL]
#
# The ONLY difference is the 765->28 core:
#   v8-quantum:  SU(16) + QViT + ANO (1,102 params)
#   Classical-D: MLP 765->64->28      (50,876 params)

python models/QuantumLatentCFM_v8.py \
    --phase=2 \
    --dataset=cifar10 \
    --vae-arch=resconv \
    --latent-dim=32 \
    --velocity-field=classical_sandwich \
    --n-qubits=8 \
    --encoding-type=sun \
    --sun-group-size=4 \
    --enc-proj-hidden=1024 \
    --core-hidden=64 \
    --k-local=2 \
    --obs-scheme=pairwise \
    --time-embed-dim=32 \
    --lr=1e-3 \
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
    --job-id=classical_cfm_D_${SLURM_JOB_ID}
