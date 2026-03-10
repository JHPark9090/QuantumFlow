#!/bin/bash
#SBATCH -A m4807_g
#SBATCH -C gpu&hbm80g
#SBATCH -q shared
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH -t 48:00:00
#SBATCH -J 10a_256
#SBATCH -o /pscratch/sd/j/junghoon/QuantumFlow/logs/qlcfm_v10a_lat256_%j.out
#SBATCH -e /pscratch/sd/j/junghoon/QuantumFlow/logs/qlcfm_v10a_lat256_%j.err

export PYTHONNOUSERSITE=1
eval "$(conda shell.bash hook 2>/dev/null)"
conda activate /pscratch/sd/j/junghoon/conda-envs/qml_eeg

cd /pscratch/sd/j/junghoon/QuantumFlow

# QLCFM v10a — CIFAR-10 32x32, lat256, concat conditioning, pairwise ANO
# VAE: v6 architecture, c_z=16, latent_dim=256 (weights_vae_v8_cifar_lat256.pt)
# Quantum: 4 qubits, SU(16) 255 generators, no QViT (vqc-type=none), 1 circuit
# Time: concat [z_t(256), t_emb(256)] = 512 input, time_embed_dim=256
# ANO: pairwise k=2, C(4,2)=6 observables
# enc_proj ratio: 512 → 255 (2.01:1)

PREV_JOB_ID="${PREV_JOB_ID:-}"
RESUME_FLAG=""
JOB_SUFFIX="${SLURM_JOB_ID}"
if [ -n "$PREV_JOB_ID" ]; then
    RESUME_FLAG="--resume"
    JOB_SUFFIX="${PREV_JOB_ID}"
    echo "Resuming from checkpoint: qlcfm_v10a_lat256_${PREV_JOB_ID}"
fi

VAE_CKPT="checkpoints/weights_vae_v8_cifar_lat256.pt"
if [ ! -f "$VAE_CKPT" ]; then
    echo "ERROR: VAE weights not found at $VAE_CKPT"
    exit 1
fi

python -u models/QuantumLatentCFM_v10.py \
    --phase=2 \
    --dataset=cifar10 \
    --img-size=32 \
    --vae-arch=v6 \
    --c-z=16 \
    --latent-dim=256 \
    --vae-ckpt="$VAE_CKPT" \
    --velocity-field=quantum \
    --n-qubits=4 \
    --vqc-type=none \
    --n-circuits=1 \
    --time-conditioning=concat \
    --time-embed-dim=256 \
    --ano-type=pairwise \
    --n-observables=6 \
    --lr=1e-3 \
    --lr-H=1e-1 \
    --epochs=200 \
    --batch-size=64 \
    --n-train=10000 \
    --n-valtest=2000 \
    --seed=2025 \
    --logit-normal-std=1.0 \
    --ode-solver=midpoint \
    --ode-steps=50 \
    --vf-ema-decay=0.999 \
    --compute-metrics \
    $RESUME_FLAG \
    --job-id=qlcfm_v10a_lat256_${JOB_SUFFIX}
