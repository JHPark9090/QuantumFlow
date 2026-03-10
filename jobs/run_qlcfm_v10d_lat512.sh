#!/bin/bash
#SBATCH -A m4807_g
#SBATCH -C gpu&hbm80g
#SBATCH -q shared
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH -t 48:00:00
#SBATCH -J 10d_512
#SBATCH -o /pscratch/sd/j/junghoon/QuantumFlow/logs/qlcfm_v10d_lat512_%j.out
#SBATCH -e /pscratch/sd/j/junghoon/QuantumFlow/logs/qlcfm_v10d_lat512_%j.err

export PYTHONNOUSERSITE=1
eval "$(conda shell.bash hook 2>/dev/null)"
conda activate /pscratch/sd/j/junghoon/conda-envs/qml_eeg

cd /pscratch/sd/j/junghoon/QuantumFlow

# QLCFM v10d — CIFAR-10 32x32, lat512, additive conditioning, global ANO
# VAE: v6 architecture, c_z=32, latent_dim=512 (weights_vae_cifar_lat512.pt)
# Quantum: 4 qubits, SU(16) 255 generators, no QViT (vqc-type=none), 1 circuit
# Time: additive (t_emb added to z_t), time_embed_dim=512
# ANO: global, 6 observables (full 2^4=16 dim Hermitians)
# enc_proj ratio: 512 → 255 (2.01:1)

PREV_JOB_ID="${PREV_JOB_ID:-}"
RESUME_FLAG=""
JOB_SUFFIX="${SLURM_JOB_ID}"
if [ -n "$PREV_JOB_ID" ]; then
    RESUME_FLAG="--resume"
    JOB_SUFFIX="${PREV_JOB_ID}"
    echo "Resuming from checkpoint: qlcfm_v10d_lat512_${PREV_JOB_ID}"
fi

VAE_CKPT="checkpoints/weights_vae_cifar_lat512.pt"
if [ ! -f "$VAE_CKPT" ]; then
    echo "ERROR: VAE weights not found at $VAE_CKPT"
    exit 1
fi

python -u models/QuantumLatentCFM_v10.py \
    --phase=2 \
    --dataset=cifar10 \
    --img-size=32 \
    --vae-arch=v6 \
    --c-z=32 \
    --latent-dim=512 \
    --vae-ckpt="$VAE_CKPT" \
    --velocity-field=quantum \
    --n-qubits=4 \
    --vqc-type=none \
    --n-circuits=1 \
    --time-conditioning=additive \
    --time-embed-dim=512 \
    --ano-type=global \
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
    --job-id=qlcfm_v10d_lat512_${JOB_SUFFIX}
