#!/bin/bash
#SBATCH -A m4807_g
#SBATCH -C gpu&hbm80g
#SBATCH -q shared
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH -t 48:00:00
#SBATCH -J v10a_cif
#SBATCH -o /pscratch/sd/j/junghoon/QuantumFlow/logs/qlcfm_v10a_cifar_%j.out
#SBATCH -e /pscratch/sd/j/junghoon/QuantumFlow/logs/qlcfm_v10a_cifar_%j.err

export PYTHONNOUSERSITE=1
eval "$(conda shell.bash hook 2>/dev/null)"
conda activate /pscratch/sd/j/junghoon/conda-envs/qml_eeg

cd /pscratch/sd/j/junghoon/QuantumFlow

# QLCFM v10a — CIFAR-10, lat128, concat conditioning, pairwise ANO
# VAE: resconv lat128 (completed, 200 epochs, all 128 dims active)
# Quantum: 4 qubits SU(16), 255 generators, no QViT
# Time: concat [z_t(128), t_emb(128)] = 256 input → 255 enc (1.00:1)
# ANO: pairwise k=2, C(4,2)=6 observables, 4×4 Hermitians, 96 ANO params
#
# Resume: PREV_JOB_ID=XXXXX sbatch jobs/run_qlcfm_v10a.sh

PREV_JOB_ID="${PREV_JOB_ID:-}"

RESUME_FLAG=""
JOB_SUFFIX="${SLURM_JOB_ID}"
if [ -n "$PREV_JOB_ID" ]; then
    RESUME_FLAG="--resume"
    JOB_SUFFIX="${PREV_JOB_ID}"
    echo "Resuming from checkpoint: v10a_cifar_${PREV_JOB_ID}"
fi

VAE_CKPT="checkpoints/weights_vae_v2_cifar10_ratio_lat128.pt"
if [ ! -f "$VAE_CKPT" ]; then
    echo "ERROR: VAE weights not found at $VAE_CKPT"
    exit 1
fi

python -u models/QuantumLatentCFM_v10.py \
    --phase=2 \
    --dataset=cifar10 \
    --img-size=32 \
    --latent-dim=128 \
    --vae-arch=resconv \
    --vae-ckpt="$VAE_CKPT" \
    --velocity-field=quantum \
    --n-qubits=4 \
    --n-circuits=1 \
    --vqc-type=none \
    --time-conditioning=concat \
    --time-embed-dim=128 \
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
    --job-id=v10a_cifar_${JOB_SUFFIX}
