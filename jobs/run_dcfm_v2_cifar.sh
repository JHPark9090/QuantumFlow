#!/bin/bash
#SBATCH -A m4807_g
#SBATCH -C gpu&hbm80g
#SBATCH -q shared
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH -t 48:00:00
#SBATCH -J dcfm2_cif
#SBATCH -o /pscratch/sd/j/junghoon/QuantumFlow/logs/dcfm_v2_cifar_%j.out
#SBATCH -e /pscratch/sd/j/junghoon/QuantumFlow/logs/dcfm_v2_cifar_%j.err

export PYTHONNOUSERSITE=1
eval "$(conda shell.bash hook 2>/dev/null)"
conda activate /pscratch/sd/j/junghoon/conda-envs/qml_eeg

cd /pscratch/sd/j/junghoon/QuantumFlow

# Quantum Direct CFM v2 — CIFAR-10 32x32
# Single SU(64), 6 qubits, no QViT, pairwise ANO k=2
# auto enc_channels=[32,64] -> d_flat=4096, enc_proj ratio 1.00:1
#
# Resume: PREV_JOB_ID=XXXXX sbatch jobs/run_dcfm_v2_cifar.sh

PREV_JOB_ID="${PREV_JOB_ID:-}"

RESUME_FLAG=""
JOB_SUFFIX="${SLURM_JOB_ID}"
if [ -n "$PREV_JOB_ID" ]; then
    RESUME_FLAG="--resume"
    JOB_SUFFIX="dcfm2_cifar_${PREV_JOB_ID}"
    echo "Resuming from checkpoint: dcfm2_cifar_${PREV_JOB_ID}"
fi

python -u models/QuantumDirectCFM_v2.py \
    --dataset=cifar10 \
    --img-size=32 \
    --velocity-field=quantum \
    --n-qubits=6 \
    --vqc-type=none \
    --k-local=2 \
    --obs-scheme=pairwise \
    --enc-channels=auto \
    --time-embed-dim=256 \
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
    --job-id=dcfm2_cifar_${JOB_SUFFIX}
