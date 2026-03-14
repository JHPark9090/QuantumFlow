#!/bin/bash
#SBATCH -A m4807_g
#SBATCH -C gpu&hbm80g
#SBATCH -q shared
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH -t 48:00:00
#SBATCH -J v13b
#SBATCH -o /pscratch/sd/j/junghoon/QuantumFlow/logs/qlcfm_v13b_%j.out
#SBATCH -e /pscratch/sd/j/junghoon/QuantumFlow/logs/qlcfm_v13b_%j.err

export PYTHONNOUSERSITE=1
eval "$(conda shell.bash hook 2>/dev/null)"
conda activate /pscratch/sd/j/junghoon/conda-envs/qml_eeg

cd /pscratch/sd/j/junghoon/QuantumFlow

# QLCFM v13b — Multi-Chip Ensemble, additive time conditioning (no enc_proj)
# SD3.5 VAE (no bottleneck), 16 chips x 4q SU(16), pairwise ANO k=2
# Per chip: (chunk(256) + t_emb(256))[:, :255] = 255 (1.00:1, direct slice)
# Per chip output: 6 ANO obs → vel_head → 256, concat all → 4096
#
# Resume: PREV_JOB_ID=XXXXX sbatch jobs/run_qlcfm_v13b.sh

PREV_JOB_ID="${PREV_JOB_ID:-}"
RESUME_FLAG=""
JOB_SUFFIX="${SLURM_JOB_ID}"
if [ -n "$PREV_JOB_ID" ]; then
    RESUME_FLAG="--resume"
    JOB_SUFFIX="${PREV_JOB_ID}"
    echo "Resuming from checkpoint: qlcfm_v13b_${PREV_JOB_ID}"
fi

python -u models/QuantumLatentCFM_v13.py \
    --phase=2 \
    --dataset=cifar10 \
    --img-size=128 \
    --vae-arch=sd3 \
    --n-chips=16 \
    --n-qubits=4 \
    --k-local=2 \
    --time-conditioning=additive \
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
    --job-id=qlcfm_v13b_${JOB_SUFFIX}
