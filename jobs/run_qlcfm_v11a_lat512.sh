#!/bin/bash
#SBATCH -A m4807_g
#SBATCH -C gpu&hbm80g
#SBATCH -q shared
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH -t 48:00:00
#SBATCH -J 11a_512
#SBATCH -o /pscratch/sd/j/junghoon/QuantumFlow/logs/qlcfm_v11a_lat512_%j.out
#SBATCH -e /pscratch/sd/j/junghoon/QuantumFlow/logs/qlcfm_v11a_lat512_%j.err

export PYTHONNOUSERSITE=1
eval "$(conda shell.bash hook 2>/dev/null)"
conda activate /pscratch/sd/j/junghoon/conda-envs/qml_eeg

cd /pscratch/sd/j/junghoon/QuantumFlow

# QLCFM v11a lat512 — CIFAR-10 128x128, SD3.5 VAE, bottleneck=512, concat + pairwise ANO k=2
# Bottleneck: Linear(4096→512) → quantum → Linear(512→4096)
# Quantum: 4 qubits, SU(16) 255 generators, vqc-type=none, 1 circuit
# Time: concat [z_bottleneck(512), t_emb(512)] = 1024 input
# ANO: pairwise k=2, C(4,2)=6 wire groups, 4×4 Hermitians, 96 ANO params
# enc_proj ratio: 1024 → 255 (4.02:1)
#
# Resume: PREV_JOB_ID=XXXXX sbatch jobs/run_qlcfm_v11a_lat512.sh

PREV_JOB_ID="${PREV_JOB_ID:-}"
RESUME_FLAG=""
JOB_SUFFIX="${SLURM_JOB_ID}"
if [ -n "$PREV_JOB_ID" ]; then
    RESUME_FLAG="--resume"
    JOB_SUFFIX="${PREV_JOB_ID}"
    echo "Resuming from checkpoint: qlcfm_v11a_lat512_${PREV_JOB_ID}"
fi

python -u models/QuantumLatentCFM_v11.py \
    --phase=2 \
    --dataset=cifar10 \
    --img-size=128 \
    --vae-arch=sd3 \
    --bottleneck-dim=512 \
    --velocity-field=quantum \
    --n-qubits=4 \
    --vqc-type=none \
    --n-circuits=1 \
    --time-conditioning=concat \
    --time-embed-dim=512 \
    --ano-type=pairwise \
    --k-local=2 \
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
    --job-id=qlcfm_v11a_lat512_${JOB_SUFFIX}
