#!/bin/bash
#SBATCH -A m4807_g
#SBATCH -C gpu&hbm80g
#SBATCH -q shared
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH -t 48:00:00
#SBATCH -J v11d
#SBATCH -o /pscratch/sd/j/junghoon/QuantumFlow/logs/qlcfm_v11d_%j.out
#SBATCH -e /pscratch/sd/j/junghoon/QuantumFlow/logs/qlcfm_v11d_%j.err

export PYTHONNOUSERSITE=1
eval "$(conda shell.bash hook 2>/dev/null)"
conda activate /pscratch/sd/j/junghoon/conda-envs/qml_eeg

cd /pscratch/sd/j/junghoon/QuantumFlow

# QLCFM v11d — CIFAR-10 128x128, SD3.5 VAE, bottleneck=256, additive + global ANO
# VAE: SD3.5 pretrained (16ch, 8x downsample), latent_dim=4096 (auto-computed)
# Bottleneck: Linear(4096→256) → quantum → Linear(256→4096)
# Quantum: 4 qubits, SU(16) 255 generators, vqc-type=none, 1 circuit
# Time: additive (t_emb added to z_bottleneck), time_embed_dim=256
# ANO: global k=4, 6 independent 16×16 Hermitians, 1536 ANO params
# enc_proj ratio: 256 → 255 (1.00:1)
#
# Resume: PREV_JOB_ID=XXXXX sbatch jobs/run_qlcfm_v11d.sh

PREV_JOB_ID="${PREV_JOB_ID:-}"
RESUME_FLAG=""
JOB_SUFFIX="${SLURM_JOB_ID}"
if [ -n "$PREV_JOB_ID" ]; then
    RESUME_FLAG="--resume"
    JOB_SUFFIX="${PREV_JOB_ID}"
    echo "Resuming from checkpoint: qlcfm_v11d_${PREV_JOB_ID}"
fi

python -u models/QuantumLatentCFM_v11.py \
    --phase=2 \
    --dataset=cifar10 \
    --img-size=128 \
    --vae-arch=sd3 \
    --bottleneck-dim=256 \
    --velocity-field=quantum \
    --n-qubits=4 \
    --vqc-type=none \
    --n-circuits=1 \
    --time-conditioning=additive \
    --time-embed-dim=256 \
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
    --job-id=qlcfm_v11d_${JOB_SUFFIX}
