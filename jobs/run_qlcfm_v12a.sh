#!/bin/bash
#SBATCH -A m4807_g
#SBATCH -C gpu&hbm80g
#SBATCH -q shared
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH -t 48:00:00
#SBATCH -J v12a
#SBATCH -o /pscratch/sd/j/junghoon/QuantumFlow/logs/qlcfm_v12a_%j.out
#SBATCH -e /pscratch/sd/j/junghoon/QuantumFlow/logs/qlcfm_v12a_%j.err

export PYTHONNOUSERSITE=1
eval "$(conda shell.bash hook 2>/dev/null)"
conda activate /pscratch/sd/j/junghoon/conda-envs/qml_eeg

cd /pscratch/sd/j/junghoon/QuantumFlow

# QLCFM v12a — CIFAR-10 128x128, SD3.5 VAE, bottleneck=256, v9 circuit
# VAE: SD3.5 pretrained (16ch, 8x downsample), latent_dim=4096 (auto-computed)
# Bottleneck: Linear(4096→256) → quantum → Linear(256→4096)
# Quantum: 8 qubits, brick-layer SU(4), 7 gates x 15 = 105 encoding params
# QViT: pyramid ansatz, depth=2
# Time: concat [z_bottleneck(256), t_emb(256)] = 512 input
# ANO: pairwise k=2, C(8,2)=28 wire groups, 4×4 Hermitians, 448 ANO params
# enc_proj ratio: 512 → 105 (4.88:1)
#
# Resume: PREV_JOB_ID=XXXXX sbatch jobs/run_qlcfm_v12a.sh

PREV_JOB_ID="${PREV_JOB_ID:-}"
RESUME_FLAG=""
JOB_SUFFIX="${SLURM_JOB_ID}"
if [ -n "$PREV_JOB_ID" ]; then
    RESUME_FLAG="--resume"
    JOB_SUFFIX="${PREV_JOB_ID}"
    echo "Resuming from checkpoint: qlcfm_v12a_${PREV_JOB_ID}"
fi

python -u models/QuantumLatentCFM_v12.py \
    --phase=2 \
    --dataset=cifar10 \
    --img-size=128 \
    --vae-arch=sd3 \
    --bottleneck-dim=256 \
    --velocity-field=quantum \
    --n-qubits=8 \
    --encoding-type=sun \
    --vqc-type=qvit \
    --qvit-circuit=pyramid \
    --vqc-depth=2 \
    --n-circuits=1 \
    --k-local=2 \
    --obs-scheme=pairwise \
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
    --job-id=qlcfm_v12a_${JOB_SUFFIX}
