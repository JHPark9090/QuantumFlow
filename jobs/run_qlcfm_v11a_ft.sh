#!/bin/bash
#SBATCH -A m4807_g
#SBATCH -C gpu&hbm80g
#SBATCH -q shared
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH -t 48:00:00
#SBATCH -J v11a_ft
#SBATCH -o /pscratch/sd/j/junghoon/QuantumFlow/logs/qlcfm_v11a_ft_%j.out
#SBATCH -e /pscratch/sd/j/junghoon/QuantumFlow/logs/qlcfm_v11a_ft_%j.err

export PYTHONNOUSERSITE=1
eval "$(conda shell.bash hook 2>/dev/null)"
conda activate /pscratch/sd/j/junghoon/conda-envs/qml_eeg

cd /pscratch/sd/j/junghoon/QuantumFlow

# QLCFM v11a_ft — CIFAR-10 32x32, fine-tuned SD3.5 VAE, bottleneck=64, concat + pairwise ANO k=2
# VAE: SD3.5 fine-tuned on CIFAR-10 32x32 (Recon FID 15.79, PSNR 26.40)
# latent_dim = 16 * (32/8)^2 = 256
# Bottleneck: 256 -> 64 -> quantum -> 64 -> 256
# Quantum: 4 qubits, SU(16) 255 generators, pairwise ANO k=2
#
# Resume: PREV_JOB_ID=XXXXX sbatch jobs/run_qlcfm_v11a_ft.sh

SD3_WEIGHTS="./checkpoints/weights_sd3vae_finetuned_sd3_finetune_49973661.pt"

PREV_JOB_ID="${PREV_JOB_ID:-}"
RESUME_FLAG=""
JOB_SUFFIX="${SLURM_JOB_ID}"
if [ -n "$PREV_JOB_ID" ]; then
    RESUME_FLAG="--resume"
    JOB_SUFFIX="${PREV_JOB_ID}"
    echo "Resuming from checkpoint: qlcfm_v11a_ft_${PREV_JOB_ID}"
fi

python -u models/QuantumLatentCFM_v11.py \
    --phase=2 \
    --dataset=cifar10 \
    --img-size=32 \
    --vae-arch=sd3 \
    --sd3-weights="${SD3_WEIGHTS}" \
    --bottleneck-dim=64 \
    --velocity-field=quantum \
    --n-qubits=4 \
    --vqc-type=none \
    --n-circuits=1 \
    --time-conditioning=concat \
    --time-embed-dim=64 \
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
    --job-id=qlcfm_v11a_ft_${JOB_SUFFIX}
