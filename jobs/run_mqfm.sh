#!/bin/bash
#SBATCH -A m4807_g
#SBATCH -J ModularQFM
#SBATCH -C gpu&hbm80g
#SBATCH --qos shared
#SBATCH -t 4:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=32
#SBATCH --chdir='/pscratch/sd/j/junghoon/QuantumFlow'
#SBATCH --output=/pscratch/sd/j/junghoon/QuantumFlow/logs/run_mqfm.out
#SBATCH -e /pscratch/sd/j/junghoon/QuantumFlow/logs/run_mqfm.e
#SBATCH --mail-user=utopie9090@snu.ac.kr
#!/bin/bash
set +x

cd /pscratch/sd/j/junghoon/QuantumFlow
module load python
conda activate /pscratch/sd/j/junghoon/conda-envs/qml_eeg
export PYTHONNOUSERSITE=1

# ── Full config: SU(N) encoding + QCNN + ANO ──
python models/ModularQFM.py \
    --dataset=mnist \
    --n-qubits=10 \
    --n-blocks=0 \
    --encoding-type=sun \
    --encoding-mode=direct \
    --vqc-type=qcnn \
    --vqc-depth=2 \
    --k-local=2 \
    --obs-scheme=sliding \
    --lr=1e-3 \
    --lr-H=1e-1 \
    --batch-size=32 \
    --epochs=30 \
    --seed=2025 \
    --n-train=1000 \
    --n-valtest=500 \
    --base-path='/pscratch/sd/j/junghoon/QuantumFlow' \
    --job-id='mqfm_sun_qcnn_ano'
