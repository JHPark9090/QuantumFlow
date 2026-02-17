#!/bin/bash
#SBATCH -A m4807_g
#SBATCH -J QuantumFlowMatching
#SBATCH -C gpu&hbm80g
#SBATCH --qos shared
#SBATCH -t 4:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=32
#SBATCH --chdir='/pscratch/sd/j/junghoon/QuantumFlow'
#SBATCH --output=/pscratch/sd/j/junghoon/QuantumFlow/logs/run_qfm.out
#SBATCH -e /pscratch/sd/j/junghoon/QuantumFlow/logs/run_qfm.e
#SBATCH --mail-user=utopie9090@snu.ac.kr
#!/bin/bash
set +x

cd /pscratch/sd/j/junghoon/QuantumFlow
module load python
conda activate /pscratch/sd/j/junghoon/conda-envs/qml_eeg
export PYTHONNOUSERSITE=1

python models/QuantumFlowMatching.py \
    --n-qubits=4 \
    --n-blocks=2 \
    --encoding-type=sun \
    --k-local=2 \
    --obs-scheme=sliding \
    --use-variational \
    --vqc-depth=1 \
    --lr=1e-3 \
    --lr-H=1e-1 \
    --batch-size=32 \
    --epochs=30 \
    --seed=2025 \
    --n-samples=500 \
    --noise=0.15 \
    --base-path='/pscratch/sd/j/junghoon/QuantumFlow' \
    --job-id='qfm_sun_ano_var'
