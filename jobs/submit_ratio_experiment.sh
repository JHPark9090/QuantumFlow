#!/bin/bash
# =============================================================================
# Ratio Bottleneck Experiment — Master Submission Script
# =============================================================================
# Submits all 8 jobs with SLURM dependency chains:
#   - VAE lat=64 and lat=128 start immediately (in parallel)
#   - CFM lat=32 (quantum + classical) depend on existing VAE job 49387885
#   - CFM lat=64 jobs depend on VAE lat=64
#   - CFM lat=128 jobs depend on VAE lat=128
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EXISTING_VAE32_JOB=49387885

echo "=== Ratio Bottleneck Experiment ==="
echo ""

# --- Step 1: Submit VAE training (lat=64 and lat=128) ---
echo "--- Submitting VAE training jobs ---"

VAE64_JOB=$(sbatch --parsable "${SCRIPT_DIR}/run_ratio_vae_lat64.sh")
echo "  VAE lat=64:  job ${VAE64_JOB}"

VAE128_JOB=$(sbatch --parsable "${SCRIPT_DIR}/run_ratio_vae_lat128.sh")
echo "  VAE lat=128: job ${VAE128_JOB}"

# --- Step 2: Submit CFM lat=32 ---
# VAE lat=32 (job 49387885) already completed; no dependency needed.
echo ""
echo "--- Submitting CFM lat=32 (VAE already completed) ---"

Q32_JOB=$(sbatch --parsable "${SCRIPT_DIR}/run_ratio_quantum_lat32.sh")
echo "  Quantum lat=32:   job ${Q32_JOB}"

C32_JOB=$(sbatch --parsable "${SCRIPT_DIR}/run_ratio_classical_lat32.sh")
echo "  Classical lat=32:  job ${C32_JOB}"

# --- Step 3: Submit CFM lat=64 (depends on VAE lat=64) ---
echo ""
echo "--- Submitting CFM lat=64 (depends on VAE job ${VAE64_JOB}) ---"

Q64_JOB=$(sbatch --parsable --dependency=afterok:${VAE64_JOB} \
    "${SCRIPT_DIR}/run_ratio_quantum_lat64.sh")
echo "  Quantum lat=64:   job ${Q64_JOB}"

C64_JOB=$(sbatch --parsable --dependency=afterok:${VAE64_JOB} \
    "${SCRIPT_DIR}/run_ratio_classical_lat64.sh")
echo "  Classical lat=64:  job ${C64_JOB}"

# --- Step 4: Submit CFM lat=128 (depends on VAE lat=128) ---
echo ""
echo "--- Submitting CFM lat=128 (depends on VAE job ${VAE128_JOB}) ---"

Q128_JOB=$(sbatch --parsable --dependency=afterok:${VAE128_JOB} \
    "${SCRIPT_DIR}/run_ratio_quantum_lat128.sh")
echo "  Quantum lat=128:  job ${Q128_JOB}"

C128_JOB=$(sbatch --parsable --dependency=afterok:${VAE128_JOB} \
    "${SCRIPT_DIR}/run_ratio_classical_lat128.sh")
echo "  Classical lat=128: job ${C128_JOB}"

# --- Summary ---
echo ""
echo "=== All jobs submitted ==="
echo ""
echo "  VAE jobs:       ${EXISTING_VAE32_JOB} (lat32, existing), ${VAE64_JOB} (lat64), ${VAE128_JOB} (lat128)"
echo "  Quantum CFM:    ${Q32_JOB} (lat32), ${Q64_JOB} (lat64), ${Q128_JOB} (lat128)"
echo "  Classical CFM:  ${C32_JOB} (lat32), ${C64_JOB} (lat64), ${C128_JOB} (lat128)"
echo ""
echo "  Monitor: squeue -u \$USER -n ratio_vae_lat64,ratio_vae_lat128,ratio_q_lat32,ratio_q_lat64,ratio_q_lat128,ratio_c_lat32,ratio_c_lat64,ratio_c_lat128"
echo ""
echo "  After completion, run:"
echo "    python experiments/ratio_bottleneck/analyze_results.py"
