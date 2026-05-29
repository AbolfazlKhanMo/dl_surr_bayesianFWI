#!/bin/bash
# ===========================================================================
# launch_mps_sobol.sh — Ellipse DoF=3 (x_center, z_center, radius)
#
# Usage:  bash launch_mps_sobol.sh [BATCH_SIZE] [ITERATIONS]
#   BATCH_SIZE  — number of concurrent workers (default: 8)
#   ITERATIONS  — total Sobol samples (default: from config.py)
# ===========================================================================
set -euo pipefail

BATCH_SIZE=${1:-8}
ITERATIONS_ARG=""
if [ -n "${2:-}" ]; then
    ITERATIONS_ARG="--iterations $2"
fi
GPU_ID=0

echo "============================================="
echo " MPS Sobol Data Generation Launcher"
echo " GPU: ${GPU_ID}  |  Batch size: ${BATCH_SIZE}"
echo "============================================="

echo "[MPS] Stopping existing MPS daemon (if any)..."
echo quit | nvidia-cuda-mps-control 2>/dev/null || true
sleep 1

export CUDA_VISIBLE_DEVICES=${GPU_ID}

echo "[MPS] Starting MPS control daemon..."
nvidia-cuda-mps-control -d
sleep 2

THREAD_PCT=$(echo "get_default_active_thread_percentage" | nvidia-cuda-mps-control 2>/dev/null)
if [ -z "${THREAD_PCT}" ]; then
    echo "[ERROR] MPS daemon failed to start."
    exit 1
fi
echo "[MPS] Daemon is active (default thread %: ${THREAD_PCT})."

SM_PERCENT=$((100 / BATCH_SIZE))
USABLE_MB=6800
MEM_LIMIT_MB=$(( USABLE_MB / BATCH_SIZE ))

echo "[MPS] Per-client limits: ${SM_PERCENT}% SMs, ${MEM_LIMIT_MB} MB memory"
echo "set_default_active_thread_percentage ${SM_PERCENT}" | nvidia-cuda-mps-control
echo "set_default_device_pinned_mem_limit 0 ${MEM_LIMIT_MB}M" | nvidia-cuda-mps-control 2>/dev/null || \
    echo "[WARN] PINNED_MEM_LIMIT not supported on this driver version. Proceeding without memory cap."

echo ""
echo "[RUN] Launching 05_Sobol_dgen_mps.py with batch_size=${BATCH_SIZE}..."
python 05_Sobol_dgen_mps.py --batch-size "${BATCH_SIZE}" ${ITERATIONS_ARG}
EXIT_CODE=$?

echo ""
echo "[MPS] Shutting down MPS daemon..."
echo quit | nvidia-cuda-mps-control
sleep 1

echo "[DONE] Exit code: ${EXIT_CODE}"
exit ${EXIT_CODE}