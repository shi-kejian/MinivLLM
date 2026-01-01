#!/bin/bash
#SBATCH --job-name=minivbench
#SBATCH --partition=hpc-low
#SBATCH --nodes=4
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --output=logs/bench_%J.log
#SBATCH --error=logs/bench_%J.log

# 16 NVIDIA GB200 GPUs

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "$LOG_DIR"

# Environment
export CUDA_HOME=/usr/local/cuda-12.8
export TRITON_CACHE_DIR=/tmp/triton_${SLURM_JOBID:-$$}
export PYTHONPATH="${SCRIPT_DIR}/src:${PYTHONPATH}"

# NCCL settings for multi-node
export NCCL_DEBUG=WARN
export NCCL_IB_TIMEOUT=23
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800

echo "========================================"
echo "MinivLLM GB200 Benchmark"
echo "========================================"
echo "Date: $(date)"
echo "Nodes: ${SLURM_JOB_NUM_NODES:-1}"
echo "GPUs per node: 4"
echo "Total GPUs: $((${SLURM_JOB_NUM_NODES:-1} * 4))"
echo "Job ID: ${SLURM_JOBID:-local}"
echo "========================================"

if [ -n "$SLURM_JOB_ID" ]; then
    echo "Running in Slurm batch mode"
    PYTHON_CMD="srun --ntasks=1 --nodes=1 python"
else
    echo "Running in local mode"
    PYTHON_CMD="python"
fi

source ~/.bashrc 2>/dev/null || true
if command -v conda &> /dev/null; then
    conda activate minivllm 2>/dev/null || echo "Note: minivllm env not found, using current env"
fi

echo ""
echo "Python: $(which python)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU count: $(python -c 'import torch; print(torch.cuda.device_count())')"

if python -c 'import torch; assert torch.cuda.is_available()' 2>/dev/null; then
    echo "GPU 0: $(python -c 'import torch; print(torch.cuda.get_device_name(0))')"
    echo "GPU Memory: $(python -c 'import torch; print(f\"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB\")')"
fi
echo ""

echo "========================================"
echo "1. Prefilling Benchmark (Flash Attention)"
echo "========================================"
cd "$SCRIPT_DIR"
$PYTHON_CMD benchmark_prefilling.py 2>&1 | tee -a "$LOG_DIR/prefilling_${SLURM_JOBID:-local}.log"

echo ""
echo "========================================"
echo "2. Decoding Benchmark (Paged Attention)"
echo "========================================"
$PYTHON_CMD benchmark_decoding.py 2>&1 | tee -a "$LOG_DIR/decoding_${SLURM_JOBID:-local}.log"

echo ""
echo "========================================"
echo "3. Full Inference Demo"
echo "========================================"
$PYTHON_CMD main.py 2>&1 | tee -a "$LOG_DIR/inference_${SLURM_JOBID:-local}.log"

echo ""
echo "========================================"
echo "Benchmark Complete"
echo "========================================"
echo "Logs saved to: $LOG_DIR"
