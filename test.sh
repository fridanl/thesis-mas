#!/bin/bash
#SBATCH --job-name=test-run
#SBATCH --account=researchers
#SBATCH --partition=scavenge
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=04:00:00
#SBATCH --constraint="gpu_a100_80gb|gpu_h100"
#SBATCH --output=logs/%x.%j.out
#SBATCH --mail-type=BEGIN,END

echo "Host: $(hostname)"

set -euo pipefail

nvidia-smi

uv sync 

SLURM_OUTPUT_FILE="logs/${SLURM_JOB_NAME}.${SLURM_JOB_ID}.out"

#tjek
# uv run log_run.py --model_name llama-3.1-8b --repetition 10 --round 1 --batch_size 20 -limit 30 --slurm_output "${SLURM_OUTPUT_FILE}"

# tjek
# uv run log_run.py --model_name gemma-3-4b --repetition 10 --round 1 --batch_size 20 -limit 30 --slurm_output "${SLURM_OUTPUT_FILE}"

# mangler
# uv run log_run.py --model_name gemma-3-27b --repetition 10 --round 1 --batch_size 20 -limit 30 --slurm_output "${SLURM_OUTPUT_FILE}"

# mangler
uv run run.py --model_name llama-3.3-70b --repetition 2 --round 1 --batch_size 256 -limit 300 --slurm_output "${SLURM_OUTPUT_FILE}"

# mangler
#uv run log_run.py --model_name qwen-2.5-7b --repetition 10 --round 1 --batch_size 20 -limit 30 --slurm_output "${SLURM_OUTPUT_FILE}"

# mangler
#uv run log_run.py --model_name qwen-2.5-72b --repetition 10 --round 1 --batch_size 20 -limit 30 --slurm_output "${SLURM_OUTPUT_FILE}"

