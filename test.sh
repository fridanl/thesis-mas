#!/bin/bash
#SBATCH --job-name=llama-3.1-8b
#SBATCH --account=researchers
#SBATCH --partition=scavenge
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --constraint="gpu_a100_80gb|gpu_h100"
#SBATCH --output=logs/%x.%j.out
#SBATCH --mail-type=BEGIN,END

echo "Host: $(hostname)"

set -euo pipefail

nvidia-smi

uv sync 

SLURM_OUTPUT_FILE="logs/${SLURM_JOB_NAME}.${SLURM_JOB_ID}.out"

# missing
# uv run run.py --model_name llama-3.1-8b --repetition 10 --round 1 --batch_size 256 --slurm_output "${SLURM_OUTPUT_FILE}"

# missing
# uv run run.py --model_name gemma-3-4b --repetition 10 --round 1 --batch_size 256 --slurm_output "${SLURM_OUTPUT_FILE}"

# missing
# uv run run.py --model_name gemma-3-27b --repetition 10 --round 1 --batch_size 256 --slurm_output "${SLURM_OUTPUT_FILE}"

# missing
# uv run run.py --model_name llama-3.3-70b --repetition 10 --round 1 --batch_size 256 --slurm_output "${SLURM_OUTPUT_FILE}"


