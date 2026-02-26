#!/bin/bash
#SBATCH --job-name=test-run
#SBATCH --partition=scavenge
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --constraint="gpu_a100_40gb"
#SBATCH --output=logs/%x.%j.out
#SBATCH --mail-type=BEGIN,END

echo "Host: $(hostname)"

set -euo pipefail

nvidia-smi

uv sync 

SLURM_OUTPUT_FILE="logs/${SLURM_JOB_NAME}.${SLURM_JOB_ID}.out"

uv run test.py --model_name llama-3.1-8b --repetition 3 --round 1 --batch_size 2 -limit 6 --slurm_output "${SLURM_OUTPUT_FILE}"
