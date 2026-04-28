#!/bin/bash
#SBATCH --job-name=temperature-test1
#SBATCH --account=researchers
#SBATCH --partition=scavenge
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x.%j.out
#SBATCH --mail-type=BEGIN,END

echo "Host: $(hostname)"

set -euo pipefail

nvidia-smi

uv sync 

SLURM_OUTPUT_FILE="logs/${SLURM_JOB_NAME}.${SLURM_JOB_ID}.out"

# tester

uv run run.py --model_name llama-3.3-70b --repetition 10 --round 1 --batch_size 256 --slurm_output "${SLURM_OUTPUT_FILE}"