#!/bin/bash
#SBATCH --job-name=round2-testrun-gemma4b
#SBATCH --account=researchers
#SBATCH --partition=scavenge
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=2:00:00
#SBATCH --constraint="gpu_a100_80gb|gpu_h100|gpu_l40s|gpu_a30"
#SBATCH --output=logs/%x.%j.out
#SBATCH --mail-type=BEGIN,END

echo "Host: $(hostname)"

set -euo pipefail

nvidia-smi

uv sync

SLURM_OUTPUT_FILE="logs/${SLURM_JOB_NAME}.${SLURM_JOB_ID}.out"

# ROUND 2
# model_name is the receiver, set the repetition to 1, set batch size to 256, enable --history, change the dataset path! and the outdir to: /home/rp-fril-mhpe/results-round2
# uv run run.py --model_name gemma-3-4b --repetition 1 --round 2 --batch_size 256 --history --dataset_path test_data_round2.csv --slurm_output "${SLURM_OUTPUT_FILE}"
uv run run.py --model_name gemma-3-4b --repetition 1 --round 2 --batch_size 256 --history --dataset_path /home/rp-fril-mhpe/subsampled_input_round2/sarcasm/gemma-3-4b_agree_subsampled.csv --outdir /home/rp-fril-mhpe  --slurm_output "${SLURM_OUTPUT_FILE}" -limit 10

