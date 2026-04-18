#!/bin/bash
#SBATCH --job-name=failed-run-gemma27b-test
#SBATCH --account=researchers
#SBATCH --partition=scavenge
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=2:00:00
#SBATCH --constraint="gpu_h100|gpu_a100_80gb|gpu_a100_40gb"
#SBATCH --output=logs/%x.%j.out
#SBATCH --mail-type=BEGIN,END


echo "Host: $(hostname)"

set -euo pipefail

nvidia-smi

uv sync 

SLURM_OUTPUT_FILE="logs/${SLURM_JOB_NAME}.${SLURM_JOB_ID}.out"

# failed runs for sentiment (gemma 27 and qwen 7)
# testrun
uv run run.py --model_name gemma-3-27b --repetition 1 --limit 10 --round 1 --batch_size 256 --dataset sentiment --dataset_path /home/rp-fril-mhpe/first/gemma-3-27b-sentiment-failed1.csv --no_logging --outdir /home/rp-fril-mhpe/tmp/failed --slurm_output "${SLURM_OUTPUT_FILE}"

# real runs
#uv run run.py --model_name gemma-3-27b --repetition 10 --round 1 --batch_size 256 --dataset sentiment --dataset_path /home/rp-fril-mhpe/first/gemma-3-27b-sentiment-failed1.csv --outdir /home/rp-fril-mhpe/ --slurm_output "${SLURM_OUTPUT_FILE}"

#uv run run.py --model_name qwen-2.5-7b --repetition 10 --round 1 --batch_size 256 --dataset sentiment --dataset_path /home/rp-fril-mhpe/first/qwen-2.5-7b-sentiment-failed1.csv --outdir /home/rp-fril-mhpe/ --slurm_output "${SLURM_OUTPUT_FILE}"
