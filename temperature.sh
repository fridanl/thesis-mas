#!/bin/bash
#SBATCH --job-name=temperature-llamahigh-r1
#SBATCH --account=researchers
#SBATCH --partition=scavenge
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=15:00:00
#SBATCH --output=logs/%x.%j.out
#SBATCH --mail-type=BEGIN,END

echo "Host: $(hostname)"

set -euo pipefail

nvidia-smi

uv sync 

SLURM_OUTPUT_FILE="logs/${SLURM_JOB_NAME}.${SLURM_JOB_ID}.out"

# tester
#uv run run.py --model_name tester --repetition 10 --round 1 --limit 20 --batch_size 256 --outdir /home/rp-fril-mhpe/tmp/temperature --slurm_output "${SLURM_OUTPUT_FILE}"

# ROUND 1
# high llama
uv run run.py --model_name llama-3.3-70b-high-temp --repetition 10 --round 1 --limit 50_000 --batch_size 256 --outdir /home/rp-fril-mhpe/temperature --slurm_output "${SLURM_OUTPUT_FILE}"
# high qwen
#uv run run.py --model_name qwen-2.5-72b-high-temp --repetition 10 --round 1 --limit 50_000 --batch_size 256 --outdir /home/rp-fril-mhpe/temperature --slurm_output "${SLURM_OUTPUT_FILE}"


# low llama
#uv run run.py --model_name llama-3.3-70b-low-temp --repetition 10 --round 1 --limit 50_000 --batch_size 256 --outdir /home/rp-fril-mhpe/temperature --slurm_output "${SLURM_OUTPUT_FILE}"
# low qwen
#uv run run.py --model_name qwen-2.5-72b-low-temp --repetition 10 --round 1 --limit 50_000 --batch_size 256 --outdir /home/rp-fril-mhpe/temperature --slurm_output "${SLURM_OUTPUT_FILE}"


# ROUND 2
# receiver default qwen (senders low and high llama in dataset)
# uv run run.py --model_name qwen-2.5-72b --repetition 1 --round 2 --batch_size 256 --history --dataset_path /home/rp-fril-mhpe/temperature/subsampled_input_round2/sarcasm/gemma-3-4b_disagree_subsampled.csv --outdir /home/rp-fril-mhpe/temperature  --slurm_output "${SLURM_OUTPUT_FILE}" --dataset sentiment

# receiver default llama (senders low and high qwen in dataset)
# uv run run.py --model_name llama-3.3-70b --repetition 1 --round 2 --batch_size 256 --history --dataset_path /home/rp-fril-mhpe/temperature/subsampled_input_round2/sarcasm/gemma-3-4b_disagree_subsampled.csv --outdir /home/rp-fril-mhpe/temperature  --slurm_output "${SLURM_OUTPUT_FILE}" --dataset sentiment
