#!/bin/bash
#SBATCH --job-name=round2-qwen-72
#SBATCH --account=researchers
#SBATCH --partition=scavenge
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=05:00:00
#SBATCH --output=logs/%x.%j.out
#SBATCH --mail-type=BEGIN,END

echo "Host: $(hostname)"

set -euo pipefail

nvidia-smi

uv sync

SLURM_OUTPUT_FILE="logs/${SLURM_JOB_NAME}.${SLURM_JOB_ID}.out"

# ROUND 2


# COMMONSENSE SUB-RUNS

### qwen 72b
# # Disagreeing
uv run run.py --model_name qwen-2.5-72b --repetition 1 --round 2 --batch_size 256 --history --dataset_path /home/rp-fril-mhpe/subsampled_input_round2/commonsense/gpt/commonsense/qwen-2.5-72b_gpt_sender_disagree_subsampled.csv --outdir /home/rp-fril-mhpe  --slurm_output "${SLURM_OUTPUT_FILE}" --dataset commonsense
# # Agreeing
uv run run.py --model_name qwen-2.5-72b --repetition 1 --round 2 --batch_size 256 --history --dataset_path /home/rp-fril-mhpe/subsampled_input_round2/commonsense/gpt/commonsense/qwen-2.5-72b_gpt_sender_agree_subsampled.csv --outdir /home/rp-fril-mhpe  --slurm_output "${SLURM_OUTPUT_FILE}" --dataset commonsense

### llama 70b
# # Disagreeing
# uv run run.py --model_name llama-3.3-70b --repetition 1 --round 2 --batch_size 256 --history --dataset_path /home/rp-fril-mhpe/subsampled_input_round2/commonsense/gpt/commonsense/llama-3.3-70b_gpt_sender_disagree_subsampled.csv --outdir /home/rp-fril-mhpe  --slurm_output "${SLURM_OUTPUT_FILE}" --dataset commonsense
# # Agreeing
# uv run run.py --model_name llama-3.3-70b --repetition 1 --round 2 --batch_size 256 --history --dataset_path /home/rp-fril-mhpe/subsampled_input_round2/commonsense/gpt/commonsense/llama-3.3-70b_gpt_sender_agree_subsampled.csv --outdir /home/rp-fril-mhpe  --slurm_output "${SLURM_OUTPUT_FILE}" --dataset commonsense

### gemma 27b
# # Disagreeing
# uv run run.py --model_name gemma-3-27b --repetition 1 --round 2 --batch_size 256 --history --dataset_path /home/rp-fril-mhpe/subsampled_input_round2/commonsense/gpt/commonsense/gemma-3-27b_gpt_sender_disagree_subsampled.csv --outdir /home/rp-fril-mhpe  --slurm_output "${SLURM_OUTPUT_FILE}" --dataset commonsense
# # Agreeing
# uv run run.py --model_name gemma-3-27b --repetition 1 --round 2 --batch_size 256 --history --dataset_path /home/rp-fril-mhpe/subsampled_input_round2/commonsense/gpt/commonsense/gemma-3-27b_gpt_sender_agree_subsampled.csv --outdir /home/rp-fril-mhpe  --slurm_output "${SLURM_OUTPUT_FILE}" --dataset commonsense



