#!/bin/bash
#SBATCH --job-name=round2-gemma4b
#SBATCH --account=researchers
#SBATCH --partition=scavenge
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --constraint="gpu_a100_80gb|gpu_h100|gpu_l40s|gpu_a30"
#SBATCH --output=logs/%x.%j.out
#SBATCH --mail-type=BEGIN,END

echo "Host: $(hostname)"

set -euo pipefail

nvidia-smi

uv sync

SLURM_OUTPUT_FILE="logs/${SLURM_JOB_NAME}.${SLURM_JOB_ID}.out"

# ROUND 2

# TEST 
# uv run run.py --model_name gemma-3-4b --repetition 1 --round 2 --batch_size 256 --history --dataset_path test_data_round2.csv --slurm_output "${SLURM_OUTPUT_FILE}"


# RUNS

# SARCASM 

# gemma-3-4b
# Disagreeing
uv run run.py --model_name gemma-3-4b --repetition 1 --round 2 --batch_size 256 --history --dataset_path /home/rp-fril-mhpe/subsampled_input_round2/sarcasm/gemma-3-4b_disagree_subsampled.csv --outdir /home/rp-fril-mhpe  --slurm_output "${SLURM_OUTPUT_FILE}"
# Agreeing
uv run run.py --model_name gemma-3-4b --repetition 1 --round 2 --batch_size 256 --history --dataset_path /home/rp-fril-mhpe/subsampled_input_round2/sarcasm/gemma-3-4b_agree_subsampled.csv --outdir /home/rp-fril-mhpe  --slurm_output "${SLURM_OUTPUT_FILE}"

# # gemma-3-27b
# # Disagreeing
# uv run run.py --model_name gemma-3-27b --repetition 1 --round 2 --batch_size 256 --history --dataset_path /home/rp-fril-mhpe/subsampled_input_round2/sarcasm/gemma-3-27b_disagree_subsampled.csv --outdir /home/rp-fril-mhpe  --slurm_output "${SLURM_OUTPUT_FILE}"
# # Agreeing
# uv run run.py --model_name gemma-3-27b --repetition 1 --round 2 --batch_size 256 --history --dataset_path /home/rp-fril-mhpe/subsampled_input_round2/sarcasm/gemma-3-27b_agree_subsampled.csv --outdir /home/rp-fril-mhpe  --slurm_output "${SLURM_OUTPUT_FILE}"

# # llama-3.3-70b
# # Disagreeing
# uv run run.py --model_name llama-3.3-70b --repetition 1 --round 2 --batch_size 256 --history --dataset_path /home/rp-fril-mhpe/subsampled_input_round2/sarcasm/llama-3.3-70b_disagree_subsampled.csv --outdir /home/rp-fril-mhpe  --slurm_output "${SLURM_OUTPUT_FILE}"
# # Agreeing
# uv run run.py --model_name llama-3.3-70b --repetition 1 --round 2 --batch_size 256 --history --dataset_path /home/rp-fril-mhpe/subsampled_input_round2/sarcasm/llama-3.3-70b_agree_subsampled.csv --outdir /home/rp-fril-mhpe  --slurm_output "${SLURM_OUTPUT_FILE}"

# # llama-3.1-8b
# # Disagreeing
# uv run run.py --model_name llama-3.1-8b --repetition 1 --round 2 --batch_size 256 --history --dataset_path /home/rp-fril-mhpe/subsampled_input_round2/sarcasm/llama-3.1-8b_disagree_subsampled.csv --outdir /home/rp-fril-mhpe  --slurm_output "${SLURM_OUTPUT_FILE}"
# # Agreeing
# uv run run.py --model_name llama-3.1-8b --repetition 1 --round 2 --batch_size 256 --history --dataset_path /home/rp-fril-mhpe/subsampled_input_round2/sarcasm/llama-3.1-8b_agree_subsampled.csv --outdir /home/rp-fril-mhpe  --slurm_output "${SLURM_OUTPUT_FILE}"

# # qwen-2.5-72b
# # Disagreeing
# uv run run.py --model_name qwen-2.5-72b --repetition 1 --round 2 --batch_size 256 --history --dataset_path /home/rp-fril-mhpe/subsampled_input_round2/sarcasm/qwen-2.5-72b_disagree_subsampled.csv --outdir /home/rp-fril-mhpe  --slurm_output "${SLURM_OUTPUT_FILE}"
# # Agreeing
# uv run run.py --model_name qwen-2.5-72b --repetition 1 --round 2 --batch_size 256 --history --dataset_path /home/rp-fril-mhpe/subsampled_input_round2/sarcasm/qwen-2.5-72b_agree_subsampled.csv --outdir /home/rp-fril-mhpe  --slurm_output "${SLURM_OUTPUT_FILE}"

# # qwen-2.5-7b
# # Disagreeing
# uv run run.py --model_name qwen-2.5-7b --repetition 1 --round 2 --batch_size 256 --history --dataset_path /home/rp-fril-mhpe/subsampled_input_round2/sarcasm/qwen-2.5-7b_disagree_subsampled.csv --outdir /home/rp-fril-mhpe  --slurm_output "${SLURM_OUTPUT_FILE}"
# # Agreeing
# uv run run.py --model_name qwen-2.5-7b --repetition 1 --round 2 --batch_size 256 --history --dataset_path /home/rp-fril-mhpe/subsampled_input_round2/sarcasm/qwen-2.5-7b_agree_subsampled.csv --outdir /home/rp-fril-mhpe  --slurm_output "${SLURM_OUTPUT_FILE}"



