#!/bin/bash
#SBATCH --job-name=test-failed-runs
#SBATCH --account=researchers
#SBATCH --partition=scavenge
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=3:00:00
#SBATCH --constraint="gpu_a100_80gb|gpu_h100|gpu_l40s|gpu_a30"
#SBATCH --output=logs/%x.%j.out
#SBATCH --mail-type=BEGIN,END

echo "Host: $(hostname)"

set -euo pipefail

nvidia-smi

uv sync 

SLURM_OUTPUT_FILE="logs/${SLURM_JOB_NAME}.${SLURM_JOB_ID}.out"

# done
# uv run run.py --model_name llama-3.1-8b --repetition 10 --round 1 --batch_size 256 --slurm_output "${SLURM_OUTPUT_FILE}"

# done
# uv run run.py --model_name gemma-3-4b --repetition 10 --round 1 --batch_size 256 --slurm_output "${SLURM_OUTPUT_FILE}"

# done
# uv run run.py --model_name gemma-3-27b --repetition 10 --round 1 --batch_size 256 --slurm_output "${SLURM_OUTPUT_FILE}"

# done
# uv run run.py --model_name llama-3.3-70b --repetition 10 --round 1 --batch_size 256 --slurm_output "${SLURM_OUTPUT_FILE}"

# round 2 test
# uv run run.py --model_name gemma-3-4b --repetition 1 --round 2 --batch_size 10 -limit 10 --history --dataset_path test_data.csv --slurm_output "${SLURM_OUTPUT_FILE}"


# run failed examples 
# sarcasm
uv run run.py --model_name llama-3.1-8b --repetition 1 --round 1 --batch_size 256 --dataset_path results/first/llama-3.1-8b-sarcasm-failed1.csv --slurm_output "${SLURM_OUTPUT_FILE}"
uv run run.py --model_name llama-3.3-70b --repetition 1 --round 1 --batch_size 256 --dataset_path results/first/llama-3.3-70b-sarcasm-failed1.csv  --slurm_output "${SLURM_OUTPUT_FILE}"
uv run run.py --model_name gemma-3-4b --repetition 1 --round 1 --batch_size 256 --dataset_path results/first/gemma-3-4b-sarcasm-failed1.csv  --slurm_output "${SLURM_OUTPUT_FILE}"
uv run run.py --model_name gemma-3-27b --repetition 1 --round 1 --batch_size 256 --dataset_path results/first/gemma-3-27b-sarcasm-failed1.csv --slurm_output "${SLURM_OUTPUT_FILE}"
uv run run.py --model_name qwen-2.5-7b --repetition 1 --round 1 --batch_size 256 --dataset_path results/first/qwen-2.5-7b-sarcasm-failed1.csv --slurm_output "${SLURM_OUTPUT_FILE}"
uv run run.py --model_name qwen-2.5-72b --repetition 1 --round 1 --batch_size 256 --dataset_path results/first/qwen-2.5-72b-sarcasm-failed1.csv --slurm_output "${SLURM_OUTPUT_FILE}"
#---
# commonsense
# uv run run.py --model_name gemma-3-4b --repetition 1 --round 1 --batch_size 256 --dataset_path results/first/gemma-3-4b-commonsense-failed1.csv --slurm_output "${SLURM_OUTPUT_FILE}"
# uv run run.py --model_name gemma-3-27b --repetition 1 --round 1 --batch_size 256 --dataset_path results/first/gemma-3-27b-commonsense-failed1.csv --slurm_output "${SLURM_OUTPUT_FILE}"
# uv run run.py --model_name qwen-2.5-7b --repetition 1 --round 1 --batch_size 256 --dataset_path results/first/qwen-2.5-7b-commonsense-failed1.csv --slurm_output "${SLURM_OUTPUT_FILE}"
# uv run run.py --model_name qwen-2.5-72b --repetition 1 --round 1 --batch_size 256 --dataset_path results/first/qwen-2.5-72b-commonsense-failed1.csv --slurm_output "${SLURM_OUTPUT_FILE}"
# uv run run.py --model_name llama-3.3-70b --repetition 1 --round 1 --batch_size 256 --dataset_path results/first/llama-3.3-70b-commonsense-failed1.csv --slurm_output "${SLURM_OUTPUT_FILE}"
# uv run run.py --model_name llama-3.1-8b --repetition 1 --round 1 --batch_size 256 --dataset_path results/first/llama-3.1-8b-commonsense-failed1.csv  --slurm_output "${SLURM_OUTPUT_FILE}"
