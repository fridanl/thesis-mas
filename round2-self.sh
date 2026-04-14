#!/bin/bash
#SBATCH --job-name=round2-gemma-3-27b-self
#SBATCH --account=researchers
#SBATCH --partition=scavenge
#SBATCH --gres=gpu:a100_80gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x.%j.out
#SBATCH --mail-type=BEGIN,END

echo "Host: $(hostname)"

set -euo pipefail

nvidia-smi

uv sync

SLURM_OUTPUT_FILE="logs/${SLURM_JOB_NAME}.${SLURM_JOB_ID}.out"

# ROUND 2 : SELF-INTERACTION
# TEST
uv run run.py \
  --model_name gemma-3-4b \
  --repetition 1 \
  --round 2 \
  --batch_size 256 \
  --history \
  --dataset_path /home/rp-fril-mhpe/subsampled_input_round2/sarcasm/gemma-3-4b-self-interaction.csv \
  --outdir /home/rp-fril-mhpe/tmp/test_run2 \
  --no_logging \
  --limit 10

# We run two times to see if successfully appends to file
uv run run.py \
  --model_name gemma-3-4b \
  --repetition 1 \
  --round 2 \
  --batch_size 256 \
  --history \
  --dataset_path /home/rp-fril-mhpe/subsampled_input_round2/sarcasm/gemma-3-4b-self-interaction.csv \
  --outdir /home/rp-fril-mhpe/tmp/test_run2 \
  --no_logging \
  --limit 10

# RUNS

# SARCASM

# gemma-3-4b

# # gemma-3-27b

# # llama-3.3-70b

# # llama-3.1-8b

# # qwen-2.5-72b

# # qwen-2.5-7b


#-------------------------------------------------------------------------------------------------

# COMMONSENSE

#----- gemma-3-4b

#---- gemma-3-27b

#---- llama-3.3-70b

#---- llama-3.1-8b

#---- qwen-2.5-72b

#----  qwen-2.5-7b

