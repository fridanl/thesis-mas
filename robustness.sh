#!/bin/bash 
#SBATCH --job-name=test-llama-qwen-no-history
#SBATCH --account=researchers
#SBATCH --partition=scavenge
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x.%j.out
#SBATCH --mail-type=BEGIN,END
#SBATCH --gres=gpu:h100:1



echo "Host: $(hostname)"

set -euo pipefail

SLURM_OUTPUT_FILE="logs/${SLURM_JOB_NAME}.${SLURM_JOB_ID}.out"



# llama

# NO HISTORY 
uv run run.py \
  --model_name llama-3.3-70b \
  --repetition 1 \
  --round 2 \
  --batch_size 256 \
  --dataset sarcasm \
  --dataset_path /home/rp-fril-mhpe/subsampled_input_round2/robustness/sarcasm/llama-3.3-70b_disagree_subsampled.csv \
  --outdir /home/rp-fril-mhpe/no_history \
  --slurm_output "${SLURM_OUTPUT_FILE}" \
  -limit 10


# NO EXPLANATION BUT WITH HISTORY
uv run run.py \
  --model_name llama-3.3-70b \
  --repetition 1 \
  --round 2 \
  --batch_size 256 \
  --dataset sarcasm \
  --history \
  --no_explanation \
  --dataset_path /home/rp-fril-mhpe/subsampled_input_round2/robustness/sarcasm/llama-3.3-70b_disagree_subsampled.csv \
  --outdir /home/rp-fril-mhpe/no_explanation \
  --slurm_output "${SLURM_OUTPUT_FILE}" \
  -limit 10




# export PYTHONPATH=/home/fril/thesis-mas${PYTHONPATH:+:$PYTHONPATH}
# uv run src/make_match_type.py
