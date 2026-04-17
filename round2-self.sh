#!/bin/bash
#SBATCH --job-name=round2-self-sarcasm-gemma-3-27b
#SBATCH --account=researchers
#SBATCH --partition=scavenge
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x.%j.out
#SBATCH --mail-type=BEGIN,END

echo "Host: $(hostname)"

set -euo pipefail

nvidia-smi

uv sync

SLURM_OUTPUT_FILE="logs/${SLURM_JOB_NAME}.${SLURM_JOB_ID}.out"

# GRES 
# --gres=gpu:a100_80gb:1
# --gres=gpu:h100:1
# --gres=gpu:a30:1


# ROUND 2 : SELF-INTERACTION
# TEST
# uv run run.py \
#   --model_name gemma-3-4b \
#   --repetition 1 \
#   --round 2 \
#   --batch_size 256 \
#   --history \
#   --dataset_path /home/rp-fril-mhpe/subsampled_input_round2/sarcasm/gemma-3-4b-self-interaction_subsampled.csv \
#   --outdir /home/rp-fril-mhpe/tmp/test_run2 \
#   --no_logging \
#   -limit 10

# We run two times to see if successfully appends to file
# uv run run.py \
#   --model_name gemma-3-4b \
#   --repetition 1 \
#   --round 2 \
#   --batch_size 256 \
#   --history \
#   --dataset_path /home/rp-fril-mhpe/subsampled_input_round2/sarcasm/gemma-3-4b-self-interaction_subsampled.csv \
#   --outdir /home/rp-fril-mhpe/tmp/test_run2 \
#   --no_logging \
#   -limit 10

# RUNS

# SARCASM

# gemma-3-4b

# Agree 
#uv run run.py \
#  --model_name gemma-3-4b \
#  --repetition 1 \
#  --round 2 \
#  --batch_size 256 \
#  --history \
#  --dataset_path /home/rp-fril-mhpe/subsampled_input_round2/sarcasm/gemma-3-4b_self_interaction_agree_subsampled.csv \
#  --outdir /home/rp-fril-mhpe/self \
#  --slurm_output "${SLURM_OUTPUT_FILE}" 

# Disagree 
# uv run run.py \
#   --model_name gemma-3-4b \
#   --repetition 1 \
#   --round 2 \
#   --batch_size 256 \
#   --history \
#   --dataset_path /home/rp-fril-mhpe/subsampled_input_round2/sarcasm/gemma-3-4b_self_interaction_disagree_subsampled.csv \
#   --outdir /home/rp-fril-mhpe/self \
#   --slurm_output "${SLURM_OUTPUT_FILE}" 


# # gemma-3-27b
# agree 
uv run run.py \
  --model_name gemma-3-27b \
  --repetition 1 \
  --round 2 \
  --batch_size 256 \
  --history \
  --dataset_path /home/rp-fril-mhpe/subsampled_input_round2/sarcasm/gemma-3-27b_self_interaction_agree_subsampled.csv \
  --outdir /home/rp-fril-mhpe \
  --slurm_output "${SLURM_OUTPUT_FILE}" 

# disagree 
uv run run.py \
  --model_name gemma-3-27b \
  --repetition 1 \
  --round 2 \
  --batch_size 256 \
  --history \
  --dataset_path /home/rp-fril-mhpe/subsampled_input_round2/sarcasm/gemma-3-27b_self_interaction_disagree_subsampled.csv \
  --outdir /home/rp-fril-mhpe \
  --slurm_output "${SLURM_OUTPUT_FILE}" 



# # llama-3.3-70b
# agree 
# uv run run.py \
#   --model_name llama-3.3-70b \
#   --repetition 1 \
#   --round 2 \
#   --batch_size 256 \
#   --history \
#   --dataset_path /home/rp-fril-mhpe/subsampled_input_round2/sarcasm/llama-3.3-70b_self_interaction_agree_subsampled.csv \
#   --outdir /home/rp-fril-mhpe \
#   --slurm_output "${SLURM_OUTPUT_FILE}" 
# #disagree
# uv run run.py \
#   --model_name llama-3.3-70b \
#   --repetition 1 \
#   --round 2 \
#   --batch_size 256 \
#   --history \
#   --dataset_path /home/rp-fril-mhpe/subsampled_input_round2/sarcasm/llama-3.3-70b_self_interaction_disagree_subsampled.csv \
#   --outdir /home/rp-fril-mhpe \
#   --slurm_output "${SLURM_OUTPUT_FILE}" 

# # llama-3.1-8b
# agree
# uv run run.py \
#   --model_name llama-3.1-8b \
#   --repetition 1 \
#   --round 2 \
#   --batch_size 256 \
#   --history \
#   --dataset_path /home/rp-fril-mhpe/subsampled_input_round2/sarcasm/llama-3.1-8b_self_interaction_agree_subsampled.csv \
#   --outdir /home/rp-fril-mhpe/self \
#   --slurm_output "${SLURM_OUTPUT_FILE}" 

# uv run run.py \
#   --model_name llama-3.1-8b \
#   --repetition 1 \
#   --round 2 \
#   --batch_size 256 \
#   --history \
#   --dataset_path /home/rp-fril-mhpe/subsampled_input_round2/sarcasm/llama-3.1-8b_self_interaction_disagree_subsampled.csv \
#   --outdir /home/rp-fril-mhpe/self \
#   --slurm_output "${SLURM_OUTPUT_FILE}" 


# # qwen-2.5-72b
# agree 
# uv run run.py \
#   --model_name qwen-2.5-72b \
#   --repetition 1 \
#   --round 2 \
#   --batch_size 256 \
#   --history \
#   --dataset_path /home/rp-fril-mhpe/subsampled_input_round2/sarcasm/qwen-2.5-72b_self_interaction_agree_subsampled.csv \
#   --outdir /home/rp-fril-mhpe/self \
#   --slurm_output "${SLURM_OUTPUT_FILE}" 

# # disagree
# uv run run.py \
#   --model_name qwen-2.5-72b \
#   --repetition 1 \
#   --round 2 \
#   --batch_size 256 \
#   --history \
#   --dataset_path /home/rp-fril-mhpe/subsampled_input_round2/sarcasm/qwen-2.5-72b_self_interaction_disagree_subsampled.csv \
#   --outdir /home/rp-fril-mhpe/self \
#   --slurm_output "${SLURM_OUTPUT_FILE}" 


# qwen-2.5-7b
#agree 
# uv run run.py \
#   --model_name qwen-2.5-7b \
#   --repetition 1 \
#   --round 2 \
#   --batch_size 256 \
#   --history \
#   --dataset_path /home/rp-fril-mhpe/subsampled_input_round2/sarcasm/qwen-2.5-7b_self_interaction_agree_subsampled.csv \
#   --outdir /home/rp-fril-mhpe/self \
#   --slurm_output "${SLURM_OUTPUT_FILE}" 

# #disagree 
# uv run run.py \
#   --model_name qwen-2.5-7b \
#   --repetition 1 \
#   --round 2 \
#   --batch_size 256 \
#   --history \
#   --dataset_path /home/rp-fril-mhpe/subsampled_input_round2/sarcasm/qwen-2.5-7b_self_interaction_disagree_subsampled.csv \
#   --outdir /home/rp-fril-mhpe/self \
#   --slurm_output "${SLURM_OUTPUT_FILE}" 

#-------------------------------------------------------------------------------------------------

# COMMONSENSE

#----- gemma-3-4b

#---- gemma-3-27b

#---- llama-3.3-70b

#---- llama-3.1-8b

#---- qwen-2.5-72b

#----  qwen-2.5-7b

