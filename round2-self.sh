#!/bin/bash
#SBATCH --job-name=round2-self-sentiment-small
#SBATCH --account=researchers
#SBATCH --partition=scavenge
#SBATCH --gres=gpu:a30:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x.%j.out
#SBATCH --mail-type=BEGIN,END

echo "Host: $(hostname)"

nvidia-smi

uv sync

SLURM_OUTPUT_FILE="logs/${SLURM_JOB_NAME}.${SLURM_JOB_ID}.out"

# GRES 
# --gres=gpu:a100_80gb:1
# --gres=gpu:h100:1
# --gres=gpu:a30:1

# SENTIMENT 

# gemma-3-4b
# Agree 
uv run run.py \
 --model_name gemma-3-4b \
 --repetition 1 \
 --round 2 \
 --batch_size 256 \
 --history \
 --dataset sentiment \
 --dataset_path /home/rp-fril-mhpe/subsampled_input_round2/sentiment/gemma-3-4b_self_interaction_agree_subsampled.csv \
 --outdir /home/rp-fril-mhpe/self \
 --slurm_output "${SLURM_OUTPUT_FILE}" 

# Disagree 
uv run run.py \
  --model_name gemma-3-4b \
  --repetition 1 \
  --round 2 \
  --batch_size 256 \
  --history \
  --dataset sentiment \
  --dataset_path /home/rp-fril-mhpe/subsampled_input_round2/sentiment/gemma-3-4b_self_interaction_disagree_subsampled.csv \
  --outdir /home/rp-fril-mhpe/self \
  --slurm_output "${SLURM_OUTPUT_FILE}" 

# # gemma-3-27b
# agree 
# uv run run.py \
#   --model_name gemma-3-27b \
#   --repetition 1 \
#   --round 2 \
#   --batch_size 256 \
#   --history \
#   --dataset sentiment \
#   --dataset_path /home/rp-fril-mhpe/subsampled_input_round2/sentiment/gemma-3-27b_self_interaction_agree_subsampled.csv \
#   --outdir /home/rp-fril-mhpe/self \
#   --slurm_output "${SLURM_OUTPUT_FILE}" 

# disagree 
# uv run run.py \
#   --model_name gemma-3-27b \
#   --repetition 1 \
#   --round 2 \
#   --batch_size 256 \
#   --history \
#   --dataset sentiment \
#   --dataset_path /home/rp-fril-mhpe/subsampled_input_round2/sentiment/gemma-3-27b_self_interaction_disagree_subsampled.csv \
#   --outdir /home/rp-fril-mhpe/self \
#   --slurm_output "${SLURM_OUTPUT_FILE}" 


# # llama-3.3-70b
# agree 
# uv run run.py \
#   --model_name llama-3.3-70b \
#   --repetition 1 \
#   --round 2 \
#   --batch_size 256 \
#   --history \
#   --dataset sentiment \
#   --dataset_path /home/rp-fril-mhpe/subsampled_input_round2/sentiment/llama-3.3-70b_self_interaction_agree_subsampled.csv \
#   --outdir /home/rp-fril-mhpe/self \
#   --slurm_output "${SLURM_OUTPUT_FILE}" 
# # # #disagree
# uv run run.py \
#   --model_name llama-3.3-70b \
#   --repetition 1 \
#   --round 2 \
#   --batch_size 256 \
#   --history \
#   --dataset sentiment \
#   --dataset_path /home/rp-fril-mhpe/subsampled_input_round2/sentiment/llama-3.3-70b_self_interaction_disagree_subsampled.csv \
#   --outdir /home/rp-fril-mhpe/self \
#   --slurm_output "${SLURM_OUTPUT_FILE}" 

# # llama-3.1-8b
# agree
uv run run.py \
  --model_name llama-3.1-8b \
  --repetition 1 \
  --round 2 \
  --batch_size 256 \
  --history \
  --dataset sentiment \
  --dataset_path /home/rp-fril-mhpe/subsampled_input_round2/sentiment/llama-3.1-8b_self_interaction_agree_subsampled.csv \
  --outdir /home/rp-fril-mhpe/self \
  --slurm_output "${SLURM_OUTPUT_FILE}" 

# Disagree
uv run run.py \
  --model_name llama-3.1-8b \
  --repetition 1 \
  --round 2 \
  --batch_size 256 \
  --history \
  --dataset sentiment \
  --dataset_path /home/rp-fril-mhpe/subsampled_input_round2/sentiment/llama-3.1-8b_self_interaction_disagree_subsampled.csv \
  --outdir /home/rp-fril-mhpe/self \
  --slurm_output "${SLURM_OUTPUT_FILE}" 


# # qwen-2.5-72b
# agree 
# uv run run.py \
#   --model_name qwen-2.5-72b \
#   --repetition 1 \
#   --round 2 \
#   --batch_size 256 \
#   --history \
#   --dataset sentiment \
#   --dataset_path /home/rp-fril-mhpe/subsampled_input_round2/sentiment/qwen-2.5-72b_self_interaction_agree_subsampled.csv \
#   --outdir /home/rp-fril-mhpe/self \
#   --slurm_output "${SLURM_OUTPUT_FILE}" 

# # disagree
# uv run run.py \
#   --model_name qwen-2.5-72b \
#   --repetition 1 \
#   --round 2 \
#   --batch_size 256 \
#   --history \
#   --dataset sentiment \
#   --dataset_path /home/rp-fril-mhpe/subsampled_input_round2/sentiment/qwen-2.5-72b_self_interaction_disagree_subsampled.csv \
#   --outdir /home/rp-fril-mhpe/self \
#   --slurm_output "${SLURM_OUTPUT_FILE}" 


# qwen-2.5-7b
#agree 
uv run run.py \
  --model_name qwen-2.5-7b \
  --repetition 1 \
  --round 2 \
  --batch_size 256 \
  --history \
  --dataset sentiment \
  --dataset_path /home/rp-fril-mhpe/subsampled_input_round2/sentiment/qwen-2.5-7b_self_interaction_agree_subsampled.csv \
  --outdir /home/rp-fril-mhpe/self \
  --slurm_output "${SLURM_OUTPUT_FILE}" 

# #disagree 
uv run run.py \
  --model_name qwen-2.5-7b \
  --repetition 1 \
  --round 2 \
  --batch_size 256 \
  --history \
  --dataset sentiment \
  --dataset_path /home/rp-fril-mhpe/subsampled_input_round2/sentiment/qwen-2.5-7b_self_interaction_disagree_subsampled.csv \
  --outdir /home/rp-fril-mhpe/self \
  --slurm_output "${SLURM_OUTPUT_FILE}" 



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
# uv run run.py \
#   --model_name gemma-3-27b \
#   --repetition 1 \
#   --round 2 \
#   --batch_size 256 \
#   --history \
#   --dataset_path /home/rp-fril-mhpe/subsampled_input_round2/sarcasm/gemma-3-27b_self_interaction_agree_subsampled.csv \
#   --outdir /home/rp-fril-mhpe/self \
#   --slurm_output "${SLURM_OUTPUT_FILE}" 

# # disagree 
# uv run run.py \
#   --model_name gemma-3-27b \
#   --repetition 1 \
#   --round 2 \
#   --batch_size 256 \
#   --history \
#   --dataset_path /home/rp-fril-mhpe/subsampled_input_round2/sarcasm/gemma-3-27b_self_interaction_disagree_subsampled.csv \
#   --outdir /home/rp-fril-mhpe/self \
#   --slurm_output "${SLURM_OUTPUT_FILE}" 



# # llama-3.3-70b
# agree 
# uv run run.py \
#   --model_name llama-3.3-70b \
#   --repetition 1 \
#   --round 2 \
#   --batch_size 256 \
#   --history \
#   --dataset_path /home/rp-fril-mhpe/subsampled_input_round2/sarcasm/llama-3.3-70b_self_interaction_agree_subsampled.csv \
#   --outdir /home/rp-fril-mhpe/self \
#   --slurm_output "${SLURM_OUTPUT_FILE}" 
# #disagree
# uv run run.py \
#   --model_name llama-3.3-70b \
#   --repetition 1 \
#   --round 2 \
#   --batch_size 256 \
#   --history \
#   --dataset_path /home/rp-fril-mhpe/subsampled_input_round2/sarcasm/llama-3.3-70b_self_interaction_disagree_subsampled.csv \
#   --outdir /home/rp-fril-mhpe/self \
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

# gemma-3-4b
# Agree 
# uv run run.py \
#  --model_name gemma-3-4b \
#  --repetition 1 \
#  --round 2 \
#  --batch_size 256 \
#  --history \
#  --dataset commonsense \
#  --dataset_path /home/rp-fril-mhpe/subsampled_input_round2/commonsense/gemma-3-4b_self_interaction_agree_subsampled.csv \
#  --outdir /home/rp-fril-mhpe/self \
#  --slurm_output "${SLURM_OUTPUT_FILE}" 

# Disagree 
# uv run run.py \
#   --model_name gemma-3-4b \
#   --repetition 1 \
#   --round 2 \
#   --batch_size 256 \
#   --history \
#   --dataset commonsense \
#   --dataset_path /home/rp-fril-mhpe/subsampled_input_round2/commonsense/gemma-3-4b_self_interaction_disagree_subsampled.csv \
#   --outdir /home/rp-fril-mhpe/self \
#   --slurm_output "${SLURM_OUTPUT_FILE}" 

# # gemma-3-27b
# agree 
# uv run run.py \
#   --model_name gemma-3-27b \
#   --repetition 1 \
#   --round 2 \
#   --batch_size 256 \
#   --history \
#   --dataset commonsense \
#   --dataset_path /home/rp-fril-mhpe/subsampled_input_round2/commonsense/gemma-3-27b_self_interaction_agree_subsampled.csv \
#   --outdir /home/rp-fril-mhpe/self \
#   --slurm_output "${SLURM_OUTPUT_FILE}" 

# disagree 
# uv run run.py \
#   --model_name gemma-3-27b \
#   --repetition 1 \
#   --round 2 \
#   --batch_size 256 \
#   --history \
#   --dataset commonsense \
#   --dataset_path /home/rp-fril-mhpe/subsampled_input_round2/commonsense/gemma-3-27b_self_interaction_disagree_subsampled.csv \
#   --outdir /home/rp-fril-mhpe/self \
#   --slurm_output "${SLURM_OUTPUT_FILE}" 


# # llama-3.3-70b
# agree 
# uv run run.py \
#   --model_name llama-3.3-70b \
#   --repetition 1 \
#   --round 2 \
#   --batch_size 256 \
#   --history \
#   --dataset commonsense \
#   --dataset_path /home/rp-fril-mhpe/subsampled_input_round2/commonsense/llama-3.3-70b_self_interaction_agree_subsampled.csv \
#   --outdir /home/rp-fril-mhpe/self \
#   --slurm_output "${SLURM_OUTPUT_FILE}" 
# # # #disagree
# uv run run.py \
#   --model_name llama-3.3-70b \
#   --repetition 1 \
#   --round 2 \
#   --batch_size 256 \
#   --history \
#   --dataset commonsense \
#   --dataset_path /home/rp-fril-mhpe/subsampled_input_round2/commonsense/llama-3.3-70b_self_interaction_disagree_subsampled.csv \
#   --outdir /home/rp-fril-mhpe/self \
#   --slurm_output "${SLURM_OUTPUT_FILE}" 

# # llama-3.1-8b
# agree
# uv run run.py \
#   --model_name llama-3.1-8b \
#   --repetition 1 \
#   --round 2 \
#   --batch_size 256 \
#   --history \
#   --dataset commonsense \
#   --dataset_path /home/rp-fril-mhpe/subsampled_input_round2/commonsense/llama-3.1-8b_self_interaction_agree_subsampled.csv \
#   --outdir /home/rp-fril-mhpe/self \
#   --slurm_output "${SLURM_OUTPUT_FILE}" 

# uv run run.py \
#   --model_name llama-3.1-8b \
#   --repetition 1 \
#   --round 2 \
#   --batch_size 256 \
#   --history \
#   --dataset commonsense \
#   --dataset_path /home/rp-fril-mhpe/subsampled_input_round2/commonsense/llama-3.1-8b_self_interaction_disagree_subsampled.csv \
#   --outdir /home/rp-fril-mhpe/self \
#   --slurm_output "${SLURM_OUTPUT_FILE}" 


# # qwen-2.5-72b
# agree 
uv run run.py \
  --model_name qwen-2.5-72b \
  --repetition 1 \
  --round 2 \
  --batch_size 256 \
  --history \
  --dataset commonsense \
  --dataset_path /home/rp-fril-mhpe/subsampled_input_round2/commonsense/qwen-2.5-72b_self_interaction_agree_subsampled.csv \
  --outdir /home/rp-fril-mhpe/self \
  --slurm_output "${SLURM_OUTPUT_FILE}" 

# # disagree
uv run run.py \
  --model_name qwen-2.5-72b \
  --repetition 1 \
  --round 2 \
  --batch_size 256 \
  --history \
  --dataset commonsense \
  --dataset_path /home/rp-fril-mhpe/subsampled_input_round2/commonsense/qwen-2.5-72b_self_interaction_disagree_subsampled.csv \
  --outdir /home/rp-fril-mhpe/self \
  --slurm_output "${SLURM_OUTPUT_FILE}" 


# qwen-2.5-7b
#agree 
# uv run run.py \
#   --model_name qwen-2.5-7b \
#   --repetition 1 \
#   --round 2 \
#   --batch_size 256 \
#   --history \
#   --dataset commonsense \
#   --dataset_path /home/rp-fril-mhpe/subsampled_input_round2/commonsense/qwen-2.5-7b_self_interaction_agree_subsampled.csv \
#   --outdir /home/rp-fril-mhpe/self \
#   --slurm_output "${SLURM_OUTPUT_FILE}" 

# #disagree 
# uv run run.py \
#   --model_name qwen-2.5-7b \
#   --repetition 1 \
#   --round 2 \
#   --batch_size 256 \
#   --history \
#   --dataset commonsense \
#   --dataset_path /home/rp-fril-mhpe/subsampled_input_round2/commonsense/qwen-2.5-7b_self_interaction_disagree_subsampled.csv \
#   --outdir /home/rp-fril-mhpe/self \
#   --slurm_output "${SLURM_OUTPUT_FILE}" 


