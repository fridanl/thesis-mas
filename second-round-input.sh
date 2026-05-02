#!/bin/bash
#SBATCH --job-name=temperature-make_2nd_round_input
#SBATCH --account=researchers
#SBATCH --partition=scavenge
#SBATCH --cpus-per-task=4
#SBATCH --time=1:00:00
#SBATCH --output=logs/%x.%j.out
#SBATCH --mail-type=BEGIN,END

echo "Host: $(hostname)"

set -euo pipefail

#uv run -m src.make_second_round_input --dataset commonsense --output_root /home/rp-fril-mhpe/tmp


# SARCASM SELF INTERACTION
# uv run -m src.make_second_round_input --self_interaction --dataset sarcasm --output_root /home/rp-fril-mhpe/tmp


# COMMONSENSE SELF INTERACTION
# uv run -m src.make_second_round_input --self_interaction --dataset commonsense --output_root /home/rp-fril-mhpe/tmp

# SENTIMENT ALL (we now have vic's data)
#uv run -m src.make_second_round_input --dataset sentiment

#uv run -m src.make_second_round_input --self_interaction --dataset sentiment --output_root /home/rp-fril-mhpe/tmp/self-sentiment

# SENTIMENT rerun llama
uv run -m src.make_second_round_input --self_interaction --dataset sentiment --input_dir /home/rp-fril-mhpe/tmp/llama --output_root /home/rp-fril-mhpe/tmp/llama 
uv run -m src.make_second_round_input --dataset sentiment --input_dir /home/rp-fril-mhpe/tmp/llama --output_root /home/rp-fril-mhpe/tmp/llama


# TEMPERATURE EXPERIMENT

#uv run -m src.make_second_round_input --input_dir /home/rp-fril-mhpe/temperature/first --output_root /home/rp-fril-mhpe/temperature/input_round2