#!/bin/bash
#SBATCH --job-name=self-gpt-sentiment
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

uv run -m src.make_second_round_input --self_interaction --dataset sentiment --output_root /home/rp-fril-mhpe/tmp/self-sentiment
