#!/bin/bash
#SBATCH --job-name=all-results
#SBATCH --account=researchers
#SBATCH --partition=scavenge
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x.%j.out
#SBATCH --mem=80G

echo "Host: $(hostname)"

set -euo pipefail

export PYTHONPATH=/home/fril/thesis-mas${PYTHONPATH:+:$PYTHONPATH}

#uv run src/make_match_type.py

uv run src/results.py --dataset sarcasm --experiment main
uv run src/results.py --dataset sarcasm --experiment temperature
uv run src/results.py --dataset sarcasm --experiment swap
uv run src/results.py --dataset sarcasm --experiment no-explanation
uv run src/results.py --dataset sarcasm --experiment no-history


uv run src/results.py --dataset commonsense --experiment main
uv run src/results.py --dataset sentiment --experiment main






#uv run check.py
#uv run results.py --dataset sarcasm
#uv run src/first_round_results.py
#uv run check.py --dataset sarcasm
#uv run results_.py --dataset sarcasm
