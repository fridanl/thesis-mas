#!/bin/bash
#SBATCH --job-name=make-match
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


#jjuv run src/make_match_type.py

uv run src/results.py --dataset sarcasm --swap

uv run src/results.py --dataset sarcasm
uv run src/results.py --dataset commonsense
uv run src/results.py --dataset sentiment

# uv run src/results.py --dataset commonsense

#uv run check.py
#uv run results.py --dataset sarcasm
#uv run src/first_round_results.py
#uv run check.py --dataset sarcasm
#uv run results_.py --dataset sarcasm
