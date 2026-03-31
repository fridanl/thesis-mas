#!/bin/bash
#SBATCH --job-name=check
#SBATCH --account=researchers
#SBATCH --partition=scavenge
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x.%j.out
#SBATCH --mem=80G

echo "Host: $(hostname)"

set -euo pipefail

#uv run src/first_round_results.py
#uv run check.py


uv run results.py --dataset sarcasm
