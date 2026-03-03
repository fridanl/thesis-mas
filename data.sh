#!/bin/bash
#SBATCH --job-name=eda
#SBATCH --partition=scavenge
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x.%j.out


echo "Host: $(hostname)"

set -euo pipefail

uv run src/first-round-results.py