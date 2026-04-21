#!/bin/bash
#SBATCH --job-name=r2-input
#SBATCH --account=researchers
#SBATCH --partition=scavenge
#SBATCH --cpus-per-task=4
#SBATCH --time=2:00:00
#SBATCH --output=logs/%x.%j.out
#SBATCH --mail-type=BEGIN,END

echo "Host: $(hostname)"

set -euo pipefail

uv run -m src.make_second_round_input --dataset sentiment --output_root /home/rp-fril-mhpe/input_round2
#uv run -m src.make_second_round_input --self_interaction --dataset sentiment 
