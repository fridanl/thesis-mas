#!/bin/bash
#SBATCH --job-name=subsample_self_sarcasm
#SBATCH --account=researchers
#SBATCH --partition=scavenge
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x.%j.out
#SBATCH --mail-type=END



echo "Host: $(hostname)"

set -euo pipefail


uv run src/make_subsample.py --glob_pattern *-self-interaction.csv --cap 7000 --dataset sarcasm

#uv run src/first_round_results.py

# try the agreement ones:
# uv run src/make_subsample.py --suffix disagree --cap 7000 --dataset commonsense
# uv run src/make_subsample.py --suffix agree --cap 1000 --dataset commonsense

