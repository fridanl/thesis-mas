#!/bin/bash
#SBATCH --job-name=subsample_test
#SBATCH --account=researchers
#SBATCH --partition=scavenge
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x.%j.out
#SBATCH --mail-type=END



echo "Host: $(hostname)"

set -euo pipefail

#uv run src/first_round_results.py
uv run src/make_subsample.py --input_dir subsample_testdata --output_dir results/subsample