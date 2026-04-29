#!/bin/bash
#SBATCH --job-name=llama-qwen-no-history
#SBATCH --account=researchers
#SBATCH --partition=scavenge
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x.%j.out
#SBATCH --mail-type=BEGIN,END






## #SBATCH --gres=gpu:h100:1



echo "Host: $(hostname)"

set -euo pipefail

export PYTHONPATH=/home/fril/thesis-mas${PYTHONPATH:+:$PYTHONPATH}


uv run src/make_match_type.py