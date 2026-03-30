#!/bin/bash
#SBATCH --job-name=second-round-input
#SBATCH --account=researchers
#SBATCH --partition=scavenge
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=2:00:00
#SBATCH --constraint="gpu_a100_80gb|gpu_h100|gpu_l40s|gpu_a30"
#SBATCH --output=logs/%x.%j.out
#SBATCH --mail-type=END

echo "Host: $(hostname)"

set -euo pipefail

nvidia-smi

uv run -m src.make_second_round_input --dataset commonsense --output_root /home/rp-fril-mhpe/input_round2/commonsense
