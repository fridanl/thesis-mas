#!/bin/bash
#SBATCH --job-name=env
#SBATCH --partition=acltr
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x.%j.out
#SBATCH --mail-type=BEGIN,END


echo "Host: $(hostname)"

nvidia-smi

source .venv/bin/activate 

which python3

uv sync

uv python3 -c "import vllm; print('vllm', vllm.__version__)"