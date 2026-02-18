#!/bin/bash
#SBATCH --job-name=env-heterog-mas
#SBATCH --partition=acltr
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x.%j.out
#SBATCH --constraint="gpu_h100|gpu_a100_80gb"
#SBATCH --mail-type=BEGIN,END


echo "Host: $(hostname)"

nvidia-smi

python -m pip install -U pip

uv add vllm==0.10.0 --extra-index-url https://wheels.vllm.ai/0.10.0/cu126 --extra-index-url https://download.pytorch.org/whl/cu126 --index-strategy unsafe-best-match


python --version


python -c "import vllm; print('vllm', vllm.__version__)"