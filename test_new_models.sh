#!/bin/bash
#SBATCH --job-name=gemma3-27b-test
#SBATCH --account=researchers
#SBATCH --partition=scavenge
#SBATCH --gres=gpu
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --constraint="gpu_h100|gpu_a100_80gb|gpu_a100_40gb"
#SBATCH --output=logs/%x.%j.out
#SBATCH --mail-type=BEGIN,END


echo "Host: $(hostname)"

set -euo pipefail

nvidia-smi

# rm -rf .venv
# uv venv

uv sync
# uv run run-eval.py --model_name llama-3.1-8b -limit 20 --outdir results/ --repetition 1 #virker ikke

# uv run run-eval.py --model_name llama-3.2-1b --outdir results_extra/ --repetition 10 --dataset_path data/sarc/sarcasm2_llama.csv

# uv run run-eval.py --model_name llama-3.2-3b --outdir results/ --repetition 10 --dataset_path data/sarc/sarcasm2.csv

# uv run run-eval.py --model_name qwen-2.5-7b --repetition 10 --outdir results/ --dataset_path data/sarc/sarcasm50k.csv
# uv run run-eval.py --model_name qwen-2.5-7b --repetition 10 --outdir results_extra/ --dataset_path data/sarc/sarcasm2-minus-50k.csv

# uv run run-eval.py --model_name qwen-2.5-1.5b --repetition 10 --outdir results_extra/ --dataset_path data/sarc/sarcasm2-minus-50k.csv

# uv run run-eval.py --model_name mistral-0.3-7b --outdir results/ --repetition 10

#uv run test.py --model_name qwen-2.5-1.5b --repetition 3 --round 1 --batch_size 2 -limit 6

# new models
# uv run test.py --model_name gemma-3-4b --repetition 5 --round 1 --batch_size 5 -limit 25
# uv run test.py --model_name gpt-oss-9b --repetition 5 --round 1 --batch_size 5 -limit 5
# uv run test.py --model_name gpt-20b --repetition 5 --round 1 --batch_size 5 -limit 5
uv run test.py --model_name gemma-3-27b --repetition 5 --round 1 --batch_size 5 -limit 25
