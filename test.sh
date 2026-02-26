#!/bin/bash
#SBATCH --job-name=test-run
#SBATCH --partition=scavenge
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G  
#SBATCH --time=04:00:00
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

uv run test.py --model_name llama-3.1-8b --repetition 3 --round 1 --batch_size 2 -limit 6
#uv run test.py --model_name llama-3.3-70b --repetition 3 --round 1 --batch_size 2 -limit 6
