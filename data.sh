#!/bin/bash
#SBATCH --job-name=check-test-cleaned-script
#SBATCH --account=researchers
#SBATCH --partition=scavenge
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x.%j.out
#SBATCH --mem=80G

echo "Host: $(hostname)"

set -euo pipefail
<<<<<<< HEAD
export PYTHONPATH=/home/fril/thesis-mas${PYTHONPATH:+:$PYTHONPATH}
uv run src/first_round_results.py --dataset sarcasm
#uv run check.py

#uv run results.py --dataset sarcasm
=======

#uv run src/first_round_results.py
uv run check.py --dataset sarcasm

#uv run results_.py --dataset sarcasm
>>>>>>> 2195b62a0fbc93dac6c3743329f18aa9bc05335f
