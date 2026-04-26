#!/bin/bash
#SBATCH --job-name=sentiment-all-sub
#SBATCH --account=researchers
#SBATCH --partition=scavenge
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x.%j.out
#SBATCH --mail-type=END



echo "Host: $(hostname)"

set -euo pipefail


# uv run src/make_subsample.py --glob_pattern *_self_interaction_agree.csv --cap 1000 --dataset commonsense
# uv run src/make_subsample.py --glob_pattern *_self_interaction_disagree.csv --cap 7000 --dataset commonsense


# uv run src/first_round_results.py
# try the agreement ones:
# uv run src/make_subsample.py --suffix disagree --cap 7000 --dataset commonsense
# uv run src/make_subsample.py --suffix agree --cap 1000 --dataset commonsense

# sentiment small models as receivers
# uv run src/make_subsample.py --glob_pattern *_disagree2.csv --cap 7000 --dataset sentiment
# uv run src/make_subsample.py --glob_pattern *_agree2.csv --cap 1000 --dataset sentiment


# when ready for large mdoels for sentiment:
# uv run src/make_subsample.py --suffix disagree --cap 7000 --dataset sentiment
# uv run src/make_subsample.py --suffix agree --cap 1000 --dataset sentiment

# COMMONSENSE gpt 'subsample'   
#uv run src/make_subsample.py --glob_pattern *-20b_self_interaction_disagree.csv --input_dir /home/rp-fril-mhpe/tmp --output_dir /home/rp-fril-mhpe/subsampled_input_round2/commonsense/gpt/commonsense/self --cap 7000 --dataset commonsense
#uv run src/make_subsample.py --glob_pattern *_gpt_sender_agree.csv --input_dir /home/rp-fril-mhpe/input_round2/commonsense/gpt --output_dir /home/rp-fril-mhpe/subsampled_input_round2/commonsense/gpt --cap 1000 --dataset commonsense

#uv run src/make_subsample.py --glob_pattern *-oss-20b_agree.csv --input_dir /home/rp-fril-mhpe/input_round2/commonsense/gpt --output_dir /home/rp-fril-mhpe/subsampled_input_round2/commonsense/gpt --cap 1000 --dataset commonsense

# SENTIMENT subsample all input files for r2
uv run src/make_subsample.py --suffix disagree --cap 7000 --dataset sentiment
uv run src/make_subsample.py --suffix agree --cap 1000 --dataset sentiment