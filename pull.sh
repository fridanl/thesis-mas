#!/bin/bash
#SBATCH --job-name=git-pull
#SBATCH --account=researchers
#SBATCH --partition=scavenge
#SBATCH --cpus-per-task=4
#SBATCH --time=00:05:00
#SBATCH --output=logs/git/%x.%j.out

module load git

git pull --no-rebase

