#!/bin/bash
#SBATCH --job-name=git-status
#SBATCH --account=researchers
#SBATCH --partition=scavenge
#SBATCH --cpus-per-task=4
#SBATCH --time=00:05:00
#SBATCH --output=logs/%x.%j.out

module load git

git status
