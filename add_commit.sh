#!/bin/bash
#SBATCH --job-name=git-add
#SBATCH --account=researchers
#SBATCH --partition=scavenge
#SBATCH --cpus-per-task=4
#SBATCH --time=00:05:00
#SBATCH --output=logs/%x.%j.out

module load git

git checkout --theirs subsample.sh

git add subsample.sh
git commit -m "Resolve merge conflict in subsample.sh"

git add .

git commit -m 'inference logs'

git push
