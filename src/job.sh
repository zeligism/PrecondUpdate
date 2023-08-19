#!/bin/bash
#SBATCH --job-name=scaledvr
#SBATCH --output="%x-%j.out"
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=512
#SBATCH -q cpu-512
#SBATCH -p cpu
#SBATCH --time=12:00:00
source activate torch
srun python src/run_experiment.py

