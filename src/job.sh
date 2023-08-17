#!/bin/bash
#SBATCH --job-name=scaledvr
#SBATCH --output="%x-%j.out"
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=128
##SBATCH --time=02:00:00
srun python src/run_experiment.py

