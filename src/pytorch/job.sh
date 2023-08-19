#!/bin/bash
#SBATCH --job-name=scaledvr
#SBATCH --output=output.%A_%a.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH -p gpu
#SBATCH -q gpu-single
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
source activate torch
srun python src/pytorch/run_experiment.py
