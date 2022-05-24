#!/bin/bash
#SBATCH --job-name=scaledvr
#SBATCH --output=output.%A_%a.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH -p gpu
#SBATCH -q gpu-8
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
srun python src/pytorch/run_experiment.py
