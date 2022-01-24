#!/bin/bash
#SBATCH --job-name=scaledvr
#SBATCH --output="%x-%A_%a.out"
#SBATCH --ntasks=1
#SBATCH --time=0:10:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user="abdulla.almansoori@mbzuai.ac.ae"

JA_FILE="${JA_FILE:-"experiment.ja"}"

source activate opt
echo "Submitting jobs in job array file: '${JA_FILE}'"

# Get the array_id'th line of job array script and run
COMMAND="$(sed -n "${SLURM_ARRAY_TASK_ID}"p "${JA_FILE}")"
echo "job ${SLURM_ARRAY_TASK_ID}: ${COMMAND}"

# Run command
eval "$COMMAND"
