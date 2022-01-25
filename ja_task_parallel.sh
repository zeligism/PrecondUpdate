#!/bin/bash
#SBATCH --job-name=scaledvr_japarallel
#SBATCH --output="%x-%A_%a.out"
#SBATCH --ntasks=1
#SBATCH --time=0:10:00

completedjobs=".completedjobs"
sync_wait=5
# Note: this should execute before any other job finishes
rm -f $completedjobs

pwd; hostname; date
source activate opt # assuming conda env 'opt' exists

JA_FILE="${JA_FILE:-"experiment.ja"}"
LAST_JOB=$(wc -l < ${JA_FILE})
NRUNS=${NRUNS:-1}
START=$(( 1 + $NRUNS * (${SLURM_ARRAY_TASK_ID} - 1) ))
END=$(( $START + $NRUNS - 1 ))
END=$(( $END <= ${LAST_JOB} ? $END : ${LAST_JOB} ))

echo "Running jobs from file: '${JA_FILE}'"
sed -n "${START},${END}p;${END}q" "${JA_FILE}" | parallel -j$NRUNS {}

# hack to make sure all jobs finish before job scripts finishes
# this is because there seems to be a glitch in the current SLURM system
# that kills the remaining jobs in the array as soon as a single job finishes
echo ${SLURM_ARRAY_TASK_ID} >> $completedjobs
while (( $(wc -l < $completedjobs) < ${SLURM_ARRAY_TASK_COUNT} )); do
	sleep ${sync_wait}
done

