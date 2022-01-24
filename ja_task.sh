#!/bin/bash
#SBATCH --job-name=scaledvr
#SBATCH --output="%x-%A_%a.out"
#SBATCH --ntasks=1
#SBATCH --time=0:10:00
pwd; hostname; date
source activate opt # assuming conda env 'opt' exists

JA_FILE="${JA_FILE:-"experiment.ja"}"
LAST_JOB=$(wc -l < ${JA_FILE})
NRUNS=${NRUNS:-1}
START=$(( 1 + $NRUNS * (${SLURM_ARRAY_TASK_ID} - 1) ))
END=$(( $START + $NRUNS - 1 ))
END=$(( $END <= ${LAST_JOB} ? $END : ${LAST_JOB} ))

echo "Running jobs from file: '${JA_FILE}'"
for (( run=$START; run<=END; run++ )); do
	# Get the $run'th line of job array script and run it
	COMMAND="$(sed -n "$run"p "${JA_FILE}")"
	echo ""
	echo "========== job $run start"
	echo "$COMMAND"
	echo ""
	eval "$COMMAND"
	echo "========== job $run end"
done

# hack to make sure all jobs finish before job scripts finishes
# assuming all jobs are running the same number of commands
# this is because there seems to be a glitch in he current SLURM system
# that kills the remaining jobs in the array as soon as a single job finishes
sleep 30  # sleep for half a minute
