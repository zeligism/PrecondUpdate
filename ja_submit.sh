JA_FILE=${1:-"experiment.ja"}
NRUNS=${2:-1}

if [[ -f "${JA_FILE}" ]]; then
	echo "Submitting job array '${JA_FILE}'"
	NJOBS=$(wc -l < "${JA_FILE}")
	NTASKS=$(( $NJOBS / $NRUNS ))
	(( $NJOBS % $NRUNS > 0 )) && ((NTASKS++))
	echo " - Job array has $NJOBS jobs."
	echo " - Submitting $NTASKS jobs with at most $NRUNS runs each."
	sbatch --array=1-$NTASKS \
		--export=JA_FILE="${JA_FILE}",NRUNS="$NRUNS" \
		ja_task.sh
else
	echo "Job array '${JA_FILE}' not found."
fi

