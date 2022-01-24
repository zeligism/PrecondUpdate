JA_FILE=${1:-"experiment.ja"}

if [[ -f "${JA_FILE}" ]]; then
	echo "Submitting job array '${JA_FILE}'"
	sbatch --array=1-$(wc -l < "${JA_FILE}") \
		--export=JA_FILE="${JA_FILE}" \
		"ja_task.sh"
else
	echo "Job array file '${JA_FILE}' not found."
fi

