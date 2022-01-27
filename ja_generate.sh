
JA_FILE="${1:-"experiment.ja"}"
PLOT_DIR="plots"
LOG_DIR="log"

rm -f "${JA_FILE}"
mkdir -p "${PLOT_DIR}"
mkdir -p "${LOG_DIR}"

# Set up your default options here
run="python src/train.py"
defaults="-s 123 --corrupt"
default_run="${run} ${defaults}"

# Define the varying options here
BATCH_SIZES=(1 10)
GAMMAS=(1e0 1e-1 1e-2 1e-3 1e-4 1e-5 1e-6)
LAMBDAS=(0.0 1e1 1e0 1e-1 1e-3 1e-5)
BETAS=(0.99 0.999)
ALPHAS=(1e-3 1e-7)

# Then add all combinations of options
for BS in "${BATCH_SIZES[@]}"; do
    for gamma in "${GAMMAS[@]}"; do
        for lam in "${LAMBDAS[@]}"; do
            for beta in "${BETAS[@]}"; do
                for alpha in "${ALPHAS[@]}"; do
                    command="${default_run}"
                    command+=" -BS ${BS} --gamma ${gamma} --lam ${lam} --beta ${beta} --alpha ${alpha}"
                    command_info="BS=${BS},gamma=${gamma},lam=${lam},beta=${beta},alpha=${alpha}"
                    #command+=" --savefig '${PLOT_DIR}/plot(${command_info}).png'"
                    command+=" --savedata '${LOG_DIR}/data(${command_info}).pkl'"
                    echo "${command}" >> "${JA_FILE}"
                done
            done
        done
    done
done

# Check job array file and number of jobs
#echo "job array has $(wc -l < "${JA_FILE}") jobs:"
cat "${JA_FILE}"
