
ja_file="${1:-"experiment.ja"}"
mkdir -p "plots"
rm -f "${ja_file}"

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
                    command+=" --savefig 'plots/plot(BS=${BS},gamma=${gamma},lam=${lam},beta=${beta},alpha=${alpha}).png'"
                    echo "${command}" >> "${ja_file}"
                done
            done
        done
    done
done

# Check job array file and number of jobs
#echo "job array has $(wc -l < "${ja_file}") jobs:"
cat "${ja_file}"
