
ja_file="${1:-"experiment.ja"}"
mkdir -p "plots"
rm -f "${ja_file}"

# Set up your default options here
run="python src/train.py"
defaults="-s 123 --optimizer 'SARAH-AdaHessian' -T 20 -BS 10"
default_run="${run} ${defaults}"

# Define the varying options here
ALPHAS=(1e-3 1e-7)
BETAS=(0.99 0.999 0.999)
GAMMAS=(1e0 1e-1 1e-2 1e-3 1e-4 1e-5 1e-6)
LAMBDAS=(0.0 1e1 1e0 1e-1 1e-2 1e-3 1e-4 1e-5 1e-6)


# Then add all combinations of options
for alpha in "${ALPHAS[@]}"; do
    for beta in "${BETAS[@]}"; do
        for gamma in "${GAMMAS[@]}"; do
            for lam in "${LAMBDAS[@]}"; do
                command="${default_run}"
                command+=" --alpha ${alpha} --beta ${beta} --gamma ${gamma} --lam ${lam}"
                command+=" --savefig 'plots/plot(alpha=${alpha},beta=${beta},gamma=${gamma},lam=${lam}).png'"
                echo "${command}" >> "${ja_file}"
            done
        done
    done
done

# Check job array file and number of jobs
#echo "job array has $(wc -l < "${ja_file}") jobs:"
cat "${ja_file}"
