
JA_FILE="${1:-"experiment.ja"}"
LOG_DIR="logs"
rm -f "${JA_FILE}"
mkdir -p "${LOG_DIR}"

# Set up your default options here
seed=123
BS=10
T=20
lam=0.0
p=0.99
beta=0.999
alpha=1e-5
run="python src/train.py"
default_run="$run -s $seed -T $T -BS $BS"

DATASETS=("a9a", "w8a" "real-sim" "rcv1" "covtype")
OPTIMIZERS=("SGD"  "SARAH"  "SVRG")
GAMMA_POWERS=($(seq -20 5))
PRECONDS=(0 1)
CORRUPTS=(0 1)

# Create log dirs
for dataset in "${DATASETS[@]}"; do
    mkdir -p "${LOG_DIR}/$dataset"
    mkdir -p "${LOG_DIR}/${dataset}_bad"
done

# Then add all combinations of options
for dataset in "${DATASETS[@]}"; do
    mkdir -p "${LOG_DIR}/$dataset"
    mkdir -p "${LOG_DIR}/${dataset}_bad"
    for optimizer in "${OPTIMIZERS[@]}"; do
        for gammapow in "${GAMMA_POWERS[@]}"; do
            for precond in "${PRECONDS[@]}"; do
                for corrupt in "${CORRUPTS[@]}"; do
                    # Set up command
                    command="${default_run}"
                    gamma="2e${gammapow}"
                    command+=" --dataset ${dataset} --optimizer ${optimizer} --gamma ${gamma}"
                    [[ $precond == 1 ]] && commands+=" --precond hutchinson"
                    [[ $corrupt == 1 ]] && commands+=" --corrupt"
                    # Set up args info for log name
                    argsinfo="BS=${BS},gamma=${gamma},lam=${lam}"
                    [[ $optimizer == "L-SVRG" ]] && argsinfo+=",p=${p}"
                    [[ $precond == 1 ]] && argsinfo+=",precond=hutchinson,beta=${beta},alpha=${alpha}"
                    dataset_dir=$dataset
                    [[ $corrupt == 1 ]] && dataset_dir+="_bad"
                    command+=" --savedata '${LOG_DIR}/${dataset_dir}/${optimizer}(${argsinfo}).pkl'"
                    echo "${command}" >> "${JA_FILE}"
                done
            done
        done
    done
done

# Check job array file and number of jobs
cat "${JA_FILE}"
