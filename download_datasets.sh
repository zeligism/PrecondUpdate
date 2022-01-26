DATASETS=("a1a" "a9a" "rcv1" "covtype" "real-sim" "w8a" "ijcnn1" "news20")
DATASETS_DIR="datasets"
mkdir -p "${DATASETS_DIR}"
BASE_LINK="https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary"
for dataset in "${DATASETS[@]}"; do
    if [[ $dataset == "rcv1" ]]; then
        LINK="${BASE_LINK}/rcv1_train.binary"
    elif [[ $dataset == "covtype" ]]; then
        LINK="${BASE_LINK}/covtype.libsvm.binary"
    elif [[ $dataset == "news20" ]]; then
        LINK="${BASE_LINK}/news20.binary"
    else
        LINK="${BASE_LINK}/${dataset}"
    fi
    dataset_path="${DATASETS_DIR}/${dataset}"
    [[ ! -f "${dataset_path}" ]] && wget -O "${dataset_path}" "$LINK"
done
