DATASETS=("a1a")
DATASETS_DIR="datasets"
mkdir -p "${DATASETS_DIR}"
BASE_LINK="https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary"
for dataset in "${DATASETS[@]}"
    do [[ ! -f "${dataset}" ]] && wget -O "${DATASETS_DIR}/${dataset}" "${BASE_LINK}/${dataset}"
done
