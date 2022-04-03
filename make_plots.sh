plotsdir="Dratio-plots"
OPTIMIZERS=(SGD SARAH L-SVRG)
GAMMAS=(1e-1 2e-2 5e-3 1e-3 2e-4)
DATASETS=(a9a w8a rcv1 real-sim)

mkdir -p "$plotsdir"

for opt in "${OPTIMIZERS[@]}"; do
	for dataset in "${DATASETS[@]}"; do
		for gamma in "${GAMMAS[@]}"; do
			python src/train.py -s 123 -T 50 -BS 128 --optimizer $opt --dataset $dataset \
				   --gamma $gamma --precond hutchinson --beta 0.99 --alpha 0.1 \
				   --savefig "${plotsdir}/${opt}(dataset=${dataset},gamma=${gamma}).png"
		done
	done
done
