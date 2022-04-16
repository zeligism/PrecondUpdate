
import pickle
import os
import time
from argparse import Namespace
from itertools import product
from random import random
from train import *


DRY_RUN = False
LOG_DIR = "logs10"
HP_DICT = {
    "T": (50,),
    "seed": range(10),
    # "dataset": ("covtype", "ijcnn1", "news20"),
    "dataset": ("a9a", "w8a", "rcv1", "real-sim"),
    "optimizer": ("SGD", "Adam", "SARAH", "L-SVRG"),
    "corrupt": (None, (-3,0), (0,3), (-3,3)),
    "BS": (128,),
    "lr": (2**i for i in range(-16, 5, 2)),
    "weight_decay": (0.0, 0.1),
    #"lr_decay": (0.0, 0.1),
    "p": (0.99,),
    "precond": (None, "hutchinson"),
    "beta2": (0.999,),
    "alpha": (1e-1, 1e-3, 1e-7),
    # @TODO: add losses
}


def main():
    # Give other jobs a chance to avoid conflicts
    time.sleep(3 * random())

    # Create log dirs on start up
    if not os.path.isdir(LOG_DIR):
        os.mkdir(LOG_DIR)
    # Create folders for each dataset and for each corrupted version
    dataset_paths = {}
    for dataset in HP_DICT["dataset"]:
        for corrupt in HP_DICT["corrupt"]:
            if corrupt is None:
                dataset_folder = dataset
            else:
                dataset_folder = f"{dataset}({corrupt[0]},{corrupt[1]})"
            dataset_path = os.path.join(LOG_DIR, dataset_folder)
            if not os.path.isdir(dataset_path):
                os.mkdir(dataset_path)
            # Add to paths
            if dataset not in dataset_paths:
                dataset_paths[dataset] = {}
            dataset_paths[dataset][corrupt] = dataset_path

    for hyperparams in product(*HP_DICT.values()):
        hp = dict(zip(HP_DICT.keys(), hyperparams))

        # Hard settings
        if hp['optimizer'] in ("SGD", "Adam", "Adagrad", "Adadelta"):
            hp['T'] *= 2

        if 'lr_decay' not in hp:
            hp['lr_decay'] = 0

        if 'weight_decay' not in hp:
            hp['weight_decay'] = 0

        if hp['optimizer'] == "Adam":
            hp['precond'] = None
            hp['beta1'] = 0.9
            hp['beta2'] = 0.999
            hp['alpha'] = 1e-8

        # Create log file name in a way that remembers all args
        args_str = f"seed={hp['seed']}"
        args_str += f",BS={hp['BS']}"
        args_str += f",lr={hp['lr']}"
        if hp['lr_decay'] != 0:
            args_str += f",lr_decay={hp['lr_decay']}"
        if hp['weight_decay'] != 0:
            args_str += f",weight_decay={hp['weight_decay']}"

        if hp['optimizer'] in ("L-SVRG", "PAGE"):
            args_str += f",p={hp['p']}"

        if hp['precond'] == "hutchinson":
            args_str += f",precond={hp['precond']}"
            args_str += f",beta2={hp['beta2']}"
            args_str += f",alpha={hp['alpha']}"

        # log file is {LOG_DIR}/dataset[(k_min,k_max)]/optimizer(arg1=val1,...,argN=valN).pkl
        dataset_path = dataset_paths[hp['dataset']][hp['corrupt']]
        logfile = os.path.join(dataset_path, f"{hp['optimizer']}({args_str}).pkl")

        # Skip if another job already started on this
        if os.path.exists(logfile):
            continue
        # Quickly touch the log file to reserve this job!
        with open(logfile, "wb") as f:
            pickle.dump([], f)

        # Create arg namespace to pass to train
        args = parse_args(namespace=Namespace(savedata=logfile, savefig=None, **hp))

        # Run
        print(logfile)
        if not DRY_RUN:
            train(args)


if __name__ == "__main__":
    main()
