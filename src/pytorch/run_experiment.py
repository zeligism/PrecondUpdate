
import pickle
import os
import time
from argparse import Namespace
from itertools import product
from random import random, shuffle
from train import *


DRY_RUN = False  # for testing
LOG_DIR = "logs_torch"
HP_DICT = {
    "epochs": (15,),
    "seed": range(3),
    # "dataset": ("mnist", "cifar-10"),
    "dataset": ("mnist",),
    "optimizer": ("Adam", "L-SVRG", "SARAH", "SGD"),
    "batch_size": (128,),
    "p": (0.999,),
    "lr": (2**-6, 2**-8, 2**-10, 2**-12),
    "precond": ("none", "hutchinson"),
    "warmup": (10,),
    "zsamples": (50,),
    # "beta1": (0.0, 0.9),
    "beta1": (0.0,),
    "beta2": ("avg", 0.999, 0.99),
    # "alpha": (1e-1, 1e-3, 1e-7),
    "alpha": (1e-1, 1e-3),
}

HP_GRID = product(*HP_DICT.values())
# Comment out if you want experiments running in sequential order
# HP_GRID = list(HP_GRID)
# shuffle(HP_GRID)


def main():
    # Give other jobs a chance to avoid conflicts
    time.sleep(3 * random())

    # Create log dirs on start up
    dataset_paths = {}
    for dataset in HP_DICT["dataset"]:
        dataset_paths[dataset] = os.path.join(LOG_DIR, dataset)
        os.makedirs(dataset_paths[dataset], exist_ok=True)

    for hyperparams in HP_GRID:
        hp = dict(zip(HP_DICT.keys(), hyperparams))

        ### Hard settings ###
        if hp['optimizer'] in ("Adam", "Adagrad", "Adadelta"):
            hp['precond'] = "none"
            hp['alpha'] = 1e-8

        if hp['precond'] == "none" and hp['optimizer'] not in ("SVRG", "L-SVRG", "SARAH"):
            hp['epochs'] *= 2  # i.e. Adam and vanilla SGD

        if 'lr_decay' not in hp:
            hp['lr_decay'] = 0

        if 'weight_decay' not in hp:
            hp['weight_decay'] = 0

        # weight_decay only used for logistic loss
        if hp['weight_decay'] != 0 and hp["loss"] != "logistic":
            continue

        if hp['beta2'] == "avg" and hp['precond'] != "hutchinson":
            continue

        ### Create log file name in a way that remembers all relevant args ###
        args_str = f"seed={hp['seed']}"
        args_str += f",batch_size={hp['batch_size']}"
        args_str += f",lr={hp['lr']}"

        if hp['lr_decay'] != 0:
            args_str += f",lr_decay={hp['lr_decay']}"
        if hp['weight_decay'] != 0:
            args_str += f",weight_decay={hp['weight_decay']}"

        if hp['optimizer'] in ("L-SVRG", "PAGE"):
            args_str += f",p={hp['p']}"

        if hp['precond'] == "hutchinson":
            args_str += f",precond={hp['precond']}"
            args_str += f",beta1={hp['beta1']}"
            args_str += f",beta2={hp['beta2']}"
            args_str += f",alpha={hp['alpha']}"
            if 'warmup' in hp:
                args_str += f",warmup={hp['warmup']}"

        if hp['optimizer'] == "Adam":
            args_str += f",beta1={hp['beta1']}"
            args_str += f",beta2={hp['beta2']}"

        # log file is {LOG_DIR}/dataset/optimizer(arg1=val1,...,argN=valN).pkl
        dataset_path = dataset_paths[hp['dataset']]
        logfile = os.path.join(dataset_path, f"{hp['optimizer']}({args_str}).pkl")

        # Skip if another job already started on this
        if os.path.exists(logfile):
            continue
        # Quickly touch the log file to reserve this job!
        with open(logfile, "wb") as f:
            pickle.dump([], f)

        # Create arg namespace to pass to train
        hp['cuda'] = True
        hp['num_workers'] = 2
        args = parse_args(namespace=Namespace(savedata=logfile, **hp))

        # Run
        if not DRY_RUN:
            run(args)
        else:
            print(args)


if __name__ == "__main__":
    main()
