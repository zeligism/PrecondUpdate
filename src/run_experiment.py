
import pickle
import os
import time
from argparse import Namespace
from itertools import product
from random import random
from train import train

SEED = 123
T = 20
LOG_DIR = "logs"
EXPERIMENT = 2

if EXPERIMENT == 1:
    #DATASETS = ("covtype", "ijcnn1", "news20", "rcv1",)
    DATASETS = ("a9a", "rcv1", "covtype", "real-sim", "w8a",)
    OPTIMIZERS = ("SGD", "SARAH", "SVRG",)
    BATCH_SIZES = (10,)
    GAMMAS = (2**i for i in range(-20,6))
    LAMBDAS = (0.0,)
    PS = (0.99,)
    PRECONDS = (None, "hutchinson")
    BETAS = (0.999,)
    ALPHAS = (1e-5,)
    CORRUPT = (None, [])  # empty list means use default

elif EXPERIMENT == 2:
    DATASETS = ("a9a", "real-sim", "w8a",)
    OPTIMIZERS = ("SGD", "SARAH",)
    BATCH_SIZES = (32,)
    GAMMAS = (2**i for i in range(-33,10,3))
    LAMBDAS = (0.0, 1e-4)
    PS = (0.99,)
    PRECONDS = (None, "hutchinson")
    BETAS = (0.999,)
    ALPHAS = (1e-7,)
    CORRUPT = (None, [-5,0], [5,0])

HYPERPARAM_GRID = product(DATASETS,
                          OPTIMIZERS,
                          BATCH_SIZES,
                          GAMMAS,
                          LAMBDAS,
                          PS,
                          PRECONDS,
                          BETAS,
                          ALPHAS,
                          CORRUPT,
                          )


def main():
    # Give other jobs a chance to avoid conflicts
    time.sleep(3 * random())
    
    # Create log dirs on start up
    if not os.path.isdir(LOG_DIR):
        os.mkdir(LOG_DIR)
    for dataset in DATASETS:
        dataset_path = os.path.join(LOG_DIR, dataset)
        if not os.path.isdir(dataset_path):
            os.mkdir(dataset_path)

    for dataset, optimizer, BS, gamma, lam, p, precond, beta, alpha, corrupt in HYPERPARAM_GRID:
        # Create log file name in a way that remembers all args
        args_str = f"BS={BS},gamma={gamma},lam={lam}"
        if optimizer == "L-SVRG":
            args_str += f",p={p}"
        if precond is not None:
            args_str += f",precond={precond}"
        if precond == "hutchinson":
            args_str += f",beta={beta},alpha={alpha}"
        if corrupt is not None:
            args_str += f",corrupt={tuple(corrupt)}"

        # log file is {LOG_DIR}/dataset/optimizer(arg1=val1,...,argN=valN).pkl
        logfile = os.path.join(LOG_DIR, dataset, f"{optimizer}({args_str}).pkl")

        # Skip if another job already started on this
        if os.path.exists(logfile):
            continue
        # Quickly touch the log file to reserve this job
        with open(logfile, "wb") as f:
            pickle.dump([], f)

        # Create arg namespace to pass to train
        args = Namespace(alpha=alpha,
                         BS=BS,
                         beta=beta,
                         corrupt=corrupt,
                         dataset=dataset,
                         T=T,
                         gamma=gamma,
                         lam=lam,
                         optimizer=optimizer,
                         precond=precond,
                         p=p,
                         savedata=logfile,
                         savefig=None,
                         seed=SEED)

        # Run
        print(logfile)
        #train(args)

if __name__ == "__main__":
    main()
