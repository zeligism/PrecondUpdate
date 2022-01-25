
import pickle
import os
import time
from argparse import Namespace
from itertools import product
from random import random
from train import train

BATCH_SIZES = (1, 10)
GAMMAS = (1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6)
LAMBDAS = (0.0, 1e1, 1e0, 1e-1, 1e-3, 1e-5)
BETAS = (0.99, 0.999)
ALPHAS = (1e-3, 1e-7)
HYPERPARAMS = (BATCH_SIZES, GAMMAS, LAMBDAS, BETAS, ALPHAS)

def main():
    for BS, gamma, lam, beta, alpha in product(*HYPERPARAMS):
        # Give other jobs a chance
        time.sleep(3 * random())
        logfile = f"log/data(BS={BS},gamma={gamma},lam={lam},beta={beta},alpha={alpha}).pkl"
        print(logfile)
        # Skip if another job started on this
        if os.path.exists(logfile):
            continue
        # Quickly touch a log file to reserve this job
        with open(logfile, "wb") as f:
            pickle.dump([], f)

        args = Namespace(alpha=alpha,
                         batch_size=BS,
                         beta=beta,
                         corrupt=[],
                         dataset="a1a",
                         epochs=5,
                         gamma=gamma,
                         lam=lam,
                         optimizer="SARAH-OASIS",
                         savedata=logfile,
                         savefig=None, seed=123)
        train(args)

if __name__ == "__main__":
    main()
