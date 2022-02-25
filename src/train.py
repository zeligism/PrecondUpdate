
import time
import argparse
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import scipy
from joblib import Memory
from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import normalize

from optimizer import *
from plot import *

mem = Memory("./mycache")
DATASET_DIR = "datasets"
DATASETS = ("a1a", "a9a", "rcv1", "covtype", "real-sim", "w8a", "ijcnn1", "news20",)
OPTIMIZERS = ("SGD", "SARAH", "OASIS", "SVRG", "L-SVRG",)


def parse_args():
    parser = argparse.ArgumentParser(description="Optimizers with diagonal preconditioning")

    parser.add_argument("-s", "--seed", type=int, default=None, help='random seed')
    parser.add_argument("--dataset", type=str, choices=DATASETS, default="a1a",
                        help="name of dataset (in 'datasets' directory")
    parser.add_argument("--corrupt", nargs="*", type=int, default=None, help="If specified, corrupt scale of dataset."
                        "Takes at most two values: k_min, k_max, specifying the range of powers in scale."
                        "If one value k is given, the range will be (-k,k) (otherwise, default to (-3,3)).")
    parser.add_argument("--savefig", type=str, default=None, help="save plots under this name (default: don't save)")
    parser.add_argument("--savedata", type=str, default=None, help="save data log (default: don't save)")

    parser.add_argument("--optimizer", type=str, choices=OPTIMIZERS, default="SARAH", help="name of optimizer")
    parser.add_argument("-T", "--epochs", dest="T", type=int, default=5, help="number of epochs to run")
    parser.add_argument("-BS", "--batch_size", dest="BS", type=int, default=1, help="batch size")
    parser.add_argument("-lr", "--gamma", type=float, default=0.02, help="base learning rate")
    parser.add_argument("--alpha", type=float, default=1e-5, help="min value of diagonal of hessian estimate")
    parser.add_argument("--beta", type=float, default=0.999, help="adaptive rate of hessian estimate")
    parser.add_argument("--lam", type=float, default=0., help="regularization coefficient")
    parser.add_argument("-p", "--update-p", dest="p", type=float, default=0.99, help="probability of updating checkpoint in L-SVRG")
    parser.add_argument("--precond", type=str.lower, default=None, help="Diagonal preconditioning method (default: none)")
    parser.add_argument("--precond_warmup", type=int, default=1, help="Number of samples for initializing diagonal estimate")
    parser.add_argument("--precond_resample", action="store_true", help="Resample batch for preconditioning")
    parser.add_argument("--precond_zsamples", type=int, default=1, help="Number of z samples")

    # Parse command line args
    args = parser.parse_args()
    return args


@mem.cache
def get_data(filePath):
    data = load_svmlight_file(filePath)
    return data[0], data[1]


def corrupt_scale(X, k_min=-3, k_max=3):
    bad_scale = 10**np.linspace(k_min, k_max, X.shape[1])
    np.random.shuffle(bad_scale)
    return X[:].multiply(bad_scale.reshape(1,-1)).tocsr()


def savedata(data, fname):
    with open(fname, 'wb') as f:
        pickle.dump(data, f)


def train(args):
    # check if dataset is downloaded
    args.dataset = os.path.join(DATASET_DIR, args.dataset)
    if not os.path.isfile(args.dataset):
        raise FileNotFoundError(f"Could not find dataset at '{args.dataset}'")
    print(f"Using dataset '{args.dataset}'.")
    # Set seed if given
    if args.seed is not None:
        np.random.seed(args.seed)
        print(f"Setting random seed to {args.seed}.")

    X, y = get_data(args.dataset)
    X = normalize(X, norm='l2', axis=1)
    print("We have %d samples, each has up to %d features." % (X.shape[0], X.shape[1]))

    if args.corrupt is not None:
        print("Corrupting scale of data.")
        if len(args.corrupt) == 0:
            args.corrupt = (-1,1)
        elif len(args.corrupt) == 1:
            args.corrupt = (-args.corrupt[0], args.corrupt[0])
        X = corrupt_scale(X, args.corrupt[0], args.corrupt[1])

    print(f"Running {args.optimizer}...")
    kwargs = dict(T=args.T, BS=args.BS,
                  gamma=args.gamma, beta=args.beta,
                  lam=args.lam, alpha=args.alpha,
                  precond=args.precond,
                  precond_warmup=args.precond_warmup,
                  precond_resample=args.precond_resample,
                  precond_zsamples=args.precond_zsamples,
                  )
    if args.optimizer == "SGD":
        wopt, data = SGD(X, y, **kwargs)
    elif args.optimizer == "SARAH":
        wopt, data = SARAH(X, y, **kwargs)
    elif args.optimizer == "OASIS":
        wopt, data = OASIS(X, y, **kwargs)
    elif args.optimizer == "SVRG":
        wopt, data = SVRG(X, y, **kwargs)
    elif args.optimizer == "L-SVRG":
        wopt, data = L_SVRG(X, y, **kwargs, p=args.p)
    else:
        raise NotImplementedError(f"Optimizer '{args.optimizer}' not implemented yet.")
    print("Done.")

    if args.savefig is not None:
        # Create title
        title = rf"{args.optimizer}({os.path.basename(args.dataset)})"
        title += rf" with BS={args.BS}, $\gamma$={args.gamma}"
        if args.lam != 0.0:
            title += rf", $\lambda$={args.lam}"
        if args.optimizer == "L-SVRG":
            title += f", p={args.p}"
        if args.precond is not None:
            title += f", precond={args.precond}"
        if args.precond == "hutchinson":
            title += rf", $\beta$={args.beta}, $\alpha$={args.alpha}"
        if args.corrupt is not None:
            title += f", corrupt=[{args.corrupt[0]}, {args.corrupt[1]}]"
        print(f"Saving plot to '{args.savefig}'.")
        optimum = OPTIMAL_LOSS[os.path.basename(args.dataset)]
        savefig2(data, optimum, args.savefig, title=title)

    if args.savedata is not None:
        print(f"Saving data to '{args.savedata}'.")
        savedata(data, args.savedata)


def main():
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
