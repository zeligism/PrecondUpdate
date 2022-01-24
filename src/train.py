
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import sklearn
import scipy
import time

from joblib import Memory
from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import normalize

from optimizer import *

mem = Memory("./mycache")
DATASET_DIR = "datasets"
DATASETS = ("a1a",)
OPTIMIZERS = ("SGD", "SARAH", "SGD-Hessian", "SARAH-Hessian", "OASIS", "SARAH-AdaHessian")
BAD_SCALE = lambda d: 10**np.linspace(-3,3,d)
#BAD_SCALE = lambda d: 10**np.linspace(-3,5,d)**3


def parse_args():
    parser = argparse.ArgumentParser(description="SGD and SARAH with Hessian scaling")

    parser.add_argument("-s", "--seed", type=int, default=None, help='random seed')
    parser.add_argument("--dataset", type=str, choices=DATASETS, default="a1a",
                        help="name of dataset (in 'datasets' directory")
    parser.add_argument("--corrupt", action="store_true", help="corrupt scale of dataset")
    parser.add_argument("--savefig", type=str, default=None,
                        help="save plots under this name (default: don't save)")

    parser.add_argument("--optimizer", type=str, choices=OPTIMIZERS, default="SARAH-AdaHessian", help="name of optimizer")
    parser.add_argument("-T", "--epochs", type=int, default=5, help="number of iterations/epochs to run")
    parser.add_argument("-BS", "--batch_size", type=int, default=1, help="batch size")
    parser.add_argument("-lr", "--gamma", type=float, default=0.02, help="base learning rate")
    parser.add_argument("--alpha", type=float, default=1e-5, help="min value of diagonal of hessian estimate")
    parser.add_argument("--beta", type=float, default=0.999, help="adaptive rate of hessian estimate")
    parser.add_argument("--lam", type=float, default=0., help="regularization coefficient")

    # Parse command line args
    args = parser.parse_args()
    return args


@mem.cache
def get_data(filePath):
    data = load_svmlight_file(filePath)
    return data[0], data[1]


def plot_data(data, fname=None):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    fig.set_size_inches(20, 6)

    ax1.plot(data[:,0])
    ax1.set_ylabel(r"$F(w_t)$")
    ax1.set_xlabel(r"iteration $t$")  

    ax2.semilogy(data[:,1])
    ax2.set_ylabel(r"$||\nabla F(w_t)||^2$")
    ax2.set_xlabel(r"iteration $t$") 

    ax3.plot(data[:,2])
    ax3.set_ylabel(r"error")
    ax3.set_xlabel(r"iteration $t$") 

    if fname is not None:
        print(f"Saving data to '{fname}'.")
        plt.savefig(fname)
        plt.close()
    else:
        plt.show()


def corrupt_scale(X, bad_scale=None):
    if bad_scale is None:
        bad_scale = BAD_SCALE(X.shape[1])
    np.random.shuffle(bad_scale)
    return X[:].multiply(bad_scale.reshape(1,-1)).tocsr()


def main(args):
    # check if dataset is downloaded
    args.dataset = os.path.join(DATASET_DIR, args.dataset)
    if not os.path.isfile(args.dataset):
        raise FileNotFoundError(f"Could not find dataset at 'datasets/{args.dataset}'")
    print(f"Using dataset '{args.dataset}'.")
    # Set seed if given
    if args.seed is not None:
        np.random.seed(args.seed)
        print(f"Setting random seed to {args.seed}.")

    X, y = get_data(args.dataset)
    X = normalize(X, norm='l2', axis=1)
    print("We have %d samples, each has up to %d features." % (X.shape[0], X.shape[1]))

    if args.corrupt:
        print("Corrupting scale of data...")
        X = corrupt_scale(X)

    print(f"Running {args.optimizer}...")
    if args.optimizer == "SGD":
        wopt, data = SGD(X, y, gamma=args.gamma, BS=args.batch_size, T=args.epochs)
    elif args.optimizer == "SARAH":
        wopt, data = SARAH(X, y, gamma=args.gamma, BS=args.batch_size, epochs=args.epochs)
    elif args.optimizer == "SGD-Hessian":
        wopt, data = SGD_Hessian(X, y, gamma=args.gamma, BS=args.batch_size, lam=args.lam, T=args.epochs)
    elif args.optimizer == "SARAH-Hessian":
        wopt, data = SARAH_Hessian(X, y, gamma=args.gamma, BS=args.batch_size, lam=args.lam, epochs=args.epochs)
    elif args.optimizer == "OASIS":
        wopt, data = OASIS(X, y, gamma=args.gamma, beta=args.beta, alpha=args.alpha, lam=args.lam,
                           BS=args.batch_size, epochs=args.epochs)
    elif args.optimizer == "SARAH-AdaHessian":
        wopt, data = SARAH_AdaHessian(X, y, gamma=args.gamma, beta=args.beta, alpha=args.alpha, lam=args.lam,
                                      BS=args.batch_size, epochs=args.epochs)
    else:
        raise NotImplementedError(f"Optimizer '{args.optimizer}' not implemented.")

    print("Done.")
    plot_data(data, fname=args.savefig)


if __name__ == "__main__":
    args = parse_args()
    main(args)
