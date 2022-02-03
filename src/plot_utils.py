
import argparse
import pickle
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from math import log2
from itertools import cycle

from run_experiment import LOG_DIR, DATASETS, OPTIMIZERS

MARKERS = (',', '+', '.', 'o', '*', "D")
OPTIMAL_GAMMAS = {
    (None, False): 2**-5,
    ("[-3-0]", False): 2**-10,
    ("[0-3]", False): 2**-10,
    ("[-3-3]", False): 2**-10,
    (None, True): 2**-15,
    ("[-3-0]", True): 2**-10,
    ("[0-3]", True): 2**-10,
    ("[-3-3]", True): 2**-10,
}

def loaddata(fname):
    with open(fname, 'rb') as f:
        data = pickle.load(f)
    return data

def savefig(data, fname, title="Loss, gradient norm squared, and error"):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    fig.set_size_inches(20, 6)
    plt.suptitle(title)

    ax1.plot(data[:,0], data[:,1])
    ax1.set_ylabel(r"$F(w_t)$")
    ax1.set_xlabel("Effective Passes")
    ax1.grid()

    ax2.semilogy(data[:,0], data[:,2])
    ax2.set_ylabel(r"$||\nabla F(w_t)||^2$")
    ax2.set_xlabel("Effective Passes")
    ax2.grid()

    ax3.plot(data[:,0], data[:,3])
    ax3.set_ylabel("Error")
    ax3.set_xlabel("Effective Passes")
    ax3.grid()

    if fname is None:
        plt.show()
    else:
        plt.savefig(fname)
        plt.close()

def unpack_args(fname):
    # unpack path
    dirname, logname = os.path.split(fname)
    logdir, dataset = os.path.split(dirname)
    # parse args
    optimizer, argstr = logname.split("(")
    argstr, _ = argstr.split(")")  # remove ').pkl'
    args = {k:v for k,v in [s.split("=") for s in argstr.split(",")]}
    BS = int(args["BS"])
    gamma = float(args["gamma"])
    lam = float(args["lam"])
    p = 0.99 if "p" not in args else args["p"]
    if "precond" in args:
        precond = args["precond"]
        beta = float(args["beta"])
        alpha = float(args["alpha"])
    else:
        precond = None
        # any random floats would do
        beta = 0.999
        alpha = 1e-5
    corrupt = None if "corrupt" not in args else args["corrupt"]

    return dataset, optimizer, BS, gamma, lam, p, precond, beta, alpha, corrupt


def plot_gammas(corrupt, precond=None):
    # plots gammas for corrupted datasets with preconditioning

    # Gather data of gammas
    data_dict = {}
    for dataset in DATASETS:
        for optimizer in OPTIMIZERS:
            pattern = f"{LOG_DIR}/{dataset}/{optimizer}(*).pkl"
            for fname in glob.glob(pattern):
                args = unpack_args(fname)
                gamma = args[3]
                if args[4] != 0.0: continue
                if args[6] != precond or args[9] != corrupt:
                    continue
                # load data
                data = loaddata(fname)
                if len(data) == 0:
                    print(fname, "has no data!")
                    continue
                if (dataset, optimizer) not in data_dict:
                    data_dict[(dataset, optimizer)] = [(data, gamma)]
                else:
                    data_dict[(dataset, optimizer)] += [(data, gamma)]

    # Prepare overall figure
    fig, axes = plt.subplots(len(OPTIMIZERS), len(DATASETS))
    fig.set_size_inches(5*len(DATASETS), 5*len(OPTIMIZERS))
    title = rf"Optimizer performance per $\gamma$"
    title += " on scaled dataset" if corrupt else ""
    title += " with preconditioning" if precond else ""
    plt.suptitle(title)

    # Go through dataset-optimzer pairs and plot their gamma-data
    for i, optimizer in enumerate(OPTIMIZERS):
        for j, dataset in enumerate(DATASETS):
            # unpack data
            if (dataset, optimizer) not in data_dict:
                continue
            data_list = data_dict[(dataset, optimizer)]
            markers = cycle(MARKERS)
            # XXX: hack to reduce number of gammas
            for data, gamma in sorted(data_list, key=lambda t: t[1]):
                # gamma legend label
                m = next(markers)
                gamma_p = int(log2(gamma))
                gamma_str = rf"2^{{{gamma_p}}}"
                axes[i,j].set_title(f"{optimizer}({dataset})")
                axes[i,j].semilogy(data[:,0], data[:,2],
                                   label=rf"$\gamma = {gamma_str}$", marker=m)
                axes[i,j].set_ylabel(r"$||\nabla F(w_t)||^2$")
                axes[i,j].set_xlabel("Effective Passes")
                axes[i,j].legend(fontsize=10, loc=1, prop={'size': 7})
                #axes[i,j].set_ylim(bottom=10**-15)

    fig.tight_layout()
    plt.savefig(f"plots/gammas(corrupt={corrupt},precond={precond is not None}).png")


def plot_optimizers(corrupt, precond):

    gamma = OPTIMAL_GAMMAS[(corrupt, precond is not None)]

    # Gather data of gammas
    data_dict = {}
    for dataset in DATASETS:
        for optimizer in OPTIMIZERS:
            pattern = f"{LOG_DIR}/{dataset}/{optimizer}(*).pkl"
            for fname in glob.glob(pattern):
                args = unpack_args(fname)
                if args[3] != gamma:
                    continue
                if args[4] != 0.0: continue
                if args[6] != precond or args[9] != corrupt:
                    continue
                # assuming there is one file of such pattern
                data = loaddata(fname)
                if len(data) == 0: continue
                if dataset not in data_dict:
                    data_dict[dataset] = [(data, optimizer)]
                else:
                    data_dict[dataset] += [(data, optimizer)]

    fig, axes = plt.subplots(2, len(DATASETS))
    fig.set_size_inches(5*len(DATASETS), 10)
    gamma_p = int(log2(gamma))
    gamma_str = rf"2^{{{gamma_p}}}"
    plt.suptitle(rf"Optimizers best performance ($\gamma={gamma_str}$)")
    markers = cycle(MARKERS)
    for j, dataset in enumerate(DATASETS):
        if dataset not in data_dict:
            continue
        markers = cycle(MARKERS)
        for data, optimizer in data_dict[dataset]:
            m = next(markers)
            axes[0,j].set_title(dataset)
            axes[0,j].semilogy(data[:,0], data[:,2], label=optimizer, marker=m)
            axes[0,j].set_ylabel(r"$||\nabla F(w_t)||^2$")
            axes[0,j].set_xlabel("Effective Passes")
            #axes[0,j].set_ylim(bottom=10**-15)
            axes[0,j].legend()

            axes[1,j].set_title(dataset)
            axes[1,j].plot(data[:,0], data[:,3], label=optimizer, marker=m)
            axes[1,j].set_ylabel("Error")
            axes[1,j].set_xlabel("Effective Passes")
            axes[1,j].legend()

    fig.tight_layout()
    plt.savefig(f"plots/optimizers(corrupt={corrupt},precond={precond is not None}).png")


def main():
    if not os.path.isdir("plots"):
        os.mkdir("plots")

    plot_gammas(None, None)
    plot_gammas("[-5-0]", None)
    plot_gammas("[0-5]", None)
    plot_gammas(None, "hutchinson")
    plot_gammas("[-5-0]", "hutchinson")
    plot_gammas("[0-5]", "hutchinson")

    plot_optimizers(None, None)
    plot_optimizers("[-5-0]", None)
    plot_optimizers("[0-5]", None)
    plot_optimizers(None, "hutchinson")
    plot_optimizers("[-5-0]", "hutchinson")
    plot_optimizers("[0-5]", "hutchinson")

if __name__ == "__main__":
    main()

