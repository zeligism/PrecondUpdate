
import argparse
import pickle
import os
import glob
import numpy as np
import matplotlib.pyplot as plt


from run_experiment import DATASETS, OPTIMIZERS


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
    ax1.set_xlabel("effective passes")
    ax1.grid()

    ax2.semilogy(data[:,0], data[:,2])
    ax2.set_ylabel(r"$||\nabla F(w_t)||^2$")
    ax2.set_xlabel("effective passes")
    ax2.grid()

    ax3.plot(data[:,0], data[:,3])
    ax3.set_ylabel("error")
    ax3.set_xlabel("effective passes")
    ax3.grid()
    
    if fname is None:
        plt.show()
    else:
        plt.savefig(fname)
        plt.close()

def unpack_args(fname):
    # unpack path
    dirname, logname = os.path.split(fname)
    logdir, datasetname = os.path.split(dirname)
    # check if corrupted dataset
    bad_index = datasetname.find("_bad")
    if bad_index != -1:
        dataset = datasetname[:bad_index]
        corrupt = []  # default
    else:
        dataset = datasetname
        corrupt = None
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

    return dataset, optimizer, BS, gamma, lam, p, precond, beta, alpha, corrupt


def plot_gammas():
    # plots gammas for corrupted datasets with preconditioning
    # for testing
    #DATASETS = ("a1a", "w8a")
    #OPTIMIZERS = ("SGD", "SARAH")

    # Gather data of gammas
    data_dict = {}
    for dataset in DATASETS:
        for optimizer in OPTIMIZERS:
            for fname in glob.glob(f"log/{dataset}_bad/{optimizer}(*precond=hutchinson*).pkl"):
                gamma = unpack_args(fname)[3]
                data = loaddata(fname)
                if len(data) == 0: continue
                if (dataset, optimizer) not in data_dict:
                    data_dict[(dataset, optimizer)] = [(data, gamma)]
                else:
                    data_dict[(dataset, optimizer)] += [(data, gamma)]

    fig, axes = plt.subplots(len(OPTIMIZERS), len(DATASETS))
    fig.set_size_inches(20, 12)
    plt.suptitle(rf"Optimizer performance per $\gamma$ per dataset")
    for i, optimizer in enumerate(OPTIMIZERS):
        for j, dataset in enumerate(DATASETS):
            # unpack data
            if (dataset, optimizer) not in data_dict:
                continue
            data_list = data_dict[(dataset, optimizer)]
            for data, gamma in data_list:
                # XXX: hack to reduce number of gammas
                if gamma in [2**i for i in range(-20,6,2)]:
                    continue
                axes[i,j].set_title(f"{optimizer}({dataset})")
                axes[i,j].semilogy(data[:,0], data[:,2], label=rf"$\gamma$={gamma}")
                axes[i,j].set_ylabel(r"$||\nabla F(w_t)||^2$")
                axes[i,j].set_xlabel("effective passes")
                axes[i,j].legend()

    fig.tight_layout()
    plt.savefig(f"plots/gammas.png")


def plot_optimizers():
    # for testing
    #DATASETS = ("a1a", "w8a")
    #OPTIMIZERS = ("SGD", "SARAH")

    optimal_gammas = {"SGD": 0.0002, "SARAH": 0.02, "SVRG": 0.02}  # optimal gamma

    # Gather data of gammas
    data_dict = {}
    for dataset in DATASETS:
        for optimizer in OPTIMIZERS:
            gamma = optimal_gammas[optimizer]
            for fname in glob.glob(f"log/{dataset}_bad/{optimizer}(*gamma={gamma}*hutchinson*).pkl"):
                # assuming there is one file of such pattern
                data = loaddata(fname)
                if len(data) == 0: continue
                if dataset not in data_dict:
                    data_dict[dataset] = [(data, optimizer)]
                else:
                    data_dict[dataset] += [(data, optimizer)]

    fig, axes = plt.subplots(2, len(DATASETS))
    fig.set_size_inches(20, 8)
    plt.suptitle(rf"Optimizers performance per dataset")
    for j, dataset in enumerate(DATASETS):
        if dataset not in data_dict:
            continue
        for data, optimizer in data_dict[dataset]:
            axes[0,j].set_title(dataset)
            axes[0,j].semilogy(data[:,0], data[:,2], label=optimizer)
            axes[0,j].set_ylabel(r"$||\nabla F(w_t)||^2$")
            axes[0,j].set_xlabel("effective passes")
            axes[0,j].legend()

            axes[1,j].set_title(dataset)
            axes[1,j].plot(data[:,0], data[:,3], label=optimizer)
            axes[1,j].set_ylabel("error")
            axes[1,j].set_xlabel("effective passes")
            axes[1,j].legend()

    fig.tight_layout()
    plt.savefig(f"plots/optimizers.png")


def main():
    if not os.path.isdir("plots"):
        os.mkdir("plots")
    plot_gammas()
    plot_optimizers()

if __name__ == "__main__":
    main()

