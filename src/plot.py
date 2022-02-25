
import argparse
import pickle
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from math import log2
from itertools import cycle, product
from collections import defaultdict
MARKERS = (',', '+', '.', 'o', '*', "D")

LOG_DIR = "logs4"
DATASETS = ("a9a",  "w8a", "rcv1", "real-sim",)
OPTIMIZERS = ("SGD", "SARAH", "L-SVRG")


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


def savefig2(data, optimum, fname, title="Loss, gradient norm squared, and error"):
    fig, axes = plt.subplots(2, 2)
    fig.set_size_inches(15, 10)
    plt.suptitle(title)
    margin = 0.02

    axes[0,0].plot(data[:,0], data[:,1])
    axes[0,0].set_ylabel(r"$F(w_t)$")
    axes[0,0].set_xlabel("Effective Passes")
    axes[0,0].grid()
    #print(f"Final loss = {data[-1,1]:.6f}")

    axes[0,1].semilogy(data[:,0], data[:,2])
    axes[0,1].set_ylabel(r"$||\nabla F(w_t)||^2$")
    axes[0,1].set_xlabel("Effective Passes")
    axes[0,1].grid()

    axes[1,0].plot(data[:,0], data[:,3])
    axes[1,0].set_ylabel("Error")
    axes[1,0].set_xlabel("Effective Passes")
    axes[1,0].set_ylim([0-margin,1+margin])
    axes[1,0].grid()

    axes[1,1].plot(data[:,0],data[:,4])
    axes[1,1].set_ylabel(r"%($D_i > \alpha$)")
    axes[1,1].set_xlabel("Effective Passes")
    axes[1,1].set_ylim([0-margin,1+margin])
    axes[1,1].grid()

    if fname is None:
        plt.show()
    else:
        plt.savefig(fname)
        plt.close()

        fig = plt.figure()
        fig.set_size_inches(8, 6)
        plt.suptitle(r"Hessian Diagonal Estimate Relative Error at $w_t$")
        plt.plot(data[:,0], data[:,5])
        plt.ylabel("Relative Error")
        plt.xlabel("Effective Passes")
        plt.grid()
        plt.savefig("H_diag_err_t.png")
        plt.close()


def plot_H_acc(H_diag, D):
    fig = plt.figure()
    fig.set_size_inches(8, 6)
    plt.suptitle("True Hessian Diagonal Vs. Hutchinson's Diagonal Estimate")
    #plt.plot(H_diag, D, '.', label=r"$D_0$")
    plt.loglog(H_diag, D, '.', label=r"$D_0$")
    lim = max(H_diag.max(), D.max())
    plt.plot([0, lim], [0, lim], '--', label=r"$x=y$")
    plt.xlabel(r"$diag(H(w_0))$")
    plt.ylabel(r"$D_0$")
    plt.legend()
    plt.savefig("H_vs_D.png")
    plt.close()


def plot_H_approx(H_diag_errs):
    fig = plt.figure()
    fig.set_size_inches(8, 6)
    plt.suptitle(r"Hessian Diagonal Estimate Relative Error at $w_0$")
    plt.plot(H_diag_errs)
    plt.ylabel(r"Relative Error")
    plt.xlabel(r"Number of Samples")
    plt.grid()
    plt.savefig("H_diag_err_0.png")
    plt.close()


def get_logs(dataset, optimizer, **filter_args):
    """
    Return all logs having the same given args.
    """
    fpattern = f"{LOG_DIR}/{dataset}/{optimizer}(*).pkl"
    for fname in glob.glob(fpattern):
        exp_args = unpack_args(fname)
        if any(exp_args[k] != v for k, v in filter_args.items() if k in exp_args):
            continue
        # load data @TODO: logs should not be empty in the first place
        data = loaddata(fname)
        if len(data) == 0:
            print(fname, "has no data!")
            continue
        yield data, exp_args

def unpack_args(fname):
    args = {}
    # unpack path
    dirname, logname = os.path.split(fname)
    logdir, args["dataset"] = os.path.split(dirname)
    # parse args
    args["optimizer"], argstr = logname.split("(")
    argstr, _ = argstr.split(")")  # remove ').pkl'
    args_str = {k:v for k,v in [s.split("=") for s in argstr.split(",")]}
    args["BS"] = int(args_str["BS"])
    args["gamma"] = float(args_str["gamma"])
    args["lam"] = float(args_str["lam"])
    if "p" in args_str:
        args["p"] = args_str["p"]
    if "precond" in args_str:
        args["precond"] = args_str["precond"]
        args["beta"] = float(args_str["beta"])
        args["alpha"] = float(args_str["alpha"])
        args["precond_resample"] = args_str["precond_resample"] == "True"
        args["precond_warmup"] = int(args_str["precond_warmup"])
    else:
        args["precond"] = None
    if "corrupt" in args_str:
        args["corrupt"] = args_str["corrupt"]
    else:
        args["corrupt"] = None

    return args


def plot_gammas(**filter_args):
    """
    Plot performace (gradnorm^2) for all gammas given filter_args.
    Return optimal gammas for each (dataset, optimizer) given filter_args.
    """
    optimal_gammas = {}
    filter_args_str = ",".join(f"{k}={v}" for k,v in filter_args.items())

    # Gather data
    data_dict = defaultdict(list)
    for exp in product(DATASETS, OPTIMIZERS):
        for data, args in get_logs(*exp, **filter_args):
            # Record experiment data
            data_dict[exp] += [(args["gamma"], data)]
            # Track gammas with the best performances for this experiment setting
            error_gamma = (data[-1,3], args["gamma"])
            if exp not in optimal_gammas:
                optimal_gammas[exp] = error_gamma
            elif error_gamma[0] < optimal_gammas[exp][0]:
                optimal_gammas[exp] = error_gamma
    optimal_gammas = {k:v[1] for k,v in optimal_gammas.items()}  # remove errors

    # Plot data for all gammas
    fig, axes = plt.subplots(len(OPTIMIZERS), len(DATASETS))
    fig.set_size_inches(5*len(DATASETS), 5*len(OPTIMIZERS))
    title = rf"Performance for all $\gamma$ ({filter_args_str})"
    plt.suptitle(title)
    for i, optimizer in enumerate(OPTIMIZERS):
        for j, dataset in enumerate(DATASETS):
            markers = cycle(MARKERS)
            exp = (dataset, optimizer)
            if exp not in data_dict:
                continue
            for gamma, data in sorted(data_dict[exp], key=lambda t: t[0]):
                m = next(markers)
                axes[i,j].set_title(f"{optimizer}({dataset})")
                axes[i,j].semilogy(data[:,0], data[:,2],
                                   label=rf"$\gamma = 2^{{{round(log2(gamma))}}}$",
                                   marker=m)
                axes[i,j].set_ylabel(r"$||\nabla F(w_t)||^2$")
                axes[i,j].set_xlabel("Effective Passes")
                axes[i,j].legend(fontsize=10, loc=1, prop={'size': 7})
    fig.tight_layout()

    plt.savefig(f"plots/gammas({filter_args_str}).png")
    plt.close()
    return optimal_gammas


def plot_optimizers(optimal_gammas, **filter_args):
    """
    Plot performances for all optimizers (given optimal gamma)
    on each dataset given filter_args.
    """
    filter_args_str = ",".join(f"{k}={v}" for k,v in filter_args.items())

    # Gather data
    data_dict = defaultdict(list)
    for dataset, optimizer in product(DATASETS, OPTIMIZERS):
        exp = (dataset, optimizer)
        filter_args["gamma"] = optimal_gammas[exp]
        for data, args in get_logs(*exp, **filter_args):
            # Record experiment data
            data_dict[dataset] += [(optimizer, data)]

    fig, axes = plt.subplots(2, len(DATASETS))
    fig.set_size_inches(5*len(DATASETS), 10)
    plt.suptitle(rf"Top performance per optimizer")
    for j, dataset in enumerate(data_dict.keys()):
        markers = cycle(MARKERS)
        for optimizer, data in data_dict[dataset]:
            m = next(markers)
            gamma = optimal_gammas[(dataset, optimizer)]
            label = rf"{optimizer} ($\gamma = 2^{{{round(log2(gamma))}}}$)"
            axes[0,j].set_title(dataset)
            axes[0,j].semilogy(data[:,0], data[:,2], label=label, marker=m)
            axes[0,j].set_ylabel(r"$||\nabla F(w_t)||^2$")
            axes[0,j].set_xlabel("Effective Passes")
            axes[0,j].legend()
            axes[1,j].set_title(dataset)
            axes[1,j].plot(data[:,0], data[:,3], label=label, marker=m)
            axes[1,j].set_ylabel("Error")
            axes[1,j].set_xlabel("Effective Passes")
            axes[1,j].legend()

    fig.tight_layout()
    plt.savefig(f"plots/optimizers({filter_args_str}).png")
    plt.close()


def main():
    if not os.path.isdir("plots"):
        os.mkdir("plots")

    hyperparams_dict = {
        "corrupt": ("[-2-5]", "[-6-3]"),
        "precond_resample": (False, True),
        "precond_warmup": (1,2,4,8),
    }
    for hyperparams in product(*hyperparams_dict.values()):
        filter_args = dict(zip(hyperparams_dict.keys(), hyperparams))
        print("Plotting:", filter_args, "...")
        optimal_gammas = plot_gammas(**filter_args)
        plot_optimizers(optimal_gammas, **filter_args)

if __name__ == "__main__":
    main()

