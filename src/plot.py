
import argparse
import pickle
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.colors import LogNorm
from math import log2
from itertools import cycle, product
from collections import defaultdict
MARKERS = (',', '+', '.', 'o', '*', "D")
METRICS = ("loss", "gradnorm", "error")
AGGS = ("median", "mean")
AGG = AGGS[0]
LOSS = METRICS.index("loss") + 1
GRADNORM = METRICS.index("gradnorm") + 1
ERROR = METRICS.index("error") + 1

# These should be the same as the one used in run_experiment.py
DATASETS = ("a9a", "w8a", "rcv1", "real-sim",)
OPTIMIZERS = ("SGD", "SARAH", "L-SVRG")
T = 50

# Use Seaborn for plots
SEABORN = True

# Helps for accelerating seaborn plots for multi-seed runs
EP_DILUTION = 1

### Restrict plots for each combination of hyperparameter setting
HYPERPARAMS_DICT = {
    #"alpha": (1e-3,),
    "BS": (128,),
    "corrupt": ("[-3-0]", "[0-3]", "[-3-3]"),
}

# These are always clearly worse than optimal, so just ignore them
IGNORE_HYPERPARAMS = {
    "alpha": [1e-9],
    "BS": [2048],
    "gamma": [2**-16, 2**-18, 2**-20],
}


def ignore(args):
    return any(args[hp] in IGNORE_HYPERPARAMS[hp] for hp in IGNORE_HYPERPARAMS.keys() if hp in args)


def loaddata(fname):
    with open(fname, 'rb') as f:
        data = pickle.load(f)
    return data


def contain_dict(dict1, dict2):
    return all(dict1[k] == v for k, v in dict2.items() if k in dict1)


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


def get_logs(logdir, dataset, optimizer, **filter_args):
    """
    Return all logs having the same given args.
    """
    fpattern = f"{logdir}/{dataset}/{optimizer}(*).pkl"
    for fname in glob.glob(fpattern):
        exp_args = unpack_args(fname)
        if not contain_dict(exp_args, filter_args):
            continue
        # load data @TODO: logs should not be empty in the first place
        data = loaddata(fname)
        if len(data) == 0:
            print(fname, "has no data!")
            continue
        # @XXX: correct a wrong initial ep for L-SVRG
        ep0 = data[0,0]
        if ep0 > 0.:
            data[:,0] -= ep0
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
        #args["precond_resample"] = args_str["precond_resample"] == "True"
        #args["precond_warmup"] = int(args_str["precond_warmup"])
    else:
        args["precond"] = "None"
        args["alpha"] = "None"
    if "corrupt" in args_str:
        args["corrupt"] = args_str["corrupt"]
    else:
        args["corrupt"] = "None"

    if "seed" in args_str:
        args["seed"] = int(args_str["seed"])
    else:
        args["seed"] = 123  # default value

    return args


def get_optimal_hyperparams(logdir, metric=LOSS, **filter_args):
    # Gather data
    best_args_perf = {}
    best_data_dict = {}
    all_data_dict = defaultdict(list)
    for exp in product(DATASETS, OPTIMIZERS):
        for data, args in get_logs(logdir, *exp, **filter_args):
            if ignore(args):
                continue
            # Record experiment data
            final_ep = np.where(data[:,0] <= T)[0][-1] + 1
            final_ep = min(final_ep, data.shape[0])
            args_data = (args, data[:final_ep, :])
            args_perf = (args, data[final_ep, metric])
            all_data_dict[exp] += [args_data]
            # Track gammas with the best metric for this experiment
            if exp not in best_args_perf:
                best_args_perf[exp] = args_perf
                best_data_dict[exp] = args_data
            elif args_perf[1] < best_args_perf[exp][1]:
                best_args_perf[exp] = args_perf
                best_data_dict[exp] = args_data

    # remove perf
    # best_args = {k:v[0] for k,v in best_args_perf.items()}

    return all_data_dict, best_data_dict


def sns_get_optimal_hyperparams(logdir, metric=LOSS, agg=AGG, **filter_args):
    # @XXX
    columns = ["ep", "loss", "gradnorm", "error"]
    argcols = ["gamma", "BS", "precond", "alpha"]
    metric = columns[metric]
    # Gather data
    all_df = {}
    best_df = {}
    for exp in product(DATASETS, OPTIMIZERS):
        exp_df = pd.DataFrame()
        for data, args in get_logs(logdir, *exp, **filter_args):
            if ignore(args):
                continue
            # Record experiment data
            df = pd.DataFrame(data[:, :4], columns=columns)
            for col in argcols:
                df[col] = args[col]
            # Dilute iterations if averaging across seeds @XXX
            df = df[df["ep"] <= T].iloc[::EP_DILUTION]
            exp_df = exp_df.append(df, ignore_index=True)

        # Set index to arg settings
        # Get performance at last iteration
        max_ep = exp_df.groupby(argcols, sort=False)["ep"].transform(max)
        perf = exp_df[exp_df["ep"] == max_ep].drop("ep", axis=1)
        # Find the minimum aggregated metric (based on mean, median, etc.)
        if agg == "mean":
            agg_perf = perf.groupby(argcols).mean()
        elif agg == "median":
            agg_perf = perf.groupby(argcols).median()
        min_agg_perf = agg_perf[agg_perf[metric] == agg_perf.min()[metric]]

        # Get the data associated with the args of the min aggregated metric
        best_df[exp] = exp_df.set_index(argcols).loc[min_agg_perf.index]
        all_df[exp] = exp_df

    return all_df, best_df


def plot_all_hyperparams(data_dict, **filter_args):
    filter_args_str = ",".join(f"{k}={v}" for k,v in filter_args.items())
    # Plot data for all args
    fig, axes = plt.subplots(len(OPTIMIZERS), len(DATASETS))
    fig.set_size_inches(5 * len(DATASETS), 5 * len(OPTIMIZERS))
    title = rf"Performances for ({filter_args_str})"
    plt.suptitle(title)
    for i, optimizer in enumerate(OPTIMIZERS):
        for j, dataset in enumerate(DATASETS):
            markers = cycle(MARKERS)
            exp = (dataset, optimizer)
            if exp not in data_dict:
                continue
            for args, data in sorted(data_dict[exp], key=lambda t: (t[0]["BS"], t[0]["alpha"], t[0]["gamma"])):
                m = next(markers)
                gamma_pow = round(log2(args["gamma"]))
                axes[i,j].set_title(f"{optimizer}({dataset})")
                axes[i,j].semilogy(data[:,0], data[:,2],
                                   label=rf"$\gamma = 2^{{{gamma_pow}}}$, "
                                   rf"$BS={args['BS']}$, "
                                   rf"$\alpha={args['alpha']}$",
                                   marker=m)
                axes[i,j].set_ylabel(r"$||\nabla F(w_t)||^2$")
                axes[i,j].set_xlabel("Effective Passes")
                axes[i,j].legend(fontsize=10, loc=1, prop={'size': 7})
    fig.tight_layout()

    plt.savefig(f"plots/gammas({filter_args_str}).png")
    plt.close()


def sns_plot_all_hyperparams(data_dict, **filter_args):
    columns = ("ep", "loss", "gradnorm", "error")
    argcols = ("gamma", "BS", "precond", "alpha")
    filter_args_str = ",".join(f"{k}={v}" for k,v in filter_args.items())
    # Plot data for all args
    fig, axes = plt.subplots(len(OPTIMIZERS), len(DATASETS))
    fig.set_size_inches(5 * len(DATASETS), 5 * len(OPTIMIZERS))
    title = rf"Performances for ({filter_args_str})"
    plt.suptitle(title)
    for i, optimizer in enumerate(OPTIMIZERS):
        for j, dataset in enumerate(DATASETS):
            markers = cycle(MARKERS)
            exp = (dataset, optimizer)
            if exp not in data_dict:
                continue
            exp_df = data_dict[exp]
            m = next(markers)
            axes[i,j].set_title(f"{optimizer}({dataset})")
            sns.lineplot(ax=axes[i,j],x="ep", y="gradnorm", hue="gamma",
                         hue_norm=LogNorm(), palette="vlag",
                         size="BS", sizes=(1, 2), style="alpha", data=exp_df)
            axes[i,j].set(yscale="log")
            axes[i,j].set_ylabel(r"$||\nabla F(w_t)||^2$")
            axes[i,j].set_xlabel("Effective Passes")
    fig.tight_layout()

    plt.savefig(f"plots/gammas({filter_args_str}).png")
    plt.close()


def plot_optimal_hyperparams(best_data, **filter_args):
    """
    Plot performances for all optimizers (given optimal gamma)
    on each dataset given filter_args.
    """
    filter_args_str = ",".join(f"{k}={v}" for k,v in filter_args.items())

    fig, axes = plt.subplots(3, len(DATASETS))
    fig.set_size_inches(5 * len(DATASETS), 5 * 3)
    plt.suptitle(rf"Top performance per optimizer for ({filter_args_str})")
    for j, dataset in enumerate(DATASETS):
        for optimizer in OPTIMIZERS:
            exp = (dataset, optimizer)
            args, data = best_data[exp]
            gamma_pow = round(log2(args["gamma"]))
            sublabel = rf"$\gamma = 2^{{{gamma_pow}}}$, $BS={args['BS']}$, $\alpha={args['alpha']}$"
            label = rf"{optimizer}({sublabel})"
            ep = data[:,0]
            loss = data[:,1]
            gradnorm = data[:,2]
            error = data[:,3]
            axes[0,j].plot(ep, loss, label=label)
            axes[1,j].semilogy(ep, gradnorm, label=label)
            axes[2,j].plot(ep, error, label=label)
        # Loss
        axes[0,j].set_title(dataset)
        axes[0,j].set_ylabel(r"$F(w_t)$")
        axes[0,j].set_xlabel("Effective Passes")
        #axes[0,j].set_ylim(bottom=loss.min(), top=np.log(2))  # Initial loss F(w_0)=log(2)
        axes[0,j].legend()
        # Gradnorm
        axes[1,j].set_title(dataset)
        axes[1,j].set_ylabel(r"$||\nabla F(w_t)||^2$")
        axes[1,j].set_xlabel("Effective Passes")
        axes[1,j].legend()
        # Error
        axes[2,j].set_title(dataset)
        axes[2,j].set_ylabel("Error")
        axes[2,j].set_xlabel("Effective Passes")
        axes[2,j].legend()

    fig.tight_layout()
    plt.savefig(f"plots/optimizers({filter_args_str}).png")
    plt.close()


def sns_plot_optimal_hyperparams(best_data, **filter_args):
    """
    Plot performances for all optimizers (given optimal gamma)
    on each dataset given filter_args.
    """
    filter_args_str = ",".join(f"{k}={v}" for k,v in filter_args.items())

    fig, axes = plt.subplots(3, len(DATASETS))
    fig.set_size_inches(5 * len(DATASETS), 5 * 3)
    plt.suptitle(rf"Top performance per optimizer for ({filter_args_str})")
    for j, dataset in enumerate(DATASETS):
        markers = cycle(MARKERS)
        for optimizer in OPTIMIZERS:
            exp = (dataset, optimizer)
            args = {k:v for k,v in zip(best_data[exp].index.names, best_data[exp].index[0])}
            exp_df = best_data[exp].reset_index()
            gamma_pow = round(log2(args['gamma']))
            #sublabel = rf"$\gamma = 2^{{{gamma_pow}}}$, $BS={args['BS']}$, $\alpha={args['alpha']}$"
            sublabel = rf"$\gamma = 2^{{{gamma_pow}}}$, $\alpha={args['alpha']}$"
            label = rf"{optimizer}({sublabel})"
            sns.lineplot(x="ep", y="loss", label=label, ax=axes[0,j], data=exp_df)
            sns.lineplot(x="ep", y="gradnorm", label=label, ax=axes[1,j], data=exp_df)
            sns.lineplot(x="ep", y="error", label=label, ax=axes[2,j], data=exp_df)
        # Loss
        axes[0,j].set_title(dataset)
        axes[0,j].set_ylabel(r"$F(w_t)$")
        axes[0,j].set_xlabel("Effective Passes")
        axes[0,j].legend()
        # Gradnorm
        axes[1,j].set(yscale="log")
        axes[1,j].set_title(dataset)
        axes[1,j].set_ylabel(r"$||\nabla F(w_t)||^2$")
        axes[1,j].set_xlabel("Effective Passes")
        axes[1,j].legend()
        # Error
        axes[2,j].set(yscale="log")
        axes[2,j].set_title(dataset)
        axes[2,j].set_ylabel("Error")
        axes[2,j].set_xlabel("Effective Passes")
        axes[2,j].legend()

    fig.tight_layout()
    plt.savefig(f"plots/optimizers({filter_args_str}).png")
    plt.close()


def generate_plots(hp_dict, metric, logdir, seaborn=SEABORN):
    for hyperparams in product(*hp_dict.values()):
        filter_args = dict(zip(hp_dict.keys(), hyperparams))
        print("Plotting:", filter_args, "...")
        if seaborn:
            all_data, best_data = sns_get_optimal_hyperparams(logdir, metric=metric, **filter_args)
            sns_plot_all_hyperparams(all_data, **filter_args)
            sns_plot_optimal_hyperparams(best_data, **filter_args)
        else:
            all_data, best_data = get_optimal_hyperparams(logdir, metric=metric, **filter_args)
            plot_all_hyperparams(all_data, **filter_args)
            plot_optimal_hyperparams(best_data, **filter_args)

        metric_name = (None, "loss", "gradnorm", "error")[metric]
        print(f"Optimal gammas using {metric_name} metric:")
        display_best_performances(best_data, seaborn=seaborn)


def generate_plots_compare_precond(hp_dict, metric, logdir):
    if "precond" in hp_dict:
        del hp_dict["precond"]
    for hyperparams in product(*hp_dict.values()):
        filter_args = dict(zip(hp_dict.keys(), hyperparams))
        filter_args_str = ",".join(f"{k}={v}" for k,v in filter_args.items())
        print("Comparing preconditoning for hyperparameters:", filter_args)
        filter_args["precond"] = "None"
        _, best_data_without = sns_get_optimal_hyperparams(logdir, metric=metric, **filter_args)
        filter_args["precond"] = "hutchinson"
        _, best_data_with = sns_get_optimal_hyperparams(logdir, metric=metric, **filter_args)
        del filter_args["precond"]

        # Plot almost same as sns_plot_optimal_hyperparams
        fig, axes = plt.subplots(3, len(DATASETS))
        fig.set_size_inches(5 * len(DATASETS), 5 * 3)
        plt.suptitle(rf"Top performance with preconditioning vs. without")
        for j, dataset in enumerate(DATASETS):
            optim_df = pd.DataFrame()
            for optimizer in OPTIMIZERS:
                exp = (dataset, optimizer)
                exp_df = best_data_without[exp].append(best_data_with[exp])
                exp_df["optimizer"] = optimizer
                optim_df = optim_df.append(exp_df)
            # reset index and combine precond with gamma
            optim_df = optim_df.reset_index()
            sns.lineplot(x="ep", y="loss", hue="optimizer", style="precond", ax=axes[0,j], data=optim_df)
            sns.lineplot(x="ep", y="gradnorm", hue="optimizer", style="precond", ax=axes[1,j], data=optim_df)
            sns.lineplot(x="ep", y="error", hue="optimizer", style="precond", ax=axes[2,j], data=optim_df)
            # Loss
            axes[0,j].set_title(dataset)
            axes[0,j].set_ylabel(r"$F(w_t)$")
            axes[0,j].set_xlabel("Effective Passes")
            # Gradnorm
            axes[1,j].set(yscale="log")
            axes[1,j].set_title(dataset)
            axes[1,j].set_ylabel(r"$||\nabla F(w_t)||^2$")
            axes[1,j].set_xlabel("Effective Passes")
            # Error
            axes[2,j].set(yscale="log")
            axes[2,j].set_title(dataset)
            axes[2,j].set_ylabel("Error")
            axes[2,j].set_xlabel("Effective Passes")

        fig.tight_layout()
        plt.savefig(f"plots/compare_optimizers({filter_args_str}).png")
        plt.close()

        metric_name = (None, "loss", "gradnorm", "error")[metric]
        print(f"Optimal gammas using {metric_name} metric WITHOUT preconditoning:")
        display_best_performances(best_data_without, seaborn=True)
        print(f"Optimal gammas using {metric_name} metric WITH preconditoning:")
        display_best_performances(best_data_with, seaborn=True)


def display_best_performances(best_data, seaborn=SEABORN):
    for dataset in DATASETS:
        for optimizer in OPTIMIZERS:
            # Extract best performance metrics for each experiment
            exp = (dataset, optimizer)
            if seaborn:
                args = {k:v for k,v in zip(best_data[exp].index.names, best_data[exp].index[0])}
                exp_df = best_data[exp].reset_index()
                loss = exp_df["loss"].iloc[-1]
                gradnorm = exp_df["gradnorm"].iloc[-1]
                error = exp_df["error"].iloc[-1]
            else:
                args, data = best_data[exp]
                loss = data[-1,LOSS]
                gradnorm = data[-1,GRADNORM]
                error = data[-1,ERROR]
            # Print report
            print(f"{exp}:"
                  f"\tgamma = 1/{int(1/args['gamma'])},"
                  f"\tBS = {args['BS']},"
                  f"\talpha = {args['alpha']},"
                  f"\tloss = {loss},"
                  f"\tgradnorm = {gradnorm},"
                  f"\terror = {error}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Generate different performances plots from optimization logs")
    parser.add_argument("--logdir", type=str, default="logs", help="name of logs directory")
    parser.add_argument("--plotdir", type=str, default="plots", help="name of plots directory")
    parser.add_argument("--metric", type=str, choices=METRICS, default=METRICS[0],
                        help="name of metric with respect to which the optimal hyperparams will be chosen")
    parser.add_argument("--compare-precond", action="store_true",
                        help="compare precond mode (logs should contain runs with precond and without)")
    args = parser.parse_args()

    ### HYPERPARAMS_DICT should be edited manually at the top, sorry ###
    if not os.path.isdir(args.plotdir):
        os.mkdir(args.plotdir)
    metric_index = METRICS.index(args.metric) + 1
    if args.compare_precond:
        generate_plots_compare_precond(HYPERPARAMS_DICT, metric_index, args.logdir)
    else:
        generate_plots(HYPERPARAMS_DICT, metric_index, args.logdir)


if __name__ == "__main__":
    main()
