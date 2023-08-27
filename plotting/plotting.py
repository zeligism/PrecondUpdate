import os
import glob
import pickle
import time
from math import log2
from itertools import cycle, product
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LogNorm


# Aspect ratio and height of subplots
ASPECT = 4. / 3.
HEIGHT = 3.
HEIGHT_LARGE = 4.
LEGEND_FONTSIZE = "x-small"
LEGEND_LOC = "upper right"
LOG_SCALE = {
    "loss": False,
    "gradnorm": True,
    "error": True,
}
TO_MATH = {
    "loss": r"$P(w_t)$",
    "gradnorm": r"$||\nabla P(w_t)||^2$",
    "error": "Error",
    "alpha": r"$\alpha$",
    "beta2": r"$\beta$",
    "lr": r"$\eta$",
    "ep": "Effective Passes",
    "time": "Time (seconds)",
}


def as_power_of_2(num):
    return "2**" + str(int(log2(float(num))))


def ignore(args, args_dict):
    return any(args_dict[arg] in map(str, args.IGNORE_ARGS[arg])
               for arg in args.IGNORE_ARGS.keys() if arg in args_dict)


def loaddata(fname):
    with open(fname, 'rb') as f:
        data = pickle.load(f)
    return np.array(data)


def contain_dict(dict1, dict2):
    return all(dict1[k] == v for k, v in dict2.items() if k in dict1)


def unpack_args(fname):
    """
    Recover all args given file path.
    """
    # unpack path
    dirname, logname = os.path.split(fname)
    logdir, dataset = os.path.split(dirname)
    optimizer, log_args = logname.split("(")
    log_args, _ = log_args.split(")")  # e.g. remove ').pkl'
    args_dict = {k:v for k,v in [s.split("=") for s in log_args.split(",")]}

    args_dict["dataset"] = dataset
    args_dict["optimizer"] = optimizer

    # It is very unlikely that the original dataset name will end with ')'
    args_dict["corrupt"] = dataset[dataset.index("("):] if dataset[-1] == ")" else "none"

    # Set to default values if field does not exists
    if "seed" not in args_dict:
        args_dict["seed"] = '0'
    if "weight_decay" not in args_dict:
        args_dict["weight_decay"] = '0.0'
    if "lr_decay" not in args_dict:
        args_dict["lr_decay"] = '0.0'
    if "precond" not in args_dict:  # these are reserved for precond algs
        args_dict["precond"] = "none"
        args_dict["alpha"] = "none"
        args_dict["beta2"] = "none"

    return args_dict


def handle_empty_file(args, fname):
    print(fname, "has no data!")
    if not args.remove_empty_files:
        if "y" == input("Remove empty log files in the future without asking? y/(n)"):
            print("Will remove without asking.")
            args.remove_empty_files = True
        else:
            print("Will ask again before removing.")
    else:
        try:
            print("Removing", fname)
            os.remove(fname)
        except OSError as e:
            print ("Error: %s - %s." % (e.filename, e.strerror))


def get_logs(args, logdir, dataset, optimizer, **filter_args):
    """
    Return all logs in 'logdir' containing the filter hyperparams.
    Dataset name should contain feature scaling, if any
    e.g. 'dataset' or 'dataset(k_min,k_max)'.
    
    Returns the data in the log file and its arguments/hyperparams.
    """
    if "corrupt" in filter_args and filter_args['corrupt'] != "none":
        dataset += filter_args['corrupt']  # add scale suffix

    # Find all files matching this pattern
    for fname in glob.glob(f"{logdir}/{dataset}/{optimizer}(*).pkl"):
        exp_args = unpack_args(fname)
        # Skip if filter_args does not match args of this file
        if not contain_dict(exp_args, filter_args):
            continue
        data = loaddata(fname)
        if len(data) == 0:
            continue
        time_idx = args.DATA_INDICES[args.LOG_COLS.index(args.idx)]
        data[:, time_idx] -= min(data[:, time_idx])  # double check that time_idx starts at 0
        if len(data) == 0:
            handle_empty_file(args, fname)
            continue

        yield data, exp_args


def downsample_dataframe(args, df):
    # Downsample by averaging metrics every 'avg_downsample' epoch.
    if args.avg_downsample < 1:
        time_range = df[args.idx].max() - df[args.idx].min()
        effective_downsample = round(time_range * args.avg_downsample)
    else:
        effective_downsample = args.avg_downsample
    df[args.idx] = np.ceil(df[args.idx] / effective_downsample) * effective_downsample
    # df = df.groupby([args.idx] + args.ARG_COLS).mean().reset_index()
    return df


def create_experiments_dataframe(args):
    # Gather data
    all_dfs = {}
    start_time = time.time()
    for experiment in product(args.DATASETS, args.OPTIMIZERS):
        dataset, optimizer = experiment
        exp_dfs = []
        # Get all log data given the experiment and filter args
        for data, args_dict in get_logs(args, args.log_dir, dataset, optimizer, **args.filter_args):
            if ignore(args, args_dict):
                continue
            # Get experiment log data
            df = pd.DataFrame(data[:, args.DATA_INDICES], columns=args.LOG_COLS)
            # Get args of interest
            for col in args.ARG_COLS:
                df[col] = args_dict[col]
            df = downsample_dataframe(args, df)
            df = df[df[args.idx] <= args.MAX_IDX[args.idx]]  # cut data up to the prespecified time index
            exp_dfs.append(df)

        # Record all runs of exp in a single dataframe
        all_dfs[experiment] = pd.concat(exp_dfs, ignore_index=True)
        if len(exp_dfs) == 0:
            print(f"No log data found! Experiment = {experiment}, filter args = {args.filter_args}")
            continue

    # Get min time for each dataset (w and w/o precond) then cut runs up to that point
    min_last_idx = {dataset: float('inf') for dataset in args.DATASETS}
    for experiment in product(args.DATASETS, args.OPTIMIZERS):
        dataset, _ = experiment
        df = all_dfs[experiment]
        if "precond" in df.columns:
            precond_last_idx = 10**10
            noprecond_last_idx = 10**10
            if len(df[df["precond"] == "hutchinson"]) > 0:
                precond_last_idx = df[df["precond"] == "hutchinson"][args.idx].max()
            if len(df[df["precond"] == "none"]) > 0:
                noprecond_last_idx = df[df["precond"] == "none"][args.idx].max()
            min_last_idx[dataset] = min(min_last_idx[dataset], precond_last_idx, noprecond_last_idx)
    print(min_last_idx)
    for experiment in product(args.DATASETS, args.OPTIMIZERS):
        dataset, _ = experiment
        df = all_dfs[experiment]
        all_dfs[experiment] = df[df[args.idx] <= min_last_idx[dataset]]

    data_gather_time = time.time() - start_time
    print(f"Data frame lengths:")
    for exp, df in all_dfs.items():
        runs = len(df) // len(args.idx)
        print(f"{exp} -> {len(df)} data rows -> {runs} runs")
    print(f"Took about {data_gather_time:.2f} seconds to gather all these data.")

    return all_dfs


# Find the minimum aggregate metric (based on mean, median, etc.)
def get_best_hyperparams(args, perf):
    if args.agg == "mean":
        agg_perf = perf.groupby(args.ARG_COLS).mean()
    elif args.agg == "median":
        agg_perf = perf.groupby(args.ARG_COLS).median()
    # Get the aggregated perf that minimizes the chosen metric
    min_agg_perf = agg_perf[agg_perf[args.metric] == agg_perf.min()[args.metric]]
    return min_agg_perf.index


def find_all_best_hyperparams(args, all_dfs):
    best_dfs = {}
    best_dfs_fixed_args = defaultdict(dict)

    # Find set of possible values for each hp adaptively
    fixed_args = defaultdict(set)
    for (dataset, optimizer), df in all_dfs.items():
        if optimizer.startswith("Adam"):
            continue
        for arg_col in args.ARG_COLS:
            fixed_args[arg_col] = set(df[arg_col])
        break

    for experiment in product(args.DATASETS, args.OPTIMIZERS):
        print("Finding best hyperparams for", experiment)
        # Get last metrics/performance (supposed to be epoch-smoothed for better results)
        exp_df = all_dfs[experiment]
        max_ep = exp_df.groupby(args.ARG_COLS, sort=False)[args.idx].transform(max)
        perf = exp_df[exp_df[args.idx] == max_ep].drop(args.idx, axis=1)

        # Get the data associated with the args of the min aggregated metric
        exp_df = exp_df.set_index(args.ARG_COLS)
        best_dfs[experiment] = exp_df.loc[get_best_hyperparams(args, perf)]
        for arg_col, arg_set in fixed_args.items():
            best_dfs_fixed_args[arg_col][experiment] = {
                arg_value: exp_df.loc[get_best_hyperparams(args, perf[perf[arg_col] == arg_value])]
                for arg_value in arg_set
            }

    return best_dfs, best_dfs_fixed_args


def plot_best_perfs(args, best_dfs):
    plt.rc('legend', fontsize=LEGEND_FONTSIZE, loc=LEGEND_LOC)
    start_time = time.time()
    # Plot 3 rows each one showing some performance metric,
    # where the columns are the dataset on which the optim is run.
    num_rows = len(args.METRICS)
    fig, axes = plt.subplots(num_rows, len(args.DATASETS))
    fig.set_size_inches(ASPECT * HEIGHT * len(args.DATASETS), HEIGHT * num_rows)
    plt.suptitle(rf"Best performances on {args.loss} loss")
    for j, dataset in enumerate(args.DATASETS):
        for i, metric in enumerate(args.METRICS):
            ax = axes[i] if len(args.DATASETS) == 1 else axes[i,j]
            for optimizer in args.OPTIMIZERS:
                exp = (dataset, optimizer)
                if exp not in best_dfs:
                    continue
                # Get hyperparams of best performance of 'optimizer' on 'dataset'
                args_dict = {k:v for k,v in zip(best_dfs[exp].index.names, best_dfs[exp].index[0])}
                exp_df = best_dfs[exp].reset_index()
                # Show power of lr as 2^lr_pow
                lr_pow = round(log2(float(args_dict['lr'])))
                if optimizer == "Adam":
                    beta1 = 0.9 if 'beta1' not in args.filter_args else args.filter_args['beta1']
                    sublabel = rf"$\eta = 2^{{{lr_pow}}}$, $\beta_1={beta1}$, $\beta_2={0.999}$"  # XXX: hardcoded
                else:
                    sublabel = rf"$\eta = 2^{{{lr_pow}}}$, $\alpha={args_dict['alpha']}$, $\beta={args_dict['beta2']}$"
                label = rf"{optimizer}({sublabel})"
                print(f"Plotting lines for {exp}...")
                sns.lineplot(x=args.idx, y=metric, label=label, ax=ax, data=exp_df)
            if LOG_SCALE[metric]:
                ax.set(yscale="log")
            ax.set_title(dataset)
            ax.set_ylabel(rf"{TO_MATH[metric]}")
            ax.set_xlabel(rf"{TO_MATH[args.idx]}")
            ax.legend()
    fig.tight_layout()

    # Create a string out of filter args and save figure
    plt.savefig(f"{args.plots_dir}/perf({args.experiment_str}).pdf")
    plt.close()
    print(f"Took about {time.time() - start_time:.2f} seconds to create this plot.")


def plot_best_perfs_given_precond(args, best_dfs_fixed_args):
    plt.rc('legend', fontsize=LEGEND_FONTSIZE, loc=LEGEND_LOC)
    start_time = time.time()
    num_rows = len(args.METRICS)
    fig, axes = plt.subplots(num_rows, len(args.DATASETS))
    fig.set_size_inches(ASPECT * HEIGHT * len(args.DATASETS), HEIGHT * num_rows)
    plt.suptitle(rf"Best performances on {args.loss} loss")
    for j, dataset in enumerate(args.DATASETS):
        optim_dfs = []
        for optimizer in args.OPTIMIZERS:
            exp = (dataset, optimizer)
            if exp not in best_dfs_fixed_args["precond"]:
                continue
            exp_df = pd.concat(best_dfs_fixed_args["precond"][exp].values()).reset_index()
            if len(exp_df) == 0:
                continue
            exp_df["optimizer"] = optimizer
            optim_dfs.append(exp_df)
        # reset index and combine precond with gamma
        print(f"Plotting lines for {dataset}...")
        optim_df = pd.concat(optim_dfs).reset_index()
        # SGD, SARAH, L-SVRG, Adam, none is solid, hutchinson is dashed
        optim_df = optim_df.sort_values(["optimizer", "precond"], ascending=False)
        # optim_df = optim_df.sort_values("precond", ascending=False)
        for i, metric in enumerate(args.METRICS):
            ax = axes[i] if len(args.DATASETS) == 1 else axes[i,j]
            sns.lineplot(x=args.idx, y=metric, hue="optimizer", style="precond", ax=ax, data=optim_df)
            if LOG_SCALE[metric]:
                ax.set(yscale="log")
            ax.set_title(dataset)
            ax.set_ylabel(rf"{TO_MATH[metric]}")
            ax.set_xlabel(rf"{TO_MATH[args.idx]}")
            # axes[i,j].legend()
    fig.tight_layout()
    plt.savefig(f"{args.plots_dir}/perf_given_precond({args.experiment_str}).pdf")
    plt.close()
    print(f"Took about {time.time() - start_time:.2f} seconds to create this plot.")


def plot_best_perfs_given_fixed_arg(args, best_dfs_fixed_args):
    plt.rc('legend', fontsize=LEGEND_FONTSIZE, loc=LEGEND_LOC)
    modes = set(best_dfs_fixed_args.keys()) - set(["precond"])  # exclude precond
    for y in ("error", "gradnorm"):
        for mode in modes:
            valid_optimizers = set(args.OPTIMIZERS) - set(["Adam"])
            # Plot data for all optim, datasets, and args
            start_time = time.time()
            fig, axes = plt.subplots(len(valid_optimizers), len(args.DATASETS))
            fig.set_size_inches(ASPECT * HEIGHT_LARGE * len(args.DATASETS), HEIGHT_LARGE * len(valid_optimizers))
            title = rf"Best {TO_MATH[y]} given {TO_MATH[mode]} on {args.loss} loss"
            plt.suptitle(title)
            for i, optimizer in enumerate(valid_optimizers):
                for j, dataset in enumerate(args.DATASETS):
                    if len(args.DATASETS) == 1 or len(args.OPTIMIZERS) == 1:
                        ax = axes[i+j]
                    else:
                        ax = axes[i,j]
                    exp = (dataset, optimizer)
                    if exp not in best_dfs_fixed_args[mode] or len(best_dfs_fixed_args[mode][exp]) == 0:
                        continue
                    exp_df = pd.concat(best_dfs_fixed_args[mode][exp].values()).reset_index()
                    for mode in modes:
                        exp_df[mode] = exp_df[mode].astype(str)
                    exp_df["lr"] = exp_df["lr"].astype(float)

                    print(f"Plotting lines for {exp}...")
                    if mode == "lr":
                        exp_df = exp_df.sort_values("alpha", ascending=False)  # none is thinest
                        exp_df = exp_df.sort_values("beta2", ascending=False)  # none is solid, avg is dashed, etc.
                        sns.lineplot(ax=ax, x=args.idx, y=y,
                                    hue="lr", hue_norm=LogNorm(), palette="vlag",
                                    size="alpha", style="beta2", data=exp_df)

                    elif mode == "beta":
                        exp_df = exp_df.sort_values("alpha", ascending=True)  # none is blue, etc.
                        exp_df = exp_df.sort_values("beta2", ascending=True)  # nums first, to be consistent with Adam
                        sns.lineplot(ax=ax, x=args.idx, y=y,
                                    hue="beta2", size="lr", size_norm=LogNorm(), style="alpha", data=exp_df)
                        print(exp_df)

                    elif mode == "alpha":
                        exp_df = exp_df.sort_values("alpha", ascending=True)  # none is blue, etc.
                        exp_df = exp_df.sort_values("beta2", ascending=False)  # none is solid, avg is dashed, etc.
                        sns.lineplot(ax=ax, x=args.idx, y=y,
                                    hue="alpha", size="lr", size_norm=LogNorm(), style="beta2", data=exp_df)

                    ax.set(yscale="log")
                    ax.set_title(rf"$\tt {optimizer}({dataset})$")
                    ax.set_ylabel(rf"{TO_MATH[y]}")
                    ax.set_xlabel(rf"{TO_MATH[args.idx]}")
            fig.tight_layout()

            # Create a string out of filter args and save figure
            plt.savefig(f"{args.plots_dir}/{y}_given_{mode}({args.experiment_str}).pdf")
            plt.close()
            print(f"Took about {time.time() - start_time:.2f} seconds to create this plot.")


def generate_plots(args, precond=False, fixed_args=False):
    # Create experiment dataframe and find best hyperparams based on given metric
    all_dfs = create_experiments_dataframe(args)
    best_dfs, best_dfs_fixed_args = find_all_best_hyperparams(args, all_dfs)
    plot_best_perfs(args, best_dfs)
    if precond:
        plot_best_perfs_given_precond(args, best_dfs_fixed_args)
    if fixed_args:
        plot_best_perfs_given_fixed_arg(args, best_dfs_fixed_args)

