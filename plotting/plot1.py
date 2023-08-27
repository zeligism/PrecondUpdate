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

from plotting import generate_plots

# Data logs root directory and plot directory
LOG_DIR = "../logs/logs1"
PLOT_DIR = "plots1"

class Args:
    # Loss function: either "logistic" regression, or nonlinear least squares ('nllsq')
    LOSSES = ["logistic", "nllsq"]
    # The following should be the same as the one used in run_experiment.py
    DATASETS = ["w8a", "rcv1", "real-sim"]
    OPTIMIZERS = ["SGD", "SARAH", "L-SVRG", "Adam"]
    MAX_IDX = {"ep": 100, "time": 60}
    # These are the metrics collected in the data logs
    METRICS = ["loss", "gradnorm", "error"]
    # These are aggregators for comparing multi-seed runs
    AGGS = ["mean", "median"]
    # These are the logs columns: effective passes + metrics + walltime
    LOG_COLS = ["ep", "loss", "gradnorm", "error", "time"]
    TIME_INDICES = ["ep", "time"]  # indices that can be counted as time indices
    DATA_INDICES = [0, 1, 2, 3, 6]  # indices corresponding to chosen cols in logs
    # These are the hyperparameters of interest
    ARG_COLS = ["lr", "alpha", "beta2", "precond"]
    # List of available filter args per col
    # Logs will be filtered for these settings when applicable (USE EXACT STRING VALUE AS IN FILENAME).
    FILTER_LIST = {
        "corrupt": ["none", "(0,3)", "(-3,0)", "(-3,3)"],
        # "corrupt": ["none"],
        # "beta1": ["0.0", "0.9"],
        "beta1": ["0.9"],
    }
    # Ignore all runs containing 'any' of these hyperparams.
    IGNORE_ARGS = {
        "alpha": [1e-11],
        "weight_decay": [0.1],
    }

    def __init__(self, log_dir, plot_dir,
                 idx = "ep",
                 loss = "logistic",
                 metric = "error",
                 agg = "mean",
                 avg_downsample = 5,
                 filter_args = {},
                 remove_empty_file = False,
                 ) -> None:
        self.base_dir = log_dir
        self.plots_dir = plot_dir
        # Choose loss, metric, and aggregation method
        self.idx = idx
        self.loss = loss
        self.metric = metric
        self.agg = agg
        self.avg_downsample = avg_downsample  # downsample by averaging
        self.filter_args = filter_args  # see above
        self.remove_empty_file = remove_empty_file  # force remove log files that are empty
        # Update and make dirs
        self.log_dir = os.path.join(self.base_dir, self.loss)
        os.makedirs(self.plots_dir, exist_ok=True)
        # Experiment identifier
        self.as_dict = dict(idx=self.idx, loss=self.loss, metric=self.metric, **self.filter_args)
        self.experiment_str = f"_".join(f"{k}_{v}" for k,v in self.as_dict.items())
        self.experiment_repr = f", ".join(f"{k}={v}" for k,v in self.as_dict.items())

    def __repr__(self) -> str:
        return f"Args({self.experiment_repr})"


def main():
    for idx, loss, metric, *filter_values \
            in product(Args.TIME_INDICES, Args.LOSSES, Args.METRICS, *Args.FILTER_LIST.values()):
        filter_args = dict(zip(Args.FILTER_LIST.keys(), filter_values))
        kwargs = dict(log_dir=LOG_DIR, plot_dir=PLOT_DIR,
                      idx=idx, loss=loss, metric=metric, filter_args=filter_args)
        # Assign arguments
        args = Args(**kwargs)
        print(f"Generating plots with args: {args}")
        generate_plots(args, precond=True, fixed_args=False)



if __name__ == "__main__":
    main()
