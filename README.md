# ScaledVR

This repository contains an implementation of a preconditioned gradient descent update. The preconditioner is the hessian diagonal estimated using Hutchinson's trace method. We show that this clearly improves on the non-preconditioned methods and competes with SOTA optimzation algortihms with small computation overhead.

# How to run
First, download the datasets using `sh download_datasets.sh`. Then, you can run the training using `python src/train.py`. For running a whole array of experiments (or reproduce our experimental results), you can run `python src/run_experiment.py`. For running the training in pytorch (for using autograd to calculate the preconditioner), run `python src/pytorch/train.py`, or `python src/pytorch/run_experiment.py` for the experiments.

# Reproducibility
Our results are completely reproducible. We generate the data logs using the `run_experiment.py` scripts and hardcode the hyperparameter arrays in the script (works for now). Then, we make the plots using the plot notebooks in the root directory, namely `plot1.ipynb`, `plot2.ipynb`, and `plot_torch.ipynb`. Each of these notebooks generate different plots, some of which were used in the paper. Sometimes, the notebooks have to be run multiple times to account for all the combinations of hyperparameters and experiment settings.

If the user wants to use readily available data logs or wants to analyze the data logs used in our plots, then they can download the data logs from the links below.
| Log file   | Plotting script   | Link |
| - | - | - |
| logs1      | plot1.ipynb       | [logs1.zip][link1] |
| logs2      | plot2.ipynb       | [logs2.zip][link2] |
| logs_torch | plot_torch.ipynb  | [logs_torch.zip][link_torch] |

# Dependencies
We still do not include the list of dependencies. In any case, they are simple and minimal (e.g. they are included in anaconda package, except pytorch). We plan to include them soon.


[link1]: https://mbzuaiac-my.sharepoint.com/:u:/g/personal/abdulla_almansoori_mbzuai_ac_ae/EexJ9vFoalxOj2beIKbuRjcBQH9oEPDfBFmCDKTSgJZEQQ?e=0mRzQF
[link2]: https://mbzuaiac-my.sharepoint.com/:u:/g/personal/abdulla_almansoori_mbzuai_ac_ae/EfoULDS7xbhPp03-6O_WiBIBsLJ1E7CbNmGqkdJEGnIEcg?e=CYOnjT
[link_torch]: https://mbzuaiac-my.sharepoint.com/:u:/g/personal/abdulla_almansoori_mbzuai_ac_ae/EeLt0JuliFtLscjnzOycaeIBtTo-SvVVRxQyNNRlcFjdaA?e=iKMbNR
