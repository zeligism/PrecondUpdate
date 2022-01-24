# ScaledVR

In order to generate a job array and submit it to SLURM, do this:
```
bash ja_generate.sh [<JA_FILE>]
bash ja_submit.sh [<JA_FILE>] [<NRUNS>]
```
The first positional argument `<JA_FILE>` is the job array file name, which is optional (will default to `experiment.ja`).
The second positional argument `<NRUNS>` for `ja_submit.sh` is the number of runs per job in the array, which defaults to 1.
For example, if you have 300 jobs in your array, and you want to run 10 of them sequentially
(i.e. array of size 30 containing 10 sequential jobs each), just pass 10 as the second argument and the script will do the rest.
