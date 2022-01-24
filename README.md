# ScaledVR

In order to generate a job array and submit it to SLURM, do this:
```
bash ja_generate.sh [<JA_FILE>]
bash ja_submit.sh [<JA_FILE>]
```
The name is job array file name is optional (will default to `experiment.ja`).
