#!/usr/bin/env python3
import os
from pprint import pprint

slurm_variables = {key: value for key, value in os.environ.items() if "SLURM" in key}
check_these = ["SLURM_NODEID", "SLURM_TASK_PID", "SLURM_PROCID", "SLURM_JOB_GID", "SLURM_JOBID", "SLURM_LOCALID"]
to_check = {key: os.environ.get(key) for key in check_these}

print(to_check)
print()
print(slurm_variables)
print()
print()
print()


