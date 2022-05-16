import os
import torch

procid = os.environ.get("SLURM_PROCID")
print(f"SLURM_PROCID: {procid}")
print(f"Number of gpus available: {torch.cuda.device_count()}")
