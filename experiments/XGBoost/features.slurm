#!/bin/bash -l
#SBATCH --account=ie-idi
#SBATCH --partition=CPUQ
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=42000
#SBATCH --job-name="XGBoost Feature Extraction"
#SBATCH --time=00-10:00:00
#SBATCH --output="/cluster/home/martimoa/.nits/slurm/xgboost_features.slurm-%j.out"

module purge
module load Anaconda3/2020.07

conda init --all
source ~/.bashrc
conda activate TDT4900
conda info --envs

srun python features.py