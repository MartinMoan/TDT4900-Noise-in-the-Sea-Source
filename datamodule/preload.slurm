#!/bin/bash -l
#SBATCH --account=ie-idi
#SBATCH --partition=CPUQ
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=42000
#SBATCH --job-name="Init data"
#SBATCH --time=00-01:00:00
#SBATCH --output="/cluster/home/martimoa/.nits/slurm/init.slurm-%j.out"

module purge
module load Anaconda3/2020.07

conda init --all
source ~/.bashrc
conda activate TDT4900
conda info --envs

python datamodule.py find /cluster/work/martimoa/hdd_copy/GLIDER\ phase\ I\ deployment/ --output ./metadata/metadata.csv --processes 24