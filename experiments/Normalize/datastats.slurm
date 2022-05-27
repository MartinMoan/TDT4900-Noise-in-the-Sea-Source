#!/bin/bash -l
#SBATCH --account=ie-idi
#SBATCH --partition=CPUQ
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=42000
#SBATCH --job-name="Datastats"
#SBATCH --time=00-10:00:00
#SBATCH --output="/cluster/home/martimoa/.nits/slurm/datastats.slurm-%j.out"

module purge
module load Anaconda3/2020.07

conda init --all
source ~/.bashrc
conda activate TDT4900
conda info --envs

python datastats.py ./datastats.json -nfft 3200 -nmels 128 -hop_length 1280 -clip_overlap_seconds 4.0 -clip_duration_seconds 10.0