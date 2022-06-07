#!/bin/sh
#SBATCH --account=ie-idi
#SBATCH --partition=CPUQ
#SBATCH --time=00-00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --job-name="Durcheck"
#SBATCH --output="/cluster/home/martimoa/.nits/slurm/durcheck-%j.slurm.out"
#SBATCH --mail-user=martin.moan1@gmail.com
#SBATCH --mail-type=ALL
module purge
module load Anaconda3/2020.07

conda init --all
source ~/.bashrc

conda activate TDT4900
conda info --envs

python durcheck.py --n_processors 12
