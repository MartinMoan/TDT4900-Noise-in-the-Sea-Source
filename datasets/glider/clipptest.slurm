#!/bin/sh
#SBATCH --account=ie-idi
#SBATCH --partition=CPUQ
#SBATCH --time=00-4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --job-name="Clipping Test"
#SBATCH --output="/cluster/home/martimoa/.nits/logs/clipping.slurm.out"
#SBATCH --mail-user=martin.moan1@gmail.com
#SBATCH --mail-type=ALL
module purge
module load Anaconda3/2020.07

conda init --all
source ~/.bashrc

conda activate TDT4900
conda info --envs

ENV=prod VIRTUAL_DATASET_LOADING=False python clipping.py

echo "ast.slurm job done!"
