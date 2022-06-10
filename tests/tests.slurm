#!/bin/bash -l
#SBATCH --account=ie-idi
#SBATCH --partition=CPUQ
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=42000
#SBATCH --job-name="Tests"
#SBATCH --time=00-01:00:00
#SBATCH --output="/cluster/home/martimoa/.nits/slurm/tests.slurm-%j.out"
#SBATCH --chdir="/cluster/home/martimoa/TDT4900-Noise-in-the-Sea-Source/"

module purge
module load Anaconda3/2020.07

conda init --all
source ~/.bashrc
conda activate TDT4900
conda info --envs

# debugging flags
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

# load cuda
module load NCCL/2.8.3-CUDA-11.1.1

python -m unittest discover -v -t ./ -s ./tests -p "test_*.py" 
