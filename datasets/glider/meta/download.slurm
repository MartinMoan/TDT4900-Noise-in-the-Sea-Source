#!/bin/sh
#SBATCH --account=ie-idi
#SBATCH --partition=CPUQ
#SBATCH --time=00-24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=42000
#SBATCH --job-name="Download GLIDER dataset missing files"
#SBATCH --output="download.slurm.out"
#SBATCH --mail-user=martin.moan1@gmail.com
#SBATCH --mail-type=ALL
module purge
module load Anaconda3/2020.07

conda init --all
source ~/.bashrc

conda activate TDT4900
conda info --envs

VIRTUAL_DATASET_LOADING=False python ./missing_files/missing_files.py
python ./missing_files/verify_files.py
VIRTUAL_DATASET_LOADING=False python ./missing_files/missing_files.py
python labels_without_files.py
python ./missing_files/duplicates.py
echo "slurmjob complete"
