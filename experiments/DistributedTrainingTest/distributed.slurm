#!/bin/sh
#SBATCH --account=ie-idi
#SBATCH --partition=GPUQ
#SBATCH --time=00-00:10:00
#SBATCH --nodes=2
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=2
#SBATCH --mem=42000
#SBATCH --job-name="SrunInSbatchScriptTest"
#SBATCH --output="sruntest.slurm-%j.out"

module purge
module load Anaconda3/2020.07

conda init --all
source ~/.bashrc
conda activate TDT4900
conda info --envs

# load cuda
module load NCCL/2.8.3-CUDA-11.1.1

echo "Without srun"
python test.py

echo "With srun"
srun python test.py
echo "Done!"