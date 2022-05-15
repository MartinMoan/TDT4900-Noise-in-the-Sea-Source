#!/bin/sh
#SBATCH --account=ie-idi
#SBATCH --partition=GPUQ
#SBATCH --time=00-00:10:00
#SBATCH --nodes=2
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=2
#SBATCH --mem=42000
#SBATCH --job-name="DistributedTrainingTest"
#SBATCH --output="/cluster/home/martimoa/.nits/slurm/dist.slurm-%j.out"
#SBATCH --mail-user=martin.moan1@gmail.com
#SBATCH --mail-type=ALL

module purge
module load Anaconda3/2020.07

conda init --all
source ~/.bashrc
conda activate TDT4900
conda info --envs

# load cuda
module load NCCL/2.8.3-CUDA-11.1.1

python experiment.py -batch_size 16 -tracking_name "Experiment" -tracking_note "Testing FashionMNIST model to ensure distributed training on SLURM cluster works." --verbose -num_gpus 2 -num_nodes 2