#!/bin/bash -l
#SBATCH --account=ie-idi
#SBATCH --partition=GPUQ
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=42000
#SBATCH --job-name="AST Prediction"
#SBATCH --time=00-00:10:00
#SBATCH --output="/cluster/home/martimoa/.nits/slurm/predict.slurm-%j.out"

module purge
module load Anaconda3/2020.07

conda init --all
source ~/.bashrc
conda activate TDT4900
conda info --envs

# load cuda
module load NCCL/2.8.3-CUDA-11.1.1

CLIP_DURATION=10.0
CLIP_OVERLAP=4.0
NMELS=128
HOPLENGTH=1280
NFFT=3200

BATCH_SIZE=8

CHECKPOINT=/cluster/home/martimoa/.nits/lightning/checkpoints/ast/noise-in-the-sea/17zzcovp/checkpoints/epoch=2-step=29.ckpt

srun python astpredict.py $CHECKPOINT -batch_size $BATCH_SIZE -nmels $NMELS -nfft $NFFT -hop_length $HOPLENGTH -clip_duration_seconds $CLIP_DURATION -clip_overlap_seconds $CLIP_OVERLAP
