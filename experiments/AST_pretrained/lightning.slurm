#!/bin/bash -l
#SBATCH --account=ie-idi
#SBATCH --partition=GPUQ
#SBATCH --nodes=4
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=1
#SBATCH --mem=42000
#SBATCH --job-name="AST Training"
#SBATCH --time=00-02:00:00
#SBATCH --output="/cluster/home/martimoa/.nits/slurm/ast.slurm-%j.out"

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

python ast_lightning.py -batch_size 16 -epochs 3 -learning_rate 0.0001 -weight_decay 5e-7 -betas 0.95 0.999 -kfolds 5 -nmels 128 -hop_length 1280 -nfft 3200 -fstride 10 -tstride 10 -model_size "base384" -clip_duration_seconds 10.0 -clip_overlap_seconds 4.0 -tracking_name "AST ImageNet Pretrained" -tracking_note "AST pretrained on ImageNet (but not AudioSet, to enable differing input shapes)" -tracking_tags "AST" "ImageNet" "No-AudioSet" --imagenet_pretrain --no-audioset_pretrain --verbose -num_gpus 2 -num_nodes 4
