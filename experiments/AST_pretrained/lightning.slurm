#!/bin/sh
#SBATCH --account=ie-idi
#SBATCH --partition=GPUQ
#SBATCH --nodes=4
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=42000
#SBATCH --job-name="AST Training"
#SBATCH --output="/cluster/home/martimoa/.nits/slurm/ast.slurm-%j.out"
#SBATCH --mail-user=martin.moan1@gmail.com
#SBATCH --mail-type=ALL

module purge
module load Anaconda3/2020.07

conda init --all
source ~/.bashrc
conda activate TDT4900
conda info --envs

python ast_lightning.py -batch_size 16 -epochs 3 -learning_rate 0.0001 -weight_decay 5e-7 -betas 0.95 0.999 -kfolds 5 -nmels 128 -hop_length 512 -nfft 2046 -fstride 10 -tstride 10 -model_size base384 -clip_duration_seconds 10.0 -clip_overlap_seconds 4.0 -tracking_name "AST ImageNet Pretrained" -tracking_note "AST pretrained on ImageNet (but not AudioSet, to enable differing input shapes)" -tracking_tags "AST" "ImageNet" "No-AudioSet" --imagenet_pretrain --no-audioset_pretrain --verbose -num_gpus 8 -num_nodes 4
