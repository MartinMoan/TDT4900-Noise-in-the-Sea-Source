#!/bin/sh
#SBATCH --account=ie-idi
#SBATCH --partition=GPUQ
#SBATCH --time=00-24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
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

ENV=prod VIRTUAL_DATASET_LOADING=False python ast_trainer.py -batch_size 16 -epochs 3 -learning_rate 0.00001 -weight_decay 5e-7 -betas 0.95 0.999 -kfolds 5 -n_mels 256 -hop_length 512 -n_fft 8192 -fstride 10 -tstride 10 -model_size base384 -clip_duration_seconds 30.0 -clip_overlap_seconds 10.0 -tracking_name "AST ImageNet Pretrained" -tracking_note "AST pretrained on ImageNet (but not AudioSet, to enable differing input shapes)" -tracking_tags "AST" "ImageNet" "No-AudioSet" -verification_dataset_limit 42 -proper_dataset_limit 0.7 --imagenet_pretrain --no-audioset_pretrain --verbose --verification_run

echo "ast.slurm job done!"
