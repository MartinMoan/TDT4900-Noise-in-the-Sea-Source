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

BATCH_SIZE=16
EPOCHS=3
LEARNING_RATE=0.00001
KFOLDS=5

SR=128000
CLIP_DURATION=10.0
CLIP_OVERLAP=4.0

NMELS=256
NFFT=8192
HOP_LENGTH=512

ENV=prod VIRTUAL_DATASET_LOADING=False python ast_trainer.py -batch_size $BATCH_SIZE -epochs $EPOCHS -learning_rate $LEARNING_RATE -weight_decay 5e-7 -betas 0.95 0.999 -kfolds $KFOLDS -n_mels $NMELS -hop_length $HOP_LENGTH -n_fft $NFFT -fstride 10 -tstride 10 -model_size base384 -clip_duration_seconds $CLIP_DURATION -clip_overlap_seconds $CLIP_OVERLAP -tracking_name "AST ImageNet Pretrained" -tracking_note "AST pretrained on ImageNet (but not AudioSet, to enable differing input shapes)" -tracking_tags "AST" "ImageNet" "No-AudioSet" -verification_dataset_limit 42 -proper_dataset_limit 0.7 --imagenet_pretrain --no-audioset_pretrain --verbose --verification_run

echo "ast.slurm job done!"
