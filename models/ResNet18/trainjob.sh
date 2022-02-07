#!/bin/sh
#SBATCH --account=ie-idi
#SBATCH --partition=GPUQ
#SBATCH --time=00-08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:P100:2
#SBATCH --mem=42000
#SBATCH --job-name="ResNet18 Training"
#SBATCH --output="resnet18_trainer.out"
#SBATCH --mail-user=martin.moan1@gmail.com
#SBATCH --mail-type=ALL
module purge
module load Anaconda3/2020.07

conda init --all
source ~/.bashrc

conda activate TDT4501
conda info --envs

python resnet18_trainer.py --learning-rate 0.001 --weight-decay 0.00001 --epochs 3 --batch-size 16 --num-workers 16 --prediction-threshold 0.5 --force-gpu --verbose
