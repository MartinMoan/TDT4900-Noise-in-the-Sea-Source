#!/bin/sh
#SBATCH --account=ie-idi
#SBATCH --partition=GPUQ
#SBATCH --time=00-16:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --mem=42000
#SBATCH --job-name="AST Training"
#SBATCH --output="/cluster/home/martimoa/.nits/logs/ast.slurm.out"
#SBATCH --mail-user=martin.moan1@gmail.com
#SBATCH --mail-type=ALL
module purge
module load Anaconda3/2020.07

conda init --all
source ~/.bashrc

conda activate TDT4900
conda info --envs

ENV=prod VIRTUAL_DATASET_LOADING=False python ast_trainer.py --learning-rate 0.0001 --weight-decay 0.00001 --epochs 3 --batch-size 16 --num-workers 32 --force-gpu

echo "ast.slurm job done!"
