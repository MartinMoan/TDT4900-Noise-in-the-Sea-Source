#!/bin/bash -l
#SBATCH --account=ie-idi
#SBATCH --partition=GPUQ
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=1
#SBATCH --mem=42000
#SBATCH --job-name="AST Training"
#SBATCH --time=00-48:00:00
#SBATCH --output="/cluster/home/martimoa/.nits/slurm/ast.slurm.out"

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

CLIP_DURATION=10.0
CLIP_OVERLAP=4.0
NMELS=128
HOPLENGTH=1280
NFFT=3200

# initialize the dataset (on a single compute node)
# python initdata.py -nmels 128 -hop_length 1280 -nfft 3200 -clip_duration_seconds 10.0 -clip_overlap_seconds 4.0
python initdata.py -nmels $NMELS -hop_length $HOPLENGTH -nfft $NFFT -clip_duration_seconds $CLIP_DURATION -clip_overlap_seconds $CLIP_OVERLAP

srun python ast_lightning.py -batch_size 8 -epochs 3 -learning_rate 0.0001 -weight_decay 5e-7 -betas 0.95 0.999 -kfolds 5 -nmels $NMELS -hop_length $HOPLENGTH -nfft $NFFT -fstride 16 -tstride 16 -model_size "tiny224" -clip_duration_seconds $CLIP_DURATION -clip_overlap_seconds $CLIP_OVERLAP -tracking_tags "AST" "ImageNet" "No-AudioSet" "Unfrozen" --imagenet_pretrain --no-audioset_pretrain --verbose -num_gpus 2 -num_nodes 1
