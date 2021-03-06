#!/bin/bash -l
#SBATCH --account=ie-idi
#SBATCH --partition=GPUQ
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=42000
#SBATCH --job-name="AST Pre-Training Sanity Check"
#SBATCH --time=00-00:10:00
#SBATCH --output="/cluster/home/martimoa/.nits/slurm/sanity.slurm-%j.out"

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
FSTRIDE=16
TSTRIDE=16

MODEL_SIZE="base384"

LEARNING_RATE=0.00001
WEIGHT_DECAY=5e-7
BETAS="0.95 0.999"

BATCH_SIZE=8
MAX_EPOCHS=1000

srun python ../train.py -batch_size $BATCH_SIZE -epochs $MAX_EPOCHS -learning_rate $LEARNING_RATE -weight_decay $WEIGHT_DECAY -betas $BETAS -nmels $NMELS -hop_length $HOPLENGTH -nfft $NFFT -fstride $FSTRIDE -tstride $TSTRIDE -model_size $MODEL_SIZE -clip_duration_seconds $CLIP_DURATION -clip_overlap_seconds $CLIP_OVERLAP --imagenet_pretrain --no-audioset_pretrain --tracking_tags "SanityCheck" --overfit_batches 0.01
