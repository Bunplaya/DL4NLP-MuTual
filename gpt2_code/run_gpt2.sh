#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --partition=gpu
#SBATCH --time=10:00:00
 
module purge
module load 2022
module load Python/3.10.4-GCCcore-11.3.0
module load CUDA/11.7.0

export PATH=$HOME/.local/bin:$PATH

python3 main.py --batch_size 2 --epochs 1 --train_dir Data/mutual/train --val_dir Data/mutual/dev --save_dir Finetuned/gpt2/freeze --model gpt2