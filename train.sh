#!/bin/sh

#SBATCH --job-name=diff_test    # job name
#SBATCH --partition=mi210q       # partition (queue)
#SBATCH --gres=gpu:1
#SBATCH -t 24:00:00             # time (D-HH:MM)
#SBATCH -o slurm.%N.%j.out      # STDOUT
#SBATCH -e slurm.%N.%j.err      # STDERR

# Create env
source ~/miniconda3/etc/profile.d/conda.sh
conda activate env_codeRepair

cd /home/machacini/codeRepair

# Train
python train.py --epochs=0 --checkpoint=huggingface/CodeBERTa-small-v1 --model_name=CodeBERT

