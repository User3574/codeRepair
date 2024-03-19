#!/bin/sh

#SBATCH --job-name=starcoder    # job name
#SBATCH --partition=hgx2q      # partition (queue)
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCh -n 1
#SBATCH -c 6
#SBATCH -t 256:00:00             # time (D-HH:MM)
#SBATCH -o starcoder.%N.%j.out      # STDOUT
#SBATCH -e starcoder.%N.%j.err      # STDERR

module purge
module load slurm/21.08.8

# Create env
source ~/miniconda3/etc/profile.d/conda.sh
conda activate env_codeRepair

cd /home/machacini/codeRepair

# Train
python finetune.py --model_name=starcoder --checkpoint=bigcode/starcoderbase-7b --batch_size_train=1 --batch_size_test=16 --max_length=512 --max_new_tokens=512
#python finetune.py --model_name=codet5p --checkpoint=Salesforce/codet5-base --max_length=512 --max_new_tokens=512
#python finetune.py --model_name=codet5p --checkpoint=Salesforce/codet5-large --max_length=512 --max_new_tokens=512

