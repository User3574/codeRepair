#!/bin/sh

#SBATCH --job-name=val_quix    # job name
#SBATCH --partition=dgx2q      # partition (queue)
#SBATCH --gres=gpu:1
#SBATCH -t 124:00:00             # time (D-HH:MM)
#SBATCH -o val_quix.%N.%j.out      # STDOUT
#SBATCH -e val_quix.%N.%j.err      # STDERR

# Create env
source ~/miniconda3/etc/profile.d/conda.sh
conda activate env_codeRepair

cd /home/machacini/codeRepair

# Validate
cd benchmarks
python validate.py codegen quixbugs False
python validate.py codellama quixbugs False
python validate.py bloom quixbugs False
python validate.py deepseekcoder quixbugs False
python validate.py starcoder quixbugs False
python validate.py codet5p quixbugs False
