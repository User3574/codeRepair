#!/bin/sh

#SBATCH --job-name=generate_outputs    # job name
#SBATCH --partition=hgx2q
#SBATCH -t 64:00:00             # time (D-HH:MM)
#SBATCH --gres=gpu:1
#SBATCH -o outputs.%N.%j.out      # STDOUT
#SBATCH -e outputs.%N.%j.err      # STDERR

# Create env
source ~/miniconda3/etc/profile.d/conda.sh
conda activate env_codeRepair

cd /home/machacini/codeRepair

# Generate inputs
cd benchmarks
python generate_outputs.py codegen humaneval
python generate_outputs.py codet5p humaneval
python generate_outputs.py codellama humaneval
python generate_outputs.py starcoder humaneval
python generate_outputs.py deepseekcoder humaneval
python generate_outputs.py bloom humaneval
