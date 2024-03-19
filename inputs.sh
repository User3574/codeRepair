#!/bin/sh

#SBATCH --job-name=generate_inputs    # job name
#SBATCH --partition=dgx2q
#SBATCH --gres=gpu:1
#SBATCH -t 48:00:00             # time (D-HH:MM)
#SBATCH -o inputs.%N.%j.out      # STDOUT
#SBATCH -e inputs.%N.%j.err      # STDERR

# Create env
source ~/miniconda3/etc/profile.d/conda.sh
conda activate env_codeRepair

cd /home/machacini/codeRepair

# Generate inputs
cd benchmarks
python generate_inputs.py codet5p humaneval
python generate_inputs.py codegen humaneval
python generate_inputs.py bloom humaneval
python generate_inputs.py starcoder humaneval
python generate_inputs.py deepseekcoder humaneval
python generate_inputs.py codellama humaneval
