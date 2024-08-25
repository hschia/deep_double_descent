#!/bin/bash

#SBATCH --job-name=testing

#SBATCH --gres=gpu:1
#SBATCH --output=run_model.out
#SBATCH --nodes=1
#SBATCH --time=1:00:00 --gpus=1 

#SBATCH --mail-type=FAIL,END 
#SBATCH --mail-user=hschia@ias.edu

nvidia-smi

python run_model.py $1
