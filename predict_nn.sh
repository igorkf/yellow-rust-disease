#!/bin/bash

#SBATCH --job-name=pred
#SBATCH --output=logs/predict_nn.log
#SBATCH --partition gpu06
#SBATCH --nodes=1
#SBATCH --tasks-per-node=8
#SBATCH --time=06:00:00

## configs 
module purge
module load gcc/9.3.1 mkl/19.0.5
module load python/anaconda-3.10
source /share/apps/bin/conda-3.10.sh
conda deactivate
conda activate crop-disease

## run
python3 -u src/predict_nn.py --dir=20230801-105502_resnet18_bs56_b00_b1125_s64_acc0_767857
echo " "
python3 -u src/predict_nn.py --dir=20230801-112157_resnet18_bs58_b00_b1125_s64_acc0_778161
