#!/bin/bash

#SBATCH --job-name=pred
#SBATCH --output=logs/predict_nn.log
#SBATCH --partition gpu06
#SBATCH --nodes=1
#SBATCH --tasks-per-node=4
#SBATCH --time=06:00:00

## configs 
module purge
module load gcc/9.3.1 mkl/19.0.5
module load python/anaconda-3.10
source /share/apps/bin/conda-3.10.sh
conda deactivate
conda activate crop-disease

## run
python3 -u src/predict_nn.py --dir=20230723-154851_resnet18_bs56_acc0_757143 --ignore=4

