#!/bin/bash

#SBATCH --job-name=crop-nn
#SBATCH --output=logs/train_nn%j.log
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
for bs in 24 40 48 56 64 72
do
    echo "---------------------------------"
    python3 -u src/train_nn.py --bs=${bs}
    echo "bs=${bs}"
    echo " "
done
