#!/bin/bash

#SBATCH --job-name=crop-nn
#SBATCH --output=logs/train_nn%j.log
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
for bs in 50 52 54 56 58 60
do
    echo "---------------------------------"
    echo "bs=${bs}"
    python3 -u src/train_nn.py --m=resnet18 --bs=${bs}
    echo " "
done

