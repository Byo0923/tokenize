#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --gpus-per-node=0


source ~/miniconda3/etc/profile.d/conda.sh

conda activate textprocess

python ./concatenation_jsonl.py 
echo ""

# Conda 環境をデアクティベート
conda deactivate

