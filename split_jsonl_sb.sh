#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --gpus-per-node=0


source ~/miniconda3/etc/profile.d/conda.sh

conda activate textprocess



python ./split_jsonl.py \
    --input "/storage5/text_corpus/phase1_japanese/backupbunkatsu/phase1_corpus.jsonl"  \
    --output "/storage5/shared/k_nishizawa/tmp_p1_token/" \
    --split_size 20 \
    --append-eod
echo ""
