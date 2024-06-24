#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --gpus-per-node=0

source ~/miniconda3/etc/profile.d/conda.sh

conda activate textprocess

# tokenize
output_prefix=$(yq -r '.output_prefix' ./config_data.yaml)
input_jsonl=$(yq -r '.input' ./config_data.yaml)
input_tokenizer_file=$(yq -r '.input_tokenizer_file' ./config_data.yaml)
echo "tokenizer-model: ${input_tokenizer_file}"

python ./preprocess_data.py \
    --tokenizer-type SentencePieceTokenizer \
    --tokenizer-model ${input_tokenizer_file} \
    --input  ${input_jsonl} \
    --output-prefix ${output_prefix} \
    --dataset-impl mmap \
    --workers 64 
echo ""

# Conda 環境をデアクティベート
conda deactivate

