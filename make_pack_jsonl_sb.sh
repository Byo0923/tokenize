#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --gpus-per-node=0

source ~/miniconda3/etc/profile.d/conda.sh
conda activate textprocess

# tokenize
# output_prefix=$(yq -r '.output_prefix' ./config_dir.yaml)
# input_jsonl=$(yq -r '.input' ./config_dir.yaml)
# input_tokenizer_file=$(yq -r '.input_tokenizer_file' ./config_dir.yaml)
# echo "tokenizer-model: ${input_tokenizer_file}"

python ./make_pack_jsonl.py \
    --tokenizer-type SentencePieceTokenizer \
    --tokenizer-model /storage5/shared/corpus/phase1_tokenizer_data/tokernizer/tokenizer_scale200.model \
    --input  /storage5/shared/corpus/moe_test/jsonl \
    --output-prefix /storage5/shared/corpus/moe_test/best-fit-packing/  \
    --dataset-impl mmap \


# Conda 環境をデアクティベート
conda deactivate
