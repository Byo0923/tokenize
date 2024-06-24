#!/bin/bash

cx



# tokenize
output_prefix=$(yq -r '.output_prefix' ./config_dir.yaml)
input_jsonl=$(yq -r '.input' ./config_dir.yaml)
input_tokenizer_file=$(yq -r '.input_tokenizer_file' ./config_dir.yaml)
worker=$(yq -r '.workers' ./config_dir.yaml)

echo "tokenizer-model: ${input_tokenizer_file}"

python ./preprocess_dir.py \
    --tokenizer-type SentencePieceTokenizer \
    --tokenizer-model ${input_tokenizer_file} \
    --input  ${input_jsonl} \
    --output-prefix ${output_prefix} \
    --dataset-impl mmap \
    --workers  ${worker}  \
    --append-eod
echo ""
