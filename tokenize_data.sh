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
    --workers 64 \
    --append-eod
echo ""