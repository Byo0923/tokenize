import json
import os
import argparse
from tqdm import tqdm

def split_and_combine_files(input_dir, output_dir, dataset_name, max_file_size_gb=10):
    max_file_size_bytes = max_file_size_gb * 1024 ** 3
    os.makedirs(output_dir, exist_ok=True)

    file_counter = 1
    current_file_size = 0
    current_file = None
    current_file_path = os.path.join(output_dir, f'{dataset_name}_batch_{file_counter}.jsonl')

    # 全てのファイルを読み込んで結合
    file_list = sorted(os.listdir(input_dir))
    for filename in tqdm(file_list, desc="Processing files"):
        file_path = os.path.join(input_dir, filename)
        if os.path.isfile(file_path) and file_path.endswith('.jsonl'):
            print(f"Reading file: {file_path}")
            with open(file_path, 'r', encoding='utf-8') as file:
                for line in file:
                    # 新しいファイルを開始するか判断
                    if current_file is None or current_file_size >= max_file_size_bytes:
                        if current_file is not None:
                            current_file.close()
                        current_file_path = os.path.join(output_dir, f'{dataset_name}_batch_{file_counter}.jsonl')
                        current_file = open(current_file_path, 'w', encoding='utf-8')
                        print(f"Writing to new file: {current_file_path}")
                        file_counter += 1
                        current_file_size = 0

                    # ラインをファイルに書き込む
                    current_file.write(line)
                    current_file_size += len(line.encode('utf-8'))

    # 最後のファイルを閉じる
    if current_file is not None:
        current_file.close()

def main():
    parser = argparse.ArgumentParser(description="Combine and split JSONL files based on size.")
    parser.add_argument('--input_dir', type=str, required=True, help="Directory containing input JSONL files.")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save the output JSONL files.")
    parser.add_argument('--dataset_name', type=str, default="data", help="Name of Dataset(str)")
    parser.add_argument('--max_size_gb', type=int, default=10, help="Maximum size of each output file in GB.")
    args = parser.parse_args()

    split_and_combine_files(args.input_dir, args.output_dir, args.dataset_name, args.max_size_gb)

if __name__ == "__main__":
    main()
