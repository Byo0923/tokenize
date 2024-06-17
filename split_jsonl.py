
import argparse
import json
import os
from tqdm import tqdm
import pytz

from datetime import datetime
# 日本のタイムゾーンを設定
japan_timezone = pytz.timezone('Asia/Tokyo')

import time
import pytz
import pandas as pd


def get_args():
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group(title='input data')
    group.add_argument('--input', type=str, required=True,
                       help='Path to input JSON')
    group.add_argument('--output', type=str, required=True,
                       help='Path to output JSON')
    group.add_argument('--split_size', type=int, default=20,
                       help='Split file size GB.')

    group.add_argument('--append-eod', action='store_true',
                       help='Append an <eod> token to the end of a document.')
    
    args = parser.parse_args()

    return args


def main():
    # 現在の日本時間を取得
    start_japan = datetime.now(japan_timezone)    
    # 日時を秒単位までのフォーマットで表示
    formatted_time = start_japan.strftime("%Y-%m-%d %H:%M:%S")
    args = get_args()

    print("Start time: ", formatted_time)
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    
    current_size = 0
    current_part = 1
    output_file = None
    output_stream = None
    max_size= args.split_size *1024**3  # max_sizeは20GB
    with open(args.input, 'r', encoding='utf-8') as file:
        for line in tqdm(file):
            if output_stream is None:
                output_file = os.path.join(args.output, f'part-{current_part}.jsonl')
                output_stream = open(output_file, 'w', encoding='utf-8')
            
            # ファイルサイズを確認して、必要に応じて新しいファイルに切り替える
            line_size = len(line.encode('utf-8'))
            if current_size + line_size > max_size:
                output_stream.close()
                current_part += 1
                current_size = 0
                output_file = os.path.join(args.output, f'part-{current_part}.jsonl')
                output_stream = open(output_file, 'w', encoding='utf-8')
            
            output_stream.write(line)
            current_size += line_size
    
    if output_stream:
        output_stream.close()

if __name__ == '__main__':
    main()
