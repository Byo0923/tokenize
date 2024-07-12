# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Processing large data for pretraining."""
import argparse
import math
import json
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir)))
import time
from datetime import datetime
import pytz
import pandas as pd

# 日本のタイムゾーンを設定
japan_timezone = pytz.timezone('Asia/Tokyo')

import gzip
import glob
import torch
import numpy as np
from tqdm import tqdm

import multiprocessing
try:
    import nltk
    nltk_available = True
except ImportError:
    nltk_available = False

import concurrent.futures

from megatron.tokenizer import build_tokenizer
from megatron.data import indexed_dataset


# https://stackoverflow.com/questions/33139531/preserve-empty-lines-with-nltks-punkt-tokenizer
class CustomLanguageVars(nltk.tokenize.punkt.PunktLanguageVars):

    _period_context_fmt = r"""
        \S*                          # some word material
        %(SentEndChars)s             # a potential sentence ending
        \s*                       #  <-- THIS is what I changed
        (?=(?P<after_tok>
            %(NonWord)s              # either other punctuation
            |
            (?P<next_tok>\S+)     #  <-- Normally you would have \s+ here
        ))"""

class IdentitySplitter(object):
    def tokenize(self, *text):
        return text


class Encoder(object):
    tokenizer = None 
    def __init__(self, args):
        self.args = args

    def initializer(self):
        # Use Encoder class as a container for global data
        Encoder.tokenizer = build_tokenizer(self.args)
        Encoder.splitter = IdentitySplitter()

    def split(self, json_line):
        data = json.loads(json_line)
        output = {}
        for key in self.args.json_keys:
            text = data[key]
            max_len = 1000000
            tokens_list = [Encoder.splitter.tokenize(text[i:i+max_len]) for i in range(0, len(text), max_len)]
            output[key] = [tokens for partial in tokens_list for tokens in partial]
        return json.dumps(output), len(json_line)

    def encode(self, json_line):
        ids = {}
        lens = {}
        try:
            data = json.loads(json_line)
        except json.decoder.JSONDecodeError as e:
            print(f"デコードエラー: {e} ") 
            json_line = ""
            text = ""
            return ids, lens, len(json_line), text
        for key in self.args.json_keys:
            if key not in data:
                print(f"Warning: Key '{key}' not found in data.")
                text = ""
                continue  # キーがない場合、このイテレーションをスキップ
            text = data[key]
            # print( "text" , text[0],":" , text[1],":" , text[-2],":" , text[-1] )
            if isinstance(text, list):
                sentences = text
            else:
                sentences = [text]
            doc_ids = []
            sentence_lens = []
            for sentence in sentences:
                sentence_ids = Encoder.tokenizer.tokenize(sentence)
                if len(sentence_ids) > 0:
                    doc_ids.extend(sentence_ids)
                    sentence_lens.append(len(sentence_ids))
            if len(doc_ids) > 0 and self.args.append_eod:
                doc_ids.append(Encoder.tokenizer.eod)
            ids[key] = doc_ids
            lens[key] = sentence_lens
        return ids, lens, len(json_line), text


class Process(object):
    def __init__(self, args, workers):
        self.args = args
        self.workers = workers

    def print_processing_stats(self, count, proc_start, total_bytes_processed):
        if count % self.args.log_interval == 0:
            current = time.time()
            elapsed = current - proc_start
            mbs = total_bytes_processed/elapsed/1024/1024
            print(f"Processed {count} documents",
                  f"({count/elapsed} docs/s, {mbs} MB/s).",
                  file=sys.stderr)

    def count_lines(self, file_path):
        with open(file_path, 'r') as file:
            return sum(1 for _ in file)

    def split_sentences(self, file_name):
        input_file_name, output_file_name = file_name
        print("Opening", input_file_name)
        fin = open(input_file_name, 'r', encoding='utf-8')
        fout = open(output_file_name, 'w')

        encoder = Encoder(self.args)
        pool = multiprocessing.Pool(self.workers, initializer=encoder.initializer)
        split_docs = pool.imap(encoder.split, fin, 32)

        proc_start = time.time()
        total_bytes_processed = 0
        for i, (doc, bytes_processed) in enumerate(split_docs, start=1):
            total_bytes_processed += bytes_processed
            fout.write(doc + "\n")
            self.print_processing_stats(i, proc_start, total_bytes_processed)

        fin.close()
        fout.close()

def get_args():
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group(title='input data')
    group.add_argument('--input', type=str, nargs='+', required=True,
                       help='Path to input JSON')
    group.add_argument('--json-keys', nargs='+', default=['text'],
                       help='space separate listed of keys to extract from json')
    group.add_argument('--keep-newlines', action='store_true',
                       help='Keep newlines between sentences when splitting.')

    group = parser.add_argument_group(title='tokenizer')
    group.add_argument('--tokenizer-type', type=str, required=True,
                       choices=['BertWordPieceLowerCase','BertWordPieceCase',
                                'GPT2BPETokenizer', 'SentencePieceTokenizer',
                                'GPTSentencePieceTokenizer', 'NullTokenizer'],
                       help='What type of tokenizer to use.')
    group.add_argument('--tokenizer-model', type=str, default=None,
                       help='YTTM tokenizer model.')
    group.add_argument('--append-eod', action='store_true',
                       help='Append an <eod> token to the end of a document.')
    group.add_argument('--lang', type=str, default='english',
                       help='Language to use for NLTK-powered sentence splitting.')
    group = parser.add_argument_group(title='output data')
    group.add_argument('--output-prefix', type=str, required=True,
                       help='Path to binary output file without suffix')
    group.add_argument('--dataset-impl', type=str, default='mmap',
                       choices=['lazy', 'cached', 'mmap'])

    group = parser.add_argument_group(title='runtime')
    group.add_argument('--max_workers', type=int, required=True,
                       help=('Number of worker processes to launch.'
                             'A good default for fast pre-processing '
                             'is: (max_workers * partitions) = available CPU cores.'))

    args = parser.parse_args()
    args.keep_empty = False

    # some default/dummy values for the tokenizer
    args.rank = 1
    args.make_vocab_size_divisible_by = 128
    args.tensor_model_parallel_size = 1
    args.vocab_extra_ids = 0

    return args

def get_file_name(args, file_id):
    file_name, extension = os.path.splitext(args.input)
    input_file_name = file_name + "_" + str(file_id) + extension
    sentence_split_file = file_name + "_ss_" + str(file_id) + extension
    output_prefix = args.output_prefix + "_" + str(file_id)
    file_names = {
        'partition': input_file_name,
        'sentence_split': sentence_split_file,
        'output_prefix': output_prefix}
    return file_names

def check_files_exist(target_files_list, key, num_partitions):
    print("L237 num_partitions" , num_partitions)
    for i in range(num_partitions):
        print( target_files_list[i][key]) 
        if not os.path.exists(target_files_list[i][key]):
            return False
    return True

def count_lines(file_path):
    with open(file_path, 'r') as file:
        return sum(1 for _ in file)

def delete_cache(directory):
    # ディレクトリを下降順で走査
    for root, dirs, files in os.walk(directory, topdown=False):
        # ファイルの削除
        for name in files:
            file_path = os.path.join(root, name)
            if not (name.endswith('.bin') or name.endswith('.idx')):
                os.remove(file_path)
                # print(f"Deleted file: {file_path}")

        # 空のディレクトリの削除
        for name in dirs:
            dir_path = os.path.join(root, name)
            try:
                os.rmdir(dir_path)
                # print(f"Deleted directory: {dir_path}")
            except OSError as e:
                print(f"Directory not empty: {dir_path}")


# 並列を行う関数
def process_item(target_files_list):
    input_file_name = target_files_list['partition']
    output_prefix = target_files_list['output_prefix']
    args = target_files_list['args']

    level = "document"
    if target_files_list['sentence_split']:
        level = "sentence"

    encoder = Encoder(args)
    encoder.initializer()

    tokenizer = build_tokenizer(args)
    ids = {}

    output_bin_files = {}
    output_idx_files = {}
    builders = {}

    token_total_num = 0
    char_total_num = 0

    for key in args.json_keys:
        output_bin_files[key] = "{}_{}_{}.bin".format(output_prefix,
                                                        key, level)
        output_idx_files[key] = "{}_{}_{}.idx".format(output_prefix,
                                                        key, level)
        builders[key] = indexed_dataset.make_builder(output_bin_files[key],
                                                impl=args.dataset_impl,
                                                vocab_size=tokenizer.vocab_size)

    # 総行数を取得
    print("Opening, counting lines ....", input_file_name)
    total_lines = count_lines(input_file_name)
    print("Opened : ", input_file_name)

    with open(input_file_name, 'r') as file:
        # tqdmを使用して進捗状況を表示
        for line in tqdm(file, total=total_lines, desc="Processing lines"):
            ids, sentence_lens, len_json_line, text =  encoder.encode(line)
            for key in args.json_keys:
                if key not in ids:
                    print(f"Warning: Key '{key}' not found in {input_file_name}.")
                    continue  # キーがない場合、このイテレーションをスキップ
                token = ids['text'] #31番始まり、7番終わりのトークンのdict
                # encoded_docs.extend(token)
                token_num = len(token) #トークン数
                char_num = len(text) #文字数
                token_total_num += token_num
                char_total_num += char_num
            for key in ids.keys():
                builders[key].add_doc(ids[key], sentence_lens[key])

    time.sleep(60)
    builders[key].finalize(output_idx_files[key])
    time.sleep(60)
    delete_cache( output_prefix )
    bin_file_size_b = os.path.getsize(output_bin_files[key]) 
    bin_file_size_gb = bin_file_size_b / (1024 ** 3)
    idx_file_size_mb = os.path.getsize(output_idx_files[key]) / (1024 ** 2)
    file_size_ratio = bin_file_size_b / token_total_num 
    if file_size_ratio == 2:
        file_size_ratio_OK = True
    else:
        file_size_ratio_OK = False

    result_dict = {
        'input_file_name': input_file_name, 
        'output_bin_files': output_bin_files[key] , 
        'token_total_[BT]': token_total_num/10**9,
        'char_total_[BW]': char_total_num/10**9,
        'jsonl_file_size_[GB]': target_files_list['jsonl_file_size_gb'] ,
        'bin_file_size_[GB]': bin_file_size_gb ,
        'bin_file_size_ratio': file_size_ratio ,
        'file_size_ratio_OK': file_size_ratio_OK ,
        'idx_file_size[MB]': idx_file_size_mb ,
        'tokenizer_model':args.tokenizer_model }

    df = pd.DataFrame(list(result_dict))
    # CSVに出力
    csv_path = "{}_result.csv".format(output_prefix)
    df.to_csv(csv_path, index=False)

    return result_dict

def main():
    # 現在の日本時間を取得
    start_japan = datetime.now(japan_timezone)    
    # 日時を秒単位までのフォーマットで表示
    formatted_time = start_japan.strftime("%Y-%m-%d %H:%M:%S")
    print("Start time: ", formatted_time)

    args = get_args()
        
    extension = ".jsonl"

    #処理するファイルリスト
    target_dir_list = (args.input )
    # JSONライブラリを使用して文字列をリストに変換
    target_dir_list = target_dir_list[1:-1]

    # リスト内の各ディレクトリに対して操作を行う
    list_dir_path = []
    for dir in target_dir_list:
        # 余分なクォートとカンマを削除
        list_dir_path.append(dir.strip('",'))
    print("L356 list_dir_path" , list_dir_path)
    #処理するファイルリスト
    list_files_in_dir = []
    target_files_list = []
    list_jsonl_files_path_in_dir = []
    list_jsonl_files_name_in_dir = []
    list_dir_name = []
    for dir in list_dir_path:
        # 現在のディレクトリ内のファイルをリストアップ
        list_files_in_dir = os.listdir(dir)
        list_jsonl_files_path_in_dir.append( list_files_in_dir )
        list_jsonl_files_name_in_dir.append( [file for file in list_files_in_dir if file.endswith(extension)] )
        list_dir_name.append( os.path.basename(dir) )

    # ファイルリストを表示
    print("list_files_in_dir" , list_files_in_dir)
    print("list_jsonl_files_path_in_dir" , list_jsonl_files_path_in_dir)
    print("list_jsonl_files_path_in_dir" , list_jsonl_files_name_in_dir)
    print("list_dir_name" , list_dir_name)

    for index, dir_path in enumerate(list_dir_path):
        dir_name = list_dir_name[index]
        for file_name in list_jsonl_files_name_in_dir[index]:
            file_path = os.path.join(dir_path , file_name)
            # ファイルサイズを取得し、ギガバイトに変換
            file_size_gb = os.path.getsize(file_path) / (1024 ** 3)
            file_name_only, extension = os.path.splitext(file_name)
            sentence_split_file = file_name + "_ss" + extension
            output_dir = os.path.join(args.output_prefix, dir_name) 
            output_prefix = os.path.join(output_dir, file_name_only)

            # フォルダが存在しないことを確認し、存在する場合はエラーを発生
            assert not os.path.exists(output_prefix), f"Please delete the folder at {output_prefix} and rerun the program."
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                print(f"Created directory: {output_dir}")

            file_dict = {
                'partition': file_path, 
                'sentence_split': sentence_split_file,
                'output_prefix': output_prefix,
                'jsonl_file_size_gb': file_size_gb, 
                'args':args }
            target_files_list.append(file_dict)

    # ProcessPoolExecutor を使用し、最大ワーカー数を30に設定
    with concurrent.futures.ProcessPoolExecutor(max_workers=30) as executor:
        # map を使用してリスト内の各ファイルに対して process_item 関数を並行実行
        list_results = list(executor.map(process_item, target_files_list))

    print( "ProcessPoolExecutor Done. Please wait ...")
    time.sleep(10)
    print( "Please wait ...")
    time.sleep(10)
    print( "Make result ...")

    list_data_path = []
    for result in list_results:
        output_bin_path = result["output_bin_files"]
        data_path, extension = os.path.splitext(output_bin_path)
        list_data_path.append( str( '"' + data_path + '"') )

    # 現在の日時を取得し、ファイル名に使用する形式にフォーマットする
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    # ファイル名に現在の日時を追加
    filename = f"DATA_PATH_LIST_{current_time}.txt"
    # ファイルに書き出す
    with open( filename, 'w') as file:
        for path in list_data_path:
            file.write(path + '\n')  # 各パスを改行文字とともに書き出す                      

    df = pd.DataFrame(list_results)
    # CSVに出力
    csv_path =  args.output_prefix  + "result.csv"
    df.to_csv(csv_path, index=False)

    # 現在の日本時間を取得
    finish_japan = datetime.now(japan_timezone)    
    # 日時を秒単位までのフォーマットで表示
    formatted_time = finish_japan.strftime("%Y-%m-%d %H:%M:%S")
    print("Finish time: ", formatted_time)
    process_time = finish_japan - start_japan
    print( " ---- Fnish   process_time : " , process_time)


if __name__ == '__main__':
    main()

