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

Read_Line = 100000

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
        {}
        # try:
        #     data = json.loads(json_line)
        # except json.decoder.JSONDecodeError as e:
        #     print(f"デコードエラー: {e} ") 
        #     json_line = ""
        #     text = ""
        #     return ids, lens, len(json_line), text
        data = json_line
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
                # print("L104 sentence" , sentence)
                sentence_ids, decode_text = Encoder.tokenizer.tokenize_and_count(sentence)
                # print("L107 sentence_ids" , sentence_ids)
                # print("L108 decode_text" , decode_text)

                if len(sentence_ids) > 0:
                    doc_ids.extend(sentence_ids)
                    sentence_lens.append(len(sentence_ids))
            if len(doc_ids) > 0 and self.args.append_eod:
                doc_ids.append(Encoder.tokenizer.eod)
            ids[key] = doc_ids
            lens[key] = sentence_lens
            input_text = text
        return ids, lens, len(json_line), input_text, decode_text

class Process(object):
    def __init__(self, args):
        self.args = args

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

def get_args():
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group(title='input data')
    group.add_argument('--input', type=str, required=True,
                       help='Path to input JSON')
    group.add_argument('--json-keys', nargs='+', default=['text'],
                       help='space separate listed of keys to extract from json')
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
    
    group = parser.add_argument_group(title='output data')
    group.add_argument('--output-prefix', type=str, required=True,
                       help='Path to binary output file without suffix')
    group.add_argument('--dataset-impl', type=str, default='mmap',
                       choices=['lazy', 'cached', 'mmap'])

    args = parser.parse_args()
    args.keep_empty = False

    # some default/dummy values for the tokenizer
    args.rank = 1
    args.make_vocab_size_divisible_by = 128
    args.tensor_model_parallel_size = 1
    args.vocab_extra_ids = 0

    return args

def count_lines(file_path):
    with open(file_path, 'r') as file:
        return sum(1 for _ in file)

def read_jsonl_in_batches(file_path, batch_size=Read_Line):
    """指定されたJSONLファイルからデータをバッチで読み込むジェネレーター関数"""
    batch = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # JSON行を辞書に変換
            record = json.loads(line.strip())
            batch.append(record)
            # バッチサイズに達したらバッチを返す
            if len(batch) == batch_size:
                yield batch
                batch = []  # バッチをリセット
        # ファイルの末尾に達し、残りのデータがある場合は返す
        if batch:
            yield batch

# 並列を行う関数
def process_item(target_files_list):
    input_file_path = target_files_list['partition']
    output_prefix = target_files_list['output_prefix']
    file_name_only = target_files_list['file_name_only']
    output_file = os.path.join(output_prefix, file_name_only) + "_packing.jsonl"

    args = target_files_list['args']

    encoder = Encoder(args)
    encoder.initializer()

    tokenizer = build_tokenizer(args)

    list_token_num_10t = []
    list_token_num_11_20t = []
    list_token_num_21_50t = []
    list_token_num_51_75t = []
    list_token_num_76_100t = []
    list_token_num_101_200t = []
    list_token_num_201_500t = []
    list_token_num_501_1000t = []
    list_token_num_1001_2048t = []
    list_complete_doc = []

    for batch in read_jsonl_in_batches(input_file_path):
        # tqdmを使用して進捗状況を表示
        for line in tqdm(batch, total=Read_Line, desc="Processing lines"):
            ids, sentence_lens, len_json_line, text, decode_text =  encoder.encode(line)
            # print("L216 ids" , ids)
            # print("L217 text" , text)
            for key in args.json_keys:
                if key not in ids:
                    print(f"Warning: Key '{key}' not found in {input_file_path}.")
                    continue  # キーがない場合、このイテレーションをスキップ
                token = ids['text'] #31番始まり、7番終わりのトークンのdict
                # encoded_docs.extend(token)
                token_num = len(token) #トークン数
                while True:
                    doc = {'length': token_num, 'text': text }
                    if token_num < 2 :
                        break
                    elif token_num < 11 :
                        list_token_num_10t.append(doc)
                        break
                    elif token_num < 21 :
                        list_token_num_11_20t.append(doc)
                        break
                    elif token_num < 51 :
                        list_token_num_21_50t.append(doc)
                        break
                    elif token_num < 76 :
                        list_token_num_51_75t.append(doc)
                        break
                    elif token_num < 101 :
                        list_token_num_76_100t.append(doc)
                        break
                    elif token_num < 201 :
                        list_token_num_101_200t.append(doc)
                        break
                    elif token_num < 501 :
                        list_token_num_201_500t.append(doc)
                        break
                    elif token_num < 1001 :
                        list_token_num_501_1000t.append(doc)
                        break
                    elif token_num < 2048 :
                        list_token_num_1001_2048t.append(doc)
                        break
                    else:
                        # print("L312 token ", token[ 2038:2058] )
                        doc_token = token[0:2048]
                        doc_token_num = len(doc_token)
                        doc_text = decode_text[0:2048]
                        doc = {'length': doc_token_num, 'text': ''.join(doc_text) }
                        list_complete_doc.append(doc)
                        token = token[2048:]
                        token_num = len(token)
                        if token_num < 2048 :
                            text = ''.join(decode_text[2048:])
                        else:
                            decode_text = decode_text[2048:]
                        # print("doc_token" , doc_token[-10:])
                        # print("token" , token[:10])
        tmp_list_token_num_1001_2048t = []
        def add_doc_process(add_list, token_num, text, list_complete_doc, tmp_list_token_num_1001_2048t):
            if len(add_list)  == 0 :
                return add_list, list_complete_doc, tmp_list_token_num_1001_2048t
            else:
                add_doc = add_list[0]
                add_list = add_list[1:]
                total_token_num = int(token_num) + int(add_doc['length'])
                # print("total_token_num" , total_token_num)
                # print("token_num" , token_num )
                # print("add_doc['length']" , add_doc['length'] )
                total_text = str(text) + str(add_doc['text'])
                doc = {'length': total_token_num, 'text': total_text }
                if total_token_num > 2040:
                    list_complete_doc.append(doc)
                else:
                    tmp_list_token_num_1001_2048t.append(doc)
                return add_list, list_complete_doc, tmp_list_token_num_1001_2048t

        pre_tmp_list_token_num_1001_2048t_len = len(list_token_num_1001_2048t)
        while True:
            if len(list_token_num_1001_2048t) == 0:
                break
            for token_num_1001_2048t in list_token_num_1001_2048t:
                token_num = token_num_1001_2048t['length']
                text = token_num_1001_2048t['text']
                diff = 2047 - token_num
                if diff > 1000:
                    list_token_num_501_1000t, list_complete_doc, tmp_list_token_num_1001_2048t =\
                        add_doc_process(list_token_num_501_1000t, token_num, text, list_complete_doc, tmp_list_token_num_1001_2048t )
                elif diff > 500:
                    list_token_num_201_500t, list_complete_doc, tmp_list_token_num_1001_2048t =\
                        add_doc_process(list_token_num_201_500t, token_num, text, list_complete_doc, tmp_list_token_num_1001_2048t )
                elif diff > 200:
                    list_token_num_101_200t, list_complete_doc, tmp_list_token_num_1001_2048t =\
                        add_doc_process(list_token_num_101_200t, token_num, text, list_complete_doc, tmp_list_token_num_1001_2048t )               
                elif diff > 100:
                    list_token_num_101_200t, list_complete_doc, tmp_list_token_num_1001_2048t =\
                        add_doc_process(list_token_num_101_200t, token_num, text, list_complete_doc, tmp_list_token_num_1001_2048t )                            
                elif diff > 75:
                    list_token_num_76_100t, list_complete_doc, tmp_list_token_num_1001_2048t =\
                        add_doc_process(list_token_num_76_100t, token_num, text, list_complete_doc, tmp_list_token_num_1001_2048t )       
                elif diff > 50:
                    list_token_num_21_50t, list_complete_doc, tmp_list_token_num_1001_2048t =\
                        add_doc_process(list_token_num_21_50t, token_num, text, list_complete_doc, tmp_list_token_num_1001_2048t )    
                elif diff > 20:
                    list_token_num_11_20t, list_complete_doc, tmp_list_token_num_1001_2048t =\
                        add_doc_process(list_token_num_11_20t, token_num, text, list_complete_doc, tmp_list_token_num_1001_2048t )    
                elif diff > 10:
                    list_token_num_10t, list_complete_doc, tmp_list_token_num_1001_2048t =\
                        add_doc_process(list_token_num_10t, token_num, text, list_complete_doc, tmp_list_token_num_1001_2048t )    
            list_token_num_1001_2048t = tmp_list_token_num_1001_2048t
            if pre_tmp_list_token_num_1001_2048t_len == len(list_token_num_1001_2048t):
                break
            else:
                pre_tmp_list_token_num_1001_2048t_len = len(list_token_num_1001_2048t)
        with open(output_file, 'a', encoding='utf-8') as f_out:
            for complete_doc in list_complete_doc:
                doc = {'text': complete_doc['text'] }
                f_out.write(json.dumps(doc, ensure_ascii=False) + '\n')
        print("Done batch " , output_file)
        print( "list_complete_doc" , len(list_complete_doc) )
        print( "list_token_num_1001_2048t" , len(list_token_num_1001_2048t) )
        print( "list_token_num_501_1000t" , len(list_token_num_501_1000t) )
        print( "list_token_num_201_500t" , len(list_token_num_201_500t) )
        print( "list_token_num_101_200t" , len(list_token_num_101_200t) )
        print( "list_token_num_76_100t" , len(list_token_num_76_100t) )
        print( "list_token_num_51_75t" , len(list_token_num_51_75t) )
        print( "list_token_num_21_50t" , len(list_token_num_21_50t) )
        print( "list_token_num_11_20t" , len(list_token_num_11_20t) )
        print( "list_token_num_10t" , len(list_token_num_10t) )
        list_complete_doc.clear()

    def write_jsonl(list_dict, file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as file:
            for doc in list_dict:
                doc_to_write = {'text': doc['text']}
                # JSON形式でファイルに書き込み
                file.write(json.dumps(doc_to_write, ensure_ascii=False) + '\n')

    leftover_file = os.path.join(output_prefix, 'leftover_file' ,file_name_only) 
    file_path_list_token_num_1001_2048t = leftover_file + "/1001_2048t.jsonl"
    write_jsonl(list_token_num_1001_2048t, file_path_list_token_num_1001_2048t)
    file_path_list_token_num_501_1000t = leftover_file + "/501_1000t.jsonl"
    write_jsonl(list_token_num_501_1000t, file_path_list_token_num_501_1000t)
    file_path_list_token_num_201_500t = leftover_file + "/201_500t.jsonl"
    write_jsonl(list_token_num_201_500t, file_path_list_token_num_201_500t)
    file_path_list_token_num_101_200t = leftover_file + "/101_200t.jsonl"
    write_jsonl(list_token_num_101_200t, file_path_list_token_num_101_200t)
    file_path_list_token_num_76_100t = leftover_file + "/76_100t.jsonl"
    write_jsonl(list_token_num_76_100t, file_path_list_token_num_76_100t)
    file_path_list_token_num_51_75t = leftover_file + "/51_75t.jsonl"
    write_jsonl(list_token_num_51_75t, file_path_list_token_num_51_75t)
    file_path_list_token_num_21_50t = leftover_file + "/21_50t.jsonl"
    write_jsonl(list_token_num_21_50t, file_path_list_token_num_21_50t)
    file_path_list_token_num_11_20t = leftover_file + "/11_20t.jsonl"
    write_jsonl(list_token_num_11_20t, file_path_list_token_num_11_20t)
    file_path_list_token_num_10t = leftover_file + "/10t.jsonl"
    write_jsonl(list_token_num_10t, file_path_list_token_num_10t)
    print("Done file " , output_file)
    print( "list_complete_doc" , len(list_complete_doc) )
    print( "list_token_num_1001_2048t" , len(list_token_num_1001_2048t) )
    print( "list_token_num_501_1000t" , len(list_token_num_501_1000t) )
    print( "list_token_num_201_500t" , len(list_token_num_201_500t) )
    print( "list_token_num_101_200t" , len(list_token_num_101_200t) )
    print( "list_token_num_76_100t" , len(list_token_num_76_100t) )
    print( "list_token_num_51_75t" , len(list_token_num_51_75t) )
    print( "list_token_num_21_50t" , len(list_token_num_21_50t) )
    print( "list_token_num_11_20t" , len(list_token_num_11_20t) )
    print( "list_token_num_10t" , len(list_token_num_10t) )

    return 0

def leftover_maket(arget_files_list):
    input_file_path = target_files_list['partition']
    output_prefix = target_files_list['output_prefix']
    file_name_only = target_files_list['file_name_only']
    output_file = os.path.join(output_prefix, file_name_only) + "_packing.jsonl"
    leftover_file = os.path.join(output_prefix, 'leftover_file') 
    



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
    # target_dir_list = target_dir_list[1:-1]

    # リスト内の各ディレクトリに対して操作を行う
    list_dir_path = []
    # for dir in target_dir_list:
    #     # 余分なクォートとカンマを削除
    #     list_dir_path.append(dir.strip('",'))
    # print("L356 list_dir_path" , list_dir_path)
    list_dir_path.append(target_dir_list)

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
    # print("list_files_in_dir" , list_files_in_dir)
    print("list_jsonl_files_path_in_dir" , list_jsonl_files_path_in_dir)
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
            

            # フォルダが存在しないことを確認し、存在する場合はエラーを発生
            # assert not os.path.exists(output_dir), f"Please delete the folder at {output_dir} and rerun the program."
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                print(f"Created directory: {output_dir}")

            file_dict = {
                'partition': file_path, 
                'sentence_split': sentence_split_file,
                'output_prefix': output_dir,
                'file_name_only': file_name_only,
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
    # ファイルに書き出す
    with open('DATA_PATH_LIST.txt', 'w') as file:
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

    # 処理開始から終了までの時間を計算
    process_time = finish_japan - start_japan
    print(" ---- Finish process_time : ", process_time)

if __name__ == '__main__':
    main()
          