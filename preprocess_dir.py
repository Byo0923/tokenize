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
        if self.args.split_sentences:
            if not nltk_available:
                print("NLTK is not available to split sentences.")
                exit()
            library = "tokenizers/punkt/{}.pickle".format(self.args.lang)
            splitter = nltk.load(library)
            if self.args.keep_newlines:
                # this prevents punkt from eating newlines after sentences
                Encoder.splitter = nltk.tokenize.punkt.PunktSentenceTokenizer(
                    train_text = splitter._params,
                    lang_vars = CustomLanguageVars())
            else:
                Encoder.splitter = splitter

        else:
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
        except json.JSONDecodeError as e:
            print(f"デコードエラー: {e} - JSONデータ: {s[:50]}...")
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

    def process_json_file(self, file_name, queue):
        input_file_name, output_prefix = file_name

        # 総行数を取得
        total_lines = count_lines(input_file_name)
        # ファイルサイズを取得し、ギガバイトに変換
        file_size_gb = os.path.getsize(input_file_name) / (1024 ** 3)

        token_total_num = 0
        char_total_num = 0
        fin = open(input_file_name, 'r', encoding='utf-8')

        startup_start = time.time()
        encoder = Encoder(self.args)
        tokenizer = build_tokenizer(self.args)
        pool = multiprocessing.Pool(self.workers, initializer=encoder.initializer)
        encoded_docs = pool.imap(encoder.encode, fin, 32)

        level = "document"
        if self.args.split_sentences:
            level = "sentence"

        output_bin_files = {}
        output_idx_files = {}
        builders = {}

        for key in self.args.json_keys:
            output_bin_files[key] = "{}_{}_{}.bin".format(output_prefix,
                                                          key, level)
            output_idx_files[key] = "{}_{}_{}.idx".format(output_prefix,
                                                          key, level)
            builders[key] = indexed_dataset.make_builder(output_bin_files[key],
                                                   impl=self.args.dataset_impl,
                                                   vocab_size=tokenizer.vocab_size)

        startup_end = time.time()
        proc_start = time.time()
        total_bytes_processed = 0

        print("Start :", input_file_name)
        print("Time to startup:", startup_end - startup_start)
        for i, (doc, sentence_lens, bytes_processed, text) in enumerate(tqdm(encoded_docs, total=total_lines), start=1):
            total_bytes_processed += bytes_processed
            for key in doc.keys():
                builders[key].add_doc(doc[key], sentence_lens[key])
            for key in self.args.json_keys:
                if key not in doc:
                    print(f"Warning: Key '{key}' not found in {input_file_name}.")
                    continue  # キーがない場合、このイテレーションをスキップ
                token = doc[key] #31番始まり、7番終わりのトークンのdict
                # encoded_docs.extend(token)
                token_num = len(token) #トークン数
                char_num =  len(text) #文字数
                token_total_num += token_num
                char_total_num += char_num
            # self.print_processing_stats(i, proc_start, total_bytes_processed)

        fin.close()
        builders[key].finalize(output_idx_files[key])

        result_dict = {
            'input_file_name': input_file_name, 
            'output_bin_files': output_bin_files[key] , 
            'token_total_[KT]': token_total_num/1000,
            'char_total_[KC]': char_total_num/1000,
            'file_size_gb': file_size_gb ,
            'tokenizer_model':self.args.tokenizer_model }
        # print("result_dict ", result_dict)

        # 結果をキューに入れる
        queue.put(result_dict)

def get_args():
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group(title='input data')
    group.add_argument('--input', type=str, required=True,
                       help='Path to input JSON')
    group.add_argument('--json-keys', nargs='+', default=['text'],
                       help='space separate listed of keys to extract from json')
    group.add_argument('--split-sentences', action='store_true',
                       help='Split documents into sentences.')
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
    group.add_argument('--vocab-file', type=str, default=None,
                       help='Path to the vocab file')
    group.add_argument('--vocab-size', default=786,
                       help='size of vocab for use with NullTokenizer')
    group.add_argument('--merge-file', type=str, default=None,
                       help='Path to the BPE merge file (if necessary).')
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
    group.add_argument('--workers', type=int, required=True,
                       help=('Number of worker processes to launch.'
                             'A good default for fast pre-processing '
                             'is: (workers * partitions) = available CPU cores.'))
    group.add_argument('--partitions', type=int, default=1,
                        help='Number of file partitions')
    group.add_argument('--log-interval', type=int, default=1000,
                       help='Interval between progress updates')
    args = parser.parse_args()
    args.keep_empty = False

    if args.tokenizer_type.lower().startswith('bert') and not args.split_sentences:
        print("Are you sure you don't want to split sentences?")

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


def check_files_exist(in_ss_out_names, key, num_partitions):
    for i in range(num_partitions):
        if not os.path.exists(in_ss_out_names[i][key]):
            return False
    return True

def count_lines(file_path):
    with open(file_path, 'r') as file:
        return sum(1 for _ in file)

# 並列を行う関数
def process_item(in_ss_out_names):
    input_file_name = in_ss_out_names['partition']
    output_prefix = in_ss_out_names['output_prefix']
    args = in_ss_out_names['args']
    print("Opening", input_file_name)

    level = "document"
    if in_ss_out_names['sentence_split']:
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
    total_lines = count_lines(input_file_name)

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

    builders[key].finalize(output_idx_files[key])
    result_dict = {
        'input_file_name': input_file_name, 
        'output_bin_files': output_bin_files[key] , 
        'token_total_[KB]': token_total_num/1000,
        'char_total_[KW]': char_total_num/1000,
        'file_size_[GB]': in_ss_out_names['jsonl_file_size_gb'] ,
        'tokenizer_model':args.tokenizer_model }
    return result_dict

def main():
    # 現在の日本時間を取得
    start_japan = datetime.now(japan_timezone)    
    # 日時を秒単位までのフォーマットで表示
    formatted_time = start_japan.strftime("%Y-%m-%d %H:%M:%S")
    print("Start time: ", formatted_time)

    args = get_args()

    if args.split_sentences:
        if nltk_available:
            nltk.download("punkt", quiet=True)
        else:
            raise Exception(
                "nltk library required for sentence splitting is not available.")

    extension = ".jsonl"

    #処理するファイルリスト
    in_ss_out_names = []

    # 現在のディレクトリ内のファイルをリストアップ
    files_in_directory = os.listdir(args.input )
    # .jsonlファイルだけをフィルタリング
    jsonl_files = [file for file in files_in_directory if file.endswith(extension)]

    # ファイルリストを表示
    print("jsonl_files" , jsonl_files)
    for file_name in jsonl_files:
        file_path = os.path.join(args.input , file_name)
        # ファイルサイズを取得し、ギガバイトに変換
        file_size_gb = os.path.getsize(file_path) / (1024 ** 3)
        file_name_only, extension = os.path.splitext(file_name)
        sentence_split_file = file_name + "_ss" + extension
        output_prefix = args.output_prefix  + file_name_only
        file_dict = {
            'partition': file_path, 
            'sentence_split': sentence_split_file,
            'output_prefix': output_prefix,
            'jsonl_file_size_gb': file_size_gb, 
            'args':args }
        in_ss_out_names.append(file_dict)

    # #処理するファイルリスト
    # in_ss_out_names = []
    # if args.partitions == 1:
    #     file_name, extension = os.path.splitext(args.input)
    #     print("file_name ", file_name)
    #     sentence_split_file = file_name + "_ss" + extension
    #     file_names = {
    #         'partition': args.input,
    #         'sentence_split': sentence_split_file,
    #         'output_prefix': args.output_prefix}
    #     in_ss_out_names.append(file_names)
    # else:
    #     in_file_names = glob.glob(args.input)

    #     # create .jsonl parition files
    #     for idx in range(args.partitions):
    #         in_ss_out_name = get_file_name(args, idx)
    #         in_ss_out_names.append(in_ss_out_name)

    #     # check to see if paritions were already created
    #     partitions_present = check_files_exist(in_ss_out_names, 'partition', args.partitions)

    #     # check to see if paritions with split sentences already created
    #     split_sentences_present = check_files_exist(in_ss_out_names, 'sentence_split', args.partitions)

    #     if not partitions_present and not split_sentences_present:
    #         # populate .jsonl partition files from parent files
    #         partitioned_input_files = []
    #         for idx in range(args.partitions):
    #             partitioned_input_file = open(in_ss_out_names[idx]['partition'], 'w')
    #             partitioned_input_files.append(partitioned_input_file)

    #         index = 0
    #         for in_file_name in in_file_names:
    #             # support for gzip files
    #             if in_file_name.endswith(".gz"):
    #                 fin = gzip.open(in_file_name, 'rt')
    #             else:
    #                 fin = open(in_file_name, 'r', encoding='utf-8')

    #             for line in fin:
    #                 print("line ", line)
    #                 partitioned_input_files[index].write(line)
    #                 index = (index + 1)%args.partitions

    #             fin.close()

    #         for idx in range(args.partitions):
    #             partitioned_input_files[idx].close()

    assert args.workers % args.partitions == 0
    process = Process(args, args.workers )

    # check to see if paritions with split sentences already created
    split_sentences_present = check_files_exist(in_ss_out_names, 'sentence_split', args.partitions)
    # print("split_sentences_present" , split_sentences_present)
    # print("args.split_sentences" , args.split_sentences)

    # split sentences in partition files
    # if args.split_sentences and not split_sentences_present:
    #     processes = []
    #     for name in in_ss_out_names:
    #         # print("name" , name)
    #         p = multiprocessing.Process(target=partition.split_sentences,
    #                                     args=((name['partition'], name['sentence_split']),))
    #         p.start()
    #         processes.append(p)

    #     for p in processes:
    #         p.join()

    #     if args.partitions == 1:
    #         return

    # encode partition files in parallel
    processes = []
    input_key = 'sentence_split' if args.split_sentences else 'partition'

    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(executor.map(process_item, in_ss_out_names))
    # for  in_ss_out_name in in_ss_out_names :
    #     process_item( in_ss_out_name )
    # print("result_dict ," , results)

    df = pd.DataFrame(results)

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
    # if args.partitions == 1:
    #     return

    # # merge bin/idx partitions
    # level = "document"
    # if args.split_sentences:
    #     level = "sentence"

    # output_bin_files = {}
    # output_idx_files = {}
    # builders = {}
    # tokenizer = build_tokenizer(args)

    # for key in args.json_keys:
    #     output_bin_files[key] = "{}_{}_{}.bin".format(args.output_prefix,
    #                                                   key, level)
    #     output_idx_files[key] = "{}_{}_{}.idx".format(args.output_prefix,
    #                                                   key, level)
    #     builders[key] = indexed_dataset.make_builder(output_bin_files[key],
    #                                                  impl=args.dataset_impl,
    #                                                  vocab_size=tokenizer.vocab_size)
    #     for name in in_ss_out_names:
    #         parition_output_prefix = name['output_prefix']
    #         full_partition_output_prefix = "{}_{}_{}".format(parition_output_prefix,
    #                                                          key, level)
    #         builders[key].merge_file_(full_partition_output_prefix)
    #     builders[key].finalize(output_idx_files[key])


if __name__ == '__main__':
    main()

