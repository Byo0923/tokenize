import json
import os
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import re
import concurrent.futures
from pathlib import Path

# 検索するディレクトリのパス
base_path = Path('speech/2000-')

model_id = "geniacllm/dMoE_8B_math_iter8999"
tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False, torch_dtype="auto", device_map="auto")
print(" load tokenizer ")

# .jsonl ファイルのフルパスを格納するリスト
jsonl_files = []
jsonl_datas = []

# base_path 以下のディレクトリを走査
for subdir in base_path.iterdir():
    if subdir.is_dir():  # サブディレクトリの場合
        # 各サブディレクトリ内の .jsonl ファイルを検索
        for jsonl_file in subdir.glob('*.jsonl'):
            jsonl_files.append(jsonl_file)
print(jsonl_files)

for file_path in jsonl_files:
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # JSON形式のデータをPythonの辞書に変換
            data = json.loads(line)
            jsonl_datas.append(data["text"])

# print("data", gsm8l_datas )
print("data", len(jsonl_datas) )
print("data", jsonl_datas[-1] )

# model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", device_map="auto")
# print(model)
# inputs = tokenizer("地球温暖化の主な原因となる物質は", return_tensors="pt").to(model.device)
# inputs

all_data = []
word_num = 0

short_full_document_list =[]
complete_full_document_list =[]
count = 0

total_items = len(jsonl_datas)
# big_list = list(range(total_items))

# サブリストの数
num_chunks = 25
# 各サブリストの長さを計算
items_per_sublist = total_items // num_chunks
# サブリストを生成
chunks = [jsonl_datas[i * items_per_sublist:(i + 1) * items_per_sublist] for i in range(num_chunks)]

# 最後のサブリストに残りのアイテムを追加する
if total_items % num_chunks != 0:
    remaining_items = jsonl_datas[num_chunks * items_per_sublist:]
    chunks[-1].extend(remaining_items)

# 並列を行う関数
def process_item(sublists):
    for index, one_data in tqdm( enumerate(sublists)):
        one_data_token = tokenizer(one_data, return_tensors="pt", add_special_tokens=True)
        one_data_token_len =  one_data_token.input_ids[0].size(0) 
        if   one_data_token_len < 50:
            pass
        elif   one_data_token_len < 1600:
            short_full_document_list.append( {'length': one_data_token_len, 'text': one_data } )
        elif  one_data_token_len < 2010:
            complete_full_document_list.append( {'length': one_data_token_len, 'text': one_data } )
        else:
            one_sentences = one_data.split('。')
            collect_one_sentences = []
            split_one_sentence_len = 0
            for index, sentence in enumerate(one_sentences):
                one_sentence_token = tokenizer(sentence, return_tensors="pt", add_special_tokens=True)
                one_sentence_token_len =  one_sentence_token.input_ids[0].size(0) 
                if split_one_sentence_len + one_sentence_token_len < 2010:
                    split_one_sentence_len = split_one_sentence_len + one_sentence_token_len
                    collect_one_sentences.append( sentence )
                elif one_sentence_token_len > 2010:
                    print(" Warning : Over one-sentence token  :", one_sentence_token_len  )
                    print( sentence  )
                    split_document = "\n".join(collect_one_sentences)
                    short_full_document_list.append( {'length': split_one_sentence_len, 'text': split_document } )
                    split_one_sentence_len = 0
                    collect_one_sentences.clear()
                else :
                    split_document = "\n".join(collect_one_sentences)
                    complete_full_document_list.append( {'length': split_one_sentence_len, 'text': split_document } )
                    split_one_sentence_len = one_sentence_token_len
                    collect_one_sentences.clear()
                    collect_one_sentences.append( sentence )
            split_document = "\n".join(collect_one_sentences)
            if split_one_sentence_len < 1500 :
                short_full_document_list.append( {'length': split_one_sentence_len, 'text': split_document } )
            elif split_one_sentence_len < 2015 :
                complete_full_document_list.append( {'length': split_one_sentence_len, 'text': split_document } )
            else:
                print(" Warning : Over split-one-sentence token  :", split_one_sentence_len  )
                print( split_document  )
        # if index > 96605:
        #     break

    complete_full_document_sub_list = []

    while True:
        if len(short_full_document_list) == 0:
            break

        if len(short_full_document_list)%500 == 0:
            print( "残り : ", len(short_full_document_list) )
        target = short_full_document_list[0]
        left_len = 2010 -  target['length'] 
        if left_len < 0 :
            print(" Error : Over token   target :", target['length']  )
        del short_full_document_list[0]  # 0番目の要素を削除
        if len(short_full_document_list) == 0:
            complete_full_document_sub_list.append( target )
            break
        else:
            # ステップ1: 最小差分を計算（デフォルト値を使用）
            closest_length_diff = min((left_len - rec['length'] for rec in short_full_document_list if rec['length'] <= left_len), default=None)

            if closest_length_diff is None:
                complete_full_document_sub_list.append( target )
                # print( "complete_full_document_list.append"  )
            else:
                # 特定の条件に合致するデータのインデックスを探す
                index = next((i for i, rec in enumerate(short_full_document_list) if rec['length'] == left_len - closest_length_diff ), None)
                merge_document = short_full_document_list[index]
                del short_full_document_list[index]  # index番目の要素を削除 
                # 結果の出力
                if index is not None:
                    merge_texts = target['text'] + "</s>" + merge_document['text'] 
                    if target['length']  + merge_document['length']  < 1500 :
                        short_full_document_list.append( {'length': target['length']  + merge_document['length'] + 1 , 'text': merge_texts } )
                        # print( "merge_texts" , merge_texts )
                        # print( "short_full_document_list" , short_full_document_list[-1] )
                    elif  target['length']  + merge_document['length']  > 2022 :
                        print(" Error : Over token   target :", target['length'] , "  merge_document : ", merge_document['length']    )
                    else:
                        complete_full_document_sub_list.append( {'length': target['length']  + merge_document['length'] + 1 , 'text': merge_texts } )
                        # print( "merge_texts" , merge_texts )
                else:
                    print(f"長さが {merge_document['length'] } のデータは見つかりませんでした。")

    return complete_full_document_sub_list

# print( "L110 sublists " , len( chunks) )
# プロセスプールを使用して各チャンクを並列処理
with concurrent.futures.ProcessPoolExecutor() as executor:
    results = list(executor.map(process_item, chunks))
# print( "L114 results " , len( results) )
# for result in results:
    # print( "L116 result " , len( result) )

# リスト内包表記を使って平坦化
# sumを使って平坦化
complete_full_document_list.extend( sum(results, [] ))

print( "L121 complete_full_document_list " , len( complete_full_document_list) )

length_list = []
collect_length_list = []
file_count = 0

complete_full_document_list_len = len(complete_full_document_list)
# 各ファイルに格納する最大ドキュメント数
max_docs_per_file = 9999

# 必要なファイルの数
jsonl_num = complete_full_document_list_len // max_docs_per_file + (1 if complete_full_document_list_len % max_docs_per_file != 0 else 0)

# ファイルにデータを分割して保存
for i in range(jsonl_num):
    start_index = i * max_docs_per_file
    end_index = start_index + max_docs_per_file
    # ファイル名を生成し、連番を付ける
    filename = f"{i}_speech_text.jsonl"  # ファイル名形式: "番号_名前.jsonl"

    # ファイルを開いてデータを書き込む
    with open(filename, 'w', encoding='utf-8') as file:
        for document in complete_full_document_list[start_index:end_index]:
            # 各テキストに対するJSONオブジェクトを作成
            json_obj = {
                "text": document['text'] ,
                "is_rejected": False,
                "reason": {}
            }
            # JSON形式の文字列に変換
            json_line = json.dumps(json_obj, ensure_ascii=False)
            # ファイルに書き込み
            file.write(json_line + '\n')
            length_list.append( document['length'] ) 
            document_tokens = tokenizer(document['text'] , return_tensors="pt", add_special_tokens=True)
            document_tokens_len =  document_tokens.input_ids[0].size(0) 
            collect_length_list.append( document_tokens_len )
            if document_tokens_len > 2021:
                print("L209 Error : Over token ", document['length'] , "  collect : ", document_tokens_len  )
print( "len(complete_full_document_list) " , len(complete_full_document_list) )
# print( "length_list " , length_list )
