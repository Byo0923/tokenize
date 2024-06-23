import json
import os
import random
from tqdm import tqdm

# 対象のディレクトリを指定
input_directory = '/storage5/shared/k_nishizawa/tmp_2nd_tounyu_2/original_jsonl'
output_directory = '/storage5/shared/k_nishizawa/tmp_2nd_tounyu_2/merged_jsonl'

# 全てのtextフィールドを格納するリスト
texts = []
i = 0
# ディレクトリ内のすべての.jsonlファイルを読み込む
for filename in os.listdir( input_directory ):
    print('filename' , filename , "  :" , i , " / " , len(os.listdir( input_directory ) ) )

    if filename.endswith('.jsonl'):
        file_path = os.path.join( input_directory, filename)
        
        # ファイルの総ライン数を取得
        total_lines = sum(1 for line in open(file_path, 'r', encoding='utf-8'))
        
        # ファイルを開き、プログレスバーと共に処理
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in tqdm(file, total=total_lines, desc=f'Processing {filename}'):
                # 各行をJSONオブジェクトとして読み込む
                data = json.loads(line)
                # キーが'text'のもののみを抽出
                if 'text' in data:
                    texts.append(data['text'])
    i += 0
# リストをランダムに並び替える
random.shuffle(texts)

# 新しい.jsonlファイルに結果を書き出す
output_filename = os.path.join( output_directory , 'merged_output.jsonl')
with open(output_filename, 'w', encoding='utf-8') as outfile:
    # プログレスバーを設定
    progress = tqdm(total=len(texts), desc="Writing texts to file")
    for text in texts:
        # JSON形式でファイルに書き込み
        json.dump({"text": text}, outfile)
        outfile.write('\n')
        # プログレスバーを更新
        progress.update(1)
    progress.close()

print("Completed writing to", output_filename)



