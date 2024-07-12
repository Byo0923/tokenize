import os

def create_sequential_files(base_path, start, end, extension='.text'):
    # ディレクトリが存在しない場合は作成する
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    # 指定の連番でファイルを作成する
    for i in range(start, end + 1):
        file_name = f"{i}{extension}"
        file_path = os.path.join(base_path, file_name)
        
        # ファイルに連番を書き込む
        with open(file_path, 'w') as file:
            file.write(str("0"))

# 使用例: 'path/to/directory' を実際のパスに置き換えてください
create_sequential_files('/storage5/shared/corpus/synthetic/SyntheticTexts/flags', 39146, 39146, '.txt')