from huggingface_hub import HfApi
import os
from typing import List, Dict, Tuple
import time

def get_file_paths(folder_path) -> Tuple[List[str], List[str]]:
    if not os.path.exists(folder_path):
        print(f"错误：文件夹 '{folder_path}' 不存在！")
        return []

    file_paths = []
    name_list = []

    for root, _, files in os.walk(folder_path):
        for file_name in files:
            name_list.append(file_name)
            file_path = os.path.join(root, file_name)
            file_paths.append(file_path)

    return file_paths, name_list


def upload_qwen():
    api = HfApi()
    path_list, name_list = get_file_paths("Qwen")
    assert len(path_list) == len(name_list)
    for path, name in zip(path_list, name_list):
        print(f"[{time.ctime()}] uploading {name}. start")
        api.upload_file(
            path_or_fileobj=path,
            path_in_repo=name,
            repo_id="zjunlp/LightThinker-Qwen",
            repo_type="model",
        )
        print(f"[{time.ctime()}] uploading {name}. done")
        # api.upload_file(
        #     path_or_fileobj="Qwen/README.md",
        #     path_in_repo="README.md",
        #     repo_id="zjunlp/LightThinker-Qwen",
        #     repo_type="model",
        # )

def upload_llama():
    api = HfApi()
    path_list, name_list = get_file_paths("Llama")
    assert len(path_list) == len(name_list)
    for path, name in zip(path_list, name_list):
        print(f"[{time.ctime()}] uploading {name}. start")
        api.upload_file(
            path_or_fileobj=path,
            path_in_repo=name,
            repo_id="zjunlp/LightThinker-Llama",
            repo_type="model",
        )
        print(f"[{time.ctime()}] uploading {name}. done")


if __name__ == '__main__':
    # print(get_file_paths("Qwen"))
    upload_qwen()
    upload_llama()
