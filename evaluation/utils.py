
import jsonlines
from typing import *
import time

def _print(messages):
    print(f"[{time.ctime()}] {messages}")

def read_jsonl(file_name:str) -> List[Dict]:
    results:List[Dict] = list()
    with jsonlines.open(file_name, 'r') as reader:
        for item in reader:
            results.append(item)
    return results


