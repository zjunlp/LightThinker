
import jsonlines
import json
from tqdm import tqdm

import sys
sys.path.append('../LightThinker')
from dataset_reader import GPQAReader, Reader, MMLUReader, BBHReader, GSM8KReader
import os

def bbh():
    reader_map_list = [
        (BBHReader(), "bbh"),
    ]

    for reader, name in reader_map_list:
        with jsonlines.open(f"reference/{name}.jsonl", "w") as writer:
            for i in range(len(reader)):
                answer = str(reader.data_list[i]['answer'])
                alias = [str(reader.data_list[i]['answer']), "\\text{" + reader.data_list[i]['answer'] + "}"]
                error = ['error']
                if answer.lower() == 'yes':
                    alias.extend(['Yes', 'yes', r'\text{yes}', r'\text{Yes}', 'A'])
                    error.extend(['No', 'no', r'\text{No}', r'\text{no}', 'B'])
                elif answer.lower() == 'no':    
                    alias.extend(['No', 'no', r'\text{No}', r'\text{no}', 'B'])
                    error.extend(['Yes', 'yes', r'\text{yes}', r'\text{Yes}', 'A'])
                if answer.lower() == 'true':
                    alias.extend(['True', 'true', r'\text{true}', r'\text{True}'])
                    error.extend(['False', 'false', r'\text{false}', r'\text{False}'])
                elif answer.lower() == 'false':    
                    alias.extend(['False', 'false', r'\text{false}', r'\text{False}'])
                    error.extend(['True', 'true', r'\text{true}', r'\text{True}'])

                if answer.lower() == 'valid':
                    alias.extend(['valid', 'Valid', r'\text{valid}', r'\text{Valid}'])
                    error.extend(['Invalid', 'invalid', r'\text{invalid}', r'\text{Invalid}'])
                elif answer.lower() == 'invalid':
                    error.extend(['valid', 'Valid', r'\text{valid}', r'\text{Valid}'])
                    alias.extend(['Invalid', 'invalid', r'\text{invalid}', r'\text{Invalid}'])
                
                if answer == 'A':
                    error.extend(['B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K'])
                elif answer == 'B':
                    error.extend(['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K'])
                elif answer == 'C':
                    error.extend(['A', 'B', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K'])
                elif answer == 'D':
                    error.extend(['A', 'B', 'C', 'E', 'F', 'G', 'H', 'I', 'J', 'K'])
                elif answer == 'E':
                    error.extend(['A', 'B', 'C', 'D', 'F', 'G', 'H', 'I', 'J', 'K'])
                elif answer == 'F':
                    error.extend(['A', 'B', 'C', 'D', 'E', 'G', 'H', 'I', 'J', 'K'])
                elif answer == 'G':
                    error.extend(['A', 'B', 'C', 'D', 'E', 'F', 'H', 'I', 'J', 'K'])
                elif answer == 'H':
                    error.extend(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'I', 'J', 'K'])
                elif answer == 'I':
                    error.extend(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K'])
                elif answer == 'J':
                    error.extend(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K'])
                elif answer == 'K':
                    error.extend(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'])
                
                alias = list(set(alias))
                error = list(set(error))

                writer.write(dict(
                    idx=i,
                    meta_info=reader.data_list[i],
                    question=reader.data_list[i]['question'],
                    answer=answer,
                    alias=alias,
                    error=error,
                ))

def mmlu():
    reader_map_list = [
        (MMLUReader(), "mmlu"),
    ]

    for reader, name in reader_map_list:
        with jsonlines.open(f"reference/{name}.jsonl", "w") as writer:
            for i in range(len(reader)):
                answer = str(reader.data_list[i]['answer'])
                alias = [str(reader.data_list[i]['answer']), "\\text{" + reader.data_list[i]['answer'] + "}"]
                error = ['error']
                if answer == 'A':
                    error.extend(['B', 'C', 'D'])
                elif answer == 'B':
                    error.extend(['A', 'C', 'D'])
                elif answer == 'C':
                    error.extend(['A', 'B', 'D'])
                elif answer == 'D':
                    error.extend(['A', 'B', 'C'])
                
                alias = list(set(alias))
                error = list(set(error))

                writer.write(dict(
                    idx=i,
                    meta_info=reader.data_list[i],
                    question=reader.data_list[i]['question'],
                    answer=answer,
                    alias=alias,
                    error=error,
                ))

def gpqa():
    reader_map_list = [
        (GPQAReader(), "gpqa"),
    ]

    for reader, name in reader_map_list:
        with jsonlines.open(f"reference/{name}.jsonl", "w") as writer:
            for i in range(len(reader)):
                answer = str(reader.data_list[i]['answer'])
                alias = [str(reader.data_list[i]['answer']), "\\text{" + reader.data_list[i]['answer'] + "}"]
                error = ['error']
                if answer == 'A':
                    error.extend(['B', 'C', 'D'])
                elif answer == 'B':
                    error.extend(['A', 'C', 'D'])
                elif answer == 'C':
                    error.extend(['A', 'B', 'D'])
                elif answer == 'D':
                    error.extend(['A', 'B', 'C'])
                
                alias = list(set(alias))
                error = list(set(error))

                writer.write(dict(
                    idx=i,
                    # meta_info=reader.data_list[i],
                    question=reader.data_list[i]['question'],
                    answer=answer,
                    alias=alias,
                    error=error,
                ))

if __name__ == '__main__':

    reader_map_list = [
        (GPQAReader(), "gpqa"),
        (MMLUReader(), "mmlu"),
        (BBHReader(), "bbh"),
        (GSM8KReader(), "gsm8k")
    ]

    for reader, name in reader_map_list:
        with jsonlines.open(f"reference/{name}.jsonl", "w") as writer:
            for i in range(len(reader)):
                writer.write(dict(
                    idx=i,
                    meta_info=reader.data_list[i],
                    question=reader.data_list[i]['question'],
                    answer=str(reader.data_list[i]['answer']),
                    alias=[str(reader.data_list[i]['answer']), "\\text{" + reader.data_list[i]['answer'] + "}"],
                    error=['error'],
                ))
    
    mmlu()
    gpqa()
    bbh()





