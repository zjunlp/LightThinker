import json
import jsonlines
from typing import *
import time
import regex as re

DATASET_PATH = dict(
    mmlu="data/eval/mmlu.json",
    bbh="data/eval/bbh.json",
    gpqa="data/eval/gpqa.json",
    gsm8k="data/eval/gsm8k.json",
)

def _print(messages):
    print(f"[{time.ctime()}] {messages}")

def read_json(file_path:str) -> Dict:
    _print(f"reading json file from `{file_path}` ...")
    with open(file_path, 'r') as reader:
        data:dict = json.load(reader)
    return data

class Reader:
    def __init__(self):
        pass
    
    def get_prompt(self, idx:int) -> str:
        pass

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, idx) -> str:
        return self.get_prompt(idx)

    def extract_answer(self, s):
        pattern = r'\\boxed\{([^{}]*|\{[^{}]*\})*\}'
        
        def match_balanced_braces(text):
            stack = []
            result = []
            for char in text:
                if char == '{':
                    stack.append(char)
                elif char == '}':
                    if stack:
                        stack.pop()
                    else:
                        return None  
                result.append(char)
            return ''.join(result) if not stack else None
        
        matches = re.finditer(pattern, s)
        results = []
        for match in matches:
            content = match.group(0)[7:-1]  
            balanced_content = match_balanced_braces(content)
            if balanced_content:
                results.append(balanced_content)
        
        if len(results) == 0:
            return "error"
        else:
            return results[-1].strip()

    def get_answer(self, idx: int) -> str:
        pass

    def get_acc(self, output, idx) -> bool:
        model_answer = self.extract_answer(output)
        gt_answer = self.get_answer(idx)
        return self.compare_answer(model_answer, gt_answer, idx)

class MMLUReader(Reader):

    file_path = DATASET_PATH['mmlu']
    
    def __init__(self):
        """
        {
            "file_name": [
                {"meta_data": [], "question": "", "answer": "D", "domain": ""}
            ]
        }
        """
        self.meta_db = read_json(MMLUReader.file_path)

        self.data_list:List = list()
        for key in self.meta_db:
            for item in self.meta_db[key]:
                self.data_list.append(item)
                
    def get_prompt(self, idx:int) -> str:
        return "Return your final response within \\boxed{}. " + self.data_list[idx]['question']

    def get_prompt_list(self, idx:int) -> List[str]:
        return ["Return your final response within \\boxed{}."] + self.data_list[idx]['question_list']

    def compare_answer(self, model_answer:str, gt_answer:str, idx:int) -> Tuple[bool, str]:
        if model_answer == "error" or model_answer == "" or model_answer == None:
            left_part = model_answer
            right_part = gt_answer
            return False, f"`{left_part}` <=> `{right_part}`"
        if model_answer[0].lower() in ['a', 'b', 'c', 'd']:
            left_part = model_answer[0].lower()
            right_part = gt_answer.lower()
            return left_part == right_part, f"`{left_part}` <=> `{right_part}`"
        else:
            offset = ord(gt_answer)-ord('A')
            complete_answer = f"{gt_answer}. {self.data_list[idx]['choices_list'][offset]}"
            left_part = model_answer.lower().strip()
            right_part = complete_answer.lower().strip()
            return left_part == right_part, f"`{left_part}` <=> `{right_part}`"
    
    def get_answer(self, idx: int) -> str:
        return self.data_list[idx]['answer']

    def get_system_prompt(self) -> str:
        return "Your role as an assistant involves thoroughly exploring questions through a systematic long thinking process before providing the final precise and accurate solutions. This requires engaging in a comprehensive cycle of analysis, summarizing, exploration, reassessment, reflection, backtracing, and iteration to develop well-considered thinking process. Please structure your response into two main sections: Thought and Solution. In the Thought section, detail your reasoning process using the specified format: <|begin_of_thought|> {thought with steps separated with '\\n\\n'} <|end_of_thought|> Each step should include detailed considerations such as analisying questions, summarizing relevant findings, brainstorming new ideas, verifying the accuracy of the current steps, refining any errors, and revisiting previous steps. In the Solution section, based on various attempts, explorations, and reflections from the Thought section, systematically present the final solution that you deem correct. The solution should remain a logical, accurate, concise expression style and detail necessary step needed to reach the conclusion, formatted as follows: <|begin_of_solution|> {final formatted, precise, and clear solution} <|end_of_solution|> Now, try to solve the following question through the above guidelines:"

    def get_system_prompt_list(self) -> List[str]:
        return [
            "Your role as an assistant involves thoroughly exploring questions through a systematic long thinking process before providing the final precise and accurate solutions.",
            "This requires engaging in a comprehensive cycle of analysis, summarizing, exploration, reassessment, reflection, backtracing, and iteration to develop well-considered thinking process.",
            "Please structure your response into two main sections:",
            "Thought and Solution.",
            "In the Thought section, detail your reasoning process using the specified format:",
            "<|begin_of_thought|> {thought with steps separated with '\\n\\n'} <|end_of_thought|>",
            "Each step should include detailed considerations such as analisying questions, summarizing relevant findings, brainstorming new ideas, verifying the accuracy of the current steps, refining any errors, and revisiting previous steps.",
            "In the Solution section, based on various attempts, explorations, and reflections from the Thought section, systematically present the final solution that you deem correct.",
            "The solution should remain a logical, accurate, concise expression style and detail necessary step needed to reach the conclusion, formatted as follows:",
            "<|begin_of_solution|> {final formatted, precise, and clear solution} <|end_of_solution|>",
            "Now, try to solve the following question through the above guidelines:"
        ]

class BBHReader(Reader):

    file_path = DATASET_PATH['bbh']
    
    def __init__(self):
        """
        {
            "domain": [
                {
                    "meta_data": {
                        "input": "",
                        "target": "",
                        # `structured` is an optional choice
                        "structured": {
                            "question": "",
                            "choices": [
                                "",
                                "",

                            ],
                            "answer_flag": "", # A B C D E
                            "answer_content": ""
                        }
                    },
                    "question": "",
                    "answer": "",
                    "domain": ""
                },
            ]
        }
        """
        self.meta_db = read_json(BBHReader.file_path)
        self.data_list:List = list()
        for key in self.meta_db:
            for item in self.meta_db[key]:
                self.data_list.append(item)
    
    def is_multiple_choices_question(self, idx) -> bool:
        return "structured" in self.data_list[idx]["meta_data"]

    def get_prompt(self, idx:int) -> str:
        return "Return your final response within \\boxed{}. " + self.data_list[idx]['question']


    def get_prompt_list(self, idx:int) -> List[str]:
        return ["Return your final response within \\boxed{}."] + self.data_list[idx]['question_list']

    def compare_answer(self, model_answer:str, gt_answer:str, idx:int) -> Tuple[bool, str]:
        is_multi_choices:bool = self.is_multiple_choices_question(idx)
        if model_answer == "error":
            left_part = model_answer
            right_part = gt_answer
            return False, f"`{left_part}` <=> `{right_part}`"
        if not is_multi_choices:
            # is not a multiple-choice question
            left_part = model_answer.lower().strip()
            right_part = gt_answer.lower().strip()
            return left_part == right_part , f"`{left_part}` <=> `{right_part}`"
        else:
            if model_answer[0].lower() in [chr(ord('A')+i) for i in range(len(self.data_list[idx]["meta_data"]['structured']['choices']))]:
                left_part = model_answer[0].lower()
                right_part = gt_answer.lower()
                return left_part == right_part, f"`{left_part}` <=> `{right_part}`"
            else:
                offset = ord(gt_answer) - ord('A')
                complete_answer = f"{gt_answer}. {self.data_list[idx]['meta_data']['structured']['choices'][offset]}"
                left_part = model_answer.lower().strip()
                right_part = complete_answer.lower().strip()
                return left_part == right_part, f"`{left_part}` <=> `{right_part}`"
    
    def get_answer(self, idx: int) -> str:
        return self.data_list[idx]['answer']

    def get_system_prompt(self) -> str:
        return "Your role as an assistant involves thoroughly exploring questions through a systematic long thinking process before providing the final precise and accurate solutions. This requires engaging in a comprehensive cycle of analysis, summarizing, exploration, reassessment, reflection, backtracing, and iteration to develop well-considered thinking process. Please structure your response into two main sections: Thought and Solution. In the Thought section, detail your reasoning process using the specified format: <|begin_of_thought|> {thought with steps separated with '\\n\\n'} <|end_of_thought|> Each step should include detailed considerations such as analisying questions, summarizing relevant findings, brainstorming new ideas, verifying the accuracy of the current steps, refining any errors, and revisiting previous steps. In the Solution section, based on various attempts, explorations, and reflections from the Thought section, systematically present the final solution that you deem correct. The solution should remain a logical, accurate, concise expression style and detail necessary step needed to reach the conclusion, formatted as follows: <|begin_of_solution|> {final formatted, precise, and clear solution} <|end_of_solution|> Now, try to solve the following question through the above guidelines:"

    def get_system_prompt_list(self) -> List[str]:
        return [
            "Your role as an assistant involves thoroughly exploring questions through a systematic long thinking process before providing the final precise and accurate solutions.",
            "This requires engaging in a comprehensive cycle of analysis, summarizing, exploration, reassessment, reflection, backtracing, and iteration to develop well-considered thinking process.",
            "Please structure your response into two main sections:",
            "Thought and Solution.",
            "In the Thought section, detail your reasoning process using the specified format:",
            "<|begin_of_thought|> {thought with steps separated with '\\n\\n'} <|end_of_thought|>",
            "Each step should include detailed considerations such as analisying questions, summarizing relevant findings, brainstorming new ideas, verifying the accuracy of the current steps, refining any errors, and revisiting previous steps.",
            "In the Solution section, based on various attempts, explorations, and reflections from the Thought section, systematically present the final solution that you deem correct.",
            "The solution should remain a logical, accurate, concise expression style and detail necessary step needed to reach the conclusion, formatted as follows:",
            "<|begin_of_solution|> {final formatted, precise, and clear solution} <|end_of_solution|>",
            "Now, try to solve the following question through the above guidelines:"
        ]

class GSM8KReader(Reader):

    file_path = DATASET_PATH['gsm8k']

    def __init__(self):
        """
        {
            "math": [
                {
                    "meta_data": {
                        "question": "",
                        "answer": ""
                    },
                    "question": "",
                    "answer": "",
                }
            ]
        }
        """
        self.meta_db = read_json(GSM8KReader.file_path)
        self.data_list:List = list()
        for key in self.meta_db:
            for item in self.meta_db[key]:
                self.data_list.append(item)

    def get_prompt(self, idx:int) -> str:
        return "Return your final response within \\boxed{}. " + self.data_list[idx]['question']

    def get_prompt_list(self, idx:int) -> List[str]:
        return ["Return your final response within \\boxed{}.", self.data_list[idx]['question']]

    def compare_answer(self, model_answer:str, gt_answer:str, idx:int) -> bool:
        if model_answer == "error":
            left_part = model_answer
            right_part = gt_answer
            return False,  f"`{left_part}` <=> `{right_part}`"
        else:
            match = re.findall(r'\\boxed{(.*?)}', model_answer)
            if match:
                left_part = match[-1].strip().lower()
            else:
                left_part = model_answer.strip().lower()
            right_part = gt_answer.strip().lower()
            left_part = left_part.replace(",", "")
            right_part = right_part.replace(",", "")
            return left_part == right_part, f"`{left_part}` <=> `{right_part}`"
        
    def get_answer(self, idx: int) -> str:
        return self.data_list[idx]['answer']

    def get_system_prompt(self) -> str:
        return "Your role as an assistant involves thoroughly exploring questions through a systematic long thinking process before providing the final precise and accurate solutions. This requires engaging in a comprehensive cycle of analysis, summarizing, exploration, reassessment, reflection, backtracing, and iteration to develop well-considered thinking process. Please structure your response into two main sections: Thought and Solution. In the Thought section, detail your reasoning process using the specified format: <|begin_of_thought|> {thought with steps separated with '\\n\\n'} <|end_of_thought|> Each step should include detailed considerations such as analisying questions, summarizing relevant findings, brainstorming new ideas, verifying the accuracy of the current steps, refining any errors, and revisiting previous steps. In the Solution section, based on various attempts, explorations, and reflections from the Thought section, systematically present the final solution that you deem correct. The solution should remain a logical, accurate, concise expression style and detail necessary step needed to reach the conclusion, formatted as follows: <|begin_of_solution|> {final formatted, precise, and clear solution} <|end_of_solution|> Now, try to solve the following question through the above guidelines:"

    def get_system_prompt_list(self) -> List[str]:
        return [
            "Your role as an assistant involves thoroughly exploring questions through a systematic long thinking process before providing the final precise and accurate solutions.",
            "This requires engaging in a comprehensive cycle of analysis, summarizing, exploration, reassessment, reflection, backtracing, and iteration to develop well-considered thinking process.",
            "Please structure your response into two main sections:",
            "Thought and Solution.",
            "In the Thought section, detail your reasoning process using the specified format:",
            "<|begin_of_thought|> {thought with steps separated with '\\n\\n'} <|end_of_thought|>",
            "Each step should include detailed considerations such as analisying questions, summarizing relevant findings, brainstorming new ideas, verifying the accuracy of the current steps, refining any errors, and revisiting previous steps.",
            "In the Solution section, based on various attempts, explorations, and reflections from the Thought section, systematically present the final solution that you deem correct.",
            "The solution should remain a logical, accurate, concise expression style and detail necessary step needed to reach the conclusion, formatted as follows:",
            "<|begin_of_solution|> {final formatted, precise, and clear solution} <|end_of_solution|>",
            "Now, try to solve the following question through the above guidelines:"
        ]

class GPQAReader(Reader):
    
    file_path = DATASET_PATH['gpqa']

    def __init__(self):
        """
        {
            "math": [
                {
                    "meta_data": [],
                    "question": "",
                    "answer": "",           
                    "choice_list": [
                        "", "", "", ""
                    ],
                    "answer_content": "",   
                    "pure_question": ""
                }
            ]
        }
        """
        self.meta_db = read_json(GPQAReader.file_path)
        self.data_list:List = list()
        for key in self.meta_db:
            for item in self.meta_db[key]:
                self.data_list.append(item)

    def get_prompt(self, idx:int) -> str:
        return "Return your final response within \\boxed{}. " + self.data_list[idx]['question']
    
    def get_prompt_list(self, idx:int) -> List[str]:
        return ["Return your final response within \\boxed{}."] + self.data_list[idx]['question_list']

    def compare_answer(self, model_answer:str, gt_answer:str, idx:int) -> Tuple[bool, str]:
        if model_answer == "error":
            left_part = model_answer
            right_part = gt_answer
            return False, f"`{left_part}` <=> `{right_part}`"
        if model_answer[0].lower() in ['a', 'b', 'c', 'd']:
            left_part = model_answer[0].lower()
            right_part = gt_answer.lower()
            return left_part == right_part, f"`{left_part}` <=> `{right_part}`"
        else:
            offset = ord(gt_answer)-ord('A')
            complete_answer = f"{gt_answer}. {self.data_list[idx]['choices_list'][offset]}"
            left_part = model_answer.lower().strip()
            right_part = complete_answer.lower().strip()
            return left_part == right_part, f"`{left_part}` <=> `{right_part}`"
    
    def get_answer(self, idx: int) -> str:
        return str(self.data_list[idx]['answer'])

    def get_system_prompt(self) -> str:
        return "Your role as an assistant involves thoroughly exploring questions through a systematic long thinking process before providing the final precise and accurate solutions. This requires engaging in a comprehensive cycle of analysis, summarizing, exploration, reassessment, reflection, backtracing, and iteration to develop well-considered thinking process. Please structure your response into two main sections: Thought and Solution. In the Thought section, detail your reasoning process using the specified format: <|begin_of_thought|> {thought with steps separated with '\\n\\n'} <|end_of_thought|> Each step should include detailed considerations such as analisying questions, summarizing relevant findings, brainstorming new ideas, verifying the accuracy of the current steps, refining any errors, and revisiting previous steps. In the Solution section, based on various attempts, explorations, and reflections from the Thought section, systematically present the final solution that you deem correct. The solution should remain a logical, accurate, concise expression style and detail necessary step needed to reach the conclusion, formatted as follows: <|begin_of_solution|> {final formatted, precise, and clear solution} <|end_of_solution|> Now, try to solve the following question through the above guidelines:"

    def get_system_prompt_list(self) -> List[str]:
        return [
            "Your role as an assistant involves thoroughly exploring questions through a systematic long thinking process before providing the final precise and accurate solutions.",
            "This requires engaging in a comprehensive cycle of analysis, summarizing, exploration, reassessment, reflection, backtracing, and iteration to develop well-considered thinking process.",
            "Please structure your response into two main sections:",
            "Thought and Solution.",
            "In the Thought section, detail your reasoning process using the specified format:",
            "<|begin_of_thought|> {thought with steps separated with '\\n\\n'} <|end_of_thought|>",
            "Each step should include detailed considerations such as analisying questions, summarizing relevant findings, brainstorming new ideas, verifying the accuracy of the current steps, refining any errors, and revisiting previous steps.",
            "In the Solution section, based on various attempts, explorations, and reflections from the Thought section, systematically present the final solution that you deem correct.",
            "The solution should remain a logical, accurate, concise expression style and detail necessary step needed to reach the conclusion, formatted as follows:",
            "<|begin_of_solution|> {final formatted, precise, and clear solution} <|end_of_solution|>",
            "Now, try to solve the following question through the above guidelines:"
        ]
