
import time
import argparse
import jsonlines
from typing import *
from tqdm import tqdm

from utils import _print, read_jsonl
from LightThinker import Tokenizer, Config
from constant import CLASS_NORMAL, CLASS_CACHE, CLASS_ANCHOR, CLASS_TOKEN

class Reference:

    PATH_TEMPLATE = "reference/{name}.jsonl"

    def __init__(self, dataset_name:str, interaction:bool):
        self.dataset_name = dataset_name
        print(self.PATH_TEMPLATE.format(name=dataset_name))
        self.meta_data_list:List[Dict] = read_jsonl(
            self.PATH_TEMPLATE.format(name=dataset_name)
        )
        self.interaction:bool = interaction
    
    def _digit_compare(self, model_ans:str, gt_ans:str):
        if model_ans.isdigit() and gt_ans.isdigit():
            if model_ans != gt_ans:
                return False
        
        for item in [model_ans, gt_ans]:
            for i in item:
                if i not in "0123456789.-+":
                    return None
        try:
            if eval(model_ans) == eval(gt_ans):
                print(f"`{model_ans}` == {gt_ans}")
                return True
            else:
                print(f"`{model_ans}` != {gt_ans}")
                return False
        except:
            return None

        return None

    def compare(self, idx:int, model_ans:str, model_output:str) -> bool:
        if model_ans in self.meta_data_list[idx]['alias']:
            return True
        elif model_ans in self.meta_data_list[idx]['error']:
            return False
        else:
            _state = self._digit_compare(model_ans, self.meta_data_list[idx]['answer'])
            if _state in [True, False]:
                return _state
            # input(str(_state))
            if not self.interaction:
                return False
            print(f"does `{model_ans}` equal to `{self.meta_data_list[idx]['answer']}`: ")
            while True:
                results = input()
                if results.lower() in ['y', 'yes']:
                    self.meta_data_list[idx]['alias'].append(model_ans)
                    return True
                elif results.lower() in ['n', 'no']:
                    self.meta_data_list[idx]['error'].append(model_ans)
                    return False
                elif results.lower() in ['e']:
                    print(model_output)
                    print(f"does `{model_ans}` equal to `{self.meta_data_list[idx]['answer']}`: ")
                else:
                    print("Please enter yes or no!")
    
    def get_domain(self, idx:int):
        if self.dataset_name == 'bbh':
            return self.meta_data_list[idx]['meta_info']['domain']
        elif self.dataset_name == 'mmlu':
            return self.meta_data_list[idx]['meta_info']['domain']
    
    def update(self):
        _print(f"updating {self.PATH_TEMPLATE.format(name=self.dataset_name)} ...")
        with jsonlines.open(self.PATH_TEMPLATE.format(name=self.dataset_name), 'w') as writer:
            pbar = tqdm(total=len(self.meta_data_list))
            for item in self.meta_data_list:
                pbar.update(1)
                writer.write(item)
            pbar.close()

class Evaluator:

    def __init__(
        self,
        args,
        comp_config:Config,
        tokenizer:Tokenizer,
        splitter_token_id:int,
        file_name_list:List[str],
        reference:Reference,
    ):
        self.args = args
        self.comp_config:Config = comp_config
        self.tokenizer:Tokenizer = tokenizer
        self.reference:Reference = reference
        self.splitter_token_id:int = splitter_token_id
        self.splitter:str = self.tokenizer.tokenizer.convert_ids_to_tokens(
            self.splitter_token_id
        )
        _print(f"The splitter is `{self.splitter}`")
        
        self.eval_data_list:List[Dict] = list()
        for file_path in file_name_list:
            data_list = read_jsonl(file_path)
            for item in data_list:
                self.eval_data_list.append(item)

    def _cal_attend(
        self, 
        len_prompt:int, 
        len_output:int, 
        cache_size:int, 
        output_id_list:List[int],
    ) -> Tuple[int, List[int]]:

        frq_list:List[int] = list()

        if self.args.method in (CLASS_ANCHOR + CLASS_TOKEN):
            cnt = 0
            cur_context = list()
            cache_size = 0
            cmp_cnt = 0
            if self.args.method == 'anchor':
                offset = 0
            else:
                # continue token
                offset = 1
            # print(self.compress_cnt)
            for i in range(len(output_id_list)):
                cur_token_id = output_id_list[i]
                cur_context.append(cur_token_id)
                cnt += (len_prompt + len(cur_context) + cache_size)
                if self.args.method in CLASS_ANCHOR:
                    if cur_token_id == self.splitter_token_id:
                        cmp_cnt += 1
                        frq_list.append(len(cur_context))
                        cur_context.clear()
                        cache_size = (self.compress_cnt + offset) * cmp_cnt
                elif self.args.method in CLASS_TOKEN:
                    if (i+1) % self.comp_config.output_comp_step == 0:
                        cmp_cnt += 1
                        assert len(cur_context) == self.comp_config.output_comp_step
                        cur_context.clear()
                        cache_size = (self.compress_cnt + offset) * cmp_cnt
                else:
                    assert False
            # input(cnt)
            return cnt, frq_list
        elif self.args.method in CLASS_CACHE:
            return (cache_size + len_prompt) * (cache_size - len_prompt - 1) / 2 + (len_output+len_prompt-cache_size) * cache_size, frq_list
        elif self.args.method in CLASS_NORMAL:
            return len_prompt * len_output + (1+len_output)*len_output/2, frq_list
        else:
            assert False

    def avg(self, l:List[int], scale:int=1, rounds:int=2) -> float:
        if len(l) == 0:
            return round(0, rounds)
        return round(sum(l) / len(l) * scale, rounds)

    def metrics2str(self, metrics:Dict) -> str:
        if len(metrics['compress_list']) != 0:
            with jsonlines.open(f'frequency/{self.args.method}-{self.args.model_type}-{self.args.dataset}.jsonl', 'w') as writer:
                writer.write(dict(
                    value=metrics['compress_list']
                ))
        result_list = [
            (f'{"="*30} {self.args.dataset} {"="*30}'),
            (f"acc: {metrics['correct']}/{metrics['total']}"),
            (f"acc: {round(metrics['correct']/metrics['total']*100, 2)}"),
            (f"time: {round(sum(metrics['infer_time_list'])/3600, 2)}"),
            (f"attend: {self.avg(metrics['attend_list'])}"),
            (f"output len: {self.avg(metrics['output_len_list'])}"),
            (f"total len: {self.avg(metrics['total_len_list'])}"),
            (f"peak mem: {self.avg(metrics['peak_mem_list'])}"),
            (f"compress: {self.avg(metrics['compress_cnt_list'])}"),
            (f"{round(metrics['correct']/metrics['total']*100, 2)},{round(sum(metrics['infer_time_list'])/3600, 2)},{int(round(self.avg(metrics['attend_list']),0))},{int(round(self.avg(metrics['output_len_list']), 0))},{int(round(self.avg(metrics['peak_mem_list']),0))},{int(round(self.avg(metrics['compress_cnt_list']),0))}"),
        ]
        return "\n".join(result_list)   

    def print_metrics(self, metrics:Dict):
        print(self.metrics2str(metrics))

    def _cal_max_token(self, output_id_list:List[int], len_prompt:int) -> int:
        max_token = 0
        cur_context = list()
        cache_size = 0
        cmp_cnt = 0
        if self.args.method == 'anchor':
            offset = 0
        else:
            offset = 1
        for i in range(len(output_id_list)):
            cur_token_id = output_id_list[i]
            cur_context.append(cur_token_id)
            if cur_token_id == self.splitter_token_id:
                cmp_cnt += 1
                max_token = max(max_token, len(cur_context) + self.compress_cnt + len_prompt + cache_size + 1)
                cache_size = cmp_cnt * (self.compress_cnt + offset)
                cur_context.clear()
        # print(output_id_list.count(self.splitter_token_id))
        # print(cmp_cnt)
        # print(self.compress_cnt)
        max_token = max(max_token, len(cur_context) + len_prompt + cache_size)
        return max_token

    def eval(self, return_metrics:bool):
        metrics = dict(
            total=0,
            correct=0,
            infer_time_list=list(),
            peak_mem_list=list(),
            attend_list=list(),
            compress_cnt_list=list(),
            prompt_len_list=list(),
            output_len_list=list(),
            total_len_list=list(),
            compress_list=list(),      
        )

        pbar = tqdm(total=len(self.eval_data_list))
        for item in self.eval_data_list:
            pbar.update(1)

            metrics['total'] += 1

            if 'old_output' in item:
                model_output_key = 'old_output'
            elif 'output' in item:
                model_output_key = 'output'
            elif 'model_output' in item:
                model_output_key = 'model_output'
            else:
                assert False

            # 1. Accuracy
            acc_state = self.reference.compare(
                idx=item['idx'],
                model_ans=item['model_answer'],
                model_output=item[model_output_key] + ("\n" + item['prompt'] if 'prompt' in item else "")
            )
            if acc_state == True:
                metrics['correct'] += 1

            # 2. Token and Peak Memory
            prompt:str = item['prompt']
            output:str = item[model_output_key]
            prompt_len:int = len(self.tokenizer.tokenizer(prompt, return_tensors=None)['input_ids'])
            output_id_list:List[int] = self.tokenizer.tokenizer(output, return_tensors=None)['input_ids']
            output_len = len(output_id_list)
            # assert prompt_len == (item['prompt_len'] if 'prompt_len' in item else item['input_len']),\
            #     f"{prompt_len} != {(item['prompt_len'] if 'prompt_len' in item else item['input_len'])}"
            # print(output_len, [item['output_len'], prompt_len, item['output_len'] + prompt_len])
            # assert output_len in [item['output_len'], item['output_len'] + prompt_len], \
            #     f"{output_len} in {[item['output_len'], item['output_len'] + prompt_len]}"

            if self.args.method in CLASS_CACHE:
                metrics['peak_mem_list'].append(self.args.cache_size)
            elif self.args.method in (CLASS_ANCHOR + CLASS_TOKEN):
                if 'max_token' not in item:
                    metrics['peak_mem_list'].append(
                        self._cal_max_token(output_id_list, prompt_len)
                    )
                else:
                    # print(f"{self._cal_max_token(output_id_list, prompt_len)} != {item['max_token']}")
                    # input("hh")
                    # assert self._cal_max_token(output_id_list, prompt_len) == item['max_token'], \
                    #     f"{self._cal_max_token(output_id_list, prompt_len)} != {item['max_token']}"
                    metrics['peak_mem_list'].append(item['max_token'])
            elif self.args.method in CLASS_NORMAL:
                metrics['peak_mem_list'].append(output_len+prompt_len)
            else:
                assert False
            

            if self.args.method in CLASS_ANCHOR:
                output_len = output_len - output.count(self.splitter)
            metrics['prompt_len_list'].append(prompt_len)
            metrics['output_len_list'].append(output_len)
            metrics['total_len_list'].append(output_len+prompt_len)

            # 3. Attend (i.e., dependency)
            attend, frequency = self._cal_attend(
                len_prompt=prompt_len, 
                len_output=output_len, 
                cache_size=self.args.cache_size, 
                output_id_list=output_id_list,
            )
            metrics['attend_list'].append(attend)
            if len(frequency) != 0:
                metrics['compress_list'].extend(frequency)

            # 4. Infer_time
            metrics['infer_time_list'].append(item['infer_time'])

            # 5. Compress Cnt
            if self.args.method in CLASS_ANCHOR:
                metrics['compress_cnt_list'].append(output.count(self.splitter))
            if self.args.method in CLASS_TOKEN:
                metrics['compress_cnt_list'].append(len(output_id_list) // self.comp_config.output_comp_step)

        if not return_metrics:
            self.print_metrics(metrics)
        else:
            return self.metrics2str(metrics)

def get_parser():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument(
        '--method', 
        type=str, 
        choices=['anchor-token', 'normal', 'kvcache', 'anchor-thought']
    )

    parser.add_argument('--tokenizer_path', type=str)
    parser.add_argument('--comp_config', type=str)

    parser.add_argument('--model_type', type=str, choices=['llama', 'qwen'])
    parser.add_argument('--dataset', type=str, choices=['bbh', 'gsm8k', 'gpqa', 'mmlu'])
    parser.add_argument('--files', nargs='+', help='List of files')
    parser.add_argument('--interaction', action="store_true")

    # Only used for h2o and sep
    parser.add_argument('--cache_size', type=int, default=None)

    parser.add_argument('--eos_token', type=str)
    parser.add_argument('--bos_token', type=str)

    args = parser.parse_args()
    return args


def get_tokenizer_and_config(args) -> Tuple[Tokenizer, Config]:

    comp_config:Config = Config.from_file(
        config_path=args.comp_config
    )

    special_token_list:List[str] = list()

    tokenizer:Tokenizer = Tokenizer(
        tokenizer_path=args.tokenizer_path if args.tokenizer_path != None else model_path,
        bos_token=args.bos_token,
        eos_token=args.eos_token,
        special_token_list=None,
        add_prefix_space=False
    )

    for token in comp_config.special_token_name_list:
        if tokenizer.convert_tokens_to_ids(token) == None:
            special_token_list.append(token)
    if len(special_token_list) > 0:
        tokenizer.add_special_token(special_token_list)

    comp_config.convert2id(tokenizer)

    return tokenizer, comp_config

def print_interaction_mode_instruction(args):
    if args.interaction == True:
        print("#"*20, "Use the human evaluation mode.", "#"*20)
        print("""When string matching fails, the output will be displayed in the format "Model Answer" <=> "Standard Answer". At this point, you can input "y" or "n" to evaluate this case. If you believe the model's answer extraction is incorrect, you can input "e" to print the model's complete output, and then input "y" or "n" to evaluate this case.""")
    else:
        print("#"*20, "Use the automatic evaluation mode.", "#"*20)


def main():
    args = get_parser()
    tokenizer, comp_config = get_tokenizer_and_config(args)

    file_list:List[str] = args.files
    reference:Reference = Reference(
        dataset_name=args.dataset, interaction=args.interaction
    )

    print_interaction_mode_instruction(args)
    evaluator:Evaluator = Evaluator(
        args=args,
        comp_config=comp_config,
        tokenizer=tokenizer,
        splitter_token_id=comp_config.split_token_id if args.method not in (CLASS_ANCHOR + CLASS_TOKEN) else comp_config.output_comp_token_id_list[-1],
        file_name_list=file_list,
        reference=reference,
    )
    print(evaluator.eval(return_metrics=True))
    

if __name__ == '__main__':
    main()

