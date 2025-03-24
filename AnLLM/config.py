

from typing import *
from utils import read_json

class Config:

    @classmethod
    def from_file(cls, config_path:str):
        print(f"loading config from `{config_path}`")
        cfg:dict = read_json(config_path)
        return cls(**cfg)

    def __init__(
        self,
        template:Dict,
        prompt:Dict,
        output:Dict,
        share:bool=True
    ): 
        self.share:bool = share
        self.template_cfg:Dict = template
        self.prompt_cfg:Dict = prompt
        self.output_cfg:Dict = output

        if 'model' not in self.template_cfg:
            self.template_cfg['model'] = 'qwen'

        assert self.template_cfg['model'] in ['qwen', 'llama']

        # assert self.template_cfg['prefix'] + "{question}" + self.template_cfg['suffix'] == self.template_cfg['complete']
        assert self.template_cfg['prefix'] + "{system}" + self.template_cfg['middle'] + "{question}" + self.template_cfg['suffix'] == self.template_cfg['complete']

        self.prompt_save_template:bool = self.prompt_cfg['save_template']
        self.prompt_comp_level:str = self.prompt_cfg['level']
        self.prompt_comp_step:int = self.prompt_cfg['step']
        self.prompt_comp_n_token:int = self.prompt_cfg['n_token']
        self.prompt_comp_token_name_template = self.prompt_cfg['token_name']
        self.prompt_comp_token_desp_template = self.prompt_cfg['token_desp']

        # assert self.prompt_comp_level in ['']
        
        self.output_comp_step:int = self.output_cfg['step']
        self.output_comp_level:bool = self.output_cfg['level']
        self.output_comp_n_token:int = self.output_cfg['n_token']
        self.output_comp_token_name_template:str = self.output_cfg['token_name']
        self.output_comp_token_desp_template:str = self.output_cfg['token_desp']
        self.output_meta_compress_step:int = self.output_cfg['meta_compress_step']

        if self.share:
            assert self.output_comp_token_name_template == self.prompt_comp_token_name_template
            assert self.output_comp_token_desp_template == self.prompt_comp_token_desp_template
        else:
            assert self.output_comp_token_name_template != self.prompt_comp_token_name_template
            assert self.output_comp_token_desp_template != self.prompt_comp_token_desp_template

        self.prompt_comp_token_id_list:List[int] = None
        self.output_comp_token_id_list:List[int] = None
        self.prompt_comp_token_name_list:List[str] = list()
        self.output_comp_token_name_list:List[str] = list()
        self.prompt_comp_token_desp_list:List[str] = list()
        self.output_comp_token_desp_list:List[str] = list()

        self.split_token:str = "<|splitter|>"
        self.split_token_desp:str = "it's time to compress"
        self.split_token_desp:str = "\n\n"
        self.split_token_id:int = None

        self.continue_token:str = "<|continue|>"
        self.continue_token_desp:str = "continue to output according to previous content"
        self.continue_token_id:int = None

        self.recover_token:str = "<|recover|>"
        self.recover_token_desp:str = "recover the token according to previous content"
        self.recover_token_id:int = None

        self.begin_thought_token = "<|begin_of_thought|>"
        self.begin_thought_token_desp:str = "begin of thought"
        self.begin_thought_token_id:int = None

        self.end_thought_token = "<|end_of_thought|>"
        self.end_thought_token_desp:str = "end of thought"
        self.end_thought_token_id:int = None

        self.begin_solution_token = "<|begin_of_solution|>"
        self.begin_solution_token_desp:str = "begin of solution"
        self.begin_solution_token_id:int = None

        self.end_solution_token = "<|end_of_solution|>"
        self.end_solution_token_desp:str = "end of solution"
        self.end_solution_token_id:int = None

        self.double_new_line_token = "\n\n"
        self.double_new_line_token_desp = "\n\n"
        self.double_new_line_token_id:int = None


        self.special_token_name_list:List[str] = [
            self.split_token, 
            self.continue_token, 
            self.recover_token,
            self.begin_thought_token,
            self.end_thought_token,
            self.begin_solution_token,
            self.end_solution_token,
            self.double_new_line_token,
        ]
        self.special_token_desp_list:List[str] = [
            self.split_token_desp, 
            self.continue_token_desp,
            self.recover_token_desp,
            self.begin_thought_token_desp,
            self.end_thought_token_desp,
            self.begin_solution_token_desp,
            self.end_solution_token_desp,
            self.double_new_line_token_desp,
        ]

        # 为qwen2.5
        if self.template_cfg['model'] == 'qwen':
            self.bos_token = "<|im_start|>"
            self.bos_token_desp = "<|im_start|>"
            self.bos_token_id:int = None

            self.eos_token = "<|im_end|>"
            self.eos_token_desp = "<|im_end|>"
            self.eos_token_id:int = None

            self.special_token_name_list.extend(
                [self.eos_token, self.bos_token]
            )
            self.special_token_desp_list.extend(
                [self.eos_token_desp, self.bos_token_desp]
            )

        self.special_token_id_list: List[int] = list()

        for t_id in range(self.prompt_comp_n_token):
            token_name:str = self.prompt_comp_token_name_template.format(t_id=t_id)
            token_desp:str = self.prompt_comp_token_desp_template.format(t_id=t_id)
            self.prompt_comp_token_name_list.append(token_name)
            self.prompt_comp_token_desp_list.append(token_desp)
            self.special_token_name_list.append(token_name)
            self.special_token_desp_list.append(token_desp)

        for t_id in range(self.output_comp_n_token):
            token_name:str = self.output_comp_token_name_template.format(t_id=t_id)
            token_desp:str = self.output_comp_token_desp_template.format(t_id=t_id)
            self.output_comp_token_name_list.append(token_name)
            self.output_comp_token_desp_list.append(token_desp)
            if not self.share:
                self.special_token_name_list.append(token_name)
                self.special_token_desp_list.append(token_desp)

        if self.share:
            assert self.output_comp_token_name_list == self.prompt_comp_token_name_list


    def convert2id(self, tokenizer):
        self.continue_token_id = tokenizer.convert_tokens_to_ids(
            self.continue_token
        )
        self.split_token_id = tokenizer.convert_tokens_to_ids(
            self.split_token
        )
        self.recover_token_id = tokenizer.convert_tokens_to_ids(
            self.recover_token
        )

        self.begin_thought_token_id = tokenizer.convert_tokens_to_ids(
            self.begin_thought_token
        )
        self.end_thought_token_id = tokenizer.convert_tokens_to_ids(
            self.end_thought_token
        )
        self.begin_solution_token_id = tokenizer.convert_tokens_to_ids(
            self.begin_solution_token
        )
        self.end_solution_token_id = tokenizer.convert_tokens_to_ids(
            self.end_solution_token
        )
        self.double_new_line_token_id = tokenizer.convert_tokens_to_ids(
            self.double_new_line_token
        )

        if self.template_cfg['model'] == 'qwen':
            self.bos_token_id = tokenizer.convert_tokens_to_ids(
                self.bos_token
            )
            self.eos_token_id = tokenizer.convert_tokens_to_ids(
                self.eos_token
            )

        self.prompt_comp_token_id_list:List[int] = [
            tokenizer.convert_tokens_to_ids(token) for token in self.prompt_comp_token_name_list
        ]
        self.output_comp_token_id_list:List[int] = [
            tokenizer.convert_tokens_to_ids(token) for token in self.output_comp_token_name_list
        ]

    def get_prompt_comp_token(self, return_list:bool=False) -> Union[str, List[str]]:
        return self.prompt_comp_token_name_list if return_list else "".join(self.prompt_comp_token_name_list)

    def get_output_comp_token(self, return_list:bool=False) -> Union[str, List[str]]:
        return self.output_comp_token_name_list if return_list else "".join(self.output_comp_token_name_list)

    def get_prompt_comp_token_id(self) -> List[int]:
        assert self.prompt_comp_token_id_list is not None
        return self.prompt_comp_token_id_list

    def get_output_comp_token_id(self) -> List[int]:
        assert self.output_comp_token_id_list is not None
        return self.output_comp_token_id_list
    

