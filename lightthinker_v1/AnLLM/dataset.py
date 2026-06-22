
import torch
from typing import List, Dict, Tuple, Union, Optional
from tqdm import tqdm
from copy import deepcopy

from config import Config
from tokenizer import Tokenizer
from utils import create_attention_mask
from utils import visualize_labels, visualize_attention_mask
from utils import _print, read_jsonl, IGNORE_LABEL_ID, padding_item
from utils import create_attention_for_aug_data, create_attention_for_recover_data

class MyDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        file_path:str,
        config:Config,
        tokenizer: Tokenizer,
        padding_config: Dict,
        train_on_input:bool,
        change_rope: bool=False,
        output_compress_instruction:str=None
    ):
        """
        padding_config['max_length']
        """
        self.check_consistency = False
        self.train_on_input:bool = train_on_input
        self.change_rope = change_rope
        self.meta_data:List[Dict] = read_jsonl(
            file_path=file_path, progress=True
        )
        self.config:Config = config
        self.tokenizer:Tokenizer = tokenizer
        self.padding_config:Dict = padding_config
        
        # upper bound
        self.normal_data:List[Dict] = list()
        # without recover data
        self.aug_data:List[Dict] = list()
        # with recover data
        self.recover_data:List[Dict] = list()
        # prompt recover mode
        self.recover_prompt_data:List[Dict] = list()    
        # 
        self.aug_data_wo_prompt_comp:List[Dict] = list()

        self.output_compress_instruction:str = output_compress_instruction if output_compress_instruction != None else ""

        self.init()
        self.init_for_aug_data_wo_pc()
    
    def insert_comp_for_output(self, output:str) -> Tuple[List[str], List[List[int]]]:
        assert self.config.output_comp_level == 'token'
        output_input_ids = self.tokenizer.tokenizer(
            output, return_tensors=None
        )['input_ids']
        input_ids_list:List[List[int]] = list()
        indicator_list:List[str] = list()
        step = self.config.output_comp_step

        comp_cnt = len(output_input_ids) // step 
        for i in range(comp_cnt):
            input_ids_list.append(
                output_input_ids[i*step: (i+1)*step]
            )
            indicator_list.append("abandoned")

            input_ids_list.append(
                self.output_compress_instruction + self.config.get_output_comp_token(return_list=False) # + self.config.continue_token
            )
            indicator_list.append("compressed-output")
        if comp_cnt * step < len(output_input_ids):
            input_ids_list.append(
                output_input_ids[comp_cnt * step:len(output_input_ids)]
            )
            indicator_list.append("abandoned")
            
        return indicator_list, input_ids_list


    def insert_comp_for_prompt(
        self, 
        question:str, 
        question_list:List[str], 
        system_prompt:str, 
        system_prompt_list:List[str]
    ) -> Tuple[
        List[str],
        List[List[int]]
    ]:
        prefix_input_ids = None
        middle_input_ids = None
        suffix_input_ids = None
        indicator_list:List[str] = list()
        input_ids_list:List[List[int]] = list()
        if self.config.prompt_save_template:
            prefix_input_ids = self.tokenizer.tokenizer(
                self.tokenizer.bos_token + self.config.template_cfg['prefix'],
                return_tensors=None
            )['input_ids']
            middle_input_ids = self.tokenizer.tokenizer(
                self.config.template_cfg['middle']
            )['input_ids']
            suffix_input_ids = self.tokenizer.tokenizer(
                self.config.template_cfg['suffix']
            )['input_ids']
        else:
            if not self.train_on_input:
                suffix_input_ids = [self.config.continue_token_id]
        if self.config.prompt_comp_level == 'token':
            step = self.config.prompt_comp_step
            if not self.config.prompt_save_template:
                prompt = self.config.template_cfg['prefix'] + \
                    system_prompt + self.config.template_cfg['middle'] + question + self.config.template_cfg['suffix']
                input_ids = [self.tokenizer.bos_token_id] + self.tokenizer.tokenizer(
                    prompt, return_tensors=None
                )['input_ids']
                
                for i in range(0, len(input_ids), step):
                    # ub = i+step
                    indicator_list.append("abandoned")
                    input_ids_list.append(input_ids[i:i+step])
                    indicator_list.append("compressed-prompt")
                    input_ids_list.append(self.config.get_prompt_comp_token_id())
                
                if suffix_input_ids != None:
                    indicator_list.append("save")
                    input_ids_list.append(suffix_input_ids)
                
            else:
                system_input_ids = self.tokenizer.tokenizer(system_prompt, return_tensors=None)['input_ids']
                prompt_input_ids = self.tokenizer.tokenizer(question, return_tensors=None)['input_ids']
                if prefix_input_ids != None:
                    indicator_list.append("save")
                    input_ids_list.append(prefix_input_ids)
                
                # ub = 0
                for i in range(0, len(system_input_ids), step):
                    # ub = i+step
                    indicator_list.append("abandoned")
                    input_ids_list.append(system_input_ids[i:i+step])
                    indicator_list.append("compressed-prompt")
                    input_ids_list.append(self.config.get_prompt_comp_token_id())

                if middle_input_ids != None:
                    indicator_list.append("save")
                    input_ids_list.append(middle_input_ids)
                # ub = 0
                for i in range(0, len(prompt_input_ids), step):
                    # ub = i+step
                    indicator_list.append("abandoned")
                    input_ids_list.append(prompt_input_ids[i:i+step])
                    indicator_list.append("compressed-prompt")
                    input_ids_list.append(self.config.get_prompt_comp_token_id())

                if suffix_input_ids != None:
                    indicator_list.append("save")
                    input_ids_list.append(suffix_input_ids)
        elif self.config.prompt_comp_level == 'sentence':
            step = self.config.prompt_comp_step
            assert step == 1
            question_input_ids_list:List[List[int]] = list()
            system_prompt_input_ids_list:List[List[int]] = list()

            if not self.config.prompt_save_template:
                system_prompt_list[0] = self.config.template_cfg['prefix'] + system_prompt_list[0]
                system_prompt_list[-1] = system_prompt_list[-1] + self.config.template_cfg['middle']
                question_list[-1] = question_list[-1] + self.config.template_cfg['suffix']
                for q in question_list:
                    question_input_ids_list.append(
                        self.tokenizer.tokenizer(q, return_tensors=None)['input_ids']
                    )
                for s in system_prompt_list:
                    system_prompt_input_ids_list.append(
                        self.tokenizer.tokenizer(s, return_tensors=None)['input_ids']
                    )
                
                sentence_input_ids_list:List[List[int]] = system_prompt_input_ids_list + question_input_ids_list
                for i in range(0, len(sentence_input_ids_list)):
                    indicator_list.append("abandoned")
                    input_ids_list.append(sentence_input_ids_list[i])
                    indicator_list.append("compressed-prompt")
                    input_ids_list.append(self.config.get_prompt_comp_token_id())
            else:
                for q in question_list:
                    question_input_ids_list.append(
                        self.tokenizer.tokenizer(q, return_tensors=None)['input_ids']
                    )
                for s in system_prompt_list:
                    system_prompt_input_ids_list.append(
                        self.tokenizer.tokenizer(s, return_tensors=None)['input_ids']
                    )

                if prefix_input_ids != None:
                    indicator_list.append("save")
                    input_ids_list.append(prefix_input_ids)
                
                for i in range(0, len(system_prompt_input_ids_list)):
                    indicator_list.append("abandoned")
                    input_ids_list.append(system_prompt_input_ids_list[i])
                    indicator_list.append("compressed-prompt")
                    input_ids_list.append(self.config.get_prompt_comp_token_id())
                
                if middle_input_ids != None:
                    indicator_list.append("save")
                    input_ids_list.append(middle_input_ids)
                
                for i in range(0, len(question_input_ids_list)):
                    indicator_list.append("abandoned")
                    input_ids_list.append(question_input_ids_list[i])
                    indicator_list.append("compressed-prompt")
                    input_ids_list.append(self.config.get_prompt_comp_token_id())
                
                if suffix_input_ids != None:
                    indicator_list.append("save")
                    input_ids_list.append(suffix_input_ids)
        
        assert len(indicator_list) == len(input_ids_list)
        return indicator_list, input_ids_list

    def init(self):
        _print("initializing the dataset ...")
        self.config.convert2id(self.tokenizer)

        pbar = tqdm(total=len(self.meta_data))
        for item in self.meta_data:
            assert isinstance(item, dict)
            new_item = dict(
                meta_info=item,
                tokenized=dict()
            )

            question:str = item['question']
            system_prompt:str = item['system_prompt']
            system_prompt_list:List[str] = item['system_list']
            question_list:List[str] = item['question_list']
            thoughts_list:List[str] = item['thoughts_list']
            gt_output:str = item['gt_output']

            add_eos:bool = True
            if 'add_eos' in item:
                add_eos:bool = item['add_eos']

            if len(thoughts_list) == 0 and len(question_list) == 0 and len(system_prompt_list) == 0:
                structured_input = [
                    self.tokenizer.bos_token + self.config.template_cfg['complete'].format(system=system_prompt, question=question),
                    gt_output + self.tokenizer.eos_token
                ]
            else:
                structured_input = [
                    self.tokenizer.bos_token + self.config.template_cfg['complete'].format(system=system_prompt, question=question),
                    gt_output + self.tokenizer.eos_token
                ]
                
            self.normal_data.append(
                dict(
                    tokenized=self.tokenizer.normal_data_tokenize(
                        structured_input=structured_input,
                        max_length=self.padding_config['max_length'],
                        train_on_input=self.train_on_input,
                        check_consistency=self.check_consistency,
                    ),
                    meta_info=item,
                )
            )

            # 'save', 'abandoned', 'compressed-prompt', 'compressed-output'
            structured_input:List[List] = list()
            structured_input_indicator:List[List[str]] = list()
            mask_label_map:Dict = dict()
            mask_label_map[self.config.recover_token_id] = IGNORE_LABEL_ID
            mask_label_map[self.config.continue_token_id] = IGNORE_LABEL_ID

            # 1. insert
            prompt_indicator_list, prompt_input_ids_list = self.insert_comp_for_prompt(
                question=question, question_list=question_list, 
                system_prompt=system_prompt, system_prompt_list=system_prompt_list
            )
            structured_input.append(prompt_input_ids_list)          # [n_turn, n_sent, n_token_per_sent]
            structured_input_indicator.append(prompt_indicator_list)    # [n_turn, n_sent]
            

            # 1.1 recover prompt
            assert len(prompt_input_ids_list) == len(prompt_indicator_list)
            recover_prompt_input_ids = [self.config.recover_token_id]
            for input_ids, indicator in zip(prompt_input_ids_list, prompt_indicator_list):
                if indicator == 'abandoned':
                    recover_prompt_input_ids.extend(input_ids)
            recover_prompt_input_ids.append(self.tokenizer.eos_token_id)
            _, recover_prompt_item = self.tokenizer.aug_data_tokenize(
                structured_input=structured_input + [[recover_prompt_input_ids]],
                structured_input_indicator=structured_input_indicator + [["save"]],
                n_comp_for_prompt=self.config.prompt_comp_n_token,  
                n_continue_for_prompt=0,
                n_comp_for_output=self.config.output_comp_n_token,
                n_continue_for_output=0,
                mask_label_map=mask_label_map,
                max_length=self.padding_config['max_length'],
                train_on_input=self.train_on_input,
                check_consistency=self.check_consistency,
                recover_mode=False,
            )
            self.recover_prompt_data.append(
                dict(
                    meta_info=item,
                    tokenized=recover_prompt_item,
                )
            )

            # 2. output part
            if self.config.output_comp_level == 'token':
                output_indicator_list, output_content_list = self.insert_comp_for_output(
                    gt_output + self.tokenizer.eos_token if add_eos else gt_output
                )
            else:
                # sentence level
                output_indicator_list = list()
                output_content_list = list()
                for thought in thoughts_list:
                    output_indicator_list.append("abandoned")
                    output_content_list.append(thought + '\n') # + self.config.split_token)
                    output_indicator_list.append("compressed-output")
                    output_content_list.append(self.output_compress_instruction + self.config.get_output_comp_token(return_list=False)) # + self.config.continue_token)
                if add_eos:
                    output_indicator_list.append("save")
                    output_content_list.append(self.tokenizer.eos_token)
            
            structured_input.append(output_content_list)
            structured_input_indicator.append(output_indicator_list)

            recover_item, aug_item = self.tokenizer.aug_data_tokenize(
                structured_input=structured_input,
                structured_input_indicator=structured_input_indicator,
                n_comp_for_prompt=self.config.prompt_comp_n_token,  
                n_continue_for_prompt=0,
                n_comp_for_output=self.config.output_comp_n_token,
                n_continue_for_output=0,
                mask_label_map=mask_label_map,
                max_length=self.padding_config['max_length'],
                train_on_input=self.train_on_input,
                check_consistency=self.check_consistency,
                recover_mode=True,
            )
            self.aug_data.append(
                dict(
                    meta_info=item,
                    tokenized=aug_item,
                )
            )
            self.recover_data.append(
                recover_item
            )
            pbar.update(1)

    def init_for_aug_data_wo_pc(self, system_compression:bool=False):
        pbar = tqdm(total=len(self.meta_data))
        for item in self.meta_data:
            assert isinstance(item, dict)
            new_item = dict(
                meta_info=item,
                tokenized=dict()
            )

            question:str = item['question']
            system_prompt:str = item['system_prompt']
            system_prompt_list:List[str] = item['system_list']
            question_list:List[str] = item['question_list']
            thoughts_list:List[str] = item['thoughts_list']
            gt_output:str = item['gt_output']
            add_eos:bool=True
            if 'add_eos' in item:
                add_eos:bool = item['add_eos']
            # we will not use the system_prompt_list and question_list

            structured_input:List[List] = list()
            structured_input_indicator:List[List[str]] = list()
            mask_label_map:Dict = dict()
            mask_label_map[self.config.recover_token_id] = IGNORE_LABEL_ID
            mask_label_map[self.config.continue_token_id] = IGNORE_LABEL_ID

            # 1. prompt part
            if system_compression == False:
                structured_input.append(
                    [
                        self.tokenizer.bos_token + \
                            self.config.template_cfg['complete'].format(
                                system=system_prompt, question=question
                            )
                    ]
                )
                structured_input_indicator.append(
                    ["save"]
                )
            else:
                assert False, "we don't support it."

            # 2.output
            if self.config.output_comp_level == 'token':
                output_indicator_list, output_content_list = self.insert_comp_for_output(
                    gt_output + self.tokenizer.eos_token if add_eos else gt_output
                )
            else:
                # sentence level
                output_indicator_list = list()
                output_content_list = list()
                for thought in thoughts_list:
                    output_indicator_list.append("abandoned")
                    output_content_list.append(thought + '\n') # + self.config.split_token)
                    output_indicator_list.append("compressed-output")
                    output_content_list.append(self.output_compress_instruction + self.config.get_output_comp_token(return_list=False)) # + self.config.continue_token)
                if add_eos:
                    output_indicator_list.append("save")
                    output_content_list.append(self.tokenizer.eos_token)
            
            structured_input.append(output_content_list)
            structured_input_indicator.append(output_indicator_list)

            recover_item, aug_item = self.tokenizer.aug_data_tokenize(
                structured_input=structured_input,
                structured_input_indicator=structured_input_indicator,
                n_comp_for_prompt=self.config.prompt_comp_n_token,  
                n_continue_for_prompt=0,
                n_comp_for_output=self.config.output_comp_n_token,
                n_continue_for_output=0,
                mask_label_map=mask_label_map,
                max_length=self.padding_config['max_length'],
                train_on_input=self.train_on_input,
                check_consistency=self.check_consistency,
                recover_mode=True,
            )

            self.aug_data_wo_prompt_comp.append(
                dict(
                    meta_info=item,
                    tokenized=aug_item
                )
            )

    def __getitem__(self, idx:int) -> Tuple[Dict, Dict, Dict, Dict, Dict]:
        return (
            self.normal_data[idx],
            self.aug_data[idx],
            self.recover_data[idx],
            self.recover_prompt_data[idx],
            self.aug_data_wo_prompt_comp[idx]
            # None
        )
    
    def __len__(self) -> int:
        return len(self.normal_data)
    
class MyDataCollator:

    def __init__(
        self, 
        dataset:MyDataset,
        attention_config:Dict,
        exclude_continue:bool,
        sample_config:Dict,
    ):
        """
        attention_config:dict(
            diagonal=False,
            bi_directional=False,
            see_current=False,
            prefill_compress=True,  #
        )
        sample_config:dict(
            mode:str in [aug, normal]
            hybrid:bool=False,      
            
        )
        """
        self.dataset:MyDataset = dataset
        self.attention_config:Dict = attention_config
        self.exclude_continue:bool = exclude_continue
        self.sample_config = sample_config


    def _normal_mode(self, instances:List[Tuple]) -> Dict:
        final = dict(
            input_ids=list(),
            labels=list(),
        )
        for bsz_id, instance in enumerate(instances):
            normal_data, _, _, _, _ = instance
            new_item = padding_item(
                item=normal_data['tokenized'],
                padding_side=self.dataset.padding_config['padding_side'],
                label_padding_id=self.dataset.padding_config['label_padding_id'], 
                input_padding_id=self.dataset.padding_config['input_padding_id'], 
                max_length=self.dataset.padding_config['max_length'], 
                position_ids_padding_id=self.dataset.padding_config['position_ids_padding_id']
            )
            # print(new_item['input_ids'])
            final['input_ids'].append(new_item['input_ids'])
            final['labels'].append(new_item['labels'])
        
        # print(final['input_ids'])
        return dict(
            input_ids=torch.as_tensor(final['input_ids']),
            labels=torch.as_tensor(final['labels']),
        )
            
    def _aug_mode(self, instances:List[Tuple]) -> Dict:
        final = dict(
            input_ids=list(),
            labels=list(),
            attention_mask=list(),
            position_ids=list(),
            row_comp_index=list(),
            column_comp_index=list(),
        )

        # self.normal_data[idx],
        # self.aug_data[idx],
        # self.recover_data[idx]

        for bsz_id, instance in enumerate(instances):
            _, aug_data, _, _, _ = instance
            final['attention_mask'].append(
                create_attention_for_aug_data(
                    input_ids=aug_data['tokenized']['input_ids'],
                    locate_index_list=aug_data['tokenized']['locate_index'],
                    # [start, end, n_inst]
                    locate_indicator_list=aug_data['tokenized']['locate_indicator'],
                    bi_directional=self.attention_config['bi_directional'],
                    see_current=self.attention_config['see_current'],
                    diagonal=self.attention_config['diagonal'],
                    exclude_continue=self.exclude_continue,
                    max_length=self.dataset.padding_config['max_length'],
                    prefill_compress=self.attention_config['prefill_compress'],
                )
            )
            new_item = padding_item(
                item=aug_data['tokenized'],
                padding_side=self.dataset.padding_config['padding_side'],
                label_padding_id=self.dataset.padding_config['label_padding_id'], 
                input_padding_id=self.dataset.padding_config['input_padding_id'], 
                max_length=self.dataset.padding_config['max_length'], 
                position_ids_padding_id=self.dataset.padding_config['position_ids_padding_id']
            )
            final['input_ids'].append(new_item['input_ids'])
            final['labels'].append(new_item['labels'])
            final['position_ids'].append(new_item['position_ids'])
            # row and column
            temp_column_list = list()
            temp_row_list = list()
            for item in aug_data['tokenized']['locate_index']:
                start, end, l_inst, n_comp, n_continue = item
                temp_column_list.extend(
                    [end+l_inst+i for i in range(n_comp)]
                )
                temp_row_list.extend([bsz_id] * n_comp)
                final['row_comp_index'].extend(temp_row_list)
                final['column_comp_index'].extend(temp_column_list)
        
        return dict(
            input_ids=torch.as_tensor(
                final['input_ids']
            ),
            labels=torch.as_tensor(
                final['labels']
            ),
            attention_mask=create_attention_mask(
                final['attention_mask'],
                dtype=torch.bfloat16
            ),
            position_ids=torch.as_tensor(
                final['position_ids']
            ),
            row_comp_index=torch.as_tensor(
                final['row_comp_index']
            ),
            column_comp_index=torch.as_tensor(
                final['column_comp_index']
            ),
        )
            
    def _recover_mode(self, instances:List[Tuple]) -> Dict:
        final = dict(
            input_ids=list(),
            labels=list(),
            attention_mask=list(),
            position_ids=list(),
            row_comp_index=list(),
            column_comp_index=list(),
        )

        for bsz_id, instance in enumerate(instances):
            _, aug_data, recover_data, _, _ = instance
            if len(recover_data) == 0:
                final['attention_mask'].append(
                    create_attention_for_aug_data(
                        input_ids=aug_data['tokenized']['input_ids'],
                        locate_index_list=aug_data['tokenized']['locate_index'],
                        # [start, end, n_inst]
                        locate_indicator_list=aug_data['tokenized']['locate_indicator'],
                        bi_directional=self.attention_config['bi_directional'],
                        see_current=self.attention_config['see_current'],
                        diagonal=self.attention_config['diagonal'],
                        exclude_continue=self.exclude_continue,
                        max_length=self.dataset.padding_config['max_length'],
                        prefill_compress=self.attention_config['prefill_compress'],
                    )
                )
                new_item = padding_item(
                    item=aug_data['tokenized'],
                    padding_side=self.dataset.padding_config['padding_side'],
                    label_padding_id=self.dataset.padding_config['label_padding_id'], 
                    input_padding_id=self.dataset.padding_config['input_padding_id'], 
                    max_length=self.dataset.padding_config['max_length'], 
                    position_ids_padding_id=self.dataset.padding_config['position_ids_padding_id']
                )
                final['input_ids'].append(new_item['input_ids'])
                final['labels'].append(new_item['labels'])
                final['position_ids'].append(new_item['position_ids'])

            else:
                new_input_ids = [self.dataset.config.recover_token_id]
                new_labels = [IGNORE_LABEL_ID]
                corresp_attention = list()
                for item in recover_data:
                    new_input_ids.extend(item['input_ids'])
                    new_labels.extend(item['labels'])
                    corresp_attention.append(item['corresp_attention'])
                    assert item['indicator'] == 'compressed-output'
                new_input_ids.append(self.dataset.tokenizer.eos_token_id)
                new_labels.append(self.dataset.tokenizer.eos_token_id)
                
                attention_mask, pos_offset = create_attention_for_recover_data(
                    input_ids=aug_data['tokenized']['input_ids'],
                    locate_index_list=aug_data['tokenized']['locate_index'],
                    # [start, end, n_inst]
                    locate_indicator_list=aug_data['tokenized']['locate_indicator'],
                    bi_directional=self.attention_config['bi_directional'],
                    see_current=self.attention_config['see_current'],
                    diagonal=self.attention_config['diagonal'],
                    exclude_continue=self.exclude_continue,
                    max_length=self.dataset.padding_config['max_length'],
                    added_input_ids=new_input_ids,
                    added_labels=new_labels,
                    added_corresp_attention=corresp_attention,
                    return_offset=True,
                    prefill_compress=self.attention_config['prefill_compress'],
                )
                final['attention_mask'].append(attention_mask)
                new_position_ids = [i+pos_offset for i in range(len(new_input_ids))]
                assert len(aug_data['tokenized']['input_ids'] + new_input_ids) == len(aug_data['tokenized']['position_ids'] + new_position_ids)
                
                new_item = padding_item(
                    item=dict(
                        input_ids=aug_data['tokenized']['input_ids'] + new_input_ids,
                        labels=aug_data['tokenized']['labels'] + new_labels,
                        position_ids=aug_data['tokenized']['position_ids'] + new_position_ids,
                    ),
                    padding_side=self.dataset.padding_config['padding_side'],
                    label_padding_id=self.dataset.padding_config['label_padding_id'], 
                    input_padding_id=self.dataset.padding_config['input_padding_id'], 
                    max_length=self.dataset.padding_config['max_length'], 
                    position_ids_padding_id=self.dataset.padding_config['position_ids_padding_id']
                )

                
                final['input_ids'].append(new_item['input_ids'])
                final['labels'].append(new_item['labels'])
                final['position_ids'].append(new_item['position_ids'])

            temp_column_list = list()
            temp_row_list = list()
            for item in aug_data['tokenized']['locate_index']:
                start, end, l_inst, n_comp, n_continue = item
                temp_column_list.extend(
                    [end+l_inst+i for i in range(n_comp)]
                )
                temp_row_list.extend([bsz_id] * n_comp)
                final['row_comp_index'].extend(temp_row_list)
                final['column_comp_index'].extend(temp_column_list)

        return dict(
            input_ids=torch.as_tensor(
                final['input_ids']
            ),
            labels=torch.as_tensor(
                final['labels']
            ),
            attention_mask=create_attention_mask(
                final['attention_mask'],
                dtype=torch.bfloat16
            ),
            position_ids=torch.as_tensor(
                final['position_ids']
            ),
            row_comp_index=torch.as_tensor(
                final['row_comp_index']
            ),
            column_comp_index=torch.as_tensor(
                final['column_comp_index']
            ),
        )

    def _prompt_recover_mode(self, instances:List[Tuple]) -> Dict:
        final = dict(
            input_ids=list(),
            labels=list(),
            attention_mask=list(),
            position_ids=list(),
            row_comp_index=list(),
            column_comp_index=list(),
        )

        for bsz_id, instance in enumerate(instances):
            _, _, _, recover_prompt_data, _ = instance
            final['attention_mask'].append(
                create_attention_for_aug_data(
                    input_ids=recover_prompt_data['tokenized']['input_ids'],
                    locate_index_list=recover_prompt_data['tokenized']['locate_index'],
                    # [start, end, n_inst]
                    locate_indicator_list=recover_prompt_data['tokenized']['locate_indicator'],
                    bi_directional=self.attention_config['bi_directional'],
                    see_current=self.attention_config['see_current'],
                    diagonal=self.attention_config['diagonal'],
                    exclude_continue=self.exclude_continue,
                    max_length=self.dataset.padding_config['max_length'],
                    prefill_compress=self.attention_config['prefill_compress'],
                )
            )
            new_item = padding_item(
                item=recover_prompt_data['tokenized'],
                padding_side=self.dataset.padding_config['padding_side'],
                label_padding_id=self.dataset.padding_config['label_padding_id'], 
                input_padding_id=self.dataset.padding_config['input_padding_id'], 
                max_length=self.dataset.padding_config['max_length'], 
                position_ids_padding_id=self.dataset.padding_config['position_ids_padding_id']
            )
            final['input_ids'].append(new_item['input_ids'])
            final['labels'].append(new_item['labels'])
            final['position_ids'].append(new_item['position_ids'])
            # row and column
            temp_column_list = list()
            temp_row_list = list()
            for item in recover_prompt_data['tokenized']['locate_index']:
                start, end, l_inst, n_comp, n_continue = item
                temp_column_list.extend(
                    [end+l_inst+i for i in range(n_comp)]
                )
                temp_row_list.extend([bsz_id] * n_comp)
                final['row_comp_index'].extend(temp_row_list)
                final['column_comp_index'].extend(temp_column_list)
        
        return dict(
            input_ids=torch.as_tensor(
                final['input_ids']
            ),
            labels=torch.as_tensor(
                final['labels']
            ),
            attention_mask=create_attention_mask(
                final['attention_mask'],
                dtype=torch.bfloat16
            ),
            position_ids=torch.as_tensor(
                final['position_ids']
            ),
            row_comp_index=torch.as_tensor(
                final['row_comp_index']
            ),
            column_comp_index=torch.as_tensor(
                final['column_comp_index']
            ),
        )

    def _aug_mode_wo_pc(self, instances:List[Tuple]) -> Dict:
        final = dict(
            input_ids=list(),
            labels=list(),
            attention_mask=list(),
            position_ids=list(),
            row_comp_index=list(),
            column_comp_index=list(),
        )
        for bsz_id, instance in enumerate(instances):
            _, _, _, _, aug_data = instance
            final['attention_mask'].append(
                create_attention_for_aug_data(
                    input_ids=aug_data['tokenized']['input_ids'],
                    locate_index_list=aug_data['tokenized']['locate_index'],
                    # [start, end, n_inst]
                    locate_indicator_list=aug_data['tokenized']['locate_indicator'],
                    bi_directional=self.attention_config['bi_directional'],
                    see_current=self.attention_config['see_current'],
                    diagonal=self.attention_config['diagonal'],
                    exclude_continue=self.exclude_continue,
                    max_length=self.dataset.padding_config['max_length'],
                    prefill_compress=False,
                )
            )
            new_item = padding_item(
                item=aug_data['tokenized'],
                padding_side=self.dataset.padding_config['padding_side'],
                label_padding_id=self.dataset.padding_config['label_padding_id'], 
                input_padding_id=self.dataset.padding_config['input_padding_id'], 
                max_length=self.dataset.padding_config['max_length'], 
                position_ids_padding_id=self.dataset.padding_config['position_ids_padding_id']
            )
            final['input_ids'].append(new_item['input_ids'])
            final['labels'].append(new_item['labels'])
            final['position_ids'].append(new_item['position_ids'])
            # row and column
            temp_column_list = list()
            temp_row_list = list()
            for item in aug_data['tokenized']['locate_index']:
                start, end, l_inst, n_comp, n_continue = item
                temp_column_list.extend(
                    [end+l_inst+i for i in range(n_comp)]
                )
                temp_row_list.extend([bsz_id] * n_comp)
                final['row_comp_index'].extend(temp_row_list)
                final['column_comp_index'].extend(temp_column_list)
        
        return dict(
            input_ids=torch.as_tensor(
                final['input_ids']
            ),
            labels=torch.as_tensor(
                final['labels']
            ),
            attention_mask=create_attention_mask(
                final['attention_mask'],
                dtype=torch.bfloat16
            ),
            position_ids=torch.as_tensor(
                final['position_ids']
            ),
            row_comp_index=torch.as_tensor(
                final['row_comp_index']
            ),
            column_comp_index=torch.as_tensor(
                final['column_comp_index']
            ),
        )

    def __call__(self, instances:List[Tuple]) -> Dict:
        # 'aug', 'normal', 'recover'
        if self.sample_config['mode'] == 'aug':
            data = self._aug_mode(instances)
        elif self.sample_config['mode'] == 'recover':
            data = self._recover_mode(instances)
        elif self.sample_config['mode'] == 'normal':
            return self._normal_mode(instances)
        elif self.sample_config['mode'] == 'aug-wo-pc':
            return self._aug_mode_wo_pc(instances)
        else:
            assert False
        if self.sample_config['hybrid'] == True:
            prompt_data = self._prompt_recover_mode(instances)
            return self._merge_dict(data, prompt_data)
        else:
            return data
    
    def _merge_dict(self, dict1, dict2) -> Dict:

        return dict(
            input_ids = torch.cat(
                (dict1['input_ids'], dict2['input_ids']), 
                dim=0
            ),
            labels=torch.cat(
                (dict1['labels'], dict2['labels']),
                dim=0
            ),
            attention_mask=torch.cat(
                (dict1['attention_mask'], dict2['attention_mask']),
                dim=0,
            ),
            position_ids=torch.cat(
                (dict1['position_ids'], dict2['position_ids']),
                dim=0,
            ),
            row_comp_index=torch.cat(
                (dict1['row_comp_index'], dict2['row_comp_index'] + len(dict1['input_ids'])),
                dim=0,
            ),
            column_comp_index=torch.cat(
                (dict1['column_comp_index'], dict2['column_comp_index']),
                dim=0,
            )
        )

        
if __name__ == '__main__':
    

    config_path = ".json"
    tokenizer_path = ""
    # bos_token="<|begin_of_text|>"
    # eos_token="<|eot_id|>"
    bos_token="<|im_start|>"
    eos_token="<|im_end|>"
    dataset_path = ".jsonl"
    train_on_input = False
    exclude_continue = False

    config:Config = Config.from_file(config_path=config_path)
    # special_token_list:List[str] = config.special_token_name_list
    special_token_list:List[str] = list()

    tokenizer:Tokenizer = Tokenizer(
        tokenizer_path=tokenizer_path,
        bos_token=bos_token,
        eos_token=eos_token,
        special_token_list=None,
        add_prefix_space=False,
        change_rope=False,        
    )
    for token in config.special_token_name_list:
        if tokenizer.convert_tokens_to_ids(token) == None:
            special_token_list.append(token)
    if len(special_token_list) > 0:
        tokenizer.add_special_token(special_token_list)

    padding_config = dict(
        padding_side='right',
        label_padding_id=-100,
        input_padding_id=tokenizer.eos_token_id,
        max_length=75,
        position_ids_padding_id=0,
    )
    attention_config = dict(
        diagonal=False,         
        bi_directional=False,
        see_current=True,
        prefill_compress=False,
    )
    sample_config = dict(
        mode=['aug', 'normal', 'recover', 'aug-wo-pc'][-1],
        hybrid=False
    )

    dataset = MyDataset(
        file_path=dataset_path,
        config=config,
        tokenizer=tokenizer,
        padding_config=padding_config,
        train_on_input=train_on_input,
        change_rope=False,
        output_compress_instruction=""
    )

    data_collator = MyDataCollator(
        dataset=dataset,
        attention_config=attention_config,
        exclude_continue=exclude_continue,
        sample_config=sample_config,
    )

    batch = data_collator([dataset[0], dataset[1]])
    bsz_id = -1
    # qwen: bos/eos: 128000, 128009
    # print(batch['input_ids'][bsz_id].tolist())
    # exit()
    # print(batch['row_comp_index'])
    if 'attention_mask' in batch:
        visualize_attention_mask(
            attention_mask=batch['attention_mask'][bsz_id, 0].tolist(), 
            input_ids=batch['input_ids'][bsz_id].tolist(), 
            tokenizer=tokenizer,
            position_id=None,
            # start_offset=None,
            # end_offset=None,
            # start_offset=-50,
            # end_offset=None,
            start_offset=0,
            end_offset=50,
        )
    else:
        print("no attention")

    print(
        visualize_labels(
            input_ids=batch['input_ids'][bsz_id].tolist(), 
            labels=batch['labels'][bsz_id].tolist(), 
            tokenizer=tokenizer,
            position_ids=batch['position_ids'][bsz_id].tolist() if 'position_ids' in batch else None
        )
    )

