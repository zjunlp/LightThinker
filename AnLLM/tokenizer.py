
from transformers import AutoTokenizer
from typing import *
from utils import _print, IGNORE_LABEL_ID
from copy import deepcopy

class Tokenizer:

    def __init__(
        self,
        tokenizer_path:str,
        bos_token:str,
        eos_token:str,
        special_token_list:List[str]=None,
        add_prefix_space:bool=False,
        change_rope:bool=False,
    ):
        self.change_rope:bool = change_rope
        self.tokenizer:AutoTokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            add_prefix_space=add_prefix_space,
            mean_resizing=False
        )
        if special_token_list != None:
            self.add_special_token(special_token_list)
        self.bos_token:str = bos_token
        self.eos_token:str = eos_token
        self.tokenizer.add_eos_token = False
        self.tokenizer.add_bos_token = False
    
    def add_special_token(self, special_token_list:List[str]):
        _print("expanding tokenizer ...")
        num_added_tokens = self.tokenizer.add_tokens(
            special_token_list, special_tokens=True
        )
        assert num_added_tokens == len(special_token_list), f"{special_token_list}"
        _print(f"{num_added_tokens} tokens have been added including {special_token_list}")
        self.bos_token_id = None if self.bos_token == None else self.tokenizer.convert_tokens_to_ids(self.bos_token)
        self.eos_token_id = None if self.eos_token == None else self.tokenizer.convert_tokens_to_ids(self.eos_token)
        if self.eos_token_id is None:
            assert self.eos_token is None
        if self.bos_token_id is None:
            assert self.bos_token is None
        return self.tokenizer

    def __getattr__(self, name):
        return getattr(self.tokenizer, name)
    
    def __len__(self):
        return len(self.tokenizer)

    def normal_data_tokenize(
        self,
        structured_input:List[str],
        max_length:int,
        train_on_input:bool=False,
        check_consistency:bool=False
    ) -> Dict:
        final_item:Dict = dict(
            input_ids=list(),
            labels=list(),
            position_ids=list(),
        )

        tokenized_input_id_list:List = list()
        for segement in structured_input:
            if segement != "":
                tokenized_input_id_list.append(
                    self.tokenizer(
                        segement, return_tensors=None
                    )['input_ids']
                )
            else:
                tokenized_input_id_list.append(list())
        if check_consistency:
            whole_input:str = "".join(structured_input)
            tokenized_whole_input = self.tokenizer(
                whole_input, return_tensors=None
            )['input_ids']
            tokenized_whole_input_from_segement = list()
            for input_ids in tokenized_input_id_list:
                tokenized_whole_input_from_segement.extend(
                    input_ids
                )
            assert tokenized_whole_input_from_segement == tokenized_whole_input, \
                f"consistency check failed.\n{tokenized_whole_input_from_segement}\n{tokenized_whole_input}\n{whole_input}\n{structured_input}"

        tokenized_label_list = deepcopy(tokenized_input_id_list)
        if not train_on_input:
            for i in range(len(tokenized_label_list)):
                if i % 2 == 0:
                    tokenized_label_list[i] = [IGNORE_LABEL_ID] * len(tokenized_label_list[i])

        for input_ids in tokenized_input_id_list:
            final_item['input_ids'].extend(input_ids)
        for labels in tokenized_label_list:
            final_item['labels'].extend(labels)

        final_item['input_ids'] = final_item['input_ids'][0:max_length]
        final_item['labels'] = final_item['labels'][0:max_length]
        final_item['position_ids'] = list(range(len(final_item['input_ids'])))

        return final_item

    def aug_data_tokenize(
        self,
        # [
        #   [],
        #   [],
        # ]
        structured_input:List[List],
        # save, abandoned, compressed 
        structured_input_indicator:List[List[str]],
        n_comp_for_output:int,  
        n_continue_for_output:int,
        n_comp_for_prompt:int,
        n_continue_for_prompt:int,
        mask_label_map:Dict[int, int],
        max_length:int,
        train_on_input:bool=False,
        check_consistency:bool=False,
        recover_mode:bool=False,
    ) -> Tuple[List[Dict], Dict]:

        whole_input = ""
        tokenized_whole_input_from_segement = list()
        
        tokenized_input_id_list:List[List[List[int]]] = list()
        for segement_list in structured_input:
            tokenized_input_id_list.append(list())
            for segement in segement_list:
                if not isinstance(segement, str):
                    assert isinstance(segement, list)
                    tokenized_input_id_list[-1].append(segement)
                    whole_input += self.tokenizer.decode(segement)
                    tokenized_whole_input_from_segement.extend(segement)
                else:
                    tokenized_input_id_list[-1].append(
                        self.tokenizer(segement, return_tensors=None)['input_ids']
                    )
                    tokenized_whole_input_from_segement.extend(
                        tokenized_input_id_list[-1][-1]
                    )
                    whole_input += segement
        if check_consistency:
            tokenized_whole_input = self.tokenizer(
                whole_input, return_tensors=None
            )['input_ids']
            assert tokenized_whole_input_from_segement == tokenized_whole_input, \
                f"consistency check failed.\n{tokenized_whole_input_from_segement}\n{tokenized_whole_input}\n{whole_input}\n{structured_input}\n\n\n\n`{self.tokenizer.decode(tokenized_whole_input_from_segement)}`\n`{self.tokenizer.decode(tokenized_whole_input)}`"

        tokenized_label_list = deepcopy(tokenized_input_id_list)
        if not train_on_input:
            for i in range(len(tokenized_label_list)):
                for j in range(len(tokenized_label_list[i])):
                    if i % 2 == 0:
                        tokenized_label_list[i][j] = [IGNORE_LABEL_ID] * len(tokenized_label_list[i][j])
                    else:
                        for k in range(len(tokenized_label_list[i][j])):
                            if tokenized_label_list[i][j][k] in mask_label_map:
                                tokenized_label_list[i][j][k] = mask_label_map[
                                    tokenized_label_list[i][j][k]
                                ]
        else:
            for i in range(len(tokenized_label_list)):
                for j in range(len(tokenized_label_list[i])):
                    for k in range(len(tokenized_label_list[i][j])):
                            if tokenized_label_list[i][j][k] in mask_label_map:
                                tokenized_label_list[i][j][k] = mask_label_map[
                                    tokenized_label_list[i][j][k]
                                ]

        final_item:Dict = dict(
            input_ids=list(),
            labels=list(),
            locate_index=list(),
            position_ids=list(),
            locate_indicator=list()
        )

        for i in range(len(tokenized_label_list)):
            if len(final_item['input_ids']) >= max_length:
                break
            for j in range(len(tokenized_label_list[i])):
                if len(final_item['input_ids']) >= max_length:
                    break
                assert structured_input_indicator[i][j] in [
                    'save', 'abandoned', 'compressed-prompt', 'compressed-output'
                ]
                if structured_input_indicator[i][j] == 'abandoned':
                    # print(structured_input_indicator[i][j+1])
                    if j+1 < len(structured_input_indicator[i]):

                        assert structured_input_indicator[i][j+1] in ['compressed-prompt', 'compressed-output']
                        n_comp = n_comp_for_prompt if structured_input_indicator[i][j+1] == 'compressed-prompt' else n_comp_for_output
                        n_continue = n_continue_for_prompt if structured_input_indicator[i][j+1] == 'compressed-prompt' else n_continue_for_output
                        assert len(tokenized_input_id_list[i][j+1]) - n_comp - n_continue >= 0
                        if len(tokenized_input_id_list[i][j+1]) - n_comp - n_continue > 0:
                            tokenized_label_list[i][j+1][0:len(tokenized_input_id_list[i][j+1]) - n_comp - n_continue] = [IGNORE_LABEL_ID] * (len(tokenized_input_id_list[i][j+1]) - n_comp - n_continue)
                        final_item['locate_indicator'].append(structured_input_indicator[i][j+1])
                        final_item['locate_index'].append(
                            [
                                len(final_item['input_ids']), 
                                len(final_item['input_ids']) + len(tokenized_input_id_list[i][j]),
                                len(tokenized_input_id_list[i][j+1]) - n_comp - n_continue,
                                n_comp,
                                n_continue
                            ]
                        )
                for k in range(len(tokenized_label_list[i][j])):
                    if len(final_item['input_ids']) >= max_length:
                        break
                    final_item['position_ids'].append(len(final_item['input_ids']))
                    final_item['input_ids'].append(
                        tokenized_input_id_list[i][j][k]
                    )
                    final_item['labels'].append(
                        tokenized_label_list[i][j][k]
                    )
      
        # 4. recover
        recover_item_list:List[Dict] = list()
        if recover_mode:
            delta_length = max_length - len(final_item['input_ids']) - 2
            if delta_length <= 0:
                return recover_item_list, final_item
            for idx, index_tuple in enumerate(final_item['locate_index']):
                start, end, inst_len, n_comp, n_continue = index_tuple
                if final_item['locate_indicator'][idx] == 'compressed-prompt':
                    continue
                else:
                    new_item = dict(
                        input_ids=final_item['input_ids'][start:end],
                        labels=final_item['labels'][start:end],
                        corresp_attention=[end, inst_len, n_comp, n_continue],
                        indicator=final_item['locate_indicator'][idx]
                    )
                    delta_length -= (end-start)
                    if delta_length < 0:
                        break
                    recover_item_list.append(new_item)
    
        return recover_item_list, final_item
        