
import jsonlines
from typing import *
import time
import json
from tqdm import tqdm
from copy import deepcopy
import numpy as np
import torch

IGNORE_LABEL_ID = -100

def str2bool(str_bool:str) -> bool:
    str_bool = str_bool[0].upper() + str_bool[1:].lower()
    return eval(str_bool)

def _print(messages:Any):
    print(f"[{time.ctime()}] {messages}")

def read_jsonl(file_path:str, progress:bool=True) -> List[Dict]:
    _print(f"reading jsonl file from `{file_path}` ...")
    data:List = list()
    if progress:
        pbar = tqdm(total=1)
    with jsonlines.open(file_path, 'r') as reader:
        for item in reader:
            data.append(item)
            if progress:
                pbar.update(1)
    if progress:
        pbar.close()
    return data

def read_json(file_path:str) -> Dict:
    _print(f"reading json file from `{file_path}` ...")
    with open(file_path, 'r') as reader:
        data:dict = json.load(reader)
    return data

def padding_item(
    item:Dict, 
    padding_side:str, 
    label_padding_id:int, 
    input_padding_id:int, 
    max_length:int, 
    position_ids_padding_id:int=None
) -> Dict:
    for key in ['input_ids', 'labels']:
        assert key in item
    assert padding_side in ['left', 'right']
    copy_item = deepcopy(item)

    remainder_for_label = [label_padding_id] * (max_length - len(copy_item['labels']))
    remainder_for_input = [input_padding_id] * (max_length - len(copy_item['input_ids']))
    remainder_for_position_ids = None
    if 'position_ids' in item and position_ids_padding_id != None:
        remainder_for_position_ids = [position_ids_padding_id] * (max_length - len(copy_item['labels']))

    if padding_side == 'left':
        copy_item['input_ids'] = remainder_for_input + copy_item['input_ids']
        copy_item['labels'] = remainder_for_label + copy_item['labels']
        if remainder_for_position_ids:
            copy_item['position_ids'] = remainder_for_position_ids + copy_item['position_ids']
    else:
        copy_item['input_ids'] = copy_item['input_ids'] + remainder_for_input
        copy_item['labels'] = copy_item['labels'] + remainder_for_label
        if remainder_for_position_ids:
            copy_item['position_ids'] = copy_item['position_ids'] + remainder_for_position_ids
    
    return copy_item

def create_attention_for_aug_data(
    input_ids:List[int],
    locate_index_list:List[List[int]],
    # [start, end, n_inst]
    locate_indicator_list:List[str],
    bi_directional:bool,
    see_current:bool,
    diagonal:bool,
    exclude_continue:bool,
    prefill_compress:bool=True,     
    max_length:int=None,
):
    # 因为这里使用了prefill_compress
    # 暂时不支持多轮的那种形式
    # 因为懒，所以我没有计算连续的compressed-prompt，所以默认是从0开始的，从0到compressed_prompt设为下三角

    # 0-False mask
    # 1-True don't mask
    length = len(input_ids)
    mask = np.ones([length, length], dtype=np.bool)
    mask = np.tril(mask)

    assert len(locate_index_list) == len(locate_indicator_list)

    pre_start, pre_end, pre_n_inst, pre_n_comp, pre_n_continue = None, None, None, None, None
    pre_state = None

    # print(locate_index_list)
    # print(locate_indicator_list)
    for index_item, index_state in zip(locate_index_list, locate_indicator_list):
        assert index_state in ['compressed-prompt', 'compressed-output']
        start, end, l_inst, n_comp, n_continue = index_item
        
        # 1. attention_mask
        if exclude_continue:
            mask[end+l_inst+n_comp:, start:end+l_inst] = 0
            if pre_n_continue is not None and pre_n_continue != 0:
                mask[end+l_inst+n_comp:, pre_end+pre_n_inst+pre_n_comp:pre_end+pre_n_inst+pre_n_comp+pre_n_continue] = 0
        else:
            mask[end+l_inst+n_comp:, start:end+l_inst] = 0

        # 1.1 prefill remove compress
        if not prefill_compress and index_state == 'compressed-prompt':
            # 直接设为一个下三角得了
            mask[0:end+l_inst+n_comp, 0:end+l_inst+n_comp] = np.tri(len(mask[0:end+l_inst+n_comp, 0:end+l_inst+n_comp]), dtype=int)
            # print("prefill:", end+l_inst+n_comp)
            
        if not prefill_compress and index_state == 'compressed-output' and pre_state == 'compressed-prompt':
            mask[0:start, 0:start] = np.tri(start, dtype=int)
            # print("last:", start)
        
        # if not prefill_compress and index_state == 'save' and pre_state == 'compressed-prompt':
        #     mask[0:end, 0:end] = np.tri(start, dtype=int)
        
        pre_start, pre_end, pre_n_inst, pre_n_comp, pre_n_continue = start, end, l_inst, n_comp, n_continue

        # if not prefill_compress and index_state == 'save' and pre_state == 'compressed-prompt':
        #     pre_state = 'compressed-prompt'
        # else:
        pre_state = index_state

        # 2. bi_directional
        if bi_directional:
            mask[end+l_inst:end+l_inst+n_comp, end+l_inst:end+l_inst+n_comp] = 1
        
        # 3. diagonal
        if diagonal and end+l_inst+n_comp < len(mask):
            # assert diagonal == False, "因为前面prefill的时候，是会把整个都设为下三角，所以不行，需要改prefill的逻辑"
            # 但是我们目前prompt不进行压缩，所以应该没啥大的问题感觉
            # mask[end+l_inst:end+l_inst+n_comp, end+l_inst:end+l_inst+n_comp] = \
            #     np.eye(n_comp, dtype=int)
            mask[end+l_inst:end+l_inst+n_comp, end+l_inst:end+l_inst+n_comp] = \
                np.eye(len(mask[end+l_inst:end+l_inst+n_comp, end+l_inst:end+l_inst+n_comp]), dtype=int)
        
        # 4. see_current
        if see_current:
            mask[end+l_inst:end+l_inst+n_comp, 0:start] = 0

    # 5. padding
    if max_length is not None and max_length > length:
        padding_mask = np.zeros([max_length, max_length], dtype=np.bool)
        padding_mask[:length, :length] = mask
        padding_mask[length:, :length] = mask[-1:]
        results = torch.as_tensor(padding_mask)
    else:
        results = torch.as_tensor(mask)

    return results

def create_attention_for_recover_data(
    input_ids:List[int],
    locate_index_list:List[List[int]],
    locate_indicator_list:List[str],
    bi_directional:bool,
    see_current:bool,
    diagonal:bool,
    exclude_continue:bool,
    added_input_ids:List[int],
    added_labels:List[int],
    # [end, l_inst, n_comp, n_continnue]
    added_corresp_attention:List[int],
    max_length:int=None,
    prefill_compress:bool=True,
    return_offset:bool=False,
) -> torch.Tensor:
    # 先使用之前的内容，也就是create_attention_for_aug_data
    # 然后我们再去修改后面的部分

    # 0-False mask
    # 1-True don't mask
    length = len(input_ids) + len(added_input_ids)
    mask = np.ones([length, length], dtype=np.bool)
    mask = np.tril(mask)
    
    assert len(locate_index_list) == len(locate_indicator_list)

    # 1. augument part
    pre_start, pre_end, pre_n_inst, pre_n_comp, pre_n_continue = None, None, None, None, None
    pre_state = None
    
    for index_item, index_state in zip(locate_index_list, locate_indicator_list):
        assert index_state in ['compressed-prompt', 'compressed-output']
        start, end, l_inst, n_comp, n_continue = index_item
        
        # 1. attention_mask
        if exclude_continue:
            mask[end+l_inst+n_comp:, start:end+l_inst] = 0
            if pre_n_continue is not None and pre_n_continue != 0:
                mask[end+l_inst+n_comp:, pre_end+pre_n_inst+pre_n_comp:pre_end+pre_n_inst+pre_n_comp+pre_n_continue] = 0
        else:
            mask[end+l_inst+n_comp:, start:end+l_inst] = 0

        # 1.1 prefill remove compress
        if not prefill_compress and index_state == 'compressed-prompt':
            # 直接设为一个下三角得了
            mask[0:end+l_inst+n_comp, 0:end+l_inst+n_comp] = np.tri(len(mask[0:end+l_inst+n_comp, 0:end+l_inst+n_comp]), dtype=int)
        if not prefill_compress and index_state == 'compressed-output' and pre_state == 'compressed-prompt':
            mask[0:start, 0:start] = np.tri(len(mask[0:start, 0:start]), dtype=int)

        pre_start, pre_end, pre_n_inst, pre_n_comp, pre_n_continue = start, end, l_inst, n_comp, n_continue
        pre_state = index_state

        # 2. bi_directional
        if bi_directional:
            mask[end+l_inst:end+l_inst+n_comp, end+l_inst:end+l_inst+n_comp] = 1
        
        # 3. diagonal
        if diagonal:
            assert diagonal == False, "因为前面prefill的时候，是会把整个都设为下三角，所以不行，需要改prefill的逻辑"
            mask[end+l_inst:end+l_inst+n_comp, end+l_inst:end+l_inst+n_comp] = \
                np.eye(len(mask[end+l_inst:end+l_inst+n_comp, end+l_inst:end+l_inst+n_comp]), dtype=int)
        
        # 4. see_current
        if see_current:
            mask[end+l_inst:end+l_inst+n_comp, 0:start] = 0
        
    # 2. recover part
    # 我们这边还是期望说recover部分是连续的
    # 然后我们这边需要关心的是prompt部分的压缩内容
    # 是不能被mask的
    # 目前这里有两种写法
    # 第一种是说我们恢复原来的，就是上三角，然后去设0
    # 第二种是说我们去设1和0
    # 感觉还是使用第二种方式会更加规范一点
    # 但是save的部分的状态是丢失的
    # 这里好像不太行诶，因为answer部分没有被mask现在

    start, end, l_inst, n_comp, n_continue = None, None, None, None, None
    pre_state = None
    for index_item, index_state in zip(locate_index_list, locate_indicator_list):
        assert index_state in ['compressed-prompt', 'compressed-output']
        start, end, l_inst, n_comp, n_continue = index_item
        if index_state == 'compressed-output':
            need_mask = True
            for item in added_corresp_attention:
                _end, _l_inst, _n_comp, _n_continue = item
                if end == _end and l_inst == _l_inst and \
                    n_comp == _n_comp and n_continue == _n_continue:
                    # 刚好命中的话，那么不用动
                    need_mask = False
                    break
            if need_mask:
                # mask掉文本
                mask[
                    len(input_ids):, start:end
                ] = 0
                # mask掉注意力
                mask[
                    len(input_ids):, end:end+l_inst+n_comp+n_continue
                ] = 0

    if start != None:
        if len(input_ids) > end+l_inst+n_comp+n_continue:
            # 把最后结尾的answer还有eos token都mask掉
            mask[len(input_ids):, end+l_inst+n_comp+n_continue:len(input_ids)] = 0
    # 统计一下有多少个token是参与的，用于position_ids的offset
    offset = None
    for i in range(len(input_ids)-1, -1, -1):
        if mask[-1, i] == 1:
            offset = i+1
            break
    
    # 5. padding
    if max_length is not None and max_length > length:
        padding_mask = np.zeros([max_length, max_length], dtype=np.bool)
        padding_mask[:length, :length] = mask
        padding_mask[length:, :length] = mask[-1:]
        results = torch.as_tensor(padding_mask)
    else:
        results = torch.as_tensor(mask)

    if return_offset:
        return results, offset
    else:
        return results

def create_attention_mask(attention_mask_list:List[torch.Tensor], dtype):
    # [bs, 1, max_length, max_length]
    min_dtype = torch.finfo(dtype).min
    for i in range(len(attention_mask_list)):
        attention_mask_list[i] = attention_mask_list[i].to(dtype)
        attention_mask_list[i][torch.where(attention_mask_list[i]==0)] = min_dtype
        attention_mask_list[i][torch.where(attention_mask_list[i]==1)] = 0.
        attention_mask_list[i] = attention_mask_list[i].unsqueeze(dim=0).unsqueeze(dim=1) # [1, 1, max_length, max_length]
    return torch.cat(attention_mask_list, dim=0)

def visualize_attention_mask(
    attention_mask:List[List], 
    input_ids:List[int], 
    tokenizer,
    output_file_name:str="attention_mask4.png",
    position_id:List=None,
    start_offset:int=None,
    end_offset:int=None
):
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import numpy as np
    plt.rcParams['font.size'] = 5

    if start_offset != None:
        attention_mask = np.array(attention_mask)[start_offset:end_offset, start_offset:end_offset].tolist()
        input_ids = input_ids[start_offset:end_offset]

    if position_id == None:
        position_id = list(range(len(input_ids)))
    

    # True -> don't mask
    # False -> mask
    # in dataset.py, the value in attention_mask is 0 and -inf,
    # so if the value is 0, it means we don't mask
    for i in range(len(attention_mask)):
        for j in range(len(attention_mask[i])):
            if attention_mask[i][j] == 0.:
                attention_mask[i][j] = False
            else:
                attention_mask[i][j] = True
    attention_mask = np.array(attention_mask)
    print(input_ids)
    label = ["#"+tokenizer.convert_ids_to_tokens(token_id) for token_id in input_ids] + ['#']
    print(label)
    position_id.append(-1)
    xlabel = ["\n" + l for l in label]
    ylabel = ["\n" + ("" if position_id == None else f"({position_id[idx]})") + l for idx, l in enumerate(label)]

    cmap = mcolors.ListedColormap(['yellow', 'lightgray'])
    plt.imshow(attention_mask, cmap=cmap)
    plt.grid(which='both', color='gray', linestyle='-', linewidth=0.5)
    plt.xticks(np.arange(-.5, len(attention_mask), 1), xlabel, rotation=90)
    plt.yticks(np.arange(-.5, len(attention_mask[0]), 1), ylabel)
    plt.savefig(output_file_name, bbox_inches='tight', pad_inches=0.1)

def visualize_labels(
    input_ids:List[int],
    labels:List[int],
    tokenizer,
    position_ids:List[int]=None
):
    if not isinstance(input_ids, list):
        input_ids = input_ids.tolist()
    if position_ids is None:
        position_ids = list(range(len(input_ids)))
    if not isinstance(labels, list):
        labels = labels.tolist()
    assert len(input_ids) == len(labels)
    results = []
    for input_token_id, label_id, pos_id in zip(input_ids[0:-1], labels[1:], position_ids[0:-1]):
        input_token:str = tokenizer.convert_ids_to_tokens(input_token_id)
        if label_id == -100:
            results.append(f"`{input_token} ({pos_id})` -> `No`")
        else:
            results.append(f"`{input_token} ({pos_id})` -> `{tokenizer.convert_ids_to_tokens(label_id)}`")
    # print("\n".join(results))
    return "\n".join(results)

