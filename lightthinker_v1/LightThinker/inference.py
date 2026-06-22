
import os
import time
import torch
import argparse
import jsonlines
import numpy as np
from typing import *
from tqdm import tqdm
from copy import deepcopy
from transformers import AutoTokenizer, DynamicCache, GenerationConfig

from LightThinker.utils import *
from config import Config
from tokenizer import Tokenizer
from model_llama import LlamaForCausalLM
from model_qwen import Qwen2ForCausalLM
from dataset_reader import GPQAReader, MMLUReader, BBHReader, GSM8KReader, Reader

DEBUG:bool=False
BLOCK:bool=False
TIMER:bool=True
_SAVE:str = "save"
_COMP_OUT:str = "compressed-output"
_COMP_PMP:str = "compressed-prompt"
_ABANDONED:str = "abandoned"
INDICATOR_LIST:List[str] = [
    _SAVE,
    _COMP_OUT,
    _COMP_PMP,
    _ABANDONED
]

class DebugUtils:

    @classmethod
    def show_global_attention(
        cls, 
        tokenizer:Tokenizer,
        attention_mask:List[List[Union[bool, float]]], 
        input_ids:List[int],
        position_ids:List[int]=None,
        block:bool=False,
        start_offset:int=None,
        end_offset:int=None,
        file_name:str="debug_global.png",
    ):
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        import numpy as np
        plt.rcParams['font.size'] = 5

        if position_ids == None:
            position_ids = list(range(len(input_ids)))

        if start_offset != None:
            attention_mask = np.array(
                attention_mask
            )[start_offset:end_offset, start_offset:end_offset]
            input_ids = input_ids[start_offset:end_offset]
            if position_ids != None:
                position_ids = position_ids[start_offset:end_offset]
        
        # True -> don't mask
        # False -> mask
        for i in range(len(attention_mask)):
            for j in range(len(attention_mask[i])):
                if isinstance(attention_mask[i][j], float):
                    if attention_mask[i][j] == 0.:
                        attention_mask[i][j] = True
                    else:
                        attention_mask[i][j] = False
        attention_mask = np.array(attention_mask)
        # print(attention_mask)
        label = [
            '# ' + tokenizer.convert_ids_to_tokens(token_id) for token_id in input_ids
        ] + ['# PlaceHolder']
        position_ids.append(-1)
        xlabel = ['\n' + l for l in label]
        ylabel = ["\n" + ("" if position_ids == None else f"({position_ids[idx]})") + l for idx, l in enumerate(label)]
        
        cmap = mcolors.ListedColormap(['lightgray', 'yellow'])
        plt.imshow(attention_mask, cmap=cmap)
        plt.grid(which='both', color='gray', linestyle='-', linewidth=0.5)
        plt.xticks(np.arange(-.5, len(attention_mask[0]), 1), xlabel, rotation=90)
        plt.yticks(np.arange(-.5, len(attention_mask), 1), ylabel)
        plt.savefig(file_name, bbox_inches='tight', pad_inches=0.1)
        if block:
            input("Blocking! Please enter the enter to finish blocking ...")

    @classmethod
    def show_local_attention(
        cls,
        tokenizer:Tokenizer,
        attention_mask:List[List[Union[bool, float]]],
        input_ids:List[int],
        position_ids:List[int]=None,
        block:bool=False,
        start_offset:int=None,
        end_offset:int=None,
        file_name:str="debug_local.png",
    ):
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        import numpy as np
        plt.rcParams['font.size'] = 10
        if position_ids == None:
            position_ids = list(range(len(input_ids)))

        y_input_ids:List[int] = input_ids[-len(attention_mask):]
        y_position_ids:List[int] = position_ids[-len(attention_mask):]

        if start_offset != None:
            attention_mask = np.array(
                attention_mask
            )[:, start_offset:end_offset]
            input_ids = input_ids[start_offset:end_offset]
            if position_ids != None:
                position_ids = position_ids[start_offset:end_offset]
        
        # True -> don't mask
        # False -> mask
        for i in range(len(attention_mask)):
            for j in range(len(attention_mask[i])):
                if isinstance(attention_mask[i][j], float):
                    if attention_mask[i][j] == 0.:
                        attention_mask[i][j] = True
                    else:
                        attention_mask[i][j] = False
        attention_mask = np.array(attention_mask)

        position_ids.append(-1)
        xlabel = [
            '\n# ' + tokenizer.convert_ids_to_tokens(token_id) + f"({position_ids[idx]})" for idx, token_id in enumerate(input_ids)
        ] + ['# PlaceHolder (-1)']
        ylabel = [
            '\n# ' + tokenizer.convert_ids_to_tokens(token_id) + f"({y_position_ids[idx]})" for idx, token_id in enumerate(y_input_ids)
        ] + ['# PlaceHolder (-1)']

        cmap = mcolors.ListedColormap(['lightgray', 'yellow'])
        plt.imshow(attention_mask, cmap=cmap, aspect='auto', vmin=0, vmax=1)
        plt.grid(which='both', color='gray', linestyle='-', linewidth=0.5)
        plt.xticks(np.arange(-.5, len(attention_mask[0]), 1), xlabel, rotation=90)
        plt.yticks(np.arange(-.5, len(attention_mask), 1), ylabel)
        plt.savefig(file_name, bbox_inches='tight', pad_inches=0.1)
        if block:
            input("Blocking! Please enter the enter to finish blocking ...")

class InferenceUtils:

    @classmethod
    def get_predicted_token_ids(cls, model_output, idx:int=-1) -> int:
        # [bs, seq_length, vocab_size]
        logits = model_output.logits    
        # [vocab_size]
        target_logits = logits[0, idx, :]
        predicted_token_ids: int = torch.argmax(target_logits).item()
        return predicted_token_ids

class AttentionUtils:

    def __init__(
        self, 
        max_length:int, 
        device:str, 
        dtype,
        attention_config:Dict,
        prefill_compress:bool,
        max_comp_size:int=10,
        n_inst:int=0,
        n_continue:int=1,
    ):
        """
        args:
            - max_length: 
                max generated tokens
            - device: 
                "cuda"
            - dtype: 
                fp16 or bf16
            - attention_config: 
                attention mask
            - prefill_compress: 
                if True:
                    we will compress the prompt part
                if False:
                    we do not compress the prompt part
            - max_comp_size:
                How many tokens can a thought be compressed into
                    used for delta_attn
            - n_inst:
                instrution length
                    used for delta_attn
            - n_continue:
                used for delta_attn
                    if we compress in thought level, n_continue will be set to 1
                    if we compress in token level, n_continue will be set to 0
        """
        self.dtype = dtype
        self.device:str = device
        self.max_length:int = max_length
        self.attention_config:Dict = attention_config
        self.prefill_compress:bool = prefill_compress

        self.min_dtype = torch.finfo(dtype).min

        # this is used for global attention.
        # this is always a lower triangular matrix
        self.base_attn = torch.triu(
            torch.full(
                (max_length, max_length), fill_value=self.min_dtype, dtype=self.dtype, device=self.device
            ),
            diagonal=1
        )
        # this is used for local attention.
        # as a matter of fact, if we use kv cache, the attention mask shape is [n_new_token, all_token]
        self.delta_attn = torch.full(
            (max_comp_size + n_inst + n_continue + 1, max_length), fill_value=0., dtype=self.dtype, device=self.device
        )
        # this is used for global attention during inference.
        self.cur_attn = torch.triu(
            torch.full(
                (max_length, max_length), fill_value=self.min_dtype, dtype=self.dtype, device=self.device
            ),
            diagonal=1
        )

        # mask value
        self.mask_value = self.min_dtype
        self.show_value = 0.
        
        # Indicates which row you are currently on to start writing.
        self.last_idx = 0

        # diagonal attention, this is used for args.diagonal being True.
        self.diagonal_attn = torch.full((max_comp_size, max_comp_size), self.mask_value)
        self.diagonal_attn.fill_diagonal_(self.show_value)
        
        # used for global attention
        self.global_indicator_list:List[List[int]] = list()

    def create_prompt_attention(
        self,
        length,
        indicator_list:List[List[int]],
    ) -> torch.Tensor:
        """
        indicator_list[0]:
            [text_start, text_end, n_inst, comp_start, comp_end, n_cont]
        """
        self.reset()
        self.last_idx = length
        self.global_indicator_list.extend(indicator_list)

        prefill_compress = self.prefill_compress
        diagonal = self.attention_config['diagonal']
        see_current = self.attention_config['see_current']
        bi_attention = self.attention_config['bi_attention']

        if prefill_compress == False or len(indicator_list) == 0:
            # if we do not compress the prompt, we directly return a lower triangular matrix
            return self.cur_attn[0:length, 0:length].unsqueeze(dim=0).unsqueeze(dim=0)

        assert prefill_compress == True
        for indicator in indicator_list:
            text_start, text_end, n_inst, comp_start, comp_end, n_cont = indicator

            if text_end + n_inst != comp_start:
                assert n_inst == 0
            assert n_inst == 0, "now we only support n_inst == 0"

            # 1. the current content is invisible for later content.
            self.cur_attn[
                comp_end:length, 
                text_start:text_end
            ] = self.mask_value

            # 2. if we set the see_current, it means that the previous content is not visible during compression
            if see_current:
                self.cur_attn[
                    comp_start:comp_end,
                    0:text_start
                ] = self.mask_value
            
            # 3. attention config
            if bi_attention:
                self.cur_attn[
                    comp_start:comp_end,
                    comp_start:comp_end,
                ] = self.show_value
            if diagonal:
                self.cur_attn[
                    comp_start:comp_end,
                    comp_start:comp_end
                ] = self.diagonal_attn[
                    0:comp_end-comp_start,
                    0:comp_end-comp_start,
                ]

        return self.cur_attn[0:length, 0:length].unsqueeze(dim=0).unsqueeze(dim=0)

    def late_ajust_prompt_attention(
        self,
        length:int,
        indicator_list:List[List[int]]
    ):
        diagonal = self.attention_config['diagonal']
        see_current = self.attention_config['see_current']
        bi_attention = self.attention_config['bi_attention']

        assert self.prefill_compress == False
        for indicator in indicator_list:
            text_start, text_end, n_inst, comp_start, comp_end, n_cont = indicator

            if text_end + n_inst != comp_start:
                assert n_inst == 0
            assert n_inst == 0, "now we only support n_inst == 0"

            # 1. the current content is invisible for later content.
            self.cur_attn[
                comp_end:length,
                text_start:text_end
            ] = self.mask_value

            # 2. if we set the see_current, it means that the previous content is not visible during compression
            if see_current:
                self.cur_attn[
                    comp_start:comp_end,
                    0:text_start
                ] = self.max_value
            
            # 3. attention config
            if bi_attention:
                self.cur_attn[
                    comp_start:comp_end,
                    comp_start:comp_end,
                ] = self.show_value
            if diagonal:
                self.cur_attn[
                    comp_start:comp_end,
                    comp_start:comp_end
                ] = self.diagonal_attn[
                    0:comp_end-comp_start,
                    0:comp_end-comp_start,
                ]

    # we recommend using this function for debug.
    def update_attention_global(
        self,
        new_length:int,
        indicator:List[int],
    ):

        diagonal = self.attention_config['diagonal']
        see_current = self.attention_config['see_current']
        bi_attention = self.attention_config['bi_attention']

        # 1. copy
        self.cur_attn[
            self.last_idx:self.last_idx+new_length,
            0:self.last_idx
        ] = self.cur_attn[self.last_idx-1, 0:self.last_idx]
        
        # 2. if current is in compression mode
        if indicator != None:
            self.global_indicator_list.append(indicator)
            text_start, text_end, n_inst, comp_start, comp_end, n_cont = indicator

            self.cur_attn[
                comp_end:self.last_idx+new_length,
                text_start:text_end
            ] = self.mask_value

            if see_current:
                self.cur_attn[
                    comp_start:comp_end,
                    0:text_start
                ] = self.mask_value
            
            if bi_attention:
                self.cur_attn[
                    comp_start:comp_end,
                    comp_start:comp_end,
                ] = self.show_value
            
            if diagonal:
                self.cur_attn[
                    comp_start:comp_end,
                    comp_start:comp_end
                ] = self.diagonal_attn[
                    0:comp_end-comp_start,
                    0:comp_end-comp_start,
                ]

        self.last_idx += new_length

        # 3. truncate and return 
        self.delta_attn[:] = self.show_value
        remove_size = 0 
        if indicator != None:
            for _indicator in self.global_indicator_list[0:-1]:
                text_start, text_end, n_inst, comp_start, comp_end, n_cont = _indicator
                remove_size += (text_end-text_start)
            text_start, text_end, n_inst, comp_start, comp_end, n_cont = indicator
            # we need add an offset bias.
            text_start -= remove_size
            text_end -= remove_size
            comp_start -= remove_size
            comp_end -= remove_size
            n_prefix = new_length - (comp_end - comp_start + n_cont)
            save_length = self.last_idx - remove_size
            self.delta_attn[
                0:new_length, 
                text_start:save_length
            ] = \
                self.cur_attn[
                    self.last_idx - new_length: self.last_idx, 
                    text_start+remove_size:self.last_idx
                ]
            self.delta_attn[0:new_length, self.last_idx - remove_size-new_length:self.last_idx - remove_size] = \
                self.base_attn[0:new_length, 0:new_length]
            return self.delta_attn[0:new_length, 0:self.last_idx-remove_size].unsqueeze(dim=0).unsqueeze(dim=0)
        else:
            for _indicator in self.global_indicator_list:
                text_start, text_end, n_inst, comp_start, comp_end, n_cont = _indicator
                remove_size += (text_end-text_start)
            self.delta_attn[0:new_length, self.last_idx-remove_size-new_length:self.last_idx-remove_size] = \
                self.base_attn[0:new_length, 0:new_length]
            return self.delta_attn[0:new_length, 0:self.last_idx-remove_size].unsqueeze(dim=0).unsqueeze(dim=0)

    # This is a preferred method.
    def update_attention_local(
        self,
        origin_length:int,
        new_length:int,
        indicator:List[int],
    ) -> torch.Tensor:
        """
        args:
            - origin_length
            - new_length
        """

        # initialization
        self.delta_attn[0:new_length, 0:origin_length+new_length] = self.show_value
        self.delta_attn[0:new_length, origin_length:origin_length+new_length] = self.base_attn[0:new_length, 0:new_length]

        if indicator != None:
            text_start, text_end, n_inst, comp_start_c, comp_end_c, n_cont = indicator
            n_prefix = new_length - (comp_end_c - comp_start_c + n_cont)
            assert n_prefix >= 0
            comp_start_r = n_prefix
            comp_end_r = n_prefix + (comp_end_c - comp_start_c)

            # attention config
            diagonal = self.attention_config['diagonal']
            see_current = self.attention_config['see_current']
            bi_attention = self.attention_config['bi_attention']

            # mask
            self.delta_attn[
                comp_end_r:,
                text_start:text_end
            ] = self.mask_value

            if see_current:
                self.delta_attn[
                    comp_start_r:comp_end_r,
                    0:text_start
                ] = self.mask_value

            if bi_attention:
                self.delta_attn[
                    comp_start_r:comp_end_r,
                    comp_start_c:comp_end_c
                ] = self.show_value
            
            if diagonal:
                self.delta_attn[
                    comp_start_r:comp_end_r,
                    comp_start_c:comp_end_c
                ] = self.diagonal_attn[
                    0:comp_end_r-comp_start_r,
                    0:comp_end_r-comp_start_r,
                ]
        
        return self.delta_attn[0:new_length, 0:origin_length+new_length].unsqueeze(dim=0).unsqueeze(dim=0)
        
    def reset(self):
        self.cur_attn[:, :] = self.base_attn[:, :]
        self.last_idx = 0
        self.global_indicator_list.clear()
        self.delta_attn[:] = self.show_value

class KVUtils:
    """
    KV Cache Manager
    """

    def __init__(self):
        self.past_key_values: DynamicCache = DynamicCache()

    def get_cache(self) -> DynamicCache:
        return self.past_key_values

    def set_cache(self, past_key_values:DynamicCache):
        self.past_key_values = past_key_values

    @torch.no_grad()
    def reduce_cache(self, start:int, end:int):
        assert end <= self.past_key_values._seen_tokens
        assert self.past_key_values._seen_tokens == self.past_key_values.key_cache[0].shape[2]

        # 1. reduce the value of seen_tokens
        self.past_key_values._seen_tokens -= (end-start)

        # 2. reduce the key_cache and value_cache
        bsz, n_head, q_length, head_dim = self.past_key_values.key_cache[0].shape
        new_q_length = q_length - (end-start)
        assert self.past_key_values._seen_tokens == new_q_length
        for layer_id in range(len(self.past_key_values.key_cache)):
            if new_q_length > end:
                # overlap
                # start:end
                for i in range(new_q_length-start):
                    self.past_key_values.key_cache[layer_id][:, :, start+i, :] = self.past_key_values.key_cache[layer_id][:, :, end+i, :]
                    self.past_key_values.value_cache[layer_id][:, :, start+i, :] = self.past_key_values.value_cache[layer_id][:, :, end+i, :]
            else:
                self.past_key_values.key_cache[layer_id][:, :, start:new_q_length, :] = self.past_key_values.key_cache[layer_id][:, :, end:, :]
                self.past_key_values.value_cache[layer_id][:, :, start:new_q_length, :] = self.past_key_values.value_cache[layer_id][:, :, end:, :]

            self.past_key_values.key_cache[layer_id] = \
                self.past_key_values.key_cache[layer_id][:, :, 0:new_q_length, :]
            self.past_key_values.value_cache[layer_id] = \
                self.past_key_values.value_cache[layer_id][:, :, 0:new_q_length, :]

    def __del__(self):
        del self.past_key_values.value_cache
        del self.past_key_values.key_cache
        del self.past_key_values
        torch.cuda.empty_cache()
        time.sleep(1)

class TokenUtils:

    """
    Generated Token Manager
    """

    def __init__(self, max_length:int, device:str, rolling_rope:bool):
        self.rolling_rope:bool = rolling_rope

        # self.input_ids[..., 0:self._seen_tokens]
        self.max_length:int = max_length
        self.input_ids:torch.Tensor = torch.arange(max_length, device=device).unsqueeze(dim=0)
        self._seen_tokens:int = 0

        # complete token sequence
        self._whole_input_ids:List[int] = list()
        # dynamic token sequence (thoughts to be discarded.)
        self._current_input_ids:List[int] = list()

        # compression token will not be included
        self.show_prompt_input_ids:List[int] = list()
        self.show_output_input_ids:List[int] = list()

        self.position_ids:torch.Tensor = torch.arange(max_length, device=device).unsqueeze(dim=0)
        # used when rolling_rope==True
        self.arange_ids:torch.Tensor = torch.arange(max_length, device=device)

        self._whole_position_ids:List[int] = list()
        self._current_position_ids:List[int] = list()

        # peak token
        self.max_token = 0

    def get_input_ids(self) -> torch.Tensor:
        return self.input_ids[..., 0:self._seen_tokens]

    def get_input_ids(self, start:int, end:int) -> torch.Tensor:
        if start >= 0 and end >= 0:
            return self.input_ids[..., start:end]
        else:
            if start < 0:
                start = self._seen_tokens + start
            if end < 0:
                end = self._seen_tokens + end
            return self.input_ids[..., start:end]
    
    def get_input_ids(self, idx:int) -> torch.Tensor:
        if idx >= 0:
            return self.input_ids[..., idx:idx+1]
        else:
            idx = self._seen_tokens + idx
            return self.input_ids[..., idx:idx+1]

    def get_position_ids(self) -> torch.Tensor:
        return self.position_ids[..., 0:self._seen_tokens]

    def set_input_id(self, idx:int):
        self.input_ids[..., self._seen_tokens] = idx
        self._whole_input_ids.append(idx)
        self._current_input_ids.append(idx)

        if self.rolling_rope:
            new_pos = len(self._current_position_ids)
        else:
            new_pos = self._whole_position_ids[-1] + 1
    
        self.position_ids[..., self._seen_tokens] = new_pos
        self._current_position_ids.append(new_pos)
        self._whole_position_ids.append(new_pos)

        self._seen_tokens += 1
        self.max_token = max(self.max_token, self._seen_tokens)

    def set_input_ids(self, input_ids:List[int], return_tensors:bool=False):
        assert isinstance(input_ids, list)
        _start = self._seen_tokens
        for i in range(len(input_ids)):
            self.input_ids[..., self._seen_tokens + i] = input_ids[i]
            if not self.rolling_rope:
                if len(self._whole_position_ids) == 0:
                    self.position_ids[..., self._seen_tokens + i] = 0
                    self._current_position_ids.append(0)
                    self._whole_position_ids.append(0)
                else:
                    self.position_ids[..., self._seen_tokens + i] = self.position_ids[0,self._seen_tokens + i - 1] + 1
                    self._current_position_ids.append(self._whole_position_ids[-1] + 1)
                    self._whole_position_ids.append(self._whole_position_ids[-1] + 1)
        _end = _start + len(input_ids)
        
        if self.rolling_rope:
            self.position_ids[..., 0:self._seen_tokens+len(input_ids)] = self.arange_ids[0:self._seen_tokens]
            self._current_position_ids.extend([self._current_position_ids[-1] + i + 1 for i in range(len(input_ids))])
            self._whole_position_ids.extend([self._current_position_ids[-1] + i + 1 for i in range(len(input_ids))])
            # assert False
        self._seen_tokens += len(input_ids)
        self._whole_input_ids.extend(input_ids)
        self._current_input_ids.extend(input_ids)
        self.max_token = max(self.max_token, self._seen_tokens)
        if return_tensors:
            return self.input_ids[..., _start:_end], self.position_ids[..., _start:_end]
 
    def reduce_input_ids(self, start:int, end:int):
        origin_length = self._seen_tokens
        self._seen_tokens -= (end-start)
        if self._seen_tokens > end:
            for i in range(self._seen_tokens-start):
                self.input_ids[..., start+i] = self.input_ids[..., end+i]
                self._current_input_ids[start+i] = self._current_input_ids[end+i]
        else:
            self.input_ids[..., start:self._seen_tokens] = self.input_ids[..., end:origin_length]
            self._current_input_ids[start:self._seen_tokens] = self._current_input_ids[end:origin_length]
        
        self._current_input_ids = self._current_input_ids[0:self._seen_tokens]

        if not self.rolling_rope:
            if self._seen_tokens > end:
                for i in range(self._seen_tokens-start):
                    self.position_ids[..., start+i] = self.position_ids[..., end+i]
                    self._current_position_ids[start+i] = self._current_position_ids[end+i]
            else:
                self.position_ids[..., start:self._seen_tokens] = self.position_ids[..., end:origin_length]
                self._current_position_ids[start:self._seen_tokens] = self._current_position_ids[end:origin_length]
        else:
            self._current_position_ids[start:self._seen_tokens] = [start+i for i in range(self._seen_tokens - start)]
            self.position_ids[..., 0:self._seen_tokens] = self.arange_ids[0:self._seen_tokens]
        self._current_position_ids = self._current_position_ids[0:self._seen_tokens]

    def reset(self):
        self._seen_tokens = 0
        self.max_token = 0
        self._whole_input_ids.clear()
        self._current_input_ids.clear()
        self._whole_position_ids.clear()
        self._current_position_ids.clear()
        self.show_prompt_input_ids.clear()
        self.show_output_input_ids.clear()

# ========== CORE CODE ==========
@torch.no_grad()
def _prefill_wo_prompt_compression(
    model: Union[LlamaForCausalLM, Qwen2ForCausalLM],
    tokenizer: Tokenizer,
    comp_config: Config,
    system_prompt:str,
    system_prompt_list: List[str],
    question:str,
    question_list:List[str],
    attention_config:Dict,
    prefill_compress:bool,
    compress_prompt:bool,
    attn_utils:AttentionUtils,
    kv_utils:KVUtils,
    token_utils:TokenUtils,
) -> int:
    """
    prompt will not be compressed
    """
    assert compress_prompt is False

    past_key_values:DynamicCache = kv_utils.get_cache()

    # 1. concatenate prompt
    prompt:str = tokenizer.bos_token + comp_config.template_cfg['complete'].format(
        system=system_prompt, question=question
    )
    # 2. tokenize
    input_ids = tokenizer.tokenizer(
        prompt, return_tensors=None
    )['input_ids']
    token_utils.show_prompt_input_ids.extend(input_ids)
    token_utils.set_input_ids(input_ids)
    if DEBUG:
        DebugUtils.show_global_attention(
            tokenizer=tokenizer,
            attention_mask=attention_mask.squeeze().cpu().tolist(), 
            input_ids=input_ids,
            position_ids=token_utils.get_position_ids().squeeze().cpu().tolist(),
            block=BLOCK,
            start_offset=None,
            end_offset=None,
            file_name="debug_global.png",
        )

    # 3. model.forward()
    model_output = model(
        input_ids=torch.as_tensor(
            [input_ids], device="cuda"
        ),
        use_cache=True,
        past_key_values=past_key_values,
        return_dict=True
    )
    # 4. get the generated token id
    predicted_token_id:int = InferenceUtils.get_predicted_token_ids(
        model_output=model_output, idx=-1
    )

    return predicted_token_id
    
@torch.no_grad()
def _prefill_w_prompt_compression(
    model: Union[LlamaForCausalLM, Qwen2ForCausalLM],
    tokenizer: Tokenizer,
    comp_config: Config,
    question:str,
    question_list:List[str],
    system_prompt:str,
    system_prompt_list: List[str],
    attention_config:Dict,
    prefill_compress:bool,
    compress_prompt:bool,
    attn_utils:AttentionUtils,
    kv_utils:KVUtils,
    token_utils:TokenUtils,
) -> int:
    """
    prompt will be compressed
    
    All the code here is not reused to ensure clarity and readability as much as possible.
    """
    assert compress_prompt == True

    input_ids:List[int] = list()
    indicator_list:List[List[int]] = list()

    prefix_prompt = comp_config.template_cfg['prefix']
    middle_prompt = comp_config.template_cfg['middle']
    suffix_prompt = comp_config.template_cfg['suffix']

    past_key_values = kv_utils.get_cache()

    # 1. set the input_ids and indicator_list
    if comp_config.prompt_comp_level == 'token' and comp_config.prompt_save_template == True:
        prefix_input_ids:List[int] = [tokenizer.bos_token_id] + tokenizer.tokenizer(
            prefix_prompt, return_tensors=None
        )['input_ids']
        middle_input_ids:List[int] = tokenizer.tokenizer(
            middle_prompt, return_tensors=None
        )['input_ids']
        suffix_input_ids:List[int] = tokenizer.tokenizer(
            suffix_prompt, return_tensors=None
        )['input_ids']
        system_prompt_input_ids:List[int] = tokenizer.tokenizer(
            system_prompt, return_tensors=None
        )['input_ids']
        question_input_ids:List[int] = tokenizer.tokenizer(
            question, return_tensors=None
        )['input_ids']

        step:int = comp_config.prompt_comp_step

        input_ids.extend(prefix_input_ids)
        token_utils.show_prompt_input_ids.extend(prefix_input_ids)
        for i in range(0, len(system_prompt_input_ids), step):
            text_start = len(input_ids)
            input_ids.extend(system_prompt_input_ids[i:i+step])
            token_utils.show_prompt_input_ids.extend(system_prompt_input_ids[i:i+step])
            text_end = len(input_ids)
            n_inst = 0
            comp_start = len(input_ids)
            input_ids.extend(comp_config.get_prompt_comp_token_id())
            comp_end = len(input_ids)
            n_cont = 0
            indicator_list.append(
                [text_start, text_end, n_inst, comp_start, comp_end, n_cont]
            )
        
        input_ids.extend(middle_input_ids)
        token_utils.show_prompt_input_ids.extend(middle_input_ids)
        for i in range(0, len(question_input_ids), step):
            text_start = len(input_ids)
            input_ids.extend(question_input_ids[i:i+step])
            token_utils.show_prompt_input_ids.extend(question_input_ids[i:i+step])
            text_end = len(input_ids)
            n_inst = 0
            comp_start = len(input_ids)
            input_ids.extend(comp_config.get_prompt_comp_token_id())
            comp_end = len(input_ids)
            n_cont = 0
            indicator_list.append(
                [text_start, text_end, n_inst, comp_start, comp_end, n_cont]
            )

        input_ids.extend(suffix_input_ids)
        token_utils.show_prompt_input_ids.extend(suffix_input_ids)
    elif comp_config.prompt_comp_level == 'token' and comp_config.prompt_save_template == False:
        prompt:str = tokenizer.bos_token + \
            prefix_prompt + system_prompt + \
                middle_prompt + question + \
                    suffix_prompt
        _input_ids:List[int] = tokenizer.tokenizer(
            prompt, return_tensors=None
        )['input_ids']
        step = comp_config.prompt_comp_step
        token_utils.show_prompt_input_ids.extend(_input_ids)

        for i in range(0, len(_input_ids), step):
            text_start = len(input_ids)
            input_ids.extend(_input_ids[i:i+step])
            text_end = len(input_ids)
            n_inst = 0
            comp_start = len(input_ids)
            input_ids.extend(comp_config.get_prompt_comp_token_id())
            comp_end = len(input_ids)
            n_cont = 0
            indicator_list.append(
                [text_start, text_end, n_inst, comp_start, comp_end, n_cont]
            )
        input_ids.append(comp_config.continue_token_id)
        token_utils.show_prompt_input_ids.append(comp_config.continue_token_id)
    elif comp_config.prompt_comp_level == 'sentence' and comp_config.prompt_save_template == True:
        prefix_input_ids:List[int] = [tokenizer.bos_token_id] + tokenizer.tokenizer(
            prefix_prompt, return_tensors=None
        )['input_ids']
        middle_input_ids:List[int] = tokenizer.tokenizer(
            middle_prompt, return_tensors=None
        )['input_ids']
        suffix_input_ids:List[int] = tokenizer.tokenizer(
            suffix_prompt, return_tensors=None
        )['input_ids']

        input_ids.extend(prefix_input_ids)
        token_utils.show_prompt_input_ids.extend(prefix_input_ids)
        for sent in system_prompt_list:
            text_start = len(input_ids)
            sent_input_ids = tokenizer.tokenizer(
                sent, return_tensors=None
            )['input_ids']
            input_ids.extend(sent_input_ids)
            token_utils.show_prompt_input_ids.extend(sent_input_ids)
            text_end = len(input_ids)
            n_inst = 0
            comp_start = len(input_ids)
            input_ids.extend(comp_config.get_prompt_comp_token_id())
            comp_end = len(input_ids)
            n_cont = 0
            indicator_list.append(
                [text_start, text_end, n_inst, comp_start, comp_end, n_cont]
            )
        
        input_ids.extend(middle_input_ids)
        token_utils.show_prompt_input_ids.extend(middle_input_ids)
        for sent in question_list:
            text_start = len(input_ids)
            sent_input_ids = tokenizer.tokenizer(
                sent, return_tensors=None
            )['input_ids']
            input_ids.extend(sent_input_ids)
            token_utils.show_prompt_input_ids.extend(sent_input_ids)
            text_end = len(input_ids)
            n_inst = 0
            comp_start = len(input_ids)
            input_ids.extend(comp_config.get_prompt_comp_token_id())
            comp_end = len(input_ids)
            n_cont = 0
            indicator_list.append(
                [text_start, text_end, n_inst, comp_start, comp_end, n_cont]
            )

        input_ids.extend(suffix_input_ids)
        token_utils.show_prompt_input_ids.extend(suffix_input_ids)
    elif comp_config.prompt_comp_level == 'sentence' and comp_config.prompt_save_template == False:
        raise NotImplementedError()
    else:
        raise NotImplementedError()
    # print(input_ids)
    token_utils.set_input_ids(input_ids)

    # 2. generate attention_mask
    # [1, 1, length, length]
    attention_mask:torch.Tensor = attn_utils.create_prompt_attention(
        length=len(input_ids),
        indicator_list=indicator_list,
    )
    if DEBUG:
        # print(token_utils.get_position_ids().squeeze().cpu().tolist())
        DebugUtils.show_global_attention(
            tokenizer=tokenizer,
            attention_mask=attention_mask.squeeze().cpu().tolist(), 
            input_ids=input_ids,
            position_ids=token_utils.get_position_ids().squeeze().cpu().tolist(),
            block=BLOCK,
            start_offset=None,
            end_offset=None,
            file_name="debug_global.png",
        )

    # 3. forward
    model_output = model(
        input_ids=torch.as_tensor(
            [input_ids], device="cuda"
        ),
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        use_cache=True,
        return_dict=True,
    )

    if not prefill_compress:
        attn_utils.late_ajust_prompt_attention(
            length=len(input_ids),
            indicator_list=indicator_list,
        )
        if DEBUG:
            DebugUtils.show_global_attention(
                tokenizer=tokenizer,
                attention_mask=attn_utils.cur_attn[0:attn_utils.last_idx, 0:attn_utils.last_idx].cpu().tolist(),
                # attention_mask=attention_mask.squeeze().cpu().tolist(), 
                input_ids=input_ids,
                position_ids=token_utils.get_position_ids().squeeze().cpu().tolist(),
                block=BLOCK,
                start_offset=None,
                end_offset=None,
                file_name="debug_global.png",
            )

    # 4. generate the new token id
    predicted_token_id:int = InferenceUtils.get_predicted_token_ids(
        model_output=model_output, idx=-1
    )

    # 5. Update kv_Cache
    for indicator in indicator_list[::-1]:
        text_start, text_end, n_inst, comp_start, comp_end, n_cont = indicator
        start = text_start
        end = text_end
        assert n_inst == 0
        kv_utils.reduce_cache(start=start, end=end)
        token_utils.reduce_input_ids(start=start, end=end)

    return predicted_token_id

@torch.no_grad()
def prefill(
    model: Union[LlamaForCausalLM, Qwen2ForCausalLM],
    tokenizer: Tokenizer,
    comp_config: Config,
    question:str,
    question_list:List[str],
    system_prompt:str,
    system_prompt_list: List[str],
    attention_config:Dict,
    prefill_compress:bool,
    compress_prompt:bool,
    attn_utils:AttentionUtils,
    kv_utils:KVUtils,
    token_utils:TokenUtils,
) -> int:
    """
    There are several possible scenarios here:
        - Do not compress the prompt
        - Compress the prompt
            - Prefill stage mutually visible
            - Prefill stage mutually invisible
    """

    if not compress_prompt:
        return _prefill_wo_prompt_compression(
            model=model,
            tokenizer=tokenizer,
            comp_config=comp_config,
            system_prompt=system_prompt,
            system_prompt_list=system_prompt_list,
            question=question,
            question_list=question_list,
            attention_config=attention_config,
            prefill_compress=prefill_compress,
            attn_utils=attn_utils,
            kv_utils=kv_utils,
            token_utils=token_utils,
            compress_prompt=compress_prompt
        )
    else:
        return _prefill_w_prompt_compression(
            model=model,
            tokenizer=tokenizer,
            comp_config=comp_config,
            question=question,
            question_list=question_list,
            system_prompt=system_prompt,
            system_prompt_list=system_prompt_list,
            attention_config=attention_config,
            prefill_compress=prefill_compress,
            compress_prompt=compress_prompt,
            attn_utils=attn_utils,
            kv_utils=kv_utils,
            token_utils=token_utils,
        )

@torch.no_grad()
def _token_level_generate(
    model: Union[LlamaForCausalLM, Qwen2ForCausalLM],
    tokenizer: Tokenizer,
    comp_config: Config,
    max_new_tokens: int,
    attention_config:Dict,
    prefill_compress:bool,
    exclude_continue:bool,
    attn_utils:AttentionUtils,
    kv_utils:KVUtils,
    token_utils:TokenUtils,
    predicted_token_id:int,
    update_attention_method:str="global"
) -> Tuple[str, str]:
    
    assert update_attention_method in ["global", "local"]

    new_token_counters = 0
    eos_token_id = tokenizer.eos_token_id
    explicit_token_cnt = 1
    output_comp_step = comp_config.output_comp_step

    global_start:int = len(token_utils._whole_input_ids)
    local_start:int = len(token_utils._current_input_ids)


    assert local_start == kv_utils.get_cache()._seen_tokens, \
        f"{local_start} == {kv_utils.get_cache()._seen_tokens}"
    while predicted_token_id != eos_token_id and new_token_counters < max_new_tokens:
        new_input_ids = [predicted_token_id]
        token_utils.show_output_input_ids.append(predicted_token_id)
        IS_COMP_MODE:bool = False

        # 1. construct attention_mask
        if explicit_token_cnt == output_comp_step:
            IS_COMP_MODE = True
            new_input_ids.extend(
                comp_config.get_output_comp_token_id()
            )
            new_input_ids.append(
                comp_config.continue_token_id
            )
            explicit_token_cnt = 1
            new_length = len(new_input_ids)
            if update_attention_method == 'global':
                origin_length = len(token_utils._whole_input_ids)
                indicator = [
                    global_start,
                    origin_length + 1,  # Because the last token has not been included yet.
                    0,
                    origin_length + 1,
                    origin_length + 1 + len(comp_config.get_output_comp_token_id()),
                    1,
                ]
                attention_mask = attn_utils.update_attention_global(
                    new_length=new_length,
                    indicator=indicator,
                )
            else:
                origin_length = len(token_utils._current_input_ids)
                indicator = [
                    local_start,
                    origin_length + 1,
                    0,
                    origin_length + 1,
                    origin_length + 1 + len(comp_config.get_output_comp_token_id()),
                    1,
                ]
                attention_mask = attn_utils.update_attention_local(
                    origin_length=origin_length,
                    new_length=new_length,
                    indicator=indicator
                )
        else:
            explicit_token_cnt += 1
            if update_attention_method == 'global':
                origin_length = len(token_utils._whole_input_ids)
                attention_mask = attn_utils.update_attention_global(
                    new_length=1,
                    indicator=None
                )
            else:
                origin_length = len(token_utils._current_input_ids)
                attention_mask = attn_utils.update_attention_local(
                    origin_length=origin_length,
                    new_length=1,
                    indicator=None
                )

        # The line below marks the end of the mask during the reduce process, 
        # primarily for the purpose of reduction.
        _local_mask_end = len(token_utils._current_input_ids) + 1
        # 2. position_ids and input_ids
        input_ids, position_ids = token_utils.set_input_ids(
            new_input_ids, return_tensors=True
        ) 
        if DEBUG:
            if update_attention_method == 'global':
                DebugUtils.show_global_attention(
                    tokenizer=tokenizer,
                    attention_mask=attn_utils.cur_attn[0:attn_utils.last_idx, 0:attn_utils.last_idx].cpu().tolist(),
                    # attention_mask=attention_mask.squeeze(dim=0).squeeze(dim=0).cpu().tolist(), 
                    input_ids=deepcopy(token_utils._whole_input_ids),
                    position_ids=deepcopy(token_utils._whole_position_ids),
                    # position_ids=token_utils.get_position_ids().squeeze().cpu().tolist(),
                    block=BLOCK,
                    start_offset=None,
                    end_offset=None,
                    file_name="debug_global.png",
                )
            else:
                DebugUtils.show_local_attention(
                    tokenizer=tokenizer,
                    attention_mask=attention_mask.squeeze(dim=0).squeeze(dim=0).cpu().tolist(), 
                    input_ids=deepcopy(token_utils._current_input_ids),
                    position_ids=deepcopy(token_utils._current_position_ids),
                    # position_ids=token_utils.get_position_ids().squeeze().cpu().tolist(),
                    block=BLOCK,
                    start_offset=None,
                    end_offset=None,
                    file_name="debug_local_1.png",
                )

        # 3. generate new token
        model_output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=kv_utils.get_cache(),
            use_cache=True,
            return_dict=True,
            position_ids=position_ids,
        )

        # 4. update kv cache
        if IS_COMP_MODE:
            start = local_start
            end = _local_mask_end

            kv_utils.reduce_cache(start=start, end=end)
            token_utils.reduce_input_ids(start=start, end=end)

            # update
            global_start:int = len(token_utils._whole_input_ids)
            local_start:int = len(token_utils._current_input_ids)

        # 5. get the new predicted_tokens
        predicted_token_id:int = InferenceUtils.get_predicted_token_ids(
            model_output=model_output, idx=-1
        )
        new_token_counters += 1
    
    token_utils.show_output_input_ids.append(predicted_token_id)
    # return tokenizer.decode(token_utils._whole_input_ids)
    return tokenizer.decode(token_utils.show_prompt_input_ids), tokenizer.decode(token_utils.show_output_input_ids)

@torch.no_grad()
def _sentence_level_generate(
    model: Union[LlamaForCausalLM, Qwen2ForCausalLM],
    tokenizer: Tokenizer,
    comp_config: Config,
    max_new_tokens: int,
    attention_config:Dict,
    prefill_compress:bool,
    exclude_continue:bool,
    attn_utils:AttentionUtils,
    kv_utils:KVUtils,
    token_utils:TokenUtils,
    predicted_token_id:int,
    update_attention_method:str="global"
) -> Tuple[str,str]:
    assert update_attention_method in ["global", "local"]

    new_token_counters = 0
    eos_token_id = tokenizer.eos_token_id
    # eos_token_id = None
    output_comp_step = comp_config.output_comp_step

    global_start:int = len(token_utils._whole_input_ids)
    local_start:int = len(token_utils._current_input_ids)


    assert local_start == kv_utils.get_cache()._seen_tokens, \
        f"{local_start} == {kv_utils.get_cache()._seen_tokens}"
    while predicted_token_id != eos_token_id and new_token_counters < max_new_tokens:
        new_input_ids = [predicted_token_id]
        IS_COMP_MODE:bool = False
        token_utils.show_output_input_ids.append(predicted_token_id)

        # 1. construct attention_mask
        if predicted_token_id == comp_config.split_token_id:
            IS_COMP_MODE = True
            new_input_ids.extend(
                comp_config.get_output_comp_token_id()
            )
            new_input_ids.append(
                comp_config.continue_token_id
            )
            new_length = len(new_input_ids)
            if update_attention_method == 'global':
                origin_length = len(token_utils._whole_input_ids)
                indicator = [
                    global_start,
                    origin_length + 1,  # the last token has not been included yet.
                    0,
                    origin_length + 1,
                    origin_length + 1 + len(comp_config.get_output_comp_token_id()),
                    1,
                ]
                attention_mask = attn_utils.update_attention_global(
                    new_length=new_length,
                    indicator=indicator,
                )
            else:
                origin_length = len(token_utils._current_input_ids)
                indicator = [
                    local_start,
                    origin_length + 1,
                    0,
                    origin_length + 1,
                    origin_length + 1 + len(comp_config.get_output_comp_token_id()),
                    1,
                ]
                attention_mask = attn_utils.update_attention_local(
                    origin_length=origin_length,
                    new_length=new_length,
                    indicator=indicator
                )
        else:
            if update_attention_method == 'global':
                origin_length = len(token_utils._whole_input_ids)
                attention_mask = attn_utils.update_attention_global(
                    new_length=1,
                    indicator=None
                )
            else:
                origin_length = len(token_utils._current_input_ids)
                attention_mask = attn_utils.update_attention_local(
                    origin_length=origin_length,
                    new_length=1,
                    indicator=None
                )

        # The line below marks the end of the mask during the reduce process, 
        # primarily for the purpose of reduction.
        _local_mask_end = len(token_utils._current_input_ids) + 1
        # 2. position_ids and input_ids
        if token_utils.max_length < len(new_input_ids) + token_utils._seen_tokens:
            # exceed length
            break
        input_ids, position_ids = token_utils.set_input_ids(
            new_input_ids, return_tensors=True
        ) 
        if DEBUG:
            if update_attention_method == 'global':
                DebugUtils.show_global_attention(
                    tokenizer=tokenizer,
                    attention_mask=attn_utils.cur_attn[0:attn_utils.last_idx, 0:attn_utils.last_idx].cpu().tolist(),
                    # attention_mask=attention_mask.squeeze(dim=0).squeeze(dim=0).cpu().tolist(), 
                    input_ids=deepcopy(token_utils._whole_input_ids),
                    position_ids=deepcopy(token_utils._whole_position_ids),
                    # position_ids=token_utils.get_position_ids().squeeze().cpu().tolist(),
                    block=BLOCK,
                    start_offset=None,
                    end_offset=None,
                    file_name="debug_global.png",
                )
            else:
                DebugUtils.show_local_attention(
                    tokenizer=tokenizer,
                    attention_mask=attention_mask.squeeze(dim=0).squeeze(dim=0).cpu().tolist(), 
                    input_ids=deepcopy(token_utils._current_input_ids),
                    position_ids=deepcopy(token_utils._current_position_ids),
                    # position_ids=token_utils.get_position_ids().squeeze().cpu().tolist(),
                    block=BLOCK,
                    start_offset=None,
                    end_offset=None,
                    file_name="debug_local_1.png",
                )

        # 3. generate new token
        model_output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=kv_utils.get_cache(),
            use_cache=True,
            return_dict=True,
            position_ids=position_ids,
        )

        # 4. update kv cache
        if IS_COMP_MODE:
            start = local_start
            end = _local_mask_end

            kv_utils.reduce_cache(start=start, end=end)
            token_utils.reduce_input_ids(start=start, end=end)

            global_start:int = len(token_utils._whole_input_ids)
            local_start:int = len(token_utils._current_input_ids)

        # 5. get new predicted_tokens
        predicted_token_id:int = InferenceUtils.get_predicted_token_ids(
            model_output=model_output, idx=-1
        )
        new_token_counters += 1

    token_utils.show_output_input_ids.append(predicted_token_id)
    # return tokenizer.decode(token_utils._whole_input_ids)
    return tokenizer.decode(token_utils.show_prompt_input_ids), tokenizer.decode(token_utils.show_output_input_ids)

@torch.no_grad()
def generate(
    model: Union[LlamaForCausalLM, Qwen2ForCausalLM],
    tokenizer: Tokenizer,
    comp_config: Config,
    question:str,
    question_list:List[str],
    system_prompt: str,
    system_prompt_list: List[str],
    max_new_tokens: int,
    attention_config:Dict,
    prefill_compress:bool,
    exclude_continue:bool,
    compress_prompt:bool,
    attn_utils: AttentionUtils,
    token_utils: TokenUtils,
    update_attention_method:str,
) -> Tuple[str,str]:

    assert update_attention_method in ['global', 'local'], update_attention_method
    kv_utils = KVUtils()

    # 1. prefill
    predicted_token_id:int = prefill(
        model=model,
        tokenizer=tokenizer,
        comp_config=comp_config,
        question=question,
        question_list=question_list,
        system_prompt=system_prompt,
        system_prompt_list=system_prompt_list,
        attention_config=attention_config,
        prefill_compress=prefill_compress,
        compress_prompt=compress_prompt,
        attn_utils=attn_utils,
        kv_utils=kv_utils,
        token_utils=token_utils
    )

    # 2. auto-regressive generation
    if comp_config.output_comp_level == 'token':
        prompt, output = _token_level_generate(
            model=model,
            tokenizer=tokenizer,
            comp_config=comp_config,
            max_new_tokens=max_new_tokens,
            attention_config=attention_config,
            prefill_compress=prefill_compress,
            exclude_continue=exclude_continue,
            attn_utils=attn_utils,
            kv_utils=kv_utils,
            token_utils=token_utils,
            predicted_token_id=predicted_token_id,
            update_attention_method=update_attention_method
        )
    elif comp_config.output_comp_level == 'sentence':
        prompt, output = _sentence_level_generate(
            model=model,
            tokenizer=tokenizer,
            comp_config=comp_config,
            max_new_tokens=max_new_tokens,
            attention_config=attention_config,
            prefill_compress=prefill_compress,
            exclude_continue=exclude_continue,
            attn_utils=attn_utils,
            kv_utils=kv_utils,
            token_utils=token_utils,
            predicted_token_id=predicted_token_id,
            update_attention_method=update_attention_method
        )
    
    del kv_utils
    return prompt, output

def get_parser():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--model_tag', type=str)
    parser.add_argument('--model_short_tag', type=str, default=None)
    parser.add_argument('--ckpt', type=int)
    parser.add_argument('--tokenizer_path', type=str)
    parser.add_argument('--compress_config', type=str)
    parser.add_argument('--max_new_tokens', type=int)
    parser.add_argument('--output_tag', type=str)
    parser.add_argument('--model_type', type=str, choices=['qwen', 'llama'])
    parser.add_argument('--model_path', type=str, default=None)

    parser.add_argument('--bos_token', type=str)
    parser.add_argument('--eos_token', type=str)

    # specialized argument
    # ==============================
    parser.add_argument('--rolling_rope', type=str2bool)
    parser.add_argument('--diagonal', type=str2bool)
    parser.add_argument('--bi_directional', type=str2bool)
    parser.add_argument('--see_current', type=str2bool)
    parser.add_argument('--exclude_continue', type=str2bool)
    parser.add_argument('--output_compress_instruction', type=str)
    parser.add_argument('--prefill_compress', type=str2bool, default=True)
    parser.add_argument('--compress_prompt', type=str2bool, default=True)
    parser.add_argument('--update_attention_method', type=str, choices=['global', 'local'])
    # ==============================

    parser.add_argument('--split_size', type=int)
    # start from 1
    parser.add_argument('--index', type=int)        

    args = parser.parse_args()
    return args

def get_model_and_tokenizer(
    args, 
    comp_config:Config
) -> Tuple[
    Union[Qwen2ForCausalLM, LlamaForCausalLM],
    Tokenizer
]:
    if args.model_path == None or args.model_path == "":
        model_path = f"output/{args.model_tag}/checkpoint-{args.ckpt}"
    else:
        model_path = args.model_path
    print(f"load model from `{model_path}` ...")
    special_token_list:List[str] = list()
    tokenizer: Tokenizer = Tokenizer(
        tokenizer_path=args.tokenizer_path if args.tokenizer_path != None else model_path,
        bos_token=args.bos_token,
        eos_token=args.eos_token,
        special_token_list=None,
        add_prefix_space=False,
    )

    for token in comp_config.special_token_name_list:
        if tokenizer.convert_tokens_to_ids(token) == None:
            special_token_list.append(token)
    if len(special_token_list) > 0:
        tokenizer.add_special_token(special_token_list)

    if args.model_type.lower() == 'qwen':
        model = Qwen2ForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map="auto"
        )
    elif args.model_type.lower() == 'llama':
        model = LlamaForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map="auto"
        )

    comp_config.convert2id(tokenizer)
    
    return model, tokenizer

@torch.no_grad()
def eval_dataset(
    model:Union[Qwen2ForCausalLM, LlamaForCausalLM],
    tokenizer:Tokenizer,
    reader:Reader,
    comp_config:Config,
    output_file:str,
    max_new_tokens:int,
    max_prompt_len:int,
    device:str,
    dtype,
    max_comp_size:int,

    attention_config:Dict,
    prefill_compress:bool,
    exclude_continue:bool,
    compress_prompt:bool,

    update_attention_method:str,
    rolling_rope:bool,

    dataset_name:str,
    split_size:int=None,
    index:int=None,
):

    if split_size != None and index != None:
        assert index > 0
        assert index <= split_size
        step = len(reader) // split_size
        start = (index-1) * step
        end = index * step
        if index == split_size:
            end = len(reader)
    elif split_size == None and index == None:
        start = 0
        end = len(reader)
    else:
        assert False

    print(f"Starting test for `{dataset_name}`. Total size is {len(reader)}. Now, {index}/{split_size}: {start}-{end}")

    pbar = tqdm(total=end-start)
    
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
        except:
            pass

    # ===== CONTINUE =====
    data_copy_list = list()
    _id_list = list()
    if os.path.isfile(output_file):
        with jsonlines.open(output_file, 'r') as f:
            for item in f:
                _id_list.append(item['idx'])
                data_copy_list.append(item)
        if len(_id_list) != 0:
            assert _id_list[0] == start
            start = _id_list[-1] + 1
    # ====================

    total = 0
    acc = 0

    attn_utils = AttentionUtils(
        max_length=max_new_tokens + max_prompt_len,
        device=device,
        dtype=dtype,
        attention_config=attention_config,
        prefill_compress=prefill_compress,
        max_comp_size=max_comp_size,
        n_inst=0,
        n_continue=1
    )
    token_utils = TokenUtils(
        max_length=max_new_tokens + max_prompt_len,
        device=device,
        rolling_rope=rolling_rope
    )

    with jsonlines.open(output_file, 'w') as writer:
        # ===== CONTINUE =====
        for item in data_copy_list:
            writer.write(item)
            total += 1
            if item['acc_state'] == True:
                acc += 1
            model_answer = item['model_answer']
            gt_answer = item['gt_answer']
            acc_state = item['acc_state']
            input_len = item['input_len']
            output_len = item['output_len']
            max_token = item['max_token']
            comp_pattern = item['comp_pattern']

            pbar.set_description(f"model:`{model_answer}`; gt:`{gt_answer}`; correct:{acc_state}; acc: {acc}/{total}={round(acc/total, 5)}; input: {input_len}; output: {output_len}; `{comp_pattern}`; max_token: {max_token}")
            pbar.update(1)
        # ====================
        
        for i in range(start, end):
            total += 1
            question:str = reader.get_prompt(idx=i)
            question_list:List[str] = reader.get_prompt_list(idx=i)
            system_prompt:str = reader.get_system_prompt()
            system_prompt_list:List[str] = reader.get_system_prompt_list()

            start_time = time.time()
            prompt, output = generate(
                question=question,
                question_list=question_list,
                system_prompt=system_prompt,
                system_prompt_list=system_prompt_list,

                model=model,
                tokenizer=tokenizer,
                comp_config=comp_config,
                attention_config=attention_config,
                max_new_tokens=max_new_tokens,
                prefill_compress=prefill_compress,
                exclude_continue=exclude_continue,
                compress_prompt=compress_prompt,

                attn_utils=attn_utils,
                token_utils=token_utils,
                update_attention_method=update_attention_method,
            )
            end_time = time.time()
            input_len:int = len(token_utils.show_prompt_input_ids)
            output_len:int = len(token_utils.show_output_input_ids)
            
            model_answer:str = reader.extract_answer(output)
            gt_answer:str = reader.get_answer(i)
            acc_state, comp_pattern = reader.compare_answer(model_answer, gt_answer, i)
            if acc_state == True:
                acc += 1
                
            writer.write(dict(
                idx=i,
                model_answer=model_answer,
                gt_answer=gt_answer,
                acc_state=acc_state,
                output=output,
                prompt=prompt,
                input_len=input_len,
                output_len=output_len,
                infer_time=end_time-start_time,
                comp_pattern=comp_pattern,
                max_token=token_utils.max_token
            ))
            pbar.set_description(f"model:`{model_answer}`; gt:`{gt_answer}`; correct:{acc_state}; acc: {acc}/{total}={round(acc/total, 5)}; input: {input_len}; output: {output_len}; `{comp_pattern}`; max_token: {token_utils.max_token}")
            token_utils.reset()
            attn_utils.reset()
            pbar.update(1)
            torch.cuda.empty_cache()
    pbar.close()

def main():
    device = 'cuda'
    max_comp_size = 15
    max_prompt_len = 1000
    max_comp_size = 20
    max_prompt_len = 1100
    dtype = torch.bfloat16

    args = get_parser()
    if args.model_short_tag == None:
        args.model_short_tag = args.model_tag
    print(args)

    comp_config = Config.from_file(args.compress_config)
    attention_config = dict(
        diagonal=args.diagonal,
        bi_attention=args.bi_directional,
        see_current=args.see_current,
        prefill_compress=args.prefill_compress,
        exclude_continue=args.exclude_continue
    )


    model, tokenizer = get_model_and_tokenizer(
        args, comp_config
    )

    task_list = [
        (MMLUReader(), "mmlu"),
        (GSM8KReader(), "gsm8k"),
        (GPQAReader(), "gpqa"),
        (BBHReader(), "bbh"),
    ]

    for reader, name in task_list:
        eval_dataset(
            model=model,
            tokenizer=tokenizer,
            reader=reader,
            comp_config=comp_config,
            output_file=f"inference_results/{args.output_tag}/{name}/{args.ckpt}/{args.index}-{args.split_size}{args.model_short_tag}.jsonl",
            # output_file=f"inference_results/{args.output_tag}/{name}_{args.model_tag}_{args.ckpt}.jsonl",
            max_new_tokens=args.max_new_tokens,
            max_prompt_len=max_prompt_len,
            device=device,
            dtype=dtype,
            max_comp_size=max_comp_size,

            attention_config=attention_config,
            prefill_compress=args.prefill_compress,
            exclude_continue=args.exclude_continue,
            compress_prompt=args.compress_prompt,

            update_attention_method=args.update_attention_method,
            rolling_rope=args.rolling_rope,

            dataset_name=name,
            split_size=args.split_size,
            index=args.index,
        )

if __name__ == '__main__':
    main()
