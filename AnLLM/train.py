
import torch
import argparse
from typing import *
from tqdm import tqdm
from copy import deepcopy
from transformers import Trainer, TrainingArguments

from config import Config
from tokenizer import Tokenizer
from model_qwen import Qwen2ForCausalLM
from model_llama import LlamaForCausalLM
from dataset import MyDataset, MyDataCollator
from utils import _print, IGNORE_LABEL_ID, str2bool

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, help="just used for deepspeed.")
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--tokenizer_path', type=str, default=None)
    parser.add_argument('--train_path', type=str, help='training dataset path')
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--max_length', type=int, default=768)
    parser.add_argument('--model_type', type=str, choices=['qwen', 'llama'])

    parser.add_argument('--compress_config', type=str)
    parser.add_argument('--bos_token', type=str)
    parser.add_argument('--eos_token', type=str)
    parser.add_argument('--see_current', type=str2bool)
    parser.add_argument('--bi_directional', type=str2bool)
    parser.add_argument('--diagonal', type=str2bool)
    parser.add_argument('--mode', type=str, choices=['recover', 'normal', 'aug', 'aug-wo-pc'])
    parser.add_argument('--exclude_continue', type=str2bool)
    parser.add_argument('--qkv', type=str)
    parser.add_argument('--freeze_model', type=str2bool)
    parser.add_argument('--train_on_input', type=str2bool)
    parser.add_argument('--output_compress_instruction', type=str)
    parser.add_argument('--hybrid', type=str2bool)  
    parser.add_argument('--prefill_compress', type=str2bool, default=True)

    parser.add_argument('--epochs', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--save_steps', type=int)
    parser.add_argument('--deepspeed', type=str, help="file path")
    parser.add_argument('--micro_batch_size', type=int)
    parser.add_argument('--gradient_accumulation_steps', type=int)
    parser.add_argument('--warmup_ratio', type=float, default=0.)
    parser.add_argument('--warmup_steps', type=int, default=0)
    parser.add_argument('--lr_scheduler_type', type=str, default='linear')
    args = parser.parse_args()
    return args

def get_model_and_tokenizer(
    args,
    comp_config:Config
) -> Tuple[LlamaForCausalLM, Tokenizer]:
    special_token_list:List[str] = list()
    special_token_desp_dict = dict()
    tokenizer: Tokenizer = Tokenizer(
        tokenizer_path=args.tokenizer_path if args.tokenizer_path != None else args.model_path,
        bos_token=args.bos_token,
        eos_token=args.eos_token,
        special_token_list=None,
        add_prefix_space=False,
    )
    assert len(comp_config.special_token_desp_list) == len(comp_config.special_token_name_list)
    for desp, token in zip(comp_config.special_token_desp_list, comp_config.special_token_name_list):
        if tokenizer.convert_tokens_to_ids(token) == None:
            special_token_list.append(token)
            special_token_desp_dict[token] = desp
    if len(special_token_list) > 0:
        tokenizer.add_special_token(special_token_list)
    
    if args.model_type == 'llama':
        model_class = LlamaForCausalLM
    elif args.model_type == 'qwen':
        model_class = Qwen2ForCausalLM
    else:
        assert False, "We only support llama and qwen model."

    model = model_class.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16
    )

    model.add_qkv(
        q='q' in args.qkv,
        k='k' in args.qkv,
        v='v' in args.qkv,
    )

    if model.model.config.vocab_size != len(tokenizer):
        # Expand the token embedding and lm_head
        _print(f"before.embedding.shape={model.model.embed_tokens.weight.shape}")
        _print(f"before.lm_head.shape={model.lm_head.weight.shape}")
        model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
        _print(f"now.embedding.shape={model.model.embed_tokens.weight.shape}")
        _print(f"now.lm_head.shape={model.lm_head.weight.shape}")

    
    
    if args.freeze_model:
        _print(f"Freezing Model:\nnew_token: {len(special_token_list)}\norigin_length: {len(tokenizer) - len(special_token_list)}")
        model.freeze_embed(
            new_token_cnt=len(special_token_list), 
            origin_length=len(tokenizer) - len(special_token_list)
        )
    else:
        _print("mean ...")
        with torch.no_grad():
            for idx, token in enumerate(reversed(special_token_list), start=1):
                description = special_token_desp_dict[token]
                tokenized = tokenizer.tokenize(description)
                tokenized_ids = tokenizer.convert_tokens_to_ids(tokenized)

                # embedding layer
                new_embedding = model.model.embed_tokens.weight[tokenized_ids].mean(axis=0)
                model.model.embed_tokens.weight[-idx, :] = new_embedding.clone().detach().requires_grad_(True)

                # lm_head layer
                last_embedding = model.lm_head.weight[tokenized_ids].mean(axis=0)
                model.lm_head.weight[-idx, :] = last_embedding.clone().detach().requires_grad_(True)
    
    trainable_params = [name for name, param in model.named_parameters() if param.requires_grad]
    print("Trainable Parameters:")
    for param_name in trainable_params:
        print(param_name)

    _print(f"eos_token: {tokenizer.eos_token}; eos_token_id: {tokenizer.eos_token_id}; bos_token: {tokenizer.bos_token}; bos_token_id: {tokenizer.bos_token_id}")

    return model, tokenizer

def get_dataset_and_data_collator(
    args,
    comp_config:Config,
    tokenizer:Tokenizer,
    padding_config:Dict,
    attention_config:Dict,
    sample_config:Dict,
) -> Tuple[MyDataset, MyDataCollator]:
    
    dataset = MyDataset(
        file_path=args.train_path,
        config=comp_config,
        tokenizer=tokenizer,
        padding_config=padding_config,
        train_on_input=args.train_on_input,
        change_rope=False,
        output_compress_instruction=args.output_compress_instruction,
    )

    data_collator = MyDataCollator(
        dataset=dataset,
        attention_config=attention_config,
        exclude_continue=args.exclude_continue,
        sample_config=sample_config
    )

    return dataset, data_collator

def main():
    args = get_parser()
    if args.output_compress_instruction == "None":
        args.output_compress_instruction = ""
    print(args)

    comp_config = Config.from_file(config_path=args.compress_config)
    model, tokenizer = get_model_and_tokenizer(
        args, comp_config
    )

    sample_config:Dict = dict(
        mode=args.mode,
        hybrid=args.hybrid
    )
    attention_config:Dict = dict(
        diagonal=args.diagonal,
        bi_directional=args.bi_directional,
        see_current=args.see_current,
        prefill_compress=args.prefill_compress,
    )
    padding_config = dict(
        padding_side='right',
        label_padding_id=IGNORE_LABEL_ID,
        input_padding_id=tokenizer.eos_token_id,
        max_length=args.max_length,
        position_ids_padding_id=0,
    )

    dataset, data_collator = get_dataset_and_data_collator(
        args=args, 
        comp_config=comp_config,
        tokenizer=tokenizer,
        padding_config=padding_config,
        attention_config=attention_config,
        sample_config=sample_config,
    )

    training_config = TrainingArguments(
        lr_scheduler_type=args.lr_scheduler_type,
        local_rank=args.local_rank,
        gradient_checkpointing=True,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        bf16=True,
        logging_steps=1,
        do_eval=False,
        optim="adamw_torch",
        save_strategy="epoch",      # the default value is step
        save_steps=args.save_steps, # if the strategy is epoch, the save_steps is not used.
        output_dir=args.output_dir,
        save_only_model=True,       # don't save the global_steps
        load_best_model_at_end=False,
        deepspeed=args.deepspeed,
        save_total_limit=10,
        report_to="tensorboard",
        per_device_train_batch_size=args.micro_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=args.warmup_steps,
        warmup_ratio=args.warmup_ratio
    )
    trainer = Trainer(
        model=model,
        train_dataset=dataset,
        args=training_config,
        data_collator=data_collator
    )
    trainer.train()


if __name__ == '__main__':
    main()
