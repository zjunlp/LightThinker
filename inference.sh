# we will load model from `output/{model_tag}/checkpoint-{args.ckpt}`

# `model_tag` is the filename under the output/ folder, 
# corresponding to line 1459 of the code in LightThinker/inference.py.
model_tag=""

# `model_short_tag` is used to save file, 
# corresponding to line 1684 of the code in LightThinker/inference.py.
model_short_tag=""

model_type="qwen"
tokenizer_path="Qwen/Qwen2.5-Math-7B"
bos_token="<|im_start|>"
eos_token="<|im_end|>"
compress_config="./configs/LightThinker/qwen/v1.json"

ckpt=1045
output_tag="YOUR-TAG"
max_new_tokens=10240

root_dir="./LightThinker"

prefix=""
diagonal="false"
see_current="false"
compress_prompt="false"
rolling_rope="false"
bi_directional="false"
exclude_continue="false"
output_compress_instruction="None"
prefill_compress="false"
update_attention_method="local"

split_size=4

index=1
CUDA_VISIBLE_DEVICES=0 nohup python "${root_dir}/inference.py" \
    --model_tag $model_tag \
    --model_short_tag $model_short_tag \
    --ckpt $ckpt \
    --tokenizer_path $tokenizer_path \
    --compress_config $compress_config \
    --max_new_tokens $max_new_tokens \
    --output_tag $output_tag \
    --model_type $model_type \
    --bos_token $bos_token \
    --eos_token $eos_token \
    --rolling_rope $rolling_rope \
    --diagonal $diagonal \
    --bi_directional $bi_directional \
    --see_current $see_current \
    --exclude_continue $exclude_continue \
    --output_compress_instruction $output_compress_instruction \
    --prefill_compress $prefill_compress \
    --compress_prompt $compress_prompt \
    --update_attention_method $update_attention_method \
    --split_size $split_size \
    --index $index > "ours_infer_log/${rolling_rope}_${compress_prompt}/${index}${prefix}_${model_short_tag}_${ckpt}.txt" 2>&1 &



index=2
CUDA_VISIBLE_DEVICES=1 nohup python "${root_dir}/inference.py" \
    --model_tag $model_tag \
    --model_short_tag $model_short_tag \
    --ckpt $ckpt \
    --tokenizer_path $tokenizer_path \
    --compress_config $compress_config \
    --max_new_tokens $max_new_tokens \
    --output_tag $output_tag \
    --model_type $model_type \
    --bos_token $bos_token \
    --eos_token $eos_token \
    --rolling_rope $rolling_rope \
    --diagonal $diagonal \
    --bi_directional $bi_directional \
    --see_current $see_current \
    --exclude_continue $exclude_continue \
    --output_compress_instruction $output_compress_instruction \
    --prefill_compress $prefill_compress \
    --compress_prompt $compress_prompt \
    --update_attention_method $update_attention_method \
    --split_size $split_size \
    --index $index > "ours_infer_log/${rolling_rope}_${compress_prompt}/${index}${prefix}_${model_short_tag}_${ckpt}.txt" 2>&1 &


index=3
CUDA_VISIBLE_DEVICES=2 nohup python "${root_dir}/inference.py" \
    --model_tag $model_tag \
    --model_short_tag $model_short_tag \
    --ckpt $ckpt \
    --tokenizer_path $tokenizer_path \
    --compress_config $compress_config \
    --max_new_tokens $max_new_tokens \
    --output_tag $output_tag \
    --model_type $model_type \
    --bos_token $bos_token \
    --eos_token $eos_token \
    --rolling_rope $rolling_rope \
    --diagonal $diagonal \
    --bi_directional $bi_directional \
    --see_current $see_current \
    --exclude_continue $exclude_continue \
    --output_compress_instruction $output_compress_instruction \
    --prefill_compress $prefill_compress \
    --compress_prompt $compress_prompt \
    --update_attention_method $update_attention_method \
    --split_size $split_size \
    --index $index > "ours_infer_log/${rolling_rope}_${compress_prompt}/${index}${prefix}_${model_short_tag}_${ckpt}.txt" 2>&1 &


index=4
CUDA_VISIBLE_DEVICES=3 nohup python "${root_dir}/inference.py" \
    --model_tag $model_tag \
    --model_short_tag $model_short_tag \
    --ckpt $ckpt \
    --tokenizer_path $tokenizer_path \
    --compress_config $compress_config \
    --max_new_tokens $max_new_tokens \
    --output_tag $output_tag \
    --model_type $model_type \
    --bos_token $bos_token \
    --eos_token $eos_token \
    --rolling_rope $rolling_rope \
    --diagonal $diagonal \
    --bi_directional $bi_directional \
    --see_current $see_current \
    --exclude_continue $exclude_continue \
    --output_compress_instruction $output_compress_instruction \
    --prefill_compress $prefill_compress \
    --compress_prompt $compress_prompt \
    --update_attention_method $update_attention_method \
    --split_size $split_size \
    --index $index > "ours_infer_log/${rolling_rope}_${compress_prompt}/${index}${prefix}_${model_short_tag}_${ckpt}.txt" 2>&1 &

