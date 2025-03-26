# model
model_type="qwen"
tokenizer_path="Qwen/Qwen2.5-7B-Instruct"
model_path="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
bos_token="<|im_start|>"
eos_token="<|im_end|>"
conf_version="v1"

# training
max_length=4096
lr_scheduler_type="cosine"
epochs=5
lr=2e-5
save_steps=1
deepspeed="./configs/ds_z3_offload_config.json"
micro_batch_size=5
gradient_accumulation_steps=4
warmup_ratio=0.05
mode="aug-wo-pc"
warmup_steps=0

# others
model_size="7b"
init_tag=""
train_path="./data/train/train.jsonl"
see_current="true"
bi_directional="false"
diagonal="false"
exclude_continue="false"
qkv="no"
freeze_model="false"
train_on_input="false"
hybrid="false"
has_instruction="false"
output_compress_instruction="None"
prefill_compress="false"

echo "model_type=${model_type}"
echo "model_size=${model_size}"
echo "model_path=${model_path}"
echo "tokenizer_path=${tokenizer_path}"
echo "train_path=${train_path}"
echo "lr_scheduler_type=${lr_scheduler_type}"
echo "max_length=${max_length}"
echo "bos_token=${bos_token}"
echo "eos_token=${eos_token}"
echo "see_current=${see_current}"
echo "bi_directional=${bi_directional}"
echo "diagonal=${diagonal}"
echo "mode=${mode}"
echo "exclude_continue=${exclude_continue}"
echo "qkv=${qkv}"
echo "freeze_model=${freeze_model}"
echo "train_on_input=${train_on_input}"
echo "hybrid=${hybrid}"
echo "has_instruction=${has_instruction}"
echo "output_compress_instruction=${output_compress_instruction}"
echo "prefill_compress=${prefill_compress}"
echo "epochs=${epochs}"
echo "lr=${lr}"
echo "save_steps=${save_steps}"
echo "deepspeed=${deepspeed}"
echo "micro_batch_size=${micro_batch_size}"
echo "gradient_accumulation_steps=${gradient_accumulation_steps}"
echo "warmup_ratio=${warmup_ratio}"
echo "warmup_steps=${warmup_steps}"
echo "init_tag=${init_tag}"

att_info="${model_size}-${model_type}-len_${max_length}-see_cur_${see_current}-bi_${bi_directional}-diag_${diagonal}-mode_${mode}"
train_info="prefill_compress_${prefill_compress}-hybrid_${hybrid}-epoch_${epochs}-lr_${lr}-bsz_${micro_batch_size}-accumu_${gradient_accumulation_steps}-warm_r_${warmup_ratio}-warm_s_${warmup_steps}-freeze_model_${freeze_model}-train_input_${train_on_input}-qkv_${qkv}-ex_con_${exclude_continue}-has_instruction_${has_instruction}"
output_dir="output/${init_tag}${lr_scheduler_type}${att_info}-${train_info}"
compress_config="configs/AnLLM/${model_type}/${conf_version}.json"

deepspeed --include localhost:0,1,2,3 AnLLM/train.py \
    --model_type $model_type \
    --model_path $model_path \
    --tokenizer_path $tokenizer_path \
    --train_path $train_path \
    --output_dir $output_dir \
    --max_length $max_length \
    --compress_config $compress_config \
    --bos_token $bos_token \
    --eos_token $eos_token \
    --see_current $see_current \
    --bi_directional $bi_directional \
    --diagonal $diagonal \
    --mode $mode \
    --exclude_continue $exclude_continue \
    --qkv $qkv \
    --freeze_model $freeze_model \
    --train_on_input $train_on_input \
    --output_compress_instruction $output_compress_instruction \
    --epochs $epochs \
    --lr $lr \
    --save_steps $save_steps \
    --deepspeed $deepspeed \
    --micro_batch_size $micro_batch_size \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --warmup_ratio $warmup_ratio \
    --warmup_steps $warmup_steps \
    --hybrid $hybrid \
    --prefill_compress $prefill_compress \
    --lr_scheduler_type $lr_scheduler_type
