export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NPROC_PER_NODE=8
export MASTER_PORT=12701

MODEL_PATH="Qwen/Qwen2.5-7B-Instruct"
TRAIN_DATA_PATH=""
SAVE_PATH=""

# Make sure you are in directory ./deepanalyze/ms-swift/
swift sft \
    --model "${MODEL_PATH}" \
    --train_type "full" \
    --dataset ${TRAIN_DATA_PATH} \
    --torch_dtype "bfloat16" \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --learning_rate 2e-5 \
    --gradient_accumulation_steps 1 \
    --packing true \
    --logging_steps 1 \
    --max_length 16384 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 8 \
    --dataset_num_proc 8 \
    --save_strategy 'epoch' \
    --save_total_limit 3 \
    --save_only_model true \
    --output_dir "${SAVE_PATH}" \
    --deepspeed "zero3" \
    --use_liger_kernel true \
    --attn_impl "flash_attn"