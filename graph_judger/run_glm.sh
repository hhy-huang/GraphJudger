PRE_SEQ_LEN=8
LR=1e-2
NUM_GPUS=1

torchrun --standalone --nnodes=1  --nproc_per_node=$NUM_GPUS  ptuning_main.py \
    --do_train \
    --train_file data/WN18RR/train_instructions_glm_merge.json \
    --validation_file data/WN18RR/test_instructions_glm_merge.json \
    --preprocessing_num_workers 10 \
    --prompt_column prompt \
    --response_column response \
    --overwrite_cache \
    --model_name_or_path models/chatglm2-6b \
    --output_dir models/wn11-chatglm2-6b \
    --overwrite_output_dir \
    --max_source_length 230 \
    --max_target_length 20 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --predict_with_generate \
    --max_steps 80000 \
    --logging_steps 300 \
    --save_steps 10000 \
    --learning_rate $LR \
    --pre_seq_len $PRE_SEQ_LEN \
    # --quantization_bit 4