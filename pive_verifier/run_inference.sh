#!/bin/bash
# ./run_inference.sh /data/ChenWei/HaoyuHuang/flan-t5-large outputs_1/t5_large_only_one_error_webnlg_new/val_avg_loss=4.5331-step_count=7.ckpt 0
ROOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

GPUID=$3
CHECK_POINT=$2
MODEL=$1
FOLDER=data/only_one_error_rebel/verifier_result

export CURRENT_DIR=${ROOT_DIR}
export OUTPUT_DIR=${CURRENT_DIR}/${FOLDER}

rm -rf $OUTPUT_DIR
mkdir -p $OUTPUT_DIR

export OMP_NUM_THREADS=10


export CUDA_VISIBLE_DEVICES=${GPUID}
#--n_test 20 \
python ${ROOT_DIR}/finetune.py \
--data_dir=${ROOT_DIR}/data/rebel_itr1_generate \
--task graph2text \
--model_name_or_path=${MODEL} \
--eval_batch_size=1 \
--gpus 1 \
--output_dir=$OUTPUT_DIR \
--checkpoint=$CHECK_POINT \
--cache_dir="/home/jiuzhouh/wj84_scratch/jiuzhouh/.cache/" \
--max_source_length=512 \
--max_target_length=128 \
--val_max_target_length=128 \
--test_max_target_length=128 \
--eval_max_gen_length=128 \
--do_predict \
--eval_beams 5