#!/bin/bash
#SBATCH --job-name=polar-train
#SBATCH --nodes=1
#SBATCH --partition=general
#SBATCH --gres=gpu:A6000:4
#SBATCH --time=47:00:00
#SBATCH --mem=80G
#SBATCH --error=./polar-training.err
#SBATCH --exclude=babel-1-23,babel-4-37,babel-11-9,babel-1-27,babel-4-33,babel-0-23,babel-11-21

accelerate launch --num_processes 4 --main_process_port=29502 causal_ft.py \
    --model_type llama \
    --model_name  /data/user_data/hdiddee/llama_models/llama_checkpoint/ \
    --do_train \
    --budget 10000 \
    --num_train_epochs 3 \
    --train_file ../assets/neg_train.jsonl \
    --save_strategy epoch \
    --learning_rate 2e-5 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 16 \
    --optim paged_adamw_32bit \
    --weight_decay 0.0 \
    --fp16 False \
    --bf16 True \
    --save_total_limit 1 \
    --output_dir /data/group_data/dei-group/hdiddee/neg \
    --warmup_ratio 0.3 \
    --lr_scheduler_type cosine 

accelerate launch --num_processes 4 --main_process_port=29502 causal_ft.py \
    --model_type llama \
    --model_name  /data/user_data/hdiddee/llama_models/llama_checkpoint/ \
    --do_train \
    --budget 10000 \
    --num_train_epochs 3 \
    --train_file ../assets/pos_train.jsonl \
    --save_strategy epoch \
    --learning_rate 2e-5 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 16 \
    --optim paged_adamw_32bit \
    --weight_decay 0.0 \
    --fp16 False \
    --bf16 True \
    --save_total_limit 1 \
    --output_dir /data/group_data/dei-group/hdiddee/pos \
    --warmup_ratio 0.3 \
    --lr_scheduler_type cosine 