
#!/bin/bash

# python range
range() {
    local start=$1
    local end=$2
    local step=${3:-1}
    local i=$start
    while ((i < end)); do
        echo -n "$i "
        ((i += step))
    done
}

# Exclude specified elements from 0-28 (excluding 28)
exclude() {
    local array1=("$@")
    local array2=($(range 0 28))
    local diff=()

    for item in "${array2[@]}"; do
        if [[ ! " ${array1[@]} " =~ " $item " ]]; then
            diff+=("$item")
        fi
    done

    echo "${diff[@]}"
}


python train_split.py \
    --experiment_name "simple" \
    --model_path "/path/to/Qwen2-1.5B-Instruct" \
    --dataset_name "financial_phrasebank" \
    --num_train_epochs 20 \
    --lr_scheduler_type "constant" \
    --learning_rate 4e-4 \
    --max_len 128 \
    --train_batch_size 48 \
    --eval_batch_size 48 \
    --lora_rank 16 \
    --privacy_budget 5000 \
    --lst_enable true \
    --lst_reduce_factor 16 \
    --lst_input_type "clean" \
    --lst_skip $(exclude 0 7 14 21 27) \
    --lst_random_init false \
    --auto_skip false \
    --mi_downsample_enable false


python train_split.py \
    --experiment_name "simple" \
    --model_path "/path/to/Qwen2-1.5B-Instruct" \
    --dataset_name "financial_phrasebank" \
    --num_train_epochs 20 \
    --lr_scheduler_type "constant" \
    --learning_rate 4e-4 \
    --max_len 128 \
    --train_batch_size 48 \
    --eval_batch_size 48 \
    --lora_rank 16 \
    --privacy_budget 5000 \
    --lst_enable true \
    --lst_reduce_factor 16 \
    --lst_input_type "clean" \
    --lst_skip -1 \
    --lst_random_init false \
    --auto_skip true \
    --num_reserved_layers 3 \
    --num_integrate_step 5 \
    --num_samples 32 \
    --keep_last_layer true \
    --num_integrate_batch_size 2 \
    --mi_downsample_enable false


python train_split.py \
    --experiment_name "simple" \
    --model_path "/path/to/Qwen2-1.5B-Instruct" \
    --dataset_name "financial_phrasebank" \
    --num_train_epochs 20 \
    --lr_scheduler_type "constant" \
    --learning_rate 4e-4 \
    --max_len 128 \
    --train_batch_size 48 \
    --eval_batch_size 48 \
    --lora_rank 16 \
    --privacy_budget 5000 \
    --lst_enable true \
    --lst_reduce_factor 16 \
    --lst_input_type "clean" \
    --lst_skip -1 \
    --lst_random_init false \
    --auto_skip true \
    --num_reserved_layers 3 \
    --num_integrate_step 5 \
    --num_samples 32 \
    --keep_last_layer true \
    --num_integrate_batch_size 2 \
    --mi_downsample_enable true



