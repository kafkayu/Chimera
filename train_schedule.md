1.5 实验
13b测试
1.5 - 1.7
下载13b模型
https://huggingface.co/lmsys/vicuna-13b-v1.5

```
torchrun --nproc_per_node=1   medusa/train/train.py --model_name_or_path ../../../model/vicuna-13b-v1.3 \
    --data_path ../../../data/ShareGPT_Vicuna_unfiltered/train.json \
    --eval_data_path  "../../../data/ShareGPT_Vicuna_unfiltered/test.json" \
    --output_dir idea9_3gram \
    --num_train_epochs 4 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4\
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "steps" \
    --eval_steps 50 \
    --save_strategy "steps" \
    --save_steps 400 \
    --save_total_limit 2 \
    --learning_rate 1e-4 \
    --weight_decay 0.0 \
    --warmup_ratio 0.05 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 2048 \
    --lazy_preprocess True \
    --medusa_num_heads 1 \
    --medusa_num_layers 1
```


