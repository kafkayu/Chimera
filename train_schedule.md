1.5 实验
13b测试
1.5 - 1.7
下载13b模型
https://huggingface.co/lmsys/vicuna-13b-v1.5

```
torchrun --nproc_per_node=1   medusa/train/train.py --model_name_or_path ../../../model/vicuna-13b-v1.3 \
    --data_path ../../../data/ShareGPT_Vicuna_unfiltered/train.json \
    --eval_data_path  "../../../data/ShareGPT_Vicuna_unfiltered/test.json" \
    --output_dir idea9_3gram_0106 \
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
1.6 

| 任务| 负责人 | dll说明 | 详细说明|
|-------|-------|-------|-------|
| 跑7b多token结果 | 俞嘉鸿 | 1.6 20：00 结束| 得知预测多个token的准确率|
| 13b结果测试 | zihao-wang | 等待nvidia回复后 |得知预测多个token的准确率|
| eagle baseline 测试 | 庞千石 | 1.7晚上19：00 |查看eagle 1b,7b,13b的加速效果，eagle 预测多个token的准确率|



2.若服务器可运行，13b结果测试 

下载13b模型 
https://huggingface.co/lmsys/vicuna-13b-v1.5

```
torchrun --nproc_per_node=1   medusa/train/train.py --model_name_or_path ../../../model/vicuna-13b-v1.3 \
    --data_path ../../../data/ShareGPT_Vicuna_unfiltered/train.json \
    --eval_data_path  "../../../data/ShareGPT_Vicuna_unfiltered/test.json" \
    --output_dir idea6_3gram_0106 \
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
3.eagle baseline 测试 负责人：庞千石 ddl:1.7晚上19：00

[eagle](https://github.com/SafeAILab/EAGLE) 地址

期望结果：
查看eagle 1b,7b,13b的加速效果
eagle top5 token的预测准确率


